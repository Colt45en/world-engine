// NEXUS Event Reactions System
// Creates emergent behavior through event response chains with damping and cooldowns
// Enables complex interactions between game systems

#pragma once
#include <functional>
#include <string>
#include <unordered_map>
#include <vector>
#include <queue>
#include <memory>
#include <chrono>
#include <mutex>

namespace NEXUS {

// Forward declarations
template<typename EventType> class EventReactionSystem;

// ============ EVENT DATA STRUCTURES ============

struct EventData {
    std::string eventId;
    std::unordered_map<std::string, std::string> properties;
    std::chrono::steady_clock::time_point timestamp;
    double intensity{1.0};
    void* sourceObject{nullptr};

    EventData(const std::string& id = "")
        : eventId(id), timestamp(std::chrono::steady_clock::now()) {}

    template<typename T>
    void setProperty(const std::string& key, const T& value) {
        properties[key] = std::to_string(value);
    }

    void setProperty(const std::string& key, const std::string& value) {
        properties[key] = value;
    }

    template<typename T>
    T getProperty(const std::string& key, const T& defaultValue = T{}) const {
        auto it = properties.find(key);
        if (it != properties.end()) {
            if constexpr (std::is_same_v<T, std::string>) {
                return it->second;
            } else if constexpr (std::is_same_v<T, double>) {
                return std::stod(it->second);
            } else if constexpr (std::is_same_v<T, int>) {
                return std::stoi(it->second);
            } else if constexpr (std::is_same_v<T, bool>) {
                return it->second == "true" || it->second == "1";
            }
        }
        return defaultValue;
    }

    double getAgeSeconds() const {
        auto now = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(now - timestamp);
        return duration.count() / 1000.0;
    }
};

// ============ REACTION DEFINITION ============

template<typename EventType>
class EventReaction {
public:
    using ReactionFunction = std::function<void(const EventData&, EventReactionSystem<EventType>*)>;

private:
    std::string name;
    std::string triggerEventType;
    ReactionFunction function;
    double cooldownSeconds{0.0};
    double dampingFactor{1.0};
    int maxActivations{-1}; // -1 = unlimited
    double minIntensity{0.1};
    bool enabled{true};

    // State tracking
    mutable std::chrono::steady_clock::time_point lastActivation;
    mutable int activationCount{0};
    mutable double currentDamping{1.0};

public:
    EventReaction(const std::string& n, const std::string& eventType, ReactionFunction fn)
        : name(n), triggerEventType(eventType), function(std::move(fn)) {
        lastActivation = std::chrono::steady_clock::now() - std::chrono::seconds(1000);
    }

    // Builder pattern methods
    EventReaction& setCooldown(double seconds) { cooldownSeconds = seconds; return *this; }
    EventReaction& setDamping(double factor) { dampingFactor = factor; return *this; }
    EventReaction& setMaxActivations(int max) { maxActivations = max; return *this; }
    EventReaction& setMinIntensity(double min) { minIntensity = min; return *this; }
    EventReaction& setEnabled(bool e) { enabled = e; return *this; }

    // Check if reaction can trigger
    bool canTrigger(const EventData& eventData) const {
        if (!enabled) return false;

        // Check event type
        if (eventData.eventId != triggerEventType) return false;

        // Check intensity
        if (eventData.intensity * currentDamping < minIntensity) return false;

        // Check max activations
        if (maxActivations > 0 && activationCount >= maxActivations) return false;

        // Check cooldown
        auto now = std::chrono::steady_clock::now();
        auto timeSinceLastActivation = std::chrono::duration_cast<std::chrono::milliseconds>(now - lastActivation);
        double secondsSinceLastActivation = timeSinceLastActivation.count() / 1000.0;

        return secondsSinceLastActivation >= cooldownSeconds;
    }

    // Execute the reaction
    void trigger(const EventData& eventData, EventReactionSystem<EventType>* system) {
        if (!canTrigger(eventData)) return;

        // Update state
        lastActivation = std::chrono::steady_clock::now();
        activationCount++;

        // Apply damping
        EventData dampedEvent = eventData;
        dampedEvent.intensity *= currentDamping;

        // Execute reaction
        if (function) {
            function(dampedEvent, system);
        }

        // Update damping for next time
        currentDamping *= dampingFactor;
        if (currentDamping < 0.01) currentDamping = 0.01; // Minimum damping
    }

    // Reset reaction state
    void reset() {
        activationCount = 0;
        currentDamping = 1.0;
        lastActivation = std::chrono::steady_clock::now() - std::chrono::seconds(1000);
    }

    // Accessors
    const std::string& getName() const { return name; }
    const std::string& getTriggerEventType() const { return triggerEventType; }
    double getCooldown() const { return cooldownSeconds; }
    double getDamping() const { return dampingFactor; }
    int getActivationCount() const { return activationCount; }
    double getCurrentDamping() const { return currentDamping; }
    bool isEnabled() const { return enabled; }
};

// ============ EVENT REACTION SYSTEM ============

template<typename EventType>
class EventReactionSystem {
private:
    std::unordered_map<std::string, std::vector<std::unique_ptr<EventReaction<EventType>>>> reactions;
    std::queue<EventData> eventQueue;
    std::mutex queueMutex;
    bool processingSuspended{false};

    // Statistics
    size_t totalEventsProcessed{0};
    size_t totalReactionsTriggered{0};
    std::chrono::steady_clock::time_point lastProcessTime;

public:
    EventReactionSystem() : lastProcessTime(std::chrono::steady_clock::now()) {}

    // Register a new reaction
    void registerReaction(std::unique_ptr<EventReaction<EventType>> reaction) {
        if (reaction) {
            std::string eventType = reaction->getTriggerEventType();
            reactions[eventType].push_back(std::move(reaction));
        }
    }

    // Helper for creating and registering reactions
    EventReaction<EventType>& createReaction(const std::string& name,
                                           const std::string& eventType,
                                           typename EventReaction<EventType>::ReactionFunction function) {
        auto reaction = std::make_unique<EventReaction<EventType>>(name, eventType, std::move(function));
        EventReaction<EventType>* ptr = reaction.get();
        registerReaction(std::move(reaction));
        return *ptr;
    }

    // Queue an event for processing
    void queueEvent(const EventData& eventData) {
        std::lock_guard<std::mutex> lock(queueMutex);
        eventQueue.push(eventData);
    }

    // Process all queued events
    void processEvents() {
        if (processingSuspended) return;

        std::lock_guard<std::mutex> lock(queueMutex);

        while (!eventQueue.empty()) {
            EventData currentEvent = eventQueue.front();
            eventQueue.pop();

            processEvent(currentEvent);
            totalEventsProcessed++;
        }

        lastProcessTime = std::chrono::steady_clock::now();
    }

    // Process a single event immediately
    void processEvent(const EventData& eventData) {
        auto it = reactions.find(eventData.eventId);
        if (it != reactions.end()) {
            for (auto& reaction : it->second) {
                if (reaction->canTrigger(eventData)) {
                    reaction->trigger(eventData, this);
                    totalReactionsTriggered++;
                }
            }
        }
    }

    // Emit an event and process it immediately
    void emitEvent(const EventData& eventData) {
        processEvent(eventData);
        totalEventsProcessed++;
    }

    // Suspend/resume processing
    void suspendProcessing() { processingSuspended = true; }
    void resumeProcessing() { processingSuspended = false; }
    bool isProcessingSuspended() const { return processingSuspended; }

    // Reset all reactions
    void resetAllReactions() {
        for (auto& pair : reactions) {
            for (auto& reaction : pair.second) {
                reaction->reset();
            }
        }
    }

    // Remove reactions
    bool removeReaction(const std::string& eventType, const std::string& reactionName) {
        auto it = reactions.find(eventType);
        if (it != reactions.end()) {
            auto& reactionList = it->second;
            auto reactionIt = std::find_if(reactionList.begin(), reactionList.end(),
                [&reactionName](const std::unique_ptr<EventReaction<EventType>>& r) {
                    return r->getName() == reactionName;
                });

            if (reactionIt != reactionList.end()) {
                reactionList.erase(reactionIt);
                return true;
            }
        }
        return false;
    }

    void clearReactions() {
        reactions.clear();
    }

    // Query system
    std::vector<std::string> getEventTypes() const {
        std::vector<std::string> types;
        for (const auto& pair : reactions) {
            types.push_back(pair.first);
        }
        return types;
    }

    std::vector<std::string> getReactionsForEvent(const std::string& eventType) const {
        std::vector<std::string> names;
        auto it = reactions.find(eventType);
        if (it != reactions.end()) {
            for (const auto& reaction : it->second) {
                names.push_back(reaction->getName());
            }
        }
        return names;
    }

    const EventReaction<EventType>* getReaction(const std::string& eventType, const std::string& reactionName) const {
        auto it = reactions.find(eventType);
        if (it != reactions.end()) {
            auto reactionIt = std::find_if(it->second.begin(), it->second.end(),
                [&reactionName](const std::unique_ptr<EventReaction<EventType>>& r) {
                    return r->getName() == reactionName;
                });

            if (reactionIt != it->second.end()) {
                return reactionIt->get();
            }
        }
        return nullptr;
    }

    // Statistics
    size_t getTotalEventsProcessed() const { return totalEventsProcessed; }
    size_t getTotalReactionsTriggered() const { return totalReactionsTriggered; }
    size_t getQueuedEventCount() const {
        std::lock_guard<std::mutex> lock(const_cast<std::mutex&>(queueMutex));
        return eventQueue.size();
    }
    size_t getReactionCount() const {
        size_t total = 0;
        for (const auto& pair : reactions) {
            total += pair.second.size();
        }
        return total;
    }

    double getTimeSinceLastProcess() const {
        auto now = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(now - lastProcessTime);
        return duration.count() / 1000.0;
    }
};

// ============ PREDEFINED REACTION BUILDERS ============

template<typename EventType>
class StandardReactions {
public:
    // Chain reaction: one event triggers another
    static std::unique_ptr<EventReaction<EventType>> createChainReaction(
        const std::string& name,
        const std::string& triggerEvent,
        const std::string& chainEvent,
        double intensity = 0.8) {

        return std::make_unique<EventReaction<EventType>>(
            name,
            triggerEvent,
            [chainEvent, intensity](const EventData& eventData, EventReactionSystem<EventType>* system) {
                EventData newEvent(chainEvent);
                newEvent.intensity = eventData.intensity * intensity;
                newEvent.sourceObject = eventData.sourceObject;
                newEvent.setProperty("chain_source", eventData.eventId);
                system->queueEvent(newEvent);
            }
        );
    }

    // Dampened echo: same event triggers again with reduced intensity
    static std::unique_ptr<EventReaction<EventType>> createEchoReaction(
        const std::string& name,
        const std::string& eventType,
        double echoIntensity = 0.5,
        double delaySeconds = 1.0) {

        return std::make_unique<EventReaction<EventType>>(
            name,
            eventType,
            [eventType, echoIntensity, delaySeconds](const EventData& eventData, EventReactionSystem<EventType>* system) {
                // In a real implementation, you'd want a timer system for the delay
                EventData echoEvent(eventType + "_echo");
                echoEvent.intensity = eventData.intensity * echoIntensity;
                echoEvent.sourceObject = eventData.sourceObject;
                echoEvent.setProperty("echo_source", eventData.eventId);
                echoEvent.setProperty("echo_delay", delaySeconds);
                system->queueEvent(echoEvent);
            }
        ).get()->setCooldown(delaySeconds * 2.0) // Prevent immediate re-echo
                .setDamping(0.9); // Each echo gets weaker
    }

    // Threshold reaction: only triggers when cumulative intensity exceeds threshold
    static std::unique_ptr<EventReaction<EventType>> createThresholdReaction(
        const std::string& name,
        const std::string& triggerEvent,
        const std::string& thresholdEvent,
        double threshold = 5.0) {

        // Note: In practice, you'd want to store cumulative intensity in the system
        static double cumulativeIntensity = 0.0;

        return std::make_unique<EventReaction<EventType>>(
            name,
            triggerEvent,
            [thresholdEvent, threshold](const EventData& eventData, EventReactionSystem<EventType>* system) {
                cumulativeIntensity += eventData.intensity;

                if (cumulativeIntensity >= threshold) {
                    EventData thresholdEventData(thresholdEvent);
                    thresholdEventData.intensity = cumulativeIntensity / threshold;
                    thresholdEventData.sourceObject = eventData.sourceObject;
                    thresholdEventData.setProperty("accumulated_intensity", cumulativeIntensity);
                    system->queueEvent(thresholdEventData);

                    cumulativeIntensity = 0.0; // Reset after threshold reached
                }
            }
        );
    }
};

// ============ REACTION PATTERN ANALYZER ============

template<typename EventType>
class ReactionPatternAnalyzer {
private:
    std::vector<std::pair<std::chrono::steady_clock::time_point, std::string>> eventHistory;
    size_t maxHistorySize{1000};

public:
    void recordEvent(const std::string& eventId) {
        eventHistory.emplace_back(std::chrono::steady_clock::now(), eventId);

        // Trim history if too large
        if (eventHistory.size() > maxHistorySize) {
            eventHistory.erase(eventHistory.begin(), eventHistory.begin() + 100);
        }
    }

    // Find common event sequences
    std::vector<std::pair<std::vector<std::string>, int>> findPatterns(int sequenceLength = 2) const {
        std::unordered_map<std::string, int> patterns;

        if (eventHistory.size() < sequenceLength) return {};

        for (size_t i = 0; i <= eventHistory.size() - sequenceLength; ++i) {
            std::string pattern;
            for (int j = 0; j < sequenceLength; ++j) {
                if (j > 0) pattern += "->";
                pattern += eventHistory[i + j].second;
            }
            patterns[pattern]++;
        }

        // Convert to vector and sort by frequency
        std::vector<std::pair<std::vector<std::string>, int>> result;
        for (const auto& pair : patterns) {
            std::vector<std::string> sequence;
            std::string current;
            for (char c : pair.first) {
                if (c == '-' && current.length() > 0) {
                    // Skip the '>' part of '->'
                    continue;
                } else if (c == '>') {
                    if (!current.empty()) {
                        sequence.push_back(current);
                        current.clear();
                    }
                } else {
                    current += c;
                }
            }
            if (!current.empty()) {
                sequence.push_back(current);
            }
            result.emplace_back(sequence, pair.second);
        }

        std::sort(result.begin(), result.end(),
                 [](const auto& a, const auto& b) { return a.second > b.second; });

        return result;
    }

    void clearHistory() {
        eventHistory.clear();
    }

    size_t getHistorySize() const { return eventHistory.size(); }
};

} // namespace NEXUS
