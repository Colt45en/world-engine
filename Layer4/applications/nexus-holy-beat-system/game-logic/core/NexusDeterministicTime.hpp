// NEXUS Deterministic Time System
// Fixes audio sync issues with predictable, controllable time progression
// Supports pause, slow-motion, and tick-based determinism

#pragma once
#include <chrono>
#include <functional>

namespace NEXUS {

// ============ DETERMINISTIC CLOCK SYSTEM ============

class DeterministicClock {
private:
    double tickTime{0.0};           // Current simulation time
    double tickRate{60.0};          // Target ticks per second
    double tickDelta{1.0/60.0};     // Time per tick
    bool paused{false};
    double slowMotionFactor{1.0};

    // Frame timing
    std::chrono::high_resolution_clock::time_point lastRealTime;
    double accumulator{0.0};        // For fixed timestep
    bool initialized{false};

    // Statistics
    int totalTicks{0};
    double averageDelta{0.0};
    double maxDelta{0.0};

public:
    DeterministicClock(double targetFPS = 60.0) : tickRate(targetFPS), tickDelta(1.0/targetFPS) {
        lastRealTime = std::chrono::high_resolution_clock::now();
    }

    // Core time progression
    void tick() {
        if (paused) return;

        tickTime += tickDelta * slowMotionFactor;
        totalTicks++;

        // Update statistics
        averageDelta = (averageDelta * (totalTicks - 1) + tickDelta) / totalTicks;
        maxDelta = std::max(maxDelta, tickDelta);
    }

    // Fixed timestep update with accumulator
    int updateFixedStep() {
        if (!initialized) {
            lastRealTime = std::chrono::high_resolution_clock::now();
            initialized = true;
            return 0;
        }

        auto currentTime = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration<double>(currentTime - lastRealTime).count();
        lastRealTime = currentTime;

        // Clamp elapsed time to prevent spiral of death
        elapsed = std::min(elapsed, 0.25); // Max 250ms per frame
        accumulator += elapsed;

        int ticksProcessed = 0;
        while (accumulator >= tickDelta && ticksProcessed < 10) { // Max 10 ticks per frame
            tick();
            accumulator -= tickDelta;
            ticksProcessed++;
        }

        return ticksProcessed;
    }

    // Time accessors
    double getTime() const { return tickTime; }
    double getDelta() const { return tickDelta * slowMotionFactor; }
    double getTargetFPS() const { return tickRate; }
    int getTotalTicks() const { return totalTicks; }
    double getInterpolation() const { return accumulator / tickDelta; }

    // Time control
    void setPause(bool p) { paused = p; }
    bool isPaused() const { return paused; }

    void setSlowMotion(double factor) {
        slowMotionFactor = std::max(0.01, std::min(10.0, factor));
    }
    double getSlowMotion() const { return slowMotionFactor; }

    void setTickRate(double fps) {
        tickRate = std::max(1.0, std::min(1000.0, fps));
        tickDelta = 1.0 / tickRate;
    }

    // Time utilities
    void reset() {
        tickTime = 0.0;
        accumulator = 0.0;
        totalTicks = 0;
        averageDelta = 0.0;
        maxDelta = 0.0;
        lastRealTime = std::chrono::high_resolution_clock::now();
    }

    void jumpToTime(double newTime) {
        tickTime = newTime;
    }

    // Statistics
    struct TimeStats {
        double currentTime;
        double averageDelta;
        double maxDelta;
        int totalTicks;
        double slowMotion;
        bool paused;
        double targetFPS;
    };

    TimeStats getStats() const {
        return {
            tickTime,
            averageDelta,
            maxDelta,
            totalTicks,
            slowMotionFactor,
            paused,
            tickRate
        };
    }
};

// ============ TIMER SYSTEM FOR EVENTS ============

class Timer {
private:
    double startTime{0.0};
    double duration{0.0};
    bool running{false};
    bool looping{false};
    std::function<void()> onComplete;
    DeterministicClock* clock;

public:
    Timer(DeterministicClock* clk) : clock(clk) {}

    void start(double dur, bool loop = false, std::function<void()> callback = nullptr) {
        startTime = clock->getTime();
        duration = dur;
        running = true;
        looping = loop;
        onComplete = callback;
    }

    void stop() {
        running = false;
    }

    void reset() {
        startTime = clock->getTime();
    }

    bool update() {
        if (!running) return false;

        double elapsed = clock->getTime() - startTime;
        if (elapsed >= duration) {
            if (onComplete) onComplete();

            if (looping) {
                startTime = clock->getTime();
                return true;
            } else {
                running = false;
                return false;
            }
        }
        return true;
    }

    double getElapsed() const {
        return running ? (clock->getTime() - startTime) : 0.0;
    }

    double getRemaining() const {
        return running ? std::max(0.0, duration - getElapsed()) : 0.0;
    }

    double getProgress() const {
        return duration > 0.0 ? (getElapsed() / duration) : 1.0;
    }

    bool isRunning() const { return running; }
    bool isComplete() const { return !running && getElapsed() >= duration; }
};

// ============ SCHEDULER FOR TIMED EVENTS ============

class EventScheduler {
private:
    struct ScheduledEvent {
        double triggerTime;
        std::string eventId;
        std::function<void()> callback;
        bool repeating;
        double interval;

        bool operator>(const ScheduledEvent& other) const {
            return triggerTime > other.triggerTime; // Min-heap ordering
        }
    };

    std::priority_queue<ScheduledEvent, std::vector<ScheduledEvent>, std::greater<ScheduledEvent>> eventQueue;
    DeterministicClock* clock;
    int nextEventId{1};

public:
    EventScheduler(DeterministicClock* clk) : clock(clk) {}

    std::string scheduleEvent(double delay, std::function<void()> callback, bool repeat = false, double interval = 0.0) {
        std::string id = "evt_" + std::to_string(nextEventId++);
        ScheduledEvent event;
        event.triggerTime = clock->getTime() + delay;
        event.eventId = id;
        event.callback = callback;
        event.repeating = repeat;
        event.interval = interval > 0.0 ? interval : delay;

        eventQueue.push(event);
        return id;
    }

    void update() {
        double currentTime = clock->getTime();

        while (!eventQueue.empty() && eventQueue.top().triggerTime <= currentTime) {
            ScheduledEvent event = eventQueue.top();
            eventQueue.pop();

            // Execute callback
            if (event.callback) {
                event.callback();
            }

            // Re-schedule if repeating
            if (event.repeating) {
                event.triggerTime = currentTime + event.interval;
                eventQueue.push(event);
            }
        }
    }

    void clear() {
        eventQueue = std::priority_queue<ScheduledEvent, std::vector<ScheduledEvent>, std::greater<ScheduledEvent>>();
    }

    int getPendingEventCount() const {
        return eventQueue.size();
    }
};

// ============ INTERPOLATION HELPERS ============

template<typename T>
T lerp(const T& a, const T& b, double t) {
    return a + (b - a) * t;
}

template<typename T>
T smoothstep(const T& a, const T& b, double t) {
    t = t * t * (3.0 - 2.0 * t);
    return lerp(a, b, t);
}

// Ease functions for smooth animations
namespace Easing {
    inline double easeInQuad(double t) { return t * t; }
    inline double easeOutQuad(double t) { return 1.0 - (1.0 - t) * (1.0 - t); }
    inline double easeInOutQuad(double t) {
        return t < 0.5 ? 2.0 * t * t : 1.0 - 2.0 * (1.0 - t) * (1.0 - t);
    }

    inline double easeInCubic(double t) { return t * t * t; }
    inline double easeOutCubic(double t) { double f = (1.0 - t); return 1.0 - f * f * f; }

    inline double easeInSine(double t) { return 1.0 - std::cos(t * M_PI * 0.5); }
    inline double easeOutSine(double t) { return std::sin(t * M_PI * 0.5); }
}

} // namespace NEXUS
