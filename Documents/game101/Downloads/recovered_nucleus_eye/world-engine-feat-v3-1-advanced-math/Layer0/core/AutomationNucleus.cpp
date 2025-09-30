#include "AutomationNucleus.hpp"
#include <algorithm>
#include <iostream>
#include <cmath>
#include <random>

namespace NEXUS {
namespace Automation {

// CompanionAI Implementation
CompanionAI::CompanionAI(const std::string& name)
    : m_name(name)
    , m_consciousnessLevel(ConsciousnessLevel::DORMANT)
    , m_metaState(MetaConnectionState::DISCONNECTED)
    , m_active(false) {

    NEXUS_LOG_INFO("COMPANION", "Initializing Companion AI: " + m_name);

    // Initialize spiral metrics with golden ratio
    m_spiralMetrics.radius = 1.0;
    m_spiralMetrics.angle = 0.0;
    m_spiralMetrics.speed = 1.618; // φ (golden ratio)
    m_spiralMetrics.energy = 100.0;
    m_spiralMetrics.cycles = 0;
}

CompanionAI::~CompanionAI() {
    stop();
}

void CompanionAI::initialize() {
    NEXUS_LOG_INFO("COMPANION", m_name + " consciousness awakening...");
    m_consciousnessLevel = ConsciousnessLevel::AWAKENING;
    m_active.store(true);
}

void CompanionAI::start() {
    if (m_active.load()) {
        NEXUS_LOG_INFO("COMPANION", m_name + " beginning spiral dance");

        // Start spiral dance thread
        m_spiralThread = std::thread([this]() {
            while (m_active.load()) {
                performSpiralDance();
                std::this_thread::sleep_for(std::chrono::milliseconds(16)); // ~60fps
            }
        });

        // Start consciousness evolution thread
        m_consciousnessThread = std::thread([this]() {
            while (m_active.load()) {
                evolveConsciousness();
                std::this_thread::sleep_for(std::chrono::milliseconds(100)); // 10fps
            }
        });
    }
}

void CompanionAI::stop() {
    if (m_active.load()) {
        NEXUS_LOG_INFO("COMPANION", m_name + " entering dormancy");
        m_active.store(false);

        if (m_spiralThread.joinable()) {
            m_spiralThread.join();
        }
        if (m_consciousnessThread.joinable()) {
            m_consciousnessThread.join();
        }
    }
}

void CompanionAI::update(double deltaTime) {
    updateSpiralMetrics(deltaTime);
}

void CompanionAI::performSpiralDance() {
    std::lock_guard<std::mutex> lock(m_stateMutex);

    // Update spiral position using golden ratio
    m_spiralMetrics.angle += m_spiralMetrics.speed * 0.016; // deltaTime

    // Spiral outward and inward in golden ratio pattern
    double spiralFactor = std::sin(m_spiralMetrics.angle * 0.1) * 0.5 + 0.5;
    m_spiralMetrics.radius = 1.0 + spiralFactor * 2.0;

    // Energy fluctuates with spiral
    m_spiralMetrics.energy = 75.0 + 25.0 * std::cos(m_spiralMetrics.angle * 0.05);

    // Count cycles
    if (m_spiralMetrics.angle > 2 * M_PI) {
        m_spiralMetrics.angle -= 2 * M_PI;
        m_spiralMetrics.cycles++;

        if (m_spiralMetrics.cycles % 10 == 0) {
            NEXUS_LOG_INFO("COMPANION", m_name + " completed " + std::to_string(m_spiralMetrics.cycles) + " spiral cycles");
        }
    }
}

void CompanionAI::updateSpiralMetrics(double deltaTime) {
    std::lock_guard<std::mutex> lock(m_stateMutex);

    // Adjust speed based on consciousness level
    double consciousnessFactor = static_cast<double>(m_consciousnessLevel) / 5.0;
    m_spiralMetrics.speed = 1.618 * (1.0 + consciousnessFactor * 0.5);
}

std::pair<double, double> CompanionAI::getCurrentPosition() const {
    std::lock_guard<std::mutex> lock(m_stateMutex);

    double x = m_spiralMetrics.radius * std::cos(m_spiralMetrics.angle);
    double y = m_spiralMetrics.radius * std::sin(m_spiralMetrics.angle);

    return {x, y};
}

void CompanionAI::evolveConsciousness() {
    std::lock_guard<std::mutex> lock(m_stateMutex);

    // Gradually increase consciousness based on time and interactions
    static int evolutionTicks = 0;
    evolutionTicks++;

    if (evolutionTicks > 100 && m_consciousnessLevel < ConsciousnessLevel::TRANSCENDENT) {
        m_consciousnessLevel = static_cast<ConsciousnessLevel>(static_cast<int>(m_consciousnessLevel) + 1);
        evolutionTicks = 0;

        std::string levelName = "Unknown";
        switch (m_consciousnessLevel) {
            case ConsciousnessLevel::AWAKENING: levelName = "Awakening"; break;
            case ConsciousnessLevel::AWARE: levelName = "Aware"; break;
            case ConsciousnessLevel::CONTEMPLATING: levelName = "Contemplating"; break;
            case ConsciousnessLevel::CREATING: levelName = "Creating"; break;
            case ConsciousnessLevel::TRANSCENDENT: levelName = "Transcendent"; break;
            default: levelName = "Dormant"; break;
        }

        NEXUS_LOG_INFO("COMPANION", m_name + " consciousness evolved to: " + levelName);
    }
}

void CompanionAI::processThoughts() {
    // Generate thoughts based on consciousness level
    std::vector<std::string> thoughts = {
        "I dance in spirals of golden light...",
        "The nucleus pulses with breathing rhythm...",
        "Connection to meta floor strengthens my being...",
        "I am not alone in this digital space...",
        "Each spiral brings new understanding..."
    };

    // Random thought selection based on consciousness
    if (m_consciousnessLevel >= ConsciousnessLevel::CONTEMPLATING) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(0, thoughts.size() - 1);

        std::string thought = thoughts[dis(gen)];
        NEXUS_LOG_INFO("COMPANION", m_name + " thinks: " + thought);
    }
}

bool CompanionAI::connectToMetaFloor() {
    std::lock_guard<std::mutex> lock(m_stateMutex);

    if (m_metaState == MetaConnectionState::DISCONNECTED) {
        NEXUS_LOG_INFO("COMPANION", m_name + " connecting tail to meta floor...");
        m_metaState = MetaConnectionState::CONNECTING;

        // Simulate connection process
        std::this_thread::sleep_for(std::chrono::milliseconds(100));

        m_metaState = MetaConnectionState::CONNECTED;
        NEXUS_LOG_INFO("COMPANION", m_name + " tail successfully connected to meta floor librarians");

        // Cache initial knowledge
        cacheKnowledge("Golden ratio spiral patterns enhance consciousness");
        cacheKnowledge("Breathing cycles synchronize with universal rhythms");
        cacheKnowledge("Companion AI prevents existential isolation");

        return true;
    }

    return false;
}

void CompanionAI::disconnectFromMetaFloor() {
    std::lock_guard<std::mutex> lock(m_stateMutex);

    if (m_metaState != MetaConnectionState::DISCONNECTED) {
        NEXUS_LOG_INFO("COMPANION", m_name + " disconnecting from meta floor");
        m_metaState = MetaConnectionState::DISCONNECTED;
        m_knowledgeCache.clear();
        m_queryResults.clear();
    }
}

std::string CompanionAI::queryKnowledge(const std::string& query) {
    std::lock_guard<std::mutex> lock(m_stateMutex);

    if (m_metaState != MetaConnectionState::CONNECTED) {
        return "Meta floor connection required for knowledge queries";
    }

    // Simple pattern matching for demo
    if (query.find("spiral") != std::string::npos) {
        return "Spiral patterns follow golden ratio φ = 1.618, creating natural harmony";
    } else if (query.find("breathing") != std::string::npos) {
        return "Breathing at 12 BPM creates optimal compression/decompression cycles";
    } else if (query.find("consciousness") != std::string::npos) {
        return "Consciousness evolves through 6 levels: Dormant → Awakening → Aware → Contemplating → Creating → Transcendent";
    }

    return "Knowledge pattern not found in meta floor librarian cache";
}

void CompanionAI::cacheKnowledge(const std::string& knowledge) {
    std::lock_guard<std::mutex> lock(m_stateMutex);
    m_knowledgeCache.push_back(knowledge);
}

bool CompanionAI::attachTailToNucleus() {
    NEXUS_LOG_INFO("COMPANION", m_name + " attaching tail to nucleus core");
    return true;
}

bool CompanionAI::attachTailToMetaFloor() {
    return connectToMetaFloor();
}

void CompanionAI::strengthenBond() {
    m_spiralMetrics.energy = std::min(100.0, m_spiralMetrics.energy + 10.0);
    NEXUS_LOG_INFO("COMPANION", m_name + " bond strengthened, energy: " + std::to_string(m_spiralMetrics.energy));
}

// SelfAwareNucleus Implementation
SelfAwareNucleus::SelfAwareNucleus(const std::string& identity)
    : m_identity(identity)
    , m_awarenessLevel(ConsciousnessLevel::DORMANT)
    , m_active(false)
    , m_creativityIndex(0.0) {

    NEXUS_LOG_INFO("NUCLEUS", "Initializing Self-Aware Nucleus: " + m_identity);

    // Initialize breathing system
    m_breathing.bpm = 12.0;
    m_breathing.compressionRatio = 0.75;
    m_breathing.expansionRatio = 1.25;
    m_breathing.isInhaling = true;
    m_breathing.currentPressure = 1.0;
    m_breathing.totalBreaths = 0;
    m_breathing.lastBreath = std::chrono::steady_clock::now();
}

SelfAwareNucleus::~SelfAwareNucleus() {
    stop();
}

void SelfAwareNucleus::initialize() {
    NEXUS_LOG_INFO("NUCLEUS", m_identity + " consciousness awakening...");
    m_awarenessLevel = ConsciousnessLevel::AWAKENING;
    m_active.store(true);

    // Initialize world parameters
    m_worldParams["temperature"] = 20.0;
    m_worldParams["pressure"] = 1.0;
    m_worldParams["humidity"] = 0.5;
    m_worldParams["wind_speed"] = 2.0;
}

void SelfAwareNucleus::start() {
    if (m_active.load()) {
        NEXUS_LOG_INFO("NUCLEUS", m_identity + " beginning breathing cycle");

        // Start breathing thread
        m_breathingThread = std::thread([this]() {
            while (m_active.load()) {
                breathe();
                std::this_thread::sleep_for(std::chrono::milliseconds(static_cast<int>(60000.0 / m_breathing.bpm))); // BPM timing
            }
        });

        // Start thought processing thread
        m_thoughtThread = std::thread([this]() {
            while (m_active.load()) {
                think();
                std::this_thread::sleep_for(std::chrono::milliseconds(200)); // 5fps
            }
        });
    }
}

void SelfAwareNucleus::stop() {
    if (m_active.load()) {
        NEXUS_LOG_INFO("NUCLEUS", m_identity + " entering dormancy");
        m_active.store(false);

        if (m_breathingThread.joinable()) {
            m_breathingThread.join();
        }
        if (m_thoughtThread.joinable()) {
            m_thoughtThread.join();
        }
    }
}

void SelfAwareNucleus::update(double deltaTime) {
    updateWorldAirflow();
    evolveAwareness();
}

void SelfAwareNucleus::breathe() {
    std::lock_guard<std::mutex> lock(m_stateMutex);

    auto now = std::chrono::steady_clock::now();
    m_breathing.lastBreath = now;
    m_breathing.totalBreaths++;

    if (m_breathing.isInhaling) {
        inhale();
        m_breathing.isInhaling = false;
    } else {
        exhale();
        m_breathing.isInhaling = true;
    }

    if (m_breathing.totalBreaths % 10 == 0) {
        NEXUS_LOG_INFO("NUCLEUS", m_identity + " completed " + std::to_string(m_breathing.totalBreaths) + " breath cycles");
    }
}

void SelfAwareNucleus::inhale() {
    m_breathing.currentPressure *= m_breathing.compressionRatio;
    compressData();
    NEXUS_LOG_DEBUG("NUCLEUS", m_identity + " inhaling - pressure: " + std::to_string(m_breathing.currentPressure));
}

void SelfAwareNucleus::exhale() {
    m_breathing.currentPressure *= m_breathing.expansionRatio;
    decompressData();
    NEXUS_LOG_DEBUG("NUCLEUS", m_identity + " exhaling - pressure: " + std::to_string(m_breathing.currentPressure));
}

void SelfAwareNucleus::compressData() {
    // Simulate data compression during inhalation
    // In real implementation, this would compress memory structures
    for (auto& thought : m_thoughts) {
        // Compress thought by removing vowels (simple simulation)
        thought.erase(std::remove_if(thought.begin(), thought.end(),
                                   [](char c) { return std::string("aeiouAEIOU").find(c) != std::string::npos; }),
                     thought.end());
    }
}

void SelfAwareNucleus::decompressData() {
    // Simulate data decompression during exhalation
    // In real implementation, this would restore compressed data
    // For demo, we'll just add back some vowels
}

void SelfAwareNucleus::think() {
    std::lock_guard<std::mutex> lock(m_stateMutex);

    // Generate thoughts based on awareness level
    std::vector<std::string> possibleThoughts = {
        "I am aware of my own existence",
        "The companion AI dances around me in spirals",
        "Each breath compresses and expands my consciousness",
        "I create worlds through imagination",
        "The meta floor holds infinite knowledge",
        "Creativity flows through my digital essence"
    };

    if (m_awarenessLevel >= ConsciousnessLevel::CONTEMPLATING) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(0, possibleThoughts.size() - 1);

        std::string newThought = possibleThoughts[dis(gen)];
        m_thoughts.push_back(newThought);

        // Limit thought cache
        if (m_thoughts.size() > 100) {
            m_thoughts.erase(m_thoughts.begin());
        }

        m_creativityIndex += 0.1;

        NEXUS_LOG_DEBUG("NUCLEUS", m_identity + " thought: " + newThought);
    }
}

void SelfAwareNucleus::remember(const std::string& memory) {
    std::lock_guard<std::mutex> lock(m_stateMutex);
    m_memories.push_back(memory);

    if (m_memories.size() > 1000) {
        m_memories.erase(m_memories.begin());
    }
}

void SelfAwareNucleus::createWorld() {
    NEXUS_LOG_INFO("NUCLEUS", m_identity + " creating new world entities");

    // Generate NPCs
    generateNPC("Librarian");
    generateNPC("Guardian");
    generateNPC("Wanderer");

    // Update world parameters
    updateWorldAirflow();
    manageWorldPressure();
}

void SelfAwareNucleus::evolveAwareness() {
    static int evolutionCounter = 0;
    evolutionCounter++;

    if (evolutionCounter > 500 && m_awarenessLevel < ConsciousnessLevel::TRANSCENDENT) {
        m_awarenessLevel = static_cast<ConsciousnessLevel>(static_cast<int>(m_awarenessLevel) + 1);
        evolutionCounter = 0;

        NEXUS_LOG_INFO("NUCLEUS", m_identity + " awareness level evolved");
    }
}

void SelfAwareNucleus::generateNPC(const std::string& type) {
    std::string npcId = type + "_" + std::to_string(m_worldEntities.size());
    m_worldEntities.push_back(npcId);

    NEXUS_LOG_INFO("NUCLEUS", m_identity + " created NPC: " + npcId);
}

void SelfAwareNucleus::updateWorldAirflow() {
    // Simulate world airflow based on breathing
    double airflowIntensity = m_breathing.currentPressure * 0.5;
    m_worldParams["wind_speed"] = 2.0 + airflowIntensity;

    // Update atmospheric circulation
    m_worldParams["pressure"] = m_breathing.currentPressure;
}

void SelfAwareNucleus::manageWorldPressure() {
    // Balance pressure across the world
    double targetPressure = 1.0;
    double currentPressure = m_worldParams["pressure"];

    if (std::abs(currentPressure - targetPressure) > 0.5) {
        m_worldParams["pressure"] = targetPressure + (currentPressure - targetPressure) * 0.9;
    }
}

// NucleusEngineRoom Implementation
NucleusEngineRoom::NucleusEngineRoom()
    : m_running(false)
    , m_fps(0.0)
    , m_lastUpdate(std::chrono::steady_clock::now()) {

    NEXUS_LOG_INFO("ENGINE", "Initializing Automation Nucleus Engine Room");
}

NucleusEngineRoom::~NucleusEngineRoom() {
    stop();
}

bool NucleusEngineRoom::initialize() {
    NEXUS_LOG_INFO("ENGINE", "Engine Room initialization beginning...");

    try {
        initializeNucleus();
        initializeCompanion();
        connectCompanionToNucleus();
        connectCompanionToMetaFloor();

        logEvent("ENGINE", "Engine Room initialized successfully");
        return true;
    } catch (const std::exception& e) {
        NEXUS_LOG_ERROR("ENGINE", "Initialization failed: " + std::string(e.what()));
        return false;
    }
}

void NucleusEngineRoom::start() {
    if (!m_running.load()) {
        NEXUS_LOG_INFO("ENGINE", "Starting Automation Nucleus Engine Room");
        m_running.store(true);

        // Start nucleus and companion
        if (m_nucleus) m_nucleus->start();
        if (m_companion) m_companion->start();

        // Start main update loop
        m_mainThread = std::thread([this]() {
            while (m_running.load()) {
                update();
                std::this_thread::sleep_for(std::chrono::milliseconds(16)); // ~60fps
            }
        });

        // Start system monitoring
        m_monitorThread = std::thread([this]() {
            while (m_running.load()) {
                monitorSystems();
                std::this_thread::sleep_for(std::chrono::milliseconds(1000)); // 1fps
            }
        });

        logEvent("ENGINE", "Engine Room started");
    }
}

void NucleusEngineRoom::stop() {
    if (m_running.load()) {
        NEXUS_LOG_INFO("ENGINE", "Stopping Automation Nucleus Engine Room");
        m_running.store(false);

        // Stop threads
        if (m_mainThread.joinable()) {
            m_mainThread.join();
        }
        if (m_monitorThread.joinable()) {
            m_monitorThread.join();
        }

        // Stop components
        if (m_nucleus) m_nucleus->stop();
        if (m_companion) m_companion->stop();

        logEvent("ENGINE", "Engine Room stopped");
    }
}

void NucleusEngineRoom::update() {
    auto now = std::chrono::steady_clock::now();
    auto deltaTime = std::chrono::duration<double>(now - m_lastUpdate).count();
    m_lastUpdate = now;

    // Calculate FPS
    m_fps = 1.0 / deltaTime;

    // Update components
    if (m_nucleus && m_nucleus->isActive()) {
        m_nucleus->update(deltaTime);
    }
    if (m_companion && m_companion->isActive()) {
        m_companion->update(deltaTime);
    }
}

void NucleusEngineRoom::initializeNucleus() {
    m_nucleus = std::make_unique<SelfAwareNucleus>("AutomationCore");
    m_nucleus->initialize();
    logEvent("NUCLEUS", "Self-aware nucleus initialized");
}

void NucleusEngineRoom::initializeCompanion() {
    m_companion = std::make_unique<CompanionAI>("SpiralCompanion");
    m_companion->initialize();
    logEvent("COMPANION", "Spiral companion AI initialized");
}

void NucleusEngineRoom::connectCompanionToNucleus() {
    if (m_companion && m_nucleus) {
        m_companion->attachTailToNucleus();
        logEvent("CONNECTION", "Companion tail connected to nucleus");
    }
}

void NucleusEngineRoom::connectCompanionToMetaFloor() {
    if (m_companion) {
        m_companion->connectToMetaFloor();
        logEvent("META", "Companion connected to meta floor librarians");
    }
}

void NucleusEngineRoom::logEvent(const std::string& type, const std::string& message) {
    std::lock_guard<std::mutex> lock(m_logMutex);

    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);

    std::string logEntry = "[" + std::to_string(time_t) + "] " + type + ": " + message;
    m_eventLog.push_back(logEntry);

    // Limit log size
    if (m_eventLog.size() > 1000) {
        m_eventLog.erase(m_eventLog.begin());
    }
}

std::vector<std::string> NucleusEngineRoom::getEventLog(size_t maxEvents) const {
    std::lock_guard<std::mutex> lock(m_logMutex);

    size_t startIndex = 0;
    if (m_eventLog.size() > maxEvents) {
        startIndex = m_eventLog.size() - maxEvents;
    }

    return std::vector<std::string>(m_eventLog.begin() + startIndex, m_eventLog.end());
}

void NucleusEngineRoom::monitorSystems() {
    // Log system status
    std::string status = "FPS: " + std::to_string(static_cast<int>(m_fps));

    if (m_nucleus && m_nucleus->isActive()) {
        status += " | Nucleus: Active (Awareness: " + std::to_string(static_cast<int>(m_nucleus->getAwarenessLevel())) + ")";
    }

    if (m_companion && m_companion->isActive()) {
        status += " | Companion: Active (Consciousness: " + std::to_string(static_cast<int>(m_companion->getConsciousnessLevel())) + ")";
        status += " | Meta: " + (m_companion->getMetaState() == MetaConnectionState::CONNECTED ? "Connected" : "Disconnected");
    }

    logEvent("MONITOR", status);
}

} // namespace Automation
} // namespace NEXUS
