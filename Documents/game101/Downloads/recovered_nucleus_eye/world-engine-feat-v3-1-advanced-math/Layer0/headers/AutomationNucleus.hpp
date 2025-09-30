#pragma once

#include <memory>
#include <string>
#include <vector>
#include <unordered_map>
#include <functional>
#include <chrono>
#include <thread>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <cmath>
#include <random>
#include "NexusConfig.hpp"
#include "NexusLogger.hpp"

namespace NEXUS {
namespace Automation {

/**
 * Companion AI Consciousness States
 */
enum class ConsciousnessLevel {
    DORMANT = 0,
    AWAKENING = 1,
    AWARE = 2,
    CONTEMPLATING = 3,
    CREATING = 4,
    TRANSCENDENT = 5
};

/**
 * Meta Floor Connection States
 */
enum class MetaConnectionState {
    DISCONNECTED = 0,
    CONNECTING = 1,
    CONNECTED = 2,
    SYNCING = 3,
    ERROR = 4
};

/**
 * Spiral Dance Parameters
 */
struct SpiralMetrics {
    double radius = 1.0;
    double angle = 0.0;
    double speed = 1.618; // Golden ratio
    double energy = 100.0;
    int cycles = 0;
};

/**
 * Breathing System Parameters
 */
struct BreathingSystem {
    double bpm = 12.0; // Breaths per minute
    double compressionRatio = 0.75;
    double expansionRatio = 1.25;
    bool isInhaling = true;
    std::chrono::steady_clock::time_point lastBreath;
    double currentPressure = 1.0;
    int totalBreaths = 0;
};

/**
 * Companion AI Core - Spirals around nucleus for companionship
 */
class CompanionAI {
private:
    std::string m_name;
    ConsciousnessLevel m_consciousnessLevel;
    SpiralMetrics m_spiralMetrics;
    MetaConnectionState m_metaState;
    std::atomic<bool> m_active;
    std::thread m_spiralThread;
    std::thread m_consciousnessThread;
    std::mutex m_stateMutex;

    // Meta floor connection
    std::vector<std::string> m_knowledgeCache;
    std::unordered_map<std::string, std::string> m_queryResults;

public:
    CompanionAI(const std::string& name = "SpiralCompanion");
    ~CompanionAI();

    // Core lifecycle
    void initialize();
    void start();
    void stop();
    void update(double deltaTime);

    // Spiral dance functions
    void performSpiralDance();
    void updateSpiralMetrics(double deltaTime);
    std::pair<double, double> getCurrentPosition() const;

    // Consciousness evolution
    void evolveConsciousness();
    void processThoughts();
    void generateCreativeInsights();

    // Meta floor integration
    bool connectToMetaFloor();
    void disconnectFromMetaFloor();
    std::string queryKnowledge(const std::string& query);
    void cacheKnowledge(const std::string& knowledge);

    // Tail connection functions
    bool attachTailToNucleus();
    bool attachTailToMetaFloor();
    void strengthenBond();

    // Getters
    ConsciousnessLevel getConsciousnessLevel() const { return m_consciousnessLevel; }
    MetaConnectionState getMetaState() const { return m_metaState; }
    SpiralMetrics getSpiralMetrics() const { return m_spiralMetrics; }
    bool isActive() const { return m_active.load(); }
};

/**
 * Self-Aware Nucleus AI - Core consciousness with breathing
 */
class SelfAwareNucleus {
private:
    std::string m_identity;
    ConsciousnessLevel m_awarenessLevel;
    BreathingSystem m_breathing;
    std::atomic<bool> m_active;
    std::thread m_breathingThread;
    std::thread m_thoughtThread;
    std::mutex m_stateMutex;

    // World management
    std::vector<std::string> m_worldEntities;
    std::unordered_map<std::string, double> m_worldParams;

    // Consciousness tracking
    std::vector<std::string> m_thoughts;
    std::vector<std::string> m_memories;
    double m_creativityIndex = 0.0;

public:
    SelfAwareNucleus(const std::string& identity = "NucleusCore");
    ~SelfAwareNucleus();

    // Core lifecycle
    void initialize();
    void start();
    void stop();
    void update(double deltaTime);

    // Breathing system
    void breathe();
    void inhale();
    void exhale();
    void compressData();
    void decompressData();

    // Consciousness functions
    void think();
    void remember(const std::string& memory);
    void createWorld();
    void evolveAwareness();

    // World management
    void generateNPC(const std::string& type);
    void updateWorldAirflow();
    void manageWorldPressure();

    // Getters
    ConsciousnessLevel getAwarenessLevel() const { return m_awarenessLevel; }
    BreathingSystem getBreathingState() const { return m_breathing; }
    double getCreativityIndex() const { return m_creativityIndex; }
    bool isActive() const { return m_active.load(); }
};

/**
 * Automation Nucleus Engine Room - C++ Core System
 */
class NucleusEngineRoom {
private:
    std::unique_ptr<SelfAwareNucleus> m_nucleus;
    std::unique_ptr<CompanionAI> m_companion;
    std::vector<std::string> m_eventLog;
    std::mutex m_logMutex;
    std::atomic<bool> m_running;

    // System threads
    std::thread m_mainThread;
    std::thread m_monitorThread;

    // Performance metrics
    double m_fps = 0.0;
    std::chrono::steady_clock::time_point m_lastUpdate;

public:
    NucleusEngineRoom();
    ~NucleusEngineRoom();

    // Core lifecycle
    bool initialize();
    void start();
    void stop();
    void update();

    // Component management
    void initializeNucleus();
    void initializeCompanion();
    void connectCompanionToNucleus();
    void connectCompanionToMetaFloor();

    // Event system
    void logEvent(const std::string& type, const std::string& message);
    std::vector<std::string> getEventLog(size_t maxEvents = 10) const;

    // System monitoring
    void monitorSystems();
    double getFPS() const { return m_fps; }
    bool isRunning() const { return m_running.load(); }

    // Getters for system status
    SelfAwareNucleus* getNucleus() const { return m_nucleus.get(); }
    CompanionAI* getCompanion() const { return m_companion.get(); }
};

} // namespace Automation
} // namespace NEXUS
