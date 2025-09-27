/**
 * Layer 0 Overseer Brain - Immutable Root Authority
 * Core identity, global rules, and system governance
 * SEALED - Cannot be modified at runtime
 */

#ifndef LAYER0_OVERSEER_BRAIN_HPP
#define LAYER0_OVERSEER_BRAIN_HPP

#include <string>
#include <map>
#include <vector>
#include <memory>
#include <chrono>
#include "Layer0/KillSwitch.hpp"

namespace Layer0 {

/**
 * System Configuration Constants (Immutable)
 */
struct SystemConstants {
    static constexpr const char* SYSTEM_NAME = "World Engine Studio";
    static constexpr const char* VERSION = "2.0.0-LAYER0";
    static constexpr int MAX_LAYERS = 5;
    static constexpr int HEARTBEAT_INTERVAL_MS = 1000;
    static constexpr int CRITICAL_SECTION_TIMEOUT_S = 30;
    static constexpr size_t MAX_MEMORY_CANARIES = 64;
};

/**
 * Layer Information Structure
 */
struct LayerInfo {
    int layer_id;
    std::string name;
    std::string version;
    bool sealed;
    std::chrono::system_clock::time_point created_at;
    std::vector<std::string> slot_list;
};

/**
 * Slot Registration Entry
 */
struct SlotEntry {
    std::string path;           // e.g., "/safety/KillSwitch"
    int layer_id;              // 0-5
    std::string module_name;   // Class/module name
    bool active;               // Currently loaded
    std::string checksum;      // Integrity verification
};

/**
 * Overseer Brain - Root System Authority
 * Immutable once sealed, manages all system state
 */
class OverseerBrain {
private:
    static bool initialized_;
    static bool sealed_;
    static std::string system_id_;
    static std::chrono::system_clock::time_point birth_time_;
    static std::map<std::string, SlotEntry> slot_registry_;
    static std::map<int, LayerInfo> layer_registry_;
    static MemoryGuard* memory_guard_;

    // Private constructor - singleton pattern
    OverseerBrain() = delete;
    OverseerBrain(const OverseerBrain&) = delete;
    OverseerBrain& operator=(const OverseerBrain&) = delete;

public:
    /**
     * Initialize the Overseer Brain system
     * Can only be called once, before sealing
     */
    static bool initialize();

    /**
     * Seal the Layer 0 system - makes it immutable
     * Once sealed, no modifications are possible
     */
    static bool seal();

    /**
     * Check if system is initialized and sealed
     */
    static bool isSealed() { return sealed_; }
    static bool isInitialized() { return initialized_; }

    /**
     * Get system identity information
     */
    static std::string getSystemId() { return system_id_; }
    static std::string getSystemName() { return SystemConstants::SYSTEM_NAME; }
    static std::string getVersion() { return SystemConstants::VERSION; }
    static std::chrono::system_clock::time_point getBirthTime() { return birth_time_; }

    /**
     * Slot Management (Registration Phase Only)
     */
    static bool registerSlot(const std::string& path, int layer_id,
                           const std::string& module_name, const std::string& checksum = "");
    static bool validateSlotIntegrity();
    static std::vector<SlotEntry> getSlotRegistry();
    static bool isSlotRegistered(const std::string& path);

    /**
     * Layer Management
     */
    static bool registerLayer(int layer_id, const std::string& name, const std::string& version);
    static LayerInfo getLayerInfo(int layer_id);
    static std::vector<LayerInfo> getAllLayers();
    static bool isLayerValid(int layer_id) { return layer_id >= 0 && layer_id <= SystemConstants::MAX_LAYERS; }

    /**
     * Runtime Validation (Always Available)
     */
    static bool validateSystemState();
    static bool enforceLayerBoundaries(int requesting_layer, int target_layer);
    static bool canLayerModify(int source_layer, int target_layer);

    /**
     * Self-Repair Functions (Within Layer 0 Boundaries)
     */
    static bool detectFaults();
    static std::vector<std::string> proposeFixes();
    static bool executeSafeRecovery(const std::string& recovery_action);

    /**
     * Emergency Functions
     */
    static void emergencyShutdown(const std::string& reason);
    static bool isEmergencyActive();
    static std::string getLastEmergencyReason();

    /**
     * Diagnostic Information
     */
    static std::map<std::string, std::string> getDiagnosticInfo();
    static size_t getMemoryUsage();
    static std::chrono::milliseconds getUptime();

    /**
     * Global Rule Enforcement
     */
    static bool enforceCanvasLaw(const std::string& operation, const std::string& context);
    static bool validateModuleAccess(const std::string& module_path, int layer_id);

private:
    static void initializeCoreSystems();
    static void setupMemoryGuard();
    static void setupDefaultSlots();
    static std::string generateSystemId();
    static void logSystemEvent(const std::string& event);
    static bool emergency_active_;
    static std::string last_emergency_reason_;
};

/**
 * Canvas Law Enforcement
 * Ensures all code lives within the layered canvas architecture
 */
class CanvasEnforcer {
public:
    /**
     * Validate that a module exists in the canvas
     */
    static bool validateModuleInCanvas(const std::string& module_path);

    /**
     * Enforce layer hierarchy rules
     */
    static bool enforceLaterHierarchy(int source_layer, int target_layer, const std::string& operation);

    /**
     * Check slot uniqueness
     */
    static bool ensureSlotUniqueness(const std::string& slot_path);

    /**
     * Runtime canvas validation
     */
    static bool performRuntimeValidation();

    /**
     * Report canvas violations
     */
    static void reportViolation(const std::string& violation, const std::string& context);
};

/**
 * System Health Monitor
 * Continuous monitoring of Layer 0 integrity
 */
class HealthMonitor {
private:
    static std::chrono::steady_clock::time_point last_check_;
    static bool monitoring_active_;

public:
    /**
     * Start continuous health monitoring
     */
    static void startMonitoring();

    /**
     * Stop health monitoring
     */
    static void stopMonitoring();

    /**
     * Perform health check
     */
    static bool performHealthCheck();

    /**
     * Get health status
     */
    static std::map<std::string, bool> getHealthStatus();

    /**
     * Register health check callback
     */
    static void registerHealthCallback(std::function<bool()> callback);

private:
    static std::vector<std::function<bool()>> health_callbacks_;
    static void monitoringLoop();
};

} // namespace Layer0

// Global macros for Layer 0 operations
#define LAYER0_REGISTER_SLOT(path, layer, module) \
    Layer0::OverseerBrain::registerSlot(path, layer, module, __FILE__ ":" #__LINE__)

#define LAYER0_ENFORCE_BOUNDARY(source, target) \
    do { if (!Layer0::OverseerBrain::enforceLayerBoundaries(source, target)) \
        Layer0::KillSwitch::trigger(Layer0::KillReason::LOGIC_DEADLOCK, \
                                  "Layer boundary violation: " #source " -> " #target); } while(0)

#define LAYER0_VALIDATE_CANVAS(module) \
    do { if (!Layer0::CanvasEnforcer::validateModuleInCanvas(module)) \
        Layer0::KillSwitch::trigger(Layer0::KillReason::UNKNOWN_MODULE_DETECTED, \
                                  "Module not in canvas: " + std::string(module)); } while(0)

#endif // LAYER0_OVERSEER_BRAIN_HPP
