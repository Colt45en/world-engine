/**
 * Layer 0 Kill Switch - Emergency System Termination
 * IMMUTABLE - Cannot be bypassed or modified at runtime
 * Part of sealed Overseer layer
 */

#ifndef LAYER0_KILL_SWITCH_HPP
#define LAYER0_KILL_SWITCH_HPP

#include <string>
#include <chrono>
#include <functional>

namespace Layer0 {

enum class KillReason {
    MEMORY_CORRUPTION,
    RUNAWAY_PROCESS,
    DATA_POISONING,
    LOGIC_DEADLOCK,
    UNSEEABLE_ISSUE,
    HEALTH_MONITOR_TRIGGER,
    WATCHDOG_TIMEOUT,
    MANUAL_TRIGGER,
    UNKNOWN_MODULE_DETECTED
};

class KillSwitch {
private:
    static bool initialized_;
    static bool system_terminated_;
    static std::chrono::steady_clock::time_point last_heartbeat_;
    static std::function<void(const std::string&)> logger_;
    
    // Prevent instantiation - all static methods
    KillSwitch() = delete;
    KillSwitch(const KillSwitch&) = delete;
    KillSwitch& operator=(const KillSwitch&) = delete;

public:
    /**
     * Initialize the kill switch system
     * Called once during Layer 0 initialization
     */
    static void initialize(std::function<void(const std::string&)> log_func = nullptr);
    
    /**
     * Check if system is terminated
     */
    static bool isTerminated() { return system_terminated_; }
    
    /**
     * Emergency termination - cannot be bypassed
     * Immediately freezes all system operations
     */
    static void trigger(KillReason reason, const std::string& context = "");
    
    /**
     * Heartbeat mechanism to detect hangs
     * Must be called regularly by system components
     */
    static void heartbeat();
    
    /**
     * Check if heartbeat is still active (watchdog function)
     */
    static bool isHeartbeatHealthy(int timeout_seconds = 10);
    
    /**
     * Force system preservation for forensics
     * Saves current state before termination
     */
    static void preserveSystemState();
    
    /**
     * Get human-readable kill reason
     */
    static std::string getKillReasonString(KillReason reason);
    
    /**
     * Validate system integrity
     * Returns false if kill switch should be triggered
     */
    static bool validateSystemIntegrity();
    
    /**
     * Register emergency callback
     * Called during termination for cleanup
     */
    static void registerEmergencyCallback(std::function<void()> callback);
    
private:
    static void executeTermination(const std::string& reason_msg);
    static void forensicLog(const std::string& message);
    static std::vector<std::function<void()>> emergency_callbacks_;
};

/**
 * RAII Guard for critical sections
 * Automatically triggers kill switch if destroyed without proper cleanup
 */
class CriticalSectionGuard {
private:
    std::string section_name_;
    bool completed_;
    std::chrono::steady_clock::time_point start_time_;

public:
    explicit CriticalSectionGuard(const std::string& name);
    ~CriticalSectionGuard();
    
    void complete(); // Mark section as successfully completed
    void extend_timeout(int additional_seconds); // Extend allowed time
};

/**
 * Memory corruption detector
 * Monitors heap/stack integrity
 */
class MemoryGuard {
private:
    static constexpr uint32_t CANARY_VALUE = 0xDEADBEEF;
    uint32_t* canaries_;
    size_t canary_count_;

public:
    MemoryGuard(size_t num_canaries = 16);
    ~MemoryGuard();
    
    bool checkIntegrity() const;
    void forceCheck(); // Triggers kill switch if corruption detected
};

} // namespace Layer0

// Global macros for easy usage
#define LAYER0_HEARTBEAT() Layer0::KillSwitch::heartbeat()
#define LAYER0_CRITICAL_SECTION(name) Layer0::CriticalSectionGuard _guard(name)
#define LAYER0_COMPLETE_SECTION() _guard.complete()
#define LAYER0_KILL_IF(condition, reason, context) \
    do { if (condition) Layer0::KillSwitch::trigger(reason, context); } while(0)

#endif // LAYER0_KILL_SWITCH_HPP