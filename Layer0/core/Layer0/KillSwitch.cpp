/**
 * Layer 0 Kill Switch Implementation
 * IMMUTABLE - Part of sealed Overseer layer
 */

#include "Layer0/KillSwitch.hpp"
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <thread>
#include <atomic>
#include <vector>
#include <mutex>

namespace Layer0 {

// Static member definitions
bool KillSwitch::initialized_ = false;
bool KillSwitch::system_terminated_ = false;
std::chrono::steady_clock::time_point KillSwitch::last_heartbeat_ = std::chrono::steady_clock::now();
std::function<void(const std::string&)> KillSwitch::logger_ = nullptr;
std::vector<std::function<void()>> KillSwitch::emergency_callbacks_;

void KillSwitch::initialize(std::function<void(const std::string&)> log_func) {
    if (initialized_) return; // Already initialized
    
    logger_ = log_func ? log_func : [](const std::string& msg) {
        std::cerr << "[LAYER0:KILLSWITCH] " << msg << std::endl;
    };
    
    last_heartbeat_ = std::chrono::steady_clock::now();
    system_terminated_ = false;
    initialized_ = true;
    
    forensicLog("Kill Switch initialized - System protection active");
}

void KillSwitch::trigger(KillReason reason, const std::string& context) {
    if (system_terminated_) return; // Already terminated
    
    std::string reason_msg = getKillReasonString(reason);
    if (!context.empty()) {
        reason_msg += " | Context: " + context;
    }
    
    forensicLog("KILL SWITCH TRIGGERED: " + reason_msg);
    executeTermination(reason_msg);
}

void KillSwitch::heartbeat() {
    if (system_terminated_) return;
    last_heartbeat_ = std::chrono::steady_clock::now();
}

bool KillSwitch::isHeartbeatHealthy(int timeout_seconds) {
    if (system_terminated_) return false;
    
    auto now = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - last_heartbeat_);
    return elapsed.count() < timeout_seconds;
}

void KillSwitch::preserveSystemState() {
    forensicLog("Preserving system state for forensic analysis");
    
    try {
        // Create forensic dump
        std::ofstream dump("system_state_dump.log", std::ios::app);
        if (dump.is_open()) {
            dump << "\n=== SYSTEM STATE PRESERVATION ===\n";
            dump << "Timestamp: " << std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::system_clock::now().time_since_epoch()).count() << "ms\n";
            dump << "Thread ID: " << std::this_thread::get_id() << "\n";
            dump << "Terminated: " << (system_terminated_ ? "YES" : "NO") << "\n";
            
            auto now = std::chrono::steady_clock::now();
            auto heartbeat_age = std::chrono::duration_cast<std::chrono::seconds>(now - last_heartbeat_);
            dump << "Last heartbeat: " << heartbeat_age.count() << " seconds ago\n";
            
            dump << "=== END PRESERVATION ===\n\n";
            dump.close();
        }
    } catch (...) {
        // Even forensic logging failed - system is seriously compromised
        std::cerr << "[LAYER0:CRITICAL] Failed to preserve system state\n";
    }
}

std::string KillSwitch::getKillReasonString(KillReason reason) {
    switch (reason) {
        case KillReason::MEMORY_CORRUPTION: return "MEMORY_CORRUPTION";
        case KillReason::RUNAWAY_PROCESS: return "RUNAWAY_PROCESS";
        case KillReason::DATA_POISONING: return "DATA_POISONING";
        case KillReason::LOGIC_DEADLOCK: return "LOGIC_DEADLOCK";
        case KillReason::UNSEEABLE_ISSUE: return "UNSEEABLE_ISSUE";
        case KillReason::HEALTH_MONITOR_TRIGGER: return "HEALTH_MONITOR_TRIGGER";
        case KillReason::WATCHDOG_TIMEOUT: return "WATCHDOG_TIMEOUT";
        case KillReason::MANUAL_TRIGGER: return "MANUAL_TRIGGER";
        case KillReason::UNKNOWN_MODULE_DETECTED: return "UNKNOWN_MODULE_DETECTED";
        default: return "UNKNOWN_REASON";
    }
}

bool KillSwitch::validateSystemIntegrity() {
    if (system_terminated_) return false;
    
    // Check heartbeat health
    if (!isHeartbeatHealthy()) {
        forensicLog("System integrity check failed: Heartbeat timeout");
        return false;
    }
    
    // Check initialization state
    if (!initialized_) {
        forensicLog("System integrity check failed: Not properly initialized");
        return false;
    }
    
    return true;
}

void KillSwitch::registerEmergencyCallback(std::function<void()> callback) {
    if (callback) {
        emergency_callbacks_.push_back(callback);
    }
}

void KillSwitch::executeTermination(const std::string& reason_msg) {
    system_terminated_ = true;
    
    forensicLog("SYSTEM TERMINATION INITIATED: " + reason_msg);
    
    // Preserve system state before shutdown
    preserveSystemState();
    
    // Execute emergency callbacks
    for (auto& callback : emergency_callbacks_) {
        try {
            callback();
        } catch (...) {
            forensicLog("Emergency callback failed during termination");
        }
    }
    
    // Final log
    forensicLog("SYSTEM TERMINATED - All operations halted");
    
    // Hard stop - no recovery possible
    std::exit(EXIT_FAILURE);
}

void KillSwitch::forensicLog(const std::string& message) {
    if (logger_) {
        try {
            logger_(message);
        } catch (...) {
            // Logger itself failed - use stderr as last resort
            std::cerr << "[LAYER0:FORENSIC] " << message << std::endl;
        }
    } else {
        std::cerr << "[LAYER0:FORENSIC] " << message << std::endl;
    }
}

// CriticalSectionGuard Implementation
CriticalSectionGuard::CriticalSectionGuard(const std::string& name)
    : section_name_(name), completed_(false), start_time_(std::chrono::steady_clock::now()) {
    KillSwitch::forensicLog("Entering critical section: " + section_name_);
}

CriticalSectionGuard::~CriticalSectionGuard() {
    if (!completed_) {
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::steady_clock::now() - start_time_);
        
        std::string error_msg = "Critical section '" + section_name_ + 
                               "' not completed after " + std::to_string(elapsed.count()) + " seconds";
        
        KillSwitch::trigger(KillReason::LOGIC_DEADLOCK, error_msg);
    }
}

void CriticalSectionGuard::complete() {
    completed_ = true;
    KillSwitch::forensicLog("Completed critical section: " + section_name_);
}

void CriticalSectionGuard::extend_timeout(int additional_seconds) {
    start_time_ += std::chrono::seconds(additional_seconds);
    KillSwitch::forensicLog("Extended timeout for critical section: " + section_name_);
}

// MemoryGuard Implementation
MemoryGuard::MemoryGuard(size_t num_canaries) : canary_count_(num_canaries) {
    canaries_ = new uint32_t[canary_count_];
    for (size_t i = 0; i < canary_count_; ++i) {
        canaries_[i] = CANARY_VALUE;
    }
}

MemoryGuard::~MemoryGuard() {
    forceCheck(); // Final integrity check
    delete[] canaries_;
}

bool MemoryGuard::checkIntegrity() const {
    for (size_t i = 0; i < canary_count_; ++i) {
        if (canaries_[i] != CANARY_VALUE) {
            return false;
        }
    }
    return true;
}

void MemoryGuard::forceCheck() {
    if (!checkIntegrity()) {
        KillSwitch::trigger(KillReason::MEMORY_CORRUPTION, 
                           "Memory guard detected corruption in canary values");
    }
}

} // namespace Layer0