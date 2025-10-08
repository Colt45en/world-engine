#pragma once

#include <functional>
#include <vector>
#include <memory>
#include <string>
#include <unordered_map>

namespace NEXUS {

// Enum for different visual/audio processing modes (inspired by Quantum Protocol)
enum class ProcessingMode : uint8_t {
    MIRROR = 0,    // Reflective/symmetric processing
    COSINE,        // Wave-based harmonic processing
    CHAOS,         // Random/chaotic variations
    ABSORB,        // Dampening/absorption effects
    AMPLIFY,       // Amplification/boost effects
    PULSE,         // Beat-synchronized pulsing
    FLOW,          // Smooth flowing transitions
    FRAGMENT       // Fragmented/glitch effects
};

// Convert enum to string for debugging/logging
inline const char* ProcessingModeToString(ProcessingMode mode) {
    switch (mode) {
        case ProcessingMode::MIRROR: return "Mirror";
        case ProcessingMode::COSINE: return "Cosine";
        case ProcessingMode::CHAOS: return "Chaos";
        case ProcessingMode::ABSORB: return "Absorb";
        case ProcessingMode::AMPLIFY: return "Amplify";
        case ProcessingMode::PULSE: return "Pulse";
        case ProcessingMode::FLOW: return "Flow";
        case ProcessingMode::FRAGMENT: return "Fragment";
        default: return "Unknown";
    }
}

// Audio data structure from NEXUS server
struct AudioData {
    float volume = 0.0f;
    float bass = 0.0f;
    float midrange = 0.0f;
    float treble = 0.0f;
    float bpm = 120.0f;
    bool beat_detected = false;
    float spectral_centroid = 0.0f;
    float spectral_rolloff = 0.0f;
    uint64_t timestamp_ms = 0;
};

// Art generation data from NEXUS server
struct ArtData {
    float complexity = 0.5f;
    float brightness = 0.5f;
    float contrast = 0.5f;
    float saturation = 0.5f;
    std::string style = "abstract";
    float dominant_color[3] = {0.5f, 0.5f, 0.5f}; // RGB
    float secondary_color[3] = {0.3f, 0.3f, 0.3f};
    float texture_intensity = 0.5f;
    uint64_t timestamp_ms = 0;
};

// Callback function types (C++ equivalent of Unreal delegates)
using ProcessingModeCallback = std::function<void(ProcessingMode)>;
using AudioCallback = std::function<void(const AudioData&)>;
using ArtCallback = std::function<void(const ArtData&)>;
using BeatCallback = std::function<void(float bpm, bool is_beat)>;

// NEXUS Protocol Manager (inspired by UQuantumProtocol)
class NexusProtocol {
public:
    static NexusProtocol& Instance() {
        static NexusProtocol instance;
        return instance;
    }

    // Processing mode management
    void SetProcessingMode(ProcessingMode mode) {
        if (current_mode_ != mode) {
            ProcessingMode old_mode = current_mode_;
            current_mode_ = mode;
            NotifyModeChange(old_mode, mode);
        }
    }

    ProcessingMode GetProcessingMode() const {
        return current_mode_;
    }

    // Callback registration (like Unreal's multicast delegates)
    void RegisterModeCallback(const ProcessingModeCallback& callback) {
        mode_callbacks_.push_back(callback);
    }

    void RegisterAudioCallback(const AudioCallback& callback) {
        audio_callbacks_.push_back(callback);
    }

    void RegisterArtCallback(const ArtCallback& callback) {
        art_callbacks_.push_back(callback);
    }

    void RegisterBeatCallback(const BeatCallback& callback) {
        beat_callbacks_.push_back(callback);
    }

    // Data broadcasting from NEXUS server updates
    void BroadcastAudioData(const AudioData& data) {
        latest_audio_ = data;
        for (const auto& callback : audio_callbacks_) {
            callback(data);
        }

        // Check for beat and notify beat callbacks
        if (data.beat_detected) {
            for (const auto& callback : beat_callbacks_) {
                callback(data.bpm, true);
            }
        }
    }

    void BroadcastArtData(const ArtData& data) {
        latest_art_ = data;
        for (const auto& callback : art_callbacks_) {
            callback(data);
        }
    }

    // Get latest cached data
    const AudioData& GetLatestAudio() const { return latest_audio_; }
    const ArtData& GetLatestArt() const { return latest_art_; }

    // Auto-mode selection based on audio characteristics
    void UpdateAutoMode(const AudioData& audio) {
        if (!auto_mode_enabled_) return;

        ProcessingMode suggested_mode = current_mode_;

        // Auto-select mode based on audio characteristics
        if (audio.bass > 0.8f && audio.beat_detected) {
            suggested_mode = ProcessingMode::PULSE;
        } else if (audio.treble > 0.7f) {
            suggested_mode = ProcessingMode::FRAGMENT;
        } else if (audio.volume > 0.6f && audio.midrange > 0.5f) {
            suggested_mode = ProcessingMode::AMPLIFY;
        } else if (audio.volume < 0.3f) {
            suggested_mode = ProcessingMode::ABSORB;
        } else if (audio.spectral_centroid > 0.6f) {
            suggested_mode = ProcessingMode::COSINE;
        } else {
            suggested_mode = ProcessingMode::FLOW;
        }

        SetProcessingMode(suggested_mode);
    }

    // Configuration
    void SetAutoModeEnabled(bool enabled) { auto_mode_enabled_ = enabled; }
    bool IsAutoModeEnabled() const { return auto_mode_enabled_; }

    void SetDebugMode(bool enabled) { debug_mode_ = enabled; }
    bool IsDebugMode() const { return debug_mode_; }

    // Statistics and monitoring
    struct Statistics {
        uint32_t mode_changes = 0;
        uint32_t audio_updates = 0;
        uint32_t art_updates = 0;
        uint32_t beats_detected = 0;
        uint64_t last_update_ms = 0;
    };

    const Statistics& GetStatistics() const { return stats_; }
    void ResetStatistics() { stats_ = Statistics{}; }

private:
    NexusProtocol() = default;
    ~NexusProtocol() = default;
    NexusProtocol(const NexusProtocol&) = delete;
    NexusProtocol& operator=(const NexusProtocol&) = delete;

    void NotifyModeChange(ProcessingMode old_mode, ProcessingMode new_mode) {
        stats_.mode_changes++;

        if (debug_mode_) {
            printf("[NEXUS] Processing mode changed: %s -> %s\n",
                   ProcessingModeToString(old_mode),
                   ProcessingModeToString(new_mode));
        }

        for (const auto& callback : mode_callbacks_) {
            callback(new_mode);
        }
    }

    // Current state
    ProcessingMode current_mode_ = ProcessingMode::FLOW;
    AudioData latest_audio_;
    ArtData latest_art_;

    // Configuration
    bool auto_mode_enabled_ = false;
    bool debug_mode_ = false;

    // Callbacks (like Unreal's multicast delegates)
    std::vector<ProcessingModeCallback> mode_callbacks_;
    std::vector<AudioCallback> audio_callbacks_;
    std::vector<ArtCallback> art_callbacks_;
    std::vector<BeatCallback> beat_callbacks_;

    // Statistics
    Statistics stats_;
};

// Convenience macros for common operations
#define NEXUS_PROTOCOL NEXUS::NexusProtocol::Instance()
#define NEXUS_SET_MODE(mode) NEXUS_PROTOCOL.SetProcessingMode(NEXUS::ProcessingMode::mode)
#define NEXUS_GET_MODE() NEXUS_PROTOCOL.GetProcessingMode()
#define NEXUS_REGISTER_MODE_CALLBACK(callback) NEXUS_PROTOCOL.RegisterModeCallback(callback)
#define NEXUS_REGISTER_AUDIO_CALLBACK(callback) NEXUS_PROTOCOL.RegisterAudioCallback(callback)
#define NEXUS_REGISTER_ART_CALLBACK(callback) NEXUS_PROTOCOL.RegisterArtCallback(callback)

} // namespace NEXUS
