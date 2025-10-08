// NEXUS Recursive Keeper Engine Integration
// Combining cognitive processing with quantum protocol and audio-reactive systems
// Based on the Recursive Keeper Engine with NEXUS enhancements

#pragma once

#include "NexusProtocol.hpp"
#include "NexusVisuals.hpp"
#include <string>
#include <vector>
#include <unordered_map>
#include <chrono>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <iostream>
#include <functional>
#include <memory>

namespace NEXUS {

// Enhanced Node structure with NEXUS integration
struct CognitiveNode {
    std::string topic;
    std::string visible_infrastructure;
    std::string unseen_infrastructure;
    std::string solid_state;
    std::string liquid_state;
    std::string gas_state;
    std::string derived_topic;
    std::string timestamp; // ISO-8601

    // NEXUS enhancements
    ProcessingMode suggested_mode = ProcessingMode::FLOW;
    VisualPalette associated_palette;
    float cognitive_intensity = 1.0f;
    float conceptual_complexity = 0.5f;
    bool triggered_by_audio = false;
    bool triggered_by_art = false;
    uint64_t creation_timestamp_ms = 0;
};

inline std::string NowIso8601() {
    using namespace std::chrono;
    const auto now = system_clock::now();
    const auto t   = system_clock::to_time_t(now);
    const auto ms  = duration_cast<milliseconds>(now.time_since_epoch()) % 1000;

    std::tm tm{};
#if defined(_WIN32)
    gmtime_s(&tm, &t);
#else
    gmtime_r(&t, &tm);
#endif
    std::ostringstream oss;
    oss << std::put_time(&tm, "%Y-%m-%dT%H:%M:%S") << '.' << std::setw(3)
        << std::setfill('0') << ms.count() << "Z";
    return oss.str();
}

// NEXUS-Enhanced Recursive Infrastructure Flow Processor
class NexusRecursiveFlow {
public:
    const std::vector<CognitiveNode>& History() const { return history_; }

    CognitiveNode AnalyzeTopic(const std::string& topic, const AudioData* audio = nullptr, const ArtData* art = nullptr) const {
        CognitiveNode n;
        n.topic                   = topic;
        n.visible_infrastructure = GenerateVisibleInfrastructure(topic, audio, art);
        n.unseen_infrastructure  = GenerateUnseenInfrastructure(topic, audio, art);
        n.solid_state            = GenerateSolidState(topic, audio, art);
        n.liquid_state           = GenerateLiquidState(topic, audio, art);
        n.gas_state              = GenerateGasState(topic, audio, art);
        n.derived_topic          = GenerateDerivedTopic(topic, audio, art);
        n.timestamp              = NowIso8601();

        // NEXUS enhancements
        n.suggested_mode = DetermineSuggestedMode(topic, audio, art);
        n.associated_palette = NexusVisuals::GetPalette(n.suggested_mode, audio, art);
        n.cognitive_intensity = CalculateCognitiveIntensity(topic, audio, art);
        n.conceptual_complexity = CalculateConceptualComplexity(topic, audio, art);
        n.triggered_by_audio = (audio != nullptr);
        n.triggered_by_art = (art != nullptr);
        n.creation_timestamp_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();

        return n;
    }

    std::vector<CognitiveNode> RunFlow(const std::string& starting_topic, int iterations = 5) {
        history_.clear();
        std::string t = starting_topic;

        auto& protocol = NexusProtocol::Instance();

        for (int i = 0; i < iterations; ++i) {
            const AudioData* audio = &protocol.GetLatestAudio();
            const ArtData* art = &protocol.GetLatestArt();

            CognitiveNode n = AnalyzeTopic(t, audio, art);
            history_.push_back(n);

            // Update NEXUS protocol based on cognitive analysis
            if (n.suggested_mode != protocol.GetProcessingMode()) {
                protocol.SetProcessingMode(n.suggested_mode);
            }

            t = n.derived_topic;
        }
        return history_;
    }

private:
    std::vector<CognitiveNode> history_;

    std::string GenerateVisibleInfrastructure(const std::string& topic, const AudioData* audio, const ArtData* art) const {
        std::string base = "Tangible structures supporting " + topic;

        if (audio && audio->volume > 0.7f) {
            base += " with resonant amplification";
        }
        if (art && art->complexity > 0.6f) {
            base += " manifesting as complex geometric patterns";
        }

        return base + ".";
    }

    std::string GenerateUnseenInfrastructure(const std::string& topic, const AudioData* audio, const ArtData* art) const {
        std::string base = "Intangible frameworks supporting " + topic;

        if (audio && audio->spectral_centroid > 0.5f) {
            base += " through harmonic field resonance";
        }
        if (art && art->style == "abstract") {
            base += " via abstract conceptual networks";
        }

        return base + ".";
    }

    std::string GenerateSolidState(const std::string& topic, const AudioData* audio, const ArtData* art) const {
        std::string base = "Fixed and rigid forms of " + topic;

        if (audio && audio->bass > 0.8f) {
            base += " anchored by deep foundational frequencies";
        }
        if (art && art->contrast > 0.7f) {
            base += " with stark definitional boundaries";
        }

        return base + ".";
    }

    std::string GenerateLiquidState(const std::string& topic, const AudioData* audio, const ArtData* art) const {
        std::string base = "Adaptive and evolving forms of " + topic;

        if (audio && audio->midrange > 0.6f) {
            base += " flowing through harmonic transitions";
        }
        if (art && art->style == "organic") {
            base += " with organic fluid transformations";
        }

        return base + ".";
    }

    std::string GenerateGasState(const std::string& topic, const AudioData* audio, const ArtData* art) const {
        std::string base = "Dispersed and pervasive forms of " + topic;

        if (audio && audio->treble > 0.7f) {
            base += " dispersed through high-frequency propagation";
        }
        if (art && art->brightness > 0.8f) {
            base += " luminously distributed across conceptual space";
        }

        return base + ".";
    }

    std::string GenerateDerivedTopic(const std::string& topic, const AudioData* audio, const ArtData* art) const {
        std::string base = "The Evolution of " + topic + " in the Next Cognitive Cycle";

        if (audio && audio->beat_detected) {
            base += " Synchronized to Rhythmic Emergence";
        }
        if (art && art->texture_intensity > 0.6f) {
            base += " Through Textural Manifestation";
        }

        return base;
    }

    ProcessingMode DetermineSuggestedMode(const std::string& topic, const AudioData* audio, const ArtData* art) const {
        // Topic-based mode selection
        std::string lower_topic = topic;
        std::transform(lower_topic.begin(), lower_topic.end(), lower_topic.begin(), ::tolower);

        if (lower_topic.find("mirror") != std::string::npos || lower_topic.find("reflection") != std::string::npos) {
            return ProcessingMode::MIRROR;
        }
        if (lower_topic.find("wave") != std::string::npos || lower_topic.find("harmonic") != std::string::npos) {
            return ProcessingMode::COSINE;
        }
        if (lower_topic.find("chaos") != std::string::npos || lower_topic.find("random") != std::string::npos) {
            return ProcessingMode::CHAOS;
        }
        if (lower_topic.find("absorb") != std::string::npos || lower_topic.find("quiet") != std::string::npos) {
            return ProcessingMode::ABSORB;
        }
        if (lower_topic.find("amplify") != std::string::npos || lower_topic.find("intense") != std::string::npos) {
            return ProcessingMode::AMPLIFY;
        }
        if (lower_topic.find("pulse") != std::string::npos || lower_topic.find("beat") != std::string::npos) {
            return ProcessingMode::PULSE;
        }
        if (lower_topic.find("fragment") != std::string::npos || lower_topic.find("break") != std::string::npos) {
            return ProcessingMode::FRAGMENT;
        }

        // Audio-influenced mode selection
        if (audio) {
            if (audio->bass > 0.8f) return ProcessingMode::PULSE;
            if (audio->treble > 0.7f) return ProcessingMode::FRAGMENT;
            if (audio->volume > 0.6f) return ProcessingMode::AMPLIFY;
            if (audio->volume < 0.3f) return ProcessingMode::ABSORB;
        }

        // Art-influenced mode selection
        if (art) {
            if (art->complexity > 0.8f) return ProcessingMode::CHAOS;
            if (art->style == "geometric") return ProcessingMode::COSINE;
            if (art->brightness > 0.7f) return ProcessingMode::AMPLIFY;
        }

        return ProcessingMode::FLOW; // Default
    }

    float CalculateCognitiveIntensity(const std::string& topic, const AudioData* audio, const ArtData* art) const {
        float intensity = 1.0f;

        // Topic complexity
        intensity += static_cast<float>(topic.length()) / 100.0f;

        // Audio influence
        if (audio) {
            intensity *= (0.5f + audio->volume * 1.5f);
            if (audio->beat_detected) intensity *= 1.3f;
        }

        // Art influence
        if (art) {
            intensity *= (0.7f + art->complexity * 0.6f);
            intensity *= (0.8f + art->brightness * 0.4f);
        }

        return std::clamp(intensity, 0.1f, 3.0f);
    }

    float CalculateConceptualComplexity(const std::string& topic, const AudioData* audio, const ArtData* art) const {
        float complexity = 0.5f;

        // Topic analysis
        size_t word_count = std::count(topic.begin(), topic.end(), ' ') + 1;
        complexity += static_cast<float>(word_count) / 20.0f;

        // Audio complexity
        if (audio) {
            complexity += (audio->spectral_centroid + audio->spectral_rolloff) * 0.25f;
        }

        // Art complexity
        if (art) {
            complexity += art->complexity * 0.4f;
            complexity += art->texture_intensity * 0.2f;
        }

        return std::clamp(complexity, 0.0f, 1.0f);
    }
};

// Enhanced Dual-State Memory with NEXUS awareness
class NexusDualStateMemory {
public:
    void RecordNode(const CognitiveNode& node) {
        forward_memory_.push_back(node);
        reverse_memory_.insert(reverse_memory_.begin(), node);

        // Trigger NEXUS protocol updates based on memory patterns
        if (forward_memory_.size() >= 3) {
            AnalyzeMemoryPatterns();
        }
    }

    const std::vector<CognitiveNode>& Forward() const { return forward_memory_; }
    const std::vector<CognitiveNode>& Reverse() const { return reverse_memory_; }

    void Clear() {
        forward_memory_.clear();
        reverse_memory_.clear();
    }

    // NEXUS-specific memory analysis
    ProcessingMode GetDominantMode() const {
        if (forward_memory_.empty()) return ProcessingMode::FLOW;

        std::unordered_map<ProcessingMode, int> mode_counts;
        for (const auto& node : forward_memory_) {
            mode_counts[node.suggested_mode]++;
        }

        auto max_it = std::max_element(mode_counts.begin(), mode_counts.end(),
            [](const auto& a, const auto& b) { return a.second < b.second; });

        return max_it != mode_counts.end() ? max_it->first : ProcessingMode::FLOW;
    }

    float GetAverageIntensity() const {
        if (forward_memory_.empty()) return 1.0f;

        float sum = 0.0f;
        for (const auto& node : forward_memory_) {
            sum += node.cognitive_intensity;
        }
        return sum / static_cast<float>(forward_memory_.size());
    }

    float GetAverageComplexity() const {
        if (forward_memory_.empty()) return 0.5f;

        float sum = 0.0f;
        for (const auto& node : forward_memory_) {
            sum += node.conceptual_complexity;
        }
        return sum / static_cast<float>(forward_memory_.size());
    }

private:
    std::vector<CognitiveNode> forward_memory_;
    std::vector<CognitiveNode> reverse_memory_;

    void AnalyzeMemoryPatterns() {
        // Detect cognitive patterns and influence NEXUS protocol
        auto& protocol = NexusProtocol::Instance();

        ProcessingMode dominant = GetDominantMode();
        if (dominant != protocol.GetProcessingMode()) {
            // Cognitive consensus suggests mode change
            protocol.SetProcessingMode(dominant);
        }

        // Adjust system sensitivity based on memory patterns
        float avg_complexity = GetAverageComplexity();
        if (avg_complexity > 0.8f) {
            // High cognitive complexity detected
            // Could trigger special processing modes
        }
    }
};

// Enhanced Eternal Imprint Registry with NEXUS features
class NexusEternalImprints {
public:
    void RecordImprint(const CognitiveNode& node) {
        imprints_[node.topic] = node;

        // Track imprint statistics for NEXUS
        imprint_creation_times_[node.topic] = node.creation_timestamp_ms;
    }

    const CognitiveNode* GetImprint(const std::string& topic) const {
        auto it = imprints_.find(topic);
        return (it == imprints_.end()) ? nullptr : &it->second;
    }

    const std::unordered_map<std::string, CognitiveNode>& All() const { return imprints_; }

    // NEXUS-specific imprint analysis
    std::vector<std::string> GetRecentImprints(uint64_t since_ms) const {
        std::vector<std::string> recent;
        uint64_t current_time = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();

        for (const auto& entry : imprint_creation_times_) {
            if (current_time - entry.second <= since_ms) {
                recent.push_back(entry.first);
            }
        }
        return recent;
    }

    ProcessingMode GetMostInfluentialMode() const {
        std::unordered_map<ProcessingMode, float> mode_influence;

        for (const auto& entry : imprints_) {
            ProcessingMode mode = entry.second.suggested_mode;
            mode_influence[mode] += entry.second.cognitive_intensity;
        }

        auto max_it = std::max_element(mode_influence.begin(), mode_influence.end(),
            [](const auto& a, const auto& b) { return a.second < b.second; });

        return max_it != mode_influence.end() ? max_it->first : ProcessingMode::FLOW;
    }

private:
    std::unordered_map<std::string, CognitiveNode> imprints_;
    std::unordered_map<std::string, uint64_t> imprint_creation_times_;
};

// Enhanced Symbolic Mapping with NEXUS integration
class NexusSymbolicMapping {
public:
    void AssignSymbol(const std::string& topic, const std::string& symbol) {
        symbol_map_[topic] = symbol;

        // Associate processing mode with symbol
        auto* imprint = eternal_imprints_->GetImprint(topic);
        if (imprint) {
            symbol_modes_[symbol] = imprint->suggested_mode;
        }
    }

    void SetImprintRegistry(const NexusEternalImprints* imprints) {
        eternal_imprints_ = imprints;
    }

    const std::string* GetSymbol(const std::string& topic) const {
        auto it = symbol_map_.find(topic);
        return (it == symbol_map_.end()) ? nullptr : &it->second;
    }

    ProcessingMode GetSymbolMode(const std::string& symbol) const {
        auto it = symbol_modes_.find(symbol);
        return (it == symbol_modes_.end()) ? ProcessingMode::FLOW : it->second;
    }

    const std::unordered_map<std::string, std::string>& All() const { return symbol_map_; }

private:
    std::unordered_map<std::string, std::string> symbol_map_;
    std::unordered_map<std::string, ProcessingMode> symbol_modes_;
    const NexusEternalImprints* eternal_imprints_ = nullptr;
};

// JSON utilities for NEXUS integration
inline std::string Escape(const std::string& s) {
    std::ostringstream o;
    for (char c : s) {
        switch (c) {
            case '\"': o << "\\\""; break;
            case '\\': o << "\\\\"; break;
            case '\n': o << "\\n"; break;
            case '\r': o << "\\r"; break;
            case '\t': o << "\\t"; break;
            default:   o << c; break;
        }
    }
    return o.str();
}

inline std::string ToJson(const CognitiveNode& n) {
    std::ostringstream oss;
    oss << "{"
        << "\"topic\":\"" << Escape(n.topic) << "\","
        << "\"visible_infrastructure\":\"" << Escape(n.visible_infrastructure) << "\","
        << "\"unseen_infrastructure\":\""  << Escape(n.unseen_infrastructure)  << "\","
        << "\"solid_state\":\""            << Escape(n.solid_state)            << "\","
        << "\"liquid_state\":\""           << Escape(n.liquid_state)           << "\","
        << "\"gas_state\":\""              << Escape(n.gas_state)              << "\","
        << "\"derived_topic\":\""          << Escape(n.derived_topic)          << "\","
        << "\"timestamp\":\""              << Escape(n.timestamp)              << "\","
        << "\"suggested_mode\":\"" << ProcessingModeToString(n.suggested_mode) << "\","
        << "\"cognitive_intensity\":" << n.cognitive_intensity << ","
        << "\"conceptual_complexity\":" << n.conceptual_complexity << ","
        << "\"triggered_by_audio\":" << (n.triggered_by_audio ? "true" : "false") << ","
        << "\"triggered_by_art\":" << (n.triggered_by_art ? "true" : "false")
        << "}";
    return oss.str();
}

// Main NEXUS Recursive Keeper Engine
class NexusRecursiveKeeperEngine {
public:
    NexusRecursiveKeeperEngine() {
        symbols_.SetImprintRegistry(&imprints_);

        // Register with NEXUS protocol for real-time updates
        auto& protocol = NexusProtocol::Instance();
        protocol.RegisterAudioCallback([this](const AudioData& audio) {
            OnAudioUpdate(audio);
        });
        protocol.RegisterArtCallback([this](const ArtData& art) {
            OnArtUpdate(art);
        });
    }

    // Process a topic through N iterations with NEXUS integration
    void ProcessTopic(const std::string& starting_topic, int iterations = 5) {
        std::string t = starting_topic;
        for (int i = 0; i < iterations; ++i) {
            CognitiveNode n = flow_.AnalyzeTopic(t);
            n.timestamp = NowIso8601();

            memory_.RecordNode(n);
            imprints_.RecordImprint(n);

            t = n.derived_topic;

            // Allow NEXUS protocol updates between iterations
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }

    void AssignSymbol(const std::string& topic, const std::string& symbol) {
        symbols_.AssignSymbol(topic, symbol);
    }

    // NEXUS integration methods
    void TriggerCognitiveAnalysis(const std::string& topic) {
        auto& protocol = NexusProtocol::Instance();
        const AudioData& audio = protocol.GetLatestAudio();
        const ArtData& art = protocol.GetLatestArt();

        CognitiveNode n = flow_.AnalyzeTopic(topic, &audio, &art);
        n.timestamp = NowIso8601();

        memory_.RecordNode(n);
        imprints_.RecordImprint(n);

        // Update processing mode based on cognitive analysis
        protocol.SetProcessingMode(n.suggested_mode);
    }

    ProcessingMode GetCognitiveConsensusMode() const {
        return memory_.GetDominantMode();
    }

    float GetSystemComplexity() const {
        return memory_.GetAverageComplexity();
    }

    float GetSystemIntensity() const {
        return memory_.GetAverageIntensity();
    }

    // State inspection
    std::string GetFullStateJson() const {
        std::ostringstream oss;
        oss << "{"
            << "\"memory\":{"
                << "\"forward_memory\":[";

        const auto& forward = memory_.Forward();
        for (size_t i = 0; i < forward.size(); ++i) {
            if (i > 0) oss << ",";
            oss << ToJson(forward[i]);
        }

        oss << "],\"reverse_memory\":[";

        const auto& reverse = memory_.Reverse();
        for (size_t i = 0; i < reverse.size(); ++i) {
            if (i > 0) oss << ",";
            oss << ToJson(reverse[i]);
        }

        oss << "]},"
            << "\"cognitive_stats\":{"
                << "\"dominant_mode\":\"" << ProcessingModeToString(memory_.GetDominantMode()) << "\","
                << "\"average_intensity\":" << memory_.GetAverageIntensity() << ","
                << "\"average_complexity\":" << memory_.GetAverageComplexity()
            << "},"
            << "\"imprint_count\":" << imprints_.All().size() << ","
            << "\"symbol_count\":" << symbols_.All().size()
            << "}";
        return oss.str();
    }

    // Accessors
    const NexusRecursiveFlow&      Flow()     const { return flow_; }
    const NexusDualStateMemory&    Memory()   const { return memory_; }
    const NexusEternalImprints&    Imprints() const { return imprints_; }
    const NexusSymbolicMapping&    Symbols()  const { return symbols_; }

    void ClearMemory() { memory_.Clear(); }

private:
    NexusRecursiveFlow     flow_;
    NexusDualStateMemory   memory_;
    NexusEternalImprints   imprints_;
    NexusSymbolicMapping   symbols_;

    void OnAudioUpdate(const AudioData& audio) {
        // Audio-triggered cognitive analysis
        if (audio.beat_detected && audio.volume > 0.6f) {
            TriggerCognitiveAnalysis("Audio-Triggered Cognitive Emergence");
        }
    }

    void OnArtUpdate(const ArtData& art) {
        // Art-triggered cognitive analysis
        if (art.complexity > 0.7f) {
            TriggerCognitiveAnalysis("Art-Triggered Conceptual Evolution");
        }
    }
};

} // namespace NEXUS
