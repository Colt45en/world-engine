#pragma once

#include "NexusProtocol.hpp"
#include "NexusVisuals.hpp"
#include <vector>
#include <memory>
#include <string>
#include <chrono>

namespace NEXUS {

// Vertex structure for trail rendering (simplified)
struct TrailVertex {
    float position[3] = {0.0f, 0.0f, 0.0f};    // xyz
    float uv[2] = {0.0f, 0.0f};                 // texture coordinates
    float color[4] = {1.0f, 1.0f, 1.0f, 1.0f}; // rgba
    float time = 0.0f;                          // time stamp for animation
};

// Trail point along the spline
struct TrailPoint {
    float position[3] = {0.0f, 0.0f, 0.0f};
    float width = 1.0f;
    float alpha = 1.0f;
    uint64_t timestamp = 0;

    TrailPoint() = default;
    TrailPoint(float x, float y, float z, float w = 1.0f, float a = 1.0f)
        : width(w), alpha(a) {
        position[0] = x; position[1] = y; position[2] = z;
        timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
    }
};

// Material parameters (like Unreal's dynamic material instance)
struct TrailMaterial {
    Color start_color = Color::White;
    Color end_color = Color::Gray;
    Color glow_color = Color::Cyan;
    float intensity = 1.0f;
    float glow_factor = 0.5f;
    float texture_scale = 1.0f;
    float animation_speed = 1.0f;
    bool use_world_position = false;

    // Animation state
    bool is_animating = false;
    float anim_time = 0.0f;
    float anim_duration = 0.0f;
    Color from_start_color, from_end_color;
    Color to_start_color, to_end_color;
};

// Trail Renderer class (inspired by AQuantumTrailRenderer)
class NexusTrailRenderer {
public:
    NexusTrailRenderer() {
        auto& protocol = NexusProtocol::Instance();

        // Register for NEXUS updates
        protocol.RegisterModeCallback([this](ProcessingMode mode) {
            OnProcessingModeChanged(mode);
        });

        protocol.RegisterAudioCallback([this](const AudioData& audio) {
            OnAudioUpdate(audio);
        });

        protocol.RegisterArtCallback([this](const ArtData& art) {
            OnArtUpdate(art);
        });

        protocol.RegisterBeatCallback([this](float bpm, bool is_beat) {
            OnBeatDetected(bpm, is_beat);
        });
    }

    virtual ~NexusTrailRenderer() = default;

    // Core trail management
    void AddPoint(float x, float y, float z, float width = 1.0f) {
        TrailPoint point(x, y, z, width);
        trail_points_.push_back(point);

        // Limit trail length
        if (trail_points_.size() > max_trail_points_) {
            trail_points_.erase(trail_points_.begin());
        }

        needs_update_ = true;
    }

    void AddPoint(const float pos[3], float width = 1.0f) {
        AddPoint(pos[0], pos[1], pos[2], width);
    }

    void ClearTrail() {
        trail_points_.clear();
        vertices_.clear();
        needs_update_ = true;
    }

    // Color management (like Unreal's SetTrailColorsImmediate)
    void SetTrailColorsImmediate(const Color& start, const Color& end) {
        material_.start_color = start;
        material_.end_color = end;
        material_.is_animating = false;
        material_.anim_time = 0.0f;
        needs_update_ = true;
    }

    // Animated color transitions (like Unreal's AnimateTrailColors)
    void AnimateTrailColors(const Color& start, const Color& end, float duration_seconds) {
        if (duration_seconds <= 0.0f) {
            SetTrailColorsImmediate(start, end);
            return;
        }

        // Store current colors as start of animation
        material_.from_start_color = material_.start_color;
        material_.from_end_color = material_.end_color;

        // Set target colors
        material_.to_start_color = start;
        material_.to_end_color = end;

        // Start animation
        material_.anim_time = 0.0f;
        material_.anim_duration = duration_seconds;
        material_.is_animating = true;
    }

    // Update loop (called each frame)
    void Update(float delta_seconds) {
        UpdateAnimation(delta_seconds);
        UpdateTrailFade(delta_seconds);

        if (needs_update_) {
            RegenerateVertices();
            needs_update_ = false;
        }
    }

    // Configuration
    void SetMaxTrailPoints(size_t max_points) {
        max_trail_points_ = max_points;
        if (trail_points_.size() > max_points) {
            trail_points_.erase(trail_points_.begin(),
                               trail_points_.begin() + (trail_points_.size() - max_points));
            needs_update_ = true;
        }
    }

    void SetTrailLifetime(float seconds) { trail_lifetime_ = seconds; }
    void SetTrailWidth(float width) { default_width_ = width; }
    void SetGlowIntensity(float intensity) { material_.glow_factor = intensity; }

    void SetProcessingModeSync(bool enabled) { sync_with_processing_mode_ = enabled; }
    void SetAudioReactive(bool enabled) { audio_reactive_ = enabled; }
    void SetArtReactive(bool enabled) { art_reactive_ = enabled; }

    // Get current state
    const std::vector<TrailPoint>& GetTrailPoints() const { return trail_points_; }
    const std::vector<TrailVertex>& GetVertices() const { return vertices_; }
    const TrailMaterial& GetMaterial() const { return material_; }

    bool IsAnimating() const { return material_.is_animating; }
    size_t GetPointCount() const { return trail_points_.size(); }
    float GetTrailLength() const { return CalculateTrailLength(); }

    // Rendering interface (to be implemented by specific renderer)
    virtual void Render() = 0;
    virtual void SetupShaderParameters() {}

protected:
    // NEXUS callbacks
    virtual void OnProcessingModeChanged(ProcessingMode mode) {
        if (!sync_with_processing_mode_) return;

        auto palette = NexusVisuals::GetCurrentPalette();
        AnimateTrailColors(palette.primary, palette.secondary, 0.5f);

        // Mode-specific effects
        switch (mode) {
            case ProcessingMode::PULSE:
                // Pulsing width
                pulse_effect_enabled_ = true;
                break;
            case ProcessingMode::CHAOS:
                // Random color shifts
                chaos_effect_enabled_ = true;
                break;
            case ProcessingMode::FLOW:
                // Smooth flowing animation
                material_.animation_speed = 0.8f;
                break;
            case ProcessingMode::FRAGMENT:
                // Fragmented/glitchy effects
                fragment_effect_enabled_ = true;
                break;
            default:
                ResetEffects();
                break;
        }
    }

    virtual void OnAudioUpdate(const AudioData& audio) {
        if (!audio_reactive_) return;

        // Width based on volume
        float volume_width = default_width_ * (0.5f + audio.volume * 1.5f);
        current_width_ = volume_width;

        // Glow based on treble
        material_.glow_factor = base_glow_factor_ * (0.3f + audio.treble * 1.4f);

        // Color shifts based on frequency content
        if (audio.bass > 0.7f) {
            // Boost red/orange on heavy bass
            Color bass_tint = Color::Red * 0.3f;
            material_.glow_color = material_.glow_color * 0.7f + bass_tint;
        }

        if (audio.treble > 0.6f) {
            // Boost brightness on high treble
            material_.intensity *= 1.2f;
        }

        needs_update_ = true;
    }

    virtual void OnArtUpdate(const ArtData& art) {
        if (!art_reactive_) return;

        // Apply art-based modifications
        material_.intensity *= (0.7f + art.brightness * 0.6f);

        // Complexity affects trail detail
        if (art.complexity > 0.7f) {
            SetMaxTrailPoints(static_cast<size_t>(max_trail_points_ * 1.3f));
        } else if (art.complexity < 0.3f) {
            SetMaxTrailPoints(static_cast<size_t>(max_trail_points_ * 0.7f));
        }

        // Style influences
        if (art.style == "geometric") {
            material_.texture_scale = 2.0f;
        } else if (art.style == "organic") {
            material_.animation_speed = 0.6f;
        }

        needs_update_ = true;
    }

    virtual void OnBeatDetected(float bpm, bool is_beat) {
        if (!audio_reactive_) return;

        if (is_beat && pulse_effect_enabled_) {
            // Beat pulse effect
            float pulse_intensity = 1.5f;
            material_.intensity *= pulse_intensity;
            material_.glow_factor *= pulse_intensity;

            // Schedule fade back to normal
            beat_pulse_timer_ = 0.1f; // 100ms pulse
        }

        // BPM affects animation speed
        material_.animation_speed = std::clamp(bpm / 120.0f, 0.5f, 2.0f);
    }

private:
    void UpdateAnimation(float delta_seconds) {
        if (!material_.is_animating || material_.anim_duration <= 0.0f) return;

        material_.anim_time = std::min(material_.anim_time + delta_seconds, material_.anim_duration);

        // Smooth interpolation (like Unreal's SmoothStep)
        float t = material_.anim_time / material_.anim_duration;
        t = t * t * (3.0f - 2.0f * t); // smoothstep

        // Interpolate colors
        material_.start_color = material_.from_start_color.Lerp(material_.to_start_color, t);
        material_.end_color = material_.from_end_color.Lerp(material_.to_end_color, t);

        if (material_.anim_time >= material_.anim_duration) {
            material_.is_animating = false;
        }

        needs_update_ = true;
    }

    void UpdateTrailFade(float delta_seconds) {
        if (trail_lifetime_ <= 0.0f) return;

        uint64_t current_time = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
        uint64_t max_age = static_cast<uint64_t>(trail_lifetime_ * 1000);

        bool removed_points = false;
        auto it = trail_points_.begin();
        while (it != trail_points_.end()) {
            if (current_time - it->timestamp > max_age) {
                it = trail_points_.erase(it);
                removed_points = true;
            } else {
                // Update alpha based on age
                float age_ratio = static_cast<float>(current_time - it->timestamp) / static_cast<float>(max_age);
                it->alpha = 1.0f - age_ratio;
                ++it;
            }
        }

        if (removed_points) {
            needs_update_ = true;
        }

        // Update beat pulse
        if (beat_pulse_timer_ > 0.0f) {
            beat_pulse_timer_ -= delta_seconds;
            if (beat_pulse_timer_ <= 0.0f) {
                // Fade back to normal
                material_.intensity = base_intensity_;
                material_.glow_factor = base_glow_factor_;
            }
        }
    }

    void RegenerateVertices() {
        vertices_.clear();

        if (trail_points_.size() < 2) return;

        // Generate vertex buffer from trail points
        for (size_t i = 0; i < trail_points_.size(); ++i) {
            const TrailPoint& point = trail_points_[i];

            // Calculate UV coordinate along trail (0 to 1)
            float u = static_cast<float>(i) / static_cast<float>(trail_points_.size() - 1);

            // Interpolate colors along trail
            Color point_color = material_.start_color.Lerp(material_.end_color, u);

            // Apply effects
            if (chaos_effect_enabled_ && i % 3 == 0) {
                point_color = Color::Random();
            }

            if (fragment_effect_enabled_ && i % 2 == 0) {
                point_color = point_color * 0.3f;
            }

            // Create vertices for this point (quad strip)
            float width = point.width * current_width_;

            // Left vertex
            TrailVertex left_vertex;
            left_vertex.position[0] = point.position[0] - width * 0.5f;
            left_vertex.position[1] = point.position[1];
            left_vertex.position[2] = point.position[2];
            left_vertex.uv[0] = u;
            left_vertex.uv[1] = 0.0f;
            left_vertex.color[0] = point_color.r;
            left_vertex.color[1] = point_color.g;
            left_vertex.color[2] = point_color.b;
            left_vertex.color[3] = point_color.a * point.alpha;

            // Right vertex
            TrailVertex right_vertex;
            right_vertex.position[0] = point.position[0] + width * 0.5f;
            right_vertex.position[1] = point.position[1];
            right_vertex.position[2] = point.position[2];
            right_vertex.uv[0] = u;
            right_vertex.uv[1] = 1.0f;
            right_vertex.color[0] = point_color.r;
            right_vertex.color[1] = point_color.g;
            right_vertex.color[2] = point_color.b;
            right_vertex.color[3] = point_color.a * point.alpha;

            vertices_.push_back(left_vertex);
            vertices_.push_back(right_vertex);
        }
    }

    float CalculateTrailLength() const {
        if (trail_points_.size() < 2) return 0.0f;

        float total_length = 0.0f;
        for (size_t i = 1; i < trail_points_.size(); ++i) {
            const float* p1 = trail_points_[i-1].position;
            const float* p2 = trail_points_[i].position;

            float dx = p2[0] - p1[0];
            float dy = p2[1] - p1[1];
            float dz = p2[2] - p1[2];

            total_length += std::sqrt(dx*dx + dy*dy + dz*dz);
        }

        return total_length;
    }

    void ResetEffects() {
        pulse_effect_enabled_ = false;
        chaos_effect_enabled_ = false;
        fragment_effect_enabled_ = false;
        material_.animation_speed = 1.0f;
    }

protected:
    // Trail data
    std::vector<TrailPoint> trail_points_;
    std::vector<TrailVertex> vertices_;
    TrailMaterial material_;

    // Configuration
    size_t max_trail_points_ = 100;
    float trail_lifetime_ = 5.0f;
    float default_width_ = 1.0f;
    float current_width_ = 1.0f;

    // NEXUS integration flags
    bool sync_with_processing_mode_ = true;
    bool audio_reactive_ = true;
    bool art_reactive_ = true;

    // Effect flags
    bool pulse_effect_enabled_ = false;
    bool chaos_effect_enabled_ = false;
    bool fragment_effect_enabled_ = false;

    // Animation state
    bool needs_update_ = false;
    float beat_pulse_timer_ = 0.0f;
    float base_intensity_ = 1.0f;
    float base_glow_factor_ = 0.5f;
};

// Concrete implementation for console/debug rendering
class ConsoleTrailRenderer : public NexusTrailRenderer {
public:
    void Render() override {
        if (trail_points_.empty()) return;

        // Simple ASCII visualization
        printf("Trail [%zu points]: ", trail_points_.size());

        for (size_t i = 0; i < std::min(trail_points_.size(), size_t(20)); ++i) {
            const auto& point = trail_points_[i];
            char symbol = '*';

            if (point.alpha < 0.3f) symbol = '.';
            else if (point.alpha < 0.6f) symbol = 'o';
            else if (point.alpha < 0.9f) symbol = 'O';

            printf("%c", symbol);
        }

        printf(" [Mode: %s, Intensity: %.1f]\n",
               ProcessingModeToString(NEXUS_GET_MODE()),
               material_.intensity);
    }
};

// Trail manager for handling multiple trails
class NexusTrailManager {
public:
    static NexusTrailManager& Instance() {
        static NexusTrailManager instance;
        return instance;
    }

    std::shared_ptr<NexusTrailRenderer> CreateTrail(const std::string& name = "") {
        auto trail = std::make_shared<ConsoleTrailRenderer>();
        trails_.push_back({name, trail});
        return trail;
    }

    void UpdateAll(float delta_seconds) {
        for (auto& entry : trails_) {
            if (entry.trail) {
                entry.trail->Update(delta_seconds);
            }
        }
    }

    void RenderAll() {
        for (const auto& entry : trails_) {
            if (entry.trail) {
                if (!entry.name.empty()) {
                    printf("[%s] ", entry.name.c_str());
                }
                entry.trail->Render();
            }
        }
    }

    void SyncAllToCurrentPalette(float fade_seconds = 0.5f) {
        auto palette = NexusVisuals::GetCurrentPalette();

        for (auto& entry : trails_) {
            if (entry.trail) {
                if (fade_seconds > 0.0f) {
                    entry.trail->AnimateTrailColors(palette.primary, palette.secondary, fade_seconds);
                } else {
                    entry.trail->SetTrailColorsImmediate(palette.primary, palette.secondary);
                }
            }
        }
    }

    size_t GetTrailCount() const { return trails_.size(); }

    void ClearAll() {
        trails_.clear();
    }

private:
    struct TrailEntry {
        std::string name;
        std::shared_ptr<NexusTrailRenderer> trail;
    };

    std::vector<TrailEntry> trails_;
};

} // namespace NEXUS
