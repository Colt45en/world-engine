#pragma once

#include "NexusProtocol.hpp"
#include <array>
#include <cmath>
#include <random>

namespace NEXUS {

// Color structure (similar to Unreal's FLinearColor)
struct Color {
    float r, g, b, a;

    Color(float red = 1.0f, float green = 1.0f, float blue = 1.0f, float alpha = 1.0f)
        : r(red), g(green), b(blue), a(alpha) {}

    // Predefined colors (like Unreal's static colors)
    static const Color White;
    static const Color Black;
    static const Color Red;
    static const Color Green;
    static const Color Blue;
    static const Color Cyan;
    static const Color Magenta;
    static const Color Yellow;
    static const Color Gray;
    static const Color Orange;
    static const Color Purple;

    // Utility functions
    Color operator+(const Color& other) const {
        return Color(r + other.r, g + other.g, b + other.b, a + other.a);
    }

    Color operator*(float scalar) const {
        return Color(r * scalar, g * scalar, b * scalar, a * scalar);
    }

    Color Lerp(const Color& target, float t) const {
        t = std::clamp(t, 0.0f, 1.0f);
        return Color(
            r + (target.r - r) * t,
            g + (target.g - g) * t,
            b + (target.b - b) * t,
            a + (target.a - a) * t
        );
    }

    static Color Random() {
        static std::random_device rd;
        static std::mt19937 gen(rd());
        static std::uniform_real_distribution<float> dis(0.0f, 1.0f);
        return Color(dis(gen), dis(gen), dis(gen), 1.0f);
    }

    static Color RandomHSV(float hue_min = 0.0f, float hue_max = 1.0f,
                          float sat_min = 0.5f, float sat_max = 1.0f,
                          float val_min = 0.5f, float val_max = 1.0f) {
        static std::random_device rd;
        static std::mt19937 gen(rd());
        static std::uniform_real_distribution<float> hue_dis(hue_min, hue_max);
        static std::uniform_real_distribution<float> sat_dis(sat_min, sat_max);
        static std::uniform_real_distribution<float> val_dis(val_min, val_max);

        float h = hue_dis(gen);
        float s = sat_dis(gen);
        float v = val_dis(gen);

        return HSVToRGB(h, s, v);
    }

    static Color HSVToRGB(float h, float s, float v) {
        float c = v * s;
        float x = c * (1.0f - std::abs(std::fmod(h * 6.0f, 2.0f) - 1.0f));
        float m = v - c;

        float r, g, b;
        if (h < 1.0f/6.0f) { r = c; g = x; b = 0; }
        else if (h < 2.0f/6.0f) { r = x; g = c; b = 0; }
        else if (h < 3.0f/6.0f) { r = 0; g = c; b = x; }
        else if (h < 4.0f/6.0f) { r = 0; g = x; b = c; }
        else if (h < 5.0f/6.0f) { r = x; g = 0; b = c; }
        else { r = c; g = 0; b = x; }

        return Color(r + m, g + m, b + m, 1.0f);
    }
};

// Static color definitions
const Color Color::White(1.0f, 1.0f, 1.0f, 1.0f);
const Color Color::Black(0.0f, 0.0f, 0.0f, 1.0f);
const Color Color::Red(1.0f, 0.0f, 0.0f, 1.0f);
const Color Color::Green(0.0f, 1.0f, 0.0f, 1.0f);
const Color Color::Blue(0.0f, 0.0f, 1.0f, 1.0f);
const Color Color::Cyan(0.0f, 1.0f, 1.0f, 1.0f);
const Color Color::Magenta(1.0f, 0.0f, 1.0f, 1.0f);
const Color Color::Yellow(1.0f, 1.0f, 0.0f, 1.0f);
const Color Color::Gray(0.5f, 0.5f, 0.5f, 1.0f);
const Color Color::Orange(1.0f, 0.65f, 0.0f, 1.0f);
const Color Color::Purple(0.5f, 0.0f, 0.5f, 1.0f);

// Palette structure (like Unreal's FTrailPalette)
struct VisualPalette {
    Color primary = Color::White;
    Color secondary = Color::Gray;
    Color accent = Color::Cyan;
    Color background = Color::Black;
    float intensity = 1.0f;
    float glow_factor = 0.5f;

    VisualPalette() = default;
    VisualPalette(const Color& p, const Color& s, const Color& a = Color::Cyan, const Color& bg = Color::Black)
        : primary(p), secondary(s), accent(a), background(bg) {}
};

// NEXUS Visual System (inspired by UQuantumVisuals)
class NexusVisuals {
public:
    // Get palette for processing mode (equivalent to GetPalette in Unreal)
    static VisualPalette GetPalette(ProcessingMode mode, const AudioData* audio = nullptr, const ArtData* art = nullptr) {
        VisualPalette palette;

        switch (mode) {
            case ProcessingMode::MIRROR:
                palette.primary = Color::Cyan;
                palette.secondary = Color::White;
                palette.accent = Color::Blue;
                palette.intensity = 0.8f;
                palette.glow_factor = 0.3f;
                break;

            case ProcessingMode::COSINE:
                palette.primary = Color::Magenta;
                palette.secondary = Color::Yellow;
                palette.accent = Color::Orange;
                palette.intensity = 0.9f;
                palette.glow_factor = 0.6f;
                break;

            case ProcessingMode::CHAOS:
                palette.primary = Color::Random();
                palette.secondary = Color::Random();
                palette.accent = Color::RandomHSV(0.0f, 1.0f, 0.7f, 1.0f, 0.8f, 1.0f);
                palette.intensity = 1.2f;
                palette.glow_factor = 0.8f;
                break;

            case ProcessingMode::ABSORB:
                palette.primary = Color::Black;
                palette.secondary = Color(0.2f, 0.2f, 0.2f);
                palette.accent = Color(0.1f, 0.1f, 0.3f);
                palette.intensity = 0.3f;
                palette.glow_factor = 0.1f;
                break;

            case ProcessingMode::AMPLIFY:
                palette.primary = Color::Red;
                palette.secondary = Color::Orange;
                palette.accent = Color::Yellow;
                palette.intensity = 1.5f;
                palette.glow_factor = 1.0f;
                break;

            case ProcessingMode::PULSE:
                palette.primary = Color::White;
                palette.secondary = Color::Red;
                palette.accent = Color::Purple;
                palette.intensity = 1.3f;
                palette.glow_factor = 0.9f;
                break;

            case ProcessingMode::FLOW:
                palette.primary = Color::Blue;
                palette.secondary = Color::Cyan;
                palette.accent = Color::Green;
                palette.intensity = 0.7f;
                palette.glow_factor = 0.4f;
                break;

            case ProcessingMode::FRAGMENT:
                palette.primary = Color::Yellow;
                palette.secondary = Color::White;
                palette.accent = Color::Magenta;
                palette.intensity = 1.1f;
                palette.glow_factor = 0.7f;
                break;

            default:
                palette.primary = Color::White;
                palette.secondary = Color::Gray;
                palette.accent = Color::Cyan;
                break;
        }

        // Apply audio-reactive modifications
        if (audio) {
            ApplyAudioModifications(palette, *audio);
        }

        // Apply art-reactive modifications
        if (art) {
            ApplyArtModifications(palette, *art);
        }

        return palette;
    }

    // Audio-reactive palette modifications
    static void ApplyAudioModifications(VisualPalette& palette, const AudioData& audio) {
        // Intensity based on volume
        palette.intensity *= (0.5f + audio.volume * 1.5f);

        // Glow based on treble content
        palette.glow_factor *= (0.3f + audio.treble * 1.4f);

        // Color shifts based on frequency content
        if (audio.bass > 0.7f) {
            // Heavy bass - shift toward red/orange
            palette.primary = palette.primary * 0.7f + Color::Red * 0.3f;
            palette.accent = palette.accent * 0.8f + Color::Orange * 0.2f;
        }

        if (audio.treble > 0.7f) {
            // High treble - shift toward white/yellow
            palette.secondary = palette.secondary * 0.6f + Color::Yellow * 0.4f;
            palette.glow_factor *= 1.3f;
        }

        if (audio.midrange > 0.6f) {
            // Strong midrange - boost saturation
            palette.intensity *= 1.2f;
        }

        // Beat detection effects
        if (audio.beat_detected) {
            palette.intensity *= 1.4f;
            palette.glow_factor *= 1.2f;
        }

        // BPM-based effects
        float bpm_factor = std::clamp(audio.bpm / 140.0f, 0.5f, 2.0f);
        palette.intensity *= (0.8f + bpm_factor * 0.4f);
    }

    // Art-reactive palette modifications
    static void ApplyArtModifications(VisualPalette& palette, const ArtData& art) {
        // Complexity affects intensity and detail
        palette.intensity *= (0.6f + art.complexity * 0.8f);

        // Brightness directly affects glow
        palette.glow_factor *= art.brightness;

        // Contrast affects color separation
        float contrast_factor = art.contrast;
        palette.primary = palette.primary * (0.5f + contrast_factor * 0.5f);
        palette.secondary = palette.secondary * (0.5f + contrast_factor * 0.5f);

        // Saturation affects color vibrancy
        float sat_factor = art.saturation;
        Color desaturated_primary = Color::Gray.Lerp(palette.primary, sat_factor);
        Color desaturated_secondary = Color::Gray.Lerp(palette.secondary, sat_factor);
        palette.primary = desaturated_primary;
        palette.secondary = desaturated_secondary;

        // Dominant color influence
        Color art_dominant(art.dominant_color[0], art.dominant_color[1], art.dominant_color[2]);
        palette.accent = palette.accent.Lerp(art_dominant, 0.3f);

        // Style-based modifications
        if (art.style == "geometric") {
            palette.intensity *= 1.1f;
            palette.glow_factor *= 0.8f; // Sharper, less glow
        } else if (art.style == "abstract") {
            palette.intensity *= 0.9f;
            palette.glow_factor *= 1.2f; // Softer, more glow
        } else if (art.style == "organic") {
            palette.primary = palette.primary.Lerp(Color::Green, 0.2f);
            palette.secondary = palette.secondary.Lerp(Color::Blue, 0.1f);
        }

        // Texture intensity affects overall visual complexity
        palette.intensity *= (0.8f + art.texture_intensity * 0.4f);
    }

    // Generate complementary palette
    static VisualPalette GetComplementaryPalette(const VisualPalette& base) {
        VisualPalette comp = base;

        // Simple complementary color calculation (opposite on color wheel)
        comp.primary = Color(1.0f - base.primary.r, 1.0f - base.primary.g, 1.0f - base.primary.b);
        comp.secondary = Color(1.0f - base.secondary.r, 1.0f - base.secondary.g, 1.0f - base.secondary.b);
        comp.accent = Color(1.0f - base.accent.r, 1.0f - base.accent.g, 1.0f - base.accent.b);

        return comp;
    }

    // Lerp between palettes for smooth transitions
    static VisualPalette LerpPalettes(const VisualPalette& from, const VisualPalette& to, float t) {
        VisualPalette result;
        result.primary = from.primary.Lerp(to.primary, t);
        result.secondary = from.secondary.Lerp(to.secondary, t);
        result.accent = from.accent.Lerp(to.accent, t);
        result.background = from.background.Lerp(to.background, t);
        result.intensity = from.intensity + (to.intensity - from.intensity) * t;
        result.glow_factor = from.glow_factor + (to.glow_factor - from.glow_factor) * t;
        return result;
    }

    // Get palette based on current NEXUS state
    static VisualPalette GetCurrentPalette() {
        auto& protocol = NexusProtocol::Instance();
        return GetPalette(protocol.GetProcessingMode(),
                         &protocol.GetLatestAudio(),
                         &protocol.GetLatestArt());
    }

    // Utility functions for specific color schemes
    static VisualPalette GetNeonPalette(ProcessingMode mode) {
        auto palette = GetPalette(mode);
        palette.intensity *= 1.5f;
        palette.glow_factor *= 2.0f;
        palette.background = Color::Black;
        return palette;
    }

    static VisualPalette GetPastelPalette(ProcessingMode mode) {
        auto palette = GetPalette(mode);
        palette.primary = Color::White.Lerp(palette.primary, 0.6f);
        palette.secondary = Color::White.Lerp(palette.secondary, 0.6f);
        palette.accent = Color::White.Lerp(palette.accent, 0.6f);
        palette.intensity *= 0.7f;
        palette.glow_factor *= 0.5f;
        return palette;
    }

    static VisualPalette GetMonochromePalette(ProcessingMode mode) {
        auto palette = GetPalette(mode);
        float avg = (palette.primary.r + palette.primary.g + palette.primary.b) / 3.0f;
        palette.primary = Color(avg, avg, avg);
        avg = (palette.secondary.r + palette.secondary.g + palette.secondary.b) / 3.0f;
        palette.secondary = Color(avg, avg, avg);
        avg = (palette.accent.r + palette.accent.g + palette.accent.b) / 3.0f;
        palette.accent = Color(avg, avg, avg);
        return palette;
    }
};

// Global palette cache for performance
class PaletteCache {
public:
    static VisualPalette GetCachedPalette(ProcessingMode mode, uint64_t audio_timestamp = 0, uint64_t art_timestamp = 0) {
        static std::unordered_map<int, VisualPalette> cache;
        static std::unordered_map<int, uint64_t> cache_timestamps;

        int key = static_cast<int>(mode) + (audio_timestamp % 1000) * 10 + (art_timestamp % 1000) * 10000;

        auto it = cache.find(key);
        if (it != cache.end()) {
            return it->second;
        }

        // Generate and cache new palette
        auto& protocol = NexusProtocol::Instance();
        auto palette = NexusVisuals::GetPalette(mode, &protocol.GetLatestAudio(), &protocol.GetLatestArt());
        cache[key] = palette;
        cache_timestamps[key] = std::max(audio_timestamp, art_timestamp);

        // Clean old entries (simple cleanup)
        if (cache.size() > 100) {
            cache.clear();
            cache_timestamps.clear();
        }

        return palette;
    }
};

} // namespace NEXUS
