#include <iostream>
#include <thread>
#include <chrono>
#include <random>
#include <cmath>

// Include our new NEXUS Quantum-inspired headers
#include "NexusProtocol.hpp"
#include "NexusVisuals.hpp"
#include "NexusTrailRenderer.hpp"

// Include original NEXUS headers
#include "NexusGameEngine.hpp"
#include "GameEntity.hpp"

using namespace NEXUS;

// Simulate audio data updates (in real system, this would come from NEXUS server)
class AudioSimulator {
public:
    AudioSimulator() : gen_(rd_()), volume_dist_(0.0f, 1.0f), freq_dist_(0.0f, 1.0f),
                       bpm_dist_(80.0f, 180.0f), beat_dist_(0.0f, 1.0f) {}

    AudioData GenerateAudioData() {
        AudioData data;

        // Simulate varying audio characteristics
        static float time = 0.0f;
        time += 0.1f;

        data.volume = 0.3f + 0.7f * (std::sin(time * 0.5f) + 1.0f) * 0.5f;
        data.bass = 0.2f + 0.8f * (std::sin(time * 0.8f) + 1.0f) * 0.5f;
        data.midrange = 0.3f + 0.7f * (std::sin(time * 1.2f) + 1.0f) * 0.5f;
        data.treble = 0.1f + 0.9f * (std::sin(time * 1.5f) + 1.0f) * 0.5f;
        data.bpm = 120.0f + 20.0f * std::sin(time * 0.3f);
        data.beat_detected = beat_dist_(gen_) > 0.7f;
        data.spectral_centroid = freq_dist_(gen_);
        data.spectral_rolloff = freq_dist_(gen_);
        data.timestamp_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();

        return data;
    }

private:
    std::random_device rd_;
    std::mt19937 gen_;
    std::uniform_real_distribution<float> volume_dist_;
    std::uniform_real_distribution<float> freq_dist_;
    std::uniform_real_distribution<float> bpm_dist_;
    std::uniform_real_distribution<float> beat_dist_;
};

// Simulate art data updates
class ArtSimulator {
public:
    ArtSimulator() : gen_(rd_()), param_dist_(0.0f, 1.0f), color_dist_(0.0f, 1.0f) {}

    ArtData GenerateArtData() {
        ArtData data;

        static float art_time = 0.0f;
        art_time += 0.05f;

        data.complexity = 0.3f + 0.7f * (std::sin(art_time * 0.7f) + 1.0f) * 0.5f;
        data.brightness = 0.2f + 0.8f * (std::sin(art_time * 0.9f) + 1.0f) * 0.5f;
        data.contrast = 0.4f + 0.6f * (std::sin(art_time * 1.1f) + 1.0f) * 0.5f;
        data.saturation = 0.5f + 0.5f * (std::sin(art_time * 1.3f) + 1.0f) * 0.5f;

        // Cycle through different styles
        static int style_counter = 0;
        style_counter = (style_counter + 1) % 300; // Change every ~15 seconds
        if (style_counter < 100) data.style = "abstract";
        else if (style_counter < 200) data.style = "geometric";
        else data.style = "organic";

        // Generate dominant color
        data.dominant_color[0] = color_dist_(gen_);
        data.dominant_color[1] = color_dist_(gen_);
        data.dominant_color[2] = color_dist_(gen_);

        data.secondary_color[0] = color_dist_(gen_);
        data.secondary_color[1] = color_dist_(gen_);
        data.secondary_color[2] = color_dist_(gen_);

        data.texture_intensity = param_dist_(gen_);
        data.timestamp_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();

        return data;
    }

private:
    std::random_device rd_;
    std::mt19937 gen_;
    std::uniform_real_distribution<float> param_dist_;
    std::uniform_real_distribution<float> color_dist_;
};

// Demo entity with Quantum-inspired trail rendering
class QuantumTrailEntity : public GameEntity {
public:
    QuantumTrailEntity(const std::string& name) : name_(name) {
        // Create a trail renderer for this entity
        trail_ = NexusTrailManager::Instance().CreateTrail(name);

        // Configure trail
        trail_->SetMaxTrailPoints(50);
        trail_->SetTrailLifetime(3.0f);
        trail_->SetTrailWidth(1.0f);
        trail_->SetProcessingModeSync(true);
        trail_->SetAudioReactive(true);
        trail_->SetArtReactive(true);

        // Initialize position
        position_[0] = 0.0f;
        position_[1] = 0.0f;
        position_[2] = 0.0f;

        // Random movement parameters
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> speed_dist(0.5f, 2.0f);
        std::uniform_real_distribution<float> freq_dist(0.3f, 1.5f);

        speed_ = speed_dist(gen);
        freq_x_ = freq_dist(gen);
        freq_y_ = freq_dist(gen);
        freq_z_ = freq_dist(gen);
    }

    void Update(float deltaTime) override {
        GameEntity::Update(deltaTime);

        // Update movement (Lissajous curves for interesting patterns)
        static float time = 0.0f;
        time += deltaTime;

        position_[0] = 5.0f * std::sin(time * freq_x_ * speed_);
        position_[1] = 3.0f * std::cos(time * freq_y_ * speed_);
        position_[2] = 2.0f * std::sin(time * freq_z_ * speed_ * 0.5f);

        // Add point to trail
        trail_->AddPoint(position_, 1.0f);

        // Update trail
        trail_->Update(deltaTime);
    }

    void Render() {
        trail_->Render();
    }

    const std::string& GetName() const { return name_; }

private:
    std::string name_;
    std::shared_ptr<NexusTrailRenderer> trail_;
    float position_[3];
    float speed_;
    float freq_x_, freq_y_, freq_z_;
};

// Demo showing all processing modes
void DemoProcessingModes() {
    std::cout << "\n🌟 NEXUS Quantum Protocol - Processing Modes Demo\n";
    std::cout << "==================================================\n";

    auto& protocol = NexusProtocol::Instance();

    ProcessingMode modes[] = {
        ProcessingMode::MIRROR,
        ProcessingMode::COSINE,
        ProcessingMode::CHAOS,
        ProcessingMode::ABSORB,
        ProcessingMode::AMPLIFY,
        ProcessingMode::PULSE,
        ProcessingMode::FLOW,
        ProcessingMode::FRAGMENT
    };

    for (ProcessingMode mode : modes) {
        protocol.SetProcessingMode(mode);
        auto palette = NexusVisuals::GetPalette(mode);

        std::cout << "Mode: " << ProcessingModeToString(mode) << "\n";
        std::cout << "  Primary: R:" << palette.primary.r << " G:" << palette.primary.g
                  << " B:" << palette.primary.b << "\n";
        std::cout << "  Secondary: R:" << palette.secondary.r << " G:" << palette.secondary.g
                  << " B:" << palette.secondary.b << "\n";
        std::cout << "  Intensity: " << palette.intensity << ", Glow: " << palette.glow_factor << "\n";
        std::cout << "\n";

        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }
}

// Demo showing audio reactivity
void DemoAudioReactivity() {
    std::cout << "\n🎵 NEXUS Audio Reactivity Demo\n";
    std::cout << "==============================\n";

    auto& protocol = NexusProtocol::Instance();
    AudioSimulator audio_sim;

    // Enable auto-mode
    protocol.SetAutoModeEnabled(true);
    protocol.SetDebugMode(true);

    std::cout << "Auto-mode enabled - watch processing modes change with audio...\n\n";

    for (int i = 0; i < 20; ++i) {
        AudioData audio = audio_sim.GenerateAudioData();
        protocol.BroadcastAudioData(audio);
        protocol.UpdateAutoMode(audio);

        std::cout << "Audio: Vol:" << std::fixed << std::setprecision(2) << audio.volume
                  << " Bass:" << audio.bass << " Mid:" << audio.midrange << " Treble:" << audio.treble
                  << " BPM:" << audio.bpm;
        if (audio.beat_detected) std::cout << " [BEAT]";
        std::cout << " -> Mode: " << ProcessingModeToString(protocol.GetProcessingMode()) << "\n";

        std::this_thread::sleep_for(std::chrono::milliseconds(300));
    }
}

// Main demo loop with quantum trail entities
void RunQuantumTrailDemo() {
    std::cout << "\n✨ NEXUS Quantum Trail Rendering Demo\n";
    std::cout << "=====================================\n";

    // Create demo entities with trails
    std::vector<std::unique_ptr<QuantumTrailEntity>> entities;
    entities.push_back(std::make_unique<QuantumTrailEntity>("Alpha"));
    entities.push_back(std::make_unique<QuantumTrailEntity>("Beta"));
    entities.push_back(std::make_unique<QuantumTrailEntity>("Gamma"));

    AudioSimulator audio_sim;
    ArtSimulator art_sim;
    auto& protocol = NexusProtocol::Instance();

    protocol.SetDebugMode(false); // Less verbose for visual demo
    protocol.SetAutoModeEnabled(true);

    std::cout << "Running quantum trail simulation with audio/art reactivity...\n";
    std::cout << "Press Ctrl+C to exit\n\n";

    auto last_time = std::chrono::steady_clock::now();
    int frame_count = 0;

    // Demo loop
    while (frame_count < 200) { // Run for ~20 seconds at 10 FPS
        auto current_time = std::chrono::steady_clock::now();
        float delta_time = std::chrono::duration<float>(current_time - last_time).count();
        last_time = current_time;

        // Update audio/art data periodically
        if (frame_count % 10 == 0) {
            AudioData audio = audio_sim.GenerateAudioData();
            protocol.BroadcastAudioData(audio);
            protocol.UpdateAutoMode(audio);
        }

        if (frame_count % 20 == 0) {
            ArtData art = art_sim.GenerateArtData();
            protocol.BroadcastArtData(art);
        }

        // Update all entities
        for (auto& entity : entities) {
            entity->Update(delta_time);
        }

        // Clear screen (simple console animation)
        if (frame_count % 5 == 0) {
            system("cls"); // Windows - use "clear" on Linux/macOS

            std::cout << "🎮 NEXUS Quantum Trails [Frame " << frame_count << "]\n";
            std::cout << "Mode: " << ProcessingModeToString(protocol.GetProcessingMode()) << "\n";

            auto palette = NexusVisuals::GetCurrentPalette();
            std::cout << "Palette Intensity: " << std::fixed << std::setprecision(1)
                      << palette.intensity << ", Glow: " << palette.glow_factor << "\n\n";

            // Render all trails
            for (const auto& entity : entities) {
                entity->Render();
            }

            auto stats = protocol.GetStatistics();
            std::cout << "\n📊 Stats: Modes:" << stats.mode_changes
                      << " Audio:" << stats.audio_updates
                      << " Art:" << stats.art_updates
                      << " Beats:" << stats.beats_detected << "\n";
        }

        frame_count++;
        std::this_thread::sleep_for(std::chrono::milliseconds(100)); // ~10 FPS
    }

    std::cout << "\n✅ Quantum trail demo completed!\n";
}

// Palette comparison demo
void DemoPaletteComparison() {
    std::cout << "\n🎨 NEXUS Visual Palette Comparison\n";
    std::cout << "==================================\n";

    ProcessingMode modes[] = {ProcessingMode::MIRROR, ProcessingMode::COSINE,
                             ProcessingMode::CHAOS, ProcessingMode::ABSORB};

    for (ProcessingMode mode : modes) {
        std::cout << "🔹 " << ProcessingModeToString(mode) << " Palettes:\n";

        auto normal = NexusVisuals::GetPalette(mode);
        auto neon = NexusVisuals::GetNeonPalette(mode);
        auto pastel = NexusVisuals::GetPastelPalette(mode);
        auto mono = NexusVisuals::GetMonochromePalette(mode);

        std::cout << "  Normal   - Intensity: " << normal.intensity << ", Glow: " << normal.glow_factor << "\n";
        std::cout << "  Neon     - Intensity: " << neon.intensity << ", Glow: " << neon.glow_factor << "\n";
        std::cout << "  Pastel   - Intensity: " << pastel.intensity << ", Glow: " << pastel.glow_factor << "\n";
        std::cout << "  Mono     - Intensity: " << mono.intensity << ", Glow: " << mono.glow_factor << "\n";
        std::cout << "\n";
    }
}

int main() {
    std::cout << "🚀 NEXUS Holy Beat System - Quantum Protocol Integration Demo\n";
    std::cout << "=============================================================\n";
    std::cout << "Inspired by Unreal Engine Quantum Trail System\n";
    std::cout << "Adapted for NEXUS audio-reactive and art-reactive gameplay\n\n";

    try {
        // Initialize NEXUS Protocol
        auto& protocol = NexusProtocol::Instance();
        std::cout << "✅ NEXUS Protocol initialized\n";

        // Demo 1: Processing modes
        DemoProcessingModes();

        // Demo 2: Audio reactivity
        DemoAudioReactivity();

        // Demo 3: Palette comparison
        DemoPaletteComparison();

        // Demo 4: Main quantum trail demo
        RunQuantumTrailDemo();

        // Final stats
        auto stats = protocol.GetStatistics();
        std::cout << "\n📈 Final Statistics:\n";
        std::cout << "  Mode Changes: " << stats.mode_changes << "\n";
        std::cout << "  Audio Updates: " << stats.audio_updates << "\n";
        std::cout << "  Art Updates: " << stats.art_updates << "\n";
        std::cout << "  Beats Detected: " << stats.beats_detected << "\n";

        std::cout << "\n🎉 NEXUS Quantum Protocol Demo completed successfully!\n";
        std::cout << "Ready for integration with NEXUS Holy Beat System server.\n";

    } catch (const std::exception& e) {
        std::cerr << "❌ Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
