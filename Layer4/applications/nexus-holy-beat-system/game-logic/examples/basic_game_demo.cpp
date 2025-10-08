#include "NexusGameEngine.hpp"
#include "GameEntity.hpp"
#include "GameComponents.hpp"
#include <iostream>
#include <thread>
#include <chrono>

using namespace NexusGame;

/**
 * Basic Game Demo - Demonstrates NEXUS Game Engine integration
 * Shows audio-reactive particles synchronized with Holy Beat System
 */
int main() {
    std::cout << "🎮 NEXUS Game Engine Demo Starting...\n\n";

    // Create engine instance
    NexusGameEngine engine;

    // Initialize with NEXUS parameters (matching API: bpm=120, harmonics=6, petalCount=8)
    NexusGameEngine::SystemParameters params;
    params.bpm = 120.0;
    params.harmonics = 6;
    params.petalCount = 8;
    params.terrainRoughness = 0.4;

    if (!engine.Initialize(params)) {
        std::cerr << "❌ Failed to initialize NEXUS Game Engine!\n";
        return -1;
    }

    std::cout << "✅ Engine initialized with NEXUS parameters:\n";
    std::cout << "   🥁 BPM: " << params.bpm << "\n";
    std::cout << "   🎵 Harmonics: " << params.harmonics << "\n";
    std::cout << "   🌸 Petal Count: " << params.petalCount << "\n";
    std::cout << "   🏔️ Terrain Roughness: " << params.terrainRoughness << "\n\n";

    // Create audio-reactive particle system
    std::vector<std::shared_ptr<GameEntity>> particles;

    // Central mandala entity
    auto centerEntity = engine.CreateEntity("MandalaCenter");
    centerEntity->AddComponent<Transform>();
    auto centerArt = centerEntity->AddComponent<ArtSync>();
    centerArt->SetPatternMode(ArtSync::PatternMode::MANDALA_SYNC);
    centerArt->SetPetalCount(params.petalCount);

    std::cout << "🌸 Created mandala center with " << params.petalCount << " petals\n";

    // Create petal particles
    for (int i = 0; i < params.petalCount; ++i) {
        auto particle = engine.CreateEntity("Petal_" + std::to_string(i));

        // Add transform
        auto transform = particle->AddComponent<Transform>();
        double angle = (2.0 * M_PI * i) / params.petalCount;
        double radius = 5.0;
        transform->SetPosition(
            radius * std::cos(angle),
            0.0,
            radius * std::sin(angle)
        );

        // Add audio synchronization
        auto audioSync = particle->AddComponent<AudioSync>();
        audioSync->SetSyncMode(AudioSync::SyncMode::BPM_PULSE);
        audioSync->SetIntensity(0.8);
        audioSync->SetPhase(i * 0.1); // Slight phase offset per petal

        // Add art synchronization
        auto artSync = particle->AddComponent<ArtSync>();
        artSync->SetPatternMode(ArtSync::PatternMode::PETAL_FORMATION);
        artSync->SetPetalCount(params.petalCount);

        // Add physics
        auto physics = particle->AddComponent<Physics>();
        physics->SetMass(0.5);
        physics->SetUseGravity(false); // Float in space

        particles.push_back(particle);
    }

    std::cout << "✨ Created " << particles.size() << " audio-reactive particles\n";

    // Create harmonic oscillators
    for (int h = 1; h <= params.harmonics; ++h) {
        auto oscillator = engine.CreateEntity("Harmonic_" + std::to_string(h));

        auto transform = oscillator->AddComponent<Transform>();
        transform->SetPosition(0, h * 2.0, 0); // Stack vertically
        transform->SetScale(1.0 / h); // Smaller for higher harmonics

        auto audioSync = oscillator->AddComponent<AudioSync>();
        audioSync->SetSyncMode(AudioSync::SyncMode::HARMONIC_WAVE);
        audioSync->SetIntensity(1.0 / h); // Weaker for higher harmonics

        auto artSync = oscillator->AddComponent<ArtSync>();
        artSync->SetPatternMode(ArtSync::PatternMode::SPIRAL_MOTION);
    }

    std::cout << "🎵 Created " << params.harmonics << " harmonic oscillators\n\n";

    // Simulate NEXUS system data
    std::string nexusData = R"({
        "timestamp": "2025-01-26T12:00:00Z",
        "status": "running",
        "version": "1.0.0",
        "components": {
            "clockBus": "active",
            "audioEngine": "active",
            "artEngine": "active",
            "worldEngine": "active",
            "training": "ready"
        },
        "parameters": {
            "bpm": 120,
            "harmonics": 6,
            "petalCount": 8,
            "terrainRoughness": 0.4
        }
    })";

    // Main game loop
    std::cout << "🚀 Starting main game loop (press Ctrl+C to exit)...\n\n";

    int frameCount = 0;
    auto startTime = std::chrono::high_resolution_clock::now();

    while (engine.IsRunning() && frameCount < 300) { // Run for ~5 seconds at 60fps
        // Sync with NEXUS systems
        engine.SyncWithNexusSystems(nexusData);

        // Update game logic
        engine.Update();

        // Render (in a real implementation, this would draw to screen)
        engine.Render();

        // Print status every second
        if (frameCount % 60 == 0) {
            auto currentTime = std::chrono::high_resolution_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                currentTime - startTime).count();

            std::string status = engine.GetSystemStatusJson();
            std::cout << "⏱️ Frame " << frameCount << " | "
                     << "Time: " << elapsed << "ms | "
                     << "DeltaTime: " << std::fixed << std::setprecision(3)
                     << engine.GetDeltaTime() * 1000 << "ms\n";

            // Show entity status
            auto entities = engine.GetEntities();
            int activeEntities = 0;
            for (const auto& entity : entities) {
                if (entity->IsActive()) activeEntities++;
            }
            std::cout << "   🎯 Active Entities: " << activeEntities << "/" << entities.size() << "\n";

            // Simulate varying BPM
            double newBPM = 120.0 + 10.0 * std::sin(frameCount * 0.01);
            engine.SetBPM(newBPM);

            std::cout << "   🥁 Current BPM: " << std::fixed << std::setprecision(1) << newBPM << "\n\n";
        }

        frameCount++;

        // Sleep to maintain 60 FPS
        std::this_thread::sleep_for(std::chrono::milliseconds(16));
    }

    // Shutdown
    std::cout << "🔄 Shutting down NEXUS Game Engine...\n";
    engine.Shutdown();

    auto endTime = std::chrono::high_resolution_clock::now();
    auto totalTime = std::chrono::duration_cast<std::chrono::milliseconds>(
        endTime - startTime).count();

    std::cout << "\n📊 Demo Complete!\n";
    std::cout << "   Total Frames: " << frameCount << "\n";
    std::cout << "   Total Time: " << totalTime << "ms\n";
    std::cout << "   Average FPS: " << std::fixed << std::setprecision(1)
             << (frameCount * 1000.0 / totalTime) << "\n";
    std::cout << "\n🎵✨ NEXUS Game Logic Demo finished successfully! ✨🎵\n";

    return 0;
}
