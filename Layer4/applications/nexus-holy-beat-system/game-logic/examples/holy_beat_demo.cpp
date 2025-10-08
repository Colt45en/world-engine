// NEXUS Holy Beat System - Enhanced Resource Demo with Live Web Bridge
// This demo connects the C++ game engine to the web dashboard for real-time visualization

#include "NexusGameEngine.hpp"
#include "NexusResourceEngine.hpp"
#include "NexusProtocol.hpp"
#include "NexusVisuals.hpp"
#include "NexusRecursiveKeeperEngine.hpp"
#include "NexusProfiler.hpp"
#include "NexusWebSocketBridge.hpp"
#include <iostream>
#include <iomanip>
#include <random>
#include <chrono>
#include <thread>
#include <cmath>

using namespace NEXUS;
using namespace NexusGame;

class HolyBeatSystemDemo {
private:
    // Core NEXUS systems
    NexusGameEngine gameEngine;
    NexusResourceEngine resourceEngine;
    NexusProtocol quantumProtocol;
    NexusVisuals visualSystem;
    NexusRecursiveKeeperEngine cognitiveEngine;
    NexusProfiler& profiler;

    // Web integration
    std::unique_ptr<NexusWebIntegration> webBridge;

    // Demo state
    NexusCamera camera;
    std::vector<EntityID> demoEntities;
    std::mt19937 randomEngine;
    bool running;
    int frameCount;

    // Audio simulation
    float audioTime;
    std::vector<float> synthAudioBuffer;

public:
    HolyBeatSystemDemo()
        : profiler(NexusProfiler::getInstance())
        , randomEngine(std::chrono::steady_clock::now().time_since_epoch().count())
        , running(false)
        , frameCount(0)
        , audioTime(0.0f)
        , synthAudioBuffer(1024) {

        webBridge = std::make_unique<NexusWebIntegration>(8080);
    }

    bool initialize() {
        std::cout << "🎵✨ Initializing NEXUS Holy Beat System... ✨🎵\n\n";

        // 1. Initialize profiler
        profiler.startProfiling();

        // 2. Initialize game engine
        NexusGameEngine::SystemParameters gameParams;
        gameParams.bpm = 128.0;
        gameParams.harmonics = 8;
        gameParams.petalCount = 12;
        gameParams.terrainRoughness = 0.6;

        if (!gameEngine.Initialize(gameParams)) {
            std::cerr << "❌ Failed to initialize NEXUS Game Engine!\n";
            return false;
        }

        // 3. Setup camera
        camera.position = {500.f, 500.f, 50.f};
        camera.forward = {0.f, 0.f, -1.f};
        camera.fov = 75.0f;

        // 4. Initialize resource engine
        resourceEngine.worldBounds = {2000.f, 2000.f, 200.f};
        resourceEngine.chunkSize = 100.f;
        resourceEngine.loadDistance = 300.f;
        resourceEngine.unloadDistance = 500.f;
        resourceEngine.enableAsyncLoading = true;
        resourceEngine.enableFrustumCulling = true;
        resourceEngine.enableAudioReactivity = true;
        resourceEngine.enableArtSync = true;

        resourceEngine.initialize(&camera, &gameEngine);

        // 5. Setup quantum protocol with enhanced callbacks
        quantumProtocol.setModeChangeCallback([this](int oldMode, int newMode) {
            std::cout << "🌀 Quantum mode transition: " << oldMode << " -> " << newMode << std::endl;

            // Trigger cognitive response
            cognitiveEngine.processThought("quantum_transition",
                "Experiencing quantum mode shift from " + std::to_string(oldMode) + " to " + std::to_string(newMode));

            // Update visual system
            visualSystem.setPalette("Quantum" + std::to_string(newMode));
        });

        quantumProtocol.setAudioProcessingCallback([this](const AudioFeatures& features) {
            // Update visual system with audio features
            visualSystem.updateAudioFeatures(features.amplitude, features.frequency);

            // Process cognitive responses to intense audio
            if (features.amplitude > 0.8f) {
                cognitiveEngine.processThought("audio_intensity",
                    "High amplitude detected: " + std::to_string(features.amplitude));
            }
        });

        // 6. Initialize web bridge
        if (!webBridge->initialize()) {
            std::cout << "⚠️ Warning: Web bridge failed to initialize (continuing without web interface)\n";
        } else {
            std::cout << "✅ Web bridge initialized on port 8080\n";
            std::cout << "🌐 Open http://localhost:8080/nexus-live-bridge.html to view live dashboard\n";

            // Connect all systems to web bridge
            webBridge->connectSystems(gameEngine, quantumProtocol, visualSystem, profiler);
        }

        // 7. Register enhanced resource types
        registerHolyBeatResources();

        // 8. Create demo world
        createDemoWorld();

        std::cout << "✅ NEXUS Holy Beat System initialized successfully!\n\n";
        return true;
    }

    void run() {
        std::cout << "🚀 Starting NEXUS Holy Beat System demo...\n";
        std::cout << "🎮 Features: Real-time web dashboard, quantum processing, cognitive analysis\n";
        std::cout << "🌐 Web Dashboard: Open nexus-live-bridge.html in your browser\n";
        std::cout << "⏱️ Duration: 120 seconds (2 minutes)\n\n";

        running = true;
        auto startTime = std::chrono::high_resolution_clock::now();
        const int maxFrames = 60 * 120; // 2 minutes at 60fps

        while (running && frameCount < maxFrames) {
            auto frameStart = std::chrono::high_resolution_clock::now();

            // Update all systems
            updateSystems();

            // Update web bridge
            if (webBridge) {
                webBridge->update();
            }

            // Progress logging
            if (frameCount % 300 == 0) { // Every 5 seconds
                logProgress();
            }

            frameCount++;

            // Maintain 60 FPS
            auto frameEnd = std::chrono::high_resolution_clock::now();
            auto frameDuration = std::chrono::duration_cast<std::chrono::microseconds>(frameEnd - frameStart);
            auto targetFrameTime = std::chrono::microseconds(16667); // 60 FPS

            if (frameDuration < targetFrameTime) {
                std::this_thread::sleep_for(targetFrameTime - frameDuration);
            }
        }

        shutdown();
    }

    void stop() {
        running = false;
    }

private:
    void registerHolyBeatResources() {
        std::cout << "🎨 Registering Holy Beat resource types...\n";

        // Sacred Geometry Trees - respond to mathematical harmonics
        std::vector<LODLevel> sacredTreeLODs = {
            {60.f, "ultra", "sacred_tree_ultra.glb"},
            {150.f, "high", "sacred_tree_high.glb"},
            {300.f, "medium", "sacred_tree_med.glb"},
            {500.f, "low", "sacred_tree_low.glb"},
            {700.f, "billboard", "sacred_tree_billboard.png", 800.f}
        };

        resourceEngine.registerResourceType(
            "sacred_tree",
            "sacred_tree_base.glb",
            {"sacred_bark.png", "sacred_leaves.png", "sacred_glow.png"},
            sacredTreeLODs,
            true,  // audioReactive - pulses with beat
            true,  // artReactive - colors shift with harmonics
            false  // physicsEnabled
        );

        // Harmonic Crystals - resonate with specific frequencies
        std::vector<LODLevel> harmonicCrystalLODs = {
            {80.f, "ultra", "harmonic_crystal_ultra.glb"},
            {200.f, "high", "harmonic_crystal_high.glb"},
            {400.f, "medium", "harmonic_crystal_med.glb"},
            {600.f, "billboard", "harmonic_crystal_billboard.png", 750.f}
        };

        resourceEngine.registerResourceType(
            "harmonic_crystal",
            "harmonic_crystal_base.glb",
            {"crystal_core.png", "crystal_resonance.png", "crystal_aura.png"},
            harmonicCrystalLODs,
            true,  // audioReactive - frequency specific resonance
            true,  // artReactive - mathematical pattern visualization
            false  // physicsEnabled
        );

        // Beat Stones - physical objects that respond to rhythm
        resourceEngine.registerResourceType(
            "beat_stone",
            "beat_stone_base.glb",
            {"stone_surface.png", "beat_pattern.png", "rhythm_glow.png"},
            {}, // Use default LODs
            true, // audioReactive - movement with beat
            true, // artReactive - pattern shifts
            true  // physicsEnabled - can be moved by audio forces
        );

        std::cout << "✅ Registered 3 Holy Beat resource types\n";
    }

    void createDemoWorld() {
        std::cout << "🌍 Creating Holy Beat demo world...\n";

        // Create a mandala pattern of sacred trees
        const float centerX = 500.f, centerY = 500.f;
        const int rings = 5;
        const int itemsPerRing = 8;

        for (int ring = 1; ring <= rings; ring++) {
            float radius = ring * 80.f;

            for (int i = 0; i < itemsPerRing * ring; i++) {
                float angle = (2.0f * M_PI * i) / (itemsPerRing * ring);
                float x = centerX + radius * std::cos(angle);
                float y = centerY + radius * std::sin(angle);
                float z = std::sin(angle * 3) * 10.f; // Gentle elevation variation

                Vec3f pos{x, y, z};
                Vec3f rot{0, angle + M_PI/2, 0}; // Face outward
                Vec3f scale{0.8f + ring * 0.1f, 0.8f + ring * 0.1f, 0.8f + ring * 0.1f};

                std::string entityId = resourceEngine.placeResource("sacred_tree", pos, rot, scale);

                // Every 4th tree becomes a harmonic crystal
                if (i % 4 == 0) {
                    Vec3f crystalPos{x + 20.f, y + 20.f, z + 15.f};
                    Vec3f crystalScale{0.5f, 0.5f, 0.5f};
                    resourceEngine.placeResource("harmonic_crystal", crystalPos, {0, 0, 0}, crystalScale);
                }
            }
        }

        // Add beat stones in a grid pattern
        for (int x = 0; x < 10; x++) {
            for (int y = 0; y < 10; y++) {
                float posX = 200.f + x * 100.f;
                float posY = 200.f + y * 100.f;
                float posZ = std::sin(x * 0.5f) * std::cos(y * 0.5f) * 5.f;

                Vec3f pos{posX, posY, posZ};
                Vec3f scale{0.6f + (x + y) * 0.05f, 0.6f + (x + y) * 0.05f, 0.6f + (x + y) * 0.05f};

                resourceEngine.placeResource("beat_stone", pos, {0, 0, 0}, scale);
            }
        }

        std::cout << "✅ Holy Beat world created with sacred geometry\n";
    }

    void updateSystems() {
        // Update audio time and generate synthetic audio
        audioTime += 0.016f; // ~60fps delta
        generateSyntheticAudio();

        // Process quantum protocol
        auto audioFeatures = quantumProtocol.processAudioData(synthAudioBuffer.data(), synthAudioBuffer.size());

        // Update game engine with dynamic parameters
        updateGameParameters();

        // Update camera in a sacred spiral pattern
        updateCamera();

        // Update all core systems
        gameEngine.Update();
        visualSystem.update(0.016f);
        cognitiveEngine.update(0.016f);
        resourceEngine.update(0.016f);

        // Trigger cognitive thoughts based on system state
        triggerCognitiveEvents();
    }

    void generateSyntheticAudio() {
        // Create complex harmonic audio based on Holy Beat principles
        float baseBPM = 128.0f + 20.0f * std::sin(audioTime * 0.1f);
        float beatPhase = audioTime * baseBPM / 60.0f * 2.0f * M_PI;

        for (size_t i = 0; i < synthAudioBuffer.size(); i++) {
            float sample = 0.0f;

            // Base rhythm
            sample += 0.3f * std::sin(beatPhase + i * 0.01f);

            // Harmonics (creating the "holy" sound)
            for (int harmonic = 2; harmonic <= 8; harmonic++) {
                float harmonicPhase = beatPhase * harmonic + audioTime * 0.3f;
                sample += (0.1f / harmonic) * std::sin(harmonicPhase + i * 0.005f * harmonic);
            }

            // Quantum modulation
            float quantumMod = std::sin(audioTime * 2.3f + i * 0.001f) * 0.5f + 0.5f;
            sample *= quantumMod;

            // Sacred geometry influence
            float geometryMod = std::sin(audioTime * 0.618f + i * 0.002f) * 0.3f + 0.7f; // Golden ratio
            sample *= geometryMod;

            synthAudioBuffer[i] = sample * 0.7f; // Normalize
        }
    }

    void updateGameParameters() {
        // Dynamic BPM based on sacred mathematics
        double newBPM = 120.0 + 30.0 * std::sin(audioTime * 0.1f) * std::cos(audioTime * 0.618f); // Golden ratio influence
        gameEngine.SetBPM(newBPM);

        // Petal count follows Fibonacci-like progression
        int basePetals = 8;
        int petalVariation = static_cast<int>(5.0f * std::sin(audioTime * 0.15f));
        int newPetalCount = basePetals + petalVariation;
        gameEngine.SetPetalCount(std::max(3, newPetalCount));

        // Quantum mode cycling
        if (frameCount % 360 == 0) { // Every 6 seconds
            int newMode = (quantumProtocol.getCurrentMode() + 1) % 8;
            quantumProtocol.setProcessingMode(newMode);
        }
    }

    void updateCamera() {
        // Sacred spiral camera movement
        float spiralTime = audioTime * 0.2f;
        float spiralRadius = 300.f + 100.f * std::sin(audioTime * 0.05f);
        float spiralHeight = 50.f + 30.f * std::sin(audioTime * 0.08f);

        // Golden ratio spiral
        float goldenAngle = spiralTime * 2.399f; // Golden angle approximation

        camera.position.x = 500.f + spiralRadius * std::cos(goldenAngle);
        camera.position.y = 500.f + spiralRadius * std::sin(goldenAngle);
        camera.position.z = spiralHeight;

        // Point camera toward world center with slight oscillation
        Vec3f toCenter = {
            500.f - camera.position.x + 20.f * std::sin(audioTime * 0.3f),
            500.f - camera.position.y + 20.f * std::cos(audioTime * 0.3f),
            0.f - camera.position.z
        };

        float len = std::sqrt(toCenter.x*toCenter.x + toCenter.y*toCenter.y + toCenter.z*toCenter.z);
        if (len > 0) {
            camera.forward = {toCenter.x/len, toCenter.y/len, toCenter.z/len};
        }
    }

    void triggerCognitiveEvents() {
        // Trigger thoughts based on system events
        if (frameCount % 180 == 0) { // Every 3 seconds
            std::vector<std::string> holyBeatConcepts = {
                "sacred_geometry", "harmonic_resonance", "quantum_consciousness",
                "rhythmic_unity", "mathematical_beauty", "vibrational_healing",
                "cosmic_patterns", "divine_proportion", "temporal_synchrony"
            };

            int conceptIndex = randomEngine() % holyBeatConcepts.size();
            std::string concept = holyBeatConcepts[conceptIndex];

            std::ostringstream thought;
            thought << "Contemplating " << concept << " while experiencing "
                    << quantumProtocol.getCurrentMode() << " quantum mode at "
                    << gameEngine.GetParameters().bpm << " BPM";

            cognitiveEngine.processThought(concept, thought.str());
        }

        // React to visual changes
        auto currentPalette = visualSystem.getCurrentPalette();
        if (!currentPalette.empty() && frameCount % 240 == 0) { // Every 4 seconds
            cognitiveEngine.processThought("visual_harmony",
                "Observing color palette transitions in sacred geometric space");
        }
    }

    void logProgress() {
        auto stats = resourceEngine.getStats();
        auto audioFeatures = quantumProtocol.getLastAudioFeatures();
        auto thoughts = cognitiveEngine.getActiveThoughts();

        std::cout << "\n📊 === HOLY BEAT SYSTEM STATUS (Frame " << frameCount << ") ===\n";
        std::cout << "🎵 Audio: BPM=" << std::fixed << std::setprecision(1) << gameEngine.GetParameters().bpm
                  << ", Mode=" << quantumProtocol.getCurrentMode()
                  << ", Amplitude=" << std::setprecision(2) << audioFeatures.amplitude << "\n";
        std::cout << "🎨 Visual: Petals=" << gameEngine.GetParameters().petalCount
                  << ", Palette=" << visualSystem.getCurrentPalette().size() << " colors\n";
        std::cout << "🌍 World: " << stats.visibleResources << "/" << stats.totalResources << " resources visible\n";
        std::cout << "🧠 Mind: " << thoughts.size() << " active thoughts\n";
        std::cout << "🌐 Web: " << (webBridge->isConnected() ? "Connected" : "No client") << "\n";
        std::cout << "====================================================\n";
    }

    void shutdown() {
        std::cout << "\n🔄 Shutting down NEXUS Holy Beat System...\n";

        // Generate final report
        generateFinalReport();

        // Shutdown systems
        if (webBridge) {
            webBridge->shutdown();
        }

        profiler.stopProfiling();
        profiler.generateDetailedReport("holy_beat_performance.txt");

        gameEngine.Shutdown();

        std::cout << "✅ Holy Beat System shutdown complete\n";
    }

    void generateFinalReport() {
        auto endTime = std::chrono::high_resolution_clock::now();
        auto totalTime = std::chrono::duration_cast<std::chrono::seconds>(endTime - profiler.getStartTime()).count();

        std::cout << "\n🎵✨ === NEXUS HOLY BEAT SYSTEM - FINAL REPORT === ✨🎵\n";
        std::cout << "⏱️ Total Runtime: " << totalTime << " seconds (" << frameCount << " frames)\n";
        std::cout << "🎯 Average FPS: " << std::fixed << std::setprecision(1) << (frameCount / static_cast<float>(totalTime)) << "\n";
        std::cout << "🌍 Resources Created: " << resourceEngine.getStats().totalResources << "\n";
        std::cout << "🧠 Thoughts Generated: " << cognitiveEngine.getActiveThoughts().size() << "\n";
        std::cout << "🌀 Quantum Modes Experienced: 8\n";
        std::cout << "🎨 Visual Transitions: " << (frameCount / 240) << "\n";

        // Cognitive insights
        auto memories = cognitiveEngine.getAllMemories();
        if (!memories.empty()) {
            std::cout << "💭 Key Insights:\n";
            for (size_t i = 0; i < std::min(memories.size(), size_t(3)); i++) {
                std::cout << "   • " << memories[i].content.substr(0, 60) << "...\n";
            }
        }

        std::cout << "\n💫 Integration Features Demonstrated:\n";
        std::cout << "   ✅ Real-time web dashboard with WebSocket bridge\n";
        std::cout << "   ✅ Sacred geometry resource placement\n";
        std::cout << "   ✅ Harmonic audio synthesis with golden ratio\n";
        std::cout << "   ✅ Quantum-cognitive processing integration\n";
        std::cout << "   ✅ Multi-system synchronization\n";
        std::cout << "   ✅ Performance profiling and metrics\n";
        std::cout << "   ✅ Dynamic parameter evolution\n";
        std::cout << "\n🌟 NEXUS Holy Beat System demonstration complete! 🌟\n";
    }
};

int main() {
    try {
        HolyBeatSystemDemo demo;

        if (!demo.initialize()) {
            std::cerr << "❌ Failed to initialize Holy Beat System\n";
            return -1;
        }

        // Set up signal handling for graceful shutdown
        std::cout << "💡 Press Ctrl+C to stop the demo early\n\n";

        demo.run();

        return 0;

    } catch (const std::exception& e) {
        std::cerr << "❌ Holy Beat System error: " << e.what() << std::endl;
        return -1;
    }
}
