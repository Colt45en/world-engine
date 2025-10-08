#include "NexusGameEngine.hpp"
#include "NexusResourceEngine.hpp"
#include <iostream>
#include <iomanip>
#include <random>
#include <chrono>
#include <thread>

using namespace NexusGame;

/**
 * NEXUS Resource Engine Demo
 * Enhanced version of your resource_engine_demo.cpp integrated with NEXUS Holy Beat System
 *
 * Features:
 * - Async resource loading with throttling
 * - Frustum culling (simplified)
 * - Audio-reactive resources synchronized with BPM
 * - Art-reactive resources with petal patterns and color sync
 * - LOD system with multiple model variants
 * - 2.5D world with Z-axis support for elevation
 * - Performance monitoring and metrics
 */

// NEXUS-enhanced renderer with audio/art visualization
class NexusConsoleRenderer : public INexusRenderer {
private:
    int frameCount = 0;

public:
    void beginFrame() override {
        frameCount++;
        if (frameCount % 30 == 0) { // Every 30 frames (~0.5 seconds at 60fps)
            std::cout << "\n🎵 === NEXUS RENDER FRAME " << frameCount << " ===\n";
        }
    }

    void draw(const RenderedObject& obj, const std::string& lod) override {
        if (frameCount % 30 == 0) { // Only log every 30 frames to avoid spam
            std::cout << "🎮 [Render] " << obj.model
                      << " pos=(" << std::fixed << std::setprecision(1)
                      << obj.position.x << "," << obj.position.y << "," << obj.position.z << ")"
                      << " LOD=" << lod;

            // Show NEXUS effects
            if (obj.audioIntensity != 1.0) {
                std::cout << " 🎵audio=" << std::setprecision(2) << obj.audioIntensity;
            }

            if (obj.color.r != 1.0 || obj.color.g != 1.0 || obj.color.b != 1.0) {
                std::cout << " 🎨color=(" << std::setprecision(2)
                          << obj.color.r << "," << obj.color.g << "," << obj.color.b << ")";
            }

            std::cout << "\n";
        }
    }

    void endFrame() override {
        if (frameCount % 30 == 0) {
            std::cout << "=== FRAME " << frameCount << " END ===\n";
        }
    }
};

// Enhanced seeding with NEXUS resource types
static void seedNexusResources(NexusResourceEngine& eng, NexusGameEngine& gameEngine,
                               const std::string& typeId, int count,
                               float worldX, float worldY, float worldZ, uint32_t seed = 1337) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dx(0.f, worldX);
    std::uniform_real_distribution<float> dy(0.f, worldY);
    std::uniform_real_distribution<float> dz(0.f, worldZ);

    std::cout << "🌱 Seeding " << count << " " << typeId << " resources...\n";

    for (int i = 0; i < count; ++i) {
        Vec3f pos{dx(rng), dy(rng), dz(rng)};
        Vec3f rot{0, rng() * 6.28f, 0}; // Random Y rotation
        Vec3f scale{0.8f + rng() * 0.4f, 0.8f + rng() * 0.4f, 0.8f + rng() * 0.4f}; // Random scale 0.8-1.2

        std::string instId = eng.placeResource(typeId, pos, rot, scale);

        if (i % 100 == 0 && i > 0) {
            std::cout << "  ✨ Placed " << i << " " << typeId << " instances\n";
        }
    }
}

// Performance monitoring
void logPerformanceStats(const NexusResourceEngine& engine,
                        const NexusGameEngine& gameEngine,
                        double deltaTime) {
    static int logCounter = 0;
    logCounter++;

    if (logCounter % 60 == 0) { // Every 60 frames (~1 second)
        auto stats = engine.getStats();

        std::cout << "\n📊 === PERFORMANCE STATS ===\n";
        std::cout << "🎮 Game Engine: " << (gameEngine.IsRunning() ? "Running" : "Stopped")
                  << " | DeltaTime: " << std::fixed << std::setprecision(2) << deltaTime * 1000 << "ms\n";
        std::cout << "🗺️  Chunks: " << stats.loadedChunks << "/" << stats.totalChunks << " loaded\n";
        std::cout << "📦 Resources: " << stats.loadedResources << "/" << stats.totalResources << " loaded\n";
        std::cout << "👁️  Visible: " << stats.visibleResources << " resources\n";
        std::cout << "🎵 Audio Sync: " << (gameEngine.GetParameters().bpm) << " BPM\n";
        std::cout << "🌸 Art Sync: " << gameEngine.GetParameters().petalCount << " petals\n";
        std::cout << "============================\n\n";
    }
}

// Enhanced assertions with NEXUS integration
static void nexusAssertions(const NexusResourceEngine& engine,
                           const NexusGameEngine& gameEngine,
                           const std::string& testResourceId) {
    // Check resource exists and is properly integrated
    const auto* resource = engine.getInstance(testResourceId);
    assert(resource && "Test resource must exist");

    // Check NEXUS game entity integration
    assert(resource->gameEntity && "Resource should have associated game entity");

    // Check component integration
    auto audioSync = resource->gameEntity->GetComponent<AudioSync>();
    auto artSync = resource->gameEntity->GetComponent<ArtSync>();

    if (audioSync) {
        std::cout << "✅ Audio sync component active on " << testResourceId << "\n";
    }

    if (artSync) {
        std::cout << "✅ Art sync component active on " << testResourceId << "\n";
    }

    std::cout << "✅ All NEXUS integration assertions passed\n";
}

int main() {
    std::cout << "🎵✨ NEXUS Resource Engine Demo Starting... ✨🎵\n\n";

    // 1) Create NEXUS game engine first
    NexusGameEngine gameEngine;
    NexusGameEngine::SystemParameters gameParams;
    gameParams.bpm = 128.0;           // Slightly faster BPM for demo
    gameParams.harmonics = 8;         // More harmonics for complexity
    gameParams.petalCount = 12;       // More petals for art patterns
    gameParams.terrainRoughness = 0.6; // Moderate terrain complexity

    if (!gameEngine.Initialize(gameParams)) {
        std::cerr << "❌ Failed to initialize NEXUS Game Engine!\n";
        return -1;
    }

    std::cout << "✅ NEXUS Game Engine initialized\n";
    std::cout << "   🎵 BPM: " << gameParams.bpm << "\n";
    std::cout << "   🎶 Harmonics: " << gameParams.harmonics << "\n";
    std::cout << "   🌸 Petals: " << gameParams.petalCount << "\n\n";

    // 2) Create resource engine and camera
    NexusResourceEngine resourceEngine;
    NexusCamera camera;
    camera.position = { 500.f, 500.f, 50.f }; // Start elevated
    camera.forward = { 0.f, 0.f, -1.f };
    camera.fov = 75.0f;

    // Configure resource engine for demo
    resourceEngine.worldBounds = { 5000.f, 5000.f, 200.f }; // Smaller world for demo
    resourceEngine.chunkSize = 200.f;
    resourceEngine.loadDistance = 400.f;
    resourceEngine.unloadDistance = 600.f;
    resourceEngine.enableAsyncLoading = true;
    resourceEngine.enableFrustumCulling = true;
    resourceEngine.enableAudioReactivity = true;
    resourceEngine.enableArtSync = true;

    resourceEngine.initialize(&camera, &gameEngine);

    // 3) Register NEXUS-enhanced resource types
    std::cout << "🎨 Registering NEXUS resource types...\n";

    // Audio-reactive trees that pulse with the beat
    std::vector<LODLevel> treeLODs = {
        {80.f, "ultra", "nexus_tree_ultra.glb"},
        {200.f, "high", "nexus_tree_high.glb"},
        {400.f, "medium", "nexus_tree_med.glb"},
        {600.f, "low", "nexus_tree_low.glb"},
        {800.f, "billboard", "nexus_tree_billboard.png", 1000.f}
    };
    resourceEngine.registerResourceType(
        "nexus_tree",
        "nexus_tree_base.glb",
        {"tree_albedo.png", "tree_normal.png", "tree_ao.png"},
        treeLODs,
        true,  // audioReactive
        true,  // artReactive
        false  // physicsEnabled
    );

    // Art-reactive crystals that change color with petal patterns
    std::vector<LODLevel> crystalLODs = {
        {100.f, "ultra", "nexus_crystal_ultra.glb"},
        {250.f, "high", "nexus_crystal_high.glb"},
        {500.f, "medium", "nexus_crystal_med.glb"},
        {750.f, "billboard", "nexus_crystal_billboard.png", 900.f}
    };
    resourceEngine.registerResourceType(
        "nexus_crystal",
        "nexus_crystal_base.glb",
        {"crystal_albedo.png", "crystal_emission.png"},
        crystalLODs,
        false, // audioReactive
        true,  // artReactive
        false  // physicsEnabled
    );

    // Physics-enabled rocks that can be affected by audio forces
    resourceEngine.registerResourceType(
        "nexus_rock",
        "nexus_rock_base.glb",
        {"rock_albedo.png", "rock_normal.png", "rock_roughness.png"},
        {}, // Use default LODs
        true, // audioReactive
        false, // artReactive
        true   // physicsEnabled
    );

    std::cout << "✅ Registered 3 NEXUS resource types\n\n";

    // 4) Place test resources and seed the world
    std::cout << "🌍 Populating NEXUS world...\n";

    // Place a controlled test resource near camera
    const std::string testTreeId = resourceEngine.placeResource(
        "nexus_tree",
        Vec3f{520.f, 520.f, 0.f}, // Close to camera
        Vec3f{0, 0, 0},
        Vec3f{1.5f, 1.5f, 1.5f}  // Larger scale for visibility
    );

    // Seed the world with resources
    seedNexusResources(resourceEngine, gameEngine, "nexus_tree", 300,
                      resourceEngine.worldBounds.x, resourceEngine.worldBounds.y, 100.f, 42);
    seedNexusResources(resourceEngine, gameEngine, "nexus_crystal", 150,
                      resourceEngine.worldBounds.x, resourceEngine.worldBounds.y, 50.f, 123);
    seedNexusResources(resourceEngine, gameEngine, "nexus_rock", 200,
                      resourceEngine.worldBounds.x, resourceEngine.worldBounds.y, 20.f, 456);

    std::cout << "✅ World populated with 650+ NEXUS resources\n\n";

    // 5) Create NEXUS-enhanced renderer
    NexusConsoleRenderer renderer;

    // 6) Simulate NEXUS system data for synchronization
    std::string nexusSystemData = R"({
        "timestamp": "2025-09-26T12:00:00Z",
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
            "bpm": 128,
            "harmonics": 8,
            "petalCount": 12,
            "terrainRoughness": 0.6
        }
    })";

    // 7) Main NEXUS game loop
    std::cout << "🚀 Starting NEXUS integrated game loop...\n";
    std::cout << "🎮 Camera path: Spiral tour of the world with elevation changes\n\n";

    int frameCount = 0;
    auto startTime = std::chrono::high_resolution_clock::now();
    const int maxFrames = 600; // 10 seconds at 60fps

    while (gameEngine.IsRunning() && frameCount < maxFrames) {
        auto frameStart = std::chrono::high_resolution_clock::now();

        // Update NEXUS systems
        gameEngine.SyncWithNexusSystems(nexusSystemData);
        gameEngine.Update();

        // Move camera in an interesting spiral pattern with elevation
        float t = frameCount * 0.02f;
        float spiral = t * 0.3f;
        float radius = 200.f + std::sin(t * 0.1f) * 100.f; // Varying radius

        camera.position.x = 500.f + std::cos(spiral) * radius;
        camera.position.y = 500.f + std::sin(spiral) * radius;
        camera.position.z = 50.f + std::sin(t * 0.15f) * 30.f; // Elevation changes

        // Point camera toward world center
        Vec3f toCenter = {500.f - camera.position.x, 500.f - camera.position.y, 0.f - camera.position.z};
        float len = std::sqrt(toCenter.x*toCenter.x + toCenter.y*toCenter.y + toCenter.z*toCenter.z);
        if (len > 0) {
            camera.forward = {toCenter.x/len, toCenter.y/len, toCenter.z/len};
        }

        // Update resource engine
        resourceEngine.update(gameEngine.GetDeltaTime());

        // Render
        resourceEngine.render(renderer);

        // Dynamic parameter changes for demo
        if (frameCount % 120 == 0) { // Every 2 seconds
            double newBPM = 120.0 + 20.0 * std::sin(frameCount * 0.01);
            int newPetals = 8 + static_cast<int>(4 * std::sin(frameCount * 0.02));

            gameEngine.SetBPM(newBPM);
            gameEngine.SetPetalCount(newPetals);

            std::cout << "🎛️  [Frame " << frameCount << "] BPM: " << std::fixed << std::setprecision(1)
                      << newBPM << ", Petals: " << newPetals << "\n";
        }

        // Performance logging
        logPerformanceStats(resourceEngine, gameEngine, gameEngine.GetDeltaTime());

        frameCount++;

        // Sleep to maintain 60 FPS
        auto frameEnd = std::chrono::high_resolution_clock::now();
        auto frameDuration = std::chrono::duration_cast<std::chrono::microseconds>(frameEnd - frameStart);
        auto targetFrameTime = std::chrono::microseconds(16667); // 60 FPS

        if (frameDuration < targetFrameTime) {
            std::this_thread::sleep_for(targetFrameTime - frameDuration);
        }
    }

    // 8) Final assertions and cleanup
    std::cout << "\n🔍 Running NEXUS integration tests...\n";
    nexusAssertions(resourceEngine, gameEngine, testTreeId);

    auto endTime = std::chrono::high_resolution_clock::now();
    auto totalTime = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();

    auto finalStats = resourceEngine.getStats();

    std::cout << "\n📊 === FINAL DEMO RESULTS ===\n";
    std::cout << "⏱️  Total Runtime: " << totalTime << "ms (" << frameCount << " frames)\n";
    std::cout << "🎯 Average FPS: " << std::fixed << std::setprecision(1) << (frameCount * 1000.0 / totalTime) << "\n";
    std::cout << "🗺️  Final Chunks: " << finalStats.loadedChunks << "/" << finalStats.totalChunks << "\n";
    std::cout << "📦 Final Resources: " << finalStats.loadedResources << "/" << finalStats.totalResources << "\n";
    std::cout << "👁️  Final Visible: " << finalStats.visibleResources << "\n";
    std::cout << "🎵 Final BPM: " << gameEngine.GetParameters().bpm << "\n";
    std::cout << "🌸 Final Petals: " << gameEngine.GetParameters().petalCount << "\n";
    std::cout << "============================\n";

    // Shutdown
    std::cout << "\n🔄 Shutting down NEXUS systems...\n";
    gameEngine.Shutdown();

    std::cout << "\n🎵✨ NEXUS Resource Engine Demo completed successfully! ✨🎵\n";
    std::cout << "\n💡 Integration Features Demonstrated:\n";
    std::cout << "   ✅ Async resource loading with throttling\n";
    std::cout << "   ✅ Frustum culling integration\n";
    std::cout << "   ✅ Multi-level LOD system\n";
    std::cout << "   ✅ Audio-reactive resources synced to BPM\n";
    std::cout << "   ✅ Art-reactive resources with petal patterns\n";
    std::cout << "   ✅ Physics-enabled resources\n";
    std::cout << "   ✅ 2.5D world with Z-axis elevation\n";
    std::cout << "   ✅ Performance monitoring and metrics\n";
    std::cout << "   ✅ NEXUS Holy Beat System integration\n\n";

    return 0;
}
