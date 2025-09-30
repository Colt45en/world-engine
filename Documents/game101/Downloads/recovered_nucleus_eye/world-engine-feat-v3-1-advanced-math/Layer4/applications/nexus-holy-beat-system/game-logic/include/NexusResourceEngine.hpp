#pragma once

#include "GameComponents.hpp"
#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
#include <queue>
#include <mutex>
#include <future>

namespace NexusGame {

// Forward declarations
class NexusGameEngine;

// ------------------------------------------------------------
// Extended math types for resource engine
// ------------------------------------------------------------
struct Vec3f {
    float x{0}, y{0}, z{0};

    Vec3f() = default;
    Vec3f(float x_, float y_, float z_) : x(x_), y(y_), z(z_) {}

    // Conversion from Transform::Vector3
    Vec3f(const Transform::Vector3& v) : x(static_cast<float>(v.x)),
                                        y(static_cast<float>(v.y)),
                                        z(static_cast<float>(v.z)) {}

    Transform::Vector3 toVector3() const {
        return Transform::Vector3(static_cast<double>(x),
                                  static_cast<double>(y),
                                  static_cast<double>(z));
    }
};

static inline float distance(const Vec3f& a, const Vec3f& b) {
    const float dx = a.x - b.x;
    const float dy = a.y - b.y;
    const float dz = a.z - b.z;
    return std::sqrt(dx*dx + dy*dy + dz*dz);
}

// ------------------------------------------------------------
// NEXUS-enhanced renderer interface
// ------------------------------------------------------------
struct RenderedObject {
    std::string model;
    Vec3f position{};
    Vec3f rotation{};
    Vec3f scale{1,1,1};
    ArtSync::Color color{1,1,1,1}; // NEXUS art sync color
    double audioIntensity{1.0};    // NEXUS audio sync intensity
};

struct INexusRenderer {
    virtual ~INexusRenderer() = default;
    virtual void draw(const RenderedObject& obj, const std::string& lod) = 0;
    virtual void beginFrame() = 0;
    virtual void endFrame() = 0;
};

// Console renderer for testing
struct ConsoleRenderer : INexusRenderer {
    void beginFrame() override {
        std::cout << "\n=== NEXUS RENDER FRAME ===\n";
    }

    void draw(const RenderedObject& obj, const std::string& lod) override {
        std::cout << "[Render] " << obj.model
                  << " pos=(" << obj.position.x << "," << obj.position.y << "," << obj.position.z << ")"
                  << " LOD=" << lod
                  << " color=(" << obj.color.r << "," << obj.color.g << "," << obj.color.b << ")"
                  << " audio=" << std::fixed << std::setprecision(2) << obj.audioIntensity << "\n";
    }

    void endFrame() override {
        std::cout << "=== FRAME END ===\n";
    }
};

// ------------------------------------------------------------
// Resource engine data structures
// ------------------------------------------------------------
struct LODLevel {
    float distance;        // switch at this distance (inclusive)
    std::string detail;    // "high", "medium", "low", "billboard"
    std::string model;     // optional: specific model for this LOD
    float cullDistance{-1}; // optional: cull entirely beyond this distance
};

struct ResourceType {
    std::string id;
    std::string model;
    std::vector<std::string> textures;
    std::vector<LODLevel> lodLevels; // sorted by distance ascending
    std::vector<std::string> instanceIds; // instances of this type

    // NEXUS integration properties
    bool audioReactive{false};     // responds to audio sync
    bool artReactive{false};       // responds to art patterns
    bool physicsEnabled{false};    // has physics simulation
    float maxAudioScale{2.0f};     // maximum scale from audio intensity
};

struct ResourceInstance {
    std::string id;
    std::string typeId;
    Vec3f position{};
    Vec3f rotation{};
    Vec3f scale{1,1,1};
    std::string currentLOD;
    bool visible{false};
    bool loaded{false};
    bool loadRequested{false};
    RenderedObject renderedObject{};

    // NEXUS integration
    std::shared_ptr<GameEntity> gameEntity{nullptr}; // associated game entity
    double lastBeatTime{0.0};
    ArtSync::Color baseColor{1,1,1,1};
};

struct Chunk {
    std::string id;
    int cx{0}, cy{0};          // chunk indices
    Vec3f position{};          // lower-left corner origin (x,y)
    bool loaded{false};
    bool loadRequested{false};
    std::vector<std::string> resourceIds; // instance ids in this chunk

    // Async loading support
    std::future<void> loadFuture;
    std::chrono::steady_clock::time_point lastUpdate;
};

struct NexusCamera {
    Vec3f position{};
    Vec3f forward{0, 0, -1};   // view direction
    Vec3f up{0, 1, 0};         // up vector
    float fov{60.0f};          // field of view in degrees
    float nearPlane{0.1f};
    float farPlane{1000.0f};

    // Frustum culling support
    struct Frustum {
        std::array<Vec3f, 4> planes; // left, right, top, bottom normals
        bool initialized{false};
    } frustum;
};

// ------------------------------------------------------------
// Async resource loading
// ------------------------------------------------------------
struct LoadJob {
    enum Type { CHUNK_LOAD, CHUNK_UNLOAD, RESOURCE_LOAD, RESOURCE_UNLOAD };
    Type type;
    std::string id;
    std::chrono::steady_clock::time_point timestamp;
};

class AsyncLoader {
private:
    std::queue<LoadJob> jobQueue;
    std::mutex queueMutex;
    std::condition_variable jobCondition;
    std::atomic<bool> running{false};
    std::thread workerThread;

    static constexpr int MAX_JOBS_PER_FRAME = 4;

public:
    AsyncLoader() = default;
    ~AsyncLoader() { stop(); }

    void start() {
        running = true;
        workerThread = std::thread(&AsyncLoader::workerLoop, this);
    }

    void stop() {
        running = false;
        jobCondition.notify_all();
        if (workerThread.joinable()) {
            workerThread.join();
        }
    }

    void enqueue(LoadJob job) {
        std::lock_guard<std::mutex> lock(queueMutex);
        jobQueue.push(std::move(job));
        jobCondition.notify_one();
    }

    bool hasJobs() const {
        std::lock_guard<std::mutex> lock(queueMutex);
        return !jobQueue.empty();
    }

private:
    void workerLoop() {
        while (running) {
            std::unique_lock<std::mutex> lock(queueMutex);
            jobCondition.wait(lock, [this] { return !jobQueue.empty() || !running; });

            if (!running) break;

            int jobsProcessed = 0;
            while (!jobQueue.empty() && jobsProcessed < MAX_JOBS_PER_FRAME) {
                LoadJob job = jobQueue.front();
                jobQueue.pop();
                lock.unlock();

                // Simulate loading work
                processJob(job);

                lock.lock();
                jobsProcessed++;
            }
        }
    }

    void processJob(const LoadJob& job) {
        // Simulate I/O latency
        std::this_thread::sleep_for(std::chrono::milliseconds(1 + rand() % 10));

        // Job processing would happen here
        // For now, just simulate work
    }
};

// ------------------------------------------------------------
// NEXUS Resource Engine Core
// ------------------------------------------------------------
class NexusResourceEngine {
public:
    // Configuration
    Vec3f worldBounds{10000.f, 10000.f, 1000.f}; // Added Z dimension
    float chunkSize{250.f};
    float loadDistance{500.f};
    float unloadDistance{750.f};
    float cullingDistance{1000.f};

    // NEXUS integration
    bool enableAsyncLoading{true};
    bool enableFrustumCulling{true};
    bool enableAudioReactivity{true};
    bool enableArtSync{true};
    int maxResourcesPerFrame{100};

    // State
    NexusCamera* camera{nullptr};
    NexusGameEngine* gameEngine{nullptr};

public:
    NexusResourceEngine() {
        if (enableAsyncLoading) {
            asyncLoader.start();
        }
    }

    ~NexusResourceEngine() {
        asyncLoader.stop();
    }

    // API mirrors your JS version with NEXUS enhancements
    void initialize(NexusCamera* cam, NexusGameEngine* engine) {
        camera = cam;
        gameEngine = engine;
        generateChunks();
        std::cout << "[NEXUS ResourceEngine] Initialized with world bounds ("
                  << worldBounds.x << ", " << worldBounds.y << ", " << worldBounds.z << ")\n";
    }

    void registerResourceType(const std::string& id,
                              const std::string& model,
                              const std::vector<std::string>& textures,
                              std::vector<LODLevel> lodLevels = {},
                              bool audioReactive = false,
                              bool artReactive = false,
                              bool physicsEnabled = false) {
        if (lodLevels.empty()) {
            lodLevels = {
                {50.f, "ultra", model + "_ultra.glb"},
                {150.f, "high", model + "_high.glb"},
                {300.f, "medium", model + "_med.glb"},
                {500.f, "low", model + "_low.glb"},
                {800.f, "billboard", model + "_billboard.png", 1000.f}
            };
        }
        std::sort(lodLevels.begin(), lodLevels.end(),
                  [](const LODLevel& a, const LODLevel& b){ return a.distance < b.distance; });

        ResourceType type;
        type.id = id;
        type.model = model;
        type.textures = textures;
        type.lodLevels = std::move(lodLevels);
        type.audioReactive = audioReactive;
        type.artReactive = artReactive;
        type.physicsEnabled = physicsEnabled;

        resourceTypes[id] = std::move(type);

        std::cout << "[NEXUS ResourceEngine] Registered type '" << id
                  << "' audio=" << audioReactive
                  << " art=" << artReactive
                  << " physics=" << physicsEnabled << "\n";
    }

    // Returns instance id - enhanced with NEXUS entity creation
    std::string placeResource(const std::string& typeId,
                              const Vec3f& position,
                              const Vec3f& rotation = {},
                              const Vec3f& scale = {1,1,1}) {
        auto it = resourceTypes.find(typeId);
        if (it == resourceTypes.end()) return {};

        const std::string instId = makeInstanceId(typeId);
        ResourceInstance inst;
        inst.id = instId;
        inst.typeId = typeId;
        inst.position = position;
        inst.rotation = rotation;
        inst.scale = scale;

        // Create associated NEXUS game entity if engine is available
        if (gameEngine) {
            auto entity = gameEngine->CreateEntity("Resource_" + instId);

            // Add transform component
            auto transform = entity->AddComponent<Transform>();
            transform->SetPosition(position.toVector3());
            transform->SetRotation(Transform::Vector3(rotation.x, rotation.y, rotation.z));
            transform->SetScale(Transform::Vector3(scale.x, scale.y, scale.z));

            // Add components based on resource type
            const auto& type = it->second;
            if (type.audioReactive) {
                auto audioSync = entity->AddComponent<AudioSync>();
                audioSync->SetSyncMode(AudioSync::SyncMode::BPM_PULSE);
                audioSync->SetIntensity(0.5);
            }

            if (type.artReactive) {
                auto artSync = entity->AddComponent<ArtSync>();
                artSync->SetPatternMode(ArtSync::PatternMode::PETAL_FORMATION);
            }

            if (type.physicsEnabled) {
                auto physics = entity->AddComponent<Physics>();
                physics->SetMass(1.0);
                physics->SetUseGravity(true);
            }

            inst.gameEntity = entity;
        }

        instances.emplace(instId, std::move(inst));
        it->second.instanceIds.push_back(instId);

        const std::string chunkId = chunkIdFromPosition(position);
        if (auto* chunk = findChunk(chunkId)) {
            chunk->resourceIds.push_back(instId);
        }
        return instId;
    }

    void update(double deltaTime) {
        if (!camera) return;

        const Vec3f camPos = camera->position;

        // Update frustum if enabled
        if (enableFrustumCulling) {
            updateCameraFrustum();
        }

        // Update chunks
        updateChunks(camPos);

        // Update visibility and LOD for loaded instances
        int resourcesProcessed = 0;
        for (const auto& id : loadedResourceIds) {
            if (resourcesProcessed >= maxResourcesPerFrame) break;

            auto& r = instances[id];
            updateVisibility(r, camPos, deltaTime);
            updateNexusEffects(r, deltaTime);
            resourcesProcessed++;
        }
    }

    void render(INexusRenderer& renderer) {
        renderer.beginFrame();

        // Draw only visible resources
        for (const auto& id : visibleResourceIds) {
            const auto& r = instances[id];
            if (r.loaded && !r.currentLOD.empty()) {
                renderer.draw(r.renderedObject, r.currentLOD);
            }
        }

        renderer.endFrame();
    }

    // Enhanced diagnostics
    const Chunk* getChunk(const std::string& id) const {
        if (auto it = chunkIndex.find(id); it != chunkIndex.end()) {
            return &chunks[it->second];
        }
        return nullptr;
    }

    const ResourceInstance* getInstance(const std::string& id) const {
        if (auto it = instances.find(id); it != instances.end()) return &it->second;
        return nullptr;
    }

    // Performance metrics
    struct Stats {
        int totalChunks{0};
        int loadedChunks{0};
        int totalResources{0};
        int loadedResources{0};
        int visibleResources{0};
        int culledByDistance{0};
        int culledByFrustum{0};
        double avgLoadTime{0.0};
    };

    Stats getStats() const {
        Stats stats;
        stats.totalChunks = static_cast<int>(chunks.size());
        stats.totalResources = static_cast<int>(instances.size());
        stats.loadedResources = static_cast<int>(loadedResourceIds.size());
        stats.visibleResources = static_cast<int>(visibleResourceIds.size());

        for (const auto& chunk : chunks) {
            if (chunk.loaded) stats.loadedChunks++;
        }

        return stats;
    }

private:
    // Data stores
    std::unordered_map<std::string, ResourceType> resourceTypes;
    std::unordered_map<std::string, ResourceInstance> instances;

    std::vector<Chunk> chunks;
    std::unordered_map<std::string, size_t> chunkIndex;

    std::unordered_set<std::string> loadedResourceIds;
    std::unordered_set<std::string> visibleSet;
    std::vector<std::string> visibleResourceIds; // stable order

    // Async loading
    AsyncLoader asyncLoader;

    // ---- Enhanced chunking with Z support ----
    void generateChunks() {
        chunks.clear();
        chunkIndex.clear();
        const int xChunks = static_cast<int>(std::ceil(worldBounds.x / chunkSize));
        const int yChunks = static_cast<int>(std::ceil(worldBounds.y / chunkSize));

        // For 3D worlds, we could add Z chunks, but keeping it 2.5D for now
        chunks.reserve(xChunks * yChunks);
        for (int cx = 0; cx < xChunks; ++cx) {
            for (int cy = 0; cy < yChunks; ++cy) {
                Chunk c;
                c.cx = cx; c.cy = cy;
                c.id = chunkIdFromCoords(cx, cy);
                c.position = Vec3f{cx * chunkSize, cy * chunkSize, 0.f};
                chunks.push_back(std::move(c));
                chunkIndex[chunks.back().id] = chunks.size() - 1;
            }
        }

        std::cout << "[NEXUS ResourceEngine] Generated " << chunks.size() << " chunks\n";
    }

    std::string chunkIdFromCoords(int cx, int cy) const {
        return "chunk_" + std::to_string(cx) + "_" + std::to_string(cy);
    }

    std::string chunkIdFromPosition(const Vec3f& p) const {
        const int cx = std::max(0, static_cast<int>(std::floor(p.x / chunkSize)));
        const int cy = std::max(0, static_cast<int>(std::floor(p.y / chunkSize)));
        return chunkIdFromCoords(cx, cy);
    }

    Chunk* findChunk(const std::string& id) {
        auto it = chunkIndex.find(id);
        if (it == chunkIndex.end()) return nullptr;
        return &chunks[it->second];
    }

    void updateChunks(const Vec3f& camPos) {
        for (auto& c : chunks) {
            const Vec3f center{c.position.x + chunkSize * 0.5f,
                              c.position.y + chunkSize * 0.5f,
                              0.f};
            const float d = distance(center, camPos);

            if (d <= loadDistance && !c.loaded && !c.loadRequested) {
                if (enableAsyncLoading) {
                    c.loadRequested = true;
                    LoadJob job{LoadJob::CHUNK_LOAD, c.id, std::chrono::steady_clock::now()};
                    asyncLoader.enqueue(job);
                    // For demo, load immediately
                    loadChunk(c);
                } else {
                    loadChunk(c);
                }
            } else if (d > unloadDistance && c.loaded) {
                unloadChunk(c);
            }
        }
    }

    // ---- Enhanced loading with NEXUS integration ----
    void loadChunk(Chunk& chunk) {
        chunk.loaded = true;
        chunk.loadRequested = false;

        for (const auto& resId : chunk.resourceIds) {
            auto& r = instances[resId];
            if (!r.loaded && !r.loadRequested) {
                if (enableAsyncLoading) {
                    r.loadRequested = true;
                    LoadJob job{LoadJob::RESOURCE_LOAD, resId, std::chrono::steady_clock::now()};
                    asyncLoader.enqueue(job);
                    // For demo, load immediately
                    loadResource(r);
                } else {
                    loadResource(r);
                }
            }
        }
    }

    void unloadChunk(Chunk& chunk) {
        chunk.loaded = false;
        for (const auto& resId : chunk.resourceIds) {
            auto& r = instances[resId];
            if (r.loaded) unloadResource(r);
        }
    }

    void loadResource(ResourceInstance& r) {
        r.loaded = true;
        r.loadRequested = false;

        const auto& type = resourceTypes[r.typeId];
        r.renderedObject.model = "loaded_" + type.model;
        r.renderedObject.position = r.position;
        r.renderedObject.rotation = r.rotation;
        r.renderedObject.scale = r.scale;
        r.baseColor = ArtSync::Color{0.8, 0.9, 1.0, 1.0}; // Cool NEXUS blue-white
        r.renderedObject.color = r.baseColor;
        r.renderedObject.audioIntensity = 1.0;

        loadedResourceIds.insert(r.id);
    }

    void unloadResource(ResourceInstance& r) {
        r.loaded = false;
        r.visible = false;
        r.loadRequested = false;
        r.currentLOD.clear();
        r.renderedObject = {};
        loadedResourceIds.erase(r.id);
        if (visibleSet.erase(r.id)) {
            removeFromVisibleList(r.id);
        }
    }

    // ---- Enhanced visibility with frustum culling ----
    void updateCameraFrustum() {
        // Simplified frustum calculation for demo
        // In production, you'd calculate proper frustum planes from view-projection matrix
        camera->frustum.initialized = true;
    }

    void updateVisibility(ResourceInstance& r, const Vec3f& camPos, double deltaTime) {
        if (!r.loaded) return;

        const float d = distance(r.position, camPos);

        // Distance culling
        if (d > cullingDistance) {
            if (r.visible) {
                r.visible = false;
                removeFromVisibleList(r.id);
            }
            return;
        }

        // Frustum culling (simplified - in practice you'd test against frustum planes)
        bool inFrustum = true;
        if (enableFrustumCulling && camera->frustum.initialized) {
            // Simplified frustum test - just check if roughly in front of camera
            Vec3f toResource = {r.position.x - camPos.x, r.position.y - camPos.y, r.position.z - camPos.z};
            float dot = toResource.x * camera->forward.x + toResource.y * camera->forward.y + toResource.z * camera->forward.z;
            inFrustum = (dot > 0); // Very simple front-facing test
        }

        // Visibility decision
        if (d <= loadDistance && inFrustum) {
            const std::string lod = determineLOD(r.typeId, d);
            if (r.currentLOD != lod) r.currentLOD = lod;
            if (!r.visible) {
                r.visible = true;
                addToVisibleList(r.id);
            }
        } else if (r.visible) {
            r.visible = false;
            removeFromVisibleList(r.id);
        }
    }

    void updateNexusEffects(ResourceInstance& r, double deltaTime) {
        if (!r.gameEntity || !r.visible) return;

        const auto& type = resourceTypes[r.typeId];

        // Update audio reactivity
        if (type.audioReactive && enableAudioReactivity) {
            auto audioSync = r.gameEntity->GetComponent<AudioSync>();
            if (audioSync) {
                if (audioSync->IsOnBeat()) {
                    r.lastBeatTime = 0.0; // Reset beat timer
                    r.renderedObject.audioIntensity = 1.5;
                }

                // Fade audio intensity
                r.lastBeatTime += deltaTime;
                if (r.lastBeatTime > 0.2) { // 200ms fade
                    r.renderedObject.audioIntensity = std::max(1.0, r.renderedObject.audioIntensity - deltaTime * 2.0);
                }

                // Scale based on audio intensity
                float scaleMultiplier = 1.0f + (r.renderedObject.audioIntensity - 1.0) * 0.3f;
                r.renderedObject.scale = {r.scale.x * scaleMultiplier,
                                         r.scale.y * scaleMultiplier,
                                         r.scale.z * scaleMultiplier};
            }
        }

        // Update art sync colors
        if (type.artReactive && enableArtSync) {
            auto artSync = r.gameEntity->GetComponent<ArtSync>();
            if (artSync) {
                r.renderedObject.color = artSync->GetCurrentColor();
            }
        }
    }

    std::string determineLOD(const std::string& typeId, float dist) const {
        const auto it = resourceTypes.find(typeId);
        if (it == resourceTypes.end()) return "low";

        for (const auto& lvl : it->second.lodLevels) {
            if (dist <= lvl.distance) {
                // Check if we should cull entirely
                if (lvl.cullDistance > 0 && dist > lvl.cullDistance) {
                    return ""; // Cull this resource
                }
                return lvl.detail;
            }
        }
        return "billboard"; // Furthest LOD
    }

    // ---- Visible list bookkeeping ----
    void addToVisibleList(const std::string& id) {
        if (visibleSet.insert(id).second) {
            visibleResourceIds.push_back(id);
        }
    }

    void removeFromVisibleList(const std::string& id) {
        if (!visibleSet.count(id)) return;
        visibleSet.erase(id);
        auto it = std::find(visibleResourceIds.begin(), visibleResourceIds.end(), id);
        if (it != visibleResourceIds.end()) visibleResourceIds.erase(it);
    }

    // ---- Utils ----
    static std::string makeInstanceId(const std::string& typeId) {
        static uint64_t counter{0};
        return typeId + "_" + std::to_string(++counter);
    }
};

} // namespace NexusGame
