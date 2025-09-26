#include "NexusGameEngine.hpp"
#include "NexusResourceEngine.hpp"
#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <mutex>
#include <optional>
#include <queue>
#include <random>
#include <set>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

using namespace NexusGame;

// ================================================================
// NEXUS-Enhanced Open World Simulation
// Integrates your game_world_demo.cpp with NEXUS Holy Beat System
// ================================================================

// Enhanced Vec3 with NEXUS integration (uses existing Transform::Vector3 internally)
struct NexusVec3 {
    double x{0}, y{0}, z{0};

    NexusVec3() = default;
    NexusVec3(double X, double Y, double Z) : x(X), y(Y), z(Z) {}
    NexusVec3(const Transform::Vector3& v) : x(v.x), y(v.y), z(v.z) {}

    NexusVec3 operator+(const NexusVec3& o) const { return {x+o.x, y+o.y, z+o.z}; }
    NexusVec3 operator-(const NexusVec3& o) const { return {x-o.x, y-o.y, z-o.z}; }
    NexusVec3 operator*(double s) const { return {x*s, y*s, z*s}; }
    NexusVec3& operator+=(const NexusVec3& o){ x+=o.x; y+=o.y; z+=o.z; return *this; }
    NexusVec3& operator*=(double s){ x*=s; y*=s; z*=s; return *this; }

    double dot(const NexusVec3& o) const { return x*o.x + y*o.y + z*o.z; }
    double len2() const { return x*x + y*y + z*z; }
    double len() const { return std::sqrt(len2()); }

    NexusVec3 normalized(double eps=1e-9) const {
        double L = len();
        if (L < eps) return {1,0,0};
        return {x/L, y/L, z/L};
    }

    Transform::Vector3 toVector3() const { return Transform::Vector3(x, y, z); }
};

static inline double distance(const NexusVec3& a, const NexusVec3& b){ return (a-b).len(); }
static inline double clamp(double v, double a, double b){ return std::max(a, std::min(b, v)); }

// ================================================================
// NEXUS Asset Resource Manager - Enhanced with Holy Beat Integration
// ================================================================
struct NexusAsset {
    std::string id;
    std::string type;
    std::string path;

    // Extended metadata for NEXUS integration
    struct {
        int vertices = 0;
        int width = 0, height = 0;
        int channels = 0;
        double duration = 0;
        // NEXUS-specific metadata
        bool audioReactive = false;
        bool artReactive = false;
        ArtSync::Color baseColor{1,1,1,1};
        double harmonicResponse = 1.0;
    } meta{};

    // Runtime NEXUS state
    double lastBeatTime = 0.0;
    double audioIntensity = 1.0;
    ArtSync::Color currentColor{1,1,1,1};
};

class NexusAssetResourceManager {
public:
    struct MemoryUsage { double total=0; };
    using Clock = std::chrono::steady_clock;

    NexusAssetResourceManager(NexusGameEngine* engine = nullptr) : gameEngine(engine) {
        basePaths["models"]   = "nexus_assets/models";
        basePaths["textures"] = "nexus_assets/textures";
        basePaths["audio"]    = "nexus_assets/audio";
        basePaths["materials"]= "nexus_assets/materials";
        basePaths["shaders"]  = "nexus_assets/shaders";
        basePaths["config"]   = "nexus_assets/config";
        memoryLimitMB = 512.0; // Larger limit for NEXUS assets
    }

    void setBasePath(const std::string& type, const std::string& path){
        basePaths[type] = path;
    }
    void setMemoryLimitMB(double mb){ memoryLimitMB = mb; }
    void setGameEngine(NexusGameEngine* engine) { gameEngine = engine; }

    // Priority request with NEXUS enhancements
    const NexusAsset& request(const std::string& type, const std::string& id, int priority=0, bool audioReactive=false, bool artReactive=false){
        if (!store.count(type)) store[type] = {};
        auto& bucket = store[type];

        if (bucket.count(id)){
            touch(type,id);
            updateNexusEffects(bucket[id]);
            return bucket.at(id);
        }

        Request rq{type, id, priority, audioReactive, artReactive};
        return loadNow(rq);
    }

    // Update NEXUS effects on all loaded assets
    void updateNexusEffects(double deltaTime) {
        if (!gameEngine) return;

        for (auto& [type, bucket] : store) {
            for (auto& [id, asset] : bucket) {
                updateNexusEffects(asset);
            }
        }
    }

    double totalMB() const { return mem.total; }
    bool has(const std::string& type, const std::string& id) const {
        auto it = store.find(type);
        if (it==store.end()) return false;
        return it->second.count(id) != 0;
    }

    // Get NEXUS-enhanced asset for rendering
    const NexusAsset* getNexusAsset(const std::string& type, const std::string& id) const {
        auto typeIt = store.find(type);
        if (typeIt == store.end()) return nullptr;
        auto assetIt = typeIt->second.find(id);
        if (assetIt == typeIt->second.end()) return nullptr;
        return &assetIt->second;
    }

private:
    struct Request{
        std::string type;
        std::string id;
        int priority{0};
        bool audioReactive{false};
        bool artReactive{false};
    };

    std::unordered_map<std::string, std::unordered_map<std::string, NexusAsset>> store;
    std::unordered_map<std::string, Clock::time_point> lastAccess;
    std::unordered_map<std::string, double> sizesMB;
    std::unordered_map<std::string, std::string> basePaths;
    double memoryLimitMB;
    MemoryUsage mem;
    NexusGameEngine* gameEngine;

    std::string key(const std::string& type, const std::string& id) const {
        return type + ":" + id;
    }

    std::string fullPath(const std::string& type, const std::string& id) const {
        auto it = basePaths.find(type);
        std::string base = (it==basePaths.end()) ? "nexus_assets" : it->second;
        return base + "/" + id;
    }

    NexusAsset load(const Request& rq){
        NexusAsset a;
        a.type = rq.type;
        a.id = rq.id;
        a.path = fullPath(rq.type, rq.id);
        a.meta.audioReactive = rq.audioReactive;
        a.meta.artReactive = rq.artReactive;

        // Synthetic metadata with NEXUS enhancements
        std::mt19937 rng(std::hash<std::string>{}(a.path));
        if (rq.type=="models"){
            a.meta.vertices = 2000 + (rng()%20000);
            if (rq.audioReactive) {
                a.meta.harmonicResponse = 0.5 + (rng() % 1000) / 2000.0; // 0.5-1.0
            }
        } else if (rq.type=="textures"){
            a.meta.width = 512 * (1 + (rng()%4));
            a.meta.height= 512 * (1 + (rng()%4));
            if (rq.artReactive) {
                // Generate base color for art-reactive textures
                a.meta.baseColor = ArtSync::Color{
                    0.5 + (rng() % 500) / 1000.0,
                    0.5 + (rng() % 500) / 1000.0,
                    0.5 + (rng() % 500) / 1000.0,
                    1.0
                };
                a.currentColor = a.meta.baseColor;
            }
        } else if (rq.type=="audio"){
            a.meta.channels = (rng()%2) ? 2 : 1;
            a.meta.duration = 3 + (rng()%60);
            a.meta.audioReactive = true; // Audio assets are inherently audio-reactive
        }

        // Simulate I/O
        std::this_thread::sleep_for(std::chrono::milliseconds(2 + rng() % 5));
        return a;
    }

    double estimateMB(const NexusAsset& a){
        double baseMB = 0.1;

        if (a.type=="models"){
            baseMB = (a.meta.vertices<=0 ? 5000 : a.meta.vertices) * 0.00008;
        } else if (a.type=="textures"){
            int w = (a.meta.width>0)? a.meta.width : 1024;
            int h = (a.meta.height>0)? a.meta.height : 1024;
            int bpp = a.meta.artReactive ? 8 : 4; // Art-reactive textures need more data
            baseMB = (double(w)*double(h)*bpp) / (1024.0*1024.0);
        } else if (a.type=="audio"){
            double dur = (a.meta.duration>0)? a.meta.duration : 5;
            int ch = (a.meta.channels>0)? a.meta.channels : 1;
            baseMB = (dur * 44100.0 * 2.0 * ch) / (1024.0*1024.0);
        } else if (a.type=="materials"){
            baseMB = a.meta.artReactive ? 0.1 : 0.05; // Art-reactive materials larger
        } else if (a.type=="shaders"){
            baseMB = (a.meta.audioReactive || a.meta.artReactive) ? 0.02 : 0.01;
        } else if (a.type=="config"){
            baseMB = 0.01;
        }

        return baseMB;
    }

    void touch(const std::string& type, const std::string& id){
        lastAccess[key(type,id)] = Clock::now();
    }

    void updateNexusEffects(NexusAsset& asset) {
        if (!gameEngine) return;

        const auto& params = gameEngine->GetParameters();

        // Update audio-reactive assets
        if (asset.meta.audioReactive) {
            // Simulate beat detection - in real implementation, this would come from audio analysis
            double currentTime = std::chrono::duration<double>(
                std::chrono::high_resolution_clock::now().time_since_epoch()
            ).count();

            double beatInterval = 60.0 / params.bpm;
            double timeSinceLastBeat = fmod(currentTime, beatInterval);

            if (timeSinceLastBeat < 0.1) { // Beat window
                asset.lastBeatTime = 0.0;
                asset.audioIntensity = 1.0 + asset.meta.harmonicResponse;
            } else {
                asset.lastBeatTime += gameEngine->GetDeltaTime();
                // Fade audio intensity
                asset.audioIntensity = std::max(1.0, asset.audioIntensity - gameEngine->GetDeltaTime() * 3.0);
            }
        }

        // Update art-reactive assets
        if (asset.meta.artReactive) {
            // Simulate art pattern sync - in real implementation, this would use ArtSync component
            double t = std::chrono::duration<double>(
                std::chrono::high_resolution_clock::now().time_since_epoch()
            ).count() * 0.5;

            // Create petal-based color variations
            double petalPhase = (2.0 * M_PI * sin(t)) / params.petalCount;
            double colorMod = 0.8 + 0.2 * sin(petalPhase + t);

            asset.currentColor = ArtSync::Color{
                asset.meta.baseColor.r * colorMod,
                asset.meta.baseColor.g * colorMod,
                asset.meta.baseColor.b * colorMod,
                asset.meta.baseColor.a
            };
        }
    }

    void evictIfNeeded(){
        if (mem.total <= memoryLimitMB) return;

        std::vector<std::pair<std::string, Clock::time_point>> items(lastAccess.begin(), lastAccess.end());
        std::sort(items.begin(), items.end(),
                  [](const auto& a, const auto& b){ return a.second < b.second; });

        for (const auto& kv : items){
            if (mem.total <= memoryLimitMB) break;
            const std::string& K = kv.first;
            auto pos = K.find(':');
            if (pos==std::string::npos) continue;
            std::string type = K.substr(0,pos);
            std::string id   = K.substr(pos+1);

            auto itType = store.find(type);
            if (itType==store.end()) continue;
            auto itAsset = itType->second.find(id);
            if (itAsset==itType->second.end()) continue;

            double sz = 0.0;
            if (sizesMB.count(K)) sz = sizesMB[K];

            itType->second.erase(itAsset);
            lastAccess.erase(K);
            sizesMB.erase(K);
            mem.total = std::max(0.0, mem.total - sz);
            std::cout << "[NEXUS-ARM] Evicted " << K << " (" << std::fixed << std::setprecision(2) << sz << " MB)\n";
        }
    }

    const NexusAsset& loadNow(const Request& rq){
        NexusAsset a = load(rq);
        double sz = estimateMB(a);

        auto& bucket = store[rq.type];
        auto ins = bucket.emplace(rq.id, std::move(a)).first;

        const std::string K = key(rq.type, rq.id);
        sizesMB[K] = sz;
        mem.total += sz;
        touch(rq.type, rq.id);

        std::cout << "[NEXUS-ARM] Loaded " << K << " (" << std::fixed << std::setprecision(2) << sz << " MB)"
                  << " total=" << mem.total << " MB";
        if (ins->second.meta.audioReactive) std::cout << " [AUDIO-REACTIVE]";
        if (ins->second.meta.artReactive) std::cout << " [ART-REACTIVE]";
        std::cout << "\n";

        evictIfNeeded();
        return ins->second;
    }
};

// ================================================================
// NEXUS-Enhanced World Simulation
// ================================================================
struct NexusProjectile {
    NexusVec3 pos{}, vel{};
    double life{3.0};
    double damage{25.0};
    std::shared_ptr<GameEntity> entity{nullptr}; // NEXUS entity integration
    bool audioReactive{false};
    double audioScale{1.0};
};

struct NexusEnemy {
    NexusVec3 pos{}, vel{};
    double hp{100.0}, maxHp{100.0};
    std::shared_ptr<GameEntity> entity{nullptr}; // NEXUS entity integration
    ArtSync::Color currentColor{1.0, 0.3, 0.3, 1.0}; // Default red
    double lastBeatReaction{0.0};
};

struct NexusCollidable {
    NexusVec3 pos{};
    double radius{10.0};
    std::shared_ptr<GameEntity> entity{nullptr}; // NEXUS entity integration
    std::string assetId;
    bool audioReactive{false};
    bool artReactive{false};
};

struct NexusPlayer {
    NexusVec3 pos{}, vel{};
    double hp{100.0}, maxHp{100.0};
    double mp{50.0}, maxMp{50.0};
    double xp{0.0}, maxXp{100.0};
    double sensitivity{1.0};
    bool invertY{false};
    std::shared_ptr<GameEntity> entity{nullptr}; // NEXUS entity integration

    // NEXUS-specific player properties
    double beatSyncLevel{1.0};      // How much player syncs with beats
    ArtSync::Color auraColor{0.2, 0.8, 1.0, 0.5}; // Player aura color
};

struct NexusCamera {
    NexusVec3 pos{};
    NexusVec3 target{};
};

class NexusWorldSim {
public:
    // Enhanced tunables with NEXUS integration
    double PLAYER_RADIUS = 1.0;
    double ENEMY_RADIUS  = 1.0;
    double MOVE_SPEED    = 120.0;
    double MANA_REGEN    = 8.0;
    double camDist       = 40.0;

    // NEXUS-specific settings
    bool enableAudioReactivity{true};
    bool enableArtSync{true};
    double beatInfluenceRadius{100.0}; // How far beat effects reach
    double artPatternScale{50.0};      // Scale of art pattern effects

    // State
    NexusPlayer player;
    NexusCamera camera;
    std::vector<NexusProjectile> projectiles;
    std::vector<NexusEnemy> enemies;
    std::vector<NexusCollidable> collidables;

    struct { double h=0.0, v=-0.3; } camAngle;
    std::optional<NexusVec3> merchant;

    // NEXUS integration
    NexusGameEngine* gameEngine{nullptr};
    NexusAssetResourceManager* assetManager{nullptr};

    NexusWorldSim(NexusGameEngine* engine = nullptr, NexusAssetResourceManager* assets = nullptr)
        : gameEngine(engine), assetManager(assets) {
        if (gameEngine) {
            // Create player entity
            player.entity = gameEngine->CreateEntity("Player");
            auto playerTransform = player.entity->AddComponent<Transform>();
            auto playerAudioSync = player.entity->AddComponent<AudioSync>();
            auto playerArtSync = player.entity->AddComponent<ArtSync>();

            playerAudioSync->SetSyncMode(AudioSync::SyncMode::BPM_PULSE);
            playerAudioSync->SetIntensity(0.8);
            playerArtSync->SetPatternMode(ArtSync::PatternMode::MANDALA_SYNC);
        }
    }

    double getHeightAt(double x, double z) const {
        // Enhanced height field with NEXUS art pattern influence
        double baseHeight = std::sin(0.01*x)*5.0 + std::cos(0.01*z)*5.0;

        if (gameEngine && enableArtSync) {
            const auto& params = gameEngine->GetParameters();
            double artInfluence = sin(x * 0.02 * params.petalCount) * cos(z * 0.02 * params.petalCount);
            baseHeight += artInfluence * params.terrainRoughness * 3.0;
        }

        return baseHeight;
    }

    void update(double dt, const std::set<int>& keys, double mouseDX, double mouseDY, bool focused){
        if (player.hp <= 0) return;

        // Update NEXUS systems
        if (gameEngine) {
            updateNexusEffects(dt);
        }

        // Enhanced movement with NEXUS audio sync
        NexusVec3 move(0,0,0);
        if (keys.count('W')) move.z -= 1;
        if (keys.count('S')) move.z += 1;
        if (keys.count('A')) move.x -= 1;
        if (keys.count('D')) move.x += 1;

        NexusVec3 targetVel(0,0,0);
        if (move.len() > 0.1){
            double effectiveSpeed = MOVE_SPEED;

            // Audio-reactive movement speed
            if (gameEngine && enableAudioReactivity && player.entity) {
                auto audioSync = player.entity->GetComponent<AudioSync>();
                if (audioSync && audioSync->IsOnBeat()) {
                    effectiveSpeed *= (1.0 + player.beatSyncLevel * 0.5); // 50% speed boost on beat
                }
            }

            targetVel = move.normalized() * effectiveSpeed;
            const double ch = std::cos(camAngle.h), sh = std::sin(camAngle.h);
            targetVel = { targetVel.x*ch + targetVel.z*sh,
                          targetVel.y,
                         -targetVel.x*sh + targetVel.z*ch };
        }

        player.vel = player.vel*0.75 + targetVel*0.25;
        player.pos += player.vel * dt;

        // Enhanced collision with NEXUS effects
        for (const auto& c : collidables){
            const NexusVec3 d = player.pos - c.pos;
            const double req = PLAYER_RADIUS + c.radius;
            const double d2  = d.len2();
            const double min2 = req*req;
            if (d2 < min2){
                const double dist = std::max(1e-9, std::sqrt(d2));
                const NexusVec3 n = d * (1.0 / dist);
                const double push = req - dist;
                player.pos += n * push;

                // Audio feedback on collision
                if (gameEngine && c.audioReactive && player.entity) {
                    auto audioSync = player.entity->GetComponent<AudioSync>();
                    if (audioSync) {
                        // Trigger audio effect on collision
                    }
                }
            }
        }

        // Enhanced ground sticking with art-influenced terrain
        player.pos.y = getHeightAt(player.pos.x, player.pos.z);

        // Update player entity transform
        if (player.entity) {
            auto transform = player.entity->GetComponent<Transform>();
            if (transform) {
                transform->SetPosition(player.pos.toVector3());
            }
        }

        // Camera input (unchanged)
        const double invert = player.invertY ? -1.0 : 1.0;
        camAngle.h -= mouseDX * player.sensitivity * 0.01;
        camAngle.v -= invert * mouseDY * player.sensitivity * 0.5 * 0.01;
        camAngle.v = std::max(-M_PI*0.4, std::min(-0.1, camAngle.v));

        NexusVec3 camOffset( std::sin(camAngle.h)*std::cos(camAngle.v),
                            -std::sin(camAngle.v),
                             std::cos(camAngle.h)*std::cos(camAngle.v));
        camera.pos = camera.pos*0.9 + (player.pos + camOffset*camDist)*0.1;
        camera.target = player.pos + NexusVec3{0,15,0};

        // Enhanced mana regen with audio sync
        double manaRegenRate = MANA_REGEN;
        if (gameEngine && enableAudioReactivity && player.entity) {
            auto audioSync = player.entity->GetComponent<AudioSync>();
            if (audioSync && audioSync->IsOnBeat()) {
                manaRegenRate *= 1.5; // 50% faster mana regen on beat
            }
        }
        player.mp = std::min(player.maxMp, player.mp + manaRegenRate*dt);

        // Enhanced projectiles with NEXUS effects
        updateProjectiles(dt);

        // Enhanced enemies with NEXUS behavior
        updateEnemies(dt);

        // Merchant check (unchanged)
        if (merchant.has_value()){
            const bool near = distance(player.pos, *merchant) < 120.0;
            (void)near;
        }

        // Update asset manager NEXUS effects
        if (assetManager) {
            assetManager->updateNexusEffects(dt);
        }
    }

    // Add enhanced projectile/enemy creation methods
    void createAudioReactiveProjectile(const NexusVec3& pos, const NexusVec3& vel, double damage = 25.0) {
        NexusProjectile p;
        p.pos = pos;
        p.vel = vel;
        p.damage = damage;
        p.audioReactive = true;
        p.audioScale = 1.0;

        if (gameEngine) {
            p.entity = gameEngine->CreateEntity("AudioProjectile");
            auto transform = p.entity->AddComponent<Transform>();
            auto audioSync = p.entity->AddComponent<AudioSync>();

            transform->SetPosition(pos.toVector3());
            audioSync->SetSyncMode(AudioSync::SyncMode::BPM_PULSE);
            audioSync->SetIntensity(1.5);
        }

        projectiles.push_back(std::move(p));
    }

    void createArtReactiveEnemy(const NexusVec3& pos, double hp = 100.0) {
        NexusEnemy e;
        e.pos = pos;
        e.hp = e.maxHp = hp;

        if (gameEngine) {
            e.entity = gameEngine->CreateEntity("ArtEnemy");
            auto transform = e.entity->AddComponent<Transform>();
            auto artSync = e.entity->AddComponent<ArtSync>();
            auto physics = e.entity->AddComponent<Physics>();

            transform->SetPosition(pos.toVector3());
            artSync->SetPatternMode(ArtSync::PatternMode::SPIRAL_MOTION);
            physics->SetMass(2.0);
            physics->SetUseGravity(false); // Flying enemies
        }

        enemies.push_back(std::move(e));
    }

private:
    void updateNexusEffects(double dt) {
        const auto& params = gameEngine->GetParameters();

        // Update player NEXUS components
        if (player.entity) {
            auto audioSync = player.entity->GetComponent<AudioSync>();
            auto artSync = player.entity->GetComponent<ArtSync>();

            if (audioSync) {
                audioSync->UpdateAudioData(params.bpm, params.harmonics, 1.0, 440.0);
            }

            if (artSync) {
                artSync->UpdateArtData(params.petalCount, params.terrainRoughness);
                player.auraColor = artSync->GetCurrentColor();
            }
        }
    }

    void updateProjectiles(double dt) {
        for (int i=int(projectiles.size())-1; i>=0; --i){
            auto& p = projectiles[i];

            // Audio-reactive projectile effects
            if (p.audioReactive && gameEngine && p.entity) {
                auto audioSync = p.entity->GetComponent<AudioSync>();
                if (audioSync && audioSync->IsOnBeat()) {
                    p.audioScale = 1.0 + audioSync->GetIntensity() * 0.3;
                    p.damage *= 1.2; // 20% damage boost on beat
                } else {
                    p.audioScale = std::max(1.0, p.audioScale - dt * 2.0);
                }
            }

            p.pos += p.vel * dt;
            p.life -= dt;
            if (p.life <= 0){
                eraseProjectile(i);
                continue;
            }

            // Update entity transform
            if (p.entity) {
                auto transform = p.entity->GetComponent<Transform>();
                if (transform) {
                    transform->SetPosition(p.pos.toVector3());
                    if (p.audioReactive) {
                        transform->SetScale(Transform::Vector3(p.audioScale, p.audioScale, p.audioScale));
                    }
                }
            }

            // Enhanced enemy hits with NEXUS effects
            for (int j=int(enemies.size())-1; j>=0; --j){
                auto& e = enemies[j];
                if (distance(p.pos, e.pos) < ENEMY_RADIUS){
                    double actualDamage = p.damage;

                    // Art-reactive damage bonus
                    if (gameEngine && e.entity) {
                        auto artSync = e.entity->GetComponent<ArtSync>();
                        if (artSync) {
                            // More damage if enemy is in art pattern
                            const auto& params = gameEngine->GetParameters();
                            double patternBonus = sin(e.pos.x * 0.1 * params.petalCount) *
                                                cos(e.pos.z * 0.1 * params.petalCount);
                            if (patternBonus > 0.5) {
                                actualDamage *= 1.5;
                            }
                        }
                    }

                    e.hp -= actualDamage;
                    eraseProjectile(i);
                    if (e.hp <= 0) eraseEnemy(j);
                    break;
                }
            }
        }
    }

    void updateEnemies(double dt) {
        for (auto& e : enemies){
            const double d = distance(player.pos, e.pos);

            // Enhanced AI with NEXUS behavior
            if (d < 400){
                NexusVec3 toPlayer = (player.pos - e.pos).normalized();
                double seekForce = 80.0;

                // Audio-reactive enemy behavior
                if (gameEngine && e.entity && enableAudioReactivity) {
                    auto audioSync = e.entity->GetComponent<AudioSync>();
                    if (!audioSync) {
                        audioSync = e.entity->AddComponent<AudioSync>();
                        audioSync->SetSyncMode(AudioSync::SyncMode::BPM_PULSE);
                    }

                    if (audioSync && audioSync->IsOnBeat()) {
                        seekForce *= 2.0; // Double seek force on beat
                        e.lastBeatReaction = 0.0;
                    } else {
                        e.lastBeatReaction += dt;
                    }
                }

                e.vel += toPlayer * (seekForce * dt);
            }

            e.vel *= 0.95;
            e.pos += e.vel * dt;
            e.pos.y = getHeightAt(e.pos.x, e.pos.z);

            // Update entity transform
            if (e.entity) {
                auto transform = e.entity->GetComponent<Transform>();
                if (transform) {
                    transform->SetPosition(e.pos.toVector3());
                }

                // Update art sync color
                auto artSync = e.entity->GetComponent<ArtSync>();
                if (artSync) {
                    e.currentColor = artSync->GetCurrentColor();
                }
            }

            // Enhanced damage with NEXUS effects
            if (d < (PLAYER_RADIUS + ENEMY_RADIUS)){
                double damage = 15.0 * dt;

                // Reduce damage if player is synchronized with beat
                if (gameEngine && player.entity && enableAudioReactivity) {
                    auto audioSync = player.entity->GetComponent<AudioSync>();
                    if (audioSync && audioSync->IsOnBeat()) {
                        damage *= 0.5; // 50% damage reduction when on beat
                    }
                }

                player.hp -= damage;
            }
        }
    }

    void eraseProjectile(int i){
        // Clean up NEXUS entity
        if (projectiles[i].entity && gameEngine) {
            gameEngine->DestroyEntity(projectiles[i].entity);
        }
        projectiles[i] = projectiles.back();
        projectiles.pop_back();
    }

    void eraseEnemy(int j){
        // Clean up NEXUS entity
        if (enemies[j].entity && gameEngine) {
            gameEngine->DestroyEntity(enemies[j].entity);
        }
        enemies[j] = enemies.back();
        enemies.pop_back();
    }
};

// ================================================================
// NEXUS-Enhanced Tests & Demo Harness
// ================================================================
static void test_nexus_zero_distance_collision() {
    std::cout << "🧪 Testing NEXUS zero-distance collision handling...\n";

    NexusGameEngine engine;
    NexusGameEngine::SystemParameters params;
    params.bpm = 120;
    params.harmonics = 4;
    params.petalCount = 6;
    engine.Initialize(params);

    NexusAssetResourceManager assetManager(&engine);
    NexusWorldSim sim(&engine, &assetManager);

    sim.PLAYER_RADIUS = 1.0;
    sim.player.pos = {0,0,0};
    sim.collidables.push_back({{0,0,0}, 10.0});
    sim.update(0.016, {}, 0.0, 0.0, true);

    assert(std::isfinite(sim.player.pos.x));
    assert(std::isfinite(sim.player.pos.y));
    assert(std::isfinite(sim.player.pos.z));

    std::cout << "✅ NEXUS collision test passed\n";
    engine.Shutdown();
}

static void test_nexus_asset_cache_and_eviction() {
    std::cout << "🧪 Testing NEXUS asset cache with audio/art reactivity...\n";

    NexusGameEngine engine;
    NexusGameEngine::SystemParameters params;
    engine.Initialize(params);

    NexusAssetResourceManager arm(&engine);
    arm.setMemoryLimitMB(2.0); // Tiny limit to force evictions

    const NexusAsset& t1 = arm.request("textures","grass01.png", 3, false, true); // Art-reactive
    const NexusAsset& t2 = arm.request("textures","dirt01.png",  2);
    const NexusAsset& m1 = arm.request("models",  "tree01.glb",  1, true, false); // Audio-reactive

    const NexusAsset& t1b = arm.request("textures","grass01.png", 5);
    assert(&t1 == &t1b);
    assert(t1.meta.artReactive == true);
    assert(m1.meta.audioReactive == true);

    for (int i=0;i<4;i++){
        arm.request("textures", "tex_"+std::to_string(i)+".png", i, i%2==0, i%2==1);
    }

    assert(arm.totalMB() <= 2.0001);

    std::cout << "✅ NEXUS asset test passed\n";
    engine.Shutdown();
}

int main(){
    std::cout << "🎵✨ NEXUS World Simulation Demo Starting... ✨🎵\n\n";

    std::cout << "🧪 Running NEXUS-enhanced tests...\n";
    test_nexus_zero_distance_collision();
    test_nexus_asset_cache_and_eviction();
    std::cout << "✅ All NEXUS tests passed!\n\n";

    // Main NEXUS demo
    std::cout << "🌍 Initializing NEXUS World Simulation...\n";

    NexusGameEngine gameEngine;
    NexusGameEngine::SystemParameters params;
    params.bpm = 128.0;           // Energetic BPM for combat
    params.harmonics = 6;         // Rich harmonic content
    params.petalCount = 8;        // Octagonal art patterns
    params.terrainRoughness = 0.7; // Varied terrain

    if (!gameEngine.Initialize(params)) {
        std::cerr << "❌ Failed to initialize NEXUS Game Engine!\n";
        return -1;
    }

    std::cout << "✅ NEXUS Game Engine initialized\n";
    std::cout << "   🎵 BPM: " << params.bpm << "\n";
    std::cout << "   🎶 Harmonics: " << params.harmonics << "\n";
    std::cout << "   🌸 Art Petals: " << params.petalCount << "\n";
    std::cout << "   🏔️  Terrain Roughness: " << params.terrainRoughness << "\n\n";

    // Create enhanced asset manager
    NexusAssetResourceManager assetManager(&gameEngine);
    assetManager.setMemoryLimitMB(128.0);

    // Create NEXUS world simulation
    NexusWorldSim sim(&gameEngine, &assetManager);
    sim.enableAudioReactivity = true;
    sim.enableArtSync = true;

    // Setup world
    sim.player.pos = {120, 0, 120};
    sim.merchant = NexusVec3{150,0,150};

    std::cout << "🏗️  Setting up NEXUS world...\n";

    // Create art-reactive enemies with varied patterns
    for (int i=0;i<5;i++){
        NexusVec3 pos{200 + i*50.0, 0, 200+i*30.0};
        sim.createArtReactiveEnemy(pos, 100.0 + i*20.0);
    }

    // Create NEXUS-enhanced collidables with asset integration
    sim.collidables.push_back({NexusVec3{130,0,130}, 8.0, nullptr, "pillar_crystal.glb", false, true});
    sim.collidables.push_back({NexusVec3{180,0,160}, 12.0, nullptr, "pillar_metal.glb", true, false});

    // Pre-load some NEXUS assets
    std::cout << "📦 Loading NEXUS assets...\n";
    assetManager.request("models", "player_model.glb", 10, true, true);  // Player model
    assetManager.request("models", "enemy_spider.glb", 8, false, true);  // Art-reactive enemy
    assetManager.request("models", "projectile_orb.glb", 7, true, false); // Audio-reactive projectile
    assetManager.request("textures", "terrain_grass.png", 6, false, true); // Art-reactive terrain
    assetManager.request("audio", "ambient_nexus.ogg", 5, true, false);   // Ambient audio
    assetManager.request("materials", "crystal_shader.mat", 4, false, true); // Art-reactive material

    std::cout << "📊 Initial asset memory: " << std::fixed << std::setprecision(2)
              << assetManager.totalMB() << " MB\n\n";

    // Simulate NEXUS system data
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
            "harmonics": 6,
            "petalCount": 8,
            "terrainRoughness": 0.7
        }
    })";

    // Enhanced simulation loop
    std::set<int> keys = {'W','D'};
    double mouseDX=0.3, mouseDY=0.1;

    std::cout << "🎮 Starting NEXUS World Simulation...\n";
    std::cout << "🎯 Player controls: WASD movement with audio/art sync bonuses\n";
    std::cout << "🎵 Audio-reactive enemies seek faster on beats\n";
    std::cout << "🎨 Art-reactive terrain and enemy colors change with patterns\n\n";

    for (int f=0; f<360; ++f){ // 6 seconds at 60fps
        // Update NEXUS systems
        gameEngine.SyncWithNexusSystems(nexusSystemData);
        gameEngine.Update();

        // Update world simulation
        sim.update(1.0/60.0, keys, mouseDX, mouseDY, true);

        // Create some audio-reactive projectiles periodically
        if (f % 40 == 0 && f > 60) { // Every ~0.67 seconds after initial period
            NexusVec3 projectileVel{50 + (f%3)*20, 0, 50 + (f%2)*30};
            sim.createAudioReactiveProjectile(sim.player.pos + NexusVec3{0, 5, 0}, projectileVel, 30.0);
        }

        // Dynamic parameter changes for demo
        if (f % 90 == 0) { // Every 1.5 seconds
            double newBPM = 120.0 + 20.0 * sin(f * 0.02);
            int newPetals = 6 + (int)(2 * sin(f * 0.03));
            double newRoughness = 0.5 + 0.3 * sin(f * 0.01);

            gameEngine.SetBPM(newBPM);
            gameEngine.SetPetalCount(newPetals);
            gameEngine.SetTerrainRoughness(newRoughness);
        }

        // Status logging
        if (f%60==0 && f > 0){
            double t = f/60.0;
            std::cout << "⏱️  t=" << std::fixed << std::setprecision(1) << t << "s"
                      << "  player=(" << std::setprecision(1)
                      << sim.player.pos.x << "," << sim.player.pos.y << "," << sim.player.pos.z << ")"
                      << "  hp=" << std::setprecision(0) << sim.player.hp
                      << "  mp=" << sim.player.mp
                      << "  enemies=" << sim.enemies.size()
                      << "  projectiles=" << sim.projectiles.size() << "\n";

            if (t == 2.0) {
                const auto& params = gameEngine.GetParameters();
                std::cout << "🎵 Current BPM: " << params.bpm
                          << ", Petals: " << params.petalCount
                          << ", Terrain: " << params.terrainRoughness << "\n";
            }
        }
    }

    // Final status
    std::cout << "\n📊 Final NEXUS World Simulation Status:\n";
    std::cout << "======================================\n";
    std::cout << "🎯 Player Final Position: (" << std::fixed << std::setprecision(1)
              << sim.player.pos.x << ", " << sim.player.pos.y << ", " << sim.player.pos.z << ")\n";
    std::cout << "❤️  Player Health: " << sim.player.hp << "/" << sim.player.maxHp << "\n";
    std::cout << "💙 Player Mana: " << sim.player.mp << "/" << sim.player.maxMp << "\n";
    std::cout << "👹 Enemies Remaining: " << sim.enemies.size() << "\n";
    std::cout << "💥 Active Projectiles: " << sim.projectiles.size() << "\n";
    std::cout << "📦 Asset Memory Usage: " << std::setprecision(2) << assetManager.totalMB() << " MB\n";

    const auto& finalParams = gameEngine.GetParameters();
    std::cout << "🎵 Final NEXUS Parameters:\n";
    std::cout << "   BPM: " << finalParams.bpm << "\n";
    std::cout << "   Harmonics: " << finalParams.harmonics << "\n";
    std::cout << "   Petals: " << finalParams.petalCount << "\n";
    std::cout << "   Terrain Roughness: " << finalParams.terrainRoughness << "\n";

    // Shutdown
    std::cout << "\n🔄 Shutting down NEXUS systems...\n";
    gameEngine.Shutdown();

    std::cout << "\n🎵✨ NEXUS World Simulation completed successfully! ✨🎵\n";
    std::cout << "\n💡 Features Demonstrated:\n";
    std::cout << "   ✅ Audio-reactive player movement speed boost on beats\n";
    std::cout << "   ✅ Audio-reactive enemy AI with beat-synchronized seeking\n";
    std::cout << "   ✅ Audio-reactive projectiles with beat-synchronized scaling\n";
    std::cout << "   ✅ Art-reactive terrain height modulation\n";
    std::cout << "   ✅ Art-reactive enemy colors following petal patterns\n";
    std::cout << "   ✅ Asset management with audio/art-reactive metadata\n";
    std::cout << "   ✅ NEXUS Holy Beat System parameter synchronization\n";
    std::cout << "   ✅ Entity-component integration with game objects\n";
    std::cout << "   ✅ Real-time parameter modulation during gameplay\n";
    std::cout << "   ✅ Memory-efficient asset caching with LRU eviction\n\n";

    return 0;
}
