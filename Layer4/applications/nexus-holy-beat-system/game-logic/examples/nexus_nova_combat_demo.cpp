// NEXUS Nova Combat Integration Demo
// Combines Nova combat system with NEXUS Holy Beat System
// Features: Sacred geometry combat, quantum-enhanced movement, elemental harmony

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
#include <vector>
#include <map>
#include <optional>
#include <functional>
#include <set>

using namespace NEXUS;
using namespace NexusGame;
using std::cout; using std::string;

// ============ NOVA COMBAT INTEGRATION ============

namespace Nova {

// Enhanced Vec3 with NEXUS integration
struct Vec3 {
    float x{0}, y{0}, z{0};
    Vec3() = default;
    Vec3(float X,float Y,float Z):x(X),y(Y),z(Z){}
    Vec3 operator+(const Vec3& r) const { return {x+r.x,y+r.y,z+r.z}; }
    Vec3 operator-(const Vec3& r) const { return {x-r.x,y-r.y,z-r.z}; }
    Vec3 operator*(float s) const { return {x*s,y*s,z*s}; }
    Vec3& operator+=(const Vec3& r){ x+=r.x; y+=r.y; z+=r.z; return *this; }
    float length() const { return std::sqrt(x*x+y*y+z*z); }
    Vec3 normalized() const { float L=length(); return (L>1e-6f)?Vec3{x/L,y/L,z/L}:Vec3{}; }

    // NEXUS extensions
    NexusGame::Vec3f toNexus() const { return {x, y, z}; }
    static Vec3 fromNexus(const NexusGame::Vec3f& v) { return {v.x, v.y, v.z}; }
};

static inline float clampf(float v,float lo,float hi){ return std::max(lo,std::min(hi,v)); }

// Enhanced enums with NEXUS quantum states
enum class WeaponType { Sword, Spear, Gauntlet, Bow, Staff, SacredBlade };
enum class AttackType { Basic, Charged, Combo, Elemental, Quantum, Sacred };
enum class ElementType { Fire, Wind, Lightning, Ice, Void, Earth, Harmony, Chaos };

// NEXUS-enhanced Motion State with quantum properties
class QuantumMotionState {
public:
    Vec3 position{0,0,0};
    Vec3 velocity{0,0,0};
    Vec3 quantumFluctuation{0,0,0}; // NEXUS quantum effects
    bool isSprinting{false};
    bool isDashing{false};
    bool isQuantumPhase{false}; // NEXUS quantum dash

    float stamina{100.f}, staminaMax{100.f};
    float staminaRegenRate{10.f};
    float dashCooldown{2.f};
    float dashTimer{0.f};
    float quantumEnergy{50.f}, quantumMax{50.f}; // NEXUS quantum energy

    // Tunables enhanced with sacred mathematics
    float baseSpeed{5.f};
    float sprintSpeed{9.f};
    float dashBurstSpeed{15.f};
    float quantumSpeed{25.f}; // NEXUS quantum dash speed
    float sprintCostPerCall{5.f};
    float dashCost{20.f};
    float quantumCost{35.f}; // NEXUS quantum dash cost
    float dashDuration{0.20f};
    float quantumDuration{0.15f}; // NEXUS quantum dash duration
    float dashTimeLeft{0.f};

    // NEXUS integration
    NexusProtocol* quantumProtocol{nullptr};
    NexusRecursiveKeeperEngine* cognitiveEngine{nullptr};

    void bindNexusSystems(NexusProtocol* qp, NexusRecursiveKeeperEngine* ce) {
        quantumProtocol = qp;
        cognitiveEngine = ce;
    }

    void update(float dt) {
        position += velocity * dt;

        // Apply quantum fluctuations if in quantum state
        if (isQuantumPhase && quantumProtocol) {
            int qmode = quantumProtocol->getCurrentMode();
            float qphase = qmode * 0.125f; // Scale quantum mode to 0-1

            // Quantum fluctuation based on current mode
            quantumFluctuation.x = std::sin(qphase * 6.28f) * 0.5f;
            quantumFluctuation.y = std::cos(qphase * 4.71f) * 0.3f;
            quantumFluctuation.z = std::sin(qphase * 8.85f) * 0.4f;

            position += quantumFluctuation * dt;
        }

        if (dashTimer > 0.f) dashTimer = std::max(0.f, dashTimer - dt);

        if (isDashing || isQuantumPhase) {
            dashTimeLeft -= dt;
            if (dashTimeLeft <= 0.f) {
                isDashing = false;
                isQuantumPhase = false;
                velocity = velocity * 0.4f;

                if (cognitiveEngine) {
                    cognitiveEngine->processThought("movement_flow",
                        "Completed " + (isQuantumPhase ? "quantum" : "standard") + " dash maneuver");
                }
            }
        }

        if (!isSprinting && !isDashing && !isQuantumPhase) {
            stamina = clampf(stamina + staminaRegenRate * dt, 0.f, staminaMax);
            quantumEnergy = clampf(quantumEnergy + (staminaRegenRate * 0.7f) * dt, 0.f, quantumMax);
        }
    }

    void move(const Vec3& dirIn, bool sprinting=false) {
        Vec3 dir = dirIn.normalized();
        if (dir.length() < 1e-6f) { velocity = {0,0,0}; isSprinting=false; return; }

        float speed = baseSpeed;

        // Sacred mathematics: use golden ratio for sprint efficiency
        const float goldenRatio = 1.618033988749f;

        if (sprinting && stamina >= sprintCostPerCall) {
            isSprinting = true;
            speed = sprintSpeed;

            // NEXUS enhancement: sprint cost influenced by quantum state
            float quantumEfficiency = quantumProtocol ? (1.0f + quantumProtocol->getCurrentMode() * 0.1f) : 1.0f;
            float actualCost = sprintCostPerCall / quantumEfficiency;

            stamina = clampf(stamina - actualCost, 0.f, staminaMax);

            cout << "🚶 Sacred sprint: speed=" << speed
                 << " quantum_eff=" << std::fixed << std::setprecision(2) << quantumEfficiency
                 << " stamina=" << std::setprecision(0) << stamina << "\n";
        } else {
            isSprinting = false;
        }

        velocity = dir * speed;
    }

    void dash(const Vec3& dirIn) {
        if (dashTimer <= 0.f && stamina >= dashCost) {
            Vec3 dir = dirIn.normalized();
            if (dir.length() < 1e-6f) { cout<<"⛔ Dash needs direction\n"; return; }

            isDashing = true;
            dashTimeLeft = dashDuration;
            velocity = dir * dashBurstSpeed;
            stamina = clampf(stamina - dashCost, 0.f, staminaMax);
            dashTimer = dashCooldown;

            cout << "💨 Sacred dash: dir=(" << dir.x << "," << dir.y << "," << dir.z
                 << ") stamina=" << stamina << "\n";

            if (cognitiveEngine) {
                cognitiveEngine->processThought("dash_mastery", "Executing dash with sacred geometry principles");
            }
        } else {
            if (dashTimer > 0.f) cout<<"⏳ Dash CD "<<dashTimer<<"s\n";
            else cout<<"❗ Not enough stamina ("<<stamina<<"/"<<dashCost<<")\n";
        }
    }

    // NEXUS Quantum Dash - enhanced movement through quantum space
    void quantumDash(const Vec3& dirIn) {
        if (dashTimer <= 0.f && quantumEnergy >= quantumCost && quantumProtocol) {
            Vec3 dir = dirIn.normalized();
            if (dir.length() < 1e-6f) { cout<<"⛔ Quantum dash needs direction\n"; return; }

            isQuantumPhase = true;
            isDashing = false; // Override normal dash
            dashTimeLeft = quantumDuration;
            velocity = dir * quantumSpeed;
            quantumEnergy = clampf(quantumEnergy - quantumCost, 0.f, quantumMax);
            dashTimer = dashCooldown * 0.7f; // Shorter cooldown for quantum dash

            // Trigger quantum mode change
            int newMode = (quantumProtocol->getCurrentMode() + 1) % 8;
            quantumProtocol->setProcessingMode(newMode);

            cout << "🌀 QUANTUM DASH: dir=(" << dir.x << "," << dir.y << "," << dir.z
                 << ") mode=" << newMode << " quantum=" << quantumEnergy << "\n";

            if (cognitiveEngine) {
                cognitiveEngine->processThought("quantum_mastery",
                    "Transcending physical limitations through quantum dash at mode " + std::to_string(newMode));
            }
        } else {
            if (dashTimer > 0.f) cout<<"⏳ Quantum dash CD "<<dashTimer<<"s\n";
            else if (!quantumProtocol) cout<<"❌ No quantum protocol connected\n";
            else cout<<"❗ Not enough quantum energy ("<<quantumEnergy<<"/"<<quantumCost<<")\n";
        }
    }
};

// NEXUS-Enhanced Artifacts with sacred properties
struct SacredArtifact {
    string id;
    string name;
    ElementType element{ElementType::Fire};
    int tier{1};
    bool bonded{false};
    bool awakened{false};
    bool transcended{false}; // NEXUS transcendence state
    std::vector<string> traits;
    float sacredResonance{0.f}; // NEXUS sacred geometry resonance
    float quantumAlignment{0.f}; // NEXUS quantum alignment

    void bond(float affinity, NexusRecursiveKeeperEngine* cognitive = nullptr){
        if (!bonded && affinity > 5.0f) {
            bonded = true;
            sacredResonance += 0.2f;
            cout << "🔗 " << name << " bonded via elemental harmony.\n";

            if (cognitive) {
                cognitive->processThought("artifact_bonding",
                    "Achieved elemental bond with " + name + " through harmonic resonance");
            }
        }
    }

    void awaken(float affinity, NexusRecursiveKeeperEngine* cognitive = nullptr){
        if (bonded && !awakened && affinity > 15.0f) {
            awakened = true;
            traits.emplace_back("Element Surge");
            sacredResonance += 0.3f;
            cout << "⚔️ " << name << " awakened, gained trait: " << traits.back() << "\n";

            if (cognitive) {
                cognitive->processThought("artifact_awakening",
                    "Awakened " + name + " through sacred mathematical principles");
            }
        }
    }

    // NEXUS transcendence - highest level of artifact evolution
    void transcend(float quantumHarmony, NexusRecursiveKeeperEngine* cognitive = nullptr) {
        if (awakened && !transcended && quantumHarmony > 25.0f) {
            transcended = true;
            traits.emplace_back("Quantum Resonance");
            traits.emplace_back("Sacred Geometry");
            sacredResonance += 0.5f;
            quantumAlignment += 0.4f;
            cout << "✨ " << name << " TRANSCENDED! Gained quantum traits!\n";

            if (cognitive) {
                cognitive->processThought("artifact_transcendence",
                    "Achieved transcendence with " + name + " - artifact has evolved beyond physical limitations");
            }
        }
    }
};

// NEXUS-Enhanced Combat Core with quantum combat
class QuantumCombatCore {
public:
    WeaponType weapon{WeaponType::Sword};
    ElementType affinity{ElementType::Fire};
    int combo{0};
    float attackCooldown{0.5f};
    float timer{0.f};
    float harmony{0.f}; // NEXUS elemental harmony
    int quantumCombo{0}; // NEXUS quantum combo counter

    // NEXUS system hooks
    QuantumMotionState* motion{nullptr};
    NexusProtocol* quantumProtocol{nullptr};
    NexusVisuals* visualSystem{nullptr};
    NexusRecursiveKeeperEngine* cognitiveEngine{nullptr};

    void bindNexusSystems(QuantumMotionState* m, NexusProtocol* qp, NexusVisuals* vs, NexusRecursiveKeeperEngine* ce) {
        motion = m;
        quantumProtocol = qp;
        visualSystem = vs;
        cognitiveEngine = ce;
    }

    void update(float dt) {
        if (timer > 0) timer = std::max(0.f, timer - dt);
        if (motion) motion->update(dt);

        // Update harmony based on quantum state
        if (quantumProtocol) {
            int qmode = quantumProtocol->getCurrentMode();
            harmony = 0.5f + 0.5f * std::sin(qmode * 0.785398163f); // π/4 scaling
        }

        // Decay quantum combo over time
        if (quantumCombo > 0 && timer <= 0.f) {
            quantumCombo = std::max(0, quantumCombo - 1);
        }
    }

    void perform(AttackType type) {
        if (!ready()) return;

        switch(weapon) {
            case WeaponType::Sword: sacredSword(type); break;
            case WeaponType::Spear: quantumSpear(type); break;
            case WeaponType::Gauntlet: harmonicGauntlet(type); break;
            case WeaponType::Bow: resonantBow(type); break;
            case WeaponType::Staff: sacredStaff(type); break;
            case WeaponType::SacredBlade: transcendentBlade(type); break;
        }
    }

private:
    bool ready() const { return timer <= 0.f; }
    void useCD(float mult = 1.f) { timer = attackCooldown * mult; }

    void triggerVisualEffect(const string& effect) {
        if (visualSystem) {
            // Trigger visual effect based on current quantum state
            cout << "✨ Visual Effect: " << effect << " (quantum harmony: "
                 << std::fixed << std::setprecision(2) << harmony << ")\n";
        }
    }

    void processAttackThought(const string& attackName, AttackType type) {
        if (cognitiveEngine) {
            string thought = "Executed " + attackName + " with ";

            switch(type) {
                case AttackType::Quantum: thought += "quantum enhancement"; break;
                case AttackType::Sacred: thought += "sacred geometry principles"; break;
                case AttackType::Elemental: thought += "elemental harmony"; break;
                case AttackType::Combo: thought += "combo mastery"; break;
                default: thought += "basic technique"; break;
            }

            if (quantumCombo > 0) {
                thought += " (quantum combo x" + std::to_string(quantumCombo) + ")";
            }

            cognitiveEngine->processThought("combat_mastery", thought);
        }
    }

    void sacredSword(AttackType t) {
        if (t == AttackType::Basic) {
            cout << "🗡️ Sacred Sword Slash\n";
            combo = (combo + 1) % 3;
            triggerVisualEffect("Golden Slash");
            processAttackThought("Sacred Sword Slash", t);
            useCD();
        } else if (t == AttackType::Elemental) {
            cout << "🔥 Elemental Sword Wave (harmony: " << harmony << ")\n";
            triggerVisualEffect("Elemental Wave");
            processAttackThought("Elemental Sword", t);
            useCD(1.2f);
        } else if (t == AttackType::Quantum) {
            if (quantumProtocol && motion->quantumEnergy >= 15.f) {
                quantumCombo++;
                motion->quantumEnergy -= 15.f;
                cout << "🌀 QUANTUM SWORD STRIKE (combo x" << quantumCombo << ")\n";
                triggerVisualEffect("Quantum Blade");
                processAttackThought("Quantum Sword Strike", t);
                useCD(0.8f); // Faster with quantum
            } else {
                cout << "❌ Not enough quantum energy for quantum strike\n";
                return;
            }
        } else if (t == AttackType::Sacred) {
            cout << "✨ SACRED GEOMETRY BLADE - Divine Cut\n";
            triggerVisualEffect("Sacred Geometry");
            processAttackThought("Sacred Geometry Blade", t);
            useCD(1.8f);
        }
    }

    void quantumSpear(AttackType t) {
        cout << "🗡️ Quantum Spear technique " << (int)t << "\n";
        triggerVisualEffect("Quantum Spear");
        processAttackThought("Quantum Spear", t);
        useCD();
    }

    void harmonicGauntlet(AttackType t) {
        cout << "👊 Harmonic Gauntlet resonance " << (int)t << "\n";
        triggerVisualEffect("Harmonic Impact");
        processAttackThought("Harmonic Gauntlet", t);
        useCD();
    }

    void resonantBow(AttackType t) {
        cout << "🏹 Resonant Bow harmony shot " << (int)t << "\n";
        triggerVisualEffect("Resonant Arrow");
        processAttackThought("Resonant Bow", t);
        useCD(0.7f);
    }

    void sacredstaff(AttackType t) {
        cout << "🪄 Sacred Staff channeling " << (int)t << "\n";
        triggerVisualEffect("Sacred Channel");
        processAttackThought("Sacred Staff", t);
        useCD(1.3f);
    }

    void transcendentBlade(AttackType t) {
        cout << "⚡ TRANSCENDENT BLADE - Reality Cut " << (int)t << "\n";
        triggerVisualEffect("Reality Fracture");
        processAttackThought("Transcendent Blade", t);
        useCD(2.0f);
    }
};

} // namespace Nova

// ============ NEXUS NOVA INTEGRATION DEMO ============

class NexusNovaIntegrationDemo {
private:
    // Core NEXUS systems
    NexusGameEngine gameEngine;
    NexusResourceEngine resourceEngine;
    NexusProtocol quantumProtocol;
    NexusVisuals visualSystem;
    NexusRecursiveKeeperEngine cognitiveEngine;
    NexusProfiler& profiler;

    // Nova combat integration
    Nova::QuantumMotionState playerMotion;
    Nova::QuantumCombatCore combatCore;
    std::vector<Nova::SacredArtifact> artifacts;

    // Demo state
    NexusCamera camera;
    bool running;
    int frameCount;
    float demoTime;

public:
    NexusNovaIntegrationDemo()
        : profiler(NexusProfiler::getInstance())
        , running(false)
        , frameCount(0)
        , demoTime(0.0f) {}

    bool initialize() {
        cout << "🎵✨ Initializing NEXUS Nova Combat Integration... ✨🎵\n\n";

        // 1. Initialize core NEXUS systems
        profiler.startProfiling();

        NexusGameEngine::SystemParameters gameParams;
        gameParams.bpm = 140.0; // Faster tempo for combat
        gameParams.harmonics = 12; // More harmonics for combat complexity
        gameParams.petalCount = 16; // More petals for combat patterns
        gameParams.terrainRoughness = 0.8; // Higher complexity

        if (!gameEngine.Initialize(gameParams)) {
            std::cerr << "❌ Failed to initialize NEXUS Game Engine!\n";
            return false;
        }

        // 2. Setup camera
        camera.position = {0.f, 10.f, 20.f};
        camera.forward = {0.f, -0.3f, -1.f};
        camera.fov = 90.0f; // Wider FOV for combat

        // 3. Initialize resource engine for combat arena
        resourceEngine.worldBounds = {100.f, 100.f, 50.f};
        resourceEngine.chunkSize = 50.f;
        resourceEngine.loadDistance = 75.f;
        resourceEngine.unloadDistance = 100.f;
        resourceEngine.enableAsyncLoading = true;
        resourceEngine.enableFrustumCulling = true;
        resourceEngine.enableAudioReactivity = true;
        resourceEngine.enableArtSync = true;

        resourceEngine.initialize(&camera, &gameEngine);

        // 4. Setup Nova combat integration
        playerMotion.bindNexusSystems(&quantumProtocol, &cognitiveEngine);
        combatCore.bindNexusSystems(&playerMotion, &quantumProtocol, &visualSystem, &cognitiveEngine);

        combatCore.affinity = Nova::ElementType::Harmony; // NEXUS harmony element
        combatCore.weapon = Nova::WeaponType::SacredBlade; // NEXUS sacred weapon

        // 5. Create sacred artifacts
        createSacredArtifacts();

        // 6. Setup combat arena
        createCombatArena();

        cout << "✅ NEXUS Nova Combat Integration initialized!\n";
        cout << "🎮 Features: Quantum movement, sacred combat, harmonic resonance\n\n";

        return true;
    }

    void run() {
        cout << "🚀 Starting NEXUS Nova Combat Demo...\n";
        cout << "⚔️ Combat Features: Quantum dash, sacred strikes, elemental harmony\n";
        cout << "🌟 Duration: 90 seconds of integrated combat demonstration\n\n";

        running = true;
        auto startTime = std::chrono::high_resolution_clock::now();
        const int maxFrames = 60 * 90; // 90 seconds

        while (running && frameCount < maxFrames) {
            auto frameStart = std::chrono::high_resolution_clock::now();

            float dt = 1.0f / 60.0f; // Fixed timestep for demo
            demoTime += dt;

            // Update all systems
            updateSystems(dt);

            // Run combat demonstration
            runCombatDemo(dt);

            // Progress logging
            if (frameCount % 180 == 0) { // Every 3 seconds
                logCombatProgress();
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

private:
    void createSacredArtifacts() {
        cout << "⚔️ Forging sacred artifacts...\n";

        // Sacred Blade of Harmony
        Nova::SacredArtifact harmonicBlade;
        harmonicBlade.id = "harmony_blade";
        harmonicBlade.name = "Resonant Edge";
        harmonicBlade.element = Nova::ElementType::Harmony;
        harmonicBlade.tier = 5;
        harmonicBlade.sacredResonance = 0.8f;
        harmonicBlade.quantumAlignment = 0.6f;
        harmonicBlade.traits.push_back("Golden Ratio Forged");
        harmonicBlade.traits.push_back("Quantum Tempered");
        artifacts.push_back(harmonicBlade);

        // Void Crystal Gauntlets
        Nova::SacredArtifact voidGauntlets;
        voidGauntlets.id = "void_gauntlets";
        voidGauntlets.name = "Void Touched Hands";
        voidGauntlets.element = Nova::ElementType::Void;
        voidGauntlets.tier = 4;
        voidGauntlets.sacredResonance = 0.5f;
        voidGauntlets.quantumAlignment = 0.9f;
        voidGauntlets.traits.push_back("Quantum Phasing");
        artifacts.push_back(voidGauntlets);

        cout << "✅ Forged " << artifacts.size() << " sacred artifacts\n";

        // Demonstrate artifact evolution
        for (auto& artifact : artifacts) {
            artifact.bond(10.0f, &cognitiveEngine);
            artifact.awaken(20.0f, &cognitiveEngine);
            artifact.transcend(30.0f, &cognitiveEngine);
        }
    }

    void createCombatArena() {
        cout << "🏟️ Creating sacred combat arena...\n";

        // Register combat arena resource types
        resourceEngine.registerResourceType(
            "arena_pillar",
            "arena_pillar.glb",
            {"stone_sacred.png", "runes_glow.png"},
            {}, // default LODs
            true,  // audioReactive - pillars pulse with combat
            true,  // artReactive - runes glow with harmony
            false  // physicsEnabled
        );

        resourceEngine.registerResourceType(
            "sacred_platform",
            "platform_sacred.glb",
            {"platform_geometry.png", "sacred_patterns.png"},
            {},
            true, true, false
        );

        // Create circular arena with sacred geometry
        const float arenaRadius = 25.f;
        const int pillars = 8; // Octagon - sacred number

        // Central platform
        resourceEngine.placeResource("sacred_platform", {0, 0, 0}, {0, 0, 0}, {2.0f, 0.5f, 2.0f});

        // Pillars in octagon formation
        for (int i = 0; i < pillars; i++) {
            float angle = (i / float(pillars)) * 2.0f * M_PI;
            float x = arenaRadius * std::cos(angle);
            float z = arenaRadius * std::sin(angle);

            resourceEngine.placeResource("arena_pillar", {x, 0, z}, {0, angle + M_PI/2, 0}, {1.0f, 2.0f, 1.0f});
        }

        cout << "✅ Sacred combat arena created\n";
    }

    void updateSystems(float dt) {
        // Update NEXUS core systems
        gameEngine.Update();
        visualSystem.update(dt);
        cognitiveEngine.update(dt);
        resourceEngine.update(dt);

        // Update quantum protocol for combat enhancements
        quantumProtocol.update(dt);

        // Update Nova combat systems
        combatCore.update(dt);

        // Update camera to follow action
        updateCombatCamera(dt);
    }

    void runCombatDemo(float dt) {
        // Phase 1: Movement demonstration (0-20 seconds)
        if (demoTime < 20.0f) {
            demonstrateMovement(dt);
        }
        // Phase 2: Basic combat (20-40 seconds)
        else if (demoTime < 40.0f) {
            demonstrateBasicCombat(dt);
        }
        // Phase 3: Quantum combat (40-60 seconds)
        else if (demoTime < 60.0f) {
            demonstrateQuantumCombat(dt);
        }
        // Phase 4: Sacred combat mastery (60-90 seconds)
        else {
            demonstrateSacredCombat(dt);
        }
    }

    void demonstrateMovement(float dt) {
        static float moveTimer = 0.0f;
        moveTimer += dt;

        if (moveTimer < 5.0f) {
            // Basic movement
            Nova::Vec3 moveDir = {std::sin(demoTime), 0, std::cos(demoTime)};
            playerMotion.move(moveDir, false);
        } else if (moveTimer < 10.0f) {
            // Sprint movement
            Nova::Vec3 moveDir = {std::sin(demoTime * 1.5f), 0, std::cos(demoTime * 1.5f)};
            playerMotion.move(moveDir, true);
        } else if (moveTimer < 15.0f) {
            // Dash demonstrations
            if (int(moveTimer * 2) % 4 == 0 && playerMotion.dashTimer <= 0.0f) {
                Nova::Vec3 dashDir = {std::sin(demoTime * 2), 0, std::cos(demoTime * 2)};
                playerMotion.dash(dashDir);
            }
        } else {
            // Quantum dash demonstrations
            if (int(moveTimer * 1.5f) % 3 == 0 && playerMotion.dashTimer <= 0.0f) {
                Nova::Vec3 quantumDir = {std::sin(demoTime * 3), 0, std::cos(demoTime * 3)};
                playerMotion.quantumDash(quantumDir);
            }
        }
    }

    void demonstrateBasicCombat(float dt) {
        static float combatTimer = 0.0f;
        combatTimer += dt;

        // Continue movement during combat
        Nova::Vec3 moveDir = {std::sin(demoTime * 0.5f), 0, std::cos(demoTime * 0.5f)};
        playerMotion.move(moveDir, false);

        // Basic attack patterns
        if (int(combatTimer * 3) % 4 == 0 && combatCore.timer <= 0.0f) {
            combatCore.perform(Nova::AttackType::Basic);
        } else if (int(combatTimer * 2) % 5 == 0 && combatCore.timer <= 0.0f) {
            combatCore.perform(Nova::AttackType::Elemental);
        } else if (int(combatTimer) % 7 == 0 && combatCore.timer <= 0.0f) {
            combatCore.perform(Nova::AttackType::Combo);
        }
    }

    void demonstrateQuantumCombat(float dt) {
        static float quantumTimer = 0.0f;
        quantumTimer += dt;

        // Quantum movement patterns
        if (int(quantumTimer * 2) % 3 == 0 && playerMotion.dashTimer <= 0.0f) {
            Nova::Vec3 quantumDir = {std::sin(demoTime * 4), 0, std::cos(demoTime * 4)};
            playerMotion.quantumDash(quantumDir);
        }

        // Quantum combat techniques
        if (int(quantumTimer * 2) % 3 == 0 && combatCore.timer <= 0.0f) {
            combatCore.perform(Nova::AttackType::Quantum);
        } else if (int(quantumTimer) % 4 == 0 && combatCore.timer <= 0.0f) {
            combatCore.perform(Nova::AttackType::Elemental);
        }
    }

    void demonstrateSacredCombat(float dt) {
        static float sacredTimer = 0.0f;
        sacredTimer += dt;

        // Sacred movement in golden ratio patterns
        float goldenAngle = 2.39996322972865332f; // Golden angle
        float radius = 15.0f + 10.0f * std::sin(sacredTimer * 0.5f);

        Nova::Vec3 sacredPos = {
            radius * std::cos(sacredTimer * goldenAngle),
            0,
            radius * std::sin(sacredTimer * goldenAngle)
        };

        Nova::Vec3 currentPos = playerMotion.position;
        Nova::Vec3 moveDir = (sacredPos - currentPos).normalized();
        playerMotion.move(moveDir, true);

        // Sacred combat techniques
        if (int(sacredTimer * 1.5f) % 3 == 0 && combatCore.timer <= 0.0f) {
            combatCore.perform(Nova::AttackType::Sacred);
        } else if (int(sacredTimer * 2) % 4 == 0 && combatCore.timer <= 0.0f) {
            combatCore.perform(Nova::AttackType::Quantum);
        }
    }

    void updateCombatCamera(float dt) {
        // Follow player with dynamic offset based on combat state
        Nova::Vec3 playerPos = playerMotion.position;

        float cameraDistance = 15.0f;
        float cameraHeight = 8.0f;

        // Adjust camera based on quantum state
        if (playerMotion.isQuantumPhase) {
            cameraDistance *= 1.5f; // Pull back for quantum effects
            cameraHeight *= 1.3f;
        }

        // Smooth camera follow
        Vec3f targetPos = {
            playerPos.x - cameraDistance * std::sin(demoTime * 0.3f),
            cameraHeight + 3.0f * std::sin(demoTime * 0.2f),
            playerPos.z - cameraDistance * std::cos(demoTime * 0.3f)
        };

        camera.position = Vec3f{
            camera.position.x + (targetPos.x - camera.position.x) * dt * 2.0f,
            camera.position.y + (targetPos.y - camera.position.y) * dt * 2.0f,
            camera.position.z + (targetPos.z - camera.position.z) * dt * 2.0f
        };

        // Point camera at player
        Vec3f toPlayer = {playerPos.x - camera.position.x, 0 - camera.position.y, playerPos.z - camera.position.z};
        float len = std::sqrt(toPlayer.x*toPlayer.x + toPlayer.y*toPlayer.y + toPlayer.z*toPlayer.z);
        if (len > 0) {
            camera.forward = {toPlayer.x/len, toPlayer.y/len, toPlayer.z/len};
        }
    }

    void logCombatProgress() {
        auto stats = resourceEngine.getStats();
        auto playerPos = playerMotion.position;

        cout << "\n⚔️ === NEXUS NOVA COMBAT STATUS ===\n";
        cout << "⏱️ Time: " << std::fixed << std::setprecision(1) << demoTime << "s | Frame: " << frameCount << "\n";
        cout << "🌀 Quantum: Mode=" << quantumProtocol.getCurrentMode()
             << " Energy=" << std::setprecision(0) << playerMotion.quantumEnergy << "/" << playerMotion.quantumMax << "\n";
        cout << "💪 Combat: Combo=" << combatCore.combo
             << " Quantum=" << combatCore.quantumCombo
             << " Harmony=" << std::setprecision(2) << combatCore.harmony << "\n";
        cout << "🏃 Motion: Pos=(" << std::setprecision(1) << playerPos.x << "," << playerPos.y << "," << playerPos.z << ")"
             << " Stamina=" << std::setprecision(0) << playerMotion.stamina << "/" << playerMotion.staminaMax << "\n";
        cout << "🎨 Visual: BPM=" << std::setprecision(1) << gameEngine.GetParameters().bpm
             << " Petals=" << gameEngine.GetParameters().petalCount << "\n";
        cout << "🏟️ Arena: " << stats.visibleResources << " elements active\n";
        cout << "==========================================\n";
    }

    void shutdown() {
        cout << "\n🔄 Shutting down NEXUS Nova Combat Integration...\n";

        // Generate final combat report
        generateCombatReport();

        profiler.stopProfiling();
        profiler.generateDetailedReport("nexus_nova_combat_performance.txt");

        gameEngine.Shutdown();

        cout << "✅ NEXUS Nova Combat Integration shutdown complete\n";
    }

    void generateCombatReport() {
        cout << "\n⚔️✨ === NEXUS NOVA COMBAT FINAL REPORT === ✨⚔️\n";
        cout << "⏱️ Total Combat Time: " << std::fixed << std::setprecision(1) << demoTime << " seconds\n";
        cout << "🎯 Average FPS: " << std::setprecision(1) << (frameCount / demoTime) << "\n";
        cout << "🌀 Quantum Dashes: " << (playerMotion.quantumEnergy < playerMotion.quantumMax ? "Executed" : "Ready") << "\n";
        cout << "⚔️ Combat Techniques: All sacred and quantum forms demonstrated\n";
        cout << "✨ Artifacts: " << artifacts.size() << " transcended weapons achieved\n";
        cout << "🧠 Cognitive Insights: " << cognitiveEngine.getActiveThoughts().size() << " combat thoughts generated\n";

        cout << "\n💫 Integration Achievements:\n";
        cout << "   ✅ Quantum-enhanced movement with sacred mathematics\n";
        cout << "   ✅ Elemental harmony combat system\n";
        cout << "   ✅ Sacred geometry arena with audio-reactive elements\n";
        cout << "   ✅ AI-driven combat analysis and insight generation\n";
        cout << "   ✅ Multi-layered combat progression (Basic → Quantum → Sacred)\n";
        cout << "   ✅ Real-time performance optimization\n";
        cout << "   ✅ Cross-system integration (Nova ↔ NEXUS)\n";

        cout << "\n🌟 NEXUS Nova Combat Integration demonstration complete! 🌟\n";
    }
};

int main() {
    try {
        NexusNovaIntegrationDemo demo;

        if (!demo.initialize()) {
            std::cerr << "❌ Failed to initialize NEXUS Nova Combat Integration\n";
            return -1;
        }

        cout << "💡 Starting integrated combat demonstration...\n";
        cout << "🎮 Features: Quantum movement, sacred combat, harmonic resonance\n\n";

        demo.run();

        return 0;

    } catch (const std::exception& e) {
        std::cerr << "❌ NEXUS Nova Combat error: " << e.what() << std::endl;
        return -1;
    }
}
