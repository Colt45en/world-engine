#pragma once

#include <memory>
#include <string>
#include <vector>
#include <unordered_map>
#include <functional>
#include <chrono>

namespace NexusGame {

// Forward declarations
class AudioEngine;
class WorldEngine;
class ArtEngine;
class ClockBus;
class GameEntity;
class ComponentSystem;

/**
 * Core NEXUS Game Engine - Integrates with Holy Beat System
 * Provides C++ game logic that syncs with clockBus, audioEngine, artEngine, worldEngine
 */
class NexusGameEngine {
public:
    struct SystemParameters {
        double bpm = 120.0;
        int harmonics = 6;
        int petalCount = 8;
        double terrainRoughness = 0.4;
        double deltaTime = 0.016; // 60fps default
    };

    struct SystemStatus {
        bool clockBus = false;
        bool audioEngine = false;
        bool artEngine = false;
        bool worldEngine = false;
        bool training = false;
    };

private:
    SystemParameters m_parameters;
    SystemStatus m_systemStatus;

    std::unique_ptr<AudioEngine> m_audioEngine;
    std::unique_ptr<WorldEngine> m_worldEngine;
    std::unique_ptr<ArtEngine> m_artEngine;
    std::unique_ptr<ClockBus> m_clockBus;

    std::vector<std::shared_ptr<GameEntity>> m_entities;
    std::unordered_map<std::string, std::unique_ptr<ComponentSystem>> m_systems;

    std::chrono::high_resolution_clock::time_point m_lastUpdate;
    bool m_isRunning = false;

public:
    NexusGameEngine();
    ~NexusGameEngine();

    // Core Engine Functions
    bool Initialize(const SystemParameters& params = {});
    void Update();
    void Render();
    void Shutdown();

    // System Integration
    void SyncWithNexusSystems(const std::string& jsonData);
    std::string GetSystemStatusJson() const;
    void SetBPM(double bpm);
    void SetHarmonics(int harmonics);
    void SetPetalCount(int petalCount);
    void SetTerrainRoughness(double roughness);

    // Entity Management
    std::shared_ptr<GameEntity> CreateEntity(const std::string& name);
    void DestroyEntity(std::shared_ptr<GameEntity> entity);
    std::vector<std::shared_ptr<GameEntity>> GetEntities() const;

    // Component Systems
    template<typename T>
    void RegisterSystem(const std::string& name);

    template<typename T>
    T* GetSystem(const std::string& name);

    // Event System
    using EventCallback = std::function<void(const std::string&, void*)>;
    void RegisterEventHandler(const std::string& eventType, EventCallback callback);
    void TriggerEvent(const std::string& eventType, void* data = nullptr);

    // Getters
    const SystemParameters& GetParameters() const { return m_parameters; }
    const SystemStatus& GetSystemStatus() const { return m_systemStatus; }
    double GetDeltaTime() const;
    bool IsRunning() const { return m_isRunning; }

private:
    void UpdateSystems();
    void UpdateEntities();
    void CalculateDeltaTime();

    std::unordered_map<std::string, std::vector<EventCallback>> m_eventHandlers;
};

// Template implementations
template<typename T>
void NexusGameEngine::RegisterSystem(const std::string& name) {
    static_assert(std::is_base_of_v<ComponentSystem, T>, "T must derive from ComponentSystem");
    m_systems[name] = std::make_unique<T>(this);
}

template<typename T>
T* NexusGameEngine::GetSystem(const std::string& name) {
    auto it = m_systems.find(name);
    if (it != m_systems.end()) {
        return static_cast<T*>(it->second.get());
    }
    return nullptr;
}

} // namespace NexusGame
