#pragma once

#include <memory>
#include <vector>
#include <string>
#include <unordered_map>

namespace NexusGame {

class NexusGameEngine;
class Component;

/**
 * Game Entity - Represents any object in the NEXUS game world
 * Integrates with Holy Beat System for audio-visual synchronization
 */
class GameEntity {
public:
    using EntityID = uint64_t;

private:
    EntityID m_id;
    std::string m_name;
    bool m_active = true;

    std::unordered_map<std::string, std::unique_ptr<Component>> m_components;
    NexusGameEngine* m_engine;

public:
    GameEntity(EntityID id, const std::string& name, NexusGameEngine* engine);
    ~GameEntity();

    // Core Entity Functions
    void Update(double deltaTime);
    void Render();

    // Component Management
    template<typename T, typename... Args>
    T* AddComponent(Args&&... args);

    template<typename T>
    T* GetComponent();

    template<typename T>
    bool HasComponent() const;

    template<typename T>
    void RemoveComponent();

    // Properties
    EntityID GetID() const { return m_id; }
    const std::string& GetName() const { return m_name; }
    void SetName(const std::string& name) { m_name = name; }

    bool IsActive() const { return m_active; }
    void SetActive(bool active) { m_active = active; }

    NexusGameEngine* GetEngine() const { return m_engine; }

private:
    static EntityID s_nextID;
    static EntityID GenerateID() { return ++s_nextID; }

    friend class NexusGameEngine;
};

/**
 * Base Component class - All game components derive from this
 */
class Component {
protected:
    GameEntity* m_entity;
    bool m_enabled = true;

public:
    Component(GameEntity* entity) : m_entity(entity) {}
    virtual ~Component() = default;

    virtual void Initialize() {}
    virtual void Update(double deltaTime) {}
    virtual void Render() {}
    virtual void Destroy() {}

    GameEntity* GetEntity() const { return m_entity; }
    bool IsEnabled() const { return m_enabled; }
    void SetEnabled(bool enabled) { m_enabled = enabled; }
};

/**
 * Component System - Manages components of specific types
 */
class ComponentSystem {
protected:
    NexusGameEngine* m_engine;
    std::vector<Component*> m_components;

public:
    ComponentSystem(NexusGameEngine* engine) : m_engine(engine) {}
    virtual ~ComponentSystem() = default;

    virtual void Initialize() {}
    virtual void Update(double deltaTime) {}
    virtual void Render() {}
    virtual void Shutdown() {}

    virtual void RegisterComponent(Component* component);
    virtual void UnregisterComponent(Component* component);

    NexusGameEngine* GetEngine() const { return m_engine; }
    const std::vector<Component*>& GetComponents() const { return m_components; }
};

// Template implementations
template<typename T, typename... Args>
T* GameEntity::AddComponent(Args&&... args) {
    static_assert(std::is_base_of_v<Component, T>, "T must derive from Component");

    std::string typeName = typeid(T).name();
    if (m_components.find(typeName) != m_components.end()) {
        return static_cast<T*>(m_components[typeName].get());
    }

    auto component = std::make_unique<T>(this, std::forward<Args>(args)...);
    T* componentPtr = component.get();
    m_components[typeName] = std::move(component);

    componentPtr->Initialize();
    return componentPtr;
}

template<typename T>
T* GameEntity::GetComponent() {
    std::string typeName = typeid(T).name();
    auto it = m_components.find(typeName);
    if (it != m_components.end()) {
        return static_cast<T*>(it->second.get());
    }
    return nullptr;
}

template<typename T>
bool GameEntity::HasComponent() const {
    std::string typeName = typeid(T).name();
    return m_components.find(typeName) != m_components.end();
}

template<typename T>
void GameEntity::RemoveComponent() {
    std::string typeName = typeid(T).name();
    auto it = m_components.find(typeName);
    if (it != m_components.end()) {
        it->second->Destroy();
        m_components.erase(it);
    }
}

} // namespace NexusGame
