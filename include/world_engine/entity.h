#ifndef WORLD_ENGINE_ENTITY_H
#define WORLD_ENGINE_ENTITY_H

#include "types.h"
#include <string>
#include <vector>
#include <memory>

namespace WorldEngine {

class Component;

/**
 * @brief Entity class representing a game object in the world
 */
class Entity {
public:
    Entity();
    explicit Entity(EntityID id);
    virtual ~Entity();

    // Get entity ID
    EntityID getID() const { return id_; }

    // Name management
    void setName(const std::string& name) { name_ = name; }
    const std::string& getName() const { return name_; }

    // Active state
    void setActive(bool active) { active_ = active; }
    bool isActive() const { return active_; }

    // Component management
    void addComponent(std::shared_ptr<Component> component);
    void removeComponent(ComponentID componentID);
    std::shared_ptr<Component> getComponent(ComponentID componentID) const;
    const std::vector<std::shared_ptr<Component>>& getComponents() const { return components_; }

private:
    EntityID id_;
    std::string name_;
    bool active_;
    std::vector<std::shared_ptr<Component>> components_;
};

} // namespace WorldEngine

#endif // WORLD_ENGINE_ENTITY_H
