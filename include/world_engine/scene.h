#ifndef WORLD_ENGINE_SCENE_H
#define WORLD_ENGINE_SCENE_H

#include "types.h"
#include "entity.h"
#include "system.h"
#include <string>
#include <vector>
#include <memory>
#include <unordered_map>

namespace WorldEngine {

/**
 * @brief Scene class managing entities and systems
 */
class Scene {
public:
    Scene();
    explicit Scene(const std::string& name);
    virtual ~Scene();

    // Scene name
    void setName(const std::string& name) { name_ = name; }
    const std::string& getName() const { return name_; }

    // Entity management
    std::shared_ptr<Entity> createEntity(const std::string& name = "");
    void destroyEntity(EntityID entityID);
    std::shared_ptr<Entity> getEntity(EntityID entityID) const;
    const std::vector<std::shared_ptr<Entity>>& getEntities() const { return entities_; }

    // System management
    void addSystem(std::shared_ptr<System> system);
    void removeSystem(SystemID systemID);
    std::shared_ptr<System> getSystem(SystemID systemID) const;

    // Scene lifecycle
    void initialize();
    void update(float deltaTime);
    void shutdown();

private:
    std::string name_;
    std::vector<std::shared_ptr<Entity>> entities_;
    std::vector<std::shared_ptr<System>> systems_;
    std::unordered_map<EntityID, std::shared_ptr<Entity>> entityMap_;
    EntityID nextEntityID_;
};

} // namespace WorldEngine

#endif // WORLD_ENGINE_SCENE_H
