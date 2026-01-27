#ifndef WORLD_ENGINE_SYSTEM_H
#define WORLD_ENGINE_SYSTEM_H

#include "types.h"
#include "entity.h"
#include <string>
#include <vector>
#include <memory>

namespace WorldEngine {

/**
 * @brief Base class for all systems
 */
class System {
public:
    System();
    explicit System(SystemID id);
    virtual ~System();

    // Get system ID
    SystemID getID() const { return id_; }

    // System type name
    virtual std::string getTypeName() const { return "System"; }

    // Initialize system
    virtual void initialize() {}

    // Update system (called each frame)
    virtual void update(float deltaTime) {}

    // Shutdown system
    virtual void shutdown() {}

    // Entity management
    virtual void addEntity(std::shared_ptr<Entity> entity);
    virtual void removeEntity(EntityID entityID);

protected:
    std::vector<std::shared_ptr<Entity>> entities_;

private:
    SystemID id_;
};

} // namespace WorldEngine

#endif // WORLD_ENGINE_SYSTEM_H
