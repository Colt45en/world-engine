#include "world_engine/system.h"
#include <algorithm>

namespace WorldEngine {

static SystemID nextSystemID = 1;

System::System()
    : id_(nextSystemID++) {
}

System::System(SystemID id)
    : id_(id) {
    if (id >= nextSystemID) {
        nextSystemID = id + 1;
    }
}

System::~System() {
}

void System::addEntity(std::shared_ptr<Entity> entity) {
    if (entity) {
        entities_.push_back(entity);
    }
}

void System::removeEntity(EntityID entityID) {
    entities_.erase(
        std::remove_if(entities_.begin(), entities_.end(),
            [entityID](const std::shared_ptr<Entity>& ent) {
                return ent && ent->getID() == entityID;
            }),
        entities_.end()
    );
}

} // namespace WorldEngine
