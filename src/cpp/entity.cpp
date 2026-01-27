#include "world_engine/entity.h"
#include "world_engine/component.h"
#include <algorithm>

namespace WorldEngine {

static EntityID nextEntityID = 1;

Entity::Entity()
    : id_(nextEntityID++)
    , name_("Entity")
    , active_(true) {
}

Entity::Entity(EntityID id)
    : id_(id)
    , name_("Entity")
    , active_(true) {
    if (id >= nextEntityID) {
        nextEntityID = id + 1;
    }
}

Entity::~Entity() {
    components_.clear();
}

void Entity::addComponent(std::shared_ptr<Component> component) {
    if (component) {
        components_.push_back(component);
    }
}

void Entity::removeComponent(ComponentID componentID) {
    components_.erase(
        std::remove_if(components_.begin(), components_.end(),
            [componentID](const std::shared_ptr<Component>& comp) {
                return comp && comp->getID() == componentID;
            }),
        components_.end()
    );
}

std::shared_ptr<Component> Entity::getComponent(ComponentID componentID) const {
    for (const auto& comp : components_) {
        if (comp && comp->getID() == componentID) {
            return comp;
        }
    }
    return nullptr;
}

} // namespace WorldEngine
