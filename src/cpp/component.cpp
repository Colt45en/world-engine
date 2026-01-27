#include "world_engine/component.h"

namespace WorldEngine {

static ComponentID nextComponentID = 1;

Component::Component()
    : id_(nextComponentID++)
    , enabled_(true) {
}

Component::Component(ComponentID id)
    : id_(id)
    , enabled_(true) {
    if (id >= nextComponentID) {
        nextComponentID = id + 1;
    }
}

Component::~Component() {
}

} // namespace WorldEngine
