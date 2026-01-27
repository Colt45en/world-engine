#ifndef WORLD_ENGINE_COMPONENT_H
#define WORLD_ENGINE_COMPONENT_H

#include "types.h"
#include <string>

namespace WorldEngine {

/**
 * @brief Base class for all components
 */
class Component {
public:
    Component();
    explicit Component(ComponentID id);
    virtual ~Component();

    // Get component ID
    ComponentID getID() const { return id_; }

    // Component type name (for debugging)
    virtual std::string getTypeName() const { return "Component"; }

    // Update method (called each frame)
    virtual void update(float deltaTime) {}

    // Enable/disable component
    void setEnabled(bool enabled) { enabled_ = enabled; }
    bool isEnabled() const { return enabled_; }

private:
    ComponentID id_;
    bool enabled_;
};

} // namespace WorldEngine

#endif // WORLD_ENGINE_COMPONENT_H
