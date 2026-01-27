#include "world_engine/scene.h"
#include <algorithm>

namespace WorldEngine {

Scene::Scene()
    : name_("Scene")
    , nextEntityID_(1) {
}

Scene::Scene(const std::string& name)
    : name_(name)
    , nextEntityID_(1) {
}

Scene::~Scene() {
}

std::shared_ptr<Entity> Scene::createEntity(const std::string& name) {
    auto entity = std::make_shared<Entity>(nextEntityID_++);
    if (!name.empty()) {
        entity->setName(name);
    }
    entities_.push_back(entity);
    entityMap_[entity->getID()] = entity;
    return entity;
}

void Scene::destroyEntity(EntityID entityID) {
    auto it = entityMap_.find(entityID);
    if (it != entityMap_.end()) {
        entityMap_.erase(it);
    }
    
    entities_.erase(
        std::remove_if(entities_.begin(), entities_.end(),
            [entityID](const std::shared_ptr<Entity>& ent) {
                return ent && ent->getID() == entityID;
            }),
        entities_.end()
    );
}

std::shared_ptr<Entity> Scene::getEntity(EntityID entityID) const {
    auto it = entityMap_.find(entityID);
    if (it != entityMap_.end()) {
        return it->second;
    }
    return nullptr;
}

void Scene::addSystem(std::shared_ptr<System> system) {
    if (system) {
        systems_.push_back(system);
    }
}

void Scene::removeSystem(SystemID systemID) {
    systems_.erase(
        std::remove_if(systems_.begin(), systems_.end(),
            [systemID](const std::shared_ptr<System>& sys) {
                return sys && sys->getID() == systemID;
            }),
        systems_.end()
    );
}

std::shared_ptr<System> Scene::getSystem(SystemID systemID) const {
    for (const auto& sys : systems_) {
        if (sys && sys->getID() == systemID) {
            return sys;
        }
    }
    return nullptr;
}

void Scene::initialize() {
    for (auto& system : systems_) {
        if (system) {
            system->initialize();
        }
    }
}

void Scene::update(float deltaTime) {
    for (auto& system : systems_) {
        if (system) {
            system->update(deltaTime);
        }
    }
}

void Scene::shutdown() {
    for (auto& system : systems_) {
        if (system) {
            system->shutdown();
        }
    }
}

} // namespace WorldEngine
