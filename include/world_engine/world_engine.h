#ifndef WORLD_ENGINE_H
#define WORLD_ENGINE_H

#include "types.h"
#include "entity.h"
#include "component.h"
#include "system.h"
#include "scene.h"

namespace WorldEngine {

/**
 * @brief Main World Engine class
 */
class Engine {
public:
    Engine();
    virtual ~Engine();

    // Engine lifecycle
    bool initialize();
    void run();
    void shutdown();

    // Scene management
    void setActiveScene(std::shared_ptr<Scene> scene);
    std::shared_ptr<Scene> getActiveScene() const { return activeScene_; }

    // Engine state
    bool isRunning() const { return running_; }
    void stop() { running_ = false; }

    // Frame timing
    float getDeltaTime() const { return deltaTime_; }
    uint64_t getFrameCount() const { return frameCount_; }

private:
    bool running_;
    float deltaTime_;
    uint64_t frameCount_;
    std::shared_ptr<Scene> activeScene_;
};

// Utility functions
const char* getVersion();
void printEngineInfo();

} // namespace WorldEngine

#endif // WORLD_ENGINE_H
