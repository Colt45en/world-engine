#include "world_engine/world_engine.h"
#include <iostream>
#include <chrono>
#include <thread>

namespace WorldEngine {

Engine::Engine()
    : running_(false)
    , deltaTime_(0.0f)
    , frameCount_(0)
    , activeScene_(nullptr) {
}

Engine::~Engine() {
    if (running_) {
        shutdown();
    }
}

bool Engine::initialize() {
    std::cout << "World Engine v" << VERSION << " initializing..." << std::endl;
    running_ = true;
    deltaTime_ = 0.016f; // Default to ~60 FPS
    frameCount_ = 0;
    return true;
}

void Engine::run() {
    if (!running_) {
        std::cerr << "Engine not initialized!" << std::endl;
        return;
    }

    auto lastTime = std::chrono::high_resolution_clock::now();

    while (running_) {
        auto currentTime = std::chrono::high_resolution_clock::now();
        std::chrono::duration<float> elapsed = currentTime - lastTime;
        deltaTime_ = elapsed.count();
        lastTime = currentTime;

        // Update active scene
        if (activeScene_) {
            activeScene_->update(deltaTime_);
        }

        frameCount_++;

        // Small sleep to prevent CPU spinning
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

void Engine::shutdown() {
    std::cout << "World Engine shutting down..." << std::endl;
    if (activeScene_) {
        activeScene_->shutdown();
    }
    running_ = false;
}

void Engine::setActiveScene(std::shared_ptr<Scene> scene) {
    if (activeScene_) {
        activeScene_->shutdown();
    }
    activeScene_ = scene;
    if (activeScene_) {
        activeScene_->initialize();
    }
}

const char* getVersion() {
    return VERSION;
}

void printEngineInfo() {
    std::cout << "==================================" << std::endl;
    std::cout << "World Engine" << std::endl;
    std::cout << "Version: " << VERSION << std::endl;
    std::cout << "Multi-language game engine" << std::endl;
    std::cout << "Supports: C++, Python, TypeScript" << std::endl;
    std::cout << "==================================" << std::endl;
}

} // namespace WorldEngine
