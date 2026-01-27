#include <world_engine/world_engine.h>
#include <iostream>
#include <thread>
#include <chrono>

int main() {
    // Print engine info
    WorldEngine::printEngineInfo();
    
    // Create and initialize engine
    WorldEngine::Engine engine;
    if (!engine.initialize()) {
        std::cerr << "Failed to initialize engine!" << std::endl;
        return 1;
    }
    
    // Create a scene
    auto scene = std::make_shared<WorldEngine::Scene>("ExampleScene");
    std::cout << "\nCreated scene: " << scene->getName() << std::endl;
    
    // Create some entities
    auto player = scene->createEntity("Player");
    auto enemy1 = scene->createEntity("Enemy1");
    auto enemy2 = scene->createEntity("Enemy2");
    
    std::cout << "Created entities:" << std::endl;
    std::cout << "  - " << player->getName() << " (ID: " << player->getID() << ")" << std::endl;
    std::cout << "  - " << enemy1->getName() << " (ID: " << enemy1->getID() << ")" << std::endl;
    std::cout << "  - " << enemy2->getName() << " (ID: " << enemy2->getID() << ")" << std::endl;
    
    // Set the active scene
    engine.setActiveScene(scene);
    
    // Run for a few frames
    std::cout << "\nRunning engine for 10 frames..." << std::endl;
    for (int i = 0; i < 10; ++i) {
        // Simulate a frame update
        scene->update(0.016f);
        std::cout << "  Frame " << (i + 1) << " - Delta: " << engine.getDeltaTime() << "s" << std::endl;
        std::this_thread::sleep_for(std::chrono::milliseconds(16));
    }
    
    std::cout << "\nTotal frames: " << engine.getFrameCount() << std::endl;
    
    // Shutdown
    engine.shutdown();
    std::cout << "Engine example completed successfully!" << std::endl;
    
    return 0;
}
