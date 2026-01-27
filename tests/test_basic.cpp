#include <world_engine/world_engine.h>
#include <iostream>
#include <cassert>

int main() {
    std::cout << "Running basic World Engine tests..." << std::endl;
    
    // Test 1: Engine creation and initialization
    {
        WorldEngine::Engine engine;
        assert(engine.initialize());
        assert(engine.isRunning());
        engine.shutdown();
        assert(!engine.isRunning());
        std::cout << "✓ Test 1: Engine lifecycle passed" << std::endl;
    }
    
    // Test 2: Scene creation
    {
        WorldEngine::Scene scene("TestScene");
        assert(scene.getName() == "TestScene");
        std::cout << "✓ Test 2: Scene creation passed" << std::endl;
    }
    
    // Test 3: Entity creation
    {
        WorldEngine::Scene scene;
        auto entity = scene.createEntity("TestEntity");
        assert(entity != nullptr);
        assert(entity->getName() == "TestEntity");
        assert(entity->isActive());
        std::cout << "✓ Test 3: Entity creation passed" << std::endl;
    }
    
    // Test 4: Component creation
    {
        auto component = std::make_shared<WorldEngine::Component>();
        assert(component != nullptr);
        assert(component->isEnabled());
        std::cout << "✓ Test 4: Component creation passed" << std::endl;
    }
    
    // Test 5: System creation
    {
        auto system = std::make_shared<WorldEngine::System>();
        assert(system != nullptr);
        std::cout << "✓ Test 5: System creation passed" << std::endl;
    }
    
    // Test 6: Version info
    {
        const char* version = WorldEngine::getVersion();
        assert(version != nullptr);
        std::cout << "✓ Test 6: Version info (" << version << ") passed" << std::endl;
    }
    
    std::cout << "\nAll basic tests passed!" << std::endl;
    return 0;
}
