#!/usr/bin/env python3
"""
World Engine Python Example
"""

import world_engine as we

def main():
    # Print engine info
    we.print_info()
    
    # Create and initialize engine
    engine = we.Engine()
    if not engine.initialize():
        print("Failed to initialize engine!")
        return 1
    
    # Create a scene
    scene = we.Scene("ExampleScene")
    print(f"\nCreated scene: {scene.name}")
    
    # Create some entities
    player = scene.create_entity("Player")
    enemy1 = scene.create_entity("Enemy1")
    enemy2 = scene.create_entity("Enemy2")
    
    print("Created entities:")
    print(f"  - {player.name} (ID: {player.id})")
    print(f"  - {enemy1.name} (ID: {enemy1.id})")
    print(f"  - {enemy2.name} (ID: {enemy2.id})")
    
    # Set the active scene
    engine.active_scene = scene
    
    print("\nEngine example completed successfully!")
    print(f"World Engine version: {we.get_version()}")
    
    return 0

if __name__ == "__main__":
    exit(main())
