/**
 * World Engine TypeScript Example
 */

import { Engine, Scene, printEngineInfo, getVersion } from '../src/typescript/src/index';

function main(): void {
    // Print engine info
    printEngineInfo();
    
    // Create and initialize engine
    const engine = new Engine();
    if (!engine.initialize()) {
        console.error('Failed to initialize engine!');
        process.exit(1);
    }
    
    // Create a scene
    const scene = new Scene('ExampleScene');
    console.log(`\nCreated scene: ${scene.name}`);
    
    // Create some entities
    const player = scene.createEntity('Player');
    const enemy1 = scene.createEntity('Enemy1');
    const enemy2 = scene.createEntity('Enemy2');
    
    console.log('Created entities:');
    console.log(`  - ${player.name} (ID: ${player.id})`);
    console.log(`  - ${enemy1.name} (ID: ${enemy1.id})`);
    console.log(`  - ${enemy2.name} (ID: ${enemy2.id})`);
    
    // Set the active scene
    engine.setActiveScene(scene);
    
    // Run for a few frames
    console.log('\nRunning engine for 10 frames...');
    for (let i = 0; i < 10; i++) {
        engine.tick();
        console.log(`  Frame ${i + 1} - Delta: ${engine.getDeltaTime().toFixed(4)}s`);
    }
    
    console.log(`\nTotal frames: ${engine.getFrameCount()}`);
    
    // Shutdown
    engine.shutdown();
    console.log('Engine example completed successfully!');
    console.log(`World Engine version: ${getVersion()}`);
}

main();
