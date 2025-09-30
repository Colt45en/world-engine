// Test Node.js bindings for NEXUS Game Engine
const path = require('path');
const fs = require('fs');

console.log('🎮 Testing NEXUS Game Logic Node.js Bindings...\n');

// Check if addon exists
const addonPath = path.join(__dirname, '..', 'nexus_game.node');
if (!fs.existsSync(addonPath)) {
    console.log('⚠️  Node addon not found at:', addonPath);
    console.log('💡 Build instructions:');
    console.log('   cd game-logic');
    console.log('   npm install');
    console.log('   npm run build');
    console.log('');
    process.exit(0);
}

try {
    // Load the NEXUS Game addon
    const nexusGame = require(addonPath);
    console.log('✅ Successfully loaded NEXUS Game addon');

    // Create engine instance
    console.log('\n🚀 Creating NexusGameEngine...');
    const engine = new nexusGame.NexusGameEngine();

    // Initialize with NEXUS Holy Beat parameters
    const params = {
        bpm: 120,
        harmonics: 6,
        petalCount: 8,
        terrainRoughness: 0.4
    };

    console.log('⚙️  Initializing with parameters:', params);
    const initialized = engine.initialize(params);

    if (!initialized) {
        console.log('❌ Failed to initialize engine');
        process.exit(1);
    }

    console.log('✅ Engine initialized successfully');
    console.log('🏃 Engine running:', engine.isRunning());

    // Create test entities
    console.log('\n🎯 Creating game entities...');

    const centerEntity = engine.createEntity('MandalaCenter');
    console.log(`✨ Created center entity: ${centerEntity.getName()}`);

    // Add components
    centerEntity.addTransform();
    centerEntity.addAudioSync();
    centerEntity.addArtSync();

    console.log('🧩 Added components:');
    console.log('   - Transform:', centerEntity.getTransform());
    console.log('   - AudioSync:', centerEntity.getAudioSync());
    console.log('   - ArtSync:', centerEntity.getArtSync());

    // Create petal entities
    const petals = [];
    for (let i = 0; i < params.petalCount; i++) {
        const petal = engine.createEntity(`Petal_${i}`);
        petal.addTransform();
        petal.addAudioSync();
        petal.addArtSync();
        petal.addPhysics();
        petals.push(petal);
    }

    console.log(`🌸 Created ${petals.length} petal entities`);

    // Test parameter updates
    console.log('\n🎛️  Testing parameter updates...');
    engine.setBPM(140);
    engine.setHarmonics(8);
    engine.setPetalCount(12);
    engine.setTerrainRoughness(0.6);
    console.log('✅ Updated engine parameters');

    // Simulate NEXUS system sync
    const nexusData = JSON.stringify({
        timestamp: new Date().toISOString(),
        status: 'running',
        version: '1.0.0',
        components: {
            clockBus: 'active',
            audioEngine: 'active',
            artEngine: 'active',
            worldEngine: 'active',
            training: 'ready'
        },
        parameters: {
            bpm: 140,
            harmonics: 8,
            petalCount: 12,
            terrainRoughness: 0.6
        }
    });

    console.log('\n🔗 Syncing with NEXUS systems...');
    engine.syncWithNexusSystems(nexusData);

    const status = engine.getSystemStatusJson();
    console.log('📊 System Status:', JSON.parse(status));

    // Test game loop
    console.log('\n🔄 Running test game loop...');

    const startTime = Date.now();
    let frameCount = 0;
    const maxFrames = 60; // 1 second at 60fps

    const gameLoop = setInterval(() => {
        // Update engine
        engine.update();

        // Render (would draw to screen in real implementation)
        engine.render();

        frameCount++;

        if (frameCount % 20 === 0) {
            const elapsed = Date.now() - startTime;
            const deltaTime = engine.getDeltaTime();
            console.log(`⏱️  Frame ${frameCount} | Elapsed: ${elapsed}ms | DeltaTime: ${(deltaTime * 1000).toFixed(2)}ms`);
        }

        // Test entity operations
        if (frameCount === 30) {
            console.log('🔄 Testing entity operations...');
            petals[0].setActive(false);
            console.log(`   Deactivated ${petals[0].getName()}: Active = ${petals[0].isActive()}`);
        }

        if (frameCount === 45) {
            petals[0].setActive(true);
            petals[0].setName('ReactivatedPetal');
            console.log(`   Reactivated petal: Name = ${petals[0].getName()}, Active = ${petals[0].isActive()}`);
        }

        // Stop after max frames
        if (frameCount >= maxFrames) {
            clearInterval(gameLoop);

            // Shutdown engine
            console.log('\n🔄 Shutting down engine...');
            engine.shutdown();

            const totalTime = Date.now() - startTime;
            const avgFPS = (frameCount * 1000 / totalTime).toFixed(1);

            console.log('\n📊 Test Results:');
            console.log(`   Total Frames: ${frameCount}`);
            console.log(`   Total Time: ${totalTime}ms`);
            console.log(`   Average FPS: ${avgFPS}`);
            console.log(`   Engine Running: ${engine.isRunning()}`);

            console.log('\n🎵✨ Node.js bindings test completed successfully! ✨🎵');
        }
    }, 16); // ~60 FPS

} catch (error) {
    console.error('❌ Error testing NEXUS Game bindings:', error.message);
    console.log('\n💡 Make sure to build the addon first:');
    console.log('   cd game-logic');
    console.log('   npm install');
    console.log('   npm run build');
    process.exit(1);
}
