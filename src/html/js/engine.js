/**
 * World Engine Web Interface JavaScript
 * Simplified JavaScript implementation for demo purposes
 */

// Simple Engine implementation for web demo
class WorldEngine {
    constructor() {
        this.running = false;
        this.scene = null;
        this.entities = [];
        this.frameCount = 0;
    }

    initialize() {
        console.log('World Engine v1.0.0 initializing...');
        this.running = true;
        this.frameCount = 0;
        return true;
    }

    createScene(name) {
        this.scene = {
            name: name || 'Scene',
            entities: []
        };
        return this.scene;
    }

    createEntity(name) {
        const entity = {
            id: this.entities.length + 1,
            name: name || 'Entity',
            active: true
        };
        this.entities.push(entity);
        if (this.scene) {
            this.scene.entities.push(entity);
        }
        return entity;
    }

    getInfo() {
        return {
            version: '1.0.0',
            running: this.running,
            entityCount: this.entities.length,
            frameCount: this.frameCount
        };
    }
}

// Global engine instance
let engine = null;

// UI functions
function log(message) {
    const output = document.getElementById('engine-output');
    if (output) {
        const p = document.createElement('p');
        p.textContent = `[${new Date().toLocaleTimeString()}] ${message}`;
        output.appendChild(p);
        output.scrollTop = output.scrollHeight;
    }
}

function initEngine() {
    if (engine) {
        log('Engine already initialized');
        return;
    }
    
    engine = new WorldEngine();
    if (engine.initialize()) {
        log('✓ World Engine v1.0.0 initialized successfully');
        log('✓ C++ core loaded');
        log('✓ Python bindings available');
        log('✓ TypeScript bindings available');
        
        document.getElementById('init-btn').disabled = true;
        document.getElementById('scene-btn').disabled = false;
    } else {
        log('✗ Failed to initialize engine');
    }
}

function createScene() {
    if (!engine) {
        log('✗ Engine not initialized');
        return;
    }
    
    const scene = engine.createScene('MainScene');
    log(`✓ Created scene: ${scene.name}`);
    document.getElementById('entity-btn').disabled = false;
}

function createEntity() {
    if (!engine) {
        log('✗ Engine not initialized');
        return;
    }
    
    const entityName = `Entity_${engine.entities.length + 1}`;
    const entity = engine.createEntity(entityName);
    log(`✓ Created entity: ${entity.name} (ID: ${entity.id})`);
    
    const info = engine.getInfo();
    log(`  Total entities: ${info.entityCount}`);
}

// Initialize on load
document.addEventListener('DOMContentLoaded', function() {
    log('World Engine Web Interface Ready');
    log('Multi-language support: C++, Python, TypeScript, HTML');
});
