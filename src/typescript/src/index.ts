/**
 * World Engine TypeScript Core
 * Multi-language game engine TypeScript/JavaScript bindings
 */

export const VERSION = '1.0.0';

/**
 * Entity ID type
 */
export type EntityID = number;

/**
 * Component ID type
 */
export type ComponentID = number;

/**
 * System ID type
 */
export type SystemID = number;

/**
 * Component base class
 */
export class Component {
    private static nextID: ComponentID = 1;
    
    public readonly id: ComponentID;
    public enabled: boolean;
    
    constructor() {
        this.id = Component.nextID++;
        this.enabled = true;
    }
    
    /**
     * Get component type name
     */
    getTypeName(): string {
        return 'Component';
    }
    
    /**
     * Update method (called each frame)
     */
    update(deltaTime: number): void {
        // Override in derived classes
    }
}

/**
 * Entity class representing a game object
 */
export class Entity {
    private static nextID: EntityID = 1;
    
    public readonly id: EntityID;
    public name: string;
    public active: boolean;
    private components: Component[];
    
    constructor(id?: EntityID) {
        this.id = id !== undefined ? id : Entity.nextID++;
        this.name = 'Entity';
        this.active = true;
        this.components = [];
    }
    
    /**
     * Add a component to this entity
     */
    addComponent(component: Component): void {
        this.components.push(component);
    }
    
    /**
     * Remove a component by ID
     */
    removeComponent(componentID: ComponentID): void {
        this.components = this.components.filter(c => c.id !== componentID);
    }
    
    /**
     * Get a component by ID
     */
    getComponent(componentID: ComponentID): Component | undefined {
        return this.components.find(c => c.id === componentID);
    }
    
    /**
     * Get all components
     */
    getComponents(): Component[] {
        return this.components;
    }
}

/**
 * System base class
 */
export class System {
    private static nextID: SystemID = 1;
    
    public readonly id: SystemID;
    protected entities: Entity[];
    
    constructor() {
        this.id = System.nextID++;
        this.entities = [];
    }
    
    /**
     * Get system type name
     */
    getTypeName(): string {
        return 'System';
    }
    
    /**
     * Initialize system
     */
    initialize(): void {
        // Override in derived classes
    }
    
    /**
     * Update system (called each frame)
     */
    update(deltaTime: number): void {
        // Override in derived classes
    }
    
    /**
     * Shutdown system
     */
    shutdown(): void {
        // Override in derived classes
    }
    
    /**
     * Add entity to this system
     */
    addEntity(entity: Entity): void {
        this.entities.push(entity);
    }
    
    /**
     * Remove entity from this system
     */
    removeEntity(entityID: EntityID): void {
        this.entities = this.entities.filter(e => e.id !== entityID);
    }
}

/**
 * Scene class managing entities and systems
 */
export class Scene {
    public name: string;
    private entities: Entity[];
    private systems: System[];
    private entityMap: Map<EntityID, Entity>;
    private nextEntityID: EntityID;
    
    constructor(name: string = 'Scene') {
        this.name = name;
        this.entities = [];
        this.systems = [];
        this.entityMap = new Map();
        this.nextEntityID = 1;
    }
    
    /**
     * Create a new entity
     */
    createEntity(name?: string): Entity {
        const entity = new Entity(this.nextEntityID++);
        if (name) {
            entity.name = name;
        }
        this.entities.push(entity);
        this.entityMap.set(entity.id, entity);
        return entity;
    }
    
    /**
     * Destroy an entity
     */
    destroyEntity(entityID: EntityID): void {
        this.entityMap.delete(entityID);
        this.entities = this.entities.filter(e => e.id !== entityID);
    }
    
    /**
     * Get entity by ID
     */
    getEntity(entityID: EntityID): Entity | undefined {
        return this.entityMap.get(entityID);
    }
    
    /**
     * Get all entities
     */
    getEntities(): Entity[] {
        return this.entities;
    }
    
    /**
     * Add a system
     */
    addSystem(system: System): void {
        this.systems.push(system);
    }
    
    /**
     * Remove a system
     */
    removeSystem(systemID: SystemID): void {
        this.systems = this.systems.filter(s => s.id !== systemID);
    }
    
    /**
     * Get system by ID
     */
    getSystem(systemID: SystemID): System | undefined {
        return this.systems.find(s => s.id === systemID);
    }
    
    /**
     * Initialize scene
     */
    initialize(): void {
        for (const system of this.systems) {
            system.initialize();
        }
    }
    
    /**
     * Update scene
     */
    update(deltaTime: number): void {
        for (const system of this.systems) {
            system.update(deltaTime);
        }
    }
    
    /**
     * Shutdown scene
     */
    shutdown(): void {
        for (const system of this.systems) {
            system.shutdown();
        }
    }
}

/**
 * Main World Engine class
 */
export class Engine {
    private running: boolean;
    private deltaTime: number;
    private frameCount: number;
    private activeScene: Scene | null;
    private lastTime: number;
    
    constructor() {
        this.running = false;
        this.deltaTime = 0.016; // ~60 FPS
        this.frameCount = 0;
        this.activeScene = null;
        this.lastTime = 0;
    }
    
    /**
     * Initialize the engine
     */
    initialize(): boolean {
        console.log(`World Engine v${VERSION} initializing...`);
        this.running = true;
        this.frameCount = 0;
        this.lastTime = Date.now();
        return true;
    }
    
    /**
     * Run the engine (single frame update)
     */
    tick(): void {
        if (!this.running) {
            console.error('Engine not initialized!');
            return;
        }
        
        const currentTime = Date.now();
        this.deltaTime = (currentTime - this.lastTime) / 1000.0;
        this.lastTime = currentTime;
        
        if (this.activeScene) {
            this.activeScene.update(this.deltaTime);
        }
        
        this.frameCount++;
    }
    
    /**
     * Shutdown the engine
     */
    shutdown(): void {
        console.log('World Engine shutting down...');
        if (this.activeScene) {
            this.activeScene.shutdown();
        }
        this.running = false;
    }
    
    /**
     * Set the active scene
     */
    setActiveScene(scene: Scene): void {
        if (this.activeScene) {
            this.activeScene.shutdown();
        }
        this.activeScene = scene;
        if (this.activeScene) {
            this.activeScene.initialize();
        }
    }
    
    /**
     * Get the active scene
     */
    getActiveScene(): Scene | null {
        return this.activeScene;
    }
    
    /**
     * Check if engine is running
     */
    isRunning(): boolean {
        return this.running;
    }
    
    /**
     * Stop the engine
     */
    stop(): void {
        this.running = false;
    }
    
    /**
     * Get delta time
     */
    getDeltaTime(): number {
        return this.deltaTime;
    }
    
    /**
     * Get frame count
     */
    getFrameCount(): number {
        return this.frameCount;
    }
}

/**
 * Get World Engine version
 */
export function getVersion(): string {
    return VERSION;
}

/**
 * Print engine information
 */
export function printEngineInfo(): void {
    console.log('==================================');
    console.log('World Engine');
    console.log(`Version: ${VERSION}`);
    console.log('Multi-language game engine');
    console.log('Supports: C++, Python, TypeScript');
    console.log('==================================');
}
