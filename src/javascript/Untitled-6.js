Nexus.prototype.spawnEnvironmentalEvent = function (type, origin, radius, duration) {
    const event = new EnvironmentalEvent(type, origin, radius, duration);
    this.activeEvents.push(event);
    console.log(`[Nexus] Spawned ${type} event at (${origin.x},${origin.y})`);
};

Nexus.prototype.updateEvents = function () {
    const now = Date.now();
    this.activeEvents = this.activeEvents.filter(event => {
        event.timeElapsed += 1;

        for (const [id, glyph] of this.goldGlyphs.entries()) {
            if (event.affects(glyph.position)) {
                event.applyEffect(glyph);
            }
        }

        return event.timeElapsed < event.duration;
    });
};
nexus.spawnEnvironmentalEvent("flux_surge", { x: 5, y: 5 }, 4, 20);
nexus.updateEvents(); // Call this during the update loop

truct LabNote {
    std::string text;
    glm::vec3 offset = glm:: vec3(0, 0.5f, 0);
    glm:: vec3 * target;
    int frameTrigger = -1;
    std::string type = "info";
    glm::vec3 color = glm:: vec3(1.0f);

    bool shouldDisplay(int currentFrame) const {
        return frameTrigger == -1 || currentFrame >= frameTrigger;
}

    float getOpacity(int currentFrame, int fadeIn = 30, int fadeOutStart = 300, int fadeOutDur = 60) const {
    if (frameTrigger == -1) return 1.0f;
        int dt = currentFrame - frameTrigger;
if (dt < 0) return 0;
        float fadeInAlpha = std:: min(1.0f, dt / float(fadeIn));
        float fadeOutAlpha = std:: max(0.0f, 1.0f - (dt - fadeOutStart) / float(fadeOutDur));
return std:: min(fadeInAlpha, fadeOutAlpha);
⚡ Step 4: Environmental Event System
Each event affects an area, modifies glyphs or terrain, and may trigger reactions across the grid.

    javascript
CopyEdit
class EnvironmentalEvent {
    constructor(type, origin, radius = 3, duration = 10) {
        this.type = type;           // e.g. "storm", "flux_surge", "memory_echo"
        this.origin = origin;       // Vector3
        this.radius = radius;
        this.duration = duration;   // In update ticks or seconds
        this.timeElapsed = 0;
    }

    affects(position) {
        const dx = position.x - this.origin.x;
        const dy = position.y - this.origin.y;
        return Math.sqrt(dx * dx + dy * dy) <= this.radius;
    }

    applyEffect(glyph) {
        switch (this.type) {
            case "storm":
                glyph.energyLevel *= 0.95;
                break;
            case "flux_surge":
                glyph.energyLevel += 1;
                glyph.meta.mutated = true;
                break;
            case "memory_echo":
                glyph.meta.memoryAwakened = true;
                break;
        }
    }
}
Nexus.prototype.spawnEnvironmentalEvent = function (type, origin, radius, duration) {
    const event = new EnvironmentalEvent(type, origin, radius, duration);
    this.activeEvents.push(event);
    console.log(`[Nexus] Spawned ${type} event at (${origin.x},${origin.y})`);
};

Nexus.prototype.updateEvents = function () {
    const now = Date.now();
    this.activeEvents = this.activeEvents.filter(event => {
        event.timeElapsed += 1;

        for (const [id, glyph] of this.goldGlyphs.entries()) {
            if (event.affects(glyph.position)) {
                event.applyEffect(glyph);
            }
        }

        return event.timeElapsed < event.duration;
    });
};
nexus.spawnEnvironmentalEvent("flux_surge", { x: 5, y: 5 }, 4, 20);
nexus.updateEvents(); // Call this during the update loop

 * RNES 6.0 - Recursive Nexus Engine Subsystem Core
    * ResourceEngineCore.js - Modular, Scalable, Infinite Cultivation - Compatible World Engine
        */

class ResourceEngineCore {
    constructor() {
        this.resources = new Map();
        this.loadedResources = new Map();
        this.visibleResources = [];
        this.camera = null;
        this.worldBounds = { x: 10000, y: 10000 };
        this.chunkSize = 250;
        this.loadDistance = 500;
        this.unloadDistance = 750;
    }

    initialize(camera) {
        this.camera = camera;
        this._generateChunks();
        console.log("[RNES-6.0] Core Engine Initialized");
    }

    _generateChunks() {
        this.chunks = [];
        const xCount = Math.ceil(this.worldBounds.x / this.chunkSize);
        const yCount = Math.ceil(this.worldBounds.y / this.chunkSize);
        for (let x = 0; x < xCount; x++) {
            for (let y = 0; y < yCount; y++) {
                this.chunks.push({
                    id: `chunk_${x}_${y}`,
                    position: { x: x * this.chunkSize, y: y * this.chunkSize },
                    loaded: false,
                    resources: []
                });
            }
        }
    }

    registerResourceType(id, config) {
        this.resources.set(id, {
            id,
            model: config.model,
            textures: config.textures,
            lodLevels: config.lodLevels || [
                { distance: 100, detail: "high" },
                { distance: 300, detail: "medium" },
                { distance: 500, detail: "low" }
            ],
            instances: []
        });
        return this;
    }

    placeResource(typeId, position, rotation = { x: 0, y: 0, z: 0 }, scale = { x: 1, y: 1, z: 1 }) {
        const type = this.resources.get(typeId);
        if (!type) return null;

        const instance = {
            id: `${typeId}_${Date.now()}_${Math.floor(Math.random() * 1000)}`,
            typeId,
            position,
            rotation,
            scale,
            currentLOD: null,
            visible: false,
            loaded: false,
            renderedObject: null
        };

        type.instances.push(instance);
        const cx = Math.floor(position.x / this.chunkSize);
        const cy = Math.floor(position.y / this.chunkSize);
        const chunk = this.chunks.find(c => c.id === `chunk_${cx}_${cy}`);
        if (chunk) chunk.resources.push(instance.id);
        return instance;
    }

    _distance(a, b) {
        const dx = a.x - b.x;
        const dy = a.y - b.y;
        const dz = (a.z || 0) - (b.z || 0);
        return Math.sqrt(dx * dx + dy * dy + dz * dz);
    }

    _determineLOD(resource, distance) {
        const lods = this.resources.get(resource.typeId).lodLevels;
        for (let level of lods) {
            if (distance <= level.distance) return level.detail;
        }
        return "lowest";
    }

    async _loadResource(resource) {
        const type = this.resources.get(resource.typeId);
        await new Promise(r => setTimeout(r, 30));
        resource.loaded = true;
        resource.renderedObject = {
            model: `loaded_${type.model}`,
            position: resource.position,
            rotation: resource.rotation,
            scale: resource.scale
        };
        this.loadedResources.set(resource.id, resource);
    }

    _unloadResource(resource) {
        resource.loaded = false;
        resource.visible = false;
        resource.currentLOD = null;
        resource.renderedObject = null;
        this.loadedResources.delete(resource.id);
    }

    _updateVisibility(resource, cameraPosition) {
        if (!resource.loaded) return;
        const dist = this._distance(resource.position, cameraPosition);
        if (dist <= this.loadDistance) {
            const lod = this._determineLOD(resource, dist);
            if (resource.currentLOD !== lod) resource.currentLOD = lod;
            if (!resource.visible) {
                resource.visible = true;
                this.visibleResources.push(resource.id);
            }
        } else if (resource.visible) {
            resource.visible = false;
            this.visibleResources = this.visibleResources.filter(id => id !== resource.id);
        }
    }

    _findTypeByInstanceId(instanceId) {
        for (const type of this.resources.values()) {
            if (type.instances.some(inst => inst.id === instanceId)) return type;
        }
        return null;
    }

    async _loadChunk(chunk) {
        chunk.loaded = true;
        for (const resId of chunk.resources) {
            const type = this._findTypeByInstanceId(resId);
            if (!type) continue;
            const resource = type.instances.find(r => r.id === resId);
            if (resource && !resource.loaded) await this._loadResource(resource);
        }
    }

    _unloadChunk(chunk) {
        chunk.loaded = false;
        for (const resId of chunk.resources) {
            const resource = this.loadedResources.get(resId);
            if (resource) this._unloadResource(resource);
        }
    }

    _updateChunks(cameraPosition) {
        for (const chunk of this.chunks) {
            const center = {
                x: chunk.position.x + this.chunkSize / 2,
                y: chunk.position.y + this.chunkSize / 2,
                z: 0
            };
            const dist = this._distance(center, cameraPosition);
            if (dist <= this.loadDistance && !chunk.loaded) this._loadChunk(chunk);
            else if (dist > this.unloadDistance && chunk.loaded) this._unloadChunk(chunk);
        }
    }

    update() {
        if (!this.camera) return;
        const camPos = this.camera.position;
        this._updateChunks(camPos);
        for (const resource of this.loadedResources.values()) {
            this._updateVisibility(resource, camPos);
        }
    }

    render(renderer) {
        for (const id of this.visibleResources) {
            const resource = this.loadedResources.get(id);
            if (resource?.renderedObject) {
                console.log(`[RNES Render] ${resource.id} :: LOD ${resource.currentLOD}`);
                // Hook to renderer.renderObject(resource.renderedObject);
            }
        }
    }
}

export { ResourceEngineCore };
