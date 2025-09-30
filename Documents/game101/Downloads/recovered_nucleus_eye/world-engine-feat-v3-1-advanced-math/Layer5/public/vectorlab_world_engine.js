/*
 * VectorLab World Engine Integration
 * =================================
 *
 * This system bridges the C++ VectorLab Engine Core (Heart, Timeline, Codex, Glyphs)
 * into a comprehensive world building and visualization engine.
 *
 * Key Features:
 * • Heart Engine resonance drives environmental effects
 * • Timeline manipulation controls memory fields and terrain generation
 * • Glyph system provides symbolic programming for world events
 * • Real-time 3D visualization of all systems
 * • Codex rules enforce world integrity and coherence
//----------------------------------------------------//
// Core VectorLab Engine Bridge (JS Implementation)  //
//----------------------------------------------------//

// eslint-disable-next-line linebreak-style//

class HeartEngine {
    constructor() {
        this.resonance = 0.0;
        this.resonanceCap = 1.0;
        this.state = 'idle'; // idle, pulse, echo, silent
        this.lastPulseTime = 0;
        this.pulseHistory = [];
    }

    pulse(intensity = 0.1) {
        this.resonance = Math.max(0, Math.min(this.resonanceCap, this.resonance + intensity));
        this.state = 'pulse';
        this.lastPulseTime = Date.now();
        this.pulseHistory.push({ time: this.lastPulseTime, intensity, resonance: this.resonance });

        // Keep only last 100 pulses
        if (this.pulseHistory.length > 100) {
            this.pulseHistory.shift();
        }

        console.log(`💓 Heart pulses with intensity: ${intensity}, resonance: ${this.resonance.toFixed(3)}`);
        return this.resonance;
    }

    decay(amount = 0.01) {
        this.resonance = Math.max(0, this.resonance - amount);
        if (this.resonance === 0) {
            this.state = 'silent';
        }
        return this.resonance;
    }

    echo() {
        if (this.pulseHistory.length > 0) {
            const lastPulse = this.pulseHistory[this.pulseHistory.length - 1];
            const echoIntensity = lastPulse.intensity * 0.6;
            this.pulse(echoIntensity);
            this.state = 'echo';
            console.log(`🔁 Heart echoes previous resonance: ${this.resonance.toFixed(3)}`);
        }
        return this.resonance;
    }

    raiseCap(amount = 0.1) {
        this.resonanceCap += amount;
        console.log(`🧬 Resonance cap raised to: ${this.resonanceCap.toFixed(3)}`);
        return this.resonanceCap;
    }

    getEmotionalState() {
        if (this.resonance > 0.8) return 'euphoric';
        if (this.resonance > 0.6) return 'elevated';
        if (this.resonance > 0.4) return 'balanced';
        if (this.resonance > 0.2) return 'calm';
        if (this.resonance > 0.0) return 'gentle';
        return 'silent';
    }
}

class TimelineEngine {
    constructor() {
        this.currentFrame = 0;
        this.isPlaying = true;
        this.frameRate = 60;
        this.keyFrames = new Map();
        this.memorySnapshots = new Map();
        this.temporalEvents = [];
    }

    stepForward() {
        this.currentFrame++;
        this.processTemporalEvents();
        return this.currentFrame;
    }

    stepBack() {
        this.currentFrame = Math.max(0, this.currentFrame - 1);
        this.restoreMemorySnapshot(this.currentFrame);
        return this.currentFrame;
    }

    togglePlay() {
        this.isPlaying = !this.isPlaying;
        return this.isPlaying;
    }

    update() {
        if (this.isPlaying) {
            return this.stepForward();
        }
        return this.currentFrame;
    }

    addKeyFrame(frame, data) {
        this.keyFrames.set(frame, data);
        console.log(`⏰ Keyframe added at frame ${frame}`);
    }

    createMemorySnapshot(worldState) {
        this.memorySnapshots.set(this.currentFrame, JSON.parse(JSON.stringify(worldState)));
        console.log(`📸 Memory snapshot created at frame ${this.currentFrame}`);
    }

    restoreMemorySnapshot(frame) {
        const snapshot = this.memorySnapshots.get(frame);
        if (snapshot) {
            console.log(`🔄 Restoring memory from frame ${frame}`);
            return snapshot;
        }
        return null;
    }

    addTemporalEvent(triggerFrame, event) {
        this.temporalEvents.push({
            triggerFrame,
            event,
            triggered: false,
            id: Date.now() + Math.random()
        });
        console.log(`⚡ Temporal event scheduled for frame ${triggerFrame}`);
    }

    processTemporalEvents() {
        this.temporalEvents.forEach(te => {
            if (!te.triggered && this.currentFrame >= te.triggerFrame) {
                te.triggered = true;
                console.log(`🌀 Triggering temporal event: ${te.event.type}`);
                if (typeof te.event.execute === 'function') {
                    te.event.execute();
                }
            }
        });
    }
}

class CodexRule {
    constructor(name, description, validator) {
        this.name = name;
        this.description = description;
        this.validator = validator;
        this.violations = 0;
        this.lastCheck = 0;
    }

    validate(subject) {
        this.lastCheck = Date.now();
        try {
            const result = this.validator(subject);
            if (!result) {
                this.violations++;
                console.log(`❌ Codex Rule violated: ${this.name}`);
                return false;
            }
            return true;
        } catch (error) {
            console.error(`⚠️ Codex Rule error in ${this.name}:`, error);
            return false;
        }
    }
}

class CodexEngine {
    constructor() {
        this.rules = new Map();
        this.globalViolations = 0;
        this.validationHistory = [];
    }

    addRule(rule) {
        this.rules.set(rule.name, rule);
        console.log(`📜 Codex rule added: ${rule.name}`);
    }

    validate(subject, ruleName = null) {
        const rulesToCheck = ruleName ? [this.rules.get(ruleName)] : Array.from(this.rules.values());
        let allValid = true;
        const results = [];

        for (const rule of rulesToCheck.filter(r => r)) {
            const isValid = rule.validate(subject);
            results.push({ rule: rule.name, valid: isValid });
            if (!isValid) {
                allValid = false;
                this.globalViolations++;
            }
        }

        this.validationHistory.push({
            timestamp: Date.now(),
            subject: subject.constructor.name || 'Unknown',
            results,
            overallValid: allValid
        });

        // Keep only last 1000 validations
        if (this.validationHistory.length > 1000) {
            this.validationHistory.shift();
        }

        return allValid;
    }

    getViolationSummary() {
        const ruleStats = {};
        for (const [name, rule] of this.rules.entries()) {
            ruleStats[name] = {
                violations: rule.violations,
                lastCheck: rule.lastCheck
            };
        }
        return {
            globalViolations: this.globalViolations,
            ruleStats,
            totalValidations: this.validationHistory.length
        };
    }
}

//--------------------------------------
// Enhanced Glyph System
//--------------------------------------

class Glyph {
    constructor(name, type, tags, intensity, effect) {
        this.name = name;
        this.type = type; // "Emotional", "Mechanical", "Temporal", "Worldshift"
        this.tags = Array.isArray(tags) ? tags : [tags];
        this.intensity = Math.max(0, Math.min(1, intensity || 0.5));
        this.effect = effect;
        this.id = `glyph_${Date.now()}_${Math.floor(Math.random() * 1000)}`;
        this.activationCount = 0;
        this.lastActivation = 0;
        this.metadata = {
            created: Date.now(),
            energy: 1.0,
            mutations: []
        };
    }

    activate(context = {}) {
        this.activationCount++;
        this.lastActivation = Date.now();

        console.log(`✨ Activating Glyph: ${this.name} (${this.type})`);

        if (typeof this.effect === 'function') {
            try {
                const result = this.effect(context);
                console.log(`🔮 Glyph ${this.name} effect applied`);
                return result;
            } catch (error) {
                console.error(`⚠️ Glyph ${this.name} effect failed:`, error);
                return null;
            }
        }

        return { glyph: this.name, activated: true, context };
    }

    mutate(mutationType = 'random') {
        const oldIntensity = this.intensity;

        switch (mutationType) {
            case 'amplify':
                this.intensity = Math.min(1, this.intensity * 1.2);
                break;
            case 'dampen':
                this.intensity = Math.max(0, this.intensity * 0.8);
                break;
            case 'chaos':
                this.intensity = Math.random();
                break;
            default:
                this.intensity = Math.max(0, Math.min(1, this.intensity + (Math.random() - 0.5) * 0.2));
        }

        this.metadata.mutations.push({
            timestamp: Date.now(),
            type: mutationType,
            oldIntensity,
            newIntensity: this.intensity
        });

        console.log(`🧬 Glyph ${this.name} mutated: ${oldIntensity.toFixed(3)} → ${this.intensity.toFixed(3)}`);
        return this;
    }

    getSymbol() {
        const symbols = {
            Emotional: ['❤️', '💫', '🌸', '⭐', '🌙'],
            Mechanical: ['⚙️', '🔧', '⚡', '🔩', '🛠️'],
            Temporal: ['⏰', '🌀', '⏳', '🔄', '⚰️'],
            Worldshift: ['🌍', '💥', '🌊', '🏔️', '🔥']
        };
        const typeSymbols = symbols[this.type] || ['🔮'];
        return typeSymbols[Math.floor(Math.random() * typeSymbols.length)];
    }
}

class GlyphRegistry {
    constructor() {
        this.glyphs = new Map();
        this.categories = new Map();
        this.totalActivations = 0;
    }

    register(glyph) {
        this.glyphs.set(glyph.id, glyph);

        if (!this.categories.has(glyph.type)) {
            this.categories.set(glyph.type, []);
        }
        this.categories.get(glyph.type).push(glyph.id);

        console.log(`📋 Glyph registered: ${glyph.name} (${glyph.id})`);
        return glyph.id;
    }

    activate(glyphId, context = {}) {
        const glyph = this.glyphs.get(glyphId);
        if (!glyph) {
            console.warn(`⚠️ Glyph not found: ${glyphId}`);
            return null;
        }

        this.totalActivations++;
        return glyph.activate(context);
    }

    getByType(type) {
        const categoryIds = this.categories.get(type) || [];
        return categoryIds.map(id => this.glyphs.get(id)).filter(g => g);
    }

    getAll() {
        return Array.from(this.glyphs.values());
    }

    search(query) {
        const results = [];
        const queryLower = query.toLowerCase();

        for (const glyph of this.glyphs.values()) {
            if (glyph.name.toLowerCase().includes(queryLower) ||
                glyph.tags.some(tag => tag.toLowerCase().includes(queryLower))) {
                results.push(glyph);
            }
        }

        return results;
    }
}

//--------------------------------------
// 3D Vector and Geometry System (Enhanced from your original)
//--------------------------------------

class Vector3 {
    constructor(x = 0, y = 0, z = 0) {
        this.x = x;
        this.y = y;
        this.z = z;
    }

    add(v) {
        return new Vector3(this.x + v.x, this.y + v.y, this.z + v.z);
    }

    subtract(v) {
        return new Vector3(this.x - v.x, this.y - v.y, this.z - v.z);
    }

    multiply(scalar) {
        return new Vector3(this.x * scalar, this.y * scalar, this.z * scalar);
    }

    dot(v) {
        return this.x * v.x + this.y * v.y + this.z * v.z;
    }

    cross(v) {
        return new Vector3(
            this.y * v.z - this.z * v.y,
            this.z * v.x - this.x * v.z,
            this.x * v.y - this.y * v.x
        );
    }

    magnitude() {
        return Math.sqrt(this.x * this.x + this.y * this.y + this.z * this.z);
    }

    normalize() {
        const mag = this.magnitude();
        if (mag === 0) return new Vector3(0, 0, 0);
        return new Vector3(this.x / mag, this.y / mag, this.z / mag);
    }

    distanceTo(v) {
        return this.subtract(v).magnitude();
    }

    clone() {
        return new Vector3(this.x, this.y, this.z);
    }
}

class WorldObject {
    constructor(position = new Vector3(), type = 'generic') {
        this.position = position;
        this.type = type;
        this.id = `obj_${Date.now()}_${Math.floor(Math.random() * 1000)}`;
        this.velocity = new Vector3();
        this.scale = new Vector3(1, 1, 1);
        this.metadata = {};
        this.connections = new Set();
        this.lastUpdate = Date.now();
    }

    update(deltaTime) {
        this.position = this.position.add(this.velocity.multiply(deltaTime));
        this.lastUpdate = Date.now();
    }

    connectTo(otherObject) {
        this.connections.add(otherObject.id);
        otherObject.connections.add(this.id);
    }

    distanceTo(otherObject) {
        return this.position.distanceTo(otherObject.position);
    }
}

class TerrainNode extends WorldObject {
    constructor(position, glyphId = null, biome = 'grassland') {
        super(position, 'terrain');
        this.glyphId = glyphId;
        this.biome = biome; // grassland, desert, mountain, crystalline, void
        this.elevation = 0;
        this.moisture = Math.random();
        this.temperature = Math.random();
        this.fertility = Math.random();
        this.decorations = [];
        this.energyLevel = 1.0;
    }

    updateFromGlyph(glyph) {
        if (!glyph) return;

        this.elevation = glyph.intensity * 10;
        this.energyLevel = glyph.metadata.energy || 1.0;

        // Biome determination based on glyph type
        if (glyph.type === 'Worldshift') {
            this.biome = 'crystalline';
        } else if (glyph.type === 'Temporal') {
            this.biome = 'void';
        } else if (glyph.type === 'Emotional') {
            this.biome = this.moisture > 0.6 ? 'lush' : 'grassland';
        } else {
            this.biome = 'desert';
        }

        // Adjust properties based on tags
        if (glyph.tags.includes('memory')) {
            this.decorations.push('memory_crystal');
        }
        if (glyph.tags.includes('echo')) {
            this.decorations.push('echo_spire');
        }

        console.log(`🌍 Terrain node updated: ${this.biome}, elevation: ${this.elevation.toFixed(1)}`);
    }

    describe() {
        return `🧱 ${this.id} | Biome: ${this.biome} | Elev: ${this.elevation.toFixed(2)} | Moisture: ${this.moisture.toFixed(2)} | Energy: ${this.energyLevel.toFixed(2)}`;
    }
}

//--------------------------------------
// World Engine Integration
//--------------------------------------

class VectorLabWorldEngine {
    constructor(canvasId) {
        this.canvas = document.getElementById(canvasId);
        this.ctx = this.canvas ? this.canvas.getContext('2d') : null;

        // Core engines
        this.heartEngine = new HeartEngine();
        this.timelineEngine = new TimelineEngine();
        this.codexEngine = new CodexEngine();
        this.glyphRegistry = new GlyphRegistry();

        // World state
        this.worldObjects = new Map();
        this.terrainNodes = new Map();
        this.environmentalEvents = [];
        this.goldStrings = new Map(); // Connections between nodes

        // Rendering
        this.camera = {
            position: new Vector3(0, 0, 10),
            target: new Vector3(0, 0, 0),
            fov: 60
        };

        this.isRunning = false;
        this.lastFrameTime = 0;

        this.initializeDefaultRules();
        this.initializeDefaultGlyphs();

        console.log('🌍 VectorLab World Engine initialized');
    }

    initializeDefaultRules() {
        // Heart resonance validation
        this.codexEngine.addRule(new CodexRule(
            'Heart Resonance Bounds',
            'Heart resonance must stay within valid bounds',
            (heart) => heart.resonance >= 0 && heart.resonance <= heart.resonanceCap
        ));

        // Glyph intensity validation
        this.codexEngine.addRule(new CodexRule(
            'Glyph Intensity Bounds',
            'Glyph intensity must be between 0 and 1',
            (glyph) => glyph.intensity >= 0 && glyph.intensity <= 1
        ));

        // Timeline coherence
        this.codexEngine.addRule(new CodexRule(
            'Timeline Progression',
            'Timeline must progress forward when playing',
            (timeline) => !timeline.isPlaying || timeline.currentFrame >= 0
        ));

        // World object validation
        this.codexEngine.addRule(new CodexRule(
            'World Object Integrity',
            'World objects must have valid positions',
            (obj) => obj.position && !isNaN(obj.position.x) && !isNaN(obj.position.y) && !isNaN(obj.position.z)
        ));
    }

    initializeDefaultGlyphs() {
        // Create the core glyphs from your system
        const coreGlyphs = [
            new Glyph('Soul Thread', 'Emotional', ['bind', 'memory', 'character'], 0.85, (ctx) => {
                console.log('🧵 Soul Thread links player essence to scene memory');
                this.createMemoryEcho(ctx.position || new Vector3());
                return { type: 'memory_link', strength: 0.85 };
            }),

            new Glyph('Echo Pulse', 'Mechanical', ['radiate', 'trigger', 'cue'], 0.75, (ctx) => {
                console.log('🔊 Echo Pulse radiates across linked glyphs');
                this.heartEngine.echo();
                this.propagateEnergyWave(ctx.position || new Vector3(), 5.0);
                return { type: 'chain_reaction', radius: 5.0 };
            }),

            new Glyph('Golden Return', 'Temporal', ['flashback', 'recall', 'anchor'], 0.9, (ctx) => {
                console.log('✨ Golden Return restores prior world state');
                const restored = this.timelineEngine.restoreMemorySnapshot(this.timelineEngine.currentFrame - 10);
                return { type: 'temporal_restore', snapshot: restored };
            }),

            new Glyph('Fracture Point', 'Worldshift', ['collapse', 'split', 'choice'], 0.95, (ctx) => {
                console.log('💥 Fracture Point creates timeline branch');
                this.createTimelineBranch(ctx.choice || 'default');
                return { type: 'reality_split', branches: 2 };
            }),

            new Glyph('Aether Lock', 'Temporal', ['memory', 'echo', 'freeze'], 0.85, (ctx) => {
                console.log('🔒 Aether Lock freezes memory zones');
                this.freezeMemoryZones(3.0);
                return { type: 'temporal_freeze', duration: 3.0 };
            }),

            new Glyph('Glyph of Convergence', 'Worldshift', ['unity', 'repetition'], 0.7, (ctx) => {
                console.log('🜂 Unity in repetition');
                this.convergeNearbyNodes(ctx.position || new Vector3(), 2.0);
                return { type: 'convergence', radius: 2.0 };
            })
        ];

        coreGlyphs.forEach(glyph => this.glyphRegistry.register(glyph));
    }

    // Core world manipulation methods
    createMemoryEcho(position) {
        const echo = new WorldObject(position, 'memory_echo');
        echo.metadata.resonance = this.heartEngine.resonance;
        echo.metadata.timestamp = Date.now();
        this.worldObjects.set(echo.id, echo);
        console.log(`📞 Memory echo created at (${position.x}, ${position.y}, ${position.z})`);
    }

    propagateEnergyWave(center, radius) {
        for (const [id, obj] of this.worldObjects.entries()) {
            const distance = obj.position.distanceTo(center);
            if (distance <= radius) {
                const intensity = 1 - (distance / radius);
                obj.velocity = obj.velocity.add(
                    obj.position.subtract(center).normalize().multiply(intensity * 2)
                );
                console.log(`⚡ Energy wave affects ${obj.id} with intensity ${intensity.toFixed(2)}`);
            }
        }
    }

    createTimelineBranch(choice) {
        const currentState = {
            frame: this.timelineEngine.currentFrame,
            worldObjects: Array.from(this.worldObjects.entries()),
            terrainNodes: Array.from(this.terrainNodes.entries()),
            choice: choice
        };

        this.timelineEngine.createMemorySnapshot(currentState);
        console.log(`🌳 Timeline branch created: ${choice}`);
    }

    freezeMemoryZones(duration) {
        const memoryObjects = Array.from(this.worldObjects.values())
            .filter(obj => obj.type === 'memory_echo');

        memoryObjects.forEach(obj => {
            obj.velocity = new Vector3(); // Stop movement
            obj.metadata.frozen = Date.now() + (duration * 1000);
        });

        console.log(`❄️ Froze ${memoryObjects.length} memory zones for ${duration}s`);
    }

    convergeNearbyNodes(center, radius) {
        const nearbyTerrain = Array.from(this.terrainNodes.values())
            .filter(node => node.position.distanceTo(center) <= radius);

        nearbyTerrain.forEach(node => {
            const direction = center.subtract(node.position).normalize().multiply(0.1);
            node.position = node.position.add(direction);
        });

        console.log(`🧲 Converged ${nearbyTerrain.length} terrain nodes`);
    }

    // Terrain generation from glyphs
    generateTerrainFromGlyphs() {
        this.terrainNodes.clear();

        const glyphs = this.glyphRegistry.getAll();
        glyphs.forEach((glyph, index) => {
            // Create terrain in a grid pattern
            const x = (index % 8) * 2 - 8;
            const z = Math.floor(index / 8) * 2 - 8;
            const position = new Vector3(x, 0, z);

            const terrain = new TerrainNode(position, glyph.id);
            terrain.updateFromGlyph(glyph);

            this.terrainNodes.set(terrain.id, terrain);
        });

        console.log(`🗺️ Generated ${this.terrainNodes.size} terrain nodes from glyphs`);
    }

    // Rendering system
    render() {
        if (!this.ctx) return;

        const ctx = this.ctx;
        const width = this.canvas.width;
        const height = this.canvas.height;

        // Clear canvas
        ctx.fillStyle = '#0b0e14';
        ctx.fillRect(0, 0, width, height);

        // Draw coordinate system
        this.drawCoordinateSystem(ctx, width, height);

        // Draw heart resonance visualization
        this.drawHeartResonance(ctx, width, height);

        // Draw terrain nodes
        this.drawTerrainNodes(ctx, width, height);

        // Draw world objects
        this.drawWorldObjects(ctx, width, height);

        // Draw environmental effects
        this.drawEnvironmentalEffects(ctx, width, height);

        // Draw timeline indicator
        this.drawTimelineIndicator(ctx, width, height);

        // Draw glyph status
        this.drawGlyphStatus(ctx, width, height);
    }

    drawCoordinateSystem(ctx, width, height) {
        const centerX = width / 2;
        const centerY = height / 2;

        ctx.strokeStyle = '#1e2b46';
        ctx.lineWidth = 1;

        // Grid lines
        for (let i = -10; i <= 10; i++) {
            const x = centerX + i * 20;
            const y = centerY + i * 20;

            if (x >= 0 && x <= width) {
                ctx.beginPath();
                ctx.moveTo(x, 0);
                ctx.lineTo(x, height);
                ctx.stroke();
            }

            if (y >= 0 && y <= height) {
                ctx.beginPath();
                ctx.moveTo(0, y);
                ctx.lineTo(width, y);
                ctx.stroke();
            }
        }

        // Axes
        ctx.strokeStyle = '#54f0b8';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(centerX, 0);
        ctx.lineTo(centerX, height);
        ctx.moveTo(0, centerY);
        ctx.lineTo(width, centerY);
        ctx.stroke();
    }

    drawHeartResonance(ctx, width, height) {
        const centerX = width / 2;
        const centerY = height / 2;
        const resonance = this.heartEngine.resonance;
        const pulseRadius = 20 + resonance * 30;

        // Heart core
        ctx.beginPath();
        ctx.arc(centerX, centerY, 8, 0, Math.PI * 2);
        ctx.fillStyle = `rgba(255, 105, 180, ${resonance})`;
        ctx.fill();

        // Resonance rings
        for (let i = 1; i <= 3; i++) {
            ctx.beginPath();
            ctx.arc(centerX, centerY, pulseRadius * i, 0, Math.PI * 2);
            ctx.strokeStyle = `rgba(255, 105, 180, ${resonance / i})`;
            ctx.lineWidth = 2;
            ctx.stroke();
        }

        // Heart state indicator
        ctx.fillStyle = '#e6f0ff';
        ctx.font = '12px ui-monospace';
        ctx.fillText(`💓 ${this.heartEngine.state} (${resonance.toFixed(3)})`, 10, 20);
        ctx.fillText(`Emotional: ${this.heartEngine.getEmotionalState()}`, 10, 35);
    }

    drawTerrainNodes(ctx, width, height) {
        const centerX = width / 2;
        const centerY = height / 2;
        const scale = 20;

        for (const [id, terrain] of this.terrainNodes.entries()) {
            const screenX = centerX + terrain.position.x * scale;
            const screenY = centerY + terrain.position.z * scale;
            const size = 8 + terrain.elevation;

            // Biome colors
            const biomeColors = {
                grassland: '#90EE90',
                desert: '#F4A460',
                mountain: '#A9A9A9',
                crystalline: '#40E0D0',
                void: '#4B0082',
                lush: '#32CD32'
            };

            ctx.fillStyle = biomeColors[terrain.biome] || '#888';
            ctx.beginPath();
            ctx.rect(screenX - size / 2, screenY - size / 2, size, size);
            ctx.fill();

            // Energy glow
            if (terrain.energyLevel > 1) {
                ctx.shadowColor = ctx.fillStyle;
                ctx.shadowBlur = terrain.energyLevel * 5;
                ctx.fill();
                ctx.shadowBlur = 0;
            }

            // Decorations
            if (terrain.decorations.length > 0) {
                ctx.fillStyle = '#FFD700';
                ctx.beginPath();
                ctx.arc(screenX, screenY - size, 2, 0, Math.PI * 2);
                ctx.fill();
            }
        }
    }

    drawWorldObjects(ctx, width, height) {
        const centerX = width / 2;
        const centerY = height / 2;
        const scale = 20;

        for (const [id, obj] of this.worldObjects.entries()) {
            const screenX = centerX + obj.position.x * scale;
            const screenY = centerY + obj.position.z * scale;

            if (obj.type === 'memory_echo') {
                const alpha = obj.metadata.resonance || 0.5;
                ctx.fillStyle = `rgba(147, 197, 253, ${alpha})`;
                ctx.beginPath();
                ctx.arc(screenX, screenY, 6, 0, Math.PI * 2);
                ctx.fill();

                // Echo ripple
                const age = (Date.now() - obj.metadata.timestamp) / 1000;
                if (age < 2) {
                    const rippleRadius = age * 20;
                    ctx.strokeStyle = `rgba(147, 197, 253, ${1 - age / 2})`;
                    ctx.lineWidth = 2;
                    ctx.beginPath();
                    ctx.arc(screenX, screenY, rippleRadius, 0, Math.PI * 2);
                    ctx.stroke();
                }
            }
        }
    }

    drawEnvironmentalEffects(ctx, width, height) {
        // Draw any active environmental events
        this.environmentalEvents.forEach((event, index) => {
            if (event.duration <= 0) {
                this.environmentalEvents.splice(index, 1);
                return;
            }

            const centerX = width / 2 + event.center[0] * 20;
            const centerY = height / 2 + event.center[2] * 20;
            const alpha = event.duration / event.maxDuration;

            if (event.type === 'storm') {
                ctx.strokeStyle = `rgba(128, 0, 128, ${alpha})`;
                ctx.lineWidth = 3;
                for (let i = 0; i < 8; i++) {
                    const angle = (i / 8) * Math.PI * 2;
                    const x1 = centerX + Math.cos(angle) * 15;
                    const y1 = centerY + Math.sin(angle) * 15;
                    const x2 = centerX + Math.cos(angle) * 25;
                    const y2 = centerY + Math.sin(angle) * 25;

                    ctx.beginPath();
                    ctx.moveTo(x1, y1);
                    ctx.lineTo(x2, y2);
                    ctx.stroke();
                }
            }

            event.duration--;
        });
    }

    drawTimelineIndicator(ctx, width, height) {
        const frame = this.timelineEngine.currentFrame;
        const isPlaying = this.timelineEngine.isPlaying;

        ctx.fillStyle = '#bfeaff';
        ctx.font = '12px ui-monospace';
        ctx.fillText(`⏰ Frame: ${frame} ${isPlaying ? '▶️' : '⏸️'}`, width - 120, 20);

        // Timeline bar
        const barWidth = 100;
        const barHeight = 4;
        const barX = width - 120;
        const barY = 30;

        ctx.fillStyle = '#1e2b46';
        ctx.fillRect(barX, barY, barWidth, barHeight);

        const progress = (frame % 1000) / 1000;
        ctx.fillStyle = '#54f0b8';
        ctx.fillRect(barX, barY, barWidth * progress, barHeight);
    }

    drawGlyphStatus(ctx, width, height) {
        const glyphs = this.glyphRegistry.getAll().slice(0, 6); // Show first 6 glyphs

        ctx.fillStyle = '#e6f0ff';
        ctx.font = '11px ui-monospace';
        ctx.fillText('Active Glyphs:', 10, height - 80);

        glyphs.forEach((glyph, index) => {
            const y = height - 65 + index * 12;
            const symbol = glyph.getSymbol();
            const intensity = (glyph.intensity * 100).toFixed(0);
            const color = glyph.intensity > 0.7 ? '#54f0b8' : glyph.intensity > 0.4 ? '#9ab0d6' : '#666';

            ctx.fillStyle = color;
            ctx.fillText(`${symbol} ${glyph.name} (${intensity}%)`, 10, y);
        });
    }

    // Main update loop
    update(currentTime) {
        const deltaTime = currentTime - this.lastFrameTime;
        this.lastFrameTime = currentTime;

        // Update core engines
        this.timelineEngine.update();
        this.heartEngine.decay(0.001); // Slow natural decay

        // Update world objects
        for (const [id, obj] of this.worldObjects.entries()) {
            obj.update(deltaTime / 1000);

            // Unfreeze memory zones
            if (obj.metadata.frozen && Date.now() > obj.metadata.frozen) {
                delete obj.metadata.frozen;
            }
        }

        // Validate systems
        this.codexEngine.validate(this.heartEngine, 'Heart Resonance Bounds');
        this.codexEngine.validate(this.timelineEngine, 'Timeline Progression');

        // Auto-generate terrain periodically
        if (this.timelineEngine.currentFrame % 300 === 0) {
            this.generateTerrainFromGlyphs();
        }

        // Render everything
        this.render();
    }

    // Public API for external control
    start() {
        if (this.isRunning) return;
        this.isRunning = true;
        this.lastFrameTime = performance.now();

        const animate = (currentTime) => {
            if (this.isRunning) {
                this.update(currentTime);
                requestAnimationFrame(animate);
            }
        };

        requestAnimationFrame(animate);
        console.log('🎬 VectorLab World Engine started');
    }

    stop() {
        this.isRunning = false;
        console.log('⏹️ VectorLab World Engine stopped');
    }

    // Glyph Forge Interface
    activateGlyph(glyphName, context = {}) {
        const results = this.glyphRegistry.search(glyphName);
        if (results.length === 0) {
            console.warn(`⚠️ Glyph not found: ${glyphName}`);
            return null;
        }

        const glyph = results[0];
        this.heartEngine.pulse(glyph.intensity * 0.1);
        return this.glyphRegistry.activate(glyph.id, context);
    }

    createCustomGlyph(name, type, tags, intensity, effectCode) {
        try {
            // Create effect function from code string
            const effectFunction = new Function('context', effectCode);
            const glyph = new Glyph(name, type, tags, intensity, effectFunction);
            const glyphId = this.glyphRegistry.register(glyph);

            console.log(`🎨 Custom glyph created: ${name}`);
            return glyphId;
        } catch (error) {
            console.error('⚠️ Failed to create custom glyph:', error);
            return null;
        }
    }

    spawnEnvironmentalEvent(type, center, intensity, duration) {
        const event = {
            type,
            center: Array.isArray(center) ? center : [center.x || 0, center.y || 0, center.z || 0],
            intensity,
            duration,
            maxDuration: duration,
            id: Date.now() + Math.random()
        };

        this.environmentalEvents.push(event);
        console.log(`🌪️ Environmental event spawned: ${type} at [${event.center.join(', ')}]`);
        return event.id;
    }

    // Export/Import world state
    exportWorldState() {
        return {
            timestamp: Date.now(),
            heartEngine: {
                resonance: this.heartEngine.resonance,
                state: this.heartEngine.state,
                pulseHistory: this.heartEngine.pulseHistory.slice(-10)
            },
            timeline: {
                currentFrame: this.timelineEngine.currentFrame,
                isPlaying: this.timelineEngine.isPlaying
            },
            glyphs: this.glyphRegistry.getAll().map(g => ({
                id: g.id,
                name: g.name,
                type: g.type,
                intensity: g.intensity,
                activationCount: g.activationCount
            })),
            terrainNodes: Array.from(this.terrainNodes.values()).map(t => ({
                id: t.id,
                position: t.position,
                biome: t.biome,
                elevation: t.elevation
            })),
            codexStats: this.codexEngine.getViolationSummary()
        };
    }
}

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        VectorLabWorldEngine,
        HeartEngine,
        TimelineEngine,
        CodexEngine,
        Glyph,
        GlyphRegistry,
        Vector3,
        WorldObject,
        TerrainNode
    };
}
