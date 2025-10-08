//=============================================================================
// UNIFIED QUANTUM ENGINE (Enhanced from glyph_forge.html)
//=============================================================================

class UnifiedQuantumEngine {
    constructor() {
        this.particles = [];
        this.glyphs = [];
        this.beatSystem = null;
        this.effects = [];
        this.canvas = null;
        this.ctx = null;
    }

    connectBeatSystem(beatSystem) {
        this.beatSystem = beatSystem;
    }

    initialize(canvas, ctx) {
        this.canvas = canvas;
        this.ctx = ctx;

        // Initialize quantum particles
        this.generateQuantumField();

        console.log('🌌 Quantum Engine initialized');
    }

    update(deltaTime) {
        // Update particles
        this.particles.forEach(particle => {
            this.updateParticle(particle, deltaTime);
        });

        // Update effects
        this.effects = this.effects.filter(effect => {
            effect.update(deltaTime);
            return effect.alive;
        });

        // Generate beat-synchronized effects
        if (this.beatSystem && this.beatSystem.isBeat()) {
            this.generateBeatEffect();
        }
    }

    updateParticle(particle, deltaTime) {
        particle.x += particle.vx * deltaTime;
        particle.y += particle.vy * deltaTime;
        particle.life -= deltaTime;

        // Wrap around screen
        if (particle.x < 0) particle.x = this.canvas.width;
        if (particle.x > this.canvas.width) particle.x = 0;
        if (particle.y < 0) particle.y = this.canvas.height;
        if (particle.y > this.canvas.height) particle.y = 0;

        // Remove dead particles
        if (particle.life <= 0) {
            particle.life = particle.maxLife;
            particle.alpha = 1.0;
        }

        particle.alpha = particle.life / particle.maxLife;
    }

    generateQuantumField() {
        const particleCount = 200;

        for (let i = 0; i < particleCount; i++) {
            this.particles.push({
                x: Math.random() * this.canvas.width,
                y: Math.random() * this.canvas.height,
                vx: (Math.random() - 0.5) * 20,
                vy: (Math.random() - 0.5) * 20,
                size: Math.random() * 2 + 1,
                color: this.getQuantumColor(),
                life: Math.random() * 5 + 2,
                maxLife: Math.random() * 5 + 2,
                alpha: 1.0
            });
        }
    }

    getQuantumColor() {
        const colors = ['#00ff7f', '#ff69b4', '#9d4edd', '#f77f00', '#06ffa5'];
        return colors[Math.floor(Math.random() * colors.length)];
    }

    generateBeatEffect() {
        this.effects.push({
            x: Math.random() * this.canvas.width,
            y: Math.random() * this.canvas.height,
            size: 0,
            maxSize: 100,
            life: 2.0,
            maxLife: 2.0,
            color: '#28F49B',
            alive: true,
            update(deltaTime) {
                this.life -= deltaTime;
                this.size = (1 - this.life / this.maxLife) * this.maxSize;
                this.alive = this.life > 0;
            }
        });
    }

    render(ctx) {
        ctx.save();

        // Render quantum field
        this.particles.forEach(particle => {
            ctx.globalAlpha = particle.alpha * 0.6;
            ctx.fillStyle = particle.color;
            ctx.beginPath();
            ctx.arc(particle.x, particle.y, particle.size, 0, Math.PI * 2);
            ctx.fill();
        });

        // Render effects
        this.effects.forEach(effect => {
            ctx.globalAlpha = effect.life / effect.maxLife;
            ctx.strokeStyle = effect.color;
            ctx.lineWidth = 2;
            ctx.beginPath();
            ctx.arc(effect.x, effect.y, effect.size, 0, Math.PI * 2);
            ctx.stroke();
        });

        ctx.restore();
    }

    renderGlyphs(ctx) {
        // Render glyph visualization mode
        ctx.fillStyle = '#1a1a2e';
        ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);

        // Draw glyph grid
        this.drawGlyphGrid(ctx);

        // Render active glyphs
        this.glyphs.forEach(glyph => {
            this.renderGlyph(ctx, glyph);
        });
    }

    drawGlyphGrid(ctx) {
        const gridSize = 50;
        ctx.strokeStyle = 'rgba(0, 255, 127, 0.1)';
        ctx.lineWidth = 1;

        for (let x = 0; x < this.canvas.width; x += gridSize) {
            ctx.beginPath();
            ctx.moveTo(x, 0);
            ctx.lineTo(x, this.canvas.height);
            ctx.stroke();
        }

        for (let y = 0; y < this.canvas.height; y += gridSize) {
            ctx.beginPath();
            ctx.moveTo(0, y);
            ctx.lineTo(this.canvas.width, y);
            ctx.stroke();
        }
    }

    renderGlyph(ctx, glyph) {
        ctx.save();
        ctx.translate(glyph.x, glyph.y);
        ctx.rotate(glyph.rotation);

        ctx.strokeStyle = glyph.color;
        ctx.lineWidth = 3;
        ctx.fillStyle = glyph.color + '40';

        // Draw glyph shape based on type
        switch (glyph.type) {
            case 'circle':
                ctx.beginPath();
                ctx.arc(0, 0, glyph.size, 0, Math.PI * 2);
                ctx.stroke();
                ctx.fill();
                break;
            case 'triangle':
                ctx.beginPath();
                ctx.moveTo(0, -glyph.size);
                ctx.lineTo(-glyph.size * 0.866, glyph.size * 0.5);
                ctx.lineTo(glyph.size * 0.866, glyph.size * 0.5);
                ctx.closePath();
                ctx.stroke();
                ctx.fill();
                break;
            case 'square':
                ctx.strokeRect(-glyph.size / 2, -glyph.size / 2, glyph.size, glyph.size);
                ctx.fillRect(-glyph.size / 2, -glyph.size / 2, glyph.size, glyph.size);
                break;
        }

        ctx.restore();
    }

    handleClick(x, y) {
        // Create new glyph at click position
        this.createGlyph(x, y);
    }

    handleMouseMove(x, y) {
        // Add interactive particle effects
        this.particles.push({
            x: x,
            y: y,
            vx: (Math.random() - 0.5) * 50,
            vy: (Math.random() - 0.5) * 50,
            size: Math.random() * 3 + 1,
            color: '#00ff7f',
            life: 1.0,
            maxLife: 1.0,
            alpha: 1.0
        });

        // Keep particle count manageable
        if (this.particles.length > 300) {
            this.particles.splice(0, 50);
        }
    }

    createGlyph(x, y) {
        const glyph = {
            id: `glyph_${Date.now()}`,
            x: x,
            y: y,
            size: 30 + Math.random() * 20,
            type: ['circle', 'triangle', 'square'][Math.floor(Math.random() * 3)],
            color: this.getQuantumColor(),
            rotation: Math.random() * Math.PI * 2,
            energy: 100
        };

        this.glyphs.push(glyph);
        console.log(`✨ Created glyph at ${x}, ${y}`);
        return glyph;
    }

    getGlyphAt(x, y) {
        return this.glyphs.find(glyph => {
            const dx = x - glyph.x;
            const dy = y - glyph.y;
            return Math.sqrt(dx * dx + dy * dy) < glyph.size;
        });
    }

    openQuantumEditor() {
        console.log('⚡ Opening Quantum Physics Editor');
    }

    openGlyphForge() {
        console.log('🔮 Opening Glyph Forge');
    }
}

//=============================================================================
// UNIFIED ASSET ENGINE (Enhanced from nexus_resource_demo.cpp concepts)
//=============================================================================

class UnifiedAssetEngine {
    constructor() {
        this.assets = new Map();
        this.loadQueue = [];
        this.memoryUsage = 0;
        this.maxMemory = 512; // MB
        this.lodSystem = new LODSystem();
    }

    initialize() {
        this.loadDefaultAssets();
        console.log('📦 Asset Engine initialized');
    }

    update(deltaTime) {
        // Process load queue
        this.processLoadQueue();

        // Update LOD system
        this.lodSystem.update(deltaTime);

        // Memory management
        this.manageMemory();
    }

    loadDefaultAssets() {
        const defaultAssets = [
            { id: 'tree_model', type: 'model', path: 'models/tree.glb', size: 2.5 },
            { id: 'rock_model', type: 'model', path: 'models/rock.glb', size: 1.2 },
            { id: 'grass_texture', type: 'texture', path: 'textures/grass.png', size: 0.8 },
            { id: 'water_shader', type: 'shader', path: 'shaders/water.glsl', size: 0.1 },
            { id: 'forest_ambient', type: 'audio', path: 'audio/forest.ogg', size: 5.2 }
        ];

        defaultAssets.forEach(asset => {
            this.loadAsset(asset);
        });
    }

    loadAsset(assetInfo) {
        if (this.assets.has(assetInfo.id)) {
            return this.assets.get(assetInfo.id);
        }

        const asset = {
            ...assetInfo,
            loaded: false,
            loadTime: Date.now(),
            references: 0,
            lastUsed: Date.now(),
            audioReactive: assetInfo.audioReactive || false,
            beatScale: 1.0
        };

        this.assets.set(assetInfo.id, asset);
        this.loadQueue.push(assetInfo.id);

        console.log(`📥 Queued asset: ${assetInfo.id}`);
        return asset;
    }

    processLoadQueue() {
        if (this.loadQueue.length === 0) return;

        // Simulate async loading - process one asset per frame
        const assetId = this.loadQueue.shift();
        const asset = this.assets.get(assetId);

        if (asset && !asset.loaded) {
            // Simulate loading time
            setTimeout(() => {
                asset.loaded = true;
                this.memoryUsage += asset.size;
                console.log(`✅ Loaded asset: ${assetId} (${asset.size}MB)`);
            }, 100);
        }
    }

    manageMemory() {
        if (this.memoryUsage > this.maxMemory) {
            this.unloadLeastRecentlyUsed();
        }
    }

    unloadLeastRecentlyUsed() {
        const sortedAssets = Array.from(this.assets.entries())
            .filter(([id, asset]) => asset.loaded && asset.references === 0)
            .sort(([, a], [, b]) => a.lastUsed - b.lastUsed);

        if (sortedAssets.length > 0) {
            const [id, asset] = sortedAssets[0];
            this.unloadAsset(id);
        }
    }

    unloadAsset(assetId) {
        const asset = this.assets.get(assetId);
        if (asset && asset.loaded) {
            asset.loaded = false;
            this.memoryUsage -= asset.size;
            console.log(`🗑️ Unloaded asset: ${assetId} (${asset.size}MB)`);
        }
    }

    getAsset(assetId) {
        const asset = this.assets.get(assetId);
        if (asset) {
            asset.lastUsed = Date.now();
            asset.references++;
            return asset;
        }
        return null;
    }

    releaseAsset(assetId) {
        const asset = this.assets.get(assetId);
        if (asset) {
            asset.references = Math.max(0, asset.references - 1);
        }
    }

    getMemoryUsage() {
        return Math.floor(this.memoryUsage);
    }

    renderGlyphAssets(ctx) {
        // Render assets in glyph view mode
        const assetPositions = this.calculateAssetLayout();

        ctx.fillStyle = '#28F49B';
        ctx.font = '12px monospace';

        assetPositions.forEach(({ asset, x, y }) => {
            const status = asset.loaded ? '✅' : '⏳';
            const text = `${status} ${asset.id} (${asset.size}MB)`;
            ctx.fillText(text, x, y);
        });
    }

    calculateAssetLayout() {
        const positions = [];
        let x = 50, y = 50;

        for (const [id, asset] of this.assets) {
            positions.push({ asset, x, y });
            y += 25;
            if (y > this.canvas?.height - 100) {
                y = 50;
                x += 250;
            }
        }

        return positions;
    }

    renderDebug(ctx) {
        // Debug asset information
        ctx.fillStyle = '#00ff7f';
        ctx.font = '12px monospace';

        const debugInfo = [
            `Assets: ${this.assets.size}`,
            `Loaded: ${Array.from(this.assets.values()).filter(a => a.loaded).length}`,
            `Queue: ${this.loadQueue.length}`,
            `Memory: ${this.memoryUsage.toFixed(1)}/${this.maxMemory}MB`
        ];

        debugInfo.forEach((info, index) => {
            ctx.fillText(info, 10, 120 + index * 15);
        });
    }

    exportAssets() {
        return {
            assets: Array.from(this.assets.entries()),
            memoryUsage: this.memoryUsage,
            loadQueue: [...this.loadQueue]
        };
    }

    openAssetManager() {
        console.log('📁 Opening Asset Manager');
    }
}

//=============================================================================
// LOD SYSTEM
//=============================================================================

class LODSystem {
    constructor() {
        this.lodLevels = new Map();
    }

    update(deltaTime) {
        // Update LOD calculations based on distance/performance
    }

    calculateLOD(distance, performance = 1.0) {
        if (distance < 100 * performance) return 'ultra';
        if (distance < 300 * performance) return 'high';
        if (distance < 600 * performance) return 'medium';
        if (distance < 1000 * performance) return 'low';
        return 'billboard';
    }
}

//=============================================================================
// UNIFIED GAME DIRECTOR (Game Intelligence System)
//=============================================================================

class UnifiedGameDirector {
    constructor() {
        this.engines = {};
        this.recommendations = [];
        this.analytics = {
            playerBehavior: {},
            performance: {},
            balance: {}
        };
        this.mode = 'development';
    }

    connectAllEngines(engines) {
        this.engines = engines;
        console.log('🎬 Game Director connected to all engines');
    }

    initialize() {
        console.log('🎯 Game Director initialized');
    }

    update(deltaTime) {
        // Analyze game state across all engines
        this.analyzeGameState();

        // Update recommendations
        this.updateRecommendations();
    }

    analyzeGameState() {
        if (!this.engines.world) return;

        // Get current game metrics
        const worldState = this.engines.world.getCurrentState();
        const aiInsights = this.engines.ai?.insights || [];

        // Update analytics
        this.analytics.playerBehavior = worldState.playerData || {};
        this.analytics.performance = worldState.performance || {};

        // Generate new insights
        this.generateDirectorInsights();
    }

    generateDirectorInsights() {
        // Example: Detect if certain areas are underutilized
        if (this.analytics.playerBehavior.avoidedAreas?.includes('desert')) {
            this.addRecommendation({
                type: 'content_enhancement',
                priority: 'medium',
                title: 'Enhance Desert Biome',
                description: 'Players avoid desert areas - add unique content and rewards',
                actions: [
                    'Add desert-specific creatures',
                    'Place valuable resources in desert',
                    'Create desert dungeons/caves',
                    'Add fast travel options'
                ],
                estimatedEffort: '1-2 weeks'
            });
        }
    }

    addRecommendation(recommendation) {
        recommendation.id = `rec_${Date.now()}`;
        recommendation.timestamp = Date.now();

        this.recommendations.unshift(recommendation);

        // Keep only recent recommendations
        this.recommendations = this.recommendations.slice(0, 20);

        console.log(`💡 Director Recommendation: ${recommendation.title}`);
    }

    updateRecommendations() {
        // Remove old recommendations
        const now = Date.now();
        this.recommendations = this.recommendations.filter(rec =>
            now - rec.timestamp < 24 * 60 * 60 * 1000 // Keep for 24 hours
        );
    }

    executeRecommendation(type) {
        console.log(`🎬 Executing recommendation: ${type}`);

        switch (type) {
            case 'balancing':
                this.executeBalanceChanges();
                break;
            case 'content':
                this.executeContentGeneration();
                break;
            case 'water':
                this.executeWaterEnhancements();
                break;
            default:
                console.log('Unknown recommendation type');
        }
    }

    executeBalanceChanges() {
        console.log('⚖️ Applying automatic balance adjustments');
        // In a real implementation, this would adjust game parameters
    }

    executeContentGeneration() {
        console.log('🎨 Generating new content');
        // In a real implementation, this would create new game content
    }

    executeWaterEnhancements() {
        console.log('🌊 Adding water area enhancements');
        // In a real implementation, this would enhance water-based content
    }

    activateGlyph(glyph) {
        console.log(`🔮 Activating glyph: ${glyph.id}`);

        // Apply glyph effects to game world
        if (this.engines.world) {
            this.engines.world.applyGlyphEffect(glyph);
        }
    }

    setMode(mode) {
        this.mode = mode;
        console.log(`🎯 Director mode: ${mode}`);
    }

    exportBuild() {
        console.log('📦 Exporting game build');
        const buildData = {
            world: this.engines.world?.exportWorld(),
            assets: this.engines.assets?.exportAssets(),
            ai: this.engines.ai?.exportIntelligence(),
            recommendations: this.recommendations,
            analytics: this.analytics
        };

        // In a real implementation, this would create deployable build
        console.log('✅ Build export complete');
        return buildData;
    }
}

//=============================================================================
// UNIFIED BEAT SYSTEM (Holy Beat System)
//=============================================================================

class UnifiedBeatSystem {
    constructor() {
        this.bpm = 128;
        this.petalCount = 12;
        this.lastBeat = 0;
        this.beatInterval = 60 / this.bpm; // seconds per beat
        this.currentTime = 0;
        this.beatIntensity = 0;
        this.isPlaying = true;
    }

    initialize() {
        console.log('🎵 Beat System initialized');
    }

    update(deltaTime) {
        this.currentTime += deltaTime;

        // Calculate beat timing
        const timeSinceLastBeat = (this.currentTime - this.lastBeat) % this.beatInterval;

        if (timeSinceLastBeat < deltaTime && this.isPlaying) {
            this.onBeat();
        }

        // Calculate beat intensity (peaks at beat, fades between)
        this.beatIntensity = Math.max(0, 1 - (timeSinceLastBeat / this.beatInterval) * 2);
    }

    onBeat() {
        this.lastBeat = this.currentTime;
        console.log(`🥁 Beat! BPM: ${this.bpm}`);
    }

    isBeat() {
        const timeSinceLastBeat = (this.currentTime - this.lastBeat) % this.beatInterval;
        return timeSinceLastBeat < 0.1; // Beat window of 0.1 seconds
    }

    getBeatIntensity() {
        return this.beatIntensity;
    }

    getCurrentBPM() {
        return this.bpm;
    }

    getPetalCount() {
        return this.petalCount;
    }

    setBPM(newBPM) {
        this.bpm = newBPM;
        this.beatInterval = 60 / this.bpm;
        console.log(`🎵 BPM changed to: ${newBPM}`);
    }

    setPetalCount(count) {
        this.petalCount = count;
        console.log(`🌸 Petal count: ${count}`);
    }

    togglePlayPause() {
        this.isPlaying = !this.isPlaying;
        console.log(`🎵 Beat system: ${this.isPlaying ? 'Playing' : 'Paused'}`);
    }

    renderBeats(ctx) {
        if (!this.isPlaying) return;

        // Render beat visualization
        const centerX = ctx.canvas.width - 50;
        const centerY = 50;
        const radius = 20 + this.beatIntensity * 10;

        ctx.save();
        ctx.globalAlpha = 0.8;
        ctx.strokeStyle = '#28F49B';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.arc(centerX, centerY, radius, 0, Math.PI * 2);
        ctx.stroke();

        // Render petal pattern
        const petalAngle = (Math.PI * 2) / this.petalCount;
        ctx.fillStyle = '#28F49B';

        for (let i = 0; i < this.petalCount; i++) {
            const angle = i * petalAngle + this.currentTime;
            const x = centerX + Math.cos(angle) * (radius * 0.7);
            const y = centerY + Math.sin(angle) * (radius * 0.7);

            ctx.beginPath();
            ctx.arc(x, y, 2 + this.beatIntensity * 2, 0, Math.PI * 2);
            ctx.fill();
        }

        ctx.restore();
    }

    openBeatEditor() {
        console.log('🎼 Opening Beat System Editor');
    }
}

// Export additional classes
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        ...module.exports,
        UnifiedQuantumEngine,
        UnifiedAssetEngine,
        UnifiedGameDirector,
        UnifiedBeatSystem,
        UnifiedAIEngine,
        UnifiedWorldEngine
    };
}
