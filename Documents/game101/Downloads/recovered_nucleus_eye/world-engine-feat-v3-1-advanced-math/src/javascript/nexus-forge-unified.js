/**
 * NEXUS FORGE UNIFIED - Open World Game Development Intelligence
 * ============================================================
 *
 * Combines all NEXUS Forge components into unified open-world game development system:
 * • AI Pattern Recognition Engine (from nexus_forge_primordial.js)
 * • Quantum Graphics Engine (from glyph_forge.html)
 * • World Generation System (from nexus_world_demo.cpp concepts)
 * • Asset Management System (from nexus_resource_demo.cpp)
 * • Game Intelligence Director
 * • Beat-synchronized systems
 *
 * This eliminates duplicates and creates comprehensive tooling for open-world game production.
 */

//=============================================================================
// CORE UNIFIED SYSTEM CLASS
//=============================================================================

class NexusForgeUnified {
    constructor() {
        this.initialized = false;
        this.canvas = null;
        this.ctx = null;
        this.animationId = null;

        // Unified engines
        this.aiEngine = new UnifiedAIEngine();
        this.worldEngine = new UnifiedWorldEngine();
        this.quantumEngine = new UnifiedQuantumEngine();
        this.assetEngine = new UnifiedAssetEngine();
        this.gameDirector = new UnifiedGameDirector();
        this.beatSystem = new UnifiedBeatSystem();

        // System state
        this.currentMode = 'development';
        this.currentViewMode = 'world';
        this.currentWorld = null;
        this.stats = {
            fps: 0,
            entityCount: 0,
            chunkCount: 0,
            memoryUsage: 0
        };

        this.setupEngineConnections();
    }

    setupEngineConnections() {
        // Connect engines for data sharing
        this.aiEngine.connectWorldEngine(this.worldEngine);
        this.aiEngine.connectQuantumEngine(this.quantumEngine);
        this.worldEngine.connectAssetEngine(this.assetEngine);
        this.worldEngine.connectBeatSystem(this.beatSystem);
        this.quantumEngine.connectBeatSystem(this.beatSystem);
        this.gameDirector.connectAllEngines({
            ai: this.aiEngine,
            world: this.worldEngine,
            quantum: this.quantumEngine,
            assets: this.assetEngine,
            beat: this.beatSystem
        });
    }

    initialize() {
        if (this.initialized) return;

        this.canvas = document.getElementById('unifiedWorldCanvas');
        if (!this.canvas) {
            console.error('Canvas element not found');
            return;
        }

        this.ctx = this.canvas.getContext('2d');
        this.resizeCanvas();

        // Initialize all engines
        this.aiEngine.initialize();
        this.worldEngine.initialize(this.canvas, this.ctx);
        this.quantumEngine.initialize(this.canvas, this.ctx);
        this.assetEngine.initialize();
        this.gameDirector.initialize();
        this.beatSystem.initialize();

        // Start systems
        this.startMainLoop();
        this.startStatsUpdater();
        this.loadDefaultWorld();

        // Event listeners
        this.setupEventListeners();

        this.initialized = true;
        console.log('🌟 NEXUS FORGE UNIFIED initialized successfully');
    }

    resizeCanvas() {
        const rect = this.canvas.parentElement.getBoundingClientRect();
        this.canvas.width = rect.width;
        this.canvas.height = rect.height;
    }

    setupEventListeners() {
        window.addEventListener('resize', () => this.resizeCanvas());

        // Canvas interactions
        this.canvas.addEventListener('click', (e) => this.handleCanvasClick(e));
        this.canvas.addEventListener('mousemove', (e) => this.handleCanvasMouseMove(e));

        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => this.handleKeyPress(e));
    }

    //=========================================================================
    // MAIN LOOP AND RENDERING
    //=========================================================================

    startMainLoop() {
        const loop = (timestamp) => {
            this.update(timestamp);
            this.render(timestamp);
            this.animationId = requestAnimationFrame(loop);
        };
        this.animationId = requestAnimationFrame(loop);
    }

    update(timestamp) {
        const deltaTime = timestamp / 1000.0;

        // Update all engines
        this.beatSystem.update(deltaTime);
        this.worldEngine.update(deltaTime);
        this.quantumEngine.update(deltaTime);
        this.assetEngine.update(deltaTime);
        this.gameDirector.update(deltaTime);
        this.aiEngine.update(deltaTime);
    }

    render(timestamp) {
        // Clear canvas
        this.ctx.fillStyle = '#06101c';
        this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);

        // Render based on current view mode
        switch (this.currentViewMode) {
            case 'world':
                this.worldEngine.render(this.ctx);
                break;
            case 'quantum':
                this.quantumEngine.render(this.ctx);
                break;
            case 'glyph':
                this.renderGlyphView(this.ctx);
                break;
            case 'debug':
                this.renderDebugView(this.ctx);
                break;
        }

        // Always render UI overlays
        this.renderUIOverlays(this.ctx);
    }

    renderGlyphView(ctx) {
        // Render glyph forge visualization
        this.quantumEngine.renderGlyphs(ctx);
        this.assetEngine.renderGlyphAssets(ctx);
    }

    renderDebugView(ctx) {
        // Debug visualization
        this.worldEngine.renderDebug(ctx);
        this.assetEngine.renderDebug(ctx);
        this.renderDebugInfo(ctx);
    }

    renderDebugInfo(ctx) {
        ctx.fillStyle = '#00ff7f';
        ctx.font = '12px monospace';

        const debugInfo = [
            `Entities: ${this.stats.entityCount}`,
            `Chunks: ${this.stats.chunkCount}`,
            `Memory: ${this.stats.memoryUsage}MB`,
            `AI Patterns: ${this.aiEngine.getPatternCount()}`,
            `BPM: ${this.beatSystem.getCurrentBPM()}`,
            `Mode: ${this.currentMode}`
        ];

        debugInfo.forEach((info, index) => {
            ctx.fillText(info, 10, 20 + index * 15);
        });
    }

    renderUIOverlays(ctx) {
        // Render beat visualization
        this.beatSystem.renderBeats(ctx);

        // Render AI insights
        this.aiEngine.renderInsights(ctx);
    }

    //=========================================================================
    // EVENT HANDLING
    //=========================================================================

    handleCanvasClick(event) {
        const rect = this.canvas.getBoundingClientRect();
        const x = event.clientX - rect.left;
        const y = event.clientY - rect.top;

        switch (this.currentViewMode) {
            case 'world':
                this.worldEngine.handleClick(x, y);
                break;
            case 'quantum':
                this.quantumEngine.handleClick(x, y);
                break;
            case 'glyph':
                this.handleGlyphClick(x, y);
                break;
        }
    }

    handleCanvasMouseMove(event) {
        const rect = this.canvas.getBoundingClientRect();
        const x = event.clientX - rect.left;
        const y = event.clientY - rect.top;

        this.worldEngine.handleMouseMove(x, y);
        this.quantumEngine.handleMouseMove(x, y);
    }

    handleKeyPress(event) {
        switch (event.key) {
            case '1': this.setViewMode('world'); break;
            case '2': this.setViewMode('quantum'); break;
            case '3': this.setViewMode('glyph'); break;
            case '4': this.setViewMode('debug'); break;
            case 'g': this.worldEngine.generateNewChunk(); break;
            case 'r': this.worldEngine.regenerateWorld(); break;
            case 'p': this.beatSystem.togglePlayPause(); break;
            case 'Escape': this.openConsole(); break;
        }
    }

    handleGlyphClick(x, y) {
        // Handle glyph interactions
        const glyph = this.quantumEngine.getGlyphAt(x, y);
        if (glyph) {
            this.gameDirector.activateGlyph(glyph);
        }
    }

    //=========================================================================
    // PUBLIC API METHODS
    //=========================================================================

    // View mode switching
    setViewMode(mode) {
        this.currentViewMode = mode;
        console.log(`🎮 View mode set to: ${mode}`);

        // Update UI
        document.querySelectorAll('.canvas-btn').forEach(btn => {
            btn.classList.remove('active');
        });
        document.querySelector(`[onclick*="${mode}"]`).classList.add('active');
    }

    // Development mode switching
    setMode(mode) {
        this.currentMode = mode;
        console.log(`🛠️ Development mode: ${mode}`);

        // Update UI
        document.querySelectorAll('.dock-btn').forEach(btn => {
            if (btn.textContent.toLowerCase() === mode.toLowerCase()) {
                btn.classList.add('active');
            } else {
                btn.classList.remove('active');
            }
        });

        // Notify engines of mode change
        this.worldEngine.setMode(mode);
        this.gameDirector.setMode(mode);
    }

    // Tool activation
    activateTool(toolName) {
        console.log(`🔧 Activating tool: ${toolName}`);

        switch (toolName) {
            case 'worldGenerator':
                this.worldEngine.openWorldGenerator();
                break;
            case 'assetManager':
                this.assetEngine.openAssetManager();
                break;
            case 'aiDirector':
                this.aiEngine.openAIDirector();
                break;
            case 'quantumPhysics':
                this.quantumEngine.openQuantumEditor();
                break;
            case 'glyphForge':
                this.quantumEngine.openGlyphForge();
                break;
            case 'beatSystem':
                this.beatSystem.openBeatEditor();
                break;
        }
    }

    // World management
    loadWorld(worldName) {
        console.log(`🌍 Loading world: ${worldName}`);
        this.worldEngine.loadWorld(worldName);
        this.currentWorld = worldName;

        // Update UI
        document.querySelectorAll('.dock-btn').forEach(btn => {
            if (btn.textContent === worldName) {
                btn.classList.add('active');
            } else if (btn.onclick?.toString().includes('loadWorld')) {
                btn.classList.remove('active');
            }
        });
    }

    createWorld() {
        console.log('🌟 Creating new world');
        const worldName = prompt('World name:') || 'NewWorld';
        this.worldEngine.createWorld(worldName);
        this.loadWorld(worldName);
    }

    loadDefaultWorld() {
        this.loadWorld('test_world');
    }

    // Recommendation system
    executeRecommendation(type) {
        console.log(`💡 Executing recommendation: ${type}`);
        this.gameDirector.executeRecommendation(type);
    }

    // Project management
    saveProject() {
        console.log('💾 Saving project');
        const projectData = {
            world: this.worldEngine.exportWorld(),
            assets: this.assetEngine.exportAssets(),
            ai: this.aiEngine.exportIntelligence(),
            settings: {
                mode: this.currentMode,
                viewMode: this.currentViewMode
            }
        };

        // In a real implementation, this would save to file system
        localStorage.setItem('nexus_forge_project', JSON.stringify(projectData));
        console.log('✅ Project saved');
    }

    exportBuild() {
        console.log('📦 Exporting build');
        this.gameDirector.exportBuild();
    }

    // Console and debugging
    openConsole() {
        console.log('🖥️ Opening console');
        // In a real implementation, this would open a dev console overlay
    }

    openProfiler() {
        console.log('📊 Opening profiler');
        // In a real implementation, this would open performance profiler
    }

    // Stats updater
    startStatsUpdater() {
        setInterval(() => {
            this.updateStats();
        }, 1000); // Update every second
    }

    updateStats() {
        this.stats = {
            fps: Math.floor(1000 / 16.67), // Mock FPS
            entityCount: this.worldEngine.getEntityCount(),
            chunkCount: this.worldEngine.getChunkCount(),
            memoryUsage: this.assetEngine.getMemoryUsage()
        };

        // Update UI
        document.getElementById('fps').textContent = this.stats.fps;
        document.getElementById('entityCount').textContent = this.stats.entityCount.toLocaleString();
        document.getElementById('chunkCount').textContent = `${this.stats.chunkCount}/128`;
        document.getElementById('memoryUsage').textContent = `${this.stats.memoryUsage} MB`;
        document.getElementById('currentBPM').textContent = this.beatSystem.getCurrentBPM();
        document.getElementById('petalCount').textContent = this.beatSystem.getPetalCount();
    }
}

//=============================================================================
// UNIFIED AI ENGINE (Enhanced from nexus_forge_primordial.js)
//=============================================================================

class UnifiedAIEngine {
    constructor() {
        this.patterns = new Map();
        this.gamePatterns = new Map(); // Game-specific patterns
        this.recommendations = [];
        this.insights = [];
        this.worldEngine = null;
        this.quantumEngine = null;
        this.lastUpdate = 0;

        this.initializePatterns();
        this.initializeGamePatterns();
    }

    initializePatterns() {
        // Enhanced pattern recognition for game development
        this.patterns.set('performance_issues', [
            /lag|slow|frame.*drop|stuttering/i,
            /memory.*leak|cpu.*spike/i,
            /loading.*time|asset.*load/i
        ]);

        this.patterns.set('balance_issues', [
            /overpowered|underpowered|too.*strong|too.*weak/i,
            /unfair|imbalanced|broken.*mechanic/i,
            /damage.*high|damage.*low/i
        ]);

        this.patterns.set('content_gaps', [
            /empty.*area|nothing.*here|boring.*zone/i,
            /need.*more.*content|lacks.*variety/i,
            /repetitive|same.*thing/i
        ]);

        this.patterns.set('ui_problems', [
            /confusing.*interface|ui.*bug|menu.*broken/i,
            /hard.*to.*find|unclear.*button/i,
            /accessibility|colorblind/i
        ]);
    }

    initializeGamePatterns() {
        // Game-specific intelligence patterns
        this.gamePatterns.set('player_behavior', {
            avoidancePatterns: /avoid|stay.*away|don.*like/i,
            preferencePatterns: /prefer|love|enjoy|fun/i,
            frustrationPatterns: /frustrating|annoying|hate/i
        });

        this.gamePatterns.set('progression_issues', {
            tooEasy: /too.*easy|no.*challenge/i,
            tooHard: /too.*hard|impossible|stuck/i,
            progression: /level.*up|xp|progress/i
        });
    }

    connectWorldEngine(worldEngine) {
        this.worldEngine = worldEngine;
    }

    connectQuantumEngine(quantumEngine) {
        this.quantumEngine = quantumEngine;
    }

    initialize() {
        console.log('🧠 AI Engine initialized');
    }

    update(deltaTime) {
        this.lastUpdate += deltaTime;

        if (this.lastUpdate >= 5.0) { // Update every 5 seconds
            this.analyzeGameState();
            this.generateRecommendations();
            this.lastUpdate = 0;
        }
    }

    analyzeGameState() {
        if (!this.worldEngine) return;

        const gameState = this.worldEngine.getCurrentState();

        // Analyze player movement patterns
        this.analyzePlayerBehavior(gameState.playerData);

        // Analyze performance metrics
        this.analyzePerformance(gameState.performance);

        // Analyze content usage
        this.analyzeContentUsage(gameState.content);
    }

    analyzePlayerBehavior(playerData) {
        // Mock player behavior analysis
        const avoidedAreas = playerData?.avoidedAreas || ['water', 'desert'];
        const preferredAreas = playerData?.preferredAreas || ['forest', 'town'];

        if (avoidedAreas.includes('water')) {
            this.addInsight({
                type: 'player_behavior',
                severity: 'medium',
                message: 'Players tend to avoid water areas → Recommendation: Add underwater content or bridges',
                confidence: 0.78
            });
        }
    }

    analyzePerformance(performance) {
        // Mock performance analysis
        if (performance?.averageFPS < 45) {
            this.addInsight({
                type: 'performance_issues',
                severity: 'high',
                message: 'Frame rate below optimal → Optimize rendering pipeline',
                confidence: 0.92
            });
        }
    }

    analyzeContentUsage(content) {
        // Mock content analysis
        const emptyBiomes = content?.emptyBiomes || ['desert'];

        if (emptyBiomes.length > 0) {
            this.addInsight({
                type: 'content_gaps',
                severity: 'medium',
                message: `${emptyBiomes.join(', ')} biome(s) lack unique creatures → Generate specific entities`,
                confidence: 0.85
            });
        }
    }

    addInsight(insight) {
        insight.timestamp = Date.now();
        this.insights.unshift(insight);

        // Keep only recent insights
        this.insights = this.insights.slice(0, 50);

        console.log(`🎯 AI Insight: ${insight.message}`);
    }

    generateRecommendations() {
        this.recommendations = [];

        // Generate recommendations based on insights
        const highSeverityInsights = this.insights.filter(i => i.severity === 'high');
        const mediumSeverityInsights = this.insights.filter(i => i.severity === 'medium');

        if (highSeverityInsights.length > 0) {
            this.recommendations.push({
                id: 'high_priority_fixes',
                type: 'critical',
                title: 'Address Critical Issues',
                description: `${highSeverityInsights.length} high-priority issues detected`,
                estimatedEffort: '1-2 days',
                confidence: 0.9
            });
        }

        if (mediumSeverityInsights.length > 2) {
            this.recommendations.push({
                id: 'content_enhancement',
                type: 'enhancement',
                title: 'Enhance Game Content',
                description: `Improve content in ${mediumSeverityInsights.length} areas`,
                estimatedEffort: '1 week',
                confidence: 0.75
            });
        }
    }

    getPatternCount() {
        return this.patterns.size + this.gamePatterns.size;
    }

    exportIntelligence() {
        return {
            patterns: Array.from(this.patterns.keys()),
            insights: this.insights.slice(0, 20),
            recommendations: this.recommendations
        };
    }

    openAIDirector() {
        console.log('🎯 Opening AI Game Director interface');
        // In a real implementation, this would open AI director UI
    }

    renderInsights(ctx) {
        // Render AI insights as subtle overlays
        const recentInsights = this.insights.slice(0, 3);

        ctx.save();
        ctx.globalAlpha = 0.7;
        ctx.fillStyle = '#28F49B';
        ctx.font = '12px monospace';

        recentInsights.forEach((insight, index) => {
            const y = ctx.canvas.height - 60 + (index * 15);
            ctx.fillText(`AI: ${insight.message.slice(0, 60)}...`, 10, y);
        });

        ctx.restore();
    }
}

//=============================================================================
// UNIFIED WORLD ENGINE (Enhanced from nexus_world_demo.cpp concepts)
//=============================================================================

class UnifiedWorldEngine {
    constructor() {
        this.chunks = new Map();
        this.entities = new Map();
        this.camera = { x: 0, y: 0, zoom: 1.0 };
        this.chunkSize = 256;
        this.worldSeed = 1337;
        this.currentWorld = null;
        this.assetEngine = null;
        this.beatSystem = null;
        this.mode = 'development';

        // Biome system
        this.biomes = ['forest', 'desert', 'mountain', 'water', 'plains', 'swamp'];
        this.biomeColors = {
            forest: '#2d5016',
            desert: '#c2b280',
            mountain: '#8b7355',
            water: '#4682b4',
            plains: '#7cfc00',
            swamp: '#556b2f'
        };
    }

    connectAssetEngine(assetEngine) {
        this.assetEngine = assetEngine;
    }

    connectBeatSystem(beatSystem) {
        this.beatSystem = beatSystem;
    }

    initialize(canvas, ctx) {
        this.canvas = canvas;
        this.ctx = ctx;

        // Generate initial world chunks
        this.generateInitialChunks();

        console.log('🌍 World Engine initialized');
    }

    update(deltaTime) {
        // Update entities
        for (const [id, entity] of this.entities) {
            this.updateEntity(entity, deltaTime);
        }

        // Check for new chunk generation based on camera position
        this.updateChunkLoading();
    }

    updateEntity(entity, deltaTime) {
        // Basic entity movement
        if (entity.velocity) {
            entity.x += entity.velocity.x * deltaTime;
            entity.y += entity.velocity.y * deltaTime;
        }

        // Beat-sync effects
        if (this.beatSystem && entity.beatReactive) {
            entity.beatScale = 1.0 + this.beatSystem.getBeatIntensity() * 0.2;
        }
    }

    generateInitialChunks() {
        // Generate 3x3 chunks around origin
        for (let x = -1; x <= 1; x++) {
            for (let y = -1; y <= 1; y++) {
                this.generateChunk(x, y);
            }
        }
    }

    generateChunk(chunkX, chunkY) {
        const chunkId = `${chunkX},${chunkY}`;

        if (this.chunks.has(chunkId)) return;

        const chunk = {
            x: chunkX,
            y: chunkY,
            biome: this.determineBiome(chunkX, chunkY),
            entities: [],
            features: [],
            generated: true
        };

        // Generate chunk content based on biome
        this.populateChunk(chunk);

        this.chunks.set(chunkId, chunk);
        console.log(`🗺️ Generated chunk ${chunkId} (${chunk.biome})`);
    }

    determineBiome(x, y) {
        // Simple biome determination using noise-like function
        const noise = Math.sin(x * 0.3) * Math.cos(y * 0.3) +
            Math.sin(x * 0.1) * Math.cos(y * 0.1);

        if (noise > 0.5) return 'mountain';
        if (noise > 0.2) return 'forest';
        if (noise > -0.2) return 'plains';
        if (noise > -0.5) return 'desert';
        if (noise > -0.7) return 'swamp';
        return 'water';
    }

    populateChunk(chunk) {
        const entityCount = Math.floor(Math.random() * 10) + 5;

        for (let i = 0; i < entityCount; i++) {
            const entity = {
                id: `entity_${chunk.x}_${chunk.y}_${i}`,
                type: this.getRandomEntityType(chunk.biome),
                x: chunk.x * this.chunkSize + Math.random() * this.chunkSize,
                y: chunk.y * this.chunkSize + Math.random() * this.chunkSize,
                health: 100,
                beatReactive: Math.random() > 0.5,
                beatScale: 1.0
            };

            chunk.entities.push(entity);
            this.entities.set(entity.id, entity);
        }
    }

    getRandomEntityType(biome) {
        const entityTypes = {
            forest: ['tree', 'deer', 'wolf', 'berry_bush'],
            desert: ['cactus', 'lizard', 'oasis'],
            mountain: ['rock', 'eagle', 'cave', 'crystal'],
            water: ['fish', 'seaweed', 'coral'],
            plains: ['grass', 'rabbit', 'flower'],
            swamp: ['lily_pad', 'frog', 'vine', 'mushroom']
        };

        const types = entityTypes[biome] || ['rock'];
        return types[Math.floor(Math.random() * types.length)];
    }

    updateChunkLoading() {
        // Simple chunk loading based on camera position
        const cameraChunkX = Math.floor(this.camera.x / this.chunkSize);
        const cameraChunkY = Math.floor(this.camera.y / this.chunkSize);

        // Load chunks in 3x3 area around camera
        for (let x = cameraChunkX - 1; x <= cameraChunkX + 1; x++) {
            for (let y = cameraChunkY - 1; y <= cameraChunkY + 1; y++) {
                const chunkId = `${x},${y}`;
                if (!this.chunks.has(chunkId)) {
                    this.generateChunk(x, y);
                }
            }
        }
    }

    render(ctx) {
        ctx.save();

        // Apply camera transform
        ctx.translate(-this.camera.x, -this.camera.y);
        ctx.scale(this.camera.zoom, this.camera.zoom);

        // Render chunks
        for (const [id, chunk] of this.chunks) {
            this.renderChunk(ctx, chunk);
        }

        // Render entities
        for (const [id, entity] of this.entities) {
            this.renderEntity(ctx, entity);
        }

        ctx.restore();
    }

    renderChunk(ctx, chunk) {
        const x = chunk.x * this.chunkSize;
        const y = chunk.y * this.chunkSize;

        // Render biome background
        ctx.fillStyle = this.biomeColors[chunk.biome] || '#333';
        ctx.fillRect(x, y, this.chunkSize, this.chunkSize);

        // Render chunk border in development mode
        if (this.mode === 'development') {
            ctx.strokeStyle = '#444';
            ctx.lineWidth = 1;
            ctx.strokeRect(x, y, this.chunkSize, this.chunkSize);
        }
    }

    renderEntity(ctx, entity) {
        ctx.save();

        ctx.translate(entity.x, entity.y);

        if (entity.beatScale !== 1.0) {
            ctx.scale(entity.beatScale, entity.beatScale);
        }

        // Simple entity rendering
        const color = this.getEntityColor(entity.type);
        ctx.fillStyle = color;

        switch (entity.type) {
            case 'tree':
                ctx.fillRect(-5, -15, 10, 15);
                ctx.fillStyle = '#228B22';
                ctx.fillRect(-8, -25, 16, 10);
                break;
            case 'rock':
                ctx.fillRect(-4, -4, 8, 8);
                break;
            default:
                ctx.fillRect(-3, -3, 6, 6);
        }

        ctx.restore();
    }

    getEntityColor(type) {
        const colors = {
            tree: '#8B4513',
            deer: '#CD853F',
            wolf: '#808080',
            rock: '#696969',
            cactus: '#228B22',
            fish: '#4169E1',
            default: '#FFA500'
        };

        return colors[type] || colors.default;
    }

    renderDebug(ctx) {
        // Render debug information
        ctx.fillStyle = '#00ff7f';
        ctx.font = '12px monospace';

        const debugInfo = [
            `Camera: ${this.camera.x.toFixed(1)}, ${this.camera.y.toFixed(1)}`,
            `Zoom: ${this.camera.zoom.toFixed(2)}`,
            `Chunks: ${this.chunks.size}`,
            `Entities: ${this.entities.size}`,
            `World: ${this.currentWorld || 'None'}`
        ];

        debugInfo.forEach((info, index) => {
            ctx.fillText(info, this.canvas.width - 200, 20 + index * 15);
        });
    }

    // Public methods
    handleClick(x, y) {
        // Convert screen coordinates to world coordinates
        const worldX = x + this.camera.x;
        const worldY = y + this.camera.y;

        console.log(`🌍 World clicked at: ${worldX.toFixed(1)}, ${worldY.toFixed(1)}`);

        // In development mode, place a new entity
        if (this.mode === 'development') {
            this.placeEntity(worldX, worldY, 'rock');
        }
    }

    handleMouseMove(x, y) {
        // Handle camera movement on drag
        // Implementation would depend on mouse drag detection
    }

    placeEntity(x, y, type) {
        const entity = {
            id: `placed_${Date.now()}`,
            type: type,
            x: x,
            y: y,
            health: 100,
            beatReactive: false,
            beatScale: 1.0
        };

        this.entities.set(entity.id, entity);
        console.log(`🎯 Placed ${type} at ${x.toFixed(1)}, ${y.toFixed(1)}`);
    }

    generateNewChunk() {
        const chunkX = Math.floor(Math.random() * 10) - 5;
        const chunkY = Math.floor(Math.random() * 10) - 5;
        this.generateChunk(chunkX, chunkY);
    }

    regenerateWorld() {
        console.log('🔄 Regenerating world');
        this.chunks.clear();
        this.entities.clear();
        this.generateInitialChunks();
    }

    setMode(mode) {
        this.mode = mode;
    }

    loadWorld(worldName) {
        this.currentWorld = worldName;
        // In a real implementation, load world data from storage
        console.log(`🌍 Loading world: ${worldName}`);
    }

    createWorld(worldName) {
        this.currentWorld = worldName;
        this.regenerateWorld();
        console.log(`🌟 Created world: ${worldName}`);
    }

    getCurrentState() {
        return {
            playerData: {
                avoidedAreas: ['water'],
                preferredAreas: ['forest', 'plains']
            },
            performance: {
                averageFPS: 58
            },
            content: {
                emptyBiomes: ['desert']
            }
        };
    }

    getEntityCount() {
        return this.entities.size;
    }

    getChunkCount() {
        return this.chunks.size;
    }

    exportWorld() {
        return {
            chunks: Array.from(this.chunks.entries()),
            entities: Array.from(this.entities.entries()),
            camera: { ...this.camera },
            worldSeed: this.worldSeed
        };
    }

    openWorldGenerator() {
        console.log('🗺️ Opening World Generator');
        // In a real implementation, open world generation UI
    }
}

//=============================================================================
// Additional unified engines will continue in the next part...
//=============================================================================

// Export the main class
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { NexusForgeUnified };
}
