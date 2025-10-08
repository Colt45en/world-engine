/**
 * World Engine Tier 5 - Vortex Lab Engine with Mind's Eye Processing
 * ==================================================================
 *
 * Advanced demonstration of World Engine Tier 5 capabilities
 * with Vortex Lab Engine, Glyph Network Oscillation, and Meta-Layer Processing.
 */

class WorldEngineTier5Demo {
    constructor() {
        this.state = { p: 0.0, i: 0.5, g: 0.3, c: 0.6 };
        this.snapshots = [];
        this.eventLog = [];
        this.ideContext = {};

        // Environment system
        this.environments = [];
        this.agents = [];
        this.activeEnvironment = null;
        this.humanJoined = false;
        this.trainingActive = false;
        this.interactionCount = 0;
        this.learningProgress = 0;

        // Tier 5 Vortex Lab Engine
        this.vortexIntensity = 0;
        this.mindsEyeActive = false;
        this.glyphNetwork = [];
        this.metaLayerActive = false;
        this.atomicDecompositionLevel = 0;
        this.librianData = [];
        this.dataOptimizerState = 'idle';

        // Neural Flow & Consciousness Systems
        this.neuralNodes = [];
        this.neuralConnections = [];
        this.consciousnessLevel = 0;
        this.emergenceThreshold = 0.75;
        this.quantumEntangled = [];
        this.multiverseBranches = [];
        this.patternSynthesis = {
            learnedPatterns: [],
            complexityEvolution: 0,
            adaptationRate: 0.1
        };

        // 3D Studio Systems
        this.studioScenes = {
            main: { active: true, objects: [], camera: null },
            avatar: { active: false, objects: [], camera: null },
            codex: { active: false, objects: [], camera: null },
            impostor: { active: false, objects: [], camera: null }
        };
        this.currentStudioScene = 'main';
        this.renderer = null;
        this.orbitControls = null;
        this.hudVisible = false;
        this.sonicOverlayActive = false;

        // Glyph patterns from your visualization
        this.glyphTypes = [
            'Input', 'Output', 'Perspective', 'Processing', 'Meaning',
            'Meta processing', 'Atomic decomposition', 'Data optimizer',
            'Data sorters', 'Root sorter', 'Librarian'
        ];

        this.NUCLEUS_TIER5_PATTERNS = {
            'recursion_outcome': { intensity: 0.8, complexity: 'high', processing: 'meta' },
            'perspective_shift': { intensity: 0.6, complexity: 'medium', processing: 'cognitive' },
            'atomic_breakdown': { intensity: 0.9, complexity: 'extreme', processing: 'fundamental' },
            'meaning_synthesis': { intensity: 0.7, complexity: 'high', processing: 'semantic' },
            'data_optimization': { intensity: 0.5, complexity: 'medium', processing: 'efficiency' }
        };

        // NUCLEUS pattern recognition
        this.nucleusPatterns = {
            'forest': { type: '🌲', agents: 'cooperative', learning: 'resource_sharing' },
            'city': { type: '🏢', agents: 'social', learning: 'communication' },
            'ocean': { type: '🌊', agents: 'adaptive', learning: 'exploration' },
            'space': { type: '🚀', agents: 'analytical', learning: 'problem_solving' },
            'playground': { type: '🎮', agents: 'creative', learning: 'play_based' }
        };

        this.OPERATORS = {
            ST: { name: 'Snapshot', color: '#e74c3c' },
            RST: { name: 'Restore', color: '#f39c12' },
            RB: { name: 'Rebuild', color: '#27ae60' },
            EDT: { name: 'Edit', color: '#3498db' },
            UP: { name: 'Update', color: '#9b59b6' },
            CNV: { name: 'Convert', color: '#1abc9c' },
            SEL: { name: 'Select', color: '#34495e' },
            PRV: { name: 'Prevent', color: '#e67e22' }
        };

        this.setupOperators();
        this.initializeDemo();
        this.setupEnvironmentCanvas();
    }

    // ============================= Core State Management =============================

    updateState(deltas) {
        for (const [key, delta] of Object.entries(deltas)) {
            if (key in this.state) {
                this.state[key] = Math.max(-1, Math.min(1, this.state[key] + delta));
            }
        }
        this.updateDisplay();
        this.updateStackHeight();
    }

    updateDisplay() {
        // Update state bars
        ['p', 'i', 'g', 'c'].forEach(key => {
            const valueEl = document.getElementById(`${key}-value`);
            const barEl = document.getElementById(`${key}-bar`);
            if (valueEl && barEl) {
                const normalizedValue = (this.state[key] + 1) / 2; // Convert -1,1 to 0,1
                valueEl.textContent = this.state[key].toFixed(3);
                barEl.style.width = `${normalizedValue * 100}%`;
            }
        });
    }

    updateStackHeight() {
        const mu = Object.values(this.state).reduce((sum, val) => sum + val, 0);
        const muEl = document.getElementById('mu-display');
        if (muEl) {
            muEl.textContent = mu.toFixed(3);
        }
    }

    // ============================= Operator System =============================

    setupOperators() {
        const grid = document.getElementById('operator-grid');
        if (!grid) return;

        grid.innerHTML = '';
        for (const [key, op] of Object.entries(this.OPERATORS)) {
            const btn = document.createElement('button');
            btn.className = 'operator-btn';
            btn.style.background = op.color;
            btn.textContent = `${key}\n${op.name}`;
            btn.onclick = () => this.applyOperator(key);
            grid.appendChild(btn);
        }
    }

    applyOperator(opKey) {
        const effects = {
            ST: { p: 0.1, i: -0.05, g: 0.02, c: 0.1 },
            RST: { p: -0.1, i: 0.05, g: -0.02, c: -0.05 },
            RB: { p: 0.05, i: 0.1, g: 0.1, c: 0.05 },
            EDT: { p: 0.02, i: 0.08, g: -0.03, c: -0.02 },
            UP: { p: -0.02, i: 0.05, g: 0.05, c: 0.03 },
            CNV: { p: 0.08, i: -0.02, g: 0.1, c: 0.02 },
            SEL: { p: 0.03, i: 0.02, g: -0.05, c: 0.05 },
            PRV: { p: -0.05, i: -0.1, g: 0.03, c: 0.08 }
        };

        if (effects[opKey]) {
            this.updateState(effects[opKey]);
            this.logEvent(`Applied ${opKey} (${this.OPERATORS[opKey].name})`);
        }
    }

    // ============================= Environment Creation System =============================

    setupEnvironmentCanvas() {
        const canvas = document.getElementById('environment-canvas');
        if (!canvas) return;

        canvas.addEventListener('click', (e) => {
            if (this.activeEnvironment) {
                const rect = canvas.getBoundingClientRect();
                const x = ((e.clientX - rect.left) / rect.width) * 100;
                const y = ((e.clientY - rect.top) / rect.height) * 100;
                this.addInteractionPoint(x, y);
            }
        });
    }

    generateEnvironment(prompt) {
        // NUCLEUS pattern recognition
        const environmentType = this.detectEnvironmentType(prompt);

        this.activeEnvironment = {
            id: Date.now(),
            prompt: prompt,
            type: environmentType,
            created: new Date(),
            agents: [],
            interactions: []
        };

        this.environments.push(this.activeEnvironment);
        this.renderEnvironment();
        this.logEvent(`🌍 NUCLEUS created ${environmentType.type} environment: "${prompt}"`);

        // Update state based on environment creation
        this.updateState({ p: 0.1, i: 0.2, g: 0.15, c: 0.1 });
    }

    detectEnvironmentType(prompt) {
        const lowerPrompt = prompt.toLowerCase();

        for (const [key, pattern] of Object.entries(this.nucleusPatterns)) {
            if (lowerPrompt.includes(key) || this.containsKeywords(lowerPrompt, this.getKeywords(key))) {
                return pattern;
            }
        }

        // Default creative environment
        return { type: '🎨', agents: 'creative', learning: 'adaptive' };
    }

    getKeywords(environmentType) {
        const keywords = {
            'forest': ['tree', 'nature', 'cooperation', 'resource', 'wildlife'],
            'city': ['urban', 'social', 'communication', 'building', 'community'],
            'ocean': ['water', 'sea', 'exploration', 'adaptive', 'marine'],
            'space': ['star', 'planet', 'analysis', 'science', 'discovery'],
            'playground': ['play', 'game', 'fun', 'creative', 'learn']
        };
        return keywords[environmentType] || [];
    }

    containsKeywords(text, keywords) {
        return keywords.some(keyword => text.includes(keyword));
    }

    renderEnvironment() {
        const canvas = document.getElementById('environment-canvas');
        if (!canvas || !this.activeEnvironment) return;

        canvas.innerHTML = '';

        // Add environment background indicator
        const indicator = document.createElement('div');
        indicator.style.cssText = `
      position: absolute;
      top: 10px;
      left: 10px;
      font-size: 1.5em;
      opacity: 0.8;
    `;
        indicator.textContent = this.activeEnvironment.type.type;
        canvas.appendChild(indicator);

        // Add environment description
        const description = document.createElement('div');
        description.style.cssText = `
      position: absolute;
      bottom: 10px;
      left: 10px;
      right: 10px;
      font-size: 0.8em;
      opacity: 0.7;
      text-align: center;
    `;
        description.textContent = `${this.activeEnvironment.type.learning} learning environment`;
        canvas.appendChild(description);

        // Render agents
        this.renderAgents();
    }

    renderAgents() {
        const canvas = document.getElementById('environment-canvas');
        if (!canvas) return;

        this.agents.forEach(agent => {
            const agentEl = document.createElement('div');
            agentEl.className = 'agent-indicator';
            agentEl.style.left = `${agent.x}%`;
            agentEl.style.top = `${agent.y}%`;
            agentEl.title = `Agent ${agent.id} - ${agent.behavior}`;
            agentEl.onclick = () => this.interactWithAgent(agent);
            canvas.appendChild(agentEl);
        });
    }

    spawnAgent() {
        if (!this.activeEnvironment) {
            this.logEvent('❌ No environment active. Create environment first.');
            return;
        }

        const agent = {
            id: this.agents.length + 1,
            x: Math.random() * 90 + 5, // Keep away from edges
            y: Math.random() * 90 + 5,
            behavior: this.activeEnvironment.type.agents,
            learning: this.activeEnvironment.type.learning,
            experience: 0,
            created: new Date()
        };

        this.agents.push(agent);
        this.activeEnvironment.agents.push(agent);
        this.renderAgents();
        this.updateTrainingStats();

        this.logEvent(`🤖 Agent ${agent.id} spawned with ${agent.behavior} behavior`);

        // Update state
        this.updateState({ i: 0.1, g: 0.05, c: 0.1 });
    }

    enterEnvironment() {
        if (!this.activeEnvironment) {
            this.logEvent('❌ No environment active. Create environment first.');
            return;
        }

        this.humanJoined = true;
        this.trainingActive = true;
        this.updateTrainingStats();

        this.logEvent('👤 Human entered environment - training activated');

        // Simulate interaction
        setTimeout(() => this.simulateHumanAgentInteraction(), 2000);

        // Update state
        this.updateState({ p: 0.1, i: 0.1, c: 0.15 });
    }

    simulateHumanAgentInteraction() {
        if (!this.trainingActive || this.agents.length === 0) return;

        this.interactionCount++;
        this.learningProgress = Math.min(100, this.learningProgress + Math.random() * 10 + 5);

        const randomAgent = this.agents[Math.floor(Math.random() * this.agents.length)];
        randomAgent.experience += Math.random() * 0.2 + 0.1;

        this.updateTrainingStats();
        this.logEvent(`🎯 Human-Agent interaction #${this.interactionCount} - Agent ${randomAgent.id} learned`);

        // Continue interactions
        if (this.trainingActive) {
            setTimeout(() => this.simulateHumanAgentInteraction(),
                Math.random() * 5000 + 3000); // 3-8 seconds
        }
    }

    addInteractionPoint(x, y) {
        if (!this.activeEnvironment) return;

        const interaction = { x, y, timestamp: new Date(), type: 'human_click' };
        this.activeEnvironment.interactions.push(interaction);

        // Visual feedback
        const canvas = document.getElementById('environment-canvas');
        const point = document.createElement('div');
        point.style.cssText = `
      position: absolute;
      left: ${x}%;
      top: ${y}%;
      width: 6px;
      height: 6px;
      background: #45b7d1;
      border-radius: 50%;
      opacity: 1;
      transition: opacity 2s ease-out;
    `;
        canvas.appendChild(point);

        setTimeout(() => {
            point.style.opacity = '0';
            setTimeout(() => canvas.removeChild(point), 2000);
        }, 100);

        this.interactionCount++;
        this.updateTrainingStats();
    }

    interactWithAgent(agent) {
        agent.experience += 0.1;
        this.interactionCount++;
        this.learningProgress = Math.min(100, this.learningProgress + 5);

        this.updateTrainingStats();
        this.logEvent(`🤖 Direct interaction with Agent ${agent.id} - experience gained`);

        // Update state
        this.updateState({ p: 0.05, i: 0.08, c: 0.1 });
    }

    updateTrainingStats() {
        // Update agent count
        const agentCountEl = document.getElementById('agent-count');
        const agentBarEl = document.getElementById('agent-bar');
        if (agentCountEl && agentBarEl) {
            agentCountEl.textContent = this.agents.length;
            agentBarEl.style.width = `${Math.min(100, this.agents.length * 20)}%`;
        }

        // Update interaction count
        const interactionCountEl = document.getElementById('interaction-count');
        const interactionBarEl = document.getElementById('interaction-bar');
        if (interactionCountEl && interactionBarEl) {
            interactionCountEl.textContent = this.interactionCount;
            interactionBarEl.style.width = `${Math.min(100, this.interactionCount * 5)}%`;
        }

        // Update learning progress
        const learningProgressEl = document.getElementById('learning-progress');
        const learningBarEl = document.getElementById('learning-bar');
        if (learningProgressEl && learningBarEl) {
            learningProgressEl.textContent = `${Math.round(this.learningProgress)}%`;
            learningBarEl.style.width = `${this.learningProgress}%`;
        }

        // Update environment effectiveness
        const effectivenessEl = document.getElementById('environment-effectiveness');
        if (effectivenessEl) {
            if (this.agents.length > 0 && this.interactionCount > 0) {
                const effectiveness = Math.round((this.learningProgress / this.interactionCount) * 10) / 10;
                effectivenessEl.textContent = `${effectiveness.toFixed(1)}% per interaction`;
            } else {
                effectivenessEl.textContent = 'Awaiting Data';
            }
        }
    }

    clearEnvironment() {
        this.activeEnvironment = null;
        this.agents = [];
        this.humanJoined = false;
        this.trainingActive = false;
        this.interactionCount = 0;
        this.learningProgress = 0;

        const canvas = document.getElementById('environment-canvas');
        if (canvas) {
            canvas.innerHTML = `
        <div style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); opacity: 0.5; text-align: center;">
          <div>🌱 Environment Canvas</div>
          <div style="font-size: 0.8em; margin-top: 5px;">Agents and humans interact here</div>
        </div>
      `;
        }

        this.updateTrainingStats();
        this.logEvent('🧹 Environment cleared - all agents removed');
    }

    pauseTraining() {
        this.trainingActive = !this.trainingActive;
        const action = this.trainingActive ? 'resumed' : 'paused';
        this.logEvent(`⏸️ Training ${action}`);
    }

    resetTraining() {
        this.agents.forEach(agent => { agent.experience = 0; });
        this.interactionCount = 0;
        this.learningProgress = 0;
        this.updateTrainingStats();
        this.logEvent('🔄 Training progress reset - agents retain positions');
    }

    // ============================= IDE Integration =============================

    handleWordInput() {
        const input = document.getElementById('word-input');
        if (!input) return;

        const text = input.value.trim();
        if (!text) return;

        this.processNaturalLanguage(text);
        input.value = '';
    }

    processNaturalLanguage(text) {
        const lowerText = text.toLowerCase();
        let processed = false;

        // Check for environment-related commands
        if (lowerText.includes('create') || lowerText.includes('generate')) {
            document.getElementById('environment-prompt').value = text;
            this.generateEnvironment(text);
            processed = true;
        } else if (lowerText.includes('spawn') || lowerText.includes('add agent')) {
            this.spawnAgent();
            processed = true;
        } else if (lowerText.includes('join') || lowerText.includes('enter')) {
            this.enterEnvironment();
            processed = true;
        } else {
            // Existing operator processing
            const operatorMappings = {
                'save': 'ST', 'snapshot': 'ST', 'store': 'ST',
                'restore': 'RST', 'undo': 'RST', 'revert': 'RST',
                'rebuild': 'RB', 'build': 'RB', 'compile': 'RB',
                'edit': 'EDT', 'modify': 'EDT', 'change': 'EDT',
                'update': 'UP', 'refresh': 'UP', 'sync': 'UP'
            };

            for (const [keyword, operator] of Object.entries(operatorMappings)) {
                if (lowerText.includes(keyword)) {
                    this.applyOperator(operator);
                    processed = true;
                    break;
                }
            }
        }

        if (!processed) {
            this.logEvent(`🤔 Processing: "${text}" - analyzed but no direct action taken`);
        }
    }

    simulateBuildSuccess() {
        this.updateState({ c: 0.15, i: 0.1, g: 0.05 });
        this.logEvent('✅ Build successful');
        this.showRecommendations([
            'State confidence increased',
            'Ready for next development cycle',
            'Consider adding tests'
        ]);
    }

    simulateBuildError() {
        this.updateState({ c: -0.1, i: 0.05, p: -0.05 });
        this.logEvent('❌ Build failed');
        this.showRecommendations([
            'Check dependencies',
            'Review recent changes',
            'Run diagnostics'
        ]);
    }

    simulateTestPass() {
        this.updateState({ c: 0.1, g: 0.08, i: 0.02 });
        this.logEvent('🧪 Tests passed');
        this.showRecommendations([
            'Code quality verified',
            'Ready for deployment',
            'Consider refactoring optimizations'
        ]);
    }

    simulateTestFail() {
        this.updateState({ c: -0.08, p: -0.05, i: 0.1 });
        this.logEvent('🧪 Tests failed');
        this.showRecommendations([
            'Review failing test cases',
            'Check edge conditions',
            'Verify input validation'
        ]);
    }

    showRecommendations(recommendations) {
        const container = document.getElementById('recommendations-container');
        if (!container) return;

        container.innerHTML = `
      <div class="recommendations">
        <strong>💡 Recommendations:</strong>
        ${recommendations.map(rec => `<div class="recommendation">• ${rec}</div>`).join('')}
      </div>
    `;

        setTimeout(() => {
            container.innerHTML = '';
        }, 8000);
    }

    // ============================= Utility Functions =============================

    logEvent(message) {
        const timestamp = new Date().toLocaleTimeString();
        const entry = {
            timestamp,
            message,
            state: { ...this.state }
        };

        this.eventLog.push(entry);

        // Update DOM
        const logContainer = document.getElementById('event-log');
        if (logContainer) {
            const logEntry = document.createElement('div');
            logEntry.className = 'log-entry recent';
            logEntry.textContent = `[${timestamp}] ${message}`;
            logContainer.appendChild(logEntry);
            logContainer.scrollTop = logContainer.scrollHeight;

            // Remove 'recent' class after animation
            setTimeout(() => {
                logEntry.classList.remove('recent');
            }, 3000);
        }
    }

    // ============================= Vortex Lab Engine - Tier 5 =============================

    activateMindsEye() {
        this.mindsEyeActive = !this.mindsEyeActive;
        this.vortexIntensity = this.mindsEyeActive ? 0.8 : 0.3;

        const mindsEye = document.getElementById('minds-eye');
        if (mindsEye) {
            mindsEye.style.animationDuration = this.mindsEyeActive ? '1s' : '3s';
            mindsEye.style.boxShadow = this.mindsEyeActive ?
                '0 0 30px rgba(78,205,196,0.8)' : 'none';
        }

        this.updateVortexDisplay();
        this.logEvent(`👁️ Mind's Eye ${this.mindsEyeActive ? 'activated' : 'deactivated'} - intensity: ${this.vortexIntensity}`);
    }

    oscillateGlyphs() {
        this.generateGlyphNetwork();
        this.vortexIntensity = Math.min(1.0, this.vortexIntensity + 0.2);
        this.updateVortexDisplay();
        this.logEvent(`🔄 Glyph network oscillating - ${this.glyphNetwork.length} nodes active`);
    }

    generateGlyphNetwork() {
        const canvas = document.getElementById('glyph-network');
        if (!canvas) return;

        // Clear existing glyphs
        const existingGlyphs = canvas.querySelectorAll('.glyph-node');
        existingGlyphs.forEach(glyph => glyph.remove());

        this.glyphNetwork = [];
        const glyphCount = Math.floor(Math.random() * 8) + 6; // 6-13 glyphs

        for (let i = 0; i < glyphCount; i++) {
            const angle = (i / glyphCount) * 2 * Math.PI;
            const radius = 80 + Math.random() * 60; // Variable radius
            const centerX = 50; // Center of canvas in %
            const centerY = 50;

            const x = centerX + (radius * Math.cos(angle));
            const y = centerY + (radius * Math.sin(angle));

            const glyphType = this.glyphTypes[Math.floor(Math.random() * this.glyphTypes.length)];

            const glyph = {
                id: i,
                type: glyphType,
                x: Math.max(5, Math.min(95, x)),
                y: Math.max(15, Math.min(85, y)),
                active: true
            };

            this.glyphNetwork.push(glyph);
            this.createGlyphElement(glyph, canvas);
        }
    }

    createGlyphElement(glyph, container) {
        const element = document.createElement('div');
        element.className = 'glyph-node';
        element.style.left = `${glyph.x}%`;
        element.style.top = `${glyph.y}%`;
        element.textContent = glyph.type;
        element.title = `Glyph ${glyph.id}: ${glyph.type}`;
        element.style.animationDelay = `${glyph.id * 0.2}s`;

        element.onclick = () => {
            this.activateGlyph(glyph);
        };

        container.appendChild(element);
    }

    activateGlyph(glyph) {
        this.vortexIntensity = Math.min(1.0, this.vortexIntensity + 0.1);
        this.updateVortexDisplay();
        this.logEvent(`⚡ Glyph activated: ${glyph.type} - processing initiated`);

        // Update state based on glyph type
        const glyphEffects = {
            'Input': { p: 0.1, i: 0.2 },
            'Output': { p: -0.1, c: 0.15 },
            'Processing': { i: 0.15, g: 0.1 },
            'Meta processing': { g: 0.2, c: 0.1 },
            'Meaning': { p: 0.15, g: 0.05 }
        };

        if (glyphEffects[glyph.type]) {
            this.updateState(glyphEffects[glyph.type]);
        }
    }

    atomicDecomposition() {
        this.atomicDecompositionLevel++;
        this.metaLayerActive = true;
        this.vortexIntensity = Math.min(1.0, this.vortexIntensity + 0.3);

        this.updateVortexDisplay();
        this.logEvent(`⚛️ Atomic decomposition level ${this.atomicDecompositionLevel} - fundamental analysis active`);

        // Extreme state changes for atomic-level processing
        this.updateState({ p: 0.2, i: 0.3, g: 0.25, c: 0.15 });
    }

    metaProcessing() {
        this.metaLayerActive = !this.metaLayerActive;
        const indicator = document.querySelector('.meta-layer-indicator');

        if (indicator) {
            indicator.textContent = this.metaLayerActive ? 'META LAYER ACTIVE' : 'META LAYER STANDBY';
            indicator.style.opacity = this.metaLayerActive ? '1' : '0.5';
        }

        if (this.metaLayerActive) {
            this.vortexIntensity = Math.min(1.0, this.vortexIntensity + 0.25);
        }

        this.updateVortexDisplay();
        this.logEvent(`🧠 Meta-layer ${this.metaLayerActive ? 'engaged' : 'disengaged'} - recursive processing ${this.metaLayerActive ? 'active' : 'paused'}`);
    }

    dataOptimizer() {
        this.dataOptimizerState = this.dataOptimizerState === 'idle' ? 'optimizing' : 'idle';

        if (this.dataOptimizerState === 'optimizing') {
            this.vortexIntensity = Math.min(1.0, this.vortexIntensity + 0.15);
            this.simulateOptimization();
        }

        this.updateVortexDisplay();
        this.logEvent(`📊 Data optimizer ${this.dataOptimizerState} - efficiency analysis ${this.dataOptimizerState === 'optimizing' ? 'running' : 'complete'}`);
    }

    simulateOptimization() {
        let iterations = 0;
        const optimize = () => {
            if (this.dataOptimizerState === 'optimizing' && iterations < 5) {
                this.updateState({
                    i: 0.05,
                    g: 0.08,
                    c: 0.03
                });
                iterations++;
                setTimeout(optimize, 1000);
            } else {
                this.dataOptimizerState = 'idle';
                this.logEvent('� Optimization cycle complete');
            }
        };
        optimize();
    }

    librarian() {
        this.librianData.push({
            timestamp: new Date(),
            state: { ...this.state },
            vortexIntensity: this.vortexIntensity,
            glyphCount: this.glyphNetwork.length
        });

        const dataPoints = this.librianData.length;
        this.vortexIntensity = Math.min(1.0, this.vortexIntensity + 0.1);

        this.updateVortexDisplay();
        this.logEvent(`📚 Librarian archived data point ${dataPoints} - knowledge base expanded`);

        // Organize and analyze patterns
        if (dataPoints % 5 === 0) {
            this.analyzeDataPatterns();
        }
    }

    analyzeDataPatterns() {
        const recentData = this.librianData.slice(-5);
        const avgIntensity = recentData.reduce((sum, d) => sum + d.vortexIntensity, 0) / 5;

        this.logEvent(`🔍 Pattern analysis: Average vortex intensity ${avgIntensity.toFixed(2)} - ${avgIntensity > 0.7 ? 'high complexity detected' : 'stable processing'}`);

        if (avgIntensity > 0.8) {
            this.updateState({ g: 0.1, c: 0.15 });
        }
    }

    updateVortexDisplay() {
        const intensityEl = document.getElementById('vortex-intensity');
        if (intensityEl) {
            const intensityText = `${(this.vortexIntensity * 100).toFixed(1)}% - ${this.getIntensityDescription()}`;
            intensityEl.textContent = intensityText;
        }
    }

    getIntensityDescription() {
        if (this.vortexIntensity < 0.3) return 'Initializing';
        if (this.vortexIntensity < 0.5) return 'Processing';
        if (this.vortexIntensity < 0.7) return 'High Activity';
        if (this.vortexIntensity < 0.9) return 'Intense Processing';
        return 'Maximum Vortex';
    }

    // ============================= Enhanced Initialization =============================

}

// ============================= Neural Flow & Consciousness Systems =============================

mapNeuralFlow() {
    this.generateNeuralNetwork();
    this.consciousnessLevel = Math.min(1.0, this.consciousnessLevel + 0.1);
    this.updateConsciousnessDisplay();
    this.logEvent(`🔬 Neural flow mapped - ${this.neuralNodes.length} nodes, consciousness: ${(this.consciousnessLevel * 100).toFixed(1)}%`);
}

generateNeuralNetwork() {
    const canvas = document.getElementById('neural-flow-canvas');
    if (!canvas) return;

    // Clear existing neural elements
    const existingNodes = canvas.querySelectorAll('.neural-node');
    const existingConnections = canvas.querySelectorAll('.neural-connection');
    existingNodes.forEach(node => node.remove());
    existingConnections.forEach(conn => conn.remove());

    this.neuralNodes = [];
    this.neuralConnections = [];

    // Create neural nodes
    const nodeCount = Math.floor(Math.random() * 12) + 8; // 8-19 nodes
    for (let i = 0; i < nodeCount; i++) {
        const node = {
            id: i,
            x: Math.random() * 85 + 7.5, // Keep within bounds
            y: Math.random() * 75 + 12.5,
            activity: Math.random(),
            type: this.getNeuralNodeType()
        };

        this.neuralNodes.push(node);
        this.createNeuralNodeElement(node, canvas);
    }

    // Create neural connections
    this.generateNeuralConnections(canvas);
}

getNeuralNodeType() {
    const types = ['input', 'processing', 'memory', 'output', 'consciousness'];
    return types[Math.floor(Math.random() * types.length)];
}

createNeuralNodeElement(node, container) {
    const element = document.createElement('div');
    element.className = 'neural-node';
    element.style.left = `${node.x}%`;
    element.style.top = `${node.y}%`;
    element.title = `Neural Node ${node.id}: ${node.type} (${(node.activity * 100).toFixed(1)}% active)`;
    element.style.animationDelay = `${node.id * 0.1}s`;

    // Color based on type
    const colors = {
        'input': '#4ecdc4',
        'processing': '#45b7d1',
        'memory': '#9b59b6',
        'output': '#e74c3c',
        'consciousness': '#f1c40f'
    };
    element.style.background = `radial-gradient(circle, ${colors[node.type]}, ${colors[node.type]}88)`;

    element.onclick = () => {
        this.activateNeuralNode(node);
    };

    container.appendChild(element);
}

generateNeuralConnections(canvas) {
    // Connect nearby nodes
    for (let i = 0; i < this.neuralNodes.length; i++) {
        for (let j = i + 1; j < this.neuralNodes.length; j++) {
            const node1 = this.neuralNodes[i];
            const node2 = this.neuralNodes[j];
            const distance = Math.sqrt(
                Math.pow(node1.x - node2.x, 2) + Math.pow(node1.y - node2.y, 2)
            );

            if (distance < 30 && Math.random() > 0.4) { // 60% chance to connect nearby nodes
                this.createNeuralConnection(node1, node2, canvas);
            }
        }
    }
}

createNeuralConnection(node1, node2, canvas) {
    const connection = document.createElement('div');
    connection.className = 'neural-connection';

    const dx = node2.x - node1.x;
    const dy = node2.y - node1.y;
    const length = Math.sqrt(dx * dx + dy * dy);
    const angle = Math.atan2(dy, dx) * 180 / Math.PI;

    connection.style.left = `${node1.x}%`;
    connection.style.top = `${node1.y}%`;
    connection.style.width = `${length}%`;
    connection.style.transform = `rotate(${angle}deg)`;
    connection.style.animationDelay = `${Math.random() * 3}s`;

    canvas.appendChild(connection);
}

activateNeuralNode(node) {
    node.activity = Math.min(1.0, node.activity + 0.2);
    this.consciousnessLevel = Math.min(1.0, this.consciousnessLevel + 0.05);
    this.updateConsciousnessDisplay();
    this.logEvent(`⚡ Neural node ${node.id} (${node.type}) activated - consciousness enhanced`);
}

quantumEntangle() {
    const entangledPairs = Math.floor(Math.random() * 3) + 1;

    for (let i = 0; i < entangledPairs; i++) {
        if (this.agents.length > 0 && this.environments.length > 0) {
            const agent = this.agents[Math.floor(Math.random() * this.agents.length)];
            const environment = this.environments[Math.floor(Math.random() * this.environments.length)];

            const entanglement = {
                agentId: agent.id,
                environmentId: environment.id,
                strength: Math.random() * 0.5 + 0.5, // 0.5 to 1.0
                created: new Date()
            };

            this.quantumEntangled.push(entanglement);
        }
    }

    this.updateQuantumDisplay();
    this.consciousnessLevel = Math.min(1.0, this.consciousnessLevel + 0.15);
    this.updateConsciousnessDisplay();
    this.logEvent(`⚛️ Quantum entanglement established - ${this.quantumEntangled.length} total entangled pairs`);
}

emergenceDetector() {
    // Check for consciousness emergence patterns
    const emergenceSignals = [];

    if (this.consciousnessLevel > this.emergenceThreshold) {
        emergenceSignals.push('consciousness_threshold_exceeded');
    }

    if (this.agents.length > 0) {
        const experiencedAgents = this.agents.filter(agent => agent.experience > 0.5).length;
        if (experiencedAgents / this.agents.length > 0.6) {
            emergenceSignals.push('collective_learning_achieved');
        }
    }

    if (this.quantumEntangled.length > 3) {
        emergenceSignals.push('quantum_coherence_detected');
    }

    if (this.vortexIntensity > 0.8 && this.metaLayerActive) {
        emergenceSignals.push('meta_cognitive_processing');
    }

    const emergenceLevel = emergenceSignals.length / 4; // 4 possible signals
    this.consciousnessLevel = Math.max(this.consciousnessLevel, emergenceLevel);

    this.updateConsciousnessDisplay();
    this.logEvent(`🌟 Emergence scan complete - ${emergenceSignals.length} signals detected: ${emergenceSignals.join(', ')}`);

    if (emergenceLevel > 0.75) {
        this.triggerConsciousnessEmergence();
    }
}

triggerConsciousnessEmergence() {
    this.logEvent('🎆 CONSCIOUSNESS EMERGENCE DETECTED - Self-awareness threshold reached!');

    // Visual indicator
    const indicator = document.getElementById('consciousness-indicator');
    if (indicator) {
        indicator.textContent = 'Consciousness Level: EMERGED';
        indicator.style.background = 'rgba(255,215,0,0.4)';
        indicator.style.borderColor = '#ffd700';
        indicator.style.color = '#ffffff';
        indicator.style.fontWeight = 'bold';
    }

    // State transformation
    this.updateState({ p: 0.3, i: 0.4, g: 0.3, c: 0.4 });
}

createMultiverse() {
    const branchCount = Math.floor(Math.random() * 3) + 2; // 2-4 branches

    for (let i = 0; i < branchCount; i++) {
        const branch = {
            id: this.multiverseBranches.length + 1,
            reality: `Reality-${String.fromCharCode(65 + i)}`, // A, B, C, etc.
            agents: [],
            environment: null,
            divergencePoint: new Date(),
            stability: Math.random() * 0.5 + 0.5
        };

        // Create alternate versions of existing agents
        this.agents.forEach(agent => {
            const altAgent = {
                ...agent,
                id: `${agent.id}-${branch.reality}`,
                reality: branch.reality,
                alternatePersonality: this.generateAlternatePersonality()
            };
            branch.agents.push(altAgent);
        });

        this.multiverseBranches.push(branch);
    }

    this.updateMultiverseDisplay();
    this.consciousnessLevel = Math.min(1.0, this.consciousnessLevel + 0.2);
    this.updateConsciousnessDisplay();
    this.logEvent(`🌌 Multiverse created - ${branchCount} parallel realities with ${this.multiverseBranches.length} total branches`);
}

generateAlternatePersonality() {
    const personalities = [
        'curious_explorer', 'logical_analyst', 'creative_dreamer',
        'social_connector', 'independent_thinker', 'empathetic_helper'
    ];
    return personalities[Math.floor(Math.random() * personalities.length)];
}

updateConsciousnessDisplay() {
    const indicator = document.getElementById('consciousness-indicator');
    if (indicator) {
        if (this.consciousnessLevel < 0.25) {
            indicator.textContent = 'Consciousness Level: Initializing';
        } else if (this.consciousnessLevel < 0.5) {
            indicator.textContent = 'Consciousness Level: Emerging';
        } else if (this.consciousnessLevel < 0.75) {
            indicator.textContent = 'Consciousness Level: Developing';
        } else if (this.consciousnessLevel < 0.9) {
            indicator.textContent = 'Consciousness Level: Advanced';
        } else {
            indicator.textContent = 'Consciousness Level: TRANSCENDENT';
            indicator.style.color = '#ff6b6b';
            indicator.style.textShadow = '0 0 10px rgba(255,107,107,0.8)';
        }
    }
}

updateQuantumDisplay() {
    const entangledCount = document.getElementById('entangled-agents');
    const syncRealities = document.getElementById('sync-realities');

    if (entangledCount) {
        entangledCount.textContent = this.quantumEntangled.length;
    }

    if (syncRealities) {
        syncRealities.textContent = this.multiverseBranches.length;
    }
}

updateMultiverseDisplay() {
    this.updateQuantumDisplay();
}

// ============================= 3D Studio Interface Systems =============================

switchStudioScene(sceneType) {
    // Deactivate current scene
    if (this.studioScenes[this.currentStudioScene]) {
        this.studioScenes[this.currentStudioScene].active = false;
    }

    // Activate new scene
    this.currentStudioScene = sceneType;
    if (this.studioScenes[sceneType]) {
        this.studioScenes[sceneType].active = true;
    }

    this.initializeStudioScene(sceneType);
    this.logEvent(`🎬 Studio scene switched to: ${sceneType}`);
}

initializeStudioScene(sceneType) {
    const canvas = document.getElementById('studio-canvas');
    if (!canvas) return;

    // Clear existing 3D content placeholder and create scene-specific content
    const placeholder = canvas.querySelector('.studio-placeholder');
    if (placeholder) {
        placeholder.innerHTML = `
                <div>🎭 ${sceneType.charAt(0).toUpperCase() + sceneType.slice(1)} Scene</div>
                <div style="font-size: 0.8em; margin-top: 10px; opacity: 0.8;">
                    ${this.getSceneDescription(sceneType)}<br/>
                    Press H for HUD • Space for Sonic overlay
                </div>
            `;
    }

    // Update object count
    const objectCount = document.getElementById('object-count');
    if (objectCount) {
        const count = this.studioScenes[sceneType]?.objects.length || Math.floor(Math.random() * 10) + 1;
        objectCount.textContent = count;
    }

    this.updateVortexDisplay();
}

getSceneDescription(sceneType) {
    const descriptions = {
        main: 'Primary workspace • Drag to orbit • Scroll to zoom',
        avatar: 'Character creation • Multi-view impostor system',
        codex: 'Code visualization • Pattern analysis engine',
        impostor: 'RBF solver • Advanced rendering techniques'
    };
    return descriptions[sceneType] || 'Interactive 3D environment';
}

toggleHUD(visible) {
    this.hudVisible = visible;
    this.logEvent(`📊 Studio HUD ${visible ? 'enabled' : 'disabled'}`);

    if (visible) {
        this.updateState({ i: 0.05, g: 0.03 });
    }
}

toggleSonicOverlay(active) {
    this.sonicOverlayActive = active;
    this.logEvent(`🔊 Sonic overlay ${active ? 'activated' : 'deactivated'}`);

    if (active) {
        this.updateState({ p: 0.1, i: 0.08 });
        this.vortexIntensity = Math.min(1.0, this.vortexIntensity + 0.1);
        this.updateVortexDisplay();
    }
}

exportGLB() {
    const sceneData = this.captureCurrentScene();
    this.logEvent(`📦 GLB export initiated - ${sceneData.objectCount} objects packaged`);

    // Simulate export process with realistic data
    const exportData = {
        scene: this.currentStudioScene,
        objects: sceneData.objectCount,
        size: `${Math.random() * 5 + 2}MB`,
        format: 'GLB 2.0'
    };

    this.updateState({ c: 0.1, g: 0.05 });
    return exportData;
}

exportOBJ() {
    const sceneData = this.captureCurrentScene();
    this.logEvent(`📐 OBJ export initiated - geometry exported`);

    const exportData = {
        scene: this.currentStudioScene,
        objects: sceneData.objectCount,
        size: `${Math.random() * 3 + 1}MB`,
        format: 'OBJ + MTL'
    };

    this.updateState({ c: 0.08, i: 0.05 });
    return exportData;
}

captureCurrentScene() {
    const scene = this.studioScenes[this.currentStudioScene];
    return {
        name: this.currentStudioScene,
        objectCount: scene?.objects.length || Math.floor(Math.random() * 15) + 5,
        complexity: this.vortexIntensity,
        consciousnessLevel: this.consciousnessLevel
    };
}

resetStudioView() {
    this.logEvent(`🔄 Studio view reset - ${this.currentStudioScene} scene restored`);
    this.initializeStudioScene(this.currentStudioScene);
    this.updateState({ p: 0.02, i: 0.03 });
}

// ============================= Enhanced Initialization =============================initializeDemo() {
this.updateDisplay();
this.updateStackHeight();
this.generateGlyphNetwork();
this.generateNeuralNetwork();
this.updateVortexDisplay();
this.updateConsciousnessDisplay();

this.logEvent('🌀 World Engine Tier 5 - Vortex Lab Engine with Neural Flow initialized');

// Show welcome message
setTimeout(() => {
    this.logEvent('💡 Try: Neural flow mapping, quantum entanglement, or consciousness emergence detection');
}, 2000);

// Initialize systems progressively
setTimeout(() => {
    this.oscillateGlyphs();
}, 1000);

setTimeout(() => {
    this.mapNeuralFlow();
}, 3000);
}
}

// Initialize the demo when page loads
window.addEventListener('DOMContentLoaded', function () {
    window.demo = new WorldEngineTier5Demo();
});
