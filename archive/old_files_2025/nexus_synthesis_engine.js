/**
 * NEXUS SYNTHESIS ENGINE
 * ======================
 *
 * Unified mathematical expression system that combines:
 * • Vibe Engine (4D audio-driven state)
 * • Math Expression Engine (equation parsing/evaluation)
 * • Art Synthesizer (2D/3D visual generation)
 * • Sound Synthesizer (audio waveform generation)
 * • World Synthesizer (3D terrain/geometry)
 * • NEXUS FORGE PRIMORDIAL (AI intelligence system)
 */

class NexusSynthesisEngine {
    constructor() {
        // Core systems
        this.vibeEngine = null;
        this.mathEngine = null;
        this.artEngine = null;
        this.soundEngine = null;
        this.worldEngine = null;
        this.nexusForge = null;

        // Synthesis contexts
        this.canvasArt = null;
        this.canvasWorld = null;
        this.audioContext = null;

        // Expression libraries
        this.artExpressions = new Map();
        this.soundExpressions = new Map();
        this.worldExpressions = new Map();

        // Real-time state
        this.isActive = false;
        this.frameCount = 0;
        this.lastUpdate = 0;

        console.log('🌊 NEXUS SYNTHESIS ENGINE initialized');
    }

    async initialize() {
        try {
            // Initialize math expression engine
            this.mathEngine = new ExpressionEngine();
            this.setupMathematicalFunctions();

            // Initialize canvas contexts
            this.initializeCanvases();

            // Initialize audio context
            await this.initializeAudio();

            // Setup default expressions
            this.loadDefaultExpressions();

            // Connect to Vibe Engine integration
            this.connectVibeEngine();

            console.log('✅ All synthesis engines online');
            return true;
        } catch (error) {
            console.error('❌ Synthesis engine initialization failed:', error);
            return false;
        }
    }

    setupMathematicalFunctions() {
        // Enhanced math functions for synthesis
        this.mathEngine.defineFunction('noise', (x, y = 0, z = 0) => {
            // Simple pseudo-noise function
            const hash = Math.sin(x * 12.9898 + y * 78.233 + z * 37.719) * 43758.5453;
            return (hash - Math.floor(hash)) * 2 - 1;
        }, -1);

        this.mathEngine.defineFunction('fbm', (x, y, octaves = 4) => {
            let value = 0;
            let amplitude = 1;
            for (let i = 0; i < octaves; i++) {
                value += this.mathEngine.functions.noise.fn(x, y) * amplitude;
                x *= 2;
                y *= 2;
                amplitude *= 0.5;
            }
            return value;
        }, -1);

        // Vibe state functions
        this.mathEngine.defineFunction('vibe_p', () => this.getVibeState().p, 0);
        this.mathEngine.defineFunction('vibe_i', () => this.getVibeState().i, 0);
        this.mathEngine.defineFunction('vibe_g', () => this.getVibeState().g, 0);
        this.mathEngine.defineFunction('vibe_c', () => this.getVibeState().c, 0);
        this.mathEngine.defineFunction('vibe_time', () => this.frameCount * 0.016, 0);

        // Special synthesis functions
        this.mathEngine.defineFunction('heart', (x, y) => {
            // Heart equation: (x^2 + y^2 - 1)^3 = x^2 * y^3
            const left = Math.pow(x * x + y * y - 1, 3);
            const right = x * x * y * y * y;
            return Math.abs(left - right) < 0.1 ? 1 : 0;
        }, 2);

        this.mathEngine.defineFunction('rose', (t, n = 5) => {
            // Rose curve: r = cos(n*θ)
            return Math.cos(n * t);
        }, 2);

        this.mathEngine.defineFunction('spiral', (t, a = 1, b = 0.1) => {
            // Archimedean spiral: r = a + b*θ
            return a + b * t;
        }, 3);
    }

    initializeCanvases() {
        // Create synthesis canvases
        this.canvasArt = document.createElement('canvas');
        this.canvasArt.width = 800;
        this.canvasArt.height = 600;
        this.ctxArt = this.canvasArt.getContext('2d');

        this.canvasWorld = document.createElement('canvas');
        this.canvasWorld.width = 800;
        this.canvasWorld.height = 600;
        this.ctxWorld = this.canvasWorld.getContext('2d');
    }

    async initializeAudio() {
        this.audioContext = new (window.AudioContext || window.webkitAudioContext)();
        this.masterGain = this.audioContext.createGain();
        this.masterGain.connect(this.audioContext.destination);
        this.masterGain.gain.value = 0.3;

        // Create audio synthesis buffer
        this.audioBuffer = this.audioContext.createBuffer(1, 4096, this.audioContext.sampleRate);
        this.audioSource = null;
    }

    loadDefaultExpressions() {
        // Art expressions
        this.artExpressions.set('heart', '(x^2 + y^2 - 1)^3 - x^2 * y^3');
        this.artExpressions.set('rose', 'r - cos(5 * theta)');
        this.artExpressions.set('spiral', 'r - (0.5 + 0.1 * theta + vibe_i() * sin(theta))');
        this.artExpressions.set('vibe_circle', '(x - vibe_p())^2 + (y - vibe_i())^2 - (0.5 + vibe_c())^2');
        this.artExpressions.set('dragon', 'x^2 + y^2 - (1 + vibe_g() * cos(8 * atan(y/x)))^2');

        // Sound expressions
        this.soundExpressions.set('sine_wave', 'sin(2 * pi * 440 * t + vibe_p())');
        this.soundExpressions.set('fm_bell', 'sin(2 * pi * 440 * t + vibe_i() * sin(2 * pi * 220 * t))');
        this.soundExpressions.set('harmonics', 'sin(2*pi*440*t) + 0.5*sin(2*pi*880*t) + 0.25*sin(2*pi*1320*t)');
        this.soundExpressions.set('vibe_drone', 'sin(2*pi*(200 + vibe_c()*400)*t) * (0.5 + 0.5*sin(vibe_i()*t))');

        // World expressions
        this.worldExpressions.set('mountains', 'sin(x + vibe_time()) * cos(y + vibe_p()) + vibe_i() * fbm(x, y)');
        this.worldExpressions.set('crater', 'exp(-(x^2 + y^2)) * (1 + vibe_g() * sin(vibe_time()))');
        this.worldExpressions.set('vibe_terrain', 'noise(x, y) + 0.5*noise(2*x, 2*y) + vibe_c()*sin(sqrt(x^2+y^2))');
        this.worldExpressions.set('ocean_waves', 'sin(x - vibe_time()) + 0.5*sin(2*y + vibe_p()) + 0.3*noise(x,y)');
    }

    // Expression synthesis methods
    synthesizeArt(expression, vars = {}) {
        try {
            const compiled = this.mathEngine.compile(expression);
            const ctx = this.ctxArt;
            const canvas = this.canvasArt;

            // Clear canvas with quantum background
            const gradient = ctx.createRadialGradient(canvas.width / 2, canvas.height / 2, 0, canvas.width / 2, canvas.height / 2, canvas.width / 2);
            gradient.addColorStop(0, 'rgba(26, 0, 51, 0.1)');
            gradient.addColorStop(1, 'rgba(0, 0, 17, 1)');
            ctx.fillStyle = gradient;
            ctx.fillRect(0, 0, canvas.width, canvas.height);

            // Render equation as art
            const resolution = 2;
            const scale = 4;
            const centerX = canvas.width / 2;
            const centerY = canvas.height / 2;

            for (let px = 0; px < canvas.width; px += resolution) {
                for (let py = 0; py < canvas.height; py += resolution) {
                    const x = (px - centerX) / (canvas.width / scale);
                    const y = (py - centerY) / (canvas.height / scale);
                    const r = Math.sqrt(x * x + y * y);
                    const theta = Math.atan2(y, x);

                    try {
                        const value = compiled.evaluate({
                            x, y, r, theta,
                            t: this.frameCount * 0.016,
                            ...vars
                        });

                        if (Math.abs(value) < 0.1) {
                            // On the curve - bright quantum color
                            const vibeState = this.getVibeState();
                            const hue = (vibeState.p + 1) * 180; // Map -1,1 to 0,360
                            const sat = 50 + vibeState.i * 50;
                            const light = 50 + vibeState.c * 30;

                            ctx.fillStyle = `hsla(${hue}, ${sat}%, ${light}%, 0.8)`;
                            ctx.fillRect(px, py, resolution, resolution);

                            // Add glow effect on high energy
                            if (vibeState.i > 0.7) {
                                ctx.shadowColor = `hsla(${hue}, ${sat}%, ${light + 20}%, 0.6)`;
                                ctx.shadowBlur = 5;
                                ctx.fillRect(px, py, resolution, resolution);
                                ctx.shadowBlur = 0;
                            }
                        }
                    } catch (evalError) {
                        // Skip invalid points
                    }
                }
            }

            return canvas;
        } catch (error) {
            console.error('Art synthesis failed:', error);
            return null;
        }
    }

    synthesizeSound(expression, duration = 1.0, frequency = 440) {
        try {
            const compiled = this.mathEngine.compile(expression);
            const sampleRate = this.audioContext.sampleRate;
            const numSamples = Math.floor(duration * sampleRate);

            const buffer = this.audioContext.createBuffer(1, numSamples, sampleRate);
            const channelData = buffer.getChannelData(0);

            for (let i = 0; i < numSamples; i++) {
                const t = i / sampleRate;
                try {
                    const sample = compiled.evaluate({
                        t,
                        freq: frequency,
                        pi: Math.PI
                    });

                    // Clamp to valid audio range
                    channelData[i] = Math.max(-1, Math.min(1, sample));
                } catch (evalError) {
                    channelData[i] = 0;
                }
            }

            return buffer;
        } catch (error) {
            console.error('Sound synthesis failed:', error);
            return null;
        }
    }

    synthesizeWorld(expression, size = 64) {
        try {
            const compiled = this.mathEngine.compile(expression);
            const heightmap = [];
            const scale = 8;

            for (let i = 0; i < size; i++) {
                heightmap[i] = [];
                for (let j = 0; j < size; j++) {
                    const x = (i / size - 0.5) * scale;
                    const y = (j / size - 0.5) * scale;

                    try {
                        const height = compiled.evaluate({ x, y });
                        heightmap[i][j] = Math.max(-2, Math.min(2, height));
                    } catch (evalError) {
                        heightmap[i][j] = 0;
                    }
                }
            }

            // Render heightmap to world canvas
            this.renderHeightmap(heightmap, size);

            return heightmap;
        } catch (error) {
            console.error('World synthesis failed:', error);
            return null;
        }
    }

    renderHeightmap(heightmap, size) {
        const ctx = this.ctxWorld;
        const canvas = this.canvasWorld;
        const cellSize = Math.min(canvas.width, canvas.height) / size;

        ctx.fillStyle = '#000011';
        ctx.fillRect(0, 0, canvas.width, canvas.height);

        for (let i = 0; i < size; i++) {
            for (let j = 0; j < size; j++) {
                const height = heightmap[i][j];
                const intensity = (height + 2) / 4; // Normalize to 0-1

                // Color based on height and vibe state
                const vibeState = this.getVibeState();
                const baseHue = height > 0 ? 120 : 240; // Green for high, blue for low
                const hue = baseHue + vibeState.p * 60;
                const sat = 70 + vibeState.g * 30;
                const light = 20 + intensity * 60 + vibeState.i * 20;

                ctx.fillStyle = `hsla(${hue}, ${sat}%, ${light}%, 0.9)`;
                ctx.fillRect(i * cellSize, j * cellSize, cellSize, cellSize);
            }
        }
    }

    // Integration methods
    connectVibeEngine() {
        // Connect to existing vibe engine integration if available
        if (window.VibeEngineIntegration) {
            this.vibeIntegration = new VibeEngineIntegration();
        }
    }

    getVibeState() {
        if (this.vibeIntegration) {
            return this.vibeIntegration.getVibeState();
        }
        // Fallback simulation
        const time = this.frameCount * 0.01;
        return {
            p: Math.sin(time * 0.7) * 0.5,
            i: 0.3 + Math.sin(time * 1.2) * 0.2,
            g: 0.4 + Math.cos(time * 0.9) * 0.3,
            c: 0.6 + Math.sin(time * 0.5) * 0.2
        };
    }

    // Real-time synthesis loop
    startSynthesis() {
        this.isActive = true;
        this.lastUpdate = performance.now();
        this.synthesisLoop();
        console.log('🌊 Real-time synthesis started');
    }

    stopSynthesis() {
        this.isActive = false;
        console.log('🔇 Synthesis stopped');
    }

    synthesisLoop() {
        if (!this.isActive) return;

        const now = performance.now();
        const dt = (now - this.lastUpdate) / 1000;
        this.lastUpdate = now;
        this.frameCount++;

        // Update all synthesis engines based on vibe state
        const vibeState = this.getVibeState();

        // Real-time art synthesis (every few frames)
        if (this.frameCount % 3 === 0) {
            const artExpression = this.getCurrentArtExpression();
            this.synthesizeArt(artExpression);
        }

        // Real-time world synthesis (less frequent)
        if (this.frameCount % 10 === 0) {
            const worldExpression = this.getCurrentWorldExpression();
            this.synthesizeWorld(worldExpression);
        }

        // Continuous audio synthesis
        if (vibeState.i > 0.3) {
            const soundExpression = this.getCurrentSoundExpression();
            this.updateAudioSynthesis(soundExpression);
        }

        requestAnimationFrame(() => this.synthesisLoop());
    }

    getCurrentArtExpression() {
        const vibeState = this.getVibeState();
        // Choose expression based on vibe energy
        if (vibeState.i > 0.8) return this.artExpressions.get('dragon');
        if (vibeState.c > 0.7) return this.artExpressions.get('rose');
        if (vibeState.p < -0.3) return this.artExpressions.get('heart');
        return this.artExpressions.get('vibe_circle');
    }

    getCurrentSoundExpression() {
        const vibeState = this.getVibeState();
        if (vibeState.g > 0.6) return this.soundExpressions.get('fm_bell');
        if (vibeState.c > 0.8) return this.soundExpressions.get('harmonics');
        return this.soundExpressions.get('vibe_drone');
    }

    getCurrentWorldExpression() {
        const vibeState = this.getVibeState();
        if (vibeState.i > 0.7) return this.worldExpressions.get('mountains');
        if (vibeState.p < -0.2) return this.worldExpressions.get('crater');
        return this.worldExpressions.get('vibe_terrain');
    }

    updateAudioSynthesis(expression) {
        // Stop previous audio
        if (this.audioSource) {
            this.audioSource.stop();
        }

        // Synthesize new buffer
        const buffer = this.synthesizeSound(expression, 0.5);
        if (buffer) {
            this.audioSource = this.audioContext.createBufferSource();
            this.audioSource.buffer = buffer;
            this.audioSource.connect(this.masterGain);
            this.audioSource.loop = true;
            this.audioSource.start();
        }
    }

    // Public API
    evaluateExpression(expression, vars = {}) {
        return this.mathEngine.evaluate(expression, vars);
    }

    addCustomExpression(type, name, expression) {
        switch (type) {
            case 'art':
                this.artExpressions.set(name, expression);
                break;
            case 'sound':
                this.soundExpressions.set(name, expression);
                break;
            case 'world':
                this.worldExpressions.set(name, expression);
                break;
        }
    }

    getCanvases() {
        return {
            art: this.canvasArt,
            world: this.canvasWorld
        };
    }

    // Integration with NEXUS FORGE
    connectNexusForge(nexusSystem) {
        this.nexusForge = nexusSystem;

        // Hook into AI events
        if (nexusSystem.on) {
            nexusSystem.on('painDetected', (level) => {
                // Modify synthesis based on pain level
                if (level > 0.7) {
                    this.addCustomExpression('art', 'pain_visualization',
                        `sin(x*${level * 10}) * cos(y*${level * 10}) + ${level}`);
                }
            });

            nexusSystem.on('intelligenceGenerated', (intelligence) => {
                // Create art from intelligence patterns
                const complexity = intelligence.patterns?.length || 1;
                this.addCustomExpression('art', 'intelligence_mandala',
                    `sin(r*${complexity}) * cos(theta*${complexity}) + vibe_c()`);
            });
        }
    }
}

// Global access
window.NexusSynthesisEngine = NexusSynthesisEngine;
