/**
 * Enhanced Web Audio Integration for Audio Semantic Engine
 * Real-time microphone processing with the enhanced mathematical model
 */

import { EnhancedAudioSemanticEngine } from './enhanced-audio-semantic-engine.js';

class EnhancedWebAudioProcessor {
    constructor() {
        this.audioContext = null;
        this.microphone = null;
        this.analyser = null;
        this.processor = null;
        this.engine = null;
        this.isProcessing = false;

        // Audio settings
        this.sampleRate = 44100;
        this.bufferSize = 1024;
        this.fftSize = 2048;

        // Visualization data
        this.visualizationData = {
            waveform: new Float32Array(this.bufferSize),
            spectrum: new Float32Array(this.fftSize / 2),
            state: [0, 0.5, 0.3, 0.6],
            tempo: 120,
            phase: 0,
            events: {},
            features: {},
            visual: {
                hue: 200,
                brightness: 55,
                grain: 0.3,
                reverb: 0.6
            }
        };

        // Event callbacks
        this.callbacks = new Map();

        this.init();
    }

    async init() {
        try {
            // Initialize Web Audio Context with proper settings
            this.audioContext = new (window.AudioContext || window.webkitAudioContext)({
                sampleRate: this.sampleRate,
                latencyHint: 'interactive'
            });

            // Initialize Enhanced Audio Semantic Engine
            this.engine = new EnhancedAudioSemanticEngine(this.sampleRate, this.bufferSize);

            // Connect engine events to visualization
            this.engine.addEventListener('stateUpdate', (output) => {
                this.updateVisualizationData(output);
            });

            console.log('Enhanced Audio Semantic Processor initialized');

        } catch (error) {
            console.error('Failed to initialize enhanced audio processor:', error);
        }
    }

    async startProcessing() {
        if (this.isProcessing) return;

        try {
            // Request microphone access with high-quality settings
            const stream = await navigator.mediaDevices.getUserMedia({
                audio: {
                    echoCancellation: false,
                    noiseSuppression: false,
                    autoGainControl: false,
                    sampleRate: this.sampleRate,
                    channelCount: 1
                }
            });

            // Resume audio context if suspended
            if (this.audioContext.state === 'suspended') {
                await this.audioContext.resume();
            }

            // Create audio processing graph
            this.microphone = this.audioContext.createMediaStreamSource(stream);
            this.analyser = this.audioContext.createAnalyser();
            this.analyser.fftSize = this.fftSize;
            this.analyser.smoothingTimeConstant = 0.0; // No smoothing for accurate analysis
            this.analyser.maxDecibels = -10;
            this.analyser.minDecibels = -90;

            // Create ScriptProcessor for frame-by-frame analysis
            this.processor = this.audioContext.createScriptProcessor(this.bufferSize, 1, 1);

            // Connect audio graph
            this.microphone.connect(this.analyser);
            this.analyser.connect(this.processor);
            this.processor.connect(this.audioContext.destination);

            // Process audio frames with enhanced engine
            this.processor.onaudioprocess = (event) => {
                this.processAudioFrame(event);
            };

            this.isProcessing = true;
            this.emit('started');

            console.log('Enhanced audio processing started');

        } catch (error) {
            console.error('Failed to start enhanced audio processing:', error);
            this.emit('error', error);
        }
    }

    stopProcessing() {
        if (!this.isProcessing) return;

        try {
            // Disconnect and cleanup audio nodes
            if (this.microphone) {
                this.microphone.disconnect();
                this.microphone = null;
            }

            if (this.analyser) {
                this.analyser.disconnect();
                this.analyser = null;
            }

            if (this.processor) {
                this.processor.disconnect();
                this.processor.onaudioprocess = null;
                this.processor = null;
            }

            this.isProcessing = false;
            this.emit('stopped');

            console.log('Enhanced audio processing stopped');

        } catch (error) {
            console.error('Failed to stop enhanced audio processing:', error);
        }
    }

    processAudioFrame(event) {
        if (!this.engine) return;

        const inputBuffer = event.inputBuffer.getChannelData(0);
        const timestamp = this.audioContext.currentTime;

        // Process with enhanced semantic engine
        const output = this.engine.processFrame(inputBuffer, timestamp);

        // Update visualization data
        this.visualizationData.waveform.set(inputBuffer);

        // Get frequency spectrum from analyser
        this.analyser.getFloatFrequencyData(this.visualizationData.spectrum);

        // Emit processed frame data
        this.emit('audioFrame', {
            waveform: this.visualizationData.waveform,
            spectrum: this.visualizationData.spectrum,
            semantics: output,
            timestamp: timestamp
        });
    }

    updateVisualizationData(output) {
        // Update core visualization data
        this.visualizationData.state = output.state.slice();
        this.visualizationData.tempo = output.clock.bpm;
        this.visualizationData.phase = output.clock.phase;
        this.visualizationData.events = { ...output.events };
        this.visualizationData.features = { ...output.features };

        // Update visual parameters for graphics/synthesis
        this.visualizationData.visual = {
            hue: output.hue,
            brightness: output.brightness,
            grain: output.grain,
            reverb: output.reverb
        };

        // Emit semantic update
        this.emit('semanticUpdate', output);
    }

    // === Data Access Methods ===

    getSemanticState() {
        return this.engine ? this.engine.getSemanticState() : null;
    }

    getTemporalState() {
        return this.engine ? this.engine.getTemporalState() : null;
    }

    getVisualizationData() {
        return this.visualizationData;
    }

    getVisualParameters() {
        return this.visualizationData.visual;
    }

    getCurrentState() {
        return {
            semantic: this.getSemanticState(),
            temporal: this.getTemporalState(),
            visual: this.getVisualParameters(),
            isProcessing: this.isProcessing,
            features: this.visualizationData.features,
            events: this.visualizationData.events
        };
    }

    // === World Engine Integration ===

    sendToWorldEngine(data) {
        // Send comprehensive semantic data to World Engine
        window.postMessage({
            type: 'ENHANCED_AUDIO_SEMANTIC_UPDATE',
            payload: {
                semantic: data.state,
                temporal: {
                    bpm: data.clock.bpm,
                    phase: data.clock.phase
                },
                visual: {
                    hue: data.hue,
                    brightness: data.brightness,
                    grain: data.grain,
                    reverb: data.reverb
                },
                events: data.events,
                operators: data.operators,
                features: data.features,
                timestamp: data.timestamp
            }
        }, '*');
    }

    // === Event System ===

    on(eventType, callback) {
        if (!this.callbacks.has(eventType)) {
            this.callbacks.set(eventType, []);
        }
        this.callbacks.get(eventType).push(callback);
    }

    off(eventType, callback) {
        if (this.callbacks.has(eventType)) {
            const callbacks = this.callbacks.get(eventType);
            const index = callbacks.indexOf(callback);
            if (index > -1) {
                callbacks.splice(index, 1);
            }
        }
    }

    emit(eventType, data = null) {
        if (this.callbacks.has(eventType)) {
            this.callbacks.get(eventType).forEach(callback => {
                try {
                    callback(data);
                } catch (error) {
                    console.error(`Error in ${eventType} callback:`, error);
                }
            });
        }
    }

    // === Audio Context Management ===

    async resumeAudioContext() {
        if (this.audioContext && this.audioContext.state === 'suspended') {
            await this.audioContext.resume();
        }
    }

    getAudioContext() {
        return this.audioContext;
    }

    // === Parameter Controls ===

    setEngineParameter(paramPath, value) {
        if (!this.engine) return false;

        const pathParts = paramPath.split('.');
        let current = this.engine;

        // Navigate to the parameter
        for (let i = 0; i < pathParts.length - 1; i++) {
            if (current[pathParts[i]] !== undefined) {
                current = current[pathParts[i]];
            } else {
                return false;
            }
        }

        const finalProp = pathParts[pathParts.length - 1];
        if (current[finalProp] !== undefined) {
            current[finalProp] = value;
            return true;
        }

        return false;
    }

    getEngineParameter(paramPath) {
        if (!this.engine) return null;

        const pathParts = paramPath.split('.');
        let current = this.engine;

        for (const part of pathParts) {
            if (current[part] !== undefined) {
                current = current[part];
            } else {
                return null;
            }
        }

        return current;
    }

    // === Calibration and Analysis ===

    async calibrate(duration = 5000) {
        return new Promise((resolve) => {
            if (!this.engine) {
                resolve(false);
                return;
            }

            console.log(`Calibrating enhanced engine for ${duration}ms...`);
            const startTime = Date.now();

            // Reset medians for calibration
            if (this.engine.medians) {
                this.engine.medians = {
                    E_med: 0.2,
                    C_med: 900,
                    Z_med: 0.1,
                    F_med: 0.05
                };
            }

            const calibrationInterval = setInterval(() => {
                const elapsed = Date.now() - startTime;
                if (elapsed >= duration) {
                    clearInterval(calibrationInterval);
                    console.log('Enhanced engine calibration complete');
                    console.log('Final medians:', this.engine.medians);
                    resolve(true);
                }
            }, 100);
        });
    }

    getAnalysisReport() {
        if (!this.engine) return null;

        return {
            state: this.engine.state.vec.slice(),
            medians: { ...this.engine.medians },
            frameCount: this.engine.frameCount,
            tempo: this.engine.tempoPhase.tau,
            phase: this.engine.tempoPhase.phi,
            recentEvents: this.engine.events,
            controllerGains: { ...this.engine.controlMapper.gamma }
        };
    }
}

// Enhanced Audio Visualizer for the new engine
class EnhancedAudioVisualizer {
    constructor(canvas, processor) {
        this.canvas = canvas;
        this.ctx = canvas.getContext('2d');
        this.processor = processor;
        this.animationId = null;
        this.isRunning = false;

        // Enhanced visualization settings
        this.colors = {
            waveform: '#00ff88',
            spectrum: '#ff6b00',
            polarity: '#ff0080',
            intensity: '#ffff00',
            granularity: '#00a8ff',
            confidence: '#7cdcff',
            events: '#ff4d4f'
        };

        this.setupCanvas();
        this.bindEvents();
    }

    setupCanvas() {
        // Set canvas size and properties
        this.canvas.width = 800;
        this.canvas.height = 600;

        this.ctx.strokeStyle = this.colors.waveform;
        this.ctx.lineWidth = 2;
        this.ctx.lineCap = 'round';
        this.ctx.lineJoin = 'round';
    }

    bindEvents() {
        this.processor.on('audioFrame', (data) => {
            if (this.isRunning) {
                this.drawFrame(data);
            }
        });
    }

    start() {
        this.isRunning = true;
        this.animate();
    }

    stop() {
        this.isRunning = false;
        if (this.animationId) {
            cancelAnimationFrame(this.animationId);
            this.animationId = null;
        }
    }

    animate() {
        if (this.isRunning) {
            this.animationId = requestAnimationFrame(() => this.animate());
        }
    }

    drawFrame(data) {
        const { waveform, spectrum, semantics } = data;
        const width = this.canvas.width;
        const height = this.canvas.height;

        // Clear with fade effect
        this.ctx.fillStyle = 'rgba(0, 0, 0, 0.05)';
        this.ctx.fillRect(0, 0, width, height);

        // Draw sections
        this.drawWaveform(waveform, 0, 0, width, height * 0.3);
        this.drawSpectrum(spectrum, 0, height * 0.3, width * 0.7, height * 0.3);
        this.drawSemanticState(semantics, width * 0.7, height * 0.3, width * 0.3, height * 0.3);
        this.drawTemporalInfo(semantics, 0, height * 0.6, width * 0.5, height * 0.2);
        this.drawEvents(semantics, width * 0.5, height * 0.6, width * 0.5, height * 0.2);
        this.drawVisualParameters(semantics, 0, height * 0.8, width, height * 0.2);
    }

    drawWaveform(waveform, x, y, w, h) {
        this.ctx.strokeStyle = this.colors.waveform;
        this.ctx.lineWidth = 2;
        this.ctx.beginPath();

        const sliceWidth = w / waveform.length;
        let x1 = x;

        for (let i = 0; i < waveform.length; i++) {
            const v = waveform[i] * h / 2;
            const y1 = y + h / 2 + v;

            if (i === 0) {
                this.ctx.moveTo(x1, y1);
            } else {
                this.ctx.lineTo(x1, y1);
            }

            x1 += sliceWidth;
        }

        this.ctx.stroke();

        // Label
        this.ctx.fillStyle = '#888';
        this.ctx.font = '12px monospace';
        this.ctx.fillText('Waveform', x + 5, y + 15);
    }

    drawSpectrum(spectrum, x, y, w, h) {
        this.ctx.fillStyle = this.colors.spectrum;

        const barWidth = w / spectrum.length * 2;

        for (let i = 0; i < spectrum.length / 2; i++) {
            const barHeight = Math.max(0, (spectrum[i] + 100) * h / 100);
            this.ctx.fillRect(x + i * barWidth, y + h - barHeight, barWidth - 1, barHeight);
        }

        // Label
        this.ctx.fillStyle = '#888';
        this.ctx.font = '12px monospace';
        this.ctx.fillText('Spectrum', x + 5, y + 15);
    }

    drawSemanticState(semantics, x, y, w, h) {
        const { state } = semantics;
        const labels = ['Polarity', 'Intensity', 'Granularity', 'Confidence'];
        const colors = [this.colors.polarity, this.colors.intensity, this.colors.granularity, this.colors.confidence];
        const ranges = [[-1, 1], [0, 2.5], [0, 2.5], [0, 1]];

        this.ctx.fillStyle = 'rgba(0, 0, 0, 0.3)';
        this.ctx.fillRect(x, y, w, h);

        this.ctx.fillStyle = '#fff';
        this.ctx.font = '12px monospace';
        this.ctx.fillText('Semantic State', x + 5, y + 15);

        const barHeight = (h - 40) / 4;

        for (let i = 0; i < 4; i++) {
            const barY = y + 25 + i * barHeight;
            const [min, max] = ranges[i];
            const normalizedValue = (state[i] - min) / (max - min);
            const barWidth = Math.max(0, Math.min(w - 80, normalizedValue * (w - 80)));

            // Background bar
            this.ctx.fillStyle = 'rgba(255, 255, 255, 0.1)';
            this.ctx.fillRect(x + 5, barY, w - 80, barHeight - 5);

            // Value bar
            this.ctx.fillStyle = colors[i];
            this.ctx.fillRect(x + 5, barY, barWidth, barHeight - 5);

            // Label and value
            this.ctx.fillStyle = '#fff';
            this.ctx.font = '10px monospace';
            this.ctx.fillText(labels[i], x + w - 70, barY + 12);
            this.ctx.fillText(state[i].toFixed(3), x + w - 70, barY + 24);
        }
    }

    drawTemporalInfo(semantics, x, y, w, h) {
        const { clock } = semantics;

        this.ctx.fillStyle = 'rgba(0, 0, 0, 0.3)';
        this.ctx.fillRect(x, y, w, h);

        this.ctx.fillStyle = '#fff';
        this.ctx.font = '14px monospace';
        this.ctx.fillText('Temporal', x + 5, y + 18);

        this.ctx.font = '12px monospace';
        this.ctx.fillText(`BPM: ${clock.bpm.toFixed(1)}`, x + 5, y + 35);
        this.ctx.fillText(`Phase: ${(clock.phase * 180 / Math.PI).toFixed(1)}°`, x + 5, y + 50);

        // Phase circle
        const centerX = x + w - 30;
        const centerY = y + h / 2;
        const radius = 15;

        this.ctx.strokeStyle = this.colors.confidence;
        this.ctx.lineWidth = 2;
        this.ctx.beginPath();
        this.ctx.arc(centerX, centerY, radius, 0, 2 * Math.PI);
        this.ctx.stroke();

        // Phase indicator
        const phaseX = centerX + radius * Math.cos(clock.phase - Math.PI / 2);
        const phaseY = centerY + radius * Math.sin(clock.phase - Math.PI / 2);
        this.ctx.fillStyle = this.colors.confidence;
        this.ctx.beginPath();
        this.ctx.arc(phaseX, phaseY, 3, 0, 2 * Math.PI);
        this.ctx.fill();
    }

    drawEvents(semantics, x, y, w, h) {
        const { events, operators } = semantics;

        this.ctx.fillStyle = 'rgba(0, 0, 0, 0.3)';
        this.ctx.fillRect(x, y, w, h);

        this.ctx.fillStyle = '#fff';
        this.ctx.font = '14px monospace';
        this.ctx.fillText('Events & Ops', x + 5, y + 18);

        let yPos = y + 35;

        // Active events
        Object.entries(events).forEach(([event, active]) => {
            if (active && yPos < y + h - 10) {
                this.ctx.fillStyle = this.colors.events;
                this.ctx.font = '10px monospace';
                this.ctx.fillText(`• ${event}`, x + 5, yPos);
                yPos += 12;
            }
        });

        // Active operators
        if (operators && operators.length > 0) {
            this.ctx.fillStyle = this.colors.intensity;
            operators.forEach(op => {
                if (yPos < y + h - 10) {
                    this.ctx.fillText(`⚡ ${op}`, x + 5, yPos);
                    yPos += 12;
                }
            });
        }
    }

    drawVisualParameters(semantics, x, y, w, h) {
        const { hue, brightness, grain, reverb } = semantics;

        this.ctx.fillStyle = 'rgba(0, 0, 0, 0.3)';
        this.ctx.fillRect(x, y, w, h);

        this.ctx.fillStyle = '#fff';
        this.ctx.font = '14px monospace';
        this.ctx.fillText('Visual Parameters', x + 5, y + 18);

        // Color strip showing current hue
        const stripWidth = w - 200;
        for (let i = 0; i < stripWidth; i++) {
            const currentHue = (i / stripWidth) * 360;
            this.ctx.fillStyle = `hsl(${currentHue}, 70%, 50%)`;
            this.ctx.fillRect(x + 10 + i, y + 25, 1, 20);
        }

        // Current hue indicator
        const huePos = ((hue % 360) / 360) * stripWidth;
        this.ctx.fillStyle = '#fff';
        this.ctx.fillRect(x + 10 + huePos - 1, y + 20, 2, 30);

        // Parameter values
        this.ctx.fillStyle = '#fff';
        this.ctx.font = '12px monospace';
        this.ctx.fillText(`Hue: ${hue.toFixed(0)}°`, x + stripWidth + 20, y + 30);
        this.ctx.fillText(`Brightness: ${brightness.toFixed(1)}`, x + stripWidth + 20, y + 45);
        this.ctx.fillText(`Grain: ${grain.toFixed(2)}`, x + stripWidth + 20, y + 60);
        this.ctx.fillText(`Reverb: ${reverb.toFixed(2)}`, x + stripWidth + 20, y + 75);
    }
}

export { EnhancedWebAudioProcessor, EnhancedAudioVisualizer };
