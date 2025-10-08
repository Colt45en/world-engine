/**
 * Web Audio Integration for Audio Semantic Engine
 * Real-time microphone processing with the semantic engine
 */

import { AudioSemanticEngine } from './audio-semantic-engine.js';

class WebAudioSemanticProcessor {
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
            state: [0, 0.5, 0.4, 0.6],
            tempo: 120,
            phase: 0,
            events: {},
            features: {}
        };

        // Event callbacks
        this.callbacks = new Map();
        /**
         * @type {((arg0: any) => void) | undefined}
         */
        this.updateVisualization = undefined;

        // Note: Call init() explicitly after creating an instance.
    }

    async init() {
        try {
            // Initialize Web Audio Context
            // @ts-ignore
            const AudioCtx = window.AudioContext || (window /** @type {any} */)["webkitAudioContext"];
            this.audioContext = new AudioCtx({
                sampleRate: this.sampleRate
            });

            // Initialize Audio Semantic Engine
            this.engine = new AudioSemanticEngine(this.sampleRate, this.bufferSize);

            // Connect engine events to visualization
            this.engine.addEventListener('stateUpdate', (output) => {
                if (typeof this.updateVisualization === 'function') {
                    this.updateVisualization(output);
                }
            });

            console.log('Audio Semantic Processor initialized');

        } catch (error) {
            console.error('Failed to initialize audio processor:', error);
        }
    }

    async startProcessing() {
        if (this.isProcessing) return;

        try {
            // Request microphone access
            const stream = await navigator.mediaDevices.getUserMedia({
                audio: {
                    echoCancellation: false,
                    noiseSuppression: false,
                    autoGainControl: false,
                    sampleRate: this.sampleRate
                }
            });

            // Resume audio context if suspended
            if (this.audioContext && this.audioContext.state === 'suspended') {
                await this.audioContext.resume();
            }

            // Create audio nodes
            if (!this.audioContext) throw new Error('AudioContext not initialized');
            this.microphone = this.audioContext.createMediaStreamSource(stream);
            this.analyser = this.audioContext.createAnalyser();
            this.analyser.fftSize = this.fftSize;
            this.analyser.smoothingTimeConstant = 0;

            // Use AudioWorkletNode for real-time analysis (modern replacement for ScriptProcessorNode)
            // Register the worklet processor if not already registered
            // @ts-ignore
            if (!this.audioContext.audioWorklet.modules.includes('processor-worklet.js')) {
                await this.audioContext.audioWorklet.addModule('processor-worklet.js');
            }
            this.processor = new AudioWorkletNode(this.audioContext, 'processor-worklet', {
                numberOfInputs: 1,
                numberOfOutputs: 1,
                outputChannelCount: [1]
            });

            // Connect audio graph
            this.microphone.connect(this.analyser);
            this.analyser.connect(this.processor);
            this.processor.connect(this.audioContext.destination);

            // Listen for audio frames from the worklet
            this.processor.port.onmessage = (event) => {
                this.processAudioFrame(event.data);
            };

            this.isProcessing = true;
            this.emit('started');

            console.log('Audio processing started');

        } catch (error) {
            console.error('Failed to start audio processing:', error);
            this.emit('error', error);
        }
    }
    /**
     * @param {any} data
     */
    processAudioFrame(data) {
        // Update waveform and spectrum visualization data
        if (data.waveform && data.spectrum) {
            this.visualizationData.waveform.set(data.waveform);
            this.visualizationData.spectrum.set(data.spectrum);
        }

        // Process audio frame with the semantic engine if available
        if (this.engine && typeof this.engine.processFrame === 'function') {
            // @ts-ignore
            const semanticOutput = this.engine.processFrame(data.waveform);
            if (typeof this.updateVisualization === 'function') {
                this.updateVisualization(semanticOutput);
            }
        }

        // Emit audioFrame event for visualization
        this.emit('audioFrame', {
            waveform: this.visualizationData.waveform,
            spectrum: this.visualizationData.spectrum,
            semantics: this.visualizationData
        });
    }

    stopProcessing() {
        if (!this.isProcessing) return;

        try {
            // Disconnect audio nodes
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
                this.processor = null;
            }
        } finally {
            this.isProcessing = false;
        }
    }

    // === VISUALIZATION DATA ACCESS ===

    getSemanticState() {
        return this.engine ? this.engine.getSemanticState() : null;
    }

    getTemporalState() {
        return this.engine ? this.engine.getTemporalState() : null;
    }

    getVisualizationData() {
        return this.visualizationData;
    }

    // === WORLD ENGINE INTEGRATION ===

    sendToWorldEngine(data) {
        // Send semantic state to World Engine
        if (window.postMessage) {
            window.postMessage({
                type: 'AUDIO_SEMANTIC_UPDATE',
                payload: data
            }, '*');
        }
    }

    // === EVENT SYSTEM ===

    on(eventType, callback) {
        if (!this.callbacks.has(eventType)) {
            this.callbacks.set(eventType, []);
        }
        this.callbacks.get(eventType).push(callback);
    }

    // @ts-ignore
    // @ts-ignore
    // @ts-ignore
    off(eventType, callback) {
        // @ts-ignore
        if (this.callbacks.has(eventType)) {
            // @ts-ignore
            const callbacks = this.callbacks.get(eventType);
            // @ts-ignore
            const index = callbacks.indexOf(callback);
            if (index > -1) {
                callbacks.splice(index, 1);
            }
        }
    }

    // @ts-ignore
    // @ts-ignore
    // @ts-ignore
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

    // === AUDIO CONTEXT MANAGEMENT ===

    // @ts-ignore
    // @ts-ignore
    // @ts-ignore
    async resumeAudioContext() {
        // @ts-ignore
        if (this.audioContext && this.audioContext.state === 'suspended') {
            // @ts-ignore
            await this.audioContext.resume();
        }
    }

    // @ts-ignore
    // @ts-ignore
    // @ts-ignore
    getAudioContext() {
        return this.audioContext;
    }

    // === PARAMETER CONTROLS ===

    // @ts-ignore
    // @ts-ignore
    // @ts-ignore
    setEngineParameter(paramName, value) {
        // @ts-ignore
        if (this.engine && this.engine[paramName] !== undefined) {
            // @ts-ignore
            this.engine[paramName] = value;
        }
    }

    // @ts-ignore
    // @ts-ignore
    // @ts-ignore
    getEngineParameter(paramName) {
        // @ts-ignore
        return this.engine ? this.engine[paramName] : null;
    }

    // === CALIBRATION ===

    // @ts-ignore
    // @ts-ignore
    // @ts-ignore
    calibrate(duration = 5000) {
        return new Promise((resolve) => {
            if (!this.engine) {
                resolve(false);
                return;
            }

            // @ts-ignore
            console.log(`Calibrating for ${duration}ms...`);
            const startTime = Date.now();

            const calibrationInterval = setInterval(() => {
                const elapsed = Date.now() - startTime;
                // @ts-ignore
                if (elapsed >= duration) {
                    clearInterval(calibrationInterval);
                    console.log('Calibration complete');
                    resolve(true);
                }
            }, 100);
        });
    }
}

// === AUDIO VISUALIZATION HELPERS ===

// @ts-ignore
// @ts-ignore
// @ts-ignore
class AudioVisualizer {
    /**
     * @param {{ getContext: (arg0: string) => any; }} canvas
     * @param {any} processor
     */
    constructor(canvas, processor) {
        this.canvas = canvas;
        this.ctx = canvas.getContext('2d');
        this.processor = processor;
        this.animationId = null;
        this.isRunning = false;

        // Visualization settings
        this.colors = {
            waveform: '#00ff88',
            spectrum: '#ff6b00',
            state: '#ff0080',
            tempo: '#00a8ff',
            events: '#ffff00'
        };

        this.setupCanvas();
        this.bindEvents();
    }

    setupCanvas() {
        // Set canvas size
        // @ts-ignore
        this.canvas.width = 800;
        // @ts-ignore
        this.canvas.height = 400;

        // Set drawing context properties
        this.ctx.strokeStyle = this.colors.waveform;
        this.ctx.lineWidth = 2;
        this.ctx.lineCap = 'round';
    }

    bindEvents() {
        this.processor.on('audioFrame', (/** @type {any} */ data) => {
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

    /**
     * @param {{ waveform: any; spectrum: any; semantics: any; }} data
     */
    drawFrame(data) {
        const { waveform, spectrum, semantics } = data;
        // @ts-ignore
        const width = this.canvas.width;
        // @ts-ignore
        const height = this.canvas.height;

        // Clear canvas
        this.ctx.fillStyle = 'rgba(0, 0, 0, 0.1)';
        this.ctx.fillRect(0, 0, width, height);

        // Draw waveform (top half)
        this.drawWaveform(waveform, 0, 0, width, height / 2);

        // Draw spectrum (bottom half)
        this.drawSpectrum(spectrum, 0, height / 2, width, height / 2);

        // Draw semantic state overlay
        this.drawSemanticState(semantics, width - 200, 10, 180, 100);

        // Draw tempo and events
        this.drawTempoAndEvents(semantics, 10, 10, 200, 60);
    }

    /**
     * @param {string | any[]} waveform
     * @param {number} x
     * @param {number} y
     * @param {number} w
     * @param {number} h
     */
    drawWaveform(waveform, x, y, w, h) {
        this.ctx.strokeStyle = this.colors.waveform;
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
    }

    /**
     * @param {string | any[]} spectrum
     * @param {number} x
     * @param {number} y
     * @param {number} w
     * @param {number} h
     */
    drawSpectrum(spectrum, x, y, w, h) {
        this.ctx.fillStyle = this.colors.spectrum;

        const barWidth = w / spectrum.length;

        for (let i = 0; i < spectrum.length; i++) {
            const barHeight = (spectrum[i] + 140) * h / 140;
            this.ctx.fillRect(x + i * barWidth, y + h - barHeight, barWidth - 1, barHeight);
        }
    }

    /**
     * @param {{ state: any; }} semantics
     * @param {number} x
     * @param {number} y
     * @param {number} w
     * @param {number} h
     */
    drawSemanticState(semantics, x, y, w, h) {
        const { state } = semantics;

        this.ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
        this.ctx.fillRect(x, y, w, h);

        this.ctx.fillStyle = this.colors.state;
        this.ctx.font = '12px monospace';
        this.ctx.fillText('Semantic State:', x + 5, y + 15);
        this.ctx.fillText(`Polarity: ${state[0].toFixed(3)}`, x + 5, y + 30);
        this.ctx.fillText(`Intensity: ${state[1].toFixed(3)}`, x + 5, y + 45);
        this.ctx.fillText(`Granularity: ${state[2].toFixed(3)}`, x + 5, y + 60);
        this.ctx.fillText(`Confidence: ${state[3].toFixed(3)}`, x + 5, y + 75);
    }

    /**
     * @param {{ tempo: any; events: any; operators: any; }} semantics
     * @param {number} x
     * @param {number} y
     * @param {number} w
     * @param {number} h
     */
    drawTempoAndEvents(semantics, x, y, w, h) {
        const { tempo, events, operators } = semantics;

        this.ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
        this.ctx.fillRect(x, y, w, h);

        this.ctx.fillStyle = this.colors.tempo;
        this.ctx.font = '12px monospace';
        this.ctx.fillText(`Tempo: ${tempo.toFixed(1)} BPM`, x + 5, y + 15);

        if (operators && operators.length > 0) {
            this.ctx.fillStyle = this.colors.events;
            this.ctx.fillText(`Ops: ${operators.join(', ')}`, x + 5, y + 30);
        }

        // Event indicators
        let eventY = y + 45;
        Object.entries(events).forEach(([eventType, active]) => {
            if (active) {
                this.ctx.fillStyle = this.colors.events;
                this.ctx.fillText(`• ${eventType}`, x + 5, eventY);
                eventY += 12;
            }
        });
    }
}

export { WebAudioSemanticProcessor, AudioVisualizer };
