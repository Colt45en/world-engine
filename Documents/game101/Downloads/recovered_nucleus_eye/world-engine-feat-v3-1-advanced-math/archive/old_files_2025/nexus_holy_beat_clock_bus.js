/**
 * NEXUS HOLY BEAT CLOCK/BUS SYSTEM
 * ================================
 *
 * Evolution of the Holy Beat engine into a shared pipeline with:
 * • Clock/Bus exposure with pub-sub mechanism
 * • Quantized parameter queues (p(t+) = p* only when φ(t) = 0)
 * • Preset banks (spectral/structural parameter vectors)
 * • Cross-modal feature maps (audio ↔ art ↔ world)
 * • Unified thought-process scaffold for all engines
 */

// Global Clock with beat/bar/phase emission
class HolyBeatClock {
    constructor(bpm = 120, beatsPerBar = 4) {
        this.bpm = bpm;
        this.beatsPerBar = beatsPerBar;
        this.running = false;
        this.startTime = 0;
        this.currentTime = 0;

        // State tracking
        this.bar = 0;
        this.beat = 0;
        this.phase = 0;  // φ(t) ∈ [0,1)

        // Beat timing
        this.secPerBeat = 60 / bpm;  // T_b = 60/BPM

        // Subscriber management
        this.beatListeners = [];
        this.barListeners = [];
        this.phaseListeners = [];

        console.log('⏰ HolyBeatClock initialized');
    }

    setBPM(newBPM) {
        this.bpm = newBPM;
        this.secPerBeat = 60 / newBPM;
        console.log(`⏰ BPM updated to ${newBPM}`);
    }

    start(currentTime) {
        this.running = true;
        this.startTime = currentTime;
        this.bar = 0;
        this.beat = 0;
        this.phase = 0;
        console.log('⏰ Clock started');
    }

    stop() {
        this.running = false;
        console.log('⏰ Clock stopped');
    }

    // Core tick function - implements φ(t) = fract(t / T_b)
    tick(currentTime) {
        if (!this.running) return this.getState();

        const elapsedTime = currentTime - this.startTime;
        const totalBeats = elapsedTime / this.secPerBeat;

        const newBar = Math.floor(totalBeats / this.beatsPerBar);
        const newBeat = Math.floor(totalBeats) % this.beatsPerBar;
        const newPhase = totalBeats - Math.floor(totalBeats); // φ(t) = fract(beats(t))

        // Detect transitions for event emission
        const beatChanged = (newBeat !== this.beat) || (newBar !== this.bar);
        const barChanged = (newBar !== this.bar);

        this.currentTime = currentTime;
        this.bar = newBar;
        this.beat = newBeat;
        this.phase = newPhase;

        // Emit events on beat boundaries (φ(t) = 0)
        if (beatChanged) {
            this.emitBeat();
        }
        if (barChanged) {
            this.emitBar();
        }

        // Always emit phase for continuous subscribers
        this.emitPhase();

        return this.getState();
    }

    getState() {
        return {
            bar: this.bar,
            beat: this.beat,
            phase: this.phase,
            bpm: this.bpm,
            running: this.running,
            secPerBeat: this.secPerBeat
        };
    }

    // Pub-sub interface
    onBeat(callback) {
        this.beatListeners.push(callback);
    }

    onBar(callback) {
        this.barListeners.push(callback);
    }

    onPhase(callback) {
        this.phaseListeners.push(callback);
    }

    emitBeat() {
        this.beatListeners.forEach(callback => {
            try {
                callback(this.getState());
            } catch (error) {
                console.error('Beat listener error:', error);
            }
        });
    }

    emitBar() {
        this.barListeners.forEach(callback => {
            try {
                callback(this.getState());
            } catch (error) {
                console.error('Bar listener error:', error);
            }
        });
    }

    emitPhase() {
        this.phaseListeners.forEach(callback => {
            try {
                callback(this.getState());
            } catch (error) {
                console.error('Phase listener error:', error);
            }
        });
    }
}

// Quantized Parameter Queue - changes only on beat boundaries
class QuantizedParamQueue {
    constructor(clockState) {
        this.queue = [];  // Array of {param, value, targetBeat, targetBar}
        this.clockState = clockState;
    }

    // Schedule parameter change for specific beat
    schedule(param, value, targetBeat = null, targetBar = null) {
        const currentState = this.clockState.getState();

        // Default to next beat if no target specified
        if (targetBeat === null) {
            targetBeat = (currentState.beat + 1) % currentState.beatsPerBar || currentState.beatsPerBar;
            targetBar = targetBeat === 0 ? currentState.bar + 1 : currentState.bar;
        } else if (targetBar === null) {
            targetBar = currentState.bar;
        }

        this.queue.push({
            param,
            value,
            targetBeat,
            targetBar,
            scheduled: Date.now()
        });

        console.log(`📅 Scheduled ${param} = ${value} for beat ${targetBeat}, bar ${targetBar}`);
    }

    // Apply queued changes when φ(t) = 0 (beat boundary)
    apply(currentBeat, currentBar, targetEngine) {
        const readyItems = this.queue.filter(item =>
            item.targetBeat === currentBeat && item.targetBar === currentBar
        );

        readyItems.forEach(item => {
            this.commitParameterUpdate(item.param, item.value, targetEngine);
        });

        // Remove applied items from queue
        this.queue = this.queue.filter(item =>
            !(item.targetBeat === currentBeat && item.targetBar === currentBar)
        );

        return readyItems.length > 0;
    }

    commitParameterUpdate(param, value, targetEngine) {
        const path = param.split('.');
        let target = targetEngine;

        // Navigate to parameter location
        for (let i = 0; i < path.length - 1; i++) {
            if (target[path[i]]) {
                target = target[path[i]];
            } else {
                console.error(`Parameter path ${param} not found in engine`);
                return;
            }
        }

        const finalKey = path[path.length - 1];
        const oldValue = target[finalKey];
        target[finalKey] = value;

        console.log(`✅ Applied ${param}: ${oldValue} → ${value}`);
    }

    // Get pending updates count
    getPendingCount() {
        return this.queue.length;
    }

    // Clear all pending updates
    clear() {
        const count = this.queue.length;
        this.queue = [];
        console.log(`🗑️ Cleared ${count} pending parameter updates`);
    }
}

// Preset Bank - stored parameter vectors
class PresetBank {
    constructor() {
        this.presets = new Map();
        this.currentPreset = null;
        this.initializeDefaultPresets();
    }

    initializeDefaultPresets() {
        // Chord Stack preset: [f₀, {ratioᵢ}, D_AM, D_FM, f_c]
        this.store('chord_stack', {
            'synth.baseFreq': 220,
            'synth.harmonics': 8,
            'lfo.amDepth': 0.3,
            'lfo.fmDepth': 2.0,
            'lfo.amDivision': 4,
            'lfo.fmDivision': 8,
            'synth.filterCutoff': 3000,
            'synth.masterGain': 0.4
        });

        // FM Bell preset: [f₀, D_FM >> 0, d_FM = 4]
        this.store('fm_bell', {
            'synth.baseFreq': 440,
            'synth.harmonics': 6,
            'lfo.amDepth': 0.1,
            'lfo.fmDepth': 12.0,  // D_FM >> 0
            'lfo.amDivision': 2,
            'lfo.fmDivision': 4,   // d_FM = 4
            'synth.filterCutoff': 4000,
            'synth.masterGain': 0.3
        });

        // Drone preset: sustained tones with slow modulation
        this.store('drone', {
            'synth.baseFreq': 110,
            'synth.harmonics': 12,
            'lfo.amDepth': 0.05,
            'lfo.fmDepth': 0.5,
            'lfo.amDivision': 16,
            'lfo.fmDivision': 32,
            'synth.filterCutoff': 2000,
            'synth.masterGain': 0.6
        });

        // Percussive preset: sharp attack, fast decay
        this.store('percussive', {
            'synth.baseFreq': 80,
            'synth.harmonics': 4,
            'lfo.amDepth': 0.8,
            'lfo.fmDepth': 8.0,
            'lfo.amDivision': 1,   // AM every beat
            'lfo.fmDivision': 2,
            'synth.filterCutoff': 6000,
            'synth.masterGain': 0.7
        });

        console.log('🎛️ Initialized preset bank with default presets');
    }

    store(name, parameterVector) {
        this.presets.set(name, { ...parameterVector, created: Date.now() });
        console.log(`💾 Stored preset '${name}' with ${Object.keys(parameterVector).length} parameters`);
    }

    recall(name, paramQueue, targetBeat = null, targetBar = null) {
        const preset = this.presets.get(name);
        if (!preset) {
            console.error(`❌ Preset '${name}' not found`);
            return false;
        }

        // Schedule all parameters from preset
        Object.entries(preset).forEach(([param, value]) => {
            if (param !== 'created') {  // Skip metadata
                paramQueue.schedule(param, value, targetBeat, targetBar);
            }
        });

        this.currentPreset = name;
        console.log(`🎼 Recalled preset '${name}' - ${Object.keys(preset).length - 1} parameters scheduled`);
        return true;
    }

    list() {
        return Array.from(this.presets.keys());
    }

    getCurrentPreset() {
        return this.currentPreset;
    }

    export(name) {
        return this.presets.get(name);
    }
}

// Cross-Modal Feature Bus - maps between audio ↔ art ↔ world
class CrossModalBus {
    constructor() {
        this.features = {
            // Audio features
            spectralCentroid: 220,    // χ_s (Hz)
            rmsEnergy: 0.1,          // RMS level
            beatIntensity: 0.5,      // RMS over last bar

            // Art features
            strokeDensity: 0.5,      // ρ_a
            colorPalette: 0,         // Current palette index
            petalCount: 5,           // Rose curve petals

            // World features
            terrainRoughness: 0.3,   // ρ_w
            cameraPath: 0,           // Camera position index
            meshComplexity: 64       // Mesh resolution
        };

        this.subscribers = new Map();
        this.mappings = new Map();
        this.initializeDefaultMappings();
    }

    initializeDefaultMappings() {
        // Spectral centroid → rose-curve petals: k = α * χ_s
        this.addMapping('spectralCentroid', 'petalCount', (centroid) => {
            const α = 0.01;  // Scaling factor
            return Math.max(3, Math.min(15, Math.floor(3 + α * centroid)));
        });

        // Beat intensity (RMS over last bar) → terrain warp amplitude
        this.addMapping('beatIntensity', 'terrainRoughness', (intensity) => {
            const γ = 1.2;  // Gain factor
            return Math.min(1.0, γ * intensity);
        });

        // Bar count mod M → color palette
        this.addMapping('bar', 'colorPalette', (bar) => {
            const M = 8;  // Palette cycle length
            return bar % M;
        });

        // RMS energy → stroke density
        this.addMapping('rmsEnergy', 'strokeDensity', (rms) => {
            return Math.min(1.0, rms * 2.0);
        });

        console.log('🔗 Initialized cross-modal mappings');
    }

    addMapping(sourceFeature, targetFeature, transformFunction) {
        if (!this.mappings.has(sourceFeature)) {
            this.mappings.set(sourceFeature, []);
        }

        this.mappings.get(sourceFeature).push({
            target: targetFeature,
            transform: transformFunction
        });

        console.log(`🔗 Added mapping: ${sourceFeature} → ${targetFeature}`);
    }

    updateFeature(featureName, value) {
        this.features[featureName] = value;

        // Apply cross-modal mappings
        if (this.mappings.has(featureName)) {
            this.mappings.get(featureName).forEach(mapping => {
                const transformedValue = mapping.transform(value);
                this.features[mapping.target] = transformedValue;

                // Notify subscribers of mapped feature
                this.notifySubscribers(mapping.target, transformedValue);
            });
        }

        // Notify direct subscribers
        this.notifySubscribers(featureName, value);
    }

    subscribe(featureName, callback) {
        if (!this.subscribers.has(featureName)) {
            this.subscribers.set(featureName, []);
        }

        this.subscribers.get(featureName).push(callback);
    }

    notifySubscribers(featureName, value) {
        if (this.subscribers.has(featureName)) {
            this.subscribers.get(featureName).forEach(callback => {
                try {
                    callback(featureName, value, this.features);
                } catch (error) {
                    console.error(`Cross-modal subscriber error for ${featureName}:`, error);
                }
            });
        }
    }

    getFeature(featureName) {
        return this.features[featureName];
    }

    getAllFeatures() {
        return { ...this.features };
    }

    // Implement the mathematical transforms from your spec
    // Θ_art = f_audio(χ_s, RMS, φ)
    getArtParameters(clockState) {
        const χ_s = this.features.spectralCentroid;
        const RMS = this.features.rmsEnergy;
        const φ = clockState.phase;

        return {
            petalCount: this.features.petalCount,
            colorIntensity: RMS * (1 + 0.3 * Math.sin(2 * Math.PI * φ)),
            rotationSpeed: (χ_s / 440) * φ,
            strokeWidth: 1 + this.features.strokeDensity * 3
        };
    }

    // Θ_world = g_audio(χ_s, RMS, φ)
    getWorldParameters(clockState) {
        const χ_s = this.features.spectralCentroid;
        const RMS = this.features.rmsEnergy;
        const φ = clockState.phase;

        return {
            terrainAmplitude: this.features.terrainRoughness,
            waveFrequency: (χ_s / 220) * 0.5,
            cameraHeight: 2 + RMS * 3,
            meshDeformation: φ * this.features.terrainRoughness
        };
    }
}

// Unified Thought-Process Scaffold
class UnifiedThoughtProcessor {
    constructor(clock, paramQueue, presetBank, crossModalBus) {
        this.clock = clock;
        this.paramQueue = paramQueue;
        this.presetBank = presetBank;
        this.crossModalBus = crossModalBus;

        this.engines = new Map();
        this.lastBeat = -1;
        this.lastBar = -1;

        this.setupClockSubscriptions();
        console.log('🧠 Unified Thought Processor initialized');
    }

    setupClockSubscriptions() {
        // Subscribe to beat events for parameter queue processing
        this.clock.onBeat((clockState) => {
            this.processBeatBoundary(clockState);
        });

        // Subscribe to bar events for preset triggers
        this.clock.onBar((clockState) => {
            this.processBarBoundary(clockState);
        });

        // Subscribe to phase updates for continuous processing
        this.clock.onPhase((clockState) => {
            this.processPhaseUpdate(clockState);
        });
    }

    registerEngine(name, engine) {
        this.engines.set(name, engine);
        console.log(`🔧 Registered engine: ${name}`);
    }

    // Step 1. Clock Pulse - emit beat/bar/phase to bus
    processBeatBoundary(clockState) {
        // Step 2. Param Queue Check - apply scheduled changes
        this.engines.forEach((engine, name) => {
            const applied = this.paramQueue.apply(clockState.beat, clockState.bar, engine);
            if (applied) {
                console.log(`⚙️ Applied queued parameters to ${name} engine`);
            }
        });

        this.lastBeat = clockState.beat;

        // Update cross-modal features with beat information
        this.crossModalBus.updateFeature('beat', clockState.beat);
        this.crossModalBus.updateFeature('beatIntensity', this.calculateBeatIntensity());
    }

    processBarBoundary(clockState) {
        // Step 3. Preset Recall or Drift - handle bar-based preset triggers
        this.handlePresetTriggers(clockState);

        this.lastBar = clockState.bar;

        // Update cross-modal features
        this.crossModalBus.updateFeature('bar', clockState.bar);
    }

    processPhaseUpdate(clockState) {
        // Step 4. Cross-Modal Sync - exchange features between engines
        this.updateCrossModalFeatures(clockState);

        // Step 5. Render - engines output synchronized state
        // (Handled by individual engines)

        // Step 6. Caution/Check - monitor for overload conditions
        this.performSafetyChecks(clockState);
    }

    handlePresetTriggers(clockState) {
        // Example: switch presets every 8 bars
        const presetCycle = ['chord_stack', 'fm_bell', 'drone', 'percussive'];
        const presetIndex = Math.floor(clockState.bar / 8) % presetCycle.length;
        const targetPreset = presetCycle[presetIndex];

        if (this.presetBank.getCurrentPreset() !== targetPreset) {
            console.log(`🎼 Bar ${clockState.bar}: Switching to preset '${targetPreset}'`);
            this.presetBank.recall(targetPreset, this.paramQueue, clockState.beat + 1);
        }
    }

    updateCrossModalFeatures(clockState) {
        // Extract features from engines and update cross-modal bus
        this.engines.forEach((engine, name) => {
            if (engine.extractFeatures) {
                const features = engine.extractFeatures(clockState);
                Object.entries(features).forEach(([featureName, value]) => {
                    this.crossModalBus.updateFeature(featureName, value);
                });
            }
        });
    }

    performSafetyChecks(clockState) {
        // Monitor for overload conditions
        const rms = this.crossModalBus.getFeature('rmsEnergy');
        const spectralCentroid = this.crossModalBus.getFeature('spectralCentroid');

        // If spectrum too bright or RMS too high, back off depths next beat
        if (rms > 0.8 || spectralCentroid > 2000) {
            console.warn('⚠️ Safety check: reducing modulation depths');

            this.engines.forEach((engine) => {
                if (engine.lfo) {
                    this.paramQueue.schedule('lfo.amDepth', engine.lfo.amDepth * 0.8, clockState.beat + 1);
                    this.paramQueue.schedule('lfo.fmDepth', engine.lfo.fmDepth * 0.8, clockState.beat + 1);
                }
            });
        }
    }

    calculateBeatIntensity() {
        // Calculate RMS over last bar (simplified)
        const rms = this.crossModalBus.getFeature('rmsEnergy');
        return Math.min(1.0, rms * 1.5);  // Boost for beat intensity
    }

    // Public API for scheduling and control
    scheduleParameterChange(engineName, param, value, targetBeat = null, targetBar = null) {
        const engine = this.engines.get(engineName);
        if (!engine) {
            console.error(`❌ Engine '${engineName}' not found`);
            return false;
        }

        this.paramQueue.schedule(`${param}`, value, targetBeat, targetBar);
        return true;
    }

    recallPreset(presetName, targetBeat = null, targetBar = null) {
        return this.presetBank.recall(presetName, this.paramQueue, targetBeat, targetBar);
    }

    getSystemState() {
        return {
            clock: this.clock.getState(),
            pendingUpdates: this.paramQueue.getPendingCount(),
            currentPreset: this.presetBank.getCurrentPreset(),
            features: this.crossModalBus.getAllFeatures(),
            registeredEngines: Array.from(this.engines.keys())
        };
    }
}

// Export all classes for use in the main application
window.HolyBeatClockBusSystem = {
    HolyBeatClock,
    QuantizedParamQueue,
    PresetBank,
    CrossModalBus,
    UnifiedThoughtProcessor
};
