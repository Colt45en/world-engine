/**
 * NEXUS HOLY BEAT CLOCK/BUS CORE SYSTEM
 * ====================================
 *
 * Clean, organized version of the Clock/Bus system for production use.
 * Implements φ(t) = fract(t / T_b) timing with quantized parameter updates.
 */

// Global Clock with mathematical precision
export class HolyBeatClock {
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

        // Beat timing: T_b = 60/BPM
        this.secPerBeat = 60 / bpm;

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
    onBeat(callback) { this.beatListeners.push(callback); }
    onBar(callback) { this.barListeners.push(callback); }
    onPhase(callback) { this.phaseListeners.push(callback); }

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
export class QuantizedParamQueue {
    constructor(clockState) {
        this.queue = [];  // Array of {param, value, targetBeat, targetBar}
        this.clockState = clockState;
    }

    schedule(param, value, targetBeat = null, targetBar = null) {
        const currentState = this.clockState.getState();

        // Default to next beat if no target specified
        if (targetBeat === null) {
            targetBeat = (currentState.beat + 1) % this.clockState.beatsPerBar;
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

    getPendingCount() { return this.queue.length; }

    clear() {
        const count = this.queue.length;
        this.queue = [];
        console.log(`🗑️ Cleared ${count} pending parameter updates`);
    }
}

// Cross-Modal Feature Bus - maps between audio ↔ art ↔ world
export class CrossModalBus {
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
    }

    updateFeature(featureName, value) {
        this.features[featureName] = value;

        // Apply cross-modal mappings
        if (this.mappings.has(featureName)) {
            this.mappings.get(featureName).forEach(mapping => {
                const transformedValue = mapping.transform(value);
                this.features[mapping.target] = transformedValue;
                this.notifySubscribers(mapping.target, transformedValue);
            });
        }

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

    getFeature(featureName) { return this.features[featureName]; }
    getAllFeatures() { return { ...this.features }; }
}

console.log('⚡ NEXUS Holy Beat Core System loaded');
