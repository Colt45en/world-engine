/**
 * NEXUS HOLY BEAT MATHEMATICAL ENGINE
 * ===================================
 *
 * Implementation of the Holy Beat equation set with thought-process algorithm
 * integrated into the NEXUS FORGE mathematical synthesis framework.
 *
 * Core equations:
 * • Global clock: φ(t) = fract(t / (60/b))
 * • Phase-locked LFOs: L_AM(t) = sin(2π φ(t)/d_AM), L_FM(t) = sin(2π φ(t)/d_FM)
 * • Harmonic stack: s(t) = AM(t) Σ(1/n)sin(2π∫f_n(τ)dτ)
 * • Quantized updates: parameters change only on beat boundaries
 */

class NexusHolyBeatMathEngine {
    constructor() {
        // Core mathematical state
        this.clock = {
            bpm: 120,                    // b (beats per minute)
            beatsPerBar: 4,              // β (beats per bar)
            startTime: 0,
            running: false,
            currentTime: 0,
            beatPhase: 0,                // φ(t) ∈ [0,1)
            beat: 0,
            bar: 0
        };

        // AM/FM LFO parameters
        this.lfo = {
            amDivision: 4,               // d_AM (beats per AM cycle)
            fmDivision: 8,               // d_FM (beats per FM cycle)
            amDepth: 0.2,                // D_AM ∈ [0,1]
            fmDepth: 6.0,                // D_FM ≥ 0
            amPhase: 0,                  // L_AM(t)
            fmPhase: 0                   // L_FM(t)
        };

        // Harmonic synthesis parameters
        this.synth = {
            baseFreq: 220,               // f_0 (fundamental frequency)
            harmonics: 6,                // N (number of partials)
            partialGains: [],            // A_n = 1/n
            partialPhases: [],           // φ_n (random initial phases)
            kappa: 0.1,                  // κ_n (FM scaling per partial)
            filterCutoff: 4000,          // f_c (lowpass cutoff)
            noiseLevel: 0.05,            // η(t) scaling
            masterGain: 0.5              // Overall output level
        };

        // Parameter update queue (quantized to beat boundaries)
        this.updateQueue = [];
        this.pendingUpdates = new Map();

        // Cross-modal feature extraction
        this.features = {
            spectralCentroid: 440,       // χ_s
            rmsEnergy: 0.1,              // Energy level
            strokeDensity: 0.5,          // ρ_a (for art coupling)
            terrainRoughness: 0.3        // ρ_w (for world coupling)
        };

        // Safety thresholds (health check)
        this.safety = {
            maxRMS: 0.8,
            maxFreqDelta: 100,
            emergencyFilterCutoff: 1000,
            stabilityMode: false
        };

        // Integration with NEXUS systems
        this.nexusIntegration = {
            vibeEngine: null,
            artEngine: null,
            worldEngine: null,
            mathEngine: null
        };

        this.initializeHarmonicStack();
        console.log('🎵 NEXUS Holy Beat Mathematical Engine initialized');
    }

    initializeHarmonicStack() {
        // Initialize harmonic series: A_n = 1/n
        this.synth.partialGains = [];
        this.synth.partialPhases = [];

        for (let n = 1; n <= this.synth.harmonics; n++) {
            this.synth.partialGains.push(1.0 / n);
            this.synth.partialPhases.push(Math.random() * 2 * Math.PI);
        }
    }

    // Core equation implementations

    /**
       * Global clock equations
       * T_b = 60/b, beats(t) = t/T_b, φ(t) = fract(beats(t))
       */
    updateClockState(currentTime) {
        if (!this.clock.running) {
            this.clock.beatPhase = 0;
            this.clock.beat = 0;
            this.clock.bar = 0;
            return { changedBeat: false, changedBar: false };
        }

        const elapsedTime = currentTime - this.clock.startTime;
        const beatsPerSecond = this.clock.bpm / 60.0;           // 1/T_b
        const totalBeats = elapsedTime * beatsPerSecond;         // beats(t)

        const newBeat = Math.floor(totalBeats) % this.clock.beatsPerBar;
        const newBar = Math.floor(totalBeats / this.clock.beatsPerBar);
        const newBeatPhase = totalBeats - Math.floor(totalBeats); // φ(t) = fract(beats(t))

        const changedBeat = (newBeat !== this.clock.beat);
        const changedBar = (newBar !== this.clock.bar);

        this.clock.currentTime = currentTime;
        this.clock.beatPhase = newBeatPhase;
        this.clock.beat = newBeat;
        this.clock.bar = newBar;

        return { changedBeat, changedBar };
    }

    /**
       * Phase-locked LFO equations
       * L_AM(t) = sin(2π φ(t)/d_AM), L_FM(t) = sin(2π φ(t)/d_FM)
       */
    updateLFOState() {
        const phi = this.clock.beatPhase;

        // AM LFO: amplitude modulation locked to beat phase
        this.lfo.amPhase = Math.sin(2 * Math.PI * phi / this.lfo.amDivision);

        // FM LFO: frequency modulation locked to beat phase
        this.lfo.fmPhase = Math.sin(2 * Math.PI * phi / this.lfo.fmDivision);
    }

    /**
       * Harmonic stack synthesis equation
       * s(t) = AM(t) Σ A_n sin(2π∫f_n(τ)dτ + φ_n)
       * where f_n(t) = n*f_0 + D_FM * L_FM(t) * κ_n
       * and AM(t) = 1 + D_AM * L_AM(t)
       */
    synthesizeHarmonicStack(sampleCount, sampleRate) {
        const samples = new Float32Array(sampleCount);
        const dt = 1.0 / sampleRate;

        // AM envelope: AM(t) = 1 + D_AM * L_AM(t)
        const amEnvelope = 1.0 + this.lfo.amDepth * this.lfo.amPhase;

        for (let i = 0; i < sampleCount; i++) {
            const t = i * dt;
            let sampleValue = 0;

            // Sum harmonic partials: Σ A_n sin(2π∫f_n(τ)dτ + φ_n)
            for (let n = 1; n <= this.synth.harmonics; n++) {
                const nIdx = n - 1;

                // Partial frequency with FM modulation: f_n(t) = n*f_0 + D_FM * L_FM(t) * κ_n
                const baseFreq = n * this.synth.baseFreq;
                const fmModulation = this.lfo.fmDepth * this.lfo.fmPhase * this.synth.kappa * n;
                const instantFreq = baseFreq + fmModulation;

                // Phase accumulation: 2π∫f_n(τ)dτ
                const phaseIncrement = 2 * Math.PI * instantFreq * dt;
                this.synth.partialPhases[nIdx] += phaseIncrement;

                // Partial amplitude: A_n = 1/n
                const amplitude = this.synth.partialGains[nIdx];

                // Generate partial: A_n sin(phase + φ_n)
                const partialSample = amplitude * Math.sin(this.synth.partialPhases[nIdx]);
                sampleValue += partialSample;
            }

            // Apply AM envelope and master gain
            samples[i] = sampleValue * amEnvelope * this.synth.masterGain;

            // Add noise: η(t) * e(t)
            if (this.synth.noiseLevel > 0) {
                const noiseSample = (Math.random() * 2 - 1) * this.synth.noiseLevel;
                samples[i] += noiseSample;
            }
        }

        // Apply lowpass filter: y(t) = F_LP(s(t); f_c)
        this.applyLowpassFilter(samples, sampleRate);

        return samples;
    }

    /**
       * Simple one-pole lowpass filter: F_LP(s(t); f_c)
       */
    applyLowpassFilter(samples, sampleRate) {
        if (!this.filterState) {
            this.filterState = { previous: 0 };
        }

        const cutoff = this.safety.stabilityMode ?
            this.safety.emergencyFilterCutoff :
            this.synth.filterCutoff;

        const rc = 1.0 / (2 * Math.PI * cutoff);
        const dt = 1.0 / sampleRate;
        const alpha = dt / (rc + dt);

        for (let i = 0; i < samples.length; i++) {
            this.filterState.previous += alpha * (samples[i] - this.filterState.previous);
            samples[i] = this.filterState.previous;
        }
    }

    // Thought-process algorithm (cognitive loop)

    /**
       * Main cognitive loop - implements the thought-process algorithm
       */
    tick(currentTime) {
        // 1. Sense time: compute φ(t), beat, bar
        const clockChanges = this.updateClockState(currentTime);

        // 2. Prepare modulations: evaluate L_AM(t), L_FM(t)
        this.updateLFOState();

        // 3. Form intent (safe changes): quantized parameter updates
        this.processQuantizedUpdates(clockChanges.changedBeat);

        // 4. Compose frequency plan & 5. Shape amplitude
        // (Handled in synthesizeHarmonicStack)

        // 6. Health check (caution logic)
        this.performHealthCheck();

        // 7. Reflect features (cross-modal coupling)
        this.updateCrossModalFeatures();

        // Return current state for visualization/integration
        return {
            clockState: {
                beatPhase: this.clock.beatPhase,
                beat: this.clock.beat,
                bar: this.clock.bar,
                bpm: this.clock.bpm
            },
            lfoState: {
                amPhase: this.lfo.amPhase,
                fmPhase: this.lfo.fmPhase
            },
            synthState: {
                baseFreq: this.synth.baseFreq,
                harmonics: this.synth.harmonics,
                amDepth: this.lfo.amDepth,
                fmDepth: this.lfo.fmDepth
            },
            features: { ...this.features },
            safety: { ...this.safety }
        };
    }

    /**
       * Quantized parameter updates - changes only occur on beat boundaries
       */
    processQuantizedUpdates(beatChanged) {
        if (beatChanged && this.pendingUpdates.size > 0) {
            // Beat boundary: commit all pending updates
            for (const [param, value] of this.pendingUpdates) {
                this.commitParameterUpdate(param, value);
            }
            this.pendingUpdates.clear();
            console.log('🎵 Beat boundary: parameters updated');
        }
    }

    commitParameterUpdate(param, value) {
        const path = param.split('.');
        let target = this;

        // Navigate to the parameter location
        for (let i = 0; i < path.length - 1; i++) {
            target = target[path[i]];
        }

        const finalKey = path[path.length - 1];
        target[finalKey] = value;

        // Special handling for harmonic count changes
        if (param === 'synth.harmonics') {
            this.initializeHarmonicStack();
        }

        console.log(`🔧 Updated ${param} = ${value}`);
    }

    /**
       * Health check and stability monitoring
       */
    performHealthCheck() {
        // Check RMS levels
        const currentRMS = this.estimateRMS();
        if (currentRMS > this.safety.maxRMS) {
            console.warn('⚠️ RMS threshold exceeded - engaging safety mode');
            this.safety.stabilityMode = true;
            this.queueParameterUpdate('lfo.amDepth', this.lfo.amDepth * 0.8);
            this.queueParameterUpdate('lfo.fmDepth', this.lfo.fmDepth * 0.8);
        } else if (this.safety.stabilityMode && currentRMS < this.safety.maxRMS * 0.7) {
            console.log('✅ RMS stabilized - disabling safety mode');
            this.safety.stabilityMode = false;
        }

        // Check frequency deviation
        const maxFreqDelta = this.lfo.fmDepth * this.synth.kappa * this.synth.harmonics;
        if (maxFreqDelta > this.safety.maxFreqDelta) {
            console.warn('⚠️ Frequency deviation too large - clamping FM depth');
            this.queueParameterUpdate('lfo.fmDepth', this.safety.maxFreqDelta / (this.synth.kappa * this.synth.harmonics));
        }
    }

    estimateRMS() {
        // Simplified RMS estimation based on current AM envelope
        const amEnvelope = 1.0 + this.lfo.amDepth * Math.abs(this.lfo.amPhase);
        const harmonicSum = this.synth.partialGains.reduce((sum, gain) => sum + gain, 0);
        return amEnvelope * harmonicSum * this.synth.masterGain;
    }

    /**
       * Cross-modal feature extraction and coupling
       */
    updateCrossModalFeatures() {
        // Update internal features based on current synthesis state
        this.features.spectralCentroid = this.calculateSpectralCentroid();
        this.features.rmsEnergy = this.estimateRMS();

        // Cross-modal coupling with NEXUS systems
        if (this.nexusIntegration.vibeEngine) {
            this.updateVibeIntegration();
        }

        if (this.nexusIntegration.artEngine) {
            this.updateArtIntegration();
        }

        if (this.nexusIntegration.worldEngine) {
            this.updateWorldIntegration();
        }
    }

    calculateSpectralCentroid() {
        // Weighted average of harmonic frequencies
        let weightedSum = 0;
        let totalWeight = 0;

        for (let n = 1; n <= this.synth.harmonics; n++) {
            const freq = n * this.synth.baseFreq;
            const weight = this.synth.partialGains[n - 1];
            weightedSum += freq * weight;
            totalWeight += weight;
        }

        return totalWeight > 0 ? weightedSum / totalWeight : this.synth.baseFreq;
    }

    // Integration methods with NEXUS systems

    updateVibeIntegration() {
        if (!this.nexusIntegration.vibeEngine) return;

        const vibeState = this.nexusIntegration.vibeEngine.getVibeState();

        // Map vibe state to synthesis parameters
        const vibeInfluence = {
            amDepth: 0.1 + Math.abs(vibeState.p) * 0.3,    // Polarity affects AM depth
            fmDepth: 2 + vibeState.i * 8,                   // Intensity affects FM depth
            baseFreq: 220 + vibeState.g * 220,              // Genre affects base frequency
            filterCutoff: 2000 + vibeState.c * 4000         // Confidence affects brightness
        };

        // Queue parameter updates for next beat
        this.queueParameterUpdate('lfo.amDepth', vibeInfluence.amDepth);
        this.queueParameterUpdate('lfo.fmDepth', vibeInfluence.fmDepth);
        this.queueParameterUpdate('synth.baseFreq', vibeInfluence.baseFreq);
        this.queueParameterUpdate('synth.filterCutoff', vibeInfluence.filterCutoff);
    }

    updateArtIntegration() {
        if (!this.nexusIntegration.artEngine) return;

        // Map spectral centroid to art parameters (rose curve petals)
        const normalizedCentroid = (this.features.spectralCentroid - 220) / 880; // Normalize to 0-1
        const petalCount = Math.floor(3 + normalizedCentroid * 12); // 3-15 petals

        this.features.strokeDensity = normalizedCentroid;

        // Send to art engine
        if (this.nexusIntegration.artEngine.setRosePetals) {
            this.nexusIntegration.artEngine.setRosePetals(petalCount);
        }
    }

    updateWorldIntegration() {
        if (!this.nexusIntegration.worldEngine) return;

        // Map RMS energy to terrain roughness
        const normalizedRMS = Math.min(1, this.features.rmsEnergy / 0.5);
        this.features.terrainRoughness = normalizedRMS;

        // Send to world engine
        if (this.nexusIntegration.worldEngine.setTerrainRoughness) {
            this.nexusIntegration.worldEngine.setTerrainRoughness(normalizedRMS);
        }
    }

    // Public API for parameter control

    /**
       * Queue a parameter update (will be applied on next beat boundary)
       */
    queueParameterUpdate(param, value) {
        this.pendingUpdates.set(param, value);
    }

    /**
       * Immediate parameter update (bypasses quantization - use carefully!)
       */
    setParameterImmediate(param, value) {
        this.commitParameterUpdate(param, value);
    }

    /**
       * Start the Holy Beat engine
       */
    start(currentTime) {
        this.clock.startTime = currentTime;
        this.clock.running = true;
        console.log('🎵 Holy Beat Mathematical Engine started');
    }

    /**
       * Stop the Holy Beat engine
       */
    stop() {
        this.clock.running = false;
        this.pendingUpdates.clear();
        this.safety.stabilityMode = false;
        console.log('🔇 Holy Beat Mathematical Engine stopped');
    }

    /**
       * Get current mathematical state for external systems
       */
    getMathematicalState() {
        return {
            // Clock equations
            beatPhase: this.clock.beatPhase,               // φ(t)
            beat: this.clock.beat,
            bar: this.clock.bar,
            bpm: this.clock.bpm,                          // b

            // LFO equations
            amLFO: this.lfo.amPhase,                      // L_AM(t)
            fmLFO: this.lfo.fmPhase,                      // L_FM(t)

            // Synthesis parameters
            baseFrequency: this.synth.baseFreq,           // f_0
            harmonicCount: this.synth.harmonics,          // N
            amDepth: this.lfo.amDepth,                    // D_AM
            fmDepth: this.lfo.fmDepth,                    // D_FM

            // Features for cross-modal coupling
            spectralCentroid: this.features.spectralCentroid,  // χ_s
            rmsEnergy: this.features.rmsEnergy,
            strokeDensity: this.features.strokeDensity,        // ρ_a
            terrainRoughness: this.features.terrainRoughness,  // ρ_w

            // Safety state
            stabilityMode: this.safety.stabilityMode
        };
    }

    /**
       * Connect to NEXUS systems
       */
    connectNexusSystems(systems) {
        this.nexusIntegration = { ...systems };
        console.log('🔗 Connected to NEXUS systems:', Object.keys(systems));
    }

    /**
       * Get the ultra-compact canonical equation as a evaluable function
       * y(t) = F_LP([1 + D_AM*L_AM(t)] Σ(1/n)sin(2π∫[n*f_0 + D_FM*L_FM(τ)*κ_n]dτ + φ_n) + η(t)*e(t); f_c)
       */
    getCanonicalEquation() {
        return (t) => {
            // Update LFO phases for time t
            const phi_t = (t * this.clock.bpm / 60.0) % 1.0;
            const L_AM = Math.sin(2 * Math.PI * phi_t / this.lfo.amDivision);
            const L_FM = Math.sin(2 * Math.PI * phi_t / this.lfo.fmDivision);

            // AM envelope
            const AM_envelope = 1 + this.lfo.amDepth * L_AM;

            // Harmonic sum
            let harmonicSum = 0;
            for (let n = 1; n <= this.synth.harmonics; n++) {
                const freq = n * this.synth.baseFreq + this.lfo.fmDepth * L_FM * this.synth.kappa * n;
                const phase = 2 * Math.PI * freq * t + this.synth.partialPhases[n - 1];
                harmonicSum += (1 / n) * Math.sin(phase);
            }

            // Add noise
            const noise = (Math.random() * 2 - 1) * this.synth.noiseLevel;

            // Apply AM and filter (simplified)
            return (AM_envelope * harmonicSum + noise) * this.synth.masterGain;
        };
    }
}

// Export for integration
window.NexusHolyBeatMathEngine = NexusHolyBeatMathEngine;
