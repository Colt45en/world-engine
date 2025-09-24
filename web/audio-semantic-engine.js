/**
 * Real-Time Audio Semantic Engine
 * Continuous audio analysis with semantic state vector control
 */

class AudioSemanticEngine {
    constructor(sampleRate = 44100, frameSize = 1024) {
        this.sampleRate = sampleRate;
        this.frameSize = frameSize;
        this.H = 0.01; // Frame time in seconds

        // === INITIALIZATION ===
        this.S = new Float32Array([0, 0.5, 0.4, 0.6]); // [polarity, intensity, granularity, confidence]
        this.tau = 120; // Initial tempo (BPM)
        this.phi = 0; // Phase accumulator
        this.lastOnsetTime = -Infinity;

        // Baselines (Exponential Moving Averages)
        this.E_baseline = new EMA(0.1);
        this.C_baseline = new EMA(0.1);
        this.Z_baseline = new EMA(0.1);
        this.F_baseline = new EMA(0.1);

        // Feature storage
        this.prevMagnitude = null;
        this.featureHistory = [];
        this.onsetHistory = [];

        // Control parameters
        this.gamma = {
            C: 2.0,  // Spectral centroid sensitivity
            E: 1.5,  // Energy sensitivity
            Z: 1.0,  // Zero crossing sensitivity
            v: 0.8,  // Voiced sensitivity
            P: 0.5   // Pitch sensitivity
        };

        // State dynamics matrices
        this.A = [
            [-0.1,  0.2,  0.1,  0.0],
            [ 0.1, -0.2,  0.0,  0.1],
            [ 0.0,  0.1, -0.15, 0.2],
            [ 0.1,  0.0,  0.1, -0.1]
        ];

        this.B = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ];

        // Operator library
        this.OP_LIBRARY = {
            ALIGN: {
                M: [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],
                d: [0.1, 0, 0, 0.05]
            },
            PROJ: {
                M: [[0.9,0.1,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],
                d: [0, 0.2, 0, 0.1]
            },
            REFINE: {
                M: [[1,0,0,0],[0,0.8,0,0.1],[0,0,1.1,0],[0,0,0,1]],
                d: [0, 0, 0.1, 0.05]
            },
            IRONY: {
                M: [[-0.8,0.2,0,0],[0,1,0,0],[0,0,1,0],[0.1,0,0,0.9]],
                d: [-0.3, 0.1, 0.2, 0]
            },
            POLY: {
                M: [[1,0,0,0],[0,1,0.2,0],[0,0.1,1,0.1],[0,0,0,1]],
                d: [0, 0, 0.3, 0.1]
            },
            TEXTUREBURST: {
                M: [[1,0.1,0,0],[0.2,1,0,0],[0,0.3,1.2,0],[0,0,0.1,1]],
                d: [0.1, 0.3, 0.4, 0]
            },
            GRAPH: {
                M: [[1.1,0,0,0.1],[0,1,0,0],[0,0,0.9,0],[0,0,0.2,1.1]],
                d: [0.05, 0, -0.1, 0.2]
            },
            BATTLE: {
                M: [[0.9,0.2,0,0],[0.1,1.1,0,0],[0,0,1,0.1],[0,0,0,0.8]],
                d: [0.2, 0.4, 0.1, -0.1]
            },
            HOLY: {
                M: [[1.2,0,0,0.2],[0,0.7,0.1,0],[0,0,1.3,0],[0.1,0,0,1.2]],
                d: [0.3, -0.2, 0.2, 0.3]
            }
        };

        this.lambda = 0.3; // Operator gain factor
        this.lo = [-1, 0, 0, 0];   // State bounds
        this.hi = [1, 2, 2, 1];

        this.frameCount = 0;
        this.K_norm = 100; // Normalize every 100 frames

        // Events for World Engine integration
        this.eventCallbacks = new Map();
    }

    // === FEATURE EXTRACTION ===

    computeRMS(audioBuffer) {
        let sum = 0;
        for (let i = 0; i < audioBuffer.length; i++) {
            sum += audioBuffer[i] * audioBuffer[i];
        }
        return Math.sqrt(sum / audioBuffer.length);
    }

    computeSTFT(audioBuffer) {
        // Simplified STFT using Web Audio API concepts
        const N = audioBuffer.length;
        const magnitude = new Float32Array(N / 2);

        // Basic magnitude spectrum computation
        for (let k = 0; k < N / 2; k++) {
            let real = 0, imag = 0;
            for (let n = 0; n < N; n++) {
                const theta = -2 * Math.PI * k * n / N;
                real += audioBuffer[n] * Math.cos(theta);
                imag += audioBuffer[n] * Math.sin(theta);
            }
            magnitude[k] = Math.sqrt(real * real + imag * imag);
        }

        return { magnitude };
    }

    computeSpectralCentroid(magnitude) {
        let weightedSum = 0, magnitudeSum = 0;

        for (let k = 0; k < magnitude.length; k++) {
            const freq = k * this.sampleRate / (2 * magnitude.length);
            weightedSum += freq * magnitude[k];
            magnitudeSum += magnitude[k];
        }

        return magnitudeSum > 0 ? weightedSum / magnitudeSum : 0;
    }

    computeSpectralFlux(prevMag, currentMag) {
        if (!prevMag) return 0;

        let flux = 0;
        const minLen = Math.min(prevMag.length, currentMag.length);

        for (let k = 0; k < minLen; k++) {
            const diff = currentMag[k] - prevMag[k];
            flux += Math.max(0, diff); // Half-wave rectification
        }

        return flux / minLen;
    }

    estimatePitch(audioBuffer) {
        // Simplified autocorrelation pitch detection
        const minPeriod = Math.floor(this.sampleRate / 800); // ~800 Hz max
        const maxPeriod = Math.floor(this.sampleRate / 80);  // ~80 Hz min

        let maxCorr = 0, bestPeriod = 0;

        for (let period = minPeriod; period <= maxPeriod; period++) {
            let correlation = 0;
            for (let i = 0; i < audioBuffer.length - period; i++) {
                correlation += audioBuffer[i] * audioBuffer[i + period];
            }

            if (correlation > maxCorr) {
                maxCorr = correlation;
                bestPeriod = period;
            }
        }

        const pitch = bestPeriod > 0 ? this.sampleRate / bestPeriod : 0;
        const voiced = maxCorr > 0.3 * audioBuffer.length; // Voicing threshold

        return { pitch, voiced };
    }

    computeZeroCrossings(audioBuffer) {
        let crossings = 0;
        for (let i = 1; i < audioBuffer.length; i++) {
            if ((audioBuffer[i] >= 0) !== (audioBuffer[i-1] >= 0)) {
                crossings++;
            }
        }
        return crossings / audioBuffer.length;
    }

    // === MAIN PROCESSING LOOP ===

    processFrame(audioBuffer, timestamp) {
        const t_k = timestamp;

        // === FEATURE EXTRACTION ===
        const E = this.computeRMS(audioBuffer);
        const { magnitude } = this.computeSTFT(audioBuffer);
        const C = this.computeSpectralCentroid(magnitude);
        const F = this.computeSpectralFlux(this.prevMagnitude, magnitude);
        const { pitch: P, voiced } = this.estimatePitch(audioBuffer);
        const Z = this.computeZeroCrossings(audioBuffer);

        // Store for next frame
        this.prevMagnitude = magnitude.slice();

        // Update feature baselines
        const E_base = this.E_baseline.update(E);
        const C_base = this.C_baseline.update(C);
        const Z_base = this.Z_baseline.update(Z);
        const F_base = this.F_baseline.update(F);

        // === ONSET DETECTION ===
        const F_std = this.computeStandardDeviation(this.featureHistory.map(f => f.F));
        const onset = F > (F_base + 1.5 * F_std);

        // === TEMPO/PHASE TRACKING ===
        if (onset) {
            const dt = t_k - this.lastOnsetTime;
            if (dt > 0.18 && dt < 2.0) {
                const eta = 0.1; // Learning rate
                const newTempo = 60 / dt;
                this.tau = (1 - eta) * this.tau + eta * newTempo;
            }
            this.lastOnsetTime = t_k;
            this.onsetHistory.push(t_k);
        }

        // Update phase
        this.phi = (this.phi + 2 * Math.PI * this.tau * this.H / 60) % (2 * Math.PI);

        // === CONTROL VECTOR COMPUTATION ===
        const u_p = Math.tanh(this.gamma.C * (C - C_base));
        const u_i = Math.tanh(this.gamma.E * (E - E_base));
        const u_g = Math.tanh(this.gamma.Z * (Z - Z_base));
        const u_c = this.sigmoid(this.gamma.v * (voiced ? 1 : 0) + this.gamma.P * Math.log1p(P));

        const u = [u_p, u_i, u_g, u_c];

        // === CONTINUOUS STATE UPDATE ===
        const dt = this.H;
        const dS = this.matrixVectorMultiply(this.A, this.S);
        const Bu = this.matrixVectorMultiply(this.B, u);
        const noise = this.generateNoise(this.S);

        for (let i = 0; i < 4; i++) {
            this.S[i] += dt * (dS[i] + Bu[i] + noise[i]);
            this.S[i] = Math.max(this.lo[i], Math.min(this.hi[i], this.S[i]));
        }

        // === EVENT DETECTION & OPERATORS ===
        const ops = [];
        const events = {};

        if (this.firstStableBeat()) { ops.push('ALIGN'); events.stableBeat = true; }
        if (voiced && !onset) { ops.push('PROJ'); events.sustained = true; }
        if (this.sustainedLowFlux()) { ops.push('REFINE'); events.stable = true; }
        if (this.contradictionDetected()) { ops.push('IRONY'); events.contradiction = true; }
        if (this.highVarianceWindow()) { ops.push('POLY'); events.complex = true; }
        if (onset && this.roughTexture()) { ops.push('TEXTUREBURST'); events.rough = true; }
        if (this.sustainedVoiced()) { ops.push('GRAPH'); events.melodic = true; }
        if (this.frequentOnsets()) { ops.push('BATTLE'); events.dense = true; }
        if (this.longDrone()) { ops.push('HOLY'); events.drone = true; }

        // === APPLY OPERATORS ===
        for (const opName of ops) {
            const { M, d } = this.OP_LIBRARY[opName];
            const gain = this.lambda * Math.sqrt(this.tau / 120) * this.phaseWindow(this.phi);

            const MS = this.matrixVectorMultiply(M, this.S);
            const gaind = d.map(x => gain * x);

            for (let i = 0; i < 4; i++) {
                this.S[i] = MS[i] + gaind[i];
                this.S[i] = Math.max(this.lo[i], Math.min(this.hi[i], this.S[i]));
            }
        }

        // === PERIODIC NORMALIZATION ===
        if (this.frameCount % this.K_norm === 0) {
            for (let i = 0; i < 4; i++) {
                this.S[i] = Math.max(this.lo[i], Math.min(this.hi[i], this.S[i]));
            }
        }

        // Store features for history
        this.featureHistory.push({ E, C, F, P, Z, voiced, onset });
        if (this.featureHistory.length > 100) {
            this.featureHistory.shift(); // Keep last 100 frames
        }

        this.frameCount++;

        // === OUTPUT & EVENTS ===
        const output = {
            state: Array.from(this.S),
            tempo: this.tau,
            phase: this.phi,
            operators: ops,
            events: events,
            features: { E, C, F, P, Z, voiced, onset },
            timestamp: t_k
        };

        this.publishOutput(output);
        return output;
    }

    // === UTILITY FUNCTIONS ===

    sigmoid(x) {
        return 1 / (1 + Math.exp(-x));
    }

    matrixVectorMultiply(matrix, vector) {
        const result = new Array(matrix.length).fill(0);
        for (let i = 0; i < matrix.length; i++) {
            for (let j = 0; j < vector.length; j++) {
                result[i] += matrix[i][j] * vector[j];
            }
        }
        return result;
    }

    generateNoise(state) {
        // State-dependent noise
        return state.map((s, i) => 0.01 * (Math.random() - 0.5) * (1 + Math.abs(s)));
    }

    phaseWindow(phi) {
        // Cosine window based on phase
        return 0.5 * (1 + Math.cos(phi));
    }

    computeStandardDeviation(values) {
        if (values.length < 2) return 0;
        const mean = values.reduce((a, b) => a + b) / values.length;
        const variance = values.reduce((sum, val) => sum + (val - mean) ** 2, 0) / values.length;
        return Math.sqrt(variance);
    }

    // === EVENT DETECTION METHODS ===

    firstStableBeat() {
        return this.onsetHistory.length > 3 && this.frameCount > 200;
    }

    sustainedLowFlux() {
        const recent = this.featureHistory.slice(-10);
        return recent.length === 10 && recent.every(f => f.F < 0.1);
    }

    contradictionDetected() {
        const recent = this.featureHistory.slice(-5);
        if (recent.length < 5) return false;

        const energyTrend = recent[4].E - recent[0].E;
        const centroidTrend = recent[4].C - recent[0].C;

        return (energyTrend > 0 && centroidTrend < -100) || (energyTrend < 0 && centroidTrend > 100);
    }

    highVarianceWindow() {
        const recent = this.featureHistory.slice(-20).map(f => f.E);
        return this.computeStandardDeviation(recent) > 0.2;
    }

    roughTexture() {
        const recent = this.featureHistory.slice(-5);
        return recent.length === 5 && recent.some(f => f.Z > 0.3);
    }

    sustainedVoiced() {
        const recent = this.featureHistory.slice(-10);
        return recent.length === 10 && recent.filter(f => f.voiced).length >= 8;
    }

    frequentOnsets() {
        const recentOnsets = this.onsetHistory.filter(t => this.frameCount * this.H - t < 2.0);
        return recentOnsets.length > 8;
    }

    longDrone() {
        const recent = this.featureHistory.slice(-50);
        if (recent.length < 50) return false;

        const pitches = recent.filter(f => f.voiced).map(f => f.P);
        if (pitches.length < 40) return false;

        const pitchStd = this.computeStandardDeviation(pitches);
        return pitchStd < 10; // Very stable pitch
    }

    // === OUTPUT & INTEGRATION ===

    publishOutput(output) {
        // Emit to World Engine
        if (this.eventCallbacks.has('stateUpdate')) {
            this.eventCallbacks.get('stateUpdate')(output);
        }

        // Emit specific events
        for (const [eventType, active] of Object.entries(output.events)) {
            if (active && this.eventCallbacks.has(eventType)) {
                this.eventCallbacks.get(eventType)(output);
            }
        }
    }

    addEventListener(eventType, callback) {
        this.eventCallbacks.set(eventType, callback);
    }

    removeEventListener(eventType) {
        this.eventCallbacks.delete(eventType);
    }

    // === WORLD ENGINE INTEGRATION ===

    getSemanticState() {
        return {
            polarity: this.S[0],
            intensity: this.S[1],
            granularity: this.S[2],
            confidence: this.S[3]
        };
    }

    getTemporalState() {
        return {
            tempo: this.tau,
            phase: this.phi,
            onsetHistory: this.onsetHistory.slice(-10)
        };
    }
}

// === EXPONENTIAL MOVING AVERAGE HELPER ===

class EMA {
    constructor(alpha = 0.1) {
        this.alpha = alpha;
        this.value = null;
    }

    update(newValue) {
        if (this.value === null) {
            this.value = newValue;
        } else {
            this.value = this.alpha * newValue + (1 - this.alpha) * this.value;
        }
        return this.value;
    }
}

export { AudioSemanticEngine, EMA };
