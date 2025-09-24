/**
 * Enhanced Audio Semantic Engine - JavaScript implementation
 * Based on the comprehensive Python version with proper mathematical modeling
 */

// Utility functions
function clip(v, lo, hi) {
    if (Array.isArray(v)) {
        return v.map((val, i) => Math.max(Math.min(val, hi[i] || hi), lo[i] || lo));
    }
    return Math.max(Math.min(v, hi), lo);
}

function sigmoid(x) {
    return 1 / (1 + Math.exp(-x));
}

function tanh(x) {
    return Math.tanh(x);
}

// Matrix operations
function matrixVectorMultiply(matrix, vector) {
    return matrix.map(row =>
        row.reduce((sum, val, i) => sum + val * vector[i], 0)
    );
}

function vectorAdd(a, b) {
    return a.map((val, i) => val + b[i]);
}

function vectorScale(vector, scalar) {
    return vector.map(val => val * scalar);
}

// State model class
class SemanticState {
    constructor(p = 0.0, i = 0.5, g = 0.3, c = 0.6) {
        this.vec = [p, i, g, c]; // [polarity, intensity, granularity, confidence]
        this.lo = [-1.0, 0.0, 0.0, 0.0];
        this.hi = [1.0, 2.5, 2.5, 1.0];
    }

    update(dt, A, B, u, nonlinearFunc) {
        // S_{k+1} = clip(S_k + dt*(A @ S_k + B @ u + N(S_k)), lo, hi)
        const AS = matrixVectorMultiply(A, this.vec);
        const Bu = matrixVectorMultiply(B, u);
        const N = nonlinearFunc(this.vec);

        const delta = vectorScale(vectorAdd(vectorAdd(AS, Bu), N), dt);
        this.vec = clip(vectorAdd(this.vec, delta), this.lo, this.hi);
    }

    applyOperator(M = null, d = null) {
        if (M !== null) {
            this.vec = matrixVectorMultiply(M, this.vec);
        }
        if (d !== null) {
            this.vec = vectorAdd(this.vec, d);
        }
        this.vec = clip(this.vec, this.lo, this.hi);
    }
}

// Tempo/phase tracker
class TempoPhaseTracker {
    constructor(eta = 0.1) {
        this.tau = 120.0; // BPM
        this.phi = 0.0;   // phase [0, 2π]
        this.eta = eta;
        this.onsetTimes = [];
    }

    update(onsetTime, H) {
        if (onsetTime !== null) {
            this.onsetTimes.push(onsetTime);

            // Keep only recent onsets
            const maxHistory = 10;
            if (this.onsetTimes.length > maxHistory) {
                this.onsetTimes.shift();
            }

            // Update tempo if we have at least 2 onsets
            if (this.onsetTimes.length >= 2) {
                const delta = this.onsetTimes[this.onsetTimes.length - 1] -
                             this.onsetTimes[this.onsetTimes.length - 2];
                if (delta > 0.18 && delta < 2.0) { // Valid tempo range
                    const bpmInst = 60.0 / delta;
                    this.tau = (1 - this.eta) * this.tau + this.eta * bpmInst;
                }
            }
        }

        // Always update phase
        this.phi = (this.phi + 2 * Math.PI * this.tau * H / 60) % (2 * Math.PI);
    }
}

// Enhanced feature extraction
class FeatureExtractor {
    constructor(sampleRate) {
        this.fs = sampleRate;
        this.prevFrame = null;
        this.prevPitch = 0;
    }

    extractFeatures(audioFrame) {
        const features = {};

        // RMS Energy
        features.E = Math.sqrt(
            audioFrame.reduce((sum, sample) => sum + sample * sample, 0) / audioFrame.length
        );

        // FFT and spectral features
        const fftResult = this.computeFFT(audioFrame);
        const magnitude = fftResult.magnitude;
        const freqs = fftResult.frequencies;

        // Spectral Centroid
        const weightedSum = magnitude.reduce((sum, mag, i) => sum + freqs[i] * mag, 0);
        const magnitudeSum = magnitude.reduce((sum, mag) => sum + mag, 0);
        features.C = magnitudeSum > 0 ? weightedSum / magnitudeSum : 0;

        // Spectral Flux
        if (this.prevFrame) {
            const prevMag = this.computeFFT(this.prevFrame).magnitude;
            features.F = this.computeSpectralFlux(prevMag, magnitude);
        } else {
            features.F = 0;
        }

        // Pitch detection
        features.P = this.detectPitch(audioFrame);

        // Zero crossing rate (roughness measure)
        features.Z = this.computeZeroCrossingRate(audioFrame);

        this.prevFrame = audioFrame.slice();
        return features;
    }

    computeFFT(audioFrame) {
        // Simple DFT implementation for real signals
        const N = audioFrame.length;
        const magnitude = [];
        const frequencies = [];

        for (let k = 0; k < N / 2; k++) {
            let real = 0, imag = 0;
            for (let n = 0; n < N; n++) {
                const theta = -2 * Math.PI * k * n / N;
                real += audioFrame[n] * Math.cos(theta);
                imag += audioFrame[n] * Math.sin(theta);
            }
            magnitude.push(Math.sqrt(real * real + imag * imag));
            frequencies.push(k * this.fs / N);
        }

        return { magnitude, frequencies };
    }

    computeSpectralFlux(prevMag, currentMag) {
        let flux = 0;
        const minLen = Math.min(prevMag.length, currentMag.length);

        for (let i = 0; i < minLen; i++) {
            const diff = currentMag[i] - prevMag[i];
            flux += Math.max(0, diff); // Half-wave rectification
        }

        return flux / minLen;
    }

    detectPitch(audioFrame) {
        // Autocorrelation-based pitch detection
        const minPeriod = Math.floor(this.fs / 800); // ~800 Hz max
        const maxPeriod = Math.floor(this.fs / 80);  // ~80 Hz min

        let maxCorr = 0;
        let bestPeriod = 0;

        for (let period = minPeriod; period <= maxPeriod; period++) {
            let correlation = 0;
            for (let i = 0; i < audioFrame.length - period; i++) {
                correlation += audioFrame[i] * audioFrame[i + period];
            }

            if (correlation > maxCorr) {
                maxCorr = correlation;
                bestPeriod = period;
            }
        }

        return bestPeriod > 0 ? this.fs / bestPeriod : 0;
    }

    computeZeroCrossingRate(audioFrame) {
        let crossings = 0;
        for (let i = 1; i < audioFrame.length; i++) {
            if ((audioFrame[i] >= 0) !== (audioFrame[i-1] >= 0)) {
                crossings++;
            }
        }
        return crossings / audioFrame.length;
    }
}

// Control mapping functions
class ControlMapper {
    constructor() {
        this.gamma = {
            E: 5.0,   // Energy sensitivity
            C: 3.0,   // Centroid sensitivity
            Z: 4.0,   // Zero crossing sensitivity
            v: 1.4,   // Voiced sensitivity
            P: 0.7    // Pitch sensitivity
        };
    }

    mapControls(features, medians) {
        const { E, C, F, P, Z } = features;
        const { E_med, C_med, Z_med } = medians;

        const u_i = tanh(this.gamma.E * (E - E_med));
        const u_p = tanh(this.gamma.C * (C - C_med));
        const u_g = tanh(this.gamma.Z * (Z - Z_med));

        const voicedFlag = P > 0 ? 1 : 0;
        const u_c = sigmoid(this.gamma.v * voicedFlag + this.gamma.P * Math.log(1 + P));

        return [u_p, u_i, u_g, u_c];
    }
}

// Nonlinearity function N(S)
function nonlinear(S, betaI = 0.05, betaG = 0.06, betaC = 0.04) {
    const [p, i, g, c] = S;
    return [
        0.0,
        betaI * i * (1 - i / 2.5),
        betaG * g * (1 - g / 2.5),
        betaC * (1 - c) * c
    ];
}

// Operator definitions
const Operators = {
    ALIGN: () => [null, [0, 0.06, 0.04, 0.08]],
    PROJ: (u) => [null, [0, 0.1 * u[1], -0.04, 0.15 * u[3]]],
    REFINE: () => [[[1,0,0,0],[0,0.92,0,0],[0,0,0.88,0],[0,0,0,1.05]], null],
    POLY: () => [null, [0, 0, 0.18, -0.12]],
    IRONY: () => [[[-1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,0.94]], null],
    GRAPH: () => [null, [0.04, 0.06, 0.06, 0.04]],
    TEXTUREBURST: () => [null, [-0.06, -0.04, 0.36, -0.06]],
    BATTLE: () => [null, [0.12, 0.16, -0.08, 0.02]],
    HOLY: () => [null, [-0.08, 0.08, 0.14, 0.08]],
    RECOMPUTE: () => [null, null]
};

// Event detection
class EventDetector {
    constructor() {
        this.history = {
            F: [],
            P: []
        };
    }

    detectEvents(features, medians, prevPitch) {
        const { E, C, F, P, Z } = features;
        const { E_med, C_med, Z_med, F_med } = medians;

        // Update history
        this.history.F.push(F);
        this.history.P.push(P);
        if (this.history.F.length > 20) this.history.F.shift();
        if (this.history.P.length > 5) this.history.P.shift();

        const F_std = this.computeStd(this.history.F);

        return {
            onset: F > F_med + 1.5 * F_std,
            voiced: P > 0,
            burst: Z > Z_med + 0.6,
            ascend: P - prevPitch > 0,
            descend: P - prevPitch < 0,
            bright: C > C_med,
            dark: C < C_med
        };
    }

    computeStd(values) {
        if (values.length < 2) return 0;
        const mean = values.reduce((a, b) => a + b) / values.length;
        const variance = values.reduce((sum, val) => sum + (val - mean) ** 2, 0) / values.length;
        return Math.sqrt(variance);
    }
}

// Main Enhanced Audio Semantic Engine
class EnhancedAudioSemanticEngine {
    constructor(sampleRate = 44100, frameSize = 1024) {
        this.sampleRate = sampleRate;
        this.frameSize = frameSize;
        this.H = frameSize / sampleRate; // Frame time in seconds

        // Initialize components
        this.state = new SemanticState();
        this.tempoPhase = new TempoPhaseTracker();
        this.featureExtractor = new FeatureExtractor(sampleRate);
        this.controlMapper = new ControlMapper();
        this.eventDetector = new EventDetector();

        // Running medians for adaptive thresholding
        this.medians = {
            E_med: 0.2,
            C_med: 900,
            Z_med: 0.1,
            F_med: 0.05
        };

        // State dynamics matrices
        this.A = [
            [-0.3,  0.0,  0.0,  0.0],
            [ 0.0, -0.6,  0.0,  0.0],
            [ 0.0,  0.0, -0.4,  0.0],
            [ 0.0,  0.0,  0.0, -0.15]
        ];

        this.B = [
            [0.8, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.7, 0.0],
            [0.0, 0.0, 0.0, 0.6]
        ];

        this.frameCount = 0;
        this.prevPitch = 0;
        this.events = {};
        this.callbacks = new Map();
    }

    processFrame(audioFrame, timestamp) {
        // Extract features
        const features = this.featureExtractor.extractFeatures(audioFrame);

        // Update medians (simple exponential moving average)
        const alpha = 0.01;
        this.medians.E_med = (1 - alpha) * this.medians.E_med + alpha * features.E;
        this.medians.C_med = (1 - alpha) * this.medians.C_med + alpha * features.C;
        this.medians.Z_med = (1 - alpha) * this.medians.Z_med + alpha * features.Z;
        this.medians.F_med = (1 - alpha) * this.medians.F_med + alpha * features.F;

        // Detect events
        this.events = this.eventDetector.detectEvents(features, this.medians, this.prevPitch);

        // Update tempo/phase
        const onsetTime = this.events.onset ? timestamp : null;
        this.tempoPhase.update(onsetTime, this.H);

        // Map features to control vector
        const u = this.controlMapper.mapControls(features, this.medians);

        // Update semantic state
        this.state.update(this.H, this.A, this.B, u, nonlinear);

        // Apply operators based on events
        const appliedOps = [];

        if (this.events.onset) {
            const [M, d] = Operators.BATTLE();
            this.state.applyOperator(M, d);
            appliedOps.push('BATTLE');
        }

        if (this.events.burst) {
            const [M, d] = Operators.TEXTUREBURST();
            this.state.applyOperator(M, d);
            appliedOps.push('TEXTUREBURST');
        }

        if (this.events.voiced) {
            const [M, d] = Operators.PROJ(u);
            this.state.applyOperator(M, d);
            appliedOps.push('PROJ');
        }

        if (this.events.bright) {
            const [M, d] = Operators.GRAPH();
            this.state.applyOperator(M, d);
            appliedOps.push('GRAPH');
        }

        if (this.events.dark) {
            const [M, d] = Operators.POLY();
            this.state.applyOperator(M, d);
            appliedOps.push('POLY');
        }

        if (this.events.ascend) {
            const [M, d] = Operators.ALIGN();
            this.state.applyOperator(M, d);
            appliedOps.push('ALIGN');
        }

        if (this.events.descend) {
            const [M, d] = Operators.IRONY();
            this.state.applyOperator(M, d);
            appliedOps.push('IRONY');
        }

        // Final clip
        this.state.applyOperator(...Operators.RECOMPUTE());

        // Update previous pitch
        this.prevPitch = features.P;
        this.frameCount++;

        // Generate output
        const output = {
            state: this.state.vec.slice(),
            clock: {
                bpm: this.tempoPhase.tau,
                phase: this.tempoPhase.phi
            },
            events: { ...this.events },
            operators: appliedOps,
            features: { ...features },
            timestamp: timestamp,
            // Visual mapping
            hue: 200 + 60 * this.state.vec[0],
            brightness: 30 + 25 * this.state.vec[1],
            grain: this.state.vec[2],
            reverb: this.state.vec[3]
        };

        // Publish output
        this.publishOutput(output);
        return output;
    }

    publishOutput(output) {
        if (this.callbacks.has('stateUpdate')) {
            this.callbacks.get('stateUpdate')(output);
        }

        // Emit specific events
        for (const [eventType, active] of Object.entries(output.events)) {
            if (active && this.callbacks.has(eventType)) {
                this.callbacks.get(eventType)(output);
            }
        }
    }

    addEventListener(eventType, callback) {
        this.callbacks.set(eventType, callback);
    }

    removeEventListener(eventType) {
        this.callbacks.delete(eventType);
    }

    getSemanticState() {
        return {
            polarity: this.state.vec[0],
            intensity: this.state.vec[1],
            granularity: this.state.vec[2],
            confidence: this.state.vec[3]
        };
    }

    getTemporalState() {
        return {
            tempo: this.tempoPhase.tau,
            phase: this.tempoPhase.phi,
            onsetHistory: this.tempoPhase.onsetTimes.slice(-10)
        };
    }
}

export { EnhancedAudioSemanticEngine, SemanticState, TempoPhaseTracker, FeatureExtractor, ControlMapper, EventDetector, Operators };
