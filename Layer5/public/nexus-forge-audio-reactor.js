/**
 * NEXUS Forge Audio Reactor
 * • Real-time audio analysis and visualization
 * • Heart rate synchronization with audio
 * • Particle system audio reactivity
 * • Environmental audio effects
 * • Advanced frequency analysis
 */

class NexusForgeAudioReactor {
    constructor() {
        this.audioContext = null;
        this.analyser = null;
        this.microphone = null;
        this.dataArray = null;
        this.frequencyData = null;
        this.isActive = false;

        this.frequencyBands = {
            bass: { min: 0, max: 100, value: 0 },      // 20-100 Hz
            lowMid: { min: 100, max: 500, value: 0 },  // 100-500 Hz
            mid: { min: 500, max: 2000, value: 0 },    // 500-2000 Hz
            highMid: { min: 2000, max: 5000, value: 0 }, // 2-5 kHz
            treble: { min: 5000, max: 20000, value: 0 }  // 5-20 kHz
        };

        this.beatDetection = {
            history: [],
            threshold: 1.3,
            minTimeBetweenBeats: 200,
            lastBeatTime: 0,
            currentBeat: false,
            bpm: 0,
            beatBuffer: []
        };

        this.heartSync = {
            enabled: false,
            targetBPM: 70,
            currentPhase: 0,
            resonanceStrength: 0.5,
            emotionalState: 'neutral'
        };

        this.visualizationCallbacks = [];
        this.particleSystemCallbacks = [];

        console.log('🎵 Audio Reactor initialized');
    }

    async initialize() {
        try {
            // Initialize Web Audio API
            this.audioContext = new (window.AudioContext || window.webkitAudioContext)();

            // Create analyser
            this.analyser = this.audioContext.createAnalyser();
            this.analyser.fftSize = 2048;
            this.analyser.smoothingTimeConstant = 0.8;

            const bufferLength = this.analyser.frequencyBinCount;
            this.dataArray = new Uint8Array(bufferLength);
            this.frequencyData = new Float32Array(bufferLength);

            console.log('🎵 Audio Context initialized');
            return true;
        } catch (error) {
            console.error('❌ Failed to initialize audio:', error);
            return false;
        }
    }

    async requestMicrophoneAccess() {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({
                audio: {
                    echoCancellation: false,
                    noiseSuppression: false,
                    autoGainControl: false,
                    sampleRate: 44100
                }
            });

            this.microphone = this.audioContext.createMediaStreamSource(stream);
            this.microphone.connect(this.analyser);

            this.isActive = true;
            this.startAnalysis();

            console.log('🎤 Microphone access granted');
            return true;
        } catch (error) {
            console.error('❌ Microphone access denied:', error);
            return false;
        }
    }

    connectAudioElement(audioElement) {
        if (!this.audioContext || !this.analyser) {
            console.error('❌ Audio context not initialized');
            return false;
        }

        try {
            const source = this.audioContext.createMediaElementSource(audioElement);
            source.connect(this.analyser);
            source.connect(this.audioContext.destination);

            this.isActive = true;
            this.startAnalysis();

            console.log('🎵 Audio element connected');
            return true;
        } catch (error) {
            console.error('❌ Failed to connect audio element:', error);
            return false;
        }
    }

    startAnalysis() {
        if (!this.isActive) return;

        this.analyzeAudio();
        requestAnimationFrame(() => this.startAnalysis());
    }

    analyzeAudio() {
        if (!this.analyser || !this.isActive) return;

        // Get frequency data
        this.analyser.getByteFrequencyData(this.dataArray);
        this.analyser.getFloatFrequencyData(this.frequencyData);

        // Analyze frequency bands
        this.analyzeFrequencyBands();

        // Detect beats
        this.detectBeats();

        // Update heart synchronization
        this.updateHeartSync();

        // Trigger callbacks
        this.triggerCallbacks();
    }

    analyzeFrequencyBands() {
        const nyquist = this.audioContext.sampleRate / 2;
        const binWidth = nyquist / this.dataArray.length;

        // Reset band values
        Object.keys(this.frequencyBands).forEach(band => {
            this.frequencyBands[band].value = 0;
        });

        // Calculate band averages
        for (let i = 0; i < this.dataArray.length; i++) {
            const frequency = i * binWidth;
            const amplitude = this.dataArray[i] / 255.0;

            // Determine which band this frequency belongs to
            for (const [, band] of Object.entries(this.frequencyBands)) {
                if (frequency >= band.min && frequency <= band.max) {
                    band.value = Math.max(band.value, amplitude);
                }
            }
        }

        // Apply logarithmic scaling for better visual response
        Object.keys(this.frequencyBands).forEach(bandName => {
            const band = this.frequencyBands[bandName];
            band.value = Math.pow(band.value, 0.7); // Slight curve for better response
        });
    }

    detectBeats() {
        const bass = this.frequencyBands.bass.value;
        const lowMid = this.frequencyBands.lowMid.value;

        // Combine bass and low-mid for beat detection
        const beatStrength = (bass * 0.7) + (lowMid * 0.3);

        // Add to history
        this.beatDetection.history.push(beatStrength);
        if (this.beatDetection.history.length > 10) {
            this.beatDetection.history.shift();
        }

        // Calculate average
        const average = this.beatDetection.history.reduce((a, b) => a + b, 0) / this.beatDetection.history.length;

        // Detect beat
        const now = Date.now();
        const timeSinceLastBeat = now - this.beatDetection.lastBeatTime;

        if (beatStrength > average * this.beatDetection.threshold &&
            timeSinceLastBeat > this.beatDetection.minTimeBetweenBeats) {

            this.beatDetection.currentBeat = true;
            this.beatDetection.lastBeatTime = now;

            // Add to BPM calculation
            this.beatDetection.beatBuffer.push(now);
            if (this.beatDetection.beatBuffer.length > 10) {
                this.beatDetection.beatBuffer.shift();
            }

            // Calculate BPM
            this.calculateBPM();
        } else {
            this.beatDetection.currentBeat = false;
        }
    }

    calculateBPM() {
        if (this.beatDetection.beatBuffer.length < 3) return;

        const intervals = [];
        for (let i = 1; i < this.beatDetection.beatBuffer.length; i++) {
            intervals.push(this.beatDetection.beatBuffer[i] - this.beatDetection.beatBuffer[i - 1]);
        }

        const averageInterval = intervals.reduce((a, b) => a + b, 0) / intervals.length;
        this.beatDetection.bpm = Math.round(60000 / averageInterval);

        // Clamp to reasonable range
        this.beatDetection.bpm = Math.max(60, Math.min(180, this.beatDetection.bpm));
    }

    updateHeartSync() {
        if (!this.heartSync.enabled) return;

        // Use detected BPM or fallback to target
        const effectiveBPM = this.beatDetection.bpm > 0 ? this.beatDetection.bpm : this.heartSync.targetBPM;

        // Update heart phase
        const bps = effectiveBPM / 60; // beats per second
        const deltaPhase = (bps * 2 * Math.PI) / 60; // phase increment per frame (assuming 60fps)
        this.heartSync.currentPhase += deltaPhase;

        // Keep phase in range
        if (this.heartSync.currentPhase > 2 * Math.PI) {
            this.heartSync.currentPhase -= 2 * Math.PI;
        }

        // Determine emotional state based on audio characteristics
        this.updateEmotionalState();
    }

    updateEmotionalState() {
        const bass = this.frequencyBands.bass.value;
        const mid = this.frequencyBands.mid.value;
        const treble = this.frequencyBands.treble.value;
        const energy = bass + mid + treble;

        if (energy > 0.8) {
            this.heartSync.emotionalState = 'excited';
        } else if (energy > 0.5) {
            this.heartSync.emotionalState = 'energetic';
        } else if (energy > 0.2) {
            this.heartSync.emotionalState = 'calm';
        } else {
            this.heartSync.emotionalState = 'peaceful';
        }
    }

    triggerCallbacks() {
        const audioData = this.getAudioData();

        // Trigger visualization callbacks
        this.visualizationCallbacks.forEach(callback => {
            try {
                callback(audioData);
            } catch (error) {
                console.error('❌ Visualization callback error:', error);
            }
        });

        // Trigger particle system callbacks
        this.particleSystemCallbacks.forEach(callback => {
            try {
                callback(audioData);
            } catch (error) {
                console.error('❌ Particle system callback error:', error);
            }
        });
    }

    getAudioData() {
        return {
            frequencyBands: { ...this.frequencyBands },
            beat: {
                detected: this.beatDetection.currentBeat,
                strength: this.beatDetection.history[this.beatDetection.history.length - 1] || 0,
                bpm: this.beatDetection.bpm
            },
            heart: {
                phase: this.heartSync.currentPhase,
                resonance: this.heartSync.resonanceStrength,
                emotionalState: this.heartSync.emotionalState,
                bpm: this.heartSync.targetBPM
            },
            energy: {
                overall: (this.frequencyBands.bass.value + this.frequencyBands.mid.value + this.frequencyBands.treble.value) / 3,
                bass: this.frequencyBands.bass.value,
                mid: this.frequencyBands.mid.value,
                treble: this.frequencyBands.treble.value
            },
            rawData: this.dataArray
        };
    }

    // Integration methods
    addVisualizationCallback(callback) {
        this.visualizationCallbacks.push(callback);
        console.log('🎵 Visualization callback added');
    }

    addParticleSystemCallback(callback) {
        this.particleSystemCallbacks.push(callback);
        console.log('✨ Particle system callback added');
    }

    removeCallback(callback) {
        this.visualizationCallbacks = this.visualizationCallbacks.filter(cb => cb !== callback);
        this.particleSystemCallbacks = this.particleSystemCallbacks.filter(cb => cb !== callback);
    }

    // Heart sync controls
    enableHeartSync(targetBPM = 70, resonanceStrength = 0.5) {
        this.heartSync.enabled = true;
        this.heartSync.targetBPM = targetBPM;
        this.heartSync.resonanceStrength = resonanceStrength;
        console.log(`💗 Heart sync enabled: ${targetBPM} BPM`);
    }

    disableHeartSync() {
        this.heartSync.enabled = false;
        console.log('💗 Heart sync disabled');
    }

    // Beat detection controls
    setBeatSensitivity(threshold = 1.3) {
        this.beatDetection.threshold = threshold;
        console.log(`🥁 Beat sensitivity: ${threshold}`);
    }

    // Audio visualization helpers
    getVisualizationData(type = 'frequency') {
        if (!this.analyser) return null;

        switch (type) {
            case 'frequency':
                this.analyser.getByteFrequencyData(this.dataArray);
                return Array.from(this.dataArray);

            case 'waveform':
                this.analyser.getByteTimeDomainData(this.dataArray);
                return Array.from(this.dataArray);

            case 'bands':
                return this.frequencyBands;

            default:
                return this.getAudioData();
        }
    }

    // Cleanup
    stop() {
        this.isActive = false;

        if (this.microphone) {
            this.microphone.disconnect();
        }

        if (this.analyser) {
            this.analyser.disconnect();
        }

        if (this.audioContext) {
            this.audioContext.close();
        }

        console.log('🎵 Audio Reactor stopped');
    }

    // Static factory methods
    static async createWithMicrophone() {
        const reactor = new NexusForgeAudioReactor();
        await reactor.initialize();
        await reactor.requestMicrophoneAccess();
        return reactor;
    }

    static async createWithAudioElement(audioElement) {
        const reactor = new NexusForgeAudioReactor();
        await reactor.initialize();
        reactor.connectAudioElement(audioElement);
        return reactor;
    }
}

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = NexusForgeAudioReactor;
}

// Auto-initialize if in browser
if (typeof window !== 'undefined') {
    window.NexusForgeAudioReactor = NexusForgeAudioReactor;
    console.log('🎵 Audio Reactor available globally');
}
