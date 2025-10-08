/**
 * Vibe Engine Integration for NEXUS FORGE PRIMORDIAL
 * =================================================
 *
 * Real-time audio-driven intelligence layer that connects mic input
 * to all NEXUS FORGE systems through the Vibe Engine's 4D state:
 * • p (polarity) → Emotional context and pain sentiment
 * • i (intensity) → Development velocity and system energy
 * • g (genre) → Code complexity and pattern recognition mode
 * • c (confidence) → System certainty and decision strength
 */

class VibeEngineIntegration {
    constructor() {
        this.vibeEngine = null;
        this.audioContext = null;
        this.analyzer = null;
        this.microphone = null;

        // Integration connections
        this.nexusForge = null;
        this.quantumEngine = null;
        this.vortexLab = null;
        this.runeGrid = null;

        // Audio processing
        this.audioBuffer = new Float32Array(1024);
        this.isListening = false;
        this.frameCount = 0;
        this.lastFrameTime = 0;

        // Feature extraction state
        this.previousSpectrum = null;
        this.energyHistory = [];
        this.centroidHistory = [];
        this.onsetThreshold = 0.3;
        this.voiceConfidence = 0.0;

        // Integration state
        this.lastVibeState = { p: 0, i: 0, g: 0, c: 0 };
        this.stateChangeThreshold = 0.1;
        this.systemResponseEnabled = true;

        console.log('🌊 Vibe Engine Integration initialized');
    }

    async initialize() {
        try {
            // Initialize VibeEngine (assuming it's available)
            if (typeof VibeEngine !== 'undefined') {
                this.vibeEngine = new VibeEngine({
                    decay: { p: 0.4, i: 0.7, g: 0.5, c: 0.2 },
                    beta: { i: 0.8, g: 0.5, c: 0.4 },
                    gammas: { E: 10.0, C: 8.0, Z: 7.0, v: 3.0, P: 0.003 },
                    intensityCap: 2.5,
                    cmin: 0.15
                });
                console.log('✅ VibeEngine core initialized');
            } else {
                console.warn('⚠️ VibeEngine not available - using mock system');
                this.vibeEngine = this.createMockVibeEngine();
            }

            // Initialize audio context
            await this.initializeAudio();

            return true;
        } catch (error) {
            console.error('❌ Vibe Engine initialization failed:', error);
            return false;
        }
    }

    async initializeAudio() {
        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
            throw new Error('WebAudio/getUserMedia not supported');
        }

        this.audioContext = new (window.AudioContext || window.webkitAudioContext)();

        // Get microphone access
        const stream = await navigator.mediaDevices.getUserMedia({
            audio: {
                echoCancellation: false,
                noiseSuppression: false,
                autoGainControl: false
            }
        });

        // Create audio processing chain
        const source = this.audioContext.createMediaStreamSource(stream);
        this.analyzer = this.audioContext.createAnalyser();

        this.analyzer.fftSize = 2048;
        this.analyzer.smoothingTimeConstant = 0.3;

        source.connect(this.analyzer);
        this.microphone = stream;

        console.log('🎤 Audio system initialized');
    }

    connectSystems(systems) {
        this.nexusForge = systems.nexusForge;
        this.quantumEngine = systems.quantumEngine;
        this.vortexLab = systems.vortexLab;
        this.runeGrid = systems.runeGrid;

        console.log('🔗 Vibe Engine connected to NEXUS FORGE systems');
    }

    startListening() {
        if (!this.vibeEngine || !this.analyzer) {
            console.error('❌ Cannot start listening - system not ready');
            return false;
        }

        this.isListening = true;
        this.lastFrameTime = performance.now();
        this.processAudioFrame();

        console.log('👂 Vibe Engine listening started');
        return true;
    }

    stopListening() {
        this.isListening = false;

        if (this.microphone) {
            this.microphone.getTracks().forEach(track => track.stop());
        }

        console.log('🔇 Vibe Engine listening stopped');
    }

    processAudioFrame() {
        if (!this.isListening) return;

        const currentTime = performance.now();
        const dt = Math.max(0.001, (currentTime - this.lastFrameTime) / 1000);
        this.lastFrameTime = currentTime;
        this.frameCount++;

        // Extract audio features
        const features = this.extractAudioFeatures(dt);

        // Update Vibe Engine
        const vibeState = this.vibeEngine.step(features);

        // Check for significant state changes
        if (this.hasStateChanged(vibeState)) {
            this.updateConnectedSystems(vibeState, features);
            this.lastVibeState = { ...vibeState };
        }

        // Continue processing
        if (this.isListening) {
            requestAnimationFrame(() => this.processAudioFrame());
        }
    }

    extractAudioFeatures(dt) {
        // Get frequency domain data
        const freqData = new Uint8Array(this.analyzer.frequencyBinCount);
        const timeData = new Uint8Array(this.analyzer.fftSize);

        this.analyzer.getByteFrequencyData(freqData);
        this.analyzer.getByteTimeDomainData(timeData);

        // Calculate RMS energy
        let rms = 0;
        for (let i = 0; i < timeData.length; i++) {
            const sample = (timeData[i] - 128) / 128;
            rms += sample * sample;
        }
        rms = Math.sqrt(rms / timeData.length);

        // Calculate spectral centroid
        let weightedSum = 0;
        let magnitude = 0;
        for (let i = 1; i < freqData.length; i++) {
            const freq = i * this.audioContext.sampleRate / (2 * freqData.length);
            const mag = freqData[i] / 255;
            weightedSum += freq * mag;
            magnitude += mag;
        }
        const centroid = magnitude > 0 ? weightedSum / magnitude : 0;
        const normalizedCentroid = Math.min(1, centroid / 8000); // Normalize to 0-1

        // Calculate spectral flux (change in spectrum)
        let flux = 0;
        if (this.previousSpectrum) {
            for (let i = 0; i < freqData.length; i++) {
                const diff = (freqData[i] / 255) - this.previousSpectrum[i];
                if (diff > 0) flux += diff;
            }
        }
        this.previousSpectrum = Array.from(freqData).map(x => x / 255);

        // Calculate zero crossing rate
        let zcr = 0;
        for (let i = 1; i < timeData.length; i++) {
            const prev = (timeData[i - 1] - 128) / 128;
            const curr = (timeData[i] - 128) / 128;
            if (prev * curr < 0) zcr++;
        }
        zcr = zcr / (timeData.length - 1);

        // Simple onset detection
        this.energyHistory.push(rms);
        if (this.energyHistory.length > 10) this.energyHistory.shift();

        const avgEnergy = this.energyHistory.reduce((a, b) => a + b, 0) / this.energyHistory.length;
        const onset = rms > avgEnergy * (1 + this.onsetThreshold);

        // Voice detection (simplified)
        const voiced = rms > 0.01 && zcr < 0.1 && centroid > 200 && centroid < 4000;
        const pitchHz = voiced ? this.estimatePitch(timeData) : 0;

        // Update voice confidence
        this.voiceConfidence = voiced ?
            Math.min(1, this.voiceConfidence + 0.1) :
            Math.max(0, this.voiceConfidence - 0.05);

        return {
            rms,
            centroid: normalizedCentroid,
            flux,
            pitchHz,
            zcr,
            onset,
            voiced: this.voiceConfidence > 0.5,
            dt
        };
    }

    estimatePitch(timeData) {
        // Simple autocorrelation-based pitch detection
        const sampleRate = this.audioContext.sampleRate;
        const minPeriod = Math.floor(sampleRate / 800); // 800 Hz max
        const maxPeriod = Math.floor(sampleRate / 80);  // 80 Hz min

        let bestPeriod = 0;
        let bestCorrelation = 0;

        for (let period = minPeriod; period < maxPeriod; period++) {
            let correlation = 0;
            for (let i = 0; i < timeData.length - period; i++) {
                const sample1 = (timeData[i] - 128) / 128;
                const sample2 = (timeData[i + period] - 128) / 128;
                correlation += sample1 * sample2;
            }

            if (correlation > bestCorrelation) {
                bestCorrelation = correlation;
                bestPeriod = period;
            }
        }

        return bestPeriod > 0 ? sampleRate / bestPeriod : 0;
    }

    hasStateChanged(newState) {
        const threshold = this.stateChangeThreshold;
        return (
            Math.abs(newState.p - this.lastVibeState.p) > threshold ||
            Math.abs(newState.i - this.lastVibeState.i) > threshold ||
            Math.abs(newState.g - this.lastVibeState.g) > threshold ||
            Math.abs(newState.c - this.lastVibeState.c) > threshold
        );
    }

    updateConnectedSystems(vibeState, features) {
        if (!this.systemResponseEnabled) return;

        // Update NEXUS FORGE AI with vibe-driven metrics
        if (this.nexusForge) {
            this.updateNexusForge(vibeState, features);
        }

        // Drive Quantum Graphics with audio energy
        if (this.quantumEngine) {
            this.updateQuantumEngine(vibeState, features);
        }

        // Sync VortexLab limbs with vibe state
        if (this.vortexLab) {
            this.updateVortexLab(vibeState, features);
        }

        // Trigger Rune Grid responses
        if (this.runeGrid) {
            this.updateRuneGrid(vibeState, features);
        }

        // Log significant changes
        this.logVibeStateChange(vibeState, features);
    }

    updateNexusForge(vibeState, features) {
        // Map vibe state to AI intelligence parameters
        const aiEnhancement = {
            emotionalContext: vibeState.p, // Polarity affects pain sentiment
            processingIntensity: vibeState.i, // Intensity drives analysis depth
            patternComplexity: vibeState.g, // Genre affects pattern recognition
            decisionConfidence: vibeState.c, // Confidence influences recommendations

            // Audio-specific context
            voiceActivity: features.voiced,
            energyLevel: features.rms,
            communicationClarity: features.centroid
        };

        // Trigger AI recalibration on significant voice activity
        if (features.voiced && features.rms > 0.1) {
            this.nexusForge.recalibrateWithContext(aiEnhancement);
        }

        // Adjust pain sensitivity based on emotional state
        const painSensitivity = 0.5 + vibeState.i * 0.5; // Higher intensity = more sensitive
        this.nexusForge.setPainSensitivity(painSensitivity);
    }

    updateQuantumEngine(vibeState, features) {
        // Drive quantum particle behavior
        const quantumParams = {
            particleIntensity: Math.max(0.1, vibeState.i * 2.0),
            fieldPolarity: vibeState.p,
            waveComplexity: vibeState.g,
            coherenceLevel: vibeState.c,

            // Audio-reactive effects
            energyBurst: features.onset ? features.rms * 5 : 0,
            spectralShift: features.centroid,
            rhythmicPulse: features.voiced ? 1.0 : 0.3
        };

        this.quantumEngine.updateVibeParameters(quantumParams);

        // Trigger special effects on onsets
        if (features.onset && features.rms > 0.2) {
            this.quantumEngine.triggerQuantumBurst(features.rms * 10);
        }
    }

    updateVortexLab(vibeState, features) {
        // Map vibe dimensions to limb behavior
        const vortexParams = {
            coreEnergy: 0.5 + vibeState.i * 0.8,
            resonance: vibeState.c * 2.0,
            limbModulation: {
                polarity: vibeState.p,
                intensity: vibeState.i,
                complexity: vibeState.g,
                confidence: vibeState.c
            },

            // Audio synchronization
            rhythmSync: features.voiced,
            energyLevel: features.rms,
            spectralResponse: features.centroid
        };

        this.vortexLab.updateAIMetrics(vortexParams);

        // Trigger vortex burst on strong audio events
        if (features.onset && features.rms > 0.15) {
            this.vortexLab.triggerBurst(vibeState.i * features.rms * 2);
        }
    }

    updateRuneGrid(vibeState, features) {
        // Audio-driven rune grid responses
        if (features.onset && vibeState.i > 0.8) {
            // High-intensity onsets trigger hot patches
            this.runeGrid.triggerHotPatch('audio-event', {
                changeType: 'audio-driven',
                intensity: vibeState.i,
                confidence: vibeState.c
            });
        }

        if (features.voiced && vibeState.c > 0.7) {
            // High-confidence voice creates sync connections
            this.runeGrid.createWire('system', 'voice-input', 'SYNC_LINE', {
                intensity: vibeState.c,
                data: { voiceActive: true, clarity: features.centroid }
            });
        }

        // Update grid energy based on overall vibe
        const gridEnergy = (vibeState.i + vibeState.g + vibeState.c) / 3;
        this.runeGrid.setSystemEnergy(gridEnergy);
    }

    logVibeStateChange(vibeState, features) {
        const intensity = Math.abs(vibeState.p) + vibeState.i + vibeState.g + vibeState.c;

        if (intensity > 2.0) {
            console.log('🌊 High-energy vibe state:', {
                polarity: vibeState.p.toFixed(3),
                intensity: vibeState.i.toFixed(3),
                genre: vibeState.g.toFixed(3),
                confidence: vibeState.c.toFixed(3),
                voiced: features.voiced,
                onset: features.onset
            });
        }
    }

    // Voice command processing
    processVoiceCommand(command) {
        if (!this.vibeEngine) return [];

        const operators = this.vibeEngine.word(command, 1.0);
        console.log(`🗣️ Voice command "${command}" → operators: [${operators.join(', ')}]`);

        // Apply operators to connected systems
        operators.forEach(op => this.applyOperatorToSystems(op));

        return operators;
    }

    applyOperatorToSystems(operator) {
        switch (operator) {
            case 'RB': // Rebuild
                if (this.nexusForge) this.nexusForge.triggerClustering();
                if (this.quantumEngine) this.quantumEngine.reset();
                break;

            case 'ST': // Status/Snapshot
                console.log('📸 System snapshot triggered by vibe');
                // Save current state of all systems
                break;

            case 'PRV': // Prevent
                if (this.nexusForge) this.nexusForge.preventOverload();
                break;

            case 'UP': // Update
                this.forceSystemUpdate();
                break;
        }
    }

    forceSystemUpdate() {
        // Force update all connected systems
        if (this.nexusForge) this.nexusForge.generateIntelligence();
        if (this.vortexLab) this.vortexLab.setQuantumIntensity(1.2);
        if (this.runeGrid) this.runeGrid.updateConnectionPoints();
    }

    // Public API
    getVibeState() {
        return this.vibeEngine ? this.vibeEngine.state() : { p: 0, i: 0, g: 0, c: 0 };
    }

    getMu() {
        return this.vibeEngine ? this.vibeEngine.mu() : 0;
    }

    toggleSystemResponse(enabled) {
        this.systemResponseEnabled = enabled;
        console.log(`🔧 System response ${enabled ? 'enabled' : 'disabled'}`);
    }

    createMockVibeEngine() {
        // Mock implementation for when VibeEngine isn't available
        return {
            state: () => ({ p: 0, i: 0.3, g: 0.5, c: 0.7 }),
            step: () => ({ p: Math.sin(Date.now() * 0.001) * 0.1, i: 0.3, g: 0.5, c: 0.7 }),
            word: () => ['RB'],
            mu: () => 1.5,
            apply: () => { },
            snapshot: () => { },
            restore: () => { }
        };
    }

    // Cleanup
    destroy() {
        this.stopListening();

        if (this.audioContext) {
            this.audioContext.close();
        }

        console.log('🗑️ Vibe Engine Integration destroyed');
    }
}

// Global integration
window.VibeEngineIntegration = VibeEngineIntegration;
