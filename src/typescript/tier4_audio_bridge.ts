/**
 * Tier-4 Audio Bridge - Connect real-time audio to Tier-4 Meta System
 * ==================================================================
 *
 * Bridges web-audio-integration.js to tier4_meta_system for live processing.
 * Audio features automatically trigger Tier-4 operators in real-time.
 */

class Tier4AudioBridge {
    constructor(tier4System, audioProcessor) {
        this.tier4 = tier4System;
        this.audio = audioProcessor;
        this.isActive = false;
        this.mappingRules = new Map();
        this.operatorQueue = [];
        this.lastTrigger = 0;
        this.minInterval = 100; // ms between triggers

        this.setupAudioMappings();
        this.connectToAudio();
    }

    // ============================= Audio → Operator Mapping =============================

    setupAudioMappings() {
        // Map audio features to Tier-4 operators
        this.mappingRules.set('onset_detected', {
            operator: 'ST',
            condition: (features) => features.spectralCentroid > 0.7,
            strength: (features) => Math.min(features.energy * 2, 1.0)
        });

        this.mappingRules.set('sustained_energy', {
            operator: 'UP',
            condition: (features) => features.energy > 0.6 && features.zcr < 0.1,
            strength: (features) => features.energy
        });

        this.mappingRules.set('high_complexity', {
            operator: 'CNV',
            condition: (features) => features.zcr > 0.15 && features.spectralCentroid > 0.5,
            strength: (features) => (features.zcr + features.spectralCentroid) / 2
        });

        this.mappingRules.set('rhythmic_pattern', {
            operator: 'RB',
            condition: (features) => features.tempo > 100 && features.tempo < 140,
            strength: (features) => Math.sin(features.phase) * 0.5 + 0.5
        });

        this.mappingRules.set('silence_transition', {
            operator: 'PRV',
            condition: (features) => features.energy < 0.1,
            strength: (features) => 1.0 - features.energy
        });

        this.mappingRules.set('voice_detected', {
            operator: 'EDT',
            condition: (features) => features.spectralCentroid > 0.3 && features.spectralCentroid < 0.7,
            strength: (features) => 1.0 - Math.abs(features.spectralCentroid - 0.5) * 2
        });

        // Three Ides Macros triggered by complex audio patterns
        this.mappingRules.set('complex_analysis_pattern', {
            operator: 'IDE_A',
            condition: (features) => features.complexity > 0.8,
            strength: (features) => features.complexity
        });

        this.mappingRules.set('constrained_harmonic', {
            operator: 'IDE_B',
            condition: (features) => features.harmonic && features.energy > 0.5,
            strength: (features) => features.energy * (features.harmonic ? 1.2 : 1.0)
        });

        this.mappingRules.set('build_crescendo', {
            operator: 'IDE_C',
            condition: (features) => this.detectCrescendo(features),
            strength: (features) => this.getCrescendoStrength(features)
        });
    }

    // ============================= Audio Event Processing =============================

    connectToAudio() {
        if (!this.audio) {
            console.warn('Tier4AudioBridge: No audio processor provided');
            return;
        }

        // Connect to semantic updates from audio processor
        this.audio.addEventListener('semanticUpdate', (output) => {
            if (this.isActive) {
                this.processSemanticsToTier4(output);
            }
        });

        // Connect to raw audio frames for real-time analysis
        this.audio.addEventListener('audioFrame', (data) => {
            if (this.isActive) {
                this.processAudioFrame(data);
            }
        });
    }

    processSemanticsToTier4(semantics) {
        const now = Date.now();
        if (now - this.lastTrigger < this.minInterval) return;

        // Extract relevant features
        const features = {
            energy: semantics.features?.energy || 0,
            spectralCentroid: semantics.features?.spectralCentroid || 0.5,
            zcr: semantics.features?.zcr || 0,
            tempo: semantics.tempo || 120,
            phase: semantics.phase || 0,
            complexity: this.computeComplexity(semantics),
            harmonic: this.detectHarmonic(semantics)
        };

        // Check mapping rules and trigger operators
        for (const [eventName, rule] of this.mappingRules) {
            if (rule.condition(features)) {
                const strength = rule.strength(features);
                this.queueOperator(rule.operator, strength, eventName, features);
            }
        }

        // Process operator queue
        this.processOperatorQueue();
        this.lastTrigger = now;
    }

    processAudioFrame(data) {
        // Additional real-time processing if needed
        // For now, rely on semantics processing
    }

    // ============================= Operator Queue Management =============================

    queueOperator(operatorId, strength, trigger, features) {
        // Prevent operator spam
        const existing = this.operatorQueue.find(op => op.operator === operatorId);
        if (existing) {
            existing.strength = Math.max(existing.strength, strength);
            existing.triggers.push(trigger);
            return;
        }

        this.operatorQueue.push({
            operator: operatorId,
            strength: strength,
            triggers: [trigger],
            features: features,
            timestamp: Date.now()
        });
    }

    processOperatorQueue() {
        if (this.operatorQueue.length === 0) return;

        // Sort by strength (highest first)
        this.operatorQueue.sort((a, b) => b.strength - a.strength);

        // Apply the strongest operator
        const topOp = this.operatorQueue[0];

        try {
            this.applyTier4Operator(topOp);
        } catch (error) {
            console.error('Tier4AudioBridge: Failed to apply operator', topOp.operator, error);
        }

        // Clear queue
        this.operatorQueue = [];
    }

    applyTier4Operator(operation) {
        if (!this.tier4) {
            console.warn('Tier4AudioBridge: No Tier-4 system connected');
            return;
        }

        // Apply to Tier-4 system
        if (this.tier4.clickButton) {
            // Browser demo version
            this.tier4.clickButton(operation.operator);
        } else if (this.tier4.applyOperator) {
            // TypeScript version
            this.tier4.applyOperator(operation.operator, {
                strength: operation.strength,
                audioTrigger: true,
                triggers: operation.triggers
            });
        }

        // Log the audio-triggered operation
        console.log(`🎵 Audio triggered: ${operation.operator} (strength: ${(operation.strength * 100).toFixed(0)}%) via ${operation.triggers.join(', ')}`);

        // Update HUD if available
        if (this.tier4.updateHUD) {
            this.tier4.updateHUD({
                op: operation.operator,
                trigger: 'audio',
                strength: operation.strength,
                audioFeatures: operation.features
            });
        }
    }

    // ============================= Audio Analysis Helpers =============================

    computeComplexity(semantics) {
        if (!semantics.features) return 0;

        // Combine multiple features for complexity measure
        const f = semantics.features;
        const spectralSpread = f.spectralSpread || 0;
        const spectralFlux = f.spectralFlux || 0;
        const zcr = f.zcr || 0;

        return Math.min(1.0, (spectralSpread * 0.4 + spectralFlux * 0.4 + zcr * 0.2));
    }

    detectHarmonic(semantics) {
        if (!semantics.features) return false;

        // Simple harmonic detection based on spectral centroid stability
        const centroid = semantics.features.spectralCentroid || 0.5;
        const spread = semantics.features.spectralSpread || 1.0;

        return spread < 0.3 && centroid > 0.2 && centroid < 0.8;
    }

    detectCrescendo(features) {
        // Simple crescendo detection - would need history for real implementation
        return features.energy > 0.7 && features.tempo > 110;
    }

    getCrescendoStrength(features) {
        return Math.min(1.0, features.energy * (features.tempo / 120));
    }

    // ============================= Control Interface =============================

    start() {
        this.isActive = true;
        console.log('🎵 Tier-4 Audio Bridge: ACTIVE');
    }

    stop() {
        this.isActive = false;
        this.operatorQueue = [];
        console.log('🎵 Tier-4 Audio Bridge: STOPPED');
    }

    setMinInterval(ms) {
        this.minInterval = Math.max(50, ms);
    }

    // Add custom mapping rule
    addMappingRule(name, rule) {
        this.mappingRules.set(name, rule);
    }

    // Remove mapping rule
    removeMappingRule(name) {
        this.mappingRules.delete(name);
    }

    // Get current status
    getStatus() {
        return {
            active: this.isActive,
            mappingRules: this.mappingRules.size,
            queueLength: this.operatorQueue.length,
            lastTrigger: this.lastTrigger,
            minInterval: this.minInterval
        };
    }
}

// ============================= Integration Helper =============================

/**
 * Factory function to create and connect Tier-4 Audio Bridge
 */
export function createTier4AudioBridge(tier4System) {
    // Try to find existing audio processor
    let audioProcessor = null;

    if (window.webAudioSemanticProcessor) {
        audioProcessor = window.webAudioSemanticProcessor;
    } else if (window.WebAudioSemanticProcessor) {
        // Create new audio processor
        audioProcessor = new window.WebAudioSemanticProcessor();
        window.webAudioSemanticProcessor = audioProcessor;
    }

    if (!audioProcessor) {
        console.warn('Tier4AudioBridge: No audio processor found. Make sure web-audio-integration.js is loaded.');
        return null;
    }

    return new Tier4AudioBridge(tier4System, audioProcessor);
}

// ============================= Usage Example =============================

/*
// In your tier4_meta_system_demo.js, add this:

// Connect audio bridge
const audioBridge = createTier4AudioBridge(worldEngine);
if (audioBridge) {
    audioBridge.start();

    // Add custom mapping for specific audio pattern
    audioBridge.addMappingRule('debug_clap', {
        operator: 'RST',
        condition: (features) => features.energy > 0.9 && features.zcr > 0.2,
        strength: (features) => 1.0
    });
}

// Access the bridge later:
// audioBridge.stop(); // Stop audio processing
// audioBridge.getStatus(); // Check status
*/

export { Tier4AudioBridge, createTier4AudioBridge };
