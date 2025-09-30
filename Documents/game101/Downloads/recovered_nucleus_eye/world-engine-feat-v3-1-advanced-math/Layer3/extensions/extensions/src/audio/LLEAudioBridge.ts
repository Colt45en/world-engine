/**
 * LLE Audio Bridge - Real-time audio feedback for LLE operations
 * ==============================================================
 *
 * Integrates Tier4AudioBridge with LLE core for:
 * - Audio feedback during button clicks
 * - Real-time audio visualization of SU states
 * - Audio-triggered LLE operations
 * - Live coding session audio enhancement
 */

import { SU, Button, click as lleClick } from '../lle/core';
import { Vec, Mat } from '../lle/algebra';

export interface AudioFeatures {
    energy: number;
    spectralCentroid: number;
    zcr: number; // Zero crossing rate
    tempo: number;
    phase: number;
}

export interface AudioMapping {
    operator: string;
    condition: (features: AudioFeatures) => boolean;
    strength: (features: AudioFeatures) => number;
    cooldown?: number;
}

export interface LLEAudioEvent {
    type: 'audio_trigger' | 'lle_feedback' | 'state_change';
    operator?: string;
    su?: SU;
    features?: AudioFeatures;
    timestamp: number;
}

export class LLEAudioBridge {
    private isListening = false;
    private mappings = new Map<string, AudioMapping>();
    private lastTrigger = new Map<string, number>();
    private callbacks = new Set<(event: LLEAudioEvent) => void>();

    constructor() {
        this.setupDefaultMappings();
    }

    async initialize(): Promise<void> {
        // Audio initialization will be handled by webview client-side
        console.log('LLE Audio Bridge initialized (server-side)');
    }

    private setupDefaultMappings(): void {
        // Map audio features to LLE operators (matching Tier4AudioBridge)
        this.mappings.set('onset_detected', {
            operator: 'ST',
            condition: (features) => features.spectralCentroid > 0.7,
            strength: (features) => Math.min(features.energy * 2, 1.0),
            cooldown: 500
        });

        this.mappings.set('sustained_energy', {
            operator: 'UP',
            condition: (features) => features.energy > 0.6 && features.zcr < 0.1,
            strength: (features) => features.energy,
            cooldown: 200
        });

        this.mappings.set('high_complexity', {
            operator: 'CNV',
            condition: (features) => features.zcr > 0.15 && features.spectralCentroid > 0.5,
            strength: (features) => (features.zcr + features.spectralCentroid) / 2,
            cooldown: 300
        });

        this.mappings.set('rhythmic_pattern', {
            operator: 'RB',
            condition: (features) => features.tempo > 100 && features.tempo < 140,
            strength: (features) => Math.sin(features.phase) * 0.5 + 0.5,
            cooldown: 150
        });

        this.mappings.set('silence_break', {
            operator: 'CP',
            condition: (features) => features.energy > 0.3 && features.zcr > 0.05,
            strength: (features) => features.energy * features.zcr,
            cooldown: 400
        });
    }

    startListening(): void {
        this.isListening = true;
        console.log('Audio listening started');
    }

    stopListening(): void {
        this.isListening = false;
        console.log('Audio listening stopped');
    }

    // Process audio features from webview
    processAudioFeatures(features: AudioFeatures): void {
        if (!this.isListening) return;

        const now = Date.now();

        for (const [name, mapping] of this.mappings.entries()) {
            const lastTrigger = this.lastTrigger.get(name) || 0;
            const cooldown = mapping.cooldown || 100;

            if (now - lastTrigger < cooldown) continue;

            if (mapping.condition(features)) {
                const strength = mapping.strength(features);

                this.lastTrigger.set(name, now);

                this.emit({
                    type: 'audio_trigger',
                    operator: mapping.operator,
                    features,
                    timestamp: now
                });
            }
        }
    }

    // Generate audio feedback for LLE operations
    generateLLEFeedback(su: SU, operator: string): LLEAudioEvent {
        const event: LLEAudioEvent = {
            type: 'lle_feedback',
            operator,
            su,
            timestamp: Date.now()
        };

        this.emit(event);
        return event;
    }

    // Generate audio parameters for webview client
    generateAudioParams(su: SU, operator: string): { frequency: number; duration: number; amplitude: number } {
        return {
            frequency: this.suToFrequency(su),
            duration: this.operatorToDuration(operator),
            amplitude: Math.min(su.kappa, 0.3)
        };
    }

    private suToFrequency(su: SU): number {
        // Map SU state to audio frequency
        const base = 220; // A3
        const xMagnitude = Math.sqrt(su.x[0] ** 2 + su.x[1] ** 2 + su.x[2] ** 2);
        const kappaFactor = Math.log(su.kappa + 1);

        return base * (1 + xMagnitude) * (1 + kappaFactor);
    }

    private operatorToDuration(operator: string): number {
        const durations: Record<string, number> = {
            'ST': 0.1,   // Short spike
            'UP': 0.3,   // Medium rise
            'CNV': 0.5,  // Longer convolution
            'RB': 0.2,   // Rhythmic pulse
            'CP': 0.15,  // Copy snap
            'PR': 0.4,   // Project sweep
            'CV': 0.6,   // Convert transformation
            'MD': 0.8    // Mode shift
        };

        return durations[operator] || 0.2;
    }

    getMappings(): Map<string, AudioMapping> {
        return new Map(this.mappings);
    }

    addMapping(name: string, mapping: AudioMapping): void {
        this.mappings.set(name, mapping);
    }

    removeMapping(name: string): void {
        this.mappings.delete(name);
    }

    onAudioEvent(callback: (event: LLEAudioEvent) => void): void {
        this.callbacks.add(callback);
    }

    removeAudioEvent(callback: (event: LLEAudioEvent) => void): void {
        this.callbacks.delete(callback);
    }

    private emit(event: LLEAudioEvent): void {
        for (const callback of this.callbacks) {
            try {
                callback(event);
            } catch (error) {
                console.error('Error in audio event callback:', error);
            }
        }
    }

    dispose(): void {
        this.stopListening();
        this.callbacks.clear();
        this.mappings.clear();
        this.lastTrigger.clear();
    }
}
