// Sensory channel types for real-time scene modulation
export type SensoryChannel = 'sight' | 'sound' | 'touch' | 'scent' | 'taste' | 'inner'

export interface SensoryDetail {
    channel: SensoryChannel
    strength: number // 0..1 intensity
    description?: string // optional label/word that triggered this
    decay?: number // optional decay rate (default 0.98)
}

export interface SceneMoment {
    timestamp: number
    details: SensoryDetail[]
    // Optional metadata
    source?: string // what triggered this moment
    duration?: number // how long this moment should last
    priority?: number // for blending multiple moments
}

// Lexicon mapping: word → channel intensities
export type SensoryLexicon = Record<string, Partial<Record<SensoryChannel, number>>>

// Proximity configuration
export interface ProximityConfig {
    center: THREE.Vector3 | THREE.Object3D
    radius: number
    falloff: number // 0..1 softness
    channels?: SensoryChannel[] // which channels to affect
    debug?: boolean
}

// Token stream configuration
export interface TokenStreamConfig {
    text: string
    tps: number // tokens per second
    lexicon?: SensoryLexicon // custom lexicon override
    decay?: number // global decay rate when no token match
}

// Real-time sensory state
export interface SensoryState {
    currentMoment: SceneMoment
    baseMoment: SceneMoment
    isActive: boolean
    proximityGain: number
    tokenIndex: number
    lastUpdate: number
}

// PostFX channel mappings
export interface ChannelFXMapping {
    sight: {
        bloom?: number
        brightness?: number
        contrast?: number
    }
    sound: {
        distortion?: number
        reverb?: number
    }
    touch: {
        roughness?: number
        displacement?: number
    }
    scent: {
        particles?: number
        opacity?: number
    }
    taste: {
        colorShift?: number
        saturation?: number
    }
    inner: {
        vignette?: number
        dof?: number
        noise?: number
    }
}

// Accessibility-aware sensory settings
export interface AccessibleSensoryConfig {
    respectReducedMotion: boolean
    maxIntensity: number // 0..1 cap for motion-sensitive users
    gentleTransitions: boolean
    visualOnly: boolean // disable audio-driven effects
    highContrast: boolean
}

// Default sensory moment factory
export function createBaseMoment(): SceneMoment {
    return {
        timestamp: Date.now(),
        details: [
            { channel: 'sight', strength: 0.5 },
            { channel: 'sound', strength: 0.3 },
            { channel: 'touch', strength: 0.2 },
            { channel: 'scent', strength: 0.1 },
            { channel: 'taste', strength: 0.1 },
            { channel: 'inner', strength: 0.4 },
        ]
    }
}

// Channel strength utilities
export const ChannelUtils = {
    getStrength: (moment: SceneMoment, channel: SensoryChannel): number => {
        return moment.details.find(d => d.channel === channel)?.strength ?? 0
    },

    setStrength: (moment: SceneMoment, channel: SensoryChannel, strength: number): SceneMoment => {
        const clone = structuredClone(moment)
        const detail = clone.details.find(d => d.channel === channel)
        if (detail) {
            detail.strength = Math.max(0, Math.min(1, strength))
        }
        return clone
    },

    blend: (momentA: SceneMoment, momentB: SceneMoment, factor: number): SceneMoment => {
        const clone = structuredClone(momentA)
        clone.details.forEach(detail => {
            const bStrength = ChannelUtils.getStrength(momentB, detail.channel)
            detail.strength = detail.strength * (1 - factor) + bStrength * factor
        })
        return clone
    },

    decay: (moment: SceneMoment, rate: number = 0.98): SceneMoment => {
        const clone = structuredClone(moment)
        clone.details.forEach(detail => {
            detail.strength = Math.max(0, detail.strength * rate)
        })
        return clone
    }
}
