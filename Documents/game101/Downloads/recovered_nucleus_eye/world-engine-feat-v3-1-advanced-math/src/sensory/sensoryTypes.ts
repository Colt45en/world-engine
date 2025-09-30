// Core sensory types and utilities for World Engine
export type SensoryChannel = "sight" | "sound" | "touch" | "scent" | "taste" | "inner";

export type SensoryDetail = {
    channel: SensoryChannel;
    description: string;
    strength?: number; // 0..1 (drives intensity)
};

export type SceneMoment = {
    id: string;
    label: string;
    perspective: "attuned" | "oblivious" | "object";
    details: SensoryDetail[];
    timestamp?: number;
};

// Channel utilities
export const byChannel = (moment: SceneMoment, channel: SensoryChannel) =>
    moment.details.filter(d => d.channel === channel);

export const getChannelStrength = (moment: SceneMoment, channel: SensoryChannel): number => {
    const details = byChannel(moment, channel);
    return details.reduce((sum, d) => sum + (d.strength ?? 0.5), 0) / Math.max(details.length, 1);
};

// Channel color mapping for UI consistency
export const CHANNEL_COLORS: Record<SensoryChannel, string> = {
    sight: "#9bd4ff",
    sound: "#c6b1ff",
    touch: "#9bffc9",
    scent: "#ffd580",
    taste: "#ff9bb1",
    inner: "#fff"
};

// Perspective modifiers
export const applyPerspective = (moment: SceneMoment): SceneMoment => {
    const clone = structuredClone(moment);

    for (const detail of clone.details) {
        switch (clone.perspective) {
            case "oblivious":
                detail.strength = (detail.strength ?? 0.5) * 0.3;
                break;
            case "object":
                // Flip emphasis: sight low, inner high
                if (detail.channel === "sight") detail.strength = 0.2;
                if (detail.channel === "inner") detail.strength = Math.max(0.8, detail.strength ?? 0.5);
                break;
            case "attuned":
            default:
                // No modification for attuned perspective
                break;
        }
    }

    return clone;
};

// Preset moments for quick testing
export const PRESET_MOMENTS: Record<string, Omit<SceneMoment, 'id' | 'perspective'>> = {
    sunrise: {
        label: "Sunrise",
        details: [
            { channel: "sight", description: "Glass rinsed in amber; light bends like a slow tide.", strength: 0.8 },
            { channel: "sound", description: "Pipes exhale; the building sighs into morning.", strength: 0.6 },
            { channel: "touch", description: "Air brushes the wrist—fine as breath over water.", strength: 0.5 },
            { channel: "scent", description: "A candle's ghost, sweet and faint.", strength: 0.5 },
            { channel: "taste", description: "Storm-bright crispness at the back of the tongue.", strength: 0.4 },
            { channel: "inner", description: "Time leans forward, listening.", strength: 0.7 },
        ]
    },
    alley: {
        label: "Forgotten Alley",
        details: [
            { channel: "sight", description: "Light shards in puddles; a cat prints the surface with silence.", strength: 0.7 },
            { channel: "sound", description: "Drips keep time; a vent whispers secrets into grate-teeth.", strength: 0.7 },
            { channel: "touch", description: "Concrete breathes damp against palm.", strength: 0.6 },
            { channel: "scent", description: "Wet iron and orange peel, forgotten.", strength: 0.6 },
            { channel: "taste", description: "Salt-metal tang of old rain.", strength: 0.5 },
            { channel: "inner", description: "The brick remembers; the door does not.", strength: 0.7 },
        ]
    },
    storm: {
        label: "Approaching Storm",
        details: [
            { channel: "sight", description: "Sky bruises purple; edges rimmed in quicksilver.", strength: 0.9 },
            { channel: "sound", description: "A far drum cracks the air's shell.", strength: 0.9 },
            { channel: "touch", description: "Humidity lays a palm on the neck.", strength: 0.8 },
            { channel: "scent", description: "Ozone sharpness cuts through summer air.", strength: 0.7 },
            { channel: "taste", description: "Ozone-snap; a coin under the tongue.", strength: 0.7 },
            { channel: "inner", description: "The horizon inhales—waiting.", strength: 0.9 },
        ]
    }
};
