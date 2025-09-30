// Real-time word→channel modulation (no IO; deterministic)
import * as React from "react";

export type SensoryDetail = {
    channel: "sight" | "sound" | "touch" | "scent" | "taste" | "inner";
    strength: number; // 0..1
    description?: string;
};

export type SceneMoment = {
    timestamp: number;
    details: SensoryDetail[];
};

type Props = {
    text: string;
    base: SceneMoment;
    onMoment: (m: SceneMoment) => void;
    tps?: number; // tokens per second
};

const LEX: Record<string, Partial<Record<SensoryDetail["channel"], number>>> = {
    // visuals
    sunrise: { sight: 0.9, inner: 0.6 },
    dawn: { sight: 0.8, inner: 0.5 },
    storm: { sight: 0.8, sound: 0.9, touch: 0.7, taste: 0.6, inner: 0.8 },
    lightning: { sight: 1.0, sound: 0.8, inner: 0.7 },
    shadow: { sight: 0.6, inner: 0.6 },
    glow: { sight: 0.9, inner: 0.5 },
    shimmer: { sight: 0.8, inner: 0.4 },
    aurora: { sight: 0.95, inner: 0.8 },
    radiance: { sight: 0.9, inner: 0.7 },

    // sound
    rustle: { sound: 0.7, touch: 0.4 },
    whisper: { sound: 0.6, inner: 0.6 },
    thunder: { sound: 1.0, sight: 0.7 },
    echo: { sound: 0.8, inner: 0.5 },
    hum: { sound: 0.5, inner: 0.4 },
    resonance: { sound: 0.9, inner: 0.8 },

    // touch
    humidity: { touch: 0.8, scent: 0.5, taste: 0.4 },
    warmth: { touch: 0.7, inner: 0.4 },
    cool: { touch: 0.6, inner: 0.3 },
    smooth: { touch: 0.8, sight: 0.3 },
    rough: { touch: 0.9, sight: 0.4 },
    vibration: { touch: 0.8, sound: 0.6 },

    // scent/taste
    ozone: { taste: 0.8, scent: 0.7 },
    candle: { scent: 0.8, sight: 0.4 },
    fresh: { scent: 0.7, inner: 0.5 },
    sweet: { taste: 0.8, scent: 0.6, inner: 0.4 },
    bitter: { taste: 0.9, inner: 0.6 },
    metallic: { taste: 0.7, touch: 0.5 },

    // inner/emotional
    memory: { inner: 0.9, taste: 0.5 },
    time: { inner: 0.8 },
    peace: { inner: 0.7, touch: 0.3 },
    energy: { inner: 0.8, sight: 0.5, sound: 0.4 },
    flow: { inner: 0.6, touch: 0.4, sight: 0.3 },
    focus: { inner: 0.9, sight: 0.6 },
    wonder: { inner: 0.8, sight: 0.7 },
    mystery: { inner: 0.9, sight: 0.5 },

    // VortexLab/neural specific
    neural: { inner: 0.95, sight: 0.6 },
    synapse: { inner: 0.9, sight: 0.5 },
    pattern: { inner: 0.8, sight: 0.7 },
    oscillation: { inner: 0.85, sound: 0.6 },
    coherence: { inner: 0.9, sight: 0.5 },
    quantum: { inner: 0.95, sight: 0.8 },
    hologram: { sight: 0.9, inner: 0.7 },
    projection: { sight: 0.8, inner: 0.6 },
};

export function SensoryTokenStream({ text, base, onMoment, tps = 6 }: Props) {
    const tokens = React.useMemo(
        () => text.split(/(\b|\W)/).map(s => s.trim().toLowerCase()).filter(Boolean),
        [text]
    );

    const idxRef = React.useRef(0);
    const lastRef = React.useRef(0);

    React.useEffect(() => {
        let raf = 0;
        const step = (t: number) => {
            raf = requestAnimationFrame(step);
            const ms = 1000 / tps;
            if (t - lastRef.current < ms) return;
            lastRef.current = t;

            const tok = tokens[idxRef.current] ?? "";
            idxRef.current = (idxRef.current + 1) % Math.max(tokens.length, 1);

            const boosts = LEX[tok];
            const clone = structuredClone(base) as SceneMoment;
            clone.timestamp = Date.now();

            if (boosts) {
                for (const d of clone.details) {
                    const b = (boosts as any)[d.channel];
                    if (b != null) {
                        d.strength = Math.min(1, (d.strength ?? 0.5) * 0.6 + b * 0.7);
                        if (!d.description) d.description = tok;
                    }
                }
            } else {
                // Gradual decay for unrecognized tokens
                for (const d of clone.details) {
                    d.strength = Math.max(0, (d.strength ?? 0.5) * 0.98);
                }
            }

            onMoment(clone);
        };

        raf = requestAnimationFrame(step);
        return () => cancelAnimationFrame(raf);
    }, [tokens, tps, base, onMoment]);

    return null;
}
