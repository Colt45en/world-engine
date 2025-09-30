// Sensory moment driver with presets and controls
import * as React from "react";
import { SceneMoment, PRESET_MOMENTS, applyPerspective } from "./sensoryTypes";

export function useSensoryMoment(initialPreset = "sunrise"): {
    moment: SceneMoment;
    setPreset: (preset: keyof typeof PRESET_MOMENTS) => void;
    setPerspective: (perspective: "attuned" | "oblivious" | "object") => void;
    setCustomMoment: (moment: SceneMoment) => void;
    availablePresets: string[];
} {
    const [preset, setPreset] = React.useState<keyof typeof PRESET_MOMENTS>(initialPreset);
    const [perspective, setPerspective] = React.useState<"attuned" | "oblivious" | "object">("attuned");
    const [customMoment, setCustomMoment] = React.useState<SceneMoment | null>(null);

    const moment = React.useMemo<SceneMoment>(() => {
        // Use custom moment if provided, otherwise use preset
        const baseMoment = customMoment || {
            id: `${preset}-${perspective}`,
            perspective,
            ...PRESET_MOMENTS[preset],
            timestamp: Date.now()
        };

        return applyPerspective(baseMoment);
    }, [preset, perspective, customMoment]);

    return {
        moment,
        setPreset,
        setPerspective,
        setCustomMoment,
        availablePresets: Object.keys(PRESET_MOMENTS)
    };
}

// Hook for creating procedural moments from text input
export function useTextToSensoryMoment(text: string, updateInterval = 2000): SceneMoment | null {
    const [moment, setMoment] = React.useState<SceneMoment | null>(null);

    React.useEffect(() => {
        if (!text.trim()) {
            setMoment(null);
            return;
        }

        // Simple procedural generation based on keywords
        const generateMomentFromText = (inputText: string): SceneMoment => {
            const words = inputText.toLowerCase().split(/\s+/);
            const details = [];

            // Sight keywords
            if (words.some(w => ['light', 'glow', 'shimmer', 'bright', 'dark', 'shadow', 'color'].includes(w))) {
                details.push({
                    channel: 'sight' as const,
                    description: extractSightDescription(words),
                    strength: 0.7
                });
            }

            // Sound keywords
            if (words.some(w => ['sound', 'whisper', 'echo', 'music', 'hum', 'silence', 'noise'].includes(w))) {
                details.push({
                    channel: 'sound' as const,
                    description: extractSoundDescription(words),
                    strength: 0.6
                });
            }

            // Touch keywords
            if (words.some(w => ['touch', 'feel', 'warm', 'cold', 'soft', 'rough', 'smooth'].includes(w))) {
                details.push({
                    channel: 'touch' as const,
                    description: extractTouchDescription(words),
                    strength: 0.5
                });
            }

            // Inner keywords
            if (words.some(w => ['think', 'memory', 'feel', 'emotion', 'mind', 'heart', 'soul'].includes(w))) {
                details.push({
                    channel: 'inner' as const,
                    description: extractInnerDescription(words),
                    strength: 0.8
                });
            }

            // Default to inner perception if no specific channels detected
            if (details.length === 0) {
                details.push({
                    channel: 'inner' as const,
                    description: text.slice(0, 60) + (text.length > 60 ? '...' : ''),
                    strength: 0.5
                });
            }

            return {
                id: `text-${Date.now()}`,
                label: `Generated from: "${text.slice(0, 30)}..."`,
                perspective: 'attuned',
                details,
                timestamp: Date.now()
            };
        };

        const newMoment = generateMomentFromText(text);
        setMoment(newMoment);

        // Update periodically if text is long enough to warrant evolution
        if (text.length > 50) {
            const interval = setInterval(() => {
                setMoment(generateMomentFromText(text));
            }, updateInterval);

            return () => clearInterval(interval);
        }
    }, [text, updateInterval]);

    return moment;
}

// Helper functions for extracting descriptions from keywords
function extractSightDescription(words: string[]): string {
    const sightWords = words.filter(w =>
        ['light', 'glow', 'shimmer', 'bright', 'dark', 'shadow', 'color', 'shine', 'gleam'].includes(w)
    );

    if (sightWords.includes('light') || sightWords.includes('bright')) {
        return "Light dances across surfaces, casting shifting patterns.";
    }
    if (sightWords.includes('dark') || sightWords.includes('shadow')) {
        return "Shadows pool in corners, deep and mysterious.";
    }
    if (sightWords.includes('glow') || sightWords.includes('shimmer')) {
        return "A subtle luminescence flickers at the edge of vision.";
    }

    return "Visual textures weave through the space.";
}

function extractSoundDescription(words: string[]): string {
    const soundWords = words.filter(w =>
        ['sound', 'whisper', 'echo', 'music', 'hum', 'silence', 'noise'].includes(w)
    );

    if (soundWords.includes('whisper')) {
        return "Soft whispers carry secrets through the air.";
    }
    if (soundWords.includes('echo')) {
        return "Echoes bounce off unseen walls, multiplying.";
    }
    if (soundWords.includes('silence')) {
        return "The silence holds its breath, waiting.";
    }

    return "Ambient sounds weave a subtle tapestry.";
}

function extractTouchDescription(words: string[]): string {
    const touchWords = words.filter(w =>
        ['touch', 'feel', 'warm', 'cold', 'soft', 'rough', 'smooth'].includes(w)
    );

    if (touchWords.includes('warm')) {
        return "Warmth radiates gently against the skin.";
    }
    if (touchWords.includes('cold')) {
        return "Cool air brushes past with gentle precision.";
    }
    if (touchWords.includes('soft')) {
        return "Soft textures invite gentle exploration.";
    }

    return "Tactile sensations ripple through the space.";
}

function extractInnerDescription(words: string[]): string {
    const innerWords = words.filter(w =>
        ['think', 'memory', 'feel', 'emotion', 'mind', 'heart', 'soul', 'sense'].includes(w)
    );

    if (innerWords.includes('memory')) {
        return "Memories surface like bubbles in still water.";
    }
    if (innerWords.includes('emotion')) {
        return "Emotions shift like weather patterns within.";
    }
    if (innerWords.includes('mind')) {
        return "The mind reaches out, sensing connections.";
    }

    return "An inner knowing stirs, wordless but certain.";
}
