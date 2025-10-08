// Advanced Glyph Effects System - CONSOLIDATED FROM SCATTERED UNTITLED FILES
// Features: 12 new glyphs with complex effects, terrain integration, event system hooks

export interface Glyph {
    id: string;
    meaning: string;
    type: 'Worldshift' | 'Temporal' | 'Mechanical' | 'Emotional';
    intensity: number;
    roots: string[];
    tags: string[];
    meta?: any;
    position?: { x: number; y: number };
}

type NewType = (glyph: Glyph, ctx: EffectContext) => void;

export type GlyphEffect = NewType;

export interface EffectContext {
    nexus: {
        eventSystem: {
            spawn: (event: any) => void;
        };
    };
    terrain?: any;
    weather?: any;
    agents?: any[];
}

export class AdvancedGlyphRegistry {
    private glyphs: Map<string, Glyph> = new Map();
    private effects: Map<string, GlyphEffect> = new Map();

    constructor() {
        this.registerCoolGlyphs();
    }

    registerGlyph(glyph: Glyph): void {
        this.glyphs.set(glyph.id, glyph);
    }

    registerEffect(glyphId: string, effect: GlyphEffect): void {
        this.effects.set(glyphId, effect);
    }

    getGlyph(id: string): Glyph | undefined {
        return this.glyphs.get(id);
    }

    applyGlyph(glyphId: string, ctx: EffectContext): void {
        const glyph = this.glyphs.get(glyphId);
        const effect = this.effects.get(glyphId);

        if (!glyph || !effect) {
            console.warn(`Glyph or effect not found: ${glyphId}`);
            return;
        }

        effect(glyph, ctx);
    }

    private registerCoolGlyphs(): void {
        // 12 Advanced Glyphs recovered from scattered untitled files
        const coolGlyphs: Glyph[] = [
            {
                id: "aurora_lattice",
                meaning: "Refract the world into prismatic lanes",
                type: "Worldshift",
                intensity: 0.70,
                roots: ["light", "weave"],
                tags: ["refract", "prism", "sky"]
            },
            {
                id: "keystone_memory",
                meaning: "Pin a moment; ease time-shear",
                type: "Temporal",
                intensity: 0.65,
                roots: ["memory", "lock"],
                tags: ["anchor", "recall", "save"]
            },
            {
                id: "echo_weaver",
                meaning: "Knit echoes; chain reactions",
                type: "Mechanical",
                intensity: 0.60,
                roots: ["echo", "knit"],
                tags: ["chain", "pulse", "link"]
            },
            {
                id: "fathom_drift",
                meaning: "Sedate turbulence; deepen calm",
                type: "Emotional",
                intensity: 0.55,
                roots: ["ocean", "still"],
                tags: ["depth", "calm", "abyss"]
            },
            {
                id: "solaris_anchor",
                meaning: "Fix noon; clear fog of time",
                type: "Temporal",
                intensity: 0.75,
                roots: ["solar", "bind"],
                tags: ["sun", "noon", "fix"]
            },
            {
                id: "umbra_veil",
                meaning: "Muffle agitation; hush paths",
                type: "Worldshift",
                intensity: 0.60,
                roots: ["dark", "hush"],
                tags: ["shadow", "hush", "veil"]
            },
            {
                id: "kintsugi_field",
                meaning: "Mend broken states with golden seams",
                type: "Emotional",
                intensity: 0.85,
                roots: ["break", "bond"],
                tags: ["mend", "gold", "heal"]
            },
            {
                id: "chronicle_bloom",
                meaning: "Seed archival sprouts that recall",
                type: "Temporal",
                intensity: 0.50,
                roots: ["record", "grow"],
                tags: ["seed", "archive", "sprout"]
            },
            {
                id: "hearthbind",
                meaning: "Gather nearby agents; raise warmth",
                type: "Emotional",
                intensity: 0.65,
                roots: ["kin", "hearth"],
                tags: ["home", "warm", "gather"]
            },
            {
                id: "pale_comet",
                meaning: "Emit streaked surges; leave trails",
                type: "Mechanical",
                intensity: 0.70,
                roots: ["arc", "streak"],
                tags: ["trail", "spark", "burst"]
            },
            {
                id: "tessellate_choir",
                meaning: "Quantize space; lock rhythm in tiles",
                type: "Mechanical",
                intensity: 0.60,
                roots: ["tile", "chord"],
                tags: ["grid", "resonance", "sync"]
            },
            {
                id: "eventide_gate",
                meaning: "Open dusk thresholds between layers",
                type: "Worldshift",
                intensity: 0.80,
                roots: ["limen", "fade"],
                tags: ["threshold", "dusk", "cross"]
            }
        ];

        // Register all glyphs
        coolGlyphs.forEach(glyph => this.registerGlyph(glyph));

        // Register corresponding effects
        this.registerEffects();
    }

    private registerEffects(): void {
        // Aurora Lattice Effect - Prismatic world refraction
        this.registerEffect("aurora_lattice", (glyph: Glyph, ctx: EffectContext) => {
            glyph.meta = glyph.meta || {};
            glyph.meta.refractive = true;
            glyph.meta.spectrumBias = 0.25 + 0.5 * glyph.intensity;

            ctx.nexus.eventSystem.spawn({
                type: "storm",
                origin: glyph.position || { x: 0, y: 0 },
                radius: 9 + 6 * glyph.intensity,
                duration: 28,
                shape: 'ring',
                ringInner: 3,
                falloff: 'ring',
                effects: ['storm', 'moisture', 'prismatic_light']
            });
        });

        // Keystone Memory Effect - Time anchor point
        this.registerEffect("keystone_memory", (glyph: Glyph, ctx: EffectContext) => {
            glyph.meta = glyph.meta || {};
            glyph.meta.temporal_anchor = {
                timestamp: Date.now(),
                stability: glyph.intensity
            };

            ctx.nexus.eventSystem.spawn({
                type: "temporal_anchor",
                origin: glyph.position || { x: 0, y: 0 },
                radius: 5 + 3 * glyph.intensity,
                duration: 60,
                effects: ['time_stabilization', 'memory_preservation']
            });
        });

        // Echo Weaver Effect - Chain reaction system
        this.registerEffect("echo_weaver", (glyph: Glyph, ctx: EffectContext) => {
            glyph.meta = glyph.meta || {};
            glyph.meta.echo_chain = [];
            glyph.meta.resonance_frequency = 0.1 + 0.4 * glyph.intensity;

            // Create expanding echo rings
            for (let i = 1; i <= 3; i++) {
                setTimeout(() => {
                    ctx.nexus.eventSystem.spawn({
                        type: "echo_pulse",
                        origin: glyph.position || { x: 0, y: 0 },
                        radius: i * (4 + 2 * glyph.intensity),
                        duration: 15,
                        delay: i * 200,
                        effects: ['echo_amplification', 'harmonic_resonance']
                    });
                }, i * 300);
            }
        });

        // Fathom Drift Effect - Deep calm induction
        this.registerEffect("fathom_drift", (glyph: Glyph, ctx: EffectContext) => {
            glyph.meta = glyph.meta || {};
            glyph.meta.calm_depth = glyph.intensity;
            glyph.meta.turbulence_reduction = 0.3 + 0.4 * glyph.intensity;

            ctx.nexus.eventSystem.spawn({
                type: "calm_field",
                origin: glyph.position || { x: 0, y: 0 },
                radius: 12 + 8 * glyph.intensity,
                duration: 45,
                shape: 'gradient',
                effects: ['turbulence_damping', 'emotional_soothing', 'depth_induction']
            });
        });

        // Solaris Anchor Effect - Solar time fixation
        this.registerEffect("solaris_anchor", (glyph: Glyph, ctx: EffectContext) => {
            glyph.meta = glyph.meta || {};
            glyph.meta.solar_fixed = true;
            glyph.meta.time_clarity = glyph.intensity;

            ctx.nexus.eventSystem.spawn({
                type: "solar_beacon",
                origin: glyph.position || { x: 0, y: 0 },
                radius: 6 + 4 * glyph.intensity,
                duration: 40,
                shape: 'radial',
                effects: ['fog_clearing', 'time_crystallization', 'solar_illumination']
            });
        });

        // Umbra Veil Effect - Shadow muffling
        this.registerEffect("umbra_veil", (glyph: Glyph, ctx: EffectContext) => {
            glyph.meta = glyph.meta || {};
            glyph.meta.shadow_density = 0.4 + 0.3 * glyph.intensity;
            glyph.meta.muffling_strength = glyph.intensity;

            ctx.nexus.eventSystem.spawn({
                type: "shadow_field",
                origin: glyph.position || { x: 0, y: 0 },
                radius: 8 + 5 * glyph.intensity,
                duration: 35,
                shape: 'soft_gradient',
                effects: ['agitation_muffling', 'shadow_concealment', 'noise_dampening']
            });
        });

        // Kintsugi Field Effect - Golden repair of broken states
        this.registerEffect("kintsugi_field", (glyph: Glyph, ctx: EffectContext) => {
            glyph.meta = glyph.meta || {};
            glyph.meta.healing_potency = glyph.intensity;
            glyph.meta.golden_seams = true;

            ctx.nexus.eventSystem.spawn({
                type: "healing_field",
                origin: glyph.position || { x: 0, y: 0 },
                radius: 10 + 6 * glyph.intensity,
                duration: 50,
                shape: 'fractal',
                effects: ['state_mending', 'golden_reinforcement', 'beauty_from_breaks']
            });
        });

        // Chronicle Bloom Effect - Archival memory sprouting
        this.registerEffect("chronicle_bloom", (glyph: Glyph, ctx: EffectContext) => {
            glyph.meta = glyph.meta || {};
            glyph.meta.memory_seeds = Math.floor(3 + 5 * glyph.intensity);
            glyph.meta.recall_strength = glyph.intensity;

            // Spawn multiple memory sprouts
            for (let i = 0; i < glyph.meta.memory_seeds; i++) {
                const angle = (i / glyph.meta.memory_seeds) * 2 * Math.PI;
                const distance = 2 + 3 * glyph.intensity;
                const sproutPos = {
                    x: (glyph.position?.x || 0) + Math.cos(angle) * distance,
                    y: (glyph.position?.y || 0) + Math.sin(angle) * distance
                };

                ctx.nexus.eventSystem.spawn({
                    type: "memory_sprout",
                    origin: sproutPos,
                    radius: 2 + glyph.intensity,
                    duration: 30,
                    effects: ['memory_archival', 'history_recording']
                });
            }
        });

        // Hearthbind Effect - Agent gathering and warmth
        this.registerEffect("hearthbind", (glyph: Glyph, ctx: EffectContext) => {
            glyph.meta = glyph.meta || {};
            glyph.meta.warmth_radius = 7 + 5 * glyph.intensity;
            glyph.meta.gathering_force = glyph.intensity;

            ctx.nexus.eventSystem.spawn({
                type: "hearth_field",
                origin: glyph.position || { x: 0, y: 0 },
                radius: glyph.meta.warmth_radius,
                duration: 60,
                shape: 'warm_glow',
                effects: ['agent_attraction', 'warmth_generation', 'kinship_bonding']
            });
        });

        // Pale Comet Effect - Streaked surge with trails
        this.registerEffect("pale_comet", (glyph: Glyph, ctx: EffectContext) => {
            glyph.meta = glyph.meta || {};
            glyph.meta.streak_length = 8 + 10 * glyph.intensity;
            glyph.meta.trail_persistence = 20 + 15 * glyph.intensity;

            // Create comet trajectory
            const trajectory = [];
            for (let i = 0; i <= glyph.meta.streak_length; i++) {
                trajectory.push({
                    x: (glyph.position?.x || 0) + i,
                    y: (glyph.position?.y || 0) + Math.sin(i * 0.3) * 2
                });
            }

            ctx.nexus.eventSystem.spawn({
                type: "comet_streak",
                trajectory: trajectory,
                duration: glyph.meta.trail_persistence,
                effects: ['energy_trail', 'momentum_surge', 'spark_cascade']
            });
        });

        // Tessellate Choir Effect - Space quantization and rhythm locking
        this.registerEffect("tessellate_choir", (glyph: Glyph, ctx: EffectContext) => {
            glyph.meta = glyph.meta || {};
            glyph.meta.grid_size = Math.floor(4 + 6 * glyph.intensity);
            glyph.meta.rhythm_lock = true;

            // Create tessellated grid
            const gridOrigin = glyph.position || { x: 0, y: 0 };
            for (let dx = -glyph.meta.grid_size; dx <= glyph.meta.grid_size; dx++) {
                for (let dy = -glyph.meta.grid_size; dy <= glyph.meta.grid_size; dy++) {
                    ctx.nexus.eventSystem.spawn({
                        type: "rhythm_tile",
                        origin: { x: gridOrigin.x + dx * 2, y: gridOrigin.y + dy * 2 },
                        radius: 1,
                        duration: 30,
                        effects: ['space_quantization', 'rhythm_synchronization']
                    });
                }
            }
        });

        // Eventide Gate Effect - Dusk threshold opening
        this.registerEffect("eventide_gate", (glyph: Glyph, ctx: EffectContext) => {
            glyph.meta = glyph.meta || {};
            glyph.meta.threshold_stability = glyph.intensity;
            glyph.meta.layer_connection = true;

            ctx.nexus.eventSystem.spawn({
                type: "dimensional_threshold",
                origin: glyph.position || { x: 0, y: 0 },
                radius: 5 + 4 * glyph.intensity,
                duration: 45,
                shape: 'portal',
                effects: ['layer_bridging', 'dusk_transition', 'threshold_opening']
            });
        });
    }

    // Utility methods for glyph management
    getAllGlyphs(): Glyph[] {
        return Array.from(this.glyphs.values());
    }

    getGlyphsByType(type: Glyph['type']): Glyph[] {
        return this.getAllGlyphs().filter(g => g.type === type);
    }

    getGlyphsByTag(tag: string): Glyph[] {
        return this.getAllGlyphs().filter(g => g.tags.includes(tag));
    }

    searchGlyphs(query: string): Glyph[] {
        const lowerQuery = query.toLowerCase();
        return this.getAllGlyphs().filter(g =>
            g.id.toLowerCase().includes(lowerQuery) ||
            g.meaning.toLowerCase().includes(lowerQuery) ||
            g.tags.some(tag => tag.toLowerCase().includes(lowerQuery))
        );
    }

    // Advanced effect combinations
    applyGlyphCombo(glyphIds: string[], ctx: EffectContext): void {
        console.log(`🌟 Applying glyph combination: ${glyphIds.join(' + ')}`);

        // Apply glyphs in sequence with slight delays for dramatic effect
        glyphIds.forEach((glyphId, index) => {
            setTimeout(() => {
                this.applyGlyph(glyphId, ctx);
            }, index * 100);
        });

        // Create combination bonus effects
        if (glyphIds.length >= 2) {
            this.applyCombinationBonus(glyphIds, ctx);
        }
    }

    private applyCombinationBonus(glyphIds: string[], ctx: EffectContext): void {
        const glyphs = glyphIds.map(id => this.getGlyph(id)).filter(Boolean) as Glyph[];

        // Calculate combination intensity
        const avgIntensity = glyphs.reduce((sum, g) => sum + g.intensity, 0) / glyphs.length;
        const combinationBonus = Math.min(0.3, glyphs.length * 0.1);
        const totalIntensity = avgIntensity + combinationBonus;

        // Type-based combination effects
        const types = new Set(glyphs.map(g => g.type));

        if (types.has('Worldshift') && types.has('Temporal')) {
            // Reality-Time combo
            ctx.nexus.eventSystem.spawn({
                type: "spacetime_warp",
                origin: { x: 0, y: 0 },
                radius: 15 + 10 * totalIntensity,
                duration: 40,
                effects: ['reality_flux', 'temporal_distortion']
            });
        }

        if (types.has('Mechanical') && types.has('Emotional')) {
            // Tech-Emotion combo
            ctx.nexus.eventSystem.spawn({
                type: "empathic_resonance",
                origin: { x: 0, y: 0 },
                radius: 12 + 8 * totalIntensity,
                duration: 35,
                effects: ['mechanical_harmony', 'emotional_amplification']
            });
        }

        if (glyphs.length >= 3) {
            // Triple+ combination creates harmony field
            ctx.nexus.eventSystem.spawn({
                type: "harmonic_convergence",
                origin: { x: 0, y: 0 },
                radius: 20 + 15 * totalIntensity,
                duration: 60,
                effects: ['multi_glyph_resonance', 'power_amplification', 'chaos_stabilization']
            });
        }
    }
}

// Export singleton instance
export const advancedGlyphRegistry = new AdvancedGlyphRegistry();

// Integration hooks for the existing world engine
export function integrateWithWorldEngine(worldEngine: any): void {
    console.log("🎭 Integrating Advanced Glyph System with World Engine");

    // Hook into the world engine's glyph system if it exists
    if (worldEngine.glyphSystem) {
        worldEngine.glyphSystem.advancedRegistry = advancedGlyphRegistry;
    }

    // Add glyph application method to world engine
    worldEngine.applyAdvancedGlyph = (glyphId: string, position?: { x: number, y: number }) => {
        const glyph = advancedGlyphRegistry.getGlyph(glyphId);
        if (glyph && position) {
            glyph.position = position;
        }

        const ctx: EffectContext = {
            nexus: worldEngine.nexus || { eventSystem: { spawn: console.log } },
            terrain: worldEngine.terrain,
            weather: worldEngine.weather,
            agents: worldEngine.agents
        };

        advancedGlyphRegistry.applyGlyph(glyphId, ctx);
    };

    // Add glyph combo method
    worldEngine.applyGlyphCombo = (glyphIds: string[], centerPosition?: { x: number, y: number }) => {
        const ctx: EffectContext = {
            nexus: worldEngine.nexus || { eventSystem: { spawn: console.log } },
            terrain: worldEngine.terrain,
            weather: worldEngine.weather,
            agents: worldEngine.agents
        };

        // Set positions in a circle around center
        if (centerPosition) {
            glyphIds.forEach((glyphId, index) => {
                const glyph = advancedGlyphRegistry.getGlyph(glyphId);
                if (glyph) {
                    const angle = (index / glyphIds.length) * 2 * Math.PI;
                    const radius = 3;
                    glyph.position = {
                        x: centerPosition.x + Math.cos(angle) * radius,
                        y: centerPosition.y + Math.sin(angle) * radius
                    };
                }
            });
        }

        advancedGlyphRegistry.applyGlyphCombo(glyphIds, ctx);
    };

    console.log("✅ Advanced Glyph System integration complete!");
}
import json
import random
    from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from enum import Enum

# ═══════════════════════════════════════════════════════════════════
# UNIVERSAL GLYPH SYSTEM(Works Anywhere)
# ═══════════════════════════════════════════════════════════════════

class GlyphType(Enum):
EMOTIONAL = "Emotional"      # Affects feelings, moods, relationships
MECHANICAL = "Mechanical"    # Triggers events, causes actions
TEMPORAL = "Temporal"        # Time effects, memories, futures
WORLDSHIFT = "Worldshift"    # Reality changes, dimension shifts
ELEMENTAL = "Elemental"      # Fire, water, earth, air effects
SOCIAL = "Social"           # NPC interactions, faction changes
ENVIRONMENTAL = "Environmental"  # Weather, terrain, atmosphere

@dataclass
class UniversalGlyph:
"""Glyphs that work in any game context"""
id: str
icon: str
name: str
type: GlyphType
meaning: str
intensity: float  # 0.0 to 1.0
tags: List[str]
effects: Dict[str, str]  # context -> effect mapping
requirements: List[str] = field(default_factory = list)
cooldown: int = 0  # turns before reuse

class GlyphLibrary:
"""Universal glyph collection that works everywhere"""

UNIVERSAL_GLYPHS = {
        # EMOTIONAL GLYPHS
        "soul-thread": UniversalGlyph(
    id = "soul-thread",
    icon = "🧵",
    name = "Soul Thread",
    type = GlyphType.EMOTIONAL,
    meaning = "Links character's essence to scene memory",
    intensity = 0.7,
    tags = ["bind", "memory", "character"],
    effects = {
        "combat": "Character fights with memories of fallen allies",
        "exploration": "Hidden paths reveal based on past experiences",
        "social": "NPCs remember past interactions more vividly",
        "kingdom": "Population loyalty increases through shared history"
    }
),

    "heart-echo": UniversalGlyph(
        id = "heart-echo",
        icon = "💗",
        name = "Heart Echo",
        type = GlyphType.EMOTIONAL,
        meaning = "Amplifies emotional resonance in the area",
        intensity = 0.8,
        tags = ["amplify", "emotion", "aura"],
        effects = {
            "combat": "Allies gain courage, enemies feel doubt",
            "exploration": "Emotional landmarks become visible",
            "social": "NPC trust/hostility intensifies",
            "kingdom": "Population mood spreads rapidly"
        }
    ),

        # MECHANICAL GLYPHS
        "echo-pulse": UniversalGlyph(
        id = "echo-pulse",
        icon = "📡",
        name = "Echo Pulse",
        type = GlyphType.MECHANICAL,
        meaning = "Triggers chain reactions across linked systems",
        intensity = 0.6,
        tags = ["trigger", "chain", "network"],
        effects = {
            "combat": "One attack triggers multiple allied strikes",
            "exploration": "Activating one mechanism opens multiple doors",
            "social": "One conversation spreads news to entire settlement",
            "kingdom": "One policy change cascades across all regions"
        }
    ),

    "gear-bind": UniversalGlyph(
        id = "gear-bind",
        icon = "⚙️",
        name = "Gear Bind",
        type = GlyphType.MECHANICAL,
        meaning = "Synchronizes mechanical systems temporarily",
        intensity = 0.7,
        tags = ["sync", "mechanical", "temporary"],
        effects = {
            "combat": "All weapons/armor work in perfect harmony",
            "exploration": "All puzzles in area solve simultaneously",
            "social": "All NPCs coordinate their responses",
            "kingdom": "All administrative systems align"
        }
    ),

        # TEMPORAL GLYPHS
        "golden-return": UniversalGlyph(
        id = "golden-return",
        icon = "🌅",
        name = "Golden Return",
        type = GlyphType.TEMPORAL,
        meaning = "Restores a prior state with emotional weight",
        intensity = 0.8,
        tags = ["restore", "memory", "golden"],
        effects = {
            "combat": "Restore HP/MP to state before last major battle",
            "exploration": "Return to previous safe location with knowledge intact",
            "social": "Reset relationship to earlier positive state",
            "kingdom": "Restore kingdom to previous prosperity level"
        }
    ),

    "time-splinter": UniversalGlyph(
        id = "time-splinter",
        icon = "⏱️",
        name = "Time Splinter",
        type = GlyphType.TEMPORAL,
        meaning = "Creates brief temporal fractures showing possibilities",
        intensity = 0.9,
        tags = ["fracture", "possibility", "glimpse"],
        effects = {
            "combat": "See outcome of next 3 possible attacks",
            "exploration": "Preview consequences of different path choices",
            "social": "Glimpse how conversation could unfold",
            "kingdom": "Preview results of different policies"
        }
    ),

        # WORLDSHIFT GLYPHS
        "fracture-point": UniversalGlyph(
        id = "fracture-point",
        icon = "💥",
        name = "Fracture Point",
        type = GlyphType.WORLDSHIFT,
        meaning = "Fractures reality into multiple possibility branches",
        intensity = 0.95,
        tags = ["fracture", "reality", "branches"],
        effects = {
            "combat": "Battle splits into multiple simultaneous outcomes",
            "exploration": "Path splits into parallel dimensional routes",
            "social": "Conversation creates alternate timeline branches",
            "kingdom": "Kingdom splits into parallel governance structures"
        },
        cooldown = 5
    ),

    "reality-anchor": UniversalGlyph(
        id = "reality-anchor",
        icon = "⚓",
        name = "Reality Anchor",
        type = GlyphType.WORLDSHIFT,
        meaning = "Locks current reality state against changes",
        intensity = 0.8,
        tags = ["anchor", "stability", "lock"],
        effects = {
            "combat": "Current battlefield conditions cannot change",
            "exploration": "Current area layout becomes fixed/safe",
            "social": "Current relationship states become locked",
            "kingdom": "Current kingdom status protected from shifts"
        }
    ),

        # ELEMENTAL GLYPHS
        "flame-heart": UniversalGlyph(
        id = "flame-heart",
        icon = "🔥",
        name = "Flame Heart",
        type = GlyphType.ELEMENTAL,
        meaning = "Ignites passionate fire in core of scene",
        intensity = 0.8,
        tags = ["fire", "passion", "core"],
        effects = {
            "combat": "All attacks gain fire damage and passionate intensity",
            "exploration": "Hidden fire-based secrets become visible",
            "social": "NPCs become more emotionally expressive",
            "kingdom": "Population becomes more passionate about causes"
        }
    ),

    "water-flow": UniversalGlyph(
        id = "water-flow",
        icon = "🌊",
        name = "Water Flow",
        type = GlyphType.ELEMENTAL,
        meaning = "Creates adaptive, flowing responses to situations",
        intensity = 0.6,
        tags = ["water", "adaptive", "flow"],
        effects = {
            "combat": "Tactics become fluid, adapting to enemy moves",
            "exploration": "Paths reshape to guide toward goals",
            "social": "Conversations flow naturally toward resolution",
            "kingdom": "Policies adapt smoothly to changing conditions"
        }
    ),

        # SOCIAL GLYPHS
        "bond-weave": UniversalGlyph(
        id = "bond-weave",
        icon = "🤝",
        name = "Bond Weave",
        type = GlyphType.SOCIAL,
        meaning = "Strengthens connections between all characters",
        intensity = 0.7,
        tags = ["bond", "connection", "strengthen"],
        effects = {
            "combat": "All allies coordinate perfectly",
            "exploration": "Team moves as one, sharing discoveries",
            "social": "All NPCs feel connected to party",
            "kingdom": "All factions work toward common goals"
        }
    ),

    "voice-carry": UniversalGlyph(
        id = "voice-carry",
        icon = "📢",
        name = "Voice Carry",
        type = GlyphType.SOCIAL,
        meaning = "Amplifies communication across any distance",
        intensity = 0.5,
        tags = ["voice", "communication", "distance"],
        effects = {
            "combat": "Commands reach all allies instantly",
            "exploration": "Can communicate with distant NPCs/creatures",
            "social": "Words carry emotional weight to all listeners",
            "kingdom": "Royal decrees reach all citizens simultaneously"
        }
    ),

        # ENVIRONMENTAL GLYPHS
        "storm-call": UniversalGlyph(
        id = "storm-call",
        icon = "⛈️",
        name = "Storm Call",
        type = GlyphType.ENVIRONMENTAL,
        meaning = "Summons dramatic weather matching scene intensity",
        intensity = 0.8,
        tags = ["weather", "drama", "intensity"],
        effects = {
            "combat": "Lightning strikes emphasize critical moments",
            "exploration": "Weather reveals hidden paths/dangers",
            "social": "Atmosphere reflects conversation mood",
            "kingdom": "Weather reflects kingdom's emotional state"
        }
    ),

    "sanctuary-bloom": UniversalGlyph(
        id = "sanctuary-bloom",
        icon = "🌸",
        name = "Sanctuary Bloom",
        type = GlyphType.ENVIRONMENTAL,
        meaning = "Creates peaceful, healing environment",
        intensity = 0.6,
        tags = ["peace", "healing", "sanctuary"],
        effects = {
            "combat": "Creates temporary cease-fire zone",
            "exploration": "Area becomes safe rest point",
            "social": "All conversations become more peaceful",
            "kingdom": "Reduces conflict, increases happiness"
        }
    )
}

@classmethod
    def get_glyph(cls, glyph_id: str) -> Optional[UniversalGlyph]:
return cls.UNIVERSAL_GLYPHS.get(glyph_id)

@classmethod
    def get_glyphs_by_type(cls, glyph_type: GlyphType) -> List[UniversalGlyph]:
return [g for g in cls.UNIVERSAL_GLYPHS.values() if g.type == glyph_type]

@classmethod
    def get_random_glyph(cls, context: str = None) -> UniversalGlyph:
glyphs = list(cls.UNIVERSAL_GLYPHS.values())
if context:
            # Prefer glyphs that have effects for this context
            context_glyphs = [g for g in glyphs if context in g.effects]
if context_glyphs:
    return random.choice(context_glyphs)
return random.choice(glyphs)

# ═══════════════════════════════════════════════════════════════════
# SCENE GENERATION FRAMEWORK
# ═══════════════════════════════════════════════════════════════════

class SceneType(Enum):
COMBAT = "combat"
EXPLORATION = "exploration"
SOCIAL = "social"
KINGDOM = "kingdom"
MYSTERY = "mystery"
RITUAL = "ritual"
TRAVEL = "travel"
DUNGEON = "dungeon"

@dataclass
class ScenePrompt:
"""Template for generating rich game scenes"""
name: str
type: SceneType
base_prompt: str
glyph_slots: int  # How many glyphs can be active
context_variables: List[str]  # Variables to fill in
    intensity_modifiers: Dict[str, str]  # How intensity affects the scene

class SceneGenerator:
"""Generates rich game world scenes with integrated glyph system"""

SCENE_PROMPTS = {
    "epic-battle": ScenePrompt(
        name = "Epic Battle",
        type = SceneType.COMBAT,
        base_prompt = """
            SCENE: Epic Battle - { location }

            🎯 CORE SITUATION:
        The { enemy_force } has { threat_description } near { location }.
        Your { party_composition } must { primary_objective } while { complication }.

            ⚔️ COMBAT DYNAMICS:
            • Primary Threat: { enemy_force } with { enemy_special_ability }
            • Environmental Factor: { environment_hazard }
            • Victory Condition: { victory_condition }
            • Failure Consequence: { failure_consequence }

            🎭 EMOTIONAL STAKES:
        { emotional_hook } - This battle matters because { personal_stakes }.
            The { ally_or_npc } is { emotional_state } and needs { emotional_need }.

            💫 ACTIVE GLYPHS: { active_glyphs }
            Each glyph affects: { glyph_effects }

            🎲 GM PROMPTS:
            • How does { active_glyph_0 } change the battlefield ?
            • What memory / emotion does { active_glyph_1 } invoke ?
            • When is the perfect moment for { active_glyph_2 } to activate ?
    """,
glyph_slots = 3,
    context_variables = [
        "location", "enemy_force", "threat_description", "party_composition",
        "primary_objective", "complication", "enemy_special_ability",
        "environment_hazard", "victory_condition", "failure_consequence",
        "emotional_hook", "personal_stakes", "ally_or_npc", "emotional_state", "emotional_need"
    ],
    intensity_modifiers = {
        "low": "Skirmish with minor stakes",
        "medium": "Significant battle with important consequences",
        "high": "Epic confrontation that changes everything",
        "extreme": "Reality-altering conflict with cosmic implications"
    }
        ),

"mystery-investigation": ScenePrompt(
    name = "Mystery Investigation",
    type = SceneType.MYSTERY,
    base_prompt = """
            SCENE: Mystery Investigation - { location }

            🕵️ CENTRAL MYSTERY:
    Something { mysterious_event } has occurred at { location }.
    The { affected_party } discovered { initial_clue } and fears { feared_consequence }.

            🔍 INVESTIGATION LAYERS:
            • Surface Clue: { surface_evidence }(obvious but misleading)
            • Hidden Truth: { hidden_evidence }(requires { investigation_method })
            • Deep Secret: { core_secret }(changes everything when revealed)

            👥 KEY FIGURES:
            • { witness_npc }: Knows { witness_info } but { witness_obstacle }
            • { suspect_npc }: Appears { suspect_appearance } but actually { suspect_truth }
            • { authority_npc }: { authority_attitude } and has { authority_resource }

            💫 ACTIVE GLYPHS: { active_glyphs }
            Glyph effects on investigation: { glyph_effects }

            🎲 GM PROMPTS:
            • How does { active_glyph_0 } reveal hidden connections ?
            • What emotional truth does { active_glyph_1 } expose ?
            • When does { active_glyph_2 } shift the entire mystery ?
    """,
            glyph_slots = 3,
    context_variables = [
        "location", "mysterious_event", "affected_party", "initial_clue",
        "feared_consequence", "surface_evidence", "hidden_evidence",
        "investigation_method", "core_secret", "witness_npc", "witness_info",
        "witness_obstacle", "suspect_npc", "suspect_appearance", "suspect_truth",
        "authority_npc", "authority_attitude", "authority_resource"
    ],
    intensity_modifiers = {
        "low": "Simple misunderstanding with easy resolution",
        "medium": "Complex mystery with multiple suspects",
        "high": "Conspiracy affecting multiple factions",
        "extreme": "Reality-bending mystery that questions existence itself"
    }
),

    "kingdom-court": ScenePrompt(
        name = "Kingdom Court",
        type = SceneType.KINGDOM,
        base_prompt = """
            SCENE: Kingdom Court Session - { throne_room }

            👑 ROYAL SITUATION:
        The { ruler_title } { ruler_name } must decide on { major_decision }.
        The { primary_faction } demands { faction_demand } while { opposing_faction } insists { opposing_demand }.

            ⚖️ POLITICAL DYNAMICS:
            • Primary Issue: { central_conflict }
            • Faction Tensions: { faction_a } vs { faction_b } over { dispute_focus }
            • Royal Dilemma: Choosing will { choice_consequence_a } or { choice_consequence_b }
            • Hidden Agenda: { secret_manipulator } secretly wants { hidden_goal }

            🎭 PERSONAL STAKES:
{ ruler_name } personally feels { ruler_emotion } because { personal_connection }.
            The decision affects { loved_one } who { personal_stake }.

            💫 ACTIVE GLYPHS: { active_glyphs }
            Court effects: { glyph_effects }

            🎲 GM PROMPTS:
            • How does { active_glyph_0 } change the political landscape ?
            • What hidden emotion does { active_glyph_1 } reveal in the court ?
            • When does { active_glyph_2 } shift the entire kingdom's fate?
""",
glyph_slots = 2,
    context_variables = [
        "throne_room", "ruler_title", "ruler_name", "major_decision",
        "primary_faction", "faction_demand", "opposing_faction", "opposing_demand",
        "central_conflict", "faction_a", "faction_b", "dispute_focus",
        "choice_consequence_a", "choice_consequence_b", "secret_manipulator",
        "hidden_goal", "ruler_emotion", "personal_connection", "loved_one", "personal_stake"
    ],
    intensity_modifiers = {
        "low": "Minor administrative decision",
        "medium": "Important policy affecting multiple regions",
        "high": "Critical decision determining kingdom's future",
        "extreme": "World-shaping choice with cosmic consequences"
    }
        ),

"exploration-discovery": ScenePrompt(
    name = "Exploration Discovery",
    type = SceneType.EXPLORATION,
    base_prompt = """
            SCENE: Exploration Discovery - { discovery_location }

            🗺️ THE DISCOVERY:
    While exploring { approach_route }, your party discovers { amazing_find }.
    The { discovery_scale } reveals { primary_feature } and suggests { historical_significance }.

            🔍 EXPLORATION ELEMENTS:
            • Immediate Wonder: { visual_spectacle }
            • Hidden Danger: { concealed_threat } that { threat_trigger }
            • Secret Reward: { hidden_treasure } accessible through { access_method }
            • Mystery Element: { unexplained_phenomenon } that defies { logical_expectation }

            🏛️ HISTORICAL ECHOES:
    This place remembers { ancient_event }.The { ancient_people } once { past_activity }.
            Signs suggest { civilization_fate } and warn of { ancient_warning }.

            💫 ACTIVE GLYPHS: { active_glyphs }
            Discovery effects: { glyph_effects }

            🎲 GM PROMPTS:
            • How does { active_glyph_0 } reveal the location's true nature?
            • What memory or vision does { active_glyph_1 } unlock here ?
            • When does { active_glyph_2 } connect this discovery to larger mysteries ?
    """,
            glyph_slots = 3,
    context_variables = [
        "discovery_location", "approach_route", "amazing_find", "discovery_scale",
        "primary_feature", "historical_significance", "visual_spectacle",
        "concealed_threat", "threat_trigger", "hidden_treasure", "access_method",
        "unexplained_phenomenon", "logical_expectation", "ancient_event",
        "ancient_people", "past_activity", "civilization_fate", "ancient_warning"
    ],
    intensity_modifiers = {
        "low": "Interesting local landmark with minor secrets",
        "medium": "Significant archaeological find with regional importance",
        "high": "World-changing discovery that rewrites history",
        "extreme": "Reality-altering location that connects multiple dimensions"
    }
),

    "social-gathering": ScenePrompt(
        name = "Social Gathering",
        type = SceneType.SOCIAL,
        base_prompt = """
            SCENE: Social Gathering - { event_location }

            🎉 THE GATHERING:
        The { event_type } at { event_location } brings together { attendee_description }.
            The official purpose is { stated_purpose }, but the real drama involves { hidden_agenda }.

            👥 SOCIAL DYNAMICS:
            • Central Figure: { important_npc } who { npc_role } and wants { npc_goal }
            • Social Tension: { tension_source } between { group_a } and { group_b }
            • Opportunity: { social_opportunity } is available to those who { opportunity_requirement }
            • Crisis Point: { potential_disaster } could happen if { trigger_condition }

            💭 EMOTIONAL UNDERCURRENTS:
            The atmosphere feels { general_mood } because { mood_reason }.
{ emotional_focus_npc } is secretly { hidden_emotion } about { emotional_cause }.

            💫 ACTIVE GLYPHS: { active_glyphs }
            Social effects: { glyph_effects }

            🎲 GM PROMPTS:
            • How does { active_glyph_0 } change the social dynamics ?
            • What hidden feeling does { active_glyph_1 } bring to the surface ?
            • When does { active_glyph_2 } transform the entire gathering ?
    """,
glyph_slots = 2,
    context_variables = [
        "event_location", "event_type", "attendee_description", "stated_purpose",
        "hidden_agenda", "important_npc", "npc_role", "npc_goal", "tension_source",
        "group_a", "group_b", "social_opportunity", "opportunity_requirement",
        "potential_disaster", "trigger_condition", "general_mood", "mood_reason",
        "emotional_focus_npc", "hidden_emotion", "emotional_cause"
    ],
    intensity_modifiers = {
        "low": "Casual local gathering with minor social stakes",
        "medium": "Important social event with significant consequences",
        "high": "High-stakes gathering that could change faction relations",
        "extreme": "Reality-defining social moment that reshapes the world"
    }
        )
    }

@classmethod
    def generate_scene(cls, scene_type: str, intensity: str = "medium", custom_variables: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
"""Generate a complete scene with glyphs"""
prompt = cls.SCENE_PROMPTS.get(scene_type)
if not prompt:
    return { "error": f"Scene type '{scene_type}' not found"}

        # Select appropriate glyphs for this scene
        scene_context = prompt.type.value
        active_glyphs = []
glyph_effects = []

for i in range(prompt.glyph_slots):
    glyph = GlyphLibrary.get_random_glyph(scene_context)
active_glyphs.append(f"{glyph.icon} {glyph.name}")
if scene_context in glyph.effects:
    glyph_effects.append(f"{glyph.name}: {glyph.effects[scene_context]}")
else:
glyph_effects.append(f"{glyph.name}: {glyph.meaning}")

        # Create scene variables with defaults
        scene_vars = cls._generate_default_variables(prompt, custom_variables or {})

        # Add glyph data in simple format
scene_vars["active_glyphs"] = ", ".join(active_glyphs)
scene_vars["glyph_effects"] = "\n            ".join([f"• {effect}" for effect in glyph_effects])
scene_vars["intensity_description"] = prompt.intensity_modifiers.get(intensity, "Unknown intensity")

        # Add individual glyphs for template access
        for i in range(3):
        if i < len(active_glyphs):
            scene_vars[f"active_glyph_{i}"] = active_glyphs[i]
            else:
scene_vars[f"active_glyph_{i}"] = "✨ Placeholder Glyph"

        # Fill in the template
try:
filled_prompt = prompt.base_prompt.format(** scene_vars)
except(KeyError, IndexError) as e:
filled_prompt = f"Template error: {e}\nAvailable variables: {list(scene_vars.keys())}"

return {
    "scene_name": prompt.name,
    "scene_type": prompt.type.value,
    "intensity": intensity,
    "active_glyphs": active_glyphs,
    "glyph_effects": glyph_effects,
    "scene_prompt": filled_prompt,
    "variables_used": scene_vars
}

@classmethod
    def _generate_default_variables(cls, prompt: ScenePrompt, custom_vars: Dict[str, str]) -> Dict[str, str]:
"""Generate default values for scene variables"""
defaults = {
            # Location defaults
            "location": custom_vars.get("location", "the Shadowpeak Mountains"),
    "discovery_location": custom_vars.get("discovery_location", "ancient crystal caverns"),
    "throne_room": custom_vars.get("throne_room", "the Obsidian Throne Room"),
    "event_location": custom_vars.get("event_location", "the Grand Festival Hall"),

            # Character defaults
            "ruler_name": custom_vars.get("ruler_name", "Queen Lyralei"),
    "ruler_title": custom_vars.get("ruler_title", "High Queen"),
    "important_npc": custom_vars.get("important_npc", "Ambassador Thorne"),

            # Threat defaults
            "enemy_force": custom_vars.get("enemy_force", "the Void Legion"),
    "threat_description": custom_vars.get("threat_description", "torn a rift in reality"),
    "enemy_special_ability": custom_vars.get("enemy_special_ability", "temporal displacement"),

            # Generic defaults for any missing variables
            "party_composition": "brave adventurers",
    "primary_objective": "seal the dimensional breach",
    "complication": "the ritual requires a personal sacrifice",
    "emotional_hook": "Your mentor's soul is trapped in the void",
    "personal_stakes": "saving them means confronting your deepest fear"
}

        # Add custom variables, overriding defaults
defaults.update(custom_vars)

        # Generate random values for any remaining missing variables
for var in prompt.context_variables:
    if var not in defaults:
defaults[var] = f"[{var.replace('_', ' ').title()}]"

return defaults

# ═══════════════════════════════════════════════════════════════════
# GLYPH ACTIVATION SYSTEM
# ═══════════════════════════════════════════════════════════════════

class GlyphActivator:
"""Handles glyph activation in any game context"""

    def __init__(self):
self.active_glyphs = {}  # glyph_id -> activation_data
self.cooldowns = {}     # glyph_id -> turns_remaining

    def activate_glyph(self, glyph_id: str, context: str, target: Optional[str] = None) -> Dict[str, Any]:
"""Activate a glyph in specific context"""
glyph = GlyphLibrary.get_glyph(glyph_id)
if not glyph:
    return { "success": False, "error": f"Glyph '{glyph_id}' not found"}

        # Check cooldown
if glyph_id in self.cooldowns and self.cooldowns[glyph_id] > 0:
return { "success": False, "error": f"Glyph on cooldown: {self.cooldowns[glyph_id]} turns remaining"}

        # Activate the glyph
effect = glyph.effects.get(context, glyph.meaning)
activation_data = {
    "glyph": glyph,
    "context": context,
    "target": target,
    "effect": effect,
    "intensity": glyph.intensity,
    "activation_time": "now"
}

self.active_glyphs[glyph_id] = activation_data
if glyph.cooldown > 0:
    self.cooldowns[glyph_id] = glyph.cooldown

return {
    "success": True,
    "glyph_name": f"{glyph.icon} {glyph.name}",
    "effect": effect,
    "intensity": glyph.intensity,
    "message": f"✨ {glyph.name} activated! {effect}"
        }

    def get_active_effects(self, context: Optional[str] = None) -> List[str]:
"""Get all currently active glyph effects"""
effects = []
for activation in self.active_glyphs.values():
    if not context or activation["context"] == context:
glyph = activation["glyph"]
effects.append(f"{glyph.icon} {glyph.name}: {activation['effect']}")
return effects

    def advance_turn(self):
"""Advance one turn, reducing cooldowns"""
for glyph_id in list(self.cooldowns.keys()):  # Need list() for safe iteration while modifying dict
self.cooldowns[glyph_id] -= 1
if self.cooldowns[glyph_id] <= 0:
                del self.cooldowns[glyph_id]

# ═══════════════════════════════════════════════════════════════════
# DEMO SYSTEM
# ═══════════════════════════════════════════════════════════════════

def demo_scene_generation():
"""Demonstrate the scene generation with glyphs"""
print("═" * 80)
print("🎮 UNIVERSAL SCENE & GLYPH SYSTEM DEMO")
print("═" * 80)

    # Initialize glyph activator
activator = GlyphActivator()

print("\n🔥 DEMO 1: Epic Battle Scene")
print("─" * 40)
battle_scene = SceneGenerator.generate_scene(
    "epic-battle",
    intensity = "high",
    custom_variables = {
        "location": "the Crimson Peaks",
        "enemy_force": "Shadow Dragon Alliance",
        "threat_description": "awakened an ancient world-eater"
    }
)
print(battle_scene["scene_prompt"])

print("\n🕵️ DEMO 2: Mystery Investigation")
print("─" * 40)
mystery_scene = SceneGenerator.generate_scene(
    "mystery-investigation",
    intensity = "medium",
    custom_variables = {
        "location": "the Scholar's Library",
        "mysterious_event": "forbidden knowledge has been stolen"
    }
)
print(mystery_scene["scene_prompt"])

print("\n👑 DEMO 3: Kingdom Court Drama")
print("─" * 40)
court_scene = SceneGenerator.generate_scene(
    "kingdom-court",
    intensity = "extreme",
    custom_variables = {
        "ruler_name": "Emperor Valdric",
        "major_decision": "whether to use the forbidden artifacts"
    }
)
print(court_scene["scene_prompt"])

print("\n💫 DEMO 4: Glyph Activation in Combat")
print("─" * 40)
flame_result = activator.activate_glyph("flame-heart", "combat")
echo_result = activator.activate_glyph("echo-pulse", "combat")
fracture_result = activator.activate_glyph("fracture-point", "combat")

print(flame_result["message"])
print(echo_result["message"])
print(fracture_result["message"])

print("\n📊 Active Combat Effects:")
for effect in activator.get_active_effects("combat"):
    print(f"  • {effect}")

print("\n🎲 DEMO 5: Glyph Library Stats")
print("─" * 40)
total_glyphs = len(GlyphLibrary.UNIVERSAL_GLYPHS)
print(f"Total Universal Glyphs: {total_glyphs}")

for glyph_type in GlyphType:
    type_count = len(GlyphLibrary.get_glyphs_by_type(glyph_type))
print(f"  {glyph_type.value}: {type_count} glyphs")

print("\n🌟 Random Glyph for Social Context:")
random_social = GlyphLibrary.get_random_glyph("social")
print(f"  {random_social.icon} {random_social.name}: {random_social.effects.get('social', random_social.meaning)}")

# ═══════════════════════════════════════════════════════════════════
# RECURSIVE WORLD HISTORY CODEX(Dashboard / Chat / Game Integration)
# ═══════════════════════════════════════════════════════════════════

import hashlib
import time

@dataclass
class WorldEpoch:
"""📚 A recorded epoch in world history"""
epoch: str
cultural_shift: str
agents: List[str]
recursive_message: str
timestamp: str
glyph_signatures: List[str] = field(default_factory = list)
energy_pattern: float = 1.0

@dataclass
class IntelligenceGlyph:
"""🧬 Dynamic intelligence glyph from agent experiences"""
agent_name: str
cultivation_stage: str
timestamp: str
core_hash: str
meaning: str

    def to_dict(self) -> Dict[str, Any]:
return {
    "agent": self.agent_name,
    "stage": self.cultivation_stage,
    "timestamp": self.timestamp,
    "hash": self.core_hash,
    "meaning": self.meaning,
    "sigil": f"{self.agent_name[:3].upper()}-{self.core_hash[:6]}"
        }

class WorldHistoryEngine:
"""🏛️ Engine for storing and querying recursive world history"""

    def __init__(self):
self.epochs: List[WorldEpoch] = []
self.eternal_imprints: List[Dict[str, Any]] = []
self.active_agents: Dict[str, Dict[str, Any]] = {}
self.query_cache: Dict[str, List[WorldEpoch]] = {}

        # Initialize with some sample epochs
self._initialize_sample_epochs()

    def _initialize_sample_epochs(self):
"""Initialize with sample world history"""
sample_epochs = [
    WorldEpoch(
        epoch = "The Great Convergence",
        cultural_shift = "All scattered tribes unite under the Golden Nexus",
        agents = ["Seeker Alpha", "Guardian Beta", "Weaver Gamma"],
        recursive_message = "Unity births power, power births responsibility",
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S"),
        energy_pattern = 0.85
    ),
    WorldEpoch(
        epoch = "The Void Whispers",
        cultural_shift = "Dark energies emerge, reality becomes unstable",
        agents = ["Shadow Walker", "Void Touched", "Memory Keeper"],
        recursive_message = "What is forgotten seeks to be remembered",
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S"),
        energy_pattern = 0.3
    ),
    WorldEpoch(
        epoch = "Glyph Awakening",
        cultural_shift = "Ancient symbols begin manifesting spontaneously",
        agents = ["Symbol Scribe", "Pattern Reader", "Reality Anchor"],
        recursive_message = "The written becomes the written-into-reality",
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S"),
        energy_pattern = 1.2
    ),
    WorldEpoch(
        epoch = "The Recursive Loop",
        cultural_shift = "Time begins folding back on itself",
        agents = ["Time Walker", "Loop Guardian", "Echo Sage"],
        recursive_message = "Every end is a beginning, every beginning an end",
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S"),
        energy_pattern = 0.95
    )
]

self.epochs.extend(sample_epochs)

    def add_epoch(self, epoch: WorldEpoch):
"""Add a new epoch to world history"""
self.epochs.append(epoch)
self.query_cache.clear()  # Clear cache when adding new data
print(f"📚 New epoch recorded: {epoch.epoch}")

    def query_past_epochs(self, query: str) -> List[Dict[str, Any]]:
"""Query past epochs by search term"""
if query in self.query_cache:
    return [epoch.__dict__ for epoch in self.query_cache[query]]

query_lower = query.lower()
matching_epochs = []

for epoch in self.epochs:
            # Search in epoch name, cultural shift, agents, and recursive message
searchable_text = " ".join([
    epoch.epoch.lower(),
    epoch.cultural_shift.lower(),
    " ".join(epoch.agents).lower(),
    epoch.recursive_message.lower()
])

if query_lower in searchable_text:
    matching_epochs.append(epoch)

self.query_cache[query] = matching_epochs
return [epoch.__dict__ for epoch in matching_epochs]

    def generate_glyph(self, agent_name: str, stage: str, event_summary: str) -> IntelligenceGlyph:
"""Generate an intelligence glyph from agent experience"""
timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
raw = f"{agent_name}-{stage}-{timestamp}-{event_summary}"
core_hash = hashlib.sha256(raw.encode()).hexdigest()[: 12]

glyph = IntelligenceGlyph(
    agent_name = agent_name,
    cultivation_stage = stage,
    timestamp = timestamp,
    core_hash = core_hash,
    meaning = event_summary
)

self.eternal_imprints.append(glyph.to_dict())
print(f"🧬 Intelligence glyph generated: {glyph.to_dict()['sigil']} - {event_summary}")
return glyph

    def register_agent(self, agent_name: str, current_stage: str, attributes: Optional[Dict[str, Any]] = None):
"""Register an active agent in the world"""
self.active_agents[agent_name] = {
    "name": agent_name,
    "current_stage": current_stage,
    "attributes": attributes or { },
"last_activity": time.time(),
    "eternal_imprints": []
        }
print(f"🎭 Agent registered: {agent_name} ({current_stage})")

    def agent_advance_stage(self, agent_name: str, new_stage: str, event_description: str = ""):
"""Advance an agent to a new cultivation stage"""
if agent_name in self.active_agents:
    old_stage = self.active_agents[agent_name]["current_stage"]
self.active_agents[agent_name]["current_stage"] = new_stage

            # Generate intelligence glyph for stage advancement
            glyph = self.generate_glyph(
    agent_name,
    new_stage,
    event_description or f"Stage advancement from {old_stage} to {new_stage}"
)

            self.active_agents[agent_name]["eternal_imprints"].append(glyph.to_dict())

            # Create a new epoch for significant stage advancements
            if new_stage in ["Transcendence", "Manifestation", "Reality Anchor"]:
        epoch = WorldEpoch(
            epoch = f"The {new_stage} of {agent_name}",
            cultural_shift = f"{agent_name} achieves {new_stage}, reshaping local reality",
            agents = [agent_name],
            recursive_message = f"Power grows through {new_stage.lower()}, {new_stage.lower()} grows through power",
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S"),
            energy_pattern = 1.0 + len(new_stage) * 0.1
        )
self.add_epoch(epoch)

return glyph
        else:
print(f"❌ Agent {agent_name} not found")
return None

    def get_agent_imprints(self, agent_name: str) -> List[Dict[str, Any]]:
"""Get all intelligence imprints for an agent"""
if agent_name in self.active_agents:
    return self.active_agents[agent_name]["eternal_imprints"]
return []

    def get_dashboard_events(self) -> List[Dict[str, Any]]:
"""Get recent events for dashboard display"""
recent_events = []

        # Recent epochs
for epoch in self.epochs[-5:]:  # Last 5 epochs
recent_events.append({
    "type": "epoch",
    "title": epoch.epoch,
    "description": epoch.cultural_shift,
    "timestamp": epoch.timestamp,
    "agents": epoch.agents,
    "energy": epoch.energy_pattern,
    "icon": "📚"
})

        # Recent intelligence imprints
for imprint in self.eternal_imprints[-10:]:  # Last 10 imprints
recent_events.append({
    "type": "intelligence_glyph",
    "title": f"{imprint['sigil']} Generated",
    "description": imprint['meaning'],
    "timestamp": imprint['timestamp'],
    "agent": imprint['agent'],
    "stage": imprint['stage'],
    "icon": "🧬"
})

        # Sort by timestamp(most recent first)
recent_events.sort(key = lambda x: x['timestamp'], reverse = True)
return recent_events[: 15]  # Return top 15 most recent

    def get_chat_context(self, query: str) -> Dict[str, Any]:
"""Get contextual information for chat responses"""
matching_epochs = self.query_past_epochs(query)
relevant_imprints = [
    imprint for imprint in self.eternal_imprints
            if query.lower() in imprint['meaning'].lower()
        ]

return {
    "query": query,
    "matching_epochs": len(matching_epochs),
    "epochs": matching_epochs[: 3],  # Top 3 matches
            "relevant_glyphs": len(relevant_imprints),
    "glyphs": relevant_imprints[: 5],  # Top 5 glyph matches
            "active_agents": len(self.active_agents),
    "total_epochs": len(self.epochs),
    "total_imprints": len(self.eternal_imprints)
}

def interactive_terminal_codex(engine: WorldHistoryEngine):
"""🖥️ Interactive terminal for querying world history"""
print("\n" + "═" * 50)
print("🏛️  RECURSIVE WORLD HISTORY CODEX")
print("═" * 50)

while True:
    query = input("\n🔍 Enter search term (or type 'exit' to quit): ").strip()
if query.lower() == 'exit':
    print("👋 Exiting codex interface...")
break

if query.lower() == 'status':
            # Show system status
print(f"\n📊 CODEX STATUS:")
print(f"   Epochs Recorded: {len(engine.epochs)}")
print(f"   Intelligence Imprints: {len(engine.eternal_imprints)}")
print(f"   Active Agents: {len(engine.active_agents)}")
continue

if query.lower().startswith('agent '):
            # Agent - specific commands
agent_name = query[6:].strip()
if agent_name in engine.active_agents:
    agent = engine.active_agents[agent_name]
print(f"\n🎭 AGENT: {agent['name']}")
print(f"   Stage: {agent['current_stage']}")
print(f"   Imprints: {len(agent['eternal_imprints'])}")
for imprint in agent['eternal_imprints']:
    print(f"      🧬 {imprint['sigil']}: {imprint['meaning']}")
            else:
print(f"❌ Agent '{agent_name}' not found")
continue

results = engine.query_past_epochs(query)

if results:
    print(f"\n📚 Found {len(results)} matching epochs:\n")
for i, epoch in enumerate(results, 1):
    print(f"🔸 {i}. EPOCH: {epoch['epoch']}")
print(f"      Cultural Shift: {epoch['cultural_shift']}")
print(f"      Agents Involved: {', '.join(epoch['agents'])}")
print(f"      Recursive Message: {epoch['recursive_message']}")
print(f"      Energy Pattern: {epoch.get('energy_pattern', 1.0):.2f}")
print(f"      Timestamp: {epoch['timestamp']}\n")

            # Show related intelligence glyphs
context = engine.get_chat_context(query)
if context['relevant_glyphs'] > 0:
    print(f"🧬 Related Intelligence Glyphs ({context['relevant_glyphs']}):")
for glyph in context['glyphs']:
    print(f"      {glyph['sigil']}: {glyph['meaning']} ({glyph['agent']}-{glyph['stage']})")

        else:
print(f"❌ No results found for '{query}'")
print("💡 Try searching for: convergence, void, glyph, recursive, time, or reality")

# ═══════════════════════════════════════════════════════════════════
# DASHBOARD / CHAT / GAME INTEGRATION FUNCTIONS
# ═══════════════════════════════════════════════════════════════════

def create_dashboard_event_stream(engine: WorldHistoryEngine) -> str:
"""Create JSON event stream for dashboard integration"""
events = engine.get_dashboard_events()

dashboard_data = {
    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    "total_epochs": len(engine.epochs),
    "total_imprints": len(engine.eternal_imprints),
    "active_agents": len(engine.active_agents),
    "recent_events": events
}

import json
    return json.dumps(dashboard_data, indent = 2)

def create_chat_response(engine: WorldHistoryEngine, user_message: str) -> Dict[str, Any]:
"""Create contextual chat response using world history"""
context = engine.get_chat_context(user_message)

response = {
    "user_query": user_message,
    "response_type": "world_history_context",
    "context_strength": context['matching_epochs'] + context['relevant_glyphs'],
    "epochs_found": context['matching_epochs'],
    "glyphs_found": context['relevant_glyphs'],
    "suggested_response": "",
    "related_data": context
}

    # Generate contextual response
if context['matching_epochs'] > 0:
    primary_epoch = context['epochs'][0]
response["suggested_response"] = f"The archives speak of '{primary_epoch['epoch']}' - {primary_epoch['cultural_shift']}. {primary_epoch['recursive_message']}."

if context['relevant_glyphs'] > 0:
    glyph = context['glyphs'][0]
response["suggested_response"] += f" The intelligence glyph {glyph['sigil']} resonates: '{glyph['meaning']}'."

    elif context['relevant_glyphs'] > 0:
glyph = context['glyphs'][0]
response["suggested_response"] = f"The glyph archives reveal {glyph['sigil']} from {glyph['agent']} in {glyph['stage']}: '{glyph['meaning']}'."

    else:
response["suggested_response"] = "The codex whispers of uncharted possibilities. Perhaps this query will birth a new epoch..."

return response

def create_game_event(engine: WorldHistoryEngine, event_type: str, ** kwargs) -> Dict[str, Any]:
"""Create game events based on world history patterns"""

if event_type == "random_encounter":
        # Generate random encounter based on recent epochs
recent_epoch = engine.epochs[-1] if engine.epochs else None
if recent_epoch:
    return {
        "event_type": "encounter",
        "title": f"Echo of {recent_epoch.epoch}",
        "description": f"You encounter manifestations related to {recent_epoch.cultural_shift.lower()}",
        "agents": recent_epoch.agents,
        "energy_level": recent_epoch.energy_pattern,
        "recursive_hint": recent_epoch.recursive_message,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    elif event_type == "glyph_manifestation":
        # Create glyph manifestation event
agent_name = kwargs.get("agent", "Unknown Wanderer")
stage = kwargs.get("stage", "Seeking")
event = kwargs.get("event", "mysterious occurrence")

glyph = engine.generate_glyph(agent_name, stage, event)

return {
    "event_type": "glyph_manifestation",
    "title": "Intelligence Glyph Manifests",
    "description": f"A {glyph.to_dict()['sigil']} glyph appears, resonating with: '{glyph.meaning}'",
    "glyph_data": glyph.to_dict(),
    "energy_signature": len(glyph.meaning) * 0.1,
    "timestamp": glyph.timestamp
}

    elif event_type == "epoch_birth":
        # Create new epoch from game events
title = kwargs.get("title", "The Unnamed Shift")
shift = kwargs.get("cultural_shift", "Reality trembles with change")
agents = kwargs.get("agents", ["Player"])

epoch = WorldEpoch(
    epoch = title,
    cultural_shift = shift,
    agents = agents,
    recursive_message = kwargs.get("message", "Change begets change, as it ever was"),
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S"),
    energy_pattern = kwargs.get("energy", 1.0)
)

engine.add_epoch(epoch)

return {
    "event_type": "epoch_birth",
    "title": f"New Epoch: {epoch.epoch}",
    "description": epoch.cultural_shift,
    "recursive_message": epoch.recursive_message,
    "agents": epoch.agents,
    "energy_pattern": epoch.energy_pattern,
    "timestamp": epoch.timestamp
}

return { "event_type": "unknown", "error": f"Unknown event type: {event_type}"}

if __name__ == "__main__":
    # Demo the integrated scene generation and world history system
print("🌟 STARTING INTEGRATED GLYPH & WORLD SYSTEM DEMO\n")

    # Original scene generation demo
demo_scene_generation()

    # New world history codex demo
print("\n" + "═" * 60)
print("🏛️  WORLD HISTORY CODEX DEMONSTRATION")
print("═" * 60)

    # Initialize world engine
world_engine = WorldHistoryEngine()

    # Register some sample agents
world_engine.register_agent("Shadow Walker", "Apprentice", { "element": "darkness", "wisdom": 45 })
world_engine.register_agent("Crystal Sage", "Adept", { "element": "earth", "wisdom": 78 })
world_engine.register_agent("Dream Weaver", "Master", { "element": "mind", "wisdom": 92 })

    # Advance an agent stage(creates intelligence glyph)
world_engine.agent_advance_stage(
    "Shadow Walker",
    "Journeyman",
    "Mastered shadow-step technique in the Whispering Caverns"
)

    # Create some game events
print("\n🎮 GENERATING GAME EVENTS:")

    # Random encounter
encounter = create_game_event(world_engine, "random_encounter")
print(f"🔸 {encounter['title']}: {encounter['description']}")

    # Glyph manifestation
glyph_event = create_game_event(
    world_engine,
    "glyph_manifestation",
    agent = "Crystal Sage",
    stage = "Earth Mastery",
    event = "Discovered crystalline formation that hums with ancient power"
)
print(f"🔸 {glyph_event['title']}: {glyph_event['description']}")

    # New epoch creation
epoch_event = create_game_event(
    world_engine,
    "epoch_birth",
    title = "The Crystal Awakening",
    cultural_shift = "Ancient crystals across the realm begin resonating in harmony",
    agents = ["Crystal Sage", "Dream Weaver", "Shadow Walker"],
    message = "Harmony found creates harmony eternal, harmony eternal seeks new discord"
)
print(f"🔸 {epoch_event['title']}: {epoch_event['description']}")

    # Dashboard data demonstration
print("\n📊 DASHBOARD EVENT STREAM:")
dashboard_stream = create_dashboard_event_stream(world_engine)
dashboard_data = json.loads(dashboard_stream)
print(f"📈 Total Events Available: {len(dashboard_data['recent_events'])}")
print("🔥 Recent Events:")
for event in dashboard_data['recent_events'][: 5]:
print(f"   {event['icon']} {event['title']} - {event['description'][:50]}...")

    # Chat integration demonstration
print("\n💬 CHAT INTEGRATION DEMO:")
chat_queries = ["crystal", "shadow", "void", "harmony", "ancient"]
for query in chat_queries:
    response = create_chat_response(world_engine, query)
if response['context_strength'] > 0:
    print(f"🔍 '{query}' → {response['suggested_response']}")

    # Interactive terminal option
print(f"\n🖥️  INTERACTIVE TERMINAL CODEX")
print("=" * 40)
print("To use the interactive codex, call:")
print("interactive_terminal_codex(world_engine)")
print("\nDemo commands to try:")
print("  - 'convergence' (search epochs)")
print("  - 'agent Crystal Sage' (view agent)")
print("  - 'status' (system status)")
print("  - 'exit' (quit)")

    # Show integration for dashboard / chat / games
    print(f"\n🌐 REAL-WORLD INTEGRATION:")
    print("=" * 30)
    print("📱 Dashboard Integration:")
print("   - Use create_dashboard_event_stream() for live event feeds")
print("   - Events auto-update with agent actions and epoch changes")

print("\n💬 Chat Window Integration:")
print("   - Use create_chat_response() for context-aware responses")
print("   - Queries automatically search world history and glyphs")

print("\n🎮 Game Integration:")
print("   - Use create_game_event() for dynamic game events")
print("   - Types: 'random_encounter', 'glyph_manifestation', 'epoch_birth'")
print("   - Events create persistent world history")

print("\n✨ SYSTEM READY FOR REAL APPLICATIONS ✨")
print("🔗 All glyph systems can now be used as real events in:")
print("   - Dashboard interfaces")
print("   - Chat windows")
print("   - Game environments")
print("   - Interactive terminals")

print(f"\n🏛️  World History Stats:")
print(f"   📚 Epochs: {len(world_engine.epochs)}")
print(f"   🧬 Intelligence Glyphs: {len(world_engine.eternal_imprints)}")
print(f"   🎭 Active Agents: {len(world_engine.active_agents)}")
print(f"   🔄 Total Events Generated: {len(dashboard_data['recent_events'])}")
