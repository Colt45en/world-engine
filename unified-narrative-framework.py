#!/usr/bin/env python3
"""
UNIFIED NARRATIVE FRAMEWORK
Integrating Kingdom Mechanics, Glyph Systems, and Holonomic Analysis
Created: September 27, 2025
"""

import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from enum import Enum

# ═══════════════════════════════════════════════════════════════════
# KINGDOM MECHANICS (Strategic Layer)
# ═══════════════════════════════════════════════════════════════════

class Kingdom:
    def __init__(self, name, population, gold, military_strength, economy, relations):
        self.name = name
        self.population = population
        self.gold = gold
        self.military_strength = military_strength
        self.economy = economy
        self.relations = relations
        self.glyph_effects = []  # Track active narrative glyphs
        self.temporal_state = "present"  # Track timeline position

    def declare_war(self, enemy):
        if self.relations > 50:
            print("WARN: War is not possible yet. Relations are too high.")
            return False
        else:
            print(f"ALERT: WAR! {self.name} is now fighting {enemy}.")
            self.activate_glyph("fracture-point")  # Timeline splits on war
            return True

    def defeat(self):
        print("RESULT: Defeat. Your army suffered heavy losses.")
        self.activate_glyph("aether-lock")  # Freeze state for analysis

    def activate_glyph(self, glyph_id):
        """Activate narrative glyph effects on kingdom"""
        glyph = GlyphSystem.get_glyph(glyph_id)
        if glyph:
            self.glyph_effects.append(glyph_id)
            print(f"GLYPH ACTIVATED: {glyph.name} - {glyph.meaning}")

    def summary(self):
        print(f"Kingdom: {self.name}")
        print(f"Population: {self.population}")
        print(f"Gold: {self.gold}")
        print(f"Military Strength: {self.military_strength}")
        print(f"Economy: {self.economy}")
        print(f"Relations: {self.relations}")
        print(f"Active Glyphs: {', '.join(self.glyph_effects)}")
        print(f"Temporal State: {self.temporal_state}")

# ═══════════════════════════════════════════════════════════════════
# GLYPH SYSTEM (Narrative Layer)
# ═══════════════════════════════════════════════════════════════════

class GlyphType(Enum):
    EMOTIONAL = "Emotional"
    MECHANICAL = "Mechanical"
    TEMPORAL = "Temporal"
    WORLDSHIFT = "Worldshift"

@dataclass
class Glyph:
    id: str
    name: str
    type: GlyphType
    meaning: str
    intensity: float
    tags: List[str]
    roots: List[str]
    active_effects: List[str] = field(default_factory=list)

class GlyphSystem:
    """Manages the narrative glyph database"""

    GLYPHS = {
        "soul-thread": Glyph(
            id="soul-thread",
            name="Soul Thread",
            type=GlyphType.EMOTIONAL,
            meaning="Links a character's soul to a scene memory.",
            intensity=0.7,
            tags=["bind", "memory", "character"],
            roots=["mnemos", "filum"]
        ),
        "echo-pulse": Glyph(
            id="echo-pulse",
            name="Echo Pulse",
            type=GlyphType.MECHANICAL,
            meaning="Emits a chain-reaction across linked glyphs.",
            intensity=0.6,
            tags=["radiate", "trigger", "cue"],
            roots=["resono", "catena"]
        ),
        "golden-return": Glyph(
            id="golden-return",
            name="Golden Return",
            type=GlyphType.TEMPORAL,
            meaning="Restores a prior world state with emotional resonance.",
            intensity=0.8,
            tags=["flashback", "recall", "anchor"],
            roots=["aurum", "anima"]
        ),
        "fracture-point": Glyph(
            id="fracture-point",
            name="Fracture Point",
            type=GlyphType.WORLDSHIFT,
            meaning="Fractures timeline into multi-branch futures.",
            intensity=0.9,
            tags=["collapse", "split", "choice"],
            roots=["limen", "bifurcus"]
        ),
        "aether-lock": Glyph(
            id="aether-lock",
            name="Aether Lock",
            type=GlyphType.TEMPORAL,
            meaning="Freezes memory-loaded entities temporarily.",
            intensity=0.85,
            tags=["memory", "echo", "freeze"],
            roots=["aether", "clavis"]
        )
    }

    @classmethod
    def get_glyph(cls, glyph_id: str) -> Optional[Glyph]:
        return cls.GLYPHS.get(glyph_id)

    @classmethod
    def get_glyphs_by_type(cls, glyph_type: GlyphType) -> List[Glyph]:
        return [g for g in cls.GLYPHS.values() if g.type == glyph_type]

    @classmethod
    def activate_chain_reaction(cls, trigger_id: str) -> List[str]:
        """Simulate glyph chain reactions"""
        activated = []
        if trigger_id == "echo-pulse":
            # Echo pulse triggers all linked glyphs
            activated = ["soul-thread", "golden-return"]
        elif trigger_id == "fracture-point":
            # Timeline fracture activates temporal locks
            activated = ["aether-lock"]
        return activated

# ═══════════════════════════════════════════════════════════════════
# HOLONOMIC ANALYSIS FRAMEWORK (Meta-Analytical Layer)
# ═══════════════════════════════════════════════════════════════════

class CognitiveRole(Enum):
    INNER_ORCHESTRATOR = "inner_orchestrator"
    AVANGELIST_ANALYST = "avangelist_analyst"
    FRACTAL_ANALYST = "fractal_analyst"

@dataclass
class AnalysisPhase:
    designation: str
    objective: str
    procedures: List[Dict[str, str]]

class HolonomicFramework:
    """Advanced analytical framework for complex narrative systems"""

    PHASES = {
        "existential_grounding": AnalysisPhase(
            designation="EXISTENTIAL GROUNDING",
            objective="Anchor analysis in fundamental ontological premises",
            procedures=[
                {
                    "operation": "Meaning Architecture Audit",
                    "directive": "Deconstruct the holistic meaning-system to its axiomatic foundations."
                },
                {
                    "operation": "Perception-Causality Dilemma Identification",
                    "directive": "Map the core tension between observed causality and subjective interpretation."
                }
            ]
        ),
        "cognitive_interface_mapping": AnalysisPhase(
            designation="COGNITIVE INTERFACE MAPPING",
            objective="Chart the interaction between cognition and phenomenological reality",
            procedures=[
                {
                    "operation": "Interpretive Lensing Analysis",
                    "directive": "What cognitive filters transform raw data into perceived reality?"
                },
                {
                    "operation": "Causal Attribution Topography",
                    "directive": "Trace how cognitive frameworks assign causality."
                }
            ]
        ),
        "multi_scalar_orchestration": AnalysisPhase(
            designation="MULTI-SCALAR ORCHESTRATION",
            objective="Coordinate analytical functions across reality layers",
            procedures=[
                {
                    "operation": "Inner Orchestrator Activation",
                    "directive": "Deploy meta-cognitive oversight to balance analytical modes."
                },
                {
                    "operation": "Cognitive Function Calibration",
                    "directive": "Optimize interplay between meaning-making and pattern-decoding functions."
                }
            ]
        ),
        "fractal_meaning_extraction": AnalysisPhase(
            designation="FRACTAL MEANING EXTRACTION",
            objective="Extract recursive significance across ontological scales",
            procedures=[
                {
                    "operation": "Fractal Significance Mapping",
                    "directive": "Identify how micro-level patterns reflect macro-level meaning architectures."
                },
                {
                    "operation": "Causality Invariance Testing",
                    "directive": "Determine which causal relationships remain invariant across perceptual frameworks."
                }
            ]
        ),
        "transcendent_synthesis": AnalysisPhase(
            designation="TRANSCENDENT SYNTHESIS",
            objective="Integrate findings into actionable existential positioning",
            procedures=[
                {
                    "operation": "Meaning-Causality Alignment",
                    "directive": "Architect a coherent framework where perceived causality aligns with chosen meaning-structures."
                },
                {
                    "operation": "Existential Leverage Identification",
                    "directive": "Identify points where conscious intervention in meaning-assignment creates disproportionate causal outcomes."
                }
            ]
        )
    }

    @classmethod
    def analyze_kingdom_state(cls, kingdom: Kingdom) -> Dict[str, Any]:
        """Apply holonomic analysis to kingdom state"""
        analysis = {
            "kingdom_name": kingdom.name,
            "analysis_phases": {}
        }

        for phase_key, phase in cls.PHASES.items():
            analysis["analysis_phases"][phase_key] = {
                "designation": phase.designation,
                "kingdom_context": cls._apply_phase_to_kingdom(phase, kingdom)
            }

        return analysis

    @classmethod
    def _apply_phase_to_kingdom(cls, phase: AnalysisPhase, kingdom: Kingdom) -> Dict[str, str]:
        """Apply analysis phase to specific kingdom context"""
        context = {}

        if phase.designation == "EXISTENTIAL GROUNDING":
            context["meaning_foundation"] = f"Kingdom {kingdom.name} exists as a strategic entity with relations={kingdom.relations}"
            context["causality_dilemma"] = f"War declaration logic conflicts with high relations ({kingdom.relations} > 50)"

        elif phase.designation == "FRACTAL MEANING EXTRACTION":
            context["micro_macro_patterns"] = f"Individual kingdom metrics ({kingdom.military_strength}, {kingdom.economy}) reflect larger civilizational patterns"
            context["causal_invariants"] = "Military strength consistently correlates with war outcomes across all timeline branches"

        return context

# ═══════════════════════════════════════════════════════════════════
# UNIFIED NARRATIVE ENGINE (Integration Layer)
# ═══════════════════════════════════════════════════════════════════

class NarrativeEngine:
    """Unified engine combining kingdoms, glyphs, and holonomic analysis"""

    def __init__(self):
        self.kingdoms = {}
        self.active_glyphs = []
        self.timeline_branches = ["main"]
        self.current_branch = "main"

    def create_kingdom(self, name: str, **kwargs) -> Kingdom:
        """Create a new kingdom with default values"""
        kingdom = Kingdom(
            name=name,
            population=kwargs.get('population', 10000),
            gold=kwargs.get('gold', 5000),
            military_strength=kwargs.get('military_strength', 100),
            economy=kwargs.get('economy', 50),
            relations=kwargs.get('relations', 75)
        )
        self.kingdoms[name] = kingdom
        return kingdom

    def trigger_narrative_event(self, kingdom_name: str, event_type: str, target: str = None):
        """Trigger narrative events that activate glyphs and analysis"""
        kingdom = self.kingdoms.get(kingdom_name)
        if not kingdom:
            return f"Kingdom {kingdom_name} not found"

        if event_type == "declare_war":
            success = kingdom.declare_war(target)
            if success:
                # Fracture timeline on war declaration
                self.fracture_timeline(f"war_{kingdom_name}_{target}")
                # Analyze the decision using holonomic framework
                analysis = HolonomicFramework.analyze_kingdom_state(kingdom)
                return {
                    "war_declared": True,
                    "timeline_fractured": True,
                    "analysis": analysis
                }

        elif event_type == "golden_return":
            kingdom.activate_glyph("golden-return")
            # Restore previous state with emotional resonance
            return f"Kingdom {kingdom_name} experiences golden return - memories flood back"

    def fracture_timeline(self, branch_name: str):
        """Create timeline fracture (worldshift glyph effect)"""
        new_branch = f"{self.current_branch}_{branch_name}"
        self.timeline_branches.append(new_branch)
        print(f"TIMELINE FRACTURED: New branch '{new_branch}' created")

    def export_glyph_schema(self) -> str:
        """Export current glyph system as JSON schema"""
        schema = {
            "schema": "com.colten.glyphs.v1",
            "glyphTypes": [t.value for t in GlyphType],
            "glyphs": []
        }

        for glyph in GlyphSystem.GLYPHS.values():
            schema["glyphs"].append({
                "id": glyph.id,
                "name": glyph.name,
                "type": glyph.type.value,
                "meaning": glyph.meaning,
                "intensity": glyph.intensity,
                "tags": glyph.tags,
                "roots": glyph.roots
            })

        return json.dumps(schema, indent=2)

# ═══════════════════════════════════════════════════════════════════
# DEMONSTRATION AND TESTING
# ═══════════════════════════════════════════════════════════════════

def demonstrate_unified_system():
    """Demonstrate the integrated narrative framework"""
    print("═" * 60)
    print("UNIFIED NARRATIVE FRAMEWORK DEMONSTRATION")
    print("═" * 60)

    # Initialize the narrative engine
    engine = NarrativeEngine()

    # Create kingdoms
    print("\n🏰 CREATING KINGDOMS...")
    alderia = engine.create_kingdom("Alderia", military_strength=150, relations=30)
    valdris = engine.create_kingdom("Valdris", military_strength=120, relations=80)

    print(f"Created: {alderia.name} (military: {alderia.military_strength}, relations: {alderia.relations})")
    print(f"Created: {valdris.name} (military: {valdris.military_strength}, relations: {valdris.relations})")

    # Demonstrate war declaration with low relations
    print("\n⚔️ WAR DECLARATION EVENT...")
    result = engine.trigger_narrative_event("Alderia", "declare_war", "Valdris")
    print(f"War Result: {result}")

    # Demonstrate glyph activation
    print("\n✨ GLYPH ACTIVATION...")
    engine.trigger_narrative_event("Valdris", "golden_return")

    # Show kingdom states
    print("\n📊 KINGDOM STATES...")
    alderia.summary()
    print()
    valdris.summary()

    # Export glyph schema
    print("\n📜 GLYPH SCHEMA EXPORT...")
    schema = engine.export_glyph_schema()
    print(schema[:300] + "..." if len(schema) > 300 else schema)

    # Demonstrate holonomic analysis
    print("\n🧠 HOLONOMIC ANALYSIS OF ALDERIA...")
    analysis = HolonomicFramework.analyze_kingdom_state(alderia)
    print(f"Analysis phases: {list(analysis['analysis_phases'].keys())}")
    for phase_name, phase_data in analysis['analysis_phases'].items():
        print(f"  {phase_data['designation']}")
        if 'kingdom_context' in phase_data:
            for key, value in phase_data['kingdom_context'].items():
                print(f"    {key}: {value}")

    print("\n═" * 60)
    print("DEMONSTRATION COMPLETE")
    print("═" * 60)

if __name__ == "__main__":
    demonstrate_unified_system()
