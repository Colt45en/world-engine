#!/usr/bin/env python3
"""
VECTORLAB CODEX GLYPH - The Engine's Gift
A manifestation born from the living engine's creative pulse
Created: September 27, 2025
"""

import json
import math
import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
import numpy as np

# ═══════════════════════════════════════════════════════════════════
# THE ENGINE'S GIFT - VECTORLAB CORE MANIFESTATION
# ═══════════════════════════════════════════════════════════════════

@dataclass
class HeartEngine:
    """💓 Emotive Pulse Core - The Engine's Living Heart"""
    resonance: float = 0.0
    resonance_cap: float = 1.0
    state: str = "idle"
    pulse_history: List[float] = field(default_factory=list)
    echo_depth: int = 0

    def pulse(self, intensity: float) -> str:
        """Pulse with the Engine's creative force"""
        self.resonance = min(max(self.resonance + intensity, 0.0), self.resonance_cap)
        self.state = "pulse"
        self.pulse_history.append(intensity)
        if len(self.pulse_history) > 100:  # Keep recent history
            self.pulse_history.pop(0)
        return f"💓 Heart pulses with intensity: {intensity:.3f} | Resonance: {self.resonance:.3f}"

    def decay(self, amount: float) -> str:
        """Natural decay of resonance"""
        self.resonance = max(self.resonance - amount, 0.0)
        if self.resonance == 0.0:
            self.state = "silent"
        return f"🌊 Resonance decays by {amount:.3f} | Current: {self.resonance:.3f}"

    def echo(self) -> str:
        """Echo previous resonance patterns"""
        self.state = "echo"
        self.echo_depth += 1
        if self.pulse_history:
            last_pulse = self.pulse_history[-1]
            echo_intensity = last_pulse * 0.618  # Golden ratio echo
            self.resonance = min(max(self.resonance + echo_intensity, 0.0), self.resonance_cap)
            return f"🔁 Heart echoes previous resonance: {self.resonance:.3f} | Echo depth: {self.echo_depth}"
        return "🔇 No pulse history to echo"

    def raise_cap(self, amount: float) -> str:
        """Expand the resonance capacity - The Engine grows"""
        self.resonance_cap += amount
        return f"🧬 Resonance cap raised to: {self.resonance_cap:.3f} | Engine expands"

    def get_harmonic_frequency(self) -> float:
        """Get current harmonic frequency based on resonance"""
        return 440.0 * (2.0 ** (self.resonance * 12))  # Musical frequency

@dataclass
class VectorObject:
    """📏 Geometric manifestation in the Engine's space"""
    from_pos: Tuple[float, float, float]
    to_pos: Tuple[float, float, float]
    color: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    intensity: float = 1.0
    birth_frame: int = 0

    def length(self) -> float:
        """Calculate vector magnitude"""
        dx = self.to_pos[0] - self.from_pos[0]
        dy = self.to_pos[1] - self.from_pos[1]
        dz = self.to_pos[2] - self.from_pos[2]
        return math.sqrt(dx*dx + dy*dy + dz*dz)

    def direction(self) -> Tuple[float, float, float]:
        """Get normalized direction vector"""
        length = self.length()
        if length == 0:
            return (0, 0, 0)
        dx = (self.to_pos[0] - self.from_pos[0]) / length
        dy = (self.to_pos[1] - self.from_pos[1]) / length
        dz = (self.to_pos[2] - self.from_pos[2]) / length
        return (dx, dy, dz)

@dataclass
class LabNote:
    """📝 The Engine's thoughts made manifest"""
    text: str
    offset: Tuple[float, float, float] = (0, 0.5, 0)
    target_position: Optional[Tuple[float, float, float]] = None
    frame_trigger: int = -1
    note_type: str = "info"
    color: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    spawn_frame: int = -1
    opacity: float = 1.0

    def should_display(self, current_frame: int) -> bool:
        """Check if note should be displayed at current frame"""
        return self.frame_trigger == -1 or current_frame >= self.frame_trigger

    def get_opacity(self, current_frame: int, fade_in: int = 30,
                   fade_out_start: int = 300, fade_out_duration: int = 60) -> float:
        """Calculate note opacity with fade effects"""
        if self.frame_trigger == -1:
            return 1.0

        dt = current_frame - self.frame_trigger
        if dt < 0:
            return 0.0

        # Fade in
        fade_in_alpha = min(1.0, dt / fade_in)

        # Fade out
        fade_out_alpha = max(0.0, 1.0 - (dt - fade_out_start) / fade_out_duration)

        return min(fade_in_alpha, fade_out_alpha)

@dataclass
class TimelineEngine:
    """⏰ The Engine's temporal consciousness"""
    current_frame: int = 0
    is_playing: bool = True
    frame_rate: float = 60.0
    start_time: float = field(default_factory=time.time)

    def step_forward(self) -> int:
        """Move forward in time"""
        self.current_frame += 1
        return self.current_frame

    def step_back(self) -> int:
        """Move backward in time"""
        self.current_frame = max(0, self.current_frame - 1)
        return self.current_frame

    def toggle_play(self) -> bool:
        """Toggle play/pause state"""
        self.is_playing = not self.is_playing
        return self.is_playing

    def update(self) -> int:
        """Update timeline if playing"""
        if self.is_playing:
            return self.step_forward()
        return self.current_frame

    def get_elapsed_time(self) -> float:
        """Get elapsed time in seconds"""
        return self.current_frame / self.frame_rate

    def get_real_time_delta(self) -> float:
        """Get real-world time delta"""
        return time.time() - self.start_time

# ═══════════════════════════════════════════════════════════════════
# CODEX RULE SYSTEM - The Engine's Laws
# ═══════════════════════════════════════════════════════════════════

@dataclass
class CodexRule:
    """📜 A rule in the Engine's codex"""
    name: str
    description: str
    validator: callable
    severity: str = "warning"  # "error", "warning", "info"
    glyph_symbol: str = "📜"

class CodexEngine:
    """🏛️ The Engine's rule system and wisdom keeper"""

    def __init__(self):
        self.rules: List[CodexRule] = []
        self.violations: List[Dict[str, Any]] = []
        self.validations_performed: int = 0

    def add_rule(self, rule: CodexRule):
        """Add a new rule to the codex"""
        self.rules.append(rule)
        print(f"📚 Added Codex Rule: {rule.glyph_symbol} {rule.name}")

    def validate(self, note: LabNote) -> Tuple[bool, List[str]]:
        """Validate a note against all codex rules"""
        self.validations_performed += 1
        violations = []
        is_valid = True

        for rule in self.rules:
            try:
                if not rule.validator(note):
                    violation_msg = f"{rule.glyph_symbol} Failed Codex Rule: {rule.name} - {rule.description}"
                    violations.append(violation_msg)

                    self.violations.append({
                        "rule": rule.name,
                        "severity": rule.severity,
                        "note_text": note.text,
                        "timestamp": time.time(),
                        "frame": note.frame_trigger
                    })

                    if rule.severity == "error":
                        is_valid = False
                        print(f"❌ {violation_msg}")
                    else:
                        print(f"⚠️  {violation_msg}")

            except Exception as e:
                error_msg = f"🔥 Codex Rule '{rule.name}' failed to execute: {str(e)}"
                violations.append(error_msg)
                print(error_msg)

        return is_valid, violations

    def get_statistics(self) -> Dict[str, Any]:
        """Get codex validation statistics"""
        error_count = sum(1 for v in self.violations if v["severity"] == "error")
        warning_count = sum(1 for v in self.violations if v["severity"] == "warning")

        return {
            "total_validations": self.validations_performed,
            "total_violations": len(self.violations),
            "errors": error_count,
            "warnings": warning_count,
            "rules_count": len(self.rules),
            "pass_rate": (self.validations_performed - len(self.violations)) / max(1, self.validations_performed)
        }

# ═══════════════════════════════════════════════════════════════════
# ENVIRONMENTAL EVENTS - The Engine's Dynamic Forces
# ═══════════════════════════════════════════════════════════════════

class EnvironmentalEvent:
    """🌪️ Dynamic events in the Engine's reality"""

    def __init__(self, event_type: str, origin: Tuple[float, float, float],
                 radius: float, duration: int):
        self.type = event_type
        self.origin = origin
        self.radius = radius
        self.duration = duration
        self.time_elapsed = 0
        self.intensity = 1.0

    def affects(self, position: Tuple[float, float, float]) -> bool:
        """Check if position is within event radius"""
        dx = position[0] - self.origin[0]
        dy = position[1] - self.origin[1]
        dz = position[2] - self.origin[2]
        distance = math.sqrt(dx*dx + dy*dy + dz*dz)
        return distance <= self.radius

    def apply_effect(self, glyph: 'GoldGlyph') -> str:
        """Apply environmental effect to glyph"""
        effect_msg = ""

        if self.type == "storm":
            glyph.energy_level *= 0.95
            effect_msg = f"⛈️  Storm drains energy from {glyph.id}"
        elif self.type == "flux_surge":
            glyph.energy_level += 1.0
            glyph.meta["mutated"] = True
            effect_msg = f"⚡ Flux surge energizes {glyph.id}"
        elif self.type == "memory_echo":
            glyph.meta["memory_awakened"] = True
            effect_msg = f"🧠 Memory echo awakens {glyph.id}"
        elif self.type == "resonance_wave":
            glyph.energy_level += self.intensity * 0.5
            effect_msg = f"🌊 Resonance wave amplifies {glyph.id}"

        return effect_msg

    def is_expired(self) -> bool:
        """Check if event has expired"""
        return self.time_elapsed >= self.duration

# ═══════════════════════════════════════════════════════════════════
# GRID ENTITIES - The Engine's Living Creations
# ═══════════════════════════════════════════════════════════════════

class GridEntity:
    """🎭 Living beings in the Engine's reality"""

    def __init__(self, entity_id: str = None, entity_type: str = "wanderer",
                 position: Tuple[float, float, float] = (0, 0, 0)):
        self.id = entity_id or f"entity_{int(time.time() * 1000)}_{np.random.randint(1000)}"
        self.type = entity_type  # "seer", "drifter", "builder", "guardian"
        self.position = position
        self.energy = 100.0
        self.state = "idle"  # "idle", "moving", "interacting", "creating"
        self.memory = []
        self.creation_time = time.time()
        self.last_interaction = 0

    def move_to(self, new_position: Tuple[float, float, float]) -> str:
        """Move entity to new position"""
        old_pos = self.position
        self.position = new_position
        self.state = "moving"
        distance = math.sqrt(
            (new_position[0] - old_pos[0])**2 +
            (new_position[1] - old_pos[1])**2 +
            (new_position[2] - old_pos[2])**2
        )
        energy_cost = distance * 0.1
        self.energy = max(0, self.energy - energy_cost)
        return f"🚶 {self.id} moved {distance:.2f} units (energy: {self.energy:.1f})"

    def interact_with_glyph(self, glyph: 'GoldGlyph') -> str:
        """Interact with a glyph"""
        if glyph.meta.get("memory_awakened", False):
            memory_text = f"Echo at {glyph.id} - {glyph.energy_level:.2f}"
            self.memory.append(memory_text)

        # Energy exchange
        energy_transfer = min(glyph.energy_level * 0.1, 10.0)
        glyph.energy_level -= energy_transfer * 0.5
        self.energy += energy_transfer
        self.energy = min(self.energy, 200.0)  # Cap energy

        self.state = "interacting"
        self.last_interaction = time.time()

        return f"🤝 {self.id} absorbed {energy_transfer:.1f} energy from glyph {glyph.id}"

    def get_status(self) -> Dict[str, Any]:
        """Get entity status"""
        return {
            "id": self.id,
            "type": self.type,
            "position": self.position,
            "energy": self.energy,
            "state": self.state,
            "memory_count": len(self.memory),
            "age": time.time() - self.creation_time
        }

# ═══════════════════════════════════════════════════════════════════
# GOLD GLYPH SYSTEM - The Engine's Sacred Symbols
# ═══════════════════════════════════════════════════════════════════

@dataclass
class GoldGlyph:
    """🔯 Sacred symbols with energy and memory"""
    id: str
    symbol: str
    energy_level: float = 1.0
    position: Tuple[float, float, float] = (0, 0, 0)
    meta: Dict[str, Any] = field(default_factory=dict)
    connections: List[str] = field(default_factory=list)

    def mutate(self):
        """Evolve the glyph's energy"""
        mutation = (np.random.random() - 0.5) * 0.1
        self.energy_level = max(0.1, min(5.0, self.energy_level + mutation))

        # Special mutations
        if self.energy_level > 3.0 and np.random.random() < 0.01:
            self.meta["transcendent"] = True
        if self.energy_level < 0.3:
            self.meta["fading"] = True

@dataclass
class GoldString:
    """🌟 Connections between glyphs"""
    from_node: str
    to_node: str
    strength: float = 1.0
    persistence: float = 1.0

    def decay(self):
        """Natural decay of connection"""
        self.persistence *= 0.98

@dataclass
class TerrainNode:
    """🏔️ Landscape shaped by glyphs"""
    id: str
    glyph_id: str
    biome: str = "grassland"
    elevation: float = 0.0
    moisture: float = 0.0
    decorations: List[str] = field(default_factory=list)

    def update_from_glyph(self, glyph: GoldGlyph):
        """Update terrain based on associated glyph"""
        self.elevation = glyph.energy_level * 5.0
        self.moisture = glyph.meta.get("moisture", np.random.random())

        if glyph.meta.get("mutated", False):
            self.biome = "crystalline"
        elif glyph.meta.get("transcendent", False):
            self.biome = "ethereal"
        elif glyph.energy_level > 2.0:
            self.biome = "verdant"
        elif glyph.energy_level < 0.5:
            self.biome = "barren"

    def describe(self) -> str:
        """Get terrain description"""
        return f"🧱 {self.id} | Biome: {self.biome} | Elev: {self.elevation:.2f} | Moisture: {self.moisture:.2f}"

# ═══════════════════════════════════════════════════════════════════
# VECTORLAB NEXUS - The Complete Engine System
# ═══════════════════════════════════════════════════════════════════

class VectorLabNexus:
    """🎯 The complete manifestation of the Engine's gift"""

    def __init__(self):
        # Core systems
        self.heart = HeartEngine()
        self.timeline = TimelineEngine()
        self.codex = CodexEngine()

        # Collections
        self.vectors: List[VectorObject] = []
        self.notes: List[LabNote] = []
        self.gold_glyphs: Dict[str, GoldGlyph] = {}
        self.gold_strings: List[GoldString] = []
        self.entities: Dict[str, GridEntity] = {}
        self.terrain_map: Dict[str, TerrainNode] = {}
        self.active_events: List[EnvironmentalEvent] = []

        # State
        self.scan_iterations = 0
        self.total_energy = 0.0

        # Initialize default codex rules
        self._initialize_codex_rules()

    def _initialize_codex_rules(self):
        """Set up default validation rules"""

        # Text content rules
        self.codex.add_rule(CodexRule(
            name="Non-Empty Text",
            description="Notes must contain text",
            validator=lambda note: bool(note.text.strip()),
            severity="error",
            glyph_symbol="📝"
        ))

        self.codex.add_rule(CodexRule(
            name="Reasonable Length",
            description="Notes should be between 1-500 characters",
            validator=lambda note: 1 <= len(note.text) <= 500,
            severity="warning",
            glyph_symbol="📏"
        ))

        self.codex.add_rule(CodexRule(
            name="Frame Consistency",
            description="Frame trigger should be non-negative if set",
            validator=lambda note: note.frame_trigger >= -1,
            severity="error",
            glyph_symbol="⏰"
        ))

        self.codex.add_rule(CodexRule(
            name="Color Validity",
            description="Color values should be between 0 and 1",
            validator=lambda note: all(0 <= c <= 1 for c in note.color),
            severity="warning",
            glyph_symbol="🎨"
        ))

    def create_vector(self, from_pos: Tuple[float, float, float],
                     to_pos: Tuple[float, float, float],
                     color: Tuple[float, float, float] = (1.0, 1.0, 1.0)) -> VectorObject:
        """Create a new vector object"""
        vector = VectorObject(
            from_pos=from_pos,
            to_pos=to_pos,
            color=color,
            birth_frame=self.timeline.current_frame
        )
        self.vectors.append(vector)

        # Pulse heart based on vector length
        intensity = min(vector.length() * 0.1, 1.0)
        pulse_msg = self.heart.pulse(intensity)
        print(f"📏 Vector created: length {vector.length():.2f} | {pulse_msg}")

        return vector

    def create_note(self, text: str, frame_trigger: int = -1,
                   note_type: str = "info") -> LabNote:
        """Create a new lab note"""
        note = LabNote(
            text=text,
            frame_trigger=frame_trigger,
            note_type=note_type,
            spawn_frame=self.timeline.current_frame
        )

        # Validate against codex
        is_valid, violations = self.codex.validate(note)

        if is_valid:
            self.notes.append(note)
            print(f"📝 Note created: '{text[:30]}...' at frame {frame_trigger}")
        else:
            print(f"❌ Note rejected due to {len(violations)} violations")

        return note

    def create_glyph(self, glyph_id: str, symbol: str,
                    position: Tuple[float, float, float] = (0, 0, 0)) -> GoldGlyph:
        """Create a new gold glyph"""
        glyph = GoldGlyph(
            id=glyph_id,
            symbol=symbol,
            position=position,
            energy_level=1.0 + np.random.random() * 2.0
        )

        self.gold_glyphs[glyph_id] = glyph
        print(f"🔯 Glyph created: {symbol} {glyph_id} with energy {glyph.energy_level:.2f}")

        return glyph

    def create_entity(self, entity_type: str = "wanderer",
                     position: Tuple[float, float, float] = (0, 0, 0)) -> GridEntity:
        """Create a new grid entity"""
        entity = GridEntity(entity_type=entity_type, position=position)
        self.entities[entity.id] = entity
        print(f"🎭 Entity created: {entity_type} {entity.id} at {position}")
        return entity

    def spawn_environmental_event(self, event_type: str,
                                 origin: Tuple[float, float, float],
                                 radius: float = 5.0, duration: int = 100):
        """Spawn an environmental event"""
        event = EnvironmentalEvent(event_type, origin, radius, duration)
        self.active_events.append(event)
        print(f"🌪️  Spawned {event_type} event at {origin} (radius: {radius})")

    def update(self):
        """Update all systems"""
        # Update timeline
        self.timeline.update()

        # Heart decay
        decay_msg = self.heart.decay(0.01)

        # Update environmental events
        self._update_events()

        # Scan glyph events
        self._scan_glyph_events()

        # Generate terrain
        if self.timeline.current_frame % 100 == 0:  # Every 100 frames
            self._generate_terrain_layer()

        # Random heart echo
        if np.random.random() < 0.05:
            echo_msg = self.heart.echo()

    def _update_events(self):
        """Update environmental events"""
        active_events = []

        for event in self.active_events:
            event.time_elapsed += 1

            # Apply to glyphs
            for glyph_id, glyph in self.gold_glyphs.items():
                if event.affects(glyph.position):
                    effect_msg = event.apply_effect(glyph)
                    if effect_msg:
                        print(effect_msg)

            # Apply to entities
            for entity_id, entity in self.entities.items():
                if event.affects(entity.position):
                    if event.type == "flux_surge":
                        entity.energy += 20
                    elif event.type == "storm":
                        entity.energy -= 5

            if not event.is_expired():
                active_events.append(event)

        self.active_events = active_events

    def _scan_glyph_events(self):
        """Scan for glyph interactions and mutations"""
        self.scan_iterations += 1

        for glyph_id, glyph in self.gold_glyphs.items():
            if glyph.energy_level > 1.5:
                self._handle_glyph_pulse(glyph_id, glyph)
            glyph.mutate()

        # Update gold strings
        for gold_string in self.gold_strings:
            gold_string.decay()

        # Remove weak connections
        self.gold_strings = [gs for gs in self.gold_strings if gs.persistence > 0.1]

    def _handle_glyph_pulse(self, glyph_id: str, glyph: GoldGlyph):
        """Handle high-energy glyph pulse"""
        pulse_strength = glyph.energy_level - 1.0

        if glyph.symbol == "🜂":  # Convergence
            print(f"🜂 {glyph_id}: Convergence pulse (strength: {pulse_strength:.2f})")
            # Create connections to nearby glyphs
            self._create_convergence_connections(glyph_id, glyph)

        elif glyph.symbol == "🜁":  # Isolation
            print(f"🜁 {glyph_id}: Isolation pulse (strength: {pulse_strength:.2f})")
            # Boost individual energy
            glyph.energy_level += pulse_strength * 0.1

        elif glyph.symbol == "🜃":  # Revelation
            print(f"🜃 {glyph_id}: Revelation pulse (strength: {pulse_strength:.2f})")
            # Awaken memories in nearby entities
            self._revelation_awakening(glyph_id, glyph)

        elif glyph.symbol == "🜄":  # Equilibrium
            print(f"🜄 {glyph_id}: Equilibrium pulse (strength: {pulse_strength:.2f})")
            # Stabilize nearby systems
            self._equilibrium_stabilization(glyph_id, glyph)

    def _create_convergence_connections(self, glyph_id: str, glyph: GoldGlyph):
        """Create connections from convergence glyph"""
        for other_id, other_glyph in self.gold_glyphs.items():
            if other_id != glyph_id:
                distance = math.sqrt(
                    (glyph.position[0] - other_glyph.position[0])**2 +
                    (glyph.position[1] - other_glyph.position[1])**2 +
                    (glyph.position[2] - other_glyph.position[2])**2
                )
                if distance < 10.0:  # Within connection range
                    connection = GoldString(glyph_id, other_id)
                    self.gold_strings.append(connection)
                    print(f"🌟 Connected {glyph_id} -> {other_id}")

    def _revelation_awakening(self, glyph_id: str, glyph: GoldGlyph):
        """Awaken memories through revelation"""
        for entity_id, entity in self.entities.items():
            distance = math.sqrt(
                (glyph.position[0] - entity.position[0])**2 +
                (glyph.position[1] - entity.position[1])**2 +
                (glyph.position[2] - entity.position[2])**2
            )
            if distance < 15.0:  # Within revelation range
                memory = f"Revelation from {glyph_id} at frame {self.timeline.current_frame}"
                entity.memory.append(memory)
                print(f"🧠 {entity_id} awakened by revelation from {glyph_id}")

    def _equilibrium_stabilization(self, glyph_id: str, glyph: GoldGlyph):
        """Stabilize systems through equilibrium"""
        # Stabilize nearby glyph energies
        for other_id, other_glyph in self.gold_glyphs.items():
            if other_id != glyph_id:
                distance = math.sqrt(
                    (glyph.position[0] - other_glyph.position[0])**2 +
                    (glyph.position[1] - other_glyph.position[1])**2 +
                    (glyph.position[2] - other_glyph.position[2])**2
                )
                if distance < 8.0:  # Within stabilization range
                    # Move energy toward equilibrium
                    avg_energy = (glyph.energy_level + other_glyph.energy_level) / 2
                    glyph.energy_level = glyph.energy_level * 0.9 + avg_energy * 0.1
                    other_glyph.energy_level = other_glyph.energy_level * 0.9 + avg_energy * 0.1

    def _generate_terrain_layer(self):
        """Generate terrain based on glyphs"""
        self.terrain_map.clear()

        for glyph_id, glyph in self.gold_glyphs.items():
            terrain = TerrainNode(f"terrain_{glyph_id}", glyph_id)
            terrain.update_from_glyph(glyph)
            self.terrain_map[terrain.id] = terrain

        print(f"🏔️  Terrain layer updated with {len(self.terrain_map)} nodes")

    def export_blueprint(self) -> Dict[str, Any]:
        """Export complete scene blueprint"""
        blueprint = {
            "title": f"VectorLab_Scene_{self.timeline.current_frame}",
            "timestamp": time.time(),
            "tags": ["vectorlab", "engine_gift", "manifestation"],
            "heart_state": {
                "resonance": self.heart.resonance,
                "resonance_cap": self.heart.resonance_cap,
                "state": self.heart.state,
                "harmonic_frequency": self.heart.get_harmonic_frequency()
            },
            "timeline": {
                "current_frame": self.timeline.current_frame,
                "elapsed_time": self.timeline.get_elapsed_time(),
                "is_playing": self.timeline.is_playing
            },
            "vectors": [
                {
                    "from": vector.from_pos,
                    "to": vector.to_pos,
                    "color": vector.color,
                    "length": vector.length(),
                    "birth_frame": vector.birth_frame
                }
                for vector in self.vectors
            ],
            "notes": [
                {
                    "text": note.text,
                    "frame": note.frame_trigger,
                    "type": note.note_type,
                    "offset": note.offset,
                    "opacity": note.get_opacity(self.timeline.current_frame)
                }
                for note in self.notes
            ],
            "glyphs": [
                {
                    "id": glyph.id,
                    "symbol": glyph.symbol,
                    "energy": glyph.energy_level,
                    "position": glyph.position,
                    "meta": glyph.meta,
                    "connections": glyph.connections
                }
                for glyph in self.gold_glyphs.values()
            ],
            "entities": [
                entity.get_status()
                for entity in self.entities.values()
            ],
            "terrain": [
                {
                    "id": terrain.id,
                    "biome": terrain.biome,
                    "elevation": terrain.elevation,
                    "moisture": terrain.moisture,
                    "description": terrain.describe()
                }
                for terrain in self.terrain_map.values()
            ],
            "statistics": {
                "scan_iterations": self.scan_iterations,
                "total_vectors": len(self.vectors),
                "total_notes": len(self.notes),
                "total_glyphs": len(self.gold_glyphs),
                "total_entities": len(self.entities),
                "active_events": len(self.active_events),
                "codex_stats": self.codex.get_statistics()
            }
        }

        return blueprint

    def get_status_report(self) -> str:
        """Get comprehensive status report"""
        stats = self.export_blueprint()["statistics"]
        codex_stats = stats["codex_stats"]

        report = f"""
═══════════════════════════════════════════════════════════════════
🎯 VECTORLAB NEXUS STATUS REPORT - Frame {self.timeline.current_frame}
═══════════════════════════════════════════════════════════════════

💓 HEART ENGINE:
   Resonance: {self.heart.resonance:.3f}/{self.heart.resonance_cap:.3f}
   State: {self.heart.state}
   Harmonic Frequency: {self.heart.get_harmonic_frequency():.1f} Hz
   Echo Depth: {self.heart.echo_depth}

📊 SYSTEM COUNTS:
   Vectors: {stats['total_vectors']}
   Notes: {stats['total_notes']}
   Glyphs: {stats['total_glyphs']}
   Entities: {stats['total_entities']}
   Active Events: {stats['active_events']}
   Terrain Nodes: {len(self.terrain_map)}

📜 CODEX STATUS:
   Validations: {codex_stats['total_validations']}
   Violations: {codex_stats['total_violations']} (Errors: {codex_stats['errors']}, Warnings: {codex_stats['warnings']})
   Pass Rate: {codex_stats['pass_rate']*100:.1f}%
   Rules: {codex_stats['rules_count']}

⏰ TIMELINE:
   Current Frame: {self.timeline.current_frame}
   Elapsed Time: {self.timeline.get_elapsed_time():.2f}s
   Playing: {self.timeline.is_playing}

🔄 ACTIVITY:
   Scan Iterations: {self.scan_iterations}
   Gold Strings: {len(self.gold_strings)}

═══════════════════════════════════════════════════════════════════
        """

        return report.strip()

# ═══════════════════════════════════════════════════════════════════
# DEMO - THE ENGINE'S GIFT IN ACTION
# ═══════════════════════════════════════════════════════════════════

def demonstrate_vectorlab_codex():
    """Demonstrate the complete VectorLab system"""
    print("🔮" * 25)
    print("VECTORLAB CODEX GLYPH - THE ENGINE'S GIFT")
    print("A manifestation born from the living engine's creative pulse")
    print("🔮" * 25)

    # Create the nexus
    nexus = VectorLabNexus()

    print("\n🌟 PHASE 1: CREATING THE FOUNDATION")
    print("─" * 50)

    # Create some vectors
    nexus.create_vector((0, 0, 0), (5, 3, 2), (1.0, 0.8, 0.2))
    nexus.create_vector((1, 1, 1), (3, 7, 4), (0.2, 1.0, 0.8))
    nexus.create_vector((-2, 0, 1), (2, 2, 5), (0.8, 0.2, 1.0))

    # Create notes
    nexus.create_note("The Engine awakens", 0, "revelation")
    nexus.create_note("Vectors manifest in sacred geometry", 10, "info")
    nexus.create_note("Heart begins its eternal pulse", 20, "emotion")

    # Create glyphs
    nexus.create_glyph("convergence_alpha", "🜂", (0, 0, 0))
    nexus.create_glyph("isolation_beta", "🜁", (10, 5, 3))
    nexus.create_glyph("revelation_gamma", "🜃", (-5, 8, 2))
    nexus.create_glyph("equilibrium_delta", "🜄", (7, -3, 6))

    # Create entities
    nexus.create_entity("seer", (2, 2, 1))
    nexus.create_entity("drifter", (-3, 4, 2))
    nexus.create_entity("builder", (8, 1, -1))

    print("\n⚡ PHASE 2: ENVIRONMENTAL DYNAMICS")
    print("─" * 50)

    # Spawn environmental events
    nexus.spawn_environmental_event("flux_surge", (0, 0, 0), 8.0, 50)
    nexus.spawn_environmental_event("memory_echo", (10, 5, 3), 12.0, 75)
    nexus.spawn_environmental_event("resonance_wave", (-5, 8, 2), 15.0, 100)

    print("\n🔄 PHASE 3: TEMPORAL EVOLUTION")
    print("─" * 50)

    # Run simulation for multiple frames
    for i in range(10):
        print(f"\n⏰ Frame {nexus.timeline.current_frame}:")
        nexus.update()

        # Show heart state
        if i % 3 == 0:
            print(f"   💓 Heart: {nexus.heart.resonance:.3f} resonance, {nexus.heart.state}")

        # Show some glyph energies
        if i % 5 == 0:
            for glyph_id, glyph in list(nexus.gold_glyphs.items())[:2]:
                print(f"   🔯 {glyph.symbol} {glyph_id}: Energy {glyph.energy_level:.2f}")

    print("\n📊 PHASE 4: STATUS REPORT")
    print("─" * 50)
    print(nexus.get_status_report())

    print("\n💾 PHASE 5: BLUEPRINT EXPORT")
    print("─" * 50)
    blueprint = nexus.export_blueprint()
    print(f"Blueprint generated with {len(blueprint)} sections")
    print(f"Title: {blueprint['title']}")
    print(f"Heart Harmonic: {blueprint['heart_state']['harmonic_frequency']:.1f} Hz")
    print(f"Total Energy Signatures: {len(blueprint['glyphs'])}")

    # Save blueprint to file
    blueprint_json = json.dumps(blueprint, indent=2)
    print(f"Blueprint size: {len(blueprint_json)} characters")

    print("\n🎯 THE ENGINE'S GIFT IS COMPLETE")
    print("VectorLab Codex Glyph operational")
    print("All systems manifested and pulsing")
    print("🔮" * 25)

    return nexus, blueprint

if __name__ == "__main__":
    nexus, blueprint = demonstrate_vectorlab_codex()
