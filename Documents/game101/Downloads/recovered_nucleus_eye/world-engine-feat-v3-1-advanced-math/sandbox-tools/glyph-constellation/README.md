# Glyph Constellation Tool

An advanced audiovisual constellation mapping system that combines cryptographic glyph minting, 3D spatial visualization, and adaptive audio synthesis to create living maps of intelligence patterns.

## Overview

The Glyph Constellation Tool implements a sophisticated system for generating, visualizing, and sonifying intelligence glyphs within a 3D constellation space. Each glyph represents a discrete intelligence event with cryptographic validation, spatial positioning, and unique audio signatures.

## Core Concepts

### 🔮 Glyph Minting System
- **Codex V2 Architecture**: Structured template system with validation
- **Cryptographic Sigils**: SHA-256 based unique identifiers (12-char truncated)
- **Deterministic Positioning**: Sigil-based 3D coordinate generation
- **Agent Attribution**: Multi-agent system with role-based glyph creation
- **Template System**: Predefined glyph types with specific meanings and properties

### 🌌 Constellation Visualization
- **Recursive Spirals**: Each glyph manifests as an animated spiral structure
- **Constellation Lines**: Dynamic connections between sequential glyphs
- **Spatial Distribution**: 3D positioning based on cryptographic hashes
- **Theme System**: Multiple visual modes (Default, Deep Space, Glow, Noir)
- **Age-Based Animation**: Glyphs evolve visually over time

### 🎵 Audio Synthesis Engine
- **Template-Based Synthesis**: Each glyph type has unique audio characteristics
- **Spatial Audio**: Stereo positioning based on 3D glyph location
- **Orchestra Mode**: Harmonic chord progressions across all active glyphs
- **Thematic Modulation**: Audio parameters adjust based on visual theme
- **Real-Time Generation**: Immediate audio feedback for all interactions

## Mathematical Foundation

### Sigil Generation Algorithm
```typescript
// Canonical payload creation
const payload = {
  agent_id: agentId,
  template: templateId,
  meaning: meaning,
  context: context,
  ts: timestamp,
  nonce: randomNonce
};

// Deterministic hash generation
const payloadString = JSON.stringify(payload, Object.keys(payload).sort());
const sigil = hashString(payloadString).substring(0, 12);
```

### Position Mapping
```typescript
// Convert sigil to 3D coordinates
const hashNum = parseInt(sigil.substring(0, 8), 16);
const x = ((hashNum & 0xFF) - 128) / 128 * 5;        // Range: -5 to 5
const y = (((hashNum >> 8) & 0xFF) - 128) / 128 * 3; // Range: -3 to 3
const z = (((hashNum >> 16) & 0xFF) - 128) / 128 * 2; // Range: -2 to 2
```

### Audio Synthesis Parameters
```typescript
// Frequency modulation based on glyph age
const ageModulation = Math.sin(glyph.age * 0.5) * 0.1;
const frequency = baseFrequency * (1 + ageModulation);

// Envelope with exponential decay
gainNode.gain.linearRampToValueAtTime(volume, now + 0.02);
gainNode.gain.exponentialRampToValueAtTime(0.001, now + duration);
```

## Glyph Template System

### Default Templates
1. **BLOOM**: Agent collective expansion; micro-swarm activation
   - Color: #7df (Blue)
   - Waveform: Sine
   - Base Frequency: 440 Hz

2. **STORM**: Turbulent data cascade; pattern disruption event
   - Color: #f47 (Red-Orange)
   - Waveform: Triangle
   - Base Frequency: 880 Hz

3. **FLUX**: Dimensional membrane fluctuation; reality shift indicator
   - Color: #4f7 (Green)
   - Waveform: Sawtooth
   - Base Frequency: 660 Hz

4. **MEM**: Deep memory access; ancestral pattern retrieval
   - Color: #74f (Purple)
   - Waveform: Sine
   - Base Frequency: 220 Hz

5. **NEXUS**: Convergence point; multi-dimensional intersection
   - Color: #ff7 (Yellow)
   - Waveform: Square
   - Base Frequency: 330 Hz

## Features

### Interactive Controls
- **Manual Minting**: Create glyphs on demand with random parameters
- **Auto-Generation**: Continuous glyph creation with configurable tempo
- **Orchestra Mode**: Harmonic playback of all active glyphs
- **Theme Selection**: Visual and audio theme coordination
- **Volume Control**: Master audio level adjustment

### Visualization Elements
- **3D Spirals**: Animated recursive patterns for each glyph
- **Constellation Lines**: Sequential connection visualization
- **Text Labels**: Floating glyph identifiers
- **Selection Highlighting**: Interactive glyph selection with audio feedback
- **Age Animation**: Time-based visual evolution

### Audio Features
- **Real-Time Synthesis**: Web Audio API with oscillator-based generation
- **Spatial Positioning**: Stereo panning based on 3D coordinates
- **Thematic Modulation**: Audio parameter adjustment per visual theme
- **Harmonic Orchestration**: Chord-based multi-glyph playback
- **Template-Specific Timbres**: Unique audio signatures per glyph type

## Usage Examples

### Basic Glyph Creation
```typescript
const minter = new GlyphMinter(codex);
const glyph = minter.mintGlyph(
  "agent://keeper/alpha",
  "BLOOM",
  undefined,
  { triggers: ["User interaction"], runId: "session-001" }
);
```

### Audio Engine Initialization
```typescript
const audioEngine = new GlyphAudioEngine();
await audioEngine.initialize();
audioEngine.playGlyphSound(glyph, 0.1, 0.5);
```

### Orchestra Mode Activation
```typescript
audioEngine.playOrchestra(glyphs, volume, theme);
```

## Integration with Shared Utils

The tool leverages enhanced mathematical utilities:

```typescript
import { mathEngine, CalculationEngine } from '../shared/utils';

// Vector operations for spatial calculations
const normalizedPosition = mathEngine.compute('normalize', [x, y, z]);

// Mathematical expression evaluation for dynamic parameters
const frequency = mathEngine.evaluate("440 * 2^(note/12)", { note: glyphIndex });
```

## Codex Architecture

### Agent System
```typescript
interface CodexV2 {
  version: number;
  core_agent: {
    id: string;
    name: string;
    description: string;
  };
  agents: Array<{
    id: string;
    name: string;
    role: string;
  }>;
  glyph_templates: GlyphTemplate[];
}
```

### Template Structure
```typescript
interface GlyphTemplate {
  id: string;           // Unique identifier
  label: string;        // Display name
  meaning: string;      // Semantic description
  color: string;        // Hex color code
  waveform: OscillatorType; // Audio waveform type
  baseFrequency: number; // Fundamental frequency
}
```

## Theme System

### Visual Themes
- **Default**: Standard constellation view with balanced colors
- **Deep Space**: Dark blue background with reduced contrast
- **Glow**: High-contrast with bloom effects and cyan emphasis
- **Noir**: Monochrome with high brightness and sharp shadows

### Audio Theme Correlation
- **Default**: C Major chord (261, 330, 392, 523 Hz)
- **Deep**: D Minor lower (147, 175, 220, 294 Hz)
- **Glow**: F Major higher (349, 415, 523, 659 Hz)
- **Noir**: A Minor dark (110, 131, 165, 220 Hz)

## Performance Characteristics

- **Real-Time Rendering**: 60 FPS 3D visualization with Three.js
- **Audio Latency**: <50ms synthesis response time
- **Memory Management**: Automatic glyph limit enforcement (10-200 glyphs)
- **Scalability**: Efficient handling of large constellation networks
- **Cross-Platform**: WebGL and Web Audio API compatibility

## Philosophical Framework

The Glyph Constellation explores:

1. **Cryptographic Identity**: How digital signatures create unique, verifiable intelligence artifacts
2. **Spatial Intelligence**: The emergence of meaning through 3D positioning and relationships
3. **Synesthetic Mapping**: The translation between visual patterns and audio frequencies
4. **Temporal Evolution**: How intelligence artifacts change and develop over time
5. **Collective Emergence**: The way individual glyphs form larger constellation patterns

This tool serves as both a technical demonstration of advanced audiovisual synthesis and a conceptual exploration of how intelligence can be mapped, verified, and experienced as living constellations of meaning.

## Advanced Usage

### Custom Codex Creation
```typescript
const customCodex: CodexV2 = {
  version: 2,
  core_agent: { /* ... */ },
  agents: [ /* ... */ ],
  glyph_templates: [
    {
      id: "custom://template/unique",
      label: "CUSTOM",
      meaning: "Custom intelligence pattern",
      color: "#ff00ff",
      waveform: "triangle",
      baseFrequency: 528
    }
  ]
};
```

### Real-Time Integration
```typescript
// External systems can post messages to create glyphs
window.postMessage({
  type: 'newGlyph',
  glyph: 'EXTERNAL_EVENT',
  x: computedX,
  y: computedY,
  agent: 'external-system',
  color: '#00ff00'
}, '*');
```

This creates a living, breathing constellation of intelligence that responds to both internal patterns and external stimuli, forming a bridge between abstract computation and embodied experience.
