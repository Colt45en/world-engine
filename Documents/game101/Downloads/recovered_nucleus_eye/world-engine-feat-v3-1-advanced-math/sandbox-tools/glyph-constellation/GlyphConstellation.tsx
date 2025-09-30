import React, { useState, useCallback, useRef, useEffect } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, Line, Text } from '@react-three/drei';
import { useControls, button, folder } from 'leva';
import * as THREE from 'three';
import { mathEngine } from '../shared/utils';

// =============================================================================
// Core Types and Interfaces
// =============================================================================

interface GlyphTemplate {
  id: string;
  label: string;
  meaning: string;
  color: string;
  waveform: OscillatorType;
  baseFrequency: number;
}

interface GlyphContext {
  triggers: string[];
  runId: string;
  timestamp: string;
  [key: string]: any;
}

interface Glyph {
  glyphId: string;
  sigil: string;
  agentId: string;
  templateId: string;
  label: string;
  meaning: string;
  context: GlyphContext;
  timestampUTC: string;
  position: THREE.Vector3;
  screenPosition?: THREE.Vector2;
  age: number;
  spiralPhase: number;
}

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

// =============================================================================
// Audio Synthesis Engine
// =============================================================================

class GlyphAudioEngine {
  private audioContext: AudioContext;
  private masterGain: GainNode;
  private isInitialized = false;

  constructor() {
    this.audioContext = new (window.AudioContext || (window as any).webkitAudioContext)();
    this.masterGain = this.audioContext.createGain();
    this.masterGain.connect(this.audioContext.destination);
  }

  async initialize(): Promise<void> {
    if (this.audioContext.state !== 'running') {
      await this.audioContext.resume();
    }
    this.isInitialized = true;
  }

  playGlyphSound(glyph: Glyph, volume: number = 0.1, duration: number = 0.5): void {
    if (!this.isInitialized) return;

    const now = this.audioContext.currentTime;
    const oscillator = this.audioContext.createOscillator();
    const gainNode = this.audioContext.createGain();
    const filter = this.audioContext.createBiquadFilter();

    // Get template-based properties
    const template = this.getTemplateForGlyph(glyph);
    oscillator.type = template.waveform;
    oscillator.frequency.setValueAtTime(template.baseFrequency, now);

    // Apply age-based frequency modulation
    const ageModulation = Math.sin(glyph.age * 0.5) * 0.1;
    oscillator.frequency.exponentialRampToValueAtTime(
      template.baseFrequency * (1 + ageModulation),
      now + duration
    );

    // Configure filter
    filter.type = 'lowpass';
    filter.frequency.setValueAtTime(template.baseFrequency * 2, now);
    filter.Q.setValueAtTime(5, now);

    // Configure envelope
    gainNode.gain.setValueAtTime(0, now);
    gainNode.gain.linearRampToValueAtTime(volume, now + 0.02);
    gainNode.gain.exponentialRampToValueAtTime(0.001, now + duration);

    // Connect and play
    oscillator.connect(filter);
    filter.connect(gainNode);
    gainNode.connect(this.masterGain);

    oscillator.start(now);
    oscillator.stop(now + duration);
  }

  playOrchestra(glyphs: Glyph[], volume: number, theme: string): void {
    if (!this.isInitialized || glyphs.length === 0) return;

    const now = this.audioContext.currentTime;
    const chord = this.generateThematicChord(theme);

    glyphs.forEach((glyph, index) => {
      const template = this.getTemplateForGlyph(glyph);
      const delay = (index / glyphs.length) * 0.2; // Stagger notes
      const frequency = chord[index % chord.length];

      const oscillator = this.audioContext.createOscillator();
      const gainNode = this.audioContext.createGain();
      const panner = this.audioContext.createStereoPanner();

      oscillator.type = template.waveform;
      oscillator.frequency.setValueAtTime(frequency, now + delay);

      // Stereo positioning based on glyph position
      const panValue = Math.max(-1, Math.min(1, glyph.position.x / 10));
      panner.pan.setValueAtTime(panValue, now + delay);

      gainNode.gain.setValueAtTime(0, now + delay);
      gainNode.gain.linearRampToValueAtTime(volume / glyphs.length, now + delay + 0.05);
      gainNode.gain.exponentialRampToValueAtTime(0.001, now + delay + 1.0);

      oscillator.connect(panner);
      panner.connect(gainNode);
      gainNode.connect(this.masterGain);

      oscillator.start(now + delay);
      oscillator.stop(now + delay + 1.0);
    });
  }

  private getTemplateForGlyph(glyph: Glyph): GlyphTemplate {
    // Default template if not found in codex
    return {
      id: glyph.templateId,
      label: glyph.label,
      meaning: glyph.meaning,
      color: '#7df',
      waveform: glyph.label.includes('STORM') ? 'triangle' :
               glyph.label.includes('FLUX') ? 'sawtooth' :
               glyph.label.includes('MEM') ? 'sine' : 'sine',
      baseFrequency: glyph.label.includes('STORM') ? 880 :
                    glyph.label.includes('FLUX') ? 660 :
                    glyph.label.includes('MEM') ? 220 : 440
    };
  }

  private generateThematicChord(theme: string): number[] {
    const baseChords = {
      default: [261.63, 329.63, 392.00, 523.25], // C major
      deep: [146.83, 174.61, 220.00, 293.66], // D minor (lower)
      glow: [349.23, 415.30, 523.25, 659.25], // F major (higher)
      noir: [110.00, 130.81, 164.81, 220.00], // A minor (dark)
    };
    return baseChords[theme as keyof typeof baseChords] || baseChords.default;
  }

  setMasterVolume(volume: number): void {
    this.masterGain.gain.setValueAtTime(volume, this.audioContext.currentTime);
  }
}

// =============================================================================
// Glyph Minting System
// =============================================================================

class GlyphMinter {
  private codex: CodexV2;

  constructor(codex: CodexV2) {
    this.codex = codex;
    this.validateCodex();
  }

  private validateCodex(): void {
    if (this.codex.version !== 2) {
      throw new Error('Codex version must be 2');
    }
    if (!this.codex.core_agent || !this.codex.agents) {
      throw new Error('Missing core_agent/agents');
    }
  }

  mintGlyph(
    agentId: string,
    templateLabel: string,
    meaningOverride?: string,
    context: Partial<GlyphContext> = {}
  ): Glyph {
    // Find template
    const template = this.codex.glyph_templates.find(t => t.label === templateLabel);
    if (!template) {
      throw new Error(`Unknown glyph template '${templateLabel}'`);
    }

    const meaning = meaningOverride || template.meaning;
    const timestamp = new Date().toISOString();

    // Create canonical payload for hashing
    const payload = {
      agent_id: agentId,
      template: template.id,
      meaning,
      context: {
        triggers: [],
        runId: `r-${new Date().toISOString().split('T')[0]}-001`,
        timestamp,
        ...context
      },
      ts: timestamp,
      nonce: Math.random().toString(36).substring(2, 10)
    };

    // Generate sigil hash
    const payloadString = JSON.stringify(payload, Object.keys(payload).sort());
    const sigil = this.hashString(payloadString).substring(0, 12);

    // Generate position based on sigil
    const position = this.generatePosition(sigil);

    return {
      glyphId: `glyph://mint/${sigil}`,
      sigil,
      agentId,
      templateId: template.id,
      label: templateLabel,
      meaning,
      context: payload.context as GlyphContext,
      timestampUTC: timestamp,
      position,
      age: 0,
      spiralPhase: Math.random() * Math.PI * 2
    };
  }

  private hashString(input: string): string {
    let hash = 0;
    for (let i = 0; i < input.length; i++) {
      const char = input.charCodeAt(i);
      hash = ((hash << 5) - hash) + char;
      hash = hash & hash; // Convert to 32-bit integer
    }
    return Math.abs(hash).toString(16);
  }

  private generatePosition(sigil: string): THREE.Vector3 {
    // Use sigil to generate deterministic position
    const hashNum = parseInt(sigil.substring(0, 8), 16);
    const x = ((hashNum & 0xFF) - 128) / 128 * 5; // -5 to 5
    const y = (((hashNum >> 8) & 0xFF) - 128) / 128 * 3; // -3 to 3
    const z = (((hashNum >> 16) & 0xFF) - 128) / 128 * 2; // -2 to 2
    return new THREE.Vector3(x, y, z);
  }

  getAvailableTemplates(): GlyphTemplate[] {
    return this.codex.glyph_templates;
  }

  getAvailableAgents(): Array<{ id: string; name: string; role: string }> {
    return this.codex.agents;
  }
}

// =============================================================================
// 3D Visualization Components
// =============================================================================

interface RecursiveSpiralProps {
  position: THREE.Vector3;
  color: string;
  age: number;
  phase: number;
  scale: number;
}

function RecursiveSpiral({ position, color, age, phase, scale }: RecursiveSpiralProps) {
  const spiralRef = useRef<THREE.Group>(null);

  useFrame((state) => {
    if (spiralRef.current) {
      spiralRef.current.rotation.y = state.clock.elapsedTime * 0.5 + phase;
      spiralRef.current.rotation.x = Math.sin(state.clock.elapsedTime * 0.3 + age) * 0.2;
    }
  });

  const spiralPoints = React.useMemo(() => {
    const points: THREE.Vector3[] = [];
    const loops = 3;
    for (let a = 0; a < Math.PI * 2 * loops; a += 0.1) {
      const ease = 0.5 + 0.5 * Math.sin(a * 3 + age);
      const r = a * 0.1 * ease * scale;
      const x = r * Math.cos(a);
      const y = a * 0.02 - loops * 0.1;
      const z = r * Math.sin(a);
      points.push(new THREE.Vector3(x, y, z));
    }
    return points;
  }, [age, scale]);

  return (
    <group ref={spiralRef} position={position}>
      <Line
        points={spiralPoints}
        color={color}
        lineWidth={2}
        transparent
        opacity={0.7}
      />
      <mesh>
        <sphereGeometry args={[0.1, 8, 8]} />
        <meshStandardMaterial
          color={color}
          emissive={color}
          emissiveIntensity={0.3}
          transparent
          opacity={0.8}
        />
      </mesh>
    </group>
  );
}

interface ConstellationLinesProps {
  glyphs: Glyph[];
}

function ConstellationLines({ glyphs }: ConstellationLinesProps) {
  const linePoints = React.useMemo(() => {
    const points: THREE.Vector3[] = [];
    for (let i = 1; i < glyphs.length; i++) {
      points.push(glyphs[i - 1].position, glyphs[i].position);
    }
    return points;
  }, [glyphs]);

  if (linePoints.length < 2) return null;

  return (
    <Line
      points={linePoints}
      color="rgba(255,255,255,0.2)"
      lineWidth={1}
      segments
      transparent
      opacity={0.3}
    />
  );
}

interface GlyphVisualizationProps {
  glyphs: Glyph[];
  selectedGlyph: Glyph | null;
  onGlyphSelect: (glyph: Glyph) => void;
  theme: string;
}

function GlyphVisualization({ glyphs, selectedGlyph, onGlyphSelect, theme }: GlyphVisualizationProps) {
  useFrame((state) => {
    // Update glyph ages
    glyphs.forEach(glyph => {
      glyph.age = (Date.now() - new Date(glyph.timestampUTC).getTime()) / 1000;
    });
  });

  const themeColors = {
    default: '#7df',
    deep: '#4a90e2',
    glow: '#00ffff',
    noir: '#ffffff'
  };

  return (
    <group>
      {glyphs.map((glyph) => {
        const color = themeColors[theme as keyof typeof themeColors] || themeColors.default;
        const isSelected = selectedGlyph?.glyphId === glyph.glyphId;
        const scale = isSelected ? 1.5 : 1.0;

        return (
          <group key={glyph.glyphId}>
            <RecursiveSpiral
              position={glyph.position}
              color={color}
              age={glyph.age}
              phase={glyph.spiralPhase}
              scale={scale}
            />
            <Text
              position={[glyph.position.x, glyph.position.y + 0.5, glyph.position.z]}
              fontSize={0.2}
              color={color}
              anchorX="center"
              anchorY="middle"
            >
              {glyph.label}
            </Text>
            {/* Invisible clickable sphere */}
            <mesh
              position={glyph.position}
              onClick={() => onGlyphSelect(glyph)}
              visible={false}
            >
              <sphereGeometry args={[0.3]} />
            </mesh>
          </group>
        );
      })}
      <ConstellationLines glyphs={glyphs} />
      <ambientLight intensity={0.4} />
      <pointLight position={[10, 10, 10]} intensity={0.8} />
      <pointLight position={[-10, -5, -10]} intensity={0.3} color="#4169e1" />
    </group>
  );
}

// =============================================================================
// Main Component
// =============================================================================

export default function GlyphConstellation() {
  // Default codex configuration
  const defaultCodex: CodexV2 = {
    version: 2,
    core_agent: {
      id: "agent://core/prime-constellation",
      name: "Prime Constellation",
      description: "Core orchestrator of glyph constellation mapping"
    },
    agents: [
      { id: "agent://keeper/alpha", name: "Alpha Keeper", role: "Pattern Recognition" },
      { id: "agent://keeper/beta", name: "Beta Keeper", role: "Memory Synthesis" },
      { id: "agent://keeper/gamma", name: "Gamma Keeper", role: "Harmonic Resonance" }
    ],
    glyph_templates: [
      { id: "glyph://rsc/bloom", label: "BLOOM", meaning: "Agent collective expansion; micro-swarm activation.", color: "#7df", waveform: "sine", baseFrequency: 440 },
      { id: "glyph://rsc/storm", label: "STORM", meaning: "Turbulent data cascade; pattern disruption event.", color: "#f47", waveform: "triangle", baseFrequency: 880 },
      { id: "glyph://rsc/flux", label: "FLUX", meaning: "Dimensional membrane fluctuation; reality shift indicator.", color: "#4f7", waveform: "sawtooth", baseFrequency: 660 },
      { id: "glyph://rsc/memory", label: "MEM", meaning: "Deep memory access; ancestral pattern retrieval.", color: "#74f", waveform: "sine", baseFrequency: 220 },
      { id: "glyph://rsc/nexus", label: "NEXUS", meaning: "Convergence point; multi-dimensional intersection.", color: "#ff7", waveform: "square", baseFrequency: 330 }
    ]
  };

  const [glyphs, setGlyphs] = useState<Glyph[]>([]);
  const [selectedGlyph, setSelectedGlyph] = useState<Glyph | null>(null);
  const [minter] = useState(() => new GlyphMinter(defaultCodex));
  const [audioEngine] = useState(() => new GlyphAudioEngine());
  const [isAudioInitialized, setIsAudioInitialized] = useState(false);

  const {
    volume,
    theme,
    orchestraMode,
    tempo,
    autoGenerate,
    maxGlyphs
  } = useControls('Glyph Constellation', {
    Audio: folder({
      volume: { value: 0.1, min: 0, max: 1, step: 0.01 },
      orchestraMode: false,
      tempo: { value: 1.0, min: 0.1, max: 3.0, step: 0.1 }
    }),
    Visual: folder({
      theme: { value: 'default', options: ['default', 'deep', 'glow', 'noir'] },
      maxGlyphs: { value: 50, min: 10, max: 200, step: 10 }
    }),
    Control: folder({
      autoGenerate: false,
      'Mint Glyph': button(() => mintRandomGlyph()),
      'Play Orchestra': button(() => playOrchestra()),
      'Clear Glyphs': button(() => setGlyphs([])),
      'Initialize Audio': button(async () => {
        await audioEngine.initialize();
        setIsAudioInitialized(true);
      })
    })
  });

  // Initialize audio on first user interaction
  useEffect(() => {
    const initAudio = async () => {
      if (!isAudioInitialized) {
        await audioEngine.initialize();
        setIsAudioInitialized(true);
      }
    };

    const handleFirstClick = () => {
      initAudio();
      document.removeEventListener('click', handleFirstClick);
    };

    document.addEventListener('click', handleFirstClick);
    return () => document.removeEventListener('click', handleFirstClick);
  }, [audioEngine, isAudioInitialized]);

  // Auto-generation
  useEffect(() => {
    if (!autoGenerate) return;

    const interval = setInterval(() => {
      mintRandomGlyph();
    }, tempo * 1000);

    return () => clearInterval(interval);
  }, [autoGenerate, tempo]);

  // Limit glyph count
  useEffect(() => {
    if (glyphs.length > maxGlyphs) {
      setGlyphs(prev => prev.slice(-maxGlyphs));
    }
  }, [glyphs.length, maxGlyphs]);

  const mintRandomGlyph = useCallback(() => {
    const templates = minter.getAvailableTemplates();
    const agents = minter.getAvailableAgents();

    const randomTemplate = templates[Math.floor(Math.random() * templates.length)];
    const randomAgent = agents[Math.floor(Math.random() * agents.length)];

    const context = {
      triggers: [`Random trigger ${Date.now()}`],
      runId: `r-${new Date().toISOString().split('T')[0]}-${Math.floor(Math.random() * 1000).toString().padStart(3, '0')}`,
      timestamp: new Date().toISOString()
    };

    try {
      const newGlyph = minter.mintGlyph(
        randomAgent.id,
        randomTemplate.label,
        undefined,
        context
      );

      setGlyphs(prev => [...prev, newGlyph]);

      // Play sound
      if (isAudioInitialized) {
        audioEngine.playGlyphSound(newGlyph, volume);
      }
    } catch (error) {
      console.error('Error minting glyph:', error);
    }
  }, [minter, audioEngine, volume, isAudioInitialized]);

  const playOrchestra = useCallback(() => {
    if (isAudioInitialized && glyphs.length > 0) {
      audioEngine.playOrchestra(glyphs, volume, theme);
    }
  }, [audioEngine, glyphs, volume, theme, isAudioInitialized]);

  const handleGlyphSelect = useCallback((glyph: Glyph) => {
    setSelectedGlyph(glyph);
    if (isAudioInitialized) {
      audioEngine.playGlyphSound(glyph, volume * 1.5);
    }
  }, [audioEngine, volume, isAudioInitialized]);

  // Update audio engine volume
  useEffect(() => {
    if (isAudioInitialized) {
      audioEngine.setMasterVolume(volume);
    }
  }, [audioEngine, volume, isAudioInitialized]);

  return (
    <div style={{ width: '100%', height: '100vh', display: 'flex', flexDirection: 'column' }}>
      {/* 3D Visualization */}
      <div style={{ flex: 1, background: theme === 'deep' ? '#000814' : '#000' }}>
        <Canvas camera={{ position: [8, 6, 8], fov: 60 }}>
          <GlyphVisualization
            glyphs={glyphs}
            selectedGlyph={selectedGlyph}
            onGlyphSelect={handleGlyphSelect}
            theme={theme}
          />
          <OrbitControls enablePan={true} enableZoom={true} enableRotate={true} />
        </Canvas>
      </div>

      {/* Info Panel */}
      <div style={{
        height: '200px',
        background: '#1a1a1a',
        color: '#fff',
        padding: '20px',
        display: 'flex',
        gap: '20px',
        overflow: 'auto'
      }}>
        <div style={{ flex: 1 }}>
          <h3>🌌 Constellation Status</h3>
          <div>Active Glyphs: {glyphs.length}</div>
          <div>Audio: {isAudioInitialized ? '🔊 Ready' : '🔇 Click to Initialize'}</div>
          <div>Theme: {theme}</div>
          <div>Orchestra Mode: {orchestraMode ? 'ON' : 'OFF'}</div>
        </div>

        {selectedGlyph && (
          <div style={{ flex: 2 }}>
            <h3>📜 Selected Glyph</h3>
            <div style={{ fontSize: '12px', fontFamily: 'monospace' }}>
              <div><strong>Label:</strong> {selectedGlyph.label}</div>
              <div><strong>Sigil:</strong> {selectedGlyph.sigil}</div>
              <div><strong>Agent:</strong> {selectedGlyph.agentId}</div>
              <div><strong>Meaning:</strong> {selectedGlyph.meaning}</div>
              <div><strong>Age:</strong> {selectedGlyph.age.toFixed(1)}s</div>
              <div><strong>Position:</strong> ({selectedGlyph.position.x.toFixed(2)}, {selectedGlyph.position.y.toFixed(2)}, {selectedGlyph.position.z.toFixed(2)})</div>
            </div>
          </div>
        )}

        <div style={{ flex: 1 }}>
          <h3>🎵 Recent Activity</h3>
          <div style={{ maxHeight: '120px', overflow: 'auto', fontSize: '11px' }}>
            {glyphs.slice(-5).reverse().map(glyph => (
              <div key={glyph.glyphId} style={{
                marginBottom: '4px',
                padding: '4px',
                background: '#2a2a2a',
                cursor: 'pointer'
              }} onClick={() => setSelectedGlyph(glyph)}>
                <strong>{glyph.label}</strong> - {glyph.sigil}
                <br />
                <span style={{ color: '#aaa' }}>
                  {new Date(glyph.timestampUTC).toLocaleTimeString()}
                </span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
