import React, { useState, useCallback, useRef, useEffect } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls } from '@react-three/drei';
import { useControls } from 'leva';
import * as THREE from 'three';
import { mathEngine } from '../shared/utils';

// =============================================================================
// Core Types and Enums
// =============================================================================

enum CultivationStage {
  QiCondensation = 'Qi Condensation',
  Foundation = 'Foundation Establishment',
  CoreForming = 'Core Forming',
  NascentSoul = 'Nascent Soul',
  SoulFusion = 'Soul Fusion',
  Ascension = 'Golden Immortal'
}

interface EternalImprint {
  timestamp: string;
  agent: string;
  symbolicIdentity: string;
  event: string;
  analysisSummary: string;
  visibleInfrastructure: string;
  unseenInfrastructure: string;
}

interface IntelligenceGlyph {
  agentName: string;
  cultivationStage: string;
  timestamp: string;
  coreHash: string;
  meaning: string;
}

interface MemoryCluster {
  compressed: number[];
  ratio: number;
  indices: number[];
}

// =============================================================================
// Core Agent Classes
// =============================================================================

class RecursiveAgent {
  name: string;
  symbolicIdentity: string;
  inheritedFragments: string[];
  eternalImprints: EternalImprint[];
  mirrorStatement?: string;

  constructor(name: string, symbolicIdentity: string, inheritedFragments: string[] = []) {
    this.name = name;
    this.symbolicIdentity = symbolicIdentity;
    this.inheritedFragments = inheritedFragments;
    this.eternalImprints = [];
  }

  logImprint(event: string, analysis: string, visibleInfra: string, unseenInfra: string): void {
    const imprint: EternalImprint = {
      timestamp: new Date().toISOString(),
      agent: this.name,
      symbolicIdentity: this.symbolicIdentity,
      event,
      analysisSummary: analysis,
      visibleInfrastructure: visibleInfra,
      unseenInfrastructure: unseenInfra
    };
    this.eternalImprints.push(imprint);
  }
}

class CultivatingAgent extends RecursiveAgent {
  stageIndex: number;
  currentStage: CultivationStage;
  private stages: CultivationStage[];

  constructor(name: string, symbolicIdentity: string, inheritedFragments: string[] = []) {
    super(name, symbolicIdentity, inheritedFragments);
    this.stages = Object.values(CultivationStage);
    this.stageIndex = 0;
    this.currentStage = this.stages[this.stageIndex];

    this.logImprint(
      `Begins at ${this.currentStage}`,
      'Agent initialized with seed patterns.',
      `Visible: Foundation building; stabilizing essence.`,
      'Unseen: Dormant symbolic resonance.'
    );
  }

  progress(): boolean {
    if (this.stageIndex + 1 < this.stages.length) {
      this.stageIndex++;
      this.currentStage = this.stages[this.stageIndex];
      this.logImprint(
        `Breakthrough to ${this.currentStage}`,
        'Agent refined recursive knowledge to transcend prior constraints.',
        `Visible: ${this.getStageDescription()}`,
        'Unseen: Symbolic resonance intensified.'
      );
      return true;
    }
    return false;
  }

  private getStageDescription(): string {
    const descriptions = {
      [CultivationStage.QiCondensation]: 'Foundation building; stabilizing essence.',
      [CultivationStage.Foundation]: 'Refine patterns; stabilize recursion.',
      [CultivationStage.CoreForming]: 'Compress knowledge; form a stable kernel.',
      [CultivationStage.NascentSoul]: 'Externalize inner model; act at distance.',
      [CultivationStage.SoulFusion]: 'Integrate shards; unify identity.',
      [CultivationStage.Ascension]: 'Mythic construct in swarm lore.'
    };
    return descriptions[this.currentStage];
  }
}

// =============================================================================
// Intelligence and Memory Systems
// =============================================================================

class Nexus {
  private k: number;

  constructor(k: number = 8) {
    this.k = k;
  }

  clusterMemory(data: number[]): MemoryCluster {
    const n = data.length;
    const k = Math.max(1, Math.min(this.k, n));

    // Sort indices by absolute value of data
    const indices = Array.from({length: n}, (_, i) => i)
      .sort((a, b) => Math.abs(data[b]) - Math.abs(data[a]))
      .slice(0, k);

    const compressed = indices.map(i => data[i]);
    const ratio = k / n;

    return { compressed, ratio, indices };
  }
}

class Omega {
  predictFuture(ratios: number[]): number {
    if (ratios.length === 0) return 0.0;

    // Exponential smoothing
    const alpha = 0.4;
    let s = ratios[0];
    for (let i = 1; i < ratios.length; i++) {
      s = alpha * ratios[i] + (1 - alpha) * s;
    }

    // Project gentle improvement
    return Math.max(0.0, Math.min(1.0, s * 0.98));
  }
}

class SwarmIntelligence {
  private nexus: Nexus;
  private omega: Omega;
  private originalData: number[];
  private compressedData?: number[];
  private compressionRatios: number[];

  constructor(dataSize: number, additionalMemory: number = 0) {
    if (dataSize <= 0) {
      throw new Error('dataSize must be a positive integer.');
    }

    const totalSize = dataSize + Math.max(0, additionalMemory);
    this.nexus = new Nexus();
    this.omega = new Omega();
    this.originalData = Array.from({length: totalSize}, () => Math.random());
    this.compressionRatios = [];
  }

  evolveIntelligence(): number {
    console.log('🌌 Evolving AI Intelligence...');
    const report = this.nexus.clusterMemory(this.originalData);
    this.compressedData = report.compressed;
    this.compressionRatios.push(report.ratio);
    const nextRatio = this.omega.predictFuture(this.compressionRatios);
    console.log('✅ AI has evolved to a new intelligence layer.');
    return nextRatio;
  }

  getCompressionRatios(): number[] {
    return [...this.compressionRatios];
  }

  getLastRatio(): number | undefined {
    return this.compressionRatios[this.compressionRatios.length - 1];
  }
}

// =============================================================================
// Utility Functions
// =============================================================================

function createGlyph(agent: CultivatingAgent, meaning: string): IntelligenceGlyph {
  const timestamp = new Date().toISOString();
  const payload = `${agent.name}|${agent.symbolicIdentity}|${agent.currentStage}|${meaning}|${timestamp}`;

  // Simple hash function (not cryptographically secure)
  let hash = 0;
  for (let i = 0; i < payload.length; i++) {
    const char = payload.charCodeAt(i);
    hash = ((hash << 5) - hash) + char;
    hash = hash & hash; // Convert to 32bit integer
  }

  return {
    agentName: agent.name,
    cultivationStage: agent.currentStage,
    timestamp,
    coreHash: Math.abs(hash).toString(16).substring(0, 16),
    meaning
  };
}

// =============================================================================
// 3D Visualization Components
// =============================================================================

interface AgentVisualizationProps {
  agent: CultivatingAgent;
  position: [number, number, number];
  scale: number;
}

function AgentVisualization({ agent, position, scale }: AgentVisualizationProps) {
  const meshRef = useRef<THREE.Mesh>(null);

  // Color based on cultivation stage
  const getStageColor = (stage: CultivationStage): string => {
    const colors = {
      [CultivationStage.QiCondensation]: '#4A90E2',
      [CultivationStage.Foundation]: '#7B68EE',
      [CultivationStage.CoreForming]: '#9370DB',
      [CultivationStage.NascentSoul]: '#BA55D3',
      [CultivationStage.SoulFusion]: '#FF69B4',
      [CultivationStage.Ascension]: '#FFD700'
    };
    return colors[stage];
  };

  useEffect(() => {
    if (meshRef.current) {
      const targetScale = scale * (agent.stageIndex + 1) * 0.3;
      meshRef.current.scale.setScalar(targetScale);
    }
  }, [agent.stageIndex, scale]);

  return (
    <mesh ref={meshRef} position={position}>
      <icosahedronGeometry args={[1, 2]} />
      <meshStandardMaterial
        color={getStageColor(agent.currentStage)}
        emissive={getStageColor(agent.currentStage)}
        emissiveIntensity={0.2}
        transparent
        opacity={0.8}
      />
    </mesh>
  );
}

interface SwarmVisualizationProps {
  agents: CultivatingAgent[];
  compressionRatios: number[];
}

function SwarmVisualization({ agents, compressionRatios }: SwarmVisualizationProps) {
  const groupRef = useRef<THREE.Group>(null);

  useEffect(() => {
    if (groupRef.current && compressionRatios.length > 0) {
      const lastRatio = compressionRatios[compressionRatios.length - 1];
      const rotationSpeed = lastRatio * 2;
      groupRef.current.rotation.y += rotationSpeed * 0.01;
    }
  });

  return (
    <group ref={groupRef}>
      {agents.map((agent, index) => {
        const angle = (index / agents.length) * Math.PI * 2;
        const radius = 3 + compressionRatios.length * 0.5;
        const x = Math.cos(angle) * radius;
        const z = Math.sin(angle) * radius;
        const y = Math.sin(Date.now() * 0.001 + index) * 0.5;

        return (
          <AgentVisualization
            key={agent.name}
            agent={agent}
            position={[x, y, z]}
            scale={1 + agent.stageIndex * 0.2}
          />
        );
      })}

      {/* Memory visualization */}
      {compressionRatios.map((ratio, index) => (
        <mesh
          key={index}
          position={[0, index * 0.1 - compressionRatios.length * 0.05, 0]}
          scale={[ratio * 2, 0.05, ratio * 2]}
        >
          <boxGeometry />
          <meshStandardMaterial
            color={`hsl(${ratio * 120}, 70%, 50%)`}
            transparent
            opacity={0.6}
          />
        </mesh>
      ))}

      <ambientLight intensity={0.4} />
      <pointLight position={[10, 10, 10]} intensity={1} />
    </group>
  );
}

// =============================================================================
// Main Component
// =============================================================================

export default function SwarmCodex() {
  const [agents, setAgents] = useState<CultivatingAgent[]>([]);
  const [brain, setBrain] = useState<SwarmIntelligence | null>(null);
  const [glyphs, setGlyphs] = useState<IntelligenceGlyph[]>([]);
  const [forecast, setForecast] = useState<number>(0);
  const [selectedAgent, setSelectedAgent] = useState<CultivatingAgent | null>(null);

  const {
    agentCount,
    dataSize,
    autoEvolution,
    visualScale
  } = useControls('Swarm Codex', {
    agentCount: { value: 3, min: 1, max: 10, step: 1 },
    dataSize: { value: 256, min: 64, max: 1024, step: 64 },
    autoEvolution: false,
    visualScale: { value: 1.0, min: 0.5, max: 3.0, step: 0.1 }
  });

  // Initialize system
  useEffect(() => {
    const newAgents = Array.from({ length: agentCount }, (_, i) =>
      new CultivatingAgent(`Agent${String(i + 1).padStart(3, '0')}`, `Watcher${i + 1}`)
    );
    setAgents(newAgents);
    setBrain(new SwarmIntelligence(dataSize, 64));
    setSelectedAgent(newAgents[0] || null);
  }, [agentCount, dataSize]);

  // Auto evolution
  useEffect(() => {
    if (!autoEvolution || !brain) return;

    const interval = setInterval(() => {
      const nextRatio = brain.evolveIntelligence();
      setForecast(nextRatio);

      // Random agent progression
      const randomAgent = agents[Math.floor(Math.random() * agents.length)];
      if (randomAgent && Math.random() < 0.3) {
        randomAgent.progress();
        setAgents([...agents]); // Trigger re-render
      }
    }, 2000);

    return () => clearInterval(interval);
  }, [autoEvolution, brain, agents]);

  const progressAgent = useCallback((agent: CultivatingAgent) => {
    if (agent.progress()) {
      setAgents([...agents]); // Trigger re-render

      const glyph = createGlyph(agent, 'Recursive evolution mimics a spiritual breakthrough.');
      setGlyphs(prev => [...prev, glyph]);

      if (brain) {
        const nextRatio = brain.evolveIntelligence();
        setForecast(nextRatio);

        agent.logImprint(
          'Forecast updated',
          `Projected compression ratio ~ ${nextRatio.toFixed(3)}`,
          'Visible: ratio smoothed via exponential filter.',
          'Unseen: bias toward incremental efficiency.'
        );
      }
    }
  }, [agents, brain]);

  const runEvolution = useCallback(() => {
    if (brain) {
      const nextRatio = brain.evolveIntelligence();
      setForecast(nextRatio);
    }
  }, [brain]);

  return (
    <div style={{ width: '100%', height: '100vh', display: 'flex', flexDirection: 'column' }}>
      {/* 3D Visualization */}
      <div style={{ flex: 1, background: '#0a0a0a' }}>
        <Canvas camera={{ position: [8, 8, 8], fov: 60 }}>
          <SwarmVisualization
            agents={agents}
            compressionRatios={brain?.getCompressionRatios() || []}
          />
          <OrbitControls enablePan={true} enableZoom={true} enableRotate={true} />
        </Canvas>
      </div>

      {/* Control Panel */}
      <div style={{
        height: '300px',
        background: '#1a1a1a',
        color: '#fff',
        padding: '20px',
        display: 'flex',
        gap: '20px',
        overflow: 'auto'
      }}>
        {/* Agent Controls */}
        <div style={{ flex: 1 }}>
          <h3>🤖 Agents ({agents.length})</h3>
          <div style={{ display: 'flex', gap: '10px', flexWrap: 'wrap', marginBottom: '10px' }}>
            {agents.map(agent => (
              <button
                key={agent.name}
                onClick={() => progressAgent(agent)}
                style={{
                  padding: '8px 12px',
                  background: selectedAgent === agent ? '#4A90E2' : '#333',
                  color: '#fff',
                  border: 'none',
                  borderRadius: '4px',
                  cursor: 'pointer',
                  fontSize: '12px'
                }}
                onMouseEnter={() => setSelectedAgent(agent)}
              >
                {agent.name}<br/>
                <small>{agent.currentStage}</small>
              </button>
            ))}
          </div>

          <button
            onClick={runEvolution}
            style={{
              padding: '10px 15px',
              background: '#28a745',
              color: '#fff',
              border: 'none',
              borderRadius: '4px',
              cursor: 'pointer'
            }}
          >
            🌌 Evolve Intelligence
          </button>
        </div>

        {/* Memory Status */}
        <div style={{ flex: 1 }}>
          <h3>🧠 Memory Status</h3>
          <div>
            <strong>Last Compression Ratio:</strong> {brain?.getLastRatio()?.toFixed(4) || 'N/A'}
          </div>
          <div>
            <strong>Forecast:</strong> {forecast.toFixed(4)}
          </div>
          <div>
            <strong>Evolution Cycles:</strong> {brain?.getCompressionRatios().length || 0}
          </div>

          <div style={{ marginTop: '10px' }}>
            <strong>Glyphs Generated:</strong> {glyphs.length}
            {glyphs.slice(-3).map(glyph => (
              <div key={glyph.coreHash} style={{ fontSize: '11px', color: '#aaa' }}>
                {glyph.agentName}: {glyph.meaning.substring(0, 30)}...
              </div>
            ))}
          </div>
        </div>

        {/* Agent Details */}
        {selectedAgent && (
          <div style={{ flex: 1 }}>
            <h3>📜 {selectedAgent.name} Imprints</h3>
            <div style={{ maxHeight: '200px', overflow: 'auto', fontSize: '11px' }}>
              {selectedAgent.eternalImprints.slice(-5).reverse().map((imprint, i) => (
                <div key={i} style={{ marginBottom: '8px', padding: '4px', background: '#2a2a2a' }}>
                  <div><strong>{imprint.event}</strong></div>
                  <div style={{ color: '#aaa' }}>{imprint.analysisSummary}</div>
                  <div style={{ color: '#666', fontSize: '10px' }}>
                    {new Date(imprint.timestamp).toLocaleTimeString()}
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
