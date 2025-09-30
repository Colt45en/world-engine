import React, { useState, useCallback, useRef, useEffect } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, Line, Text, Box, Sphere } from '@react-three/drei';
import { useControls, button, folder } from 'leva';
import * as THREE from 'three';
import { mathEngine } from '../shared/utils';

// =============================================================================
// Core Nexus Runtime Classes (TypeScript Adaptation)
// =============================================================================

export class GoldenGlyph {
  id: string;
  energyLevel: number;
  active: boolean;
  timestamp: number;
  meta: Record<string, any>;
  position: { x: number; y: number; z: number };

  constructor({ energyLevel = 1.0, active = true, meta = {} }: {
    energyLevel?: number;
    active?: boolean;
    meta?: Record<string, any>;
  } = {}) {
    this.id = `gg_${Math.random().toString(36).slice(2, 10)}`;
    this.energyLevel = Number(energyLevel);
    this.active = Boolean(active);
    this.timestamp = Date.now();
    this.meta = { ...meta };
    this.position = { x: 0, y: 0, z: 0 };
  }
}

export class GoldString {
  from: string;
  to: string;
  strength: number;
  persistence: number;

  constructor(fromNodeId: string, toNodeId: string, strength: number = 1.0) {
    this.from = fromNodeId;
    this.to = toNodeId;
    this.strength = Number(strength);
    this.persistence = 1.0;
  }

  decay(factor: number = 0.98): void {
    this.persistence *= factor;
  }
}

export class TerrainNode {
  id: string;
  glyphId: string;
  biome: string;
  elevation: number;
  moisture: number;
  decorations: string[];

  constructor(id: string, glyphId: string, biome: string = "grassland") {
    this.id = id;
    this.glyphId = glyphId;
    this.biome = biome;
    this.elevation = 0;
    this.moisture = 0;
    this.decorations = [];
  }

  updateFromGlyph(glyph: GoldenGlyph): void {
    this.elevation = Number(glyph.energyLevel) * 5;
    this.moisture = typeof glyph.meta?.moisture === "number" ? glyph.meta.moisture : Math.random();
    if (glyph.meta?.mutated) this.biome = "crystalline";
  }

  describe(): string {
    return `🧱 ${this.id} | Biome: ${this.biome} | Elev: ${this.elevation.toFixed(2)} | Moisture: ${this.moisture.toFixed(2)}`;
  }
}

export class EnvironmentalEvent {
  type: string;
  origin: { x: number; y: number; z?: number };
  radius: number;
  duration: number;
  timeElapsed: number;

  constructor(type: "storm" | "flux_surge" | "memory_echo", origin: { x: number; y: number; z?: number }, radius: number = 3, duration: number = 10) {
    this.type = type;
    this.origin = origin;
    this.radius = Number(radius);
    this.duration = Number(duration);
    this.timeElapsed = 0;
  }

  private dist(a: { x: number; y: number; z?: number }, b: { x: number; y: number; z?: number }): number {
    const dx = (a.x ?? 0) - (b.x ?? 0);
    const dy = (a.y ?? 0) - (b.y ?? 0);
    const dz = (a.z ?? 0) - (b.z ?? 0);
    return Math.hypot(dx, dy, dz);
  }

  affects(position: { x: number; y: number; z?: number }): boolean {
    return this.dist(position, this.origin) <= this.radius;
  }

  applyEffect(glyph: GoldenGlyph): void {
    switch (this.type) {
      case "storm":
        glyph.energyLevel *= 0.95;
        break;
      case "flux_surge":
        glyph.energyLevel += 1;
        glyph.meta.mutated = true;
        break;
      case "memory_echo":
        glyph.meta.memoryAwakened = true;
        break;
      default:
        break;
    }
  }
}

// =============================================================================
// Event System
// =============================================================================

class Emitter {
  private _map: Map<string, Function[]>;

  constructor() {
    this._map = new Map();
  }

  on(type: string, fn: Function): () => void {
    const listeners = this._map.get(type) || [];
    listeners.push(fn);
    this._map.set(type, listeners);
    return () => this.off(type, fn);
  }

  off(type: string, fn: Function): void {
    const listeners = this._map.get(type) || [];
    const index = listeners.indexOf(fn);
    if (index > -1) listeners.splice(index, 1);
  }

  emit(type: string, payload?: any): void {
    const listeners = this._map.get(type) || [];
    for (const fn of listeners) {
      try {
        fn(payload);
      } catch (e) {
        console.error(e);
      }
    }
  }
}

// =============================================================================
// Omega and Nova Systems
// =============================================================================

export class Omega {
  private timers: Array<{ nodeId: string; start: number; snapshot: any }>;

  constructor() {
    this.timers = [];
  }

  receivePulse(nodeId: string, glyph: GoldenGlyph): void {
    console.log(`[Omega Drift] Temporal record received from ${nodeId}.`);
    const snap = JSON.parse(JSON.stringify({
      energyLevel: glyph.energyLevel,
      meta: glyph.meta,
      at: Date.now()
    }));
    this.timers.push({ nodeId, start: Date.now(), snapshot: snap });
  }

  tick(now: number = Date.now()): void {
    this.timers = this.timers.filter(t => {
      const age = now - t.start;
      if (age > 5000) { // 5s delayed trigger
        console.log(`[Omega] Triggered delayed event from ${t.nodeId}`);
        return false;
      }
      return true;
    });
  }
}

export class Nova {
  syncGlyphEvent(nodeId: string, glyph: GoldenGlyph): void {
    console.log(`[Nova] Synced glyph event from ${nodeId} (E=${glyph.energyLevel.toFixed(2)})`);
  }
}

// =============================================================================
// Main Nexus Runtime
// =============================================================================

export class Nexus {
  grid: Map<string, { id: string; position: { x: number; y: number; z: number }; connections: GoldString[]; glyph?: GoldenGlyph }>;
  goldGlyphs: Map<string, GoldenGlyph>;
  goldStrings: GoldString[];
  activeEvents: EnvironmentalEvent[];
  terrainMap: Map<string, TerrainNode>;
  omega: Omega;
  nova: Nova;
  events: Emitter;

  constructor() {
    this.grid = new Map();
    this.goldGlyphs = new Map();
    this.goldStrings = [];
    this.activeEvents = [];
    this.terrainMap = new Map();
    this.omega = new Omega();
    this.nova = new Nova();
    this.events = new Emitter();
  }

  addNode(id: string, position: { x?: number; y?: number; z?: number } = {}): void {
    this.grid.set(id, {
      id,
      position: { x: position.x ?? 0, y: position.y ?? 0, z: position.z ?? 0 },
      connections: []
    });
  }

  getNode(id: string) {
    return this.grid.get(id);
  }

  attachGlyphToNode(nodeId: string, glyph: GoldenGlyph): boolean {
    const node = this.grid.get(nodeId);
    if (!node) {
      console.warn(`[Nexus] attachGlyphToNode: node ${nodeId} missing`);
      return false;
    }
    glyph.position = { ...node.position };
    this.goldGlyphs.set(nodeId, glyph);
    node.glyph = glyph;
    console.log(`Glyph attached to ${nodeId}`);
    this.events.emit('attach', { nodeId, glyph });
    return true;
  }

  connectNodesWithString(fromId: string, toId: string, strength: number = 1.0): GoldString | false {
    const a = this.grid.get(fromId);
    const b = this.grid.get(toId);
    if (!a || !b) {
      console.warn(`[Nexus] connectNodesWithString: missing endpoint(s)`);
      return false;
    }
    const s = new GoldString(fromId, toId, strength);
    this.goldStrings.push(s);
    a.connections.push(s);
    b.connections.push(s);
    console.log(`Gold string formed: ${fromId} -> ${toId}`);
    this.events.emit('link', { fromId, toId, string: s });
    return s;
  }

  scanGlyphEvents(threshold: number = 1.5): void {
    for (const [nodeId, glyph] of this.goldGlyphs.entries()) {
      if (glyph.active && glyph.energyLevel > threshold) {
        this.handleGlyphPulse(nodeId, glyph);
      }
    }
  }

  handleGlyphPulse(nodeId: string, glyph: GoldenGlyph): void {
    console.log(`[Nexus Pulse] Node ${nodeId} emitted glyph event.`);
    glyph.energyLevel *= 0.8; // discharge
    glyph.meta.lastPulse = Date.now();
    this.omega?.receivePulse(nodeId, glyph);
    this.nova?.syncGlyphEvent(nodeId, glyph);
    this.events.emit('pulse', { nodeId, glyph });
  }

  spawnEnvironmentalEvent(type: "storm" | "flux_surge" | "memory_echo", origin: { x: number; y: number; z?: number }, radius: number = 3, duration: number = 10): EnvironmentalEvent {
    const ev = new EnvironmentalEvent(type, origin, radius, duration);
    this.activeEvents.push(ev);
    console.log(`[Nexus] Spawned ${type} at (${origin.x},${origin.y}${origin.z != null ? "," + origin.z : ""})`);
    return ev;
  }

  updateEvents(): void {
    this.activeEvents = this.activeEvents.filter(ev => {
      ev.timeElapsed += 1;
      for (const [nodeId, glyph] of this.goldGlyphs.entries()) {
        const node = this.grid.get(nodeId);
        const pos = node?.position ?? glyph.position;
        if (pos && ev.affects(pos)) {
          ev.applyEffect(glyph);
        }
      }
      return ev.timeElapsed < ev.duration;
    });
  }

  generateTerrainLayer(): void {
    this.terrainMap.clear();
    for (const [nodeId, glyph] of this.goldGlyphs.entries()) {
      const tn = new TerrainNode(`terrain_${nodeId}`, nodeId);
      tn.updateFromGlyph(glyph);
      this.terrainMap.set(tn.id, tn);
    }
    this.events.emit('terrain-update', { size: this.terrainMap.size });
    console.log(`[Nexus] Terrain layer initialized with ${this.terrainMap.size} nodes`);
  }

  tick(): void {
    for (const s of this.goldStrings) s.decay();
    this.omega.tick();
    this.updateEvents();
    this.scanGlyphEvents();
  }
}

// =============================================================================
// 3D Visualization Components
// =============================================================================

interface GoldenGlyphNodeProps {
  node: { id: string; position: { x: number; y: number; z: number }; glyph?: GoldenGlyph };
  onNodeClick: (nodeId: string) => void;
}

function GoldenGlyphNode({ node, onNodeClick }: GoldenGlyphNodeProps) {
  const meshRef = useRef<THREE.Mesh>(null);
  const glyph = node.glyph;

  useFrame((state) => {
    if (meshRef.current && glyph) {
      // Pulse animation based on energy level
      const pulseScale = 1 + Math.sin(state.clock.elapsedTime * 2 + glyph.timestamp * 0.001) * 0.1 * glyph.energyLevel;
      meshRef.current.scale.setScalar(pulseScale);

      // Energy-based color intensity
      const intensity = Math.max(0.3, Math.min(1.0, glyph.energyLevel / 3));
      (meshRef.current.material as THREE.MeshStandardMaterial).emissiveIntensity = intensity * 0.5;
    }
  });

  const getGlyphColor = (glyph: GoldenGlyph): string => {
    if (glyph.meta.mutated) return '#ff00ff'; // Crystalline purple
    if (glyph.meta.memoryAwakened) return '#00ffff'; // Memory cyan
    if (glyph.energyLevel > 2) return '#ffff00'; // High energy yellow
    return '#7df'; // Default blue
  };

  const getBiomeColor = (terrainBiome?: string): string => {
    switch (terrainBiome) {
      case 'desert': return '#daa520';
      case 'mountain': return '#708090';
      case 'crystalline': return '#da70d6';
      default: return '#228b22'; // grassland
    }
  };

  if (!glyph) {
    // Empty node
    return (
      <mesh
        ref={meshRef}
        position={[node.position.x, node.position.y, node.position.z]}
        onClick={() => onNodeClick(node.id)}
      >
        <sphereGeometry args={[0.1, 8, 8]} />
        <meshStandardMaterial color="#333" transparent opacity={0.3} />
      </mesh>
    );
  }

  return (
    <group>
      {/* Main glyph sphere */}
      <mesh
        ref={meshRef}
        position={[node.position.x, node.position.y, node.position.z]}
        onClick={() => onNodeClick(node.id)}
      >
        <sphereGeometry args={[0.2, 12, 12]} />
        <meshStandardMaterial
          color={getGlyphColor(glyph)}
          emissive={getGlyphColor(glyph)}
          emissiveIntensity={0.3}
          transparent
          opacity={0.8}
        />
      </mesh>

      {/* Terrain base */}
      <Box
        position={[node.position.x, node.position.y - 0.3, node.position.z]}
        args={[0.4, 0.1, 0.4]}
      >
        <meshStandardMaterial color={getBiomeColor()} />
      </Box>

      {/* Energy level indicator */}
      <Text
        position={[node.position.x, node.position.y + 0.4, node.position.z]}
        fontSize={0.1}
        color="#fff"
        anchorX="center"
        anchorY="middle"
      >
        {glyph.energyLevel.toFixed(1)}
      </Text>
    </group>
  );
}

interface GoldStringVisualizationProps {
  goldString: GoldString;
  nodes: Map<string, { id: string; position: { x: number; y: number; z: number }; glyph?: GoldenGlyph }>;
}

function GoldStringVisualization({ goldString, nodes }: GoldStringVisualizationProps) {
  const fromNode = nodes.get(goldString.from);
  const toNode = nodes.get(goldString.to);

  if (!fromNode || !toNode) return null;

  const points = [
    new THREE.Vector3(fromNode.position.x, fromNode.position.y, fromNode.position.z),
    new THREE.Vector3(toNode.position.x, toNode.position.y, toNode.position.z)
  ];

  const opacity = Math.max(0.1, goldString.persistence);
  const color = `rgba(255, 215, 0, ${opacity})`; // Gold color

  return (
    <Line
      points={points}
      color={color}
      lineWidth={goldString.strength * 2}
      transparent
      opacity={opacity}
    />
  );
}

interface EnvironmentalEventVisualizationProps {
  event: EnvironmentalEvent;
}

function EnvironmentalEventVisualization({ event }: EnvironmentalEventVisualizationProps) {
  const meshRef = useRef<THREE.Mesh>(null);

  useFrame((state) => {
    if (meshRef.current) {
      // Expand/contract animation
      const progress = event.timeElapsed / event.duration;
      const scale = event.radius * (0.5 + 0.5 * Math.sin(state.clock.elapsedTime * 3));
      meshRef.current.scale.setScalar(scale);

      // Fade out over time
      const opacity = 1 - progress;
      (meshRef.current.material as THREE.MeshStandardMaterial).opacity = opacity * 0.3;
    }
  });

  const getEventColor = (type: string): string => {
    switch (type) {
      case 'storm': return '#4169e1'; // Royal blue
      case 'flux_surge': return '#ff4500'; // Orange red
      case 'memory_echo': return '#9370db'; // Medium purple
      default: return '#ffffff';
    }
  };

  return (
    <mesh
      ref={meshRef}
      position={[event.origin.x, event.origin.y, event.origin.z || 0]}
    >
      <sphereGeometry args={[1, 16, 16]} />
      <meshStandardMaterial
        color={getEventColor(event.type)}
        transparent
        opacity={0.3}
        wireframe
      />
    </mesh>
  );
}

// =============================================================================
// Main Component
// =============================================================================

export default function NexusRuntime() {
  const [nexus] = useState(() => new Nexus());
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);
  const [logMessages, setLogMessages] = useState<string[]>([]);

  const {
    autoTick,
    pulseThreshold,
    eventRadius,
    gridSize,
    showTerrain,
    energyDecay
  } = useControls('Nexus Runtime', {
    System: folder({
      autoTick: false,
      pulseThreshold: { value: 1.5, min: 0.5, max: 5.0, step: 0.1 },
      energyDecay: { value: 0.98, min: 0.9, max: 1.0, step: 0.01 }
    }),
    Environment: folder({
      eventRadius: { value: 3, min: 1, max: 10, step: 1 },
      showTerrain: true
    }),
    Grid: folder({
      gridSize: { value: 5, min: 3, max: 10, step: 1 }
    }),
    Actions: folder({
      'Initialize Grid': button(() => initializeGrid()),
      'Add Random Glyph': button(() => addRandomGlyph()),
      'Connect Random Nodes': button(() => connectRandomNodes()),
      'Spawn Storm': button(() => spawnEvent('storm')),
      'Spawn Flux Surge': button(() => spawnEvent('flux_surge')),
      'Spawn Memory Echo': button(() => spawnEvent('memory_echo')),
      'Manual Tick': button(() => manualTick()),
      'Generate Terrain': button(() => nexus.generateTerrainLayer()),
      'Clear System': button(() => clearSystem())
    })
  });

  // Initialize event listeners
  useEffect(() => {
    const unsubscribers = [
      nexus.events.on('attach', ({ nodeId }) => {
        addLogMessage(`🔗 Glyph attached to ${nodeId}`);
      }),
      nexus.events.on('pulse', ({ nodeId, glyph }) => {
        addLogMessage(`⚡ Pulse from ${nodeId} (E=${glyph.energyLevel.toFixed(2)})`);
      }),
      nexus.events.on('link', ({ fromId, toId }) => {
        addLogMessage(`🌟 Gold string: ${fromId} -> ${toId}`);
      }),
      nexus.events.on('terrain-update', ({ size }) => {
        addLogMessage(`🏔️ Terrain updated: ${size} nodes`);
      })
    ];

    return () => unsubscribers.forEach(unsub => unsub());
  }, [nexus]);

  // Auto-tick system
  useEffect(() => {
    if (!autoTick) return;

    const interval = setInterval(() => {
      nexus.tick();
      // Force re-render by updating a counter or similar
      setLogMessages(prev => [...prev]); // Trigger re-render
    }, 1000);

    return () => clearInterval(interval);
  }, [autoTick, nexus]);

  const addLogMessage = useCallback((message: string) => {
    setLogMessages(prev => [...prev.slice(-9), `${new Date().toLocaleTimeString()}: ${message}`]);
  }, []);

  const initializeGrid = useCallback(() => {
    const size = gridSize;
    for (let x = 0; x < size; x++) {
      for (let y = 0; y < size; y++) {
        const nodeId = `node_${x}_${y}`;
        nexus.addNode(nodeId, {
          x: (x - size/2) * 2,
          y: (y - size/2) * 2,
          z: 0
        });
      }
    }
    addLogMessage(`🏗️ Grid initialized: ${size}x${size} nodes`);
  }, [nexus, gridSize, addLogMessage]);

  const addRandomGlyph = useCallback(() => {
    const nodes = Array.from(nexus.grid.keys());
    const emptyNodes = nodes.filter(nodeId => !nexus.goldGlyphs.has(nodeId));

    if (emptyNodes.length === 0) {
      addLogMessage('⚠️ No empty nodes available');
      return;
    }

    const randomNodeId = emptyNodes[Math.floor(Math.random() * emptyNodes.length)];
    const glyph = new GoldenGlyph({
      energyLevel: 1 + Math.random() * 3,
      meta: {
        type: ['bloom', 'storm', 'flux', 'memory'][Math.floor(Math.random() * 4)],
        moisture: Math.random()
      }
    });

    nexus.attachGlyphToNode(randomNodeId, glyph);
  }, [nexus, addLogMessage]);

  const connectRandomNodes = useCallback(() => {
    const glyphNodes = Array.from(nexus.goldGlyphs.keys());
    if (glyphNodes.length < 2) {
      addLogMessage('⚠️ Need at least 2 glyphs to connect');
      return;
    }

    const from = glyphNodes[Math.floor(Math.random() * glyphNodes.length)];
    let to = glyphNodes[Math.floor(Math.random() * glyphNodes.length)];
    while (to === from && glyphNodes.length > 1) {
      to = glyphNodes[Math.floor(Math.random() * glyphNodes.length)];
    }

    nexus.connectNodesWithString(from, to, 0.5 + Math.random() * 1.5);
  }, [nexus, addLogMessage]);

  const spawnEvent = useCallback((type: "storm" | "flux_surge" | "memory_echo") => {
    const origin = {
      x: (Math.random() - 0.5) * gridSize * 2,
      y: (Math.random() - 0.5) * gridSize * 2,
      z: 0
    };

    nexus.spawnEnvironmentalEvent(type, origin, eventRadius, 10);
    addLogMessage(`🌪️ ${type} spawned`);
  }, [nexus, eventRadius, gridSize, addLogMessage]);

  const manualTick = useCallback(() => {
    nexus.tick();
    addLogMessage('⏰ Manual tick executed');
  }, [nexus, addLogMessage]);

  const clearSystem = useCallback(() => {
    nexus.grid.clear();
    nexus.goldGlyphs.clear();
    nexus.goldStrings.length = 0;
    nexus.activeEvents.length = 0;
    nexus.terrainMap.clear();
    setSelectedNodeId(null);
    setLogMessages([]);
    addLogMessage('🧹 System cleared');
  }, [nexus, addLogMessage]);

  const handleNodeClick = useCallback((nodeId: string) => {
    setSelectedNodeId(nodeId);
    const node = nexus.getNode(nodeId);
    if (node?.glyph) {
      addLogMessage(`👆 Selected ${nodeId}: E=${node.glyph.energyLevel.toFixed(2)}`);
    }
  }, [nexus, addLogMessage]);

  // Get current state for rendering
  const nodes = Array.from(nexus.grid.values());
  const selectedNode = selectedNodeId ? nexus.getNode(selectedNodeId) : null;

  return (
    <div style={{ width: '100%', height: '100vh', display: 'flex', flexDirection: 'column' }}>
      {/* 3D Visualization */}
      <div style={{ flex: 1, background: '#000' }}>
        <Canvas camera={{ position: [10, 10, 10], fov: 60 }}>
          {/* Grid nodes and glyphs */}
          {nodes.map(node => (
            <GoldenGlyphNode
              key={node.id}
              node={node}
              onNodeClick={handleNodeClick}
            />
          ))}

          {/* Gold strings */}
          {nexus.goldStrings.map((goldString, index) => (
            <GoldStringVisualization
              key={`${goldString.from}-${goldString.to}-${index}`}
              goldString={goldString}
              nodes={nexus.grid}
            />
          ))}

          {/* Environmental events */}
          {nexus.activeEvents.map((event, index) => (
            <EnvironmentalEventVisualization
              key={`${event.type}-${index}`}
              event={event}
            />
          ))}

          <ambientLight intensity={0.4} />
          <pointLight position={[10, 10, 10]} intensity={0.8} />
          <OrbitControls enablePan={true} enableZoom={true} enableRotate={true} />
        </Canvas>
      </div>

      {/* Info Panel */}
      <div style={{
        height: '250px',
        background: '#1a1a1a',
        color: '#fff',
        padding: '20px',
        display: 'flex',
        gap: '20px',
        overflow: 'auto'
      }}>
        <div style={{ flex: 1 }}>
          <h3>⚡ System Status</h3>
          <div>Nodes: {nexus.grid.size}</div>
          <div>Glyphs: {nexus.goldGlyphs.size}</div>
          <div>Gold Strings: {nexus.goldStrings.length}</div>
          <div>Active Events: {nexus.activeEvents.length}</div>
          <div>Terrain Nodes: {nexus.terrainMap.size}</div>
          <div>Auto-Tick: {autoTick ? '🟢 ON' : '🔴 OFF'}</div>
        </div>

        {selectedNode && (
          <div style={{ flex: 1 }}>
            <h3>🎯 Selected Node</h3>
            <div style={{ fontSize: '12px', fontFamily: 'monospace' }}>
              <div><strong>ID:</strong> {selectedNode.id}</div>
              <div><strong>Position:</strong> ({selectedNode.position.x}, {selectedNode.position.y}, {selectedNode.position.z})</div>
              {selectedNode.glyph && (
                <>
                  <div><strong>Glyph ID:</strong> {selectedNode.glyph.id}</div>
                  <div><strong>Energy:</strong> {selectedNode.glyph.energyLevel.toFixed(2)}</div>
                  <div><strong>Active:</strong> {selectedNode.glyph.active ? 'Yes' : 'No'}</div>
                  <div><strong>Meta:</strong> {JSON.stringify(selectedNode.glyph.meta).substring(0, 50)}...</div>
                </>
              )}
              <div><strong>Connections:</strong> {selectedNode.connections?.length || 0}</div>
            </div>
          </div>
        )}

        <div style={{ flex: 1 }}>
          <h3>📜 System Log</h3>
          <div style={{
            maxHeight: '160px',
            overflow: 'auto',
            fontSize: '11px',
            fontFamily: 'monospace',
            background: '#2a2a2a',
            padding: '8px',
            borderRadius: '4px'
          }}>
            {logMessages.map((msg, index) => (
              <div key={index} style={{ marginBottom: '2px' }}>
                {msg}
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
