import React, { useEffect, useMemo, useRef, useState } from "react";
import * as THREE from "three";
import { Canvas, useFrame } from "@react-three/fiber";
import { OrbitControls, Html } from "@react-three/drei";
import { useControls, button, folder } from "leva";
import { mathEngine } from '../shared/utils';

// =============================================================================
// Mathematical Surface Functions
// =============================================================================

// Hyperbolic bowl function
function hyperbolicBowl(x: number, y: number, params: SurfaceParameters): number {
  const { a, b, c, scale, offset } = params;
  return scale * (-a / (b * x * x + c * y * y + offset)) + 0.02;
}

// Paraboloid function
function paraboloid(x: number, y: number, params: SurfaceParameters): number {
  const { a, b, scale } = params;
  return scale * (a * x * x + b * y * y);
}

// Saddle function
function saddle(x: number, y: number, params: SurfaceParameters): number {
  const { a, b, scale } = params;
  return scale * (a * x * x - b * y * y);
}

// Ripple function
function ripple(x: number, y: number, params: SurfaceParameters): number {
  const { a, b, c, scale } = params;
  const r = Math.sqrt(x * x + y * y);
  return scale * Math.sin(a * r + c) * Math.exp(-b * r);
}

// Gaussian peak function
function gaussian(x: number, y: number, params: SurfaceParameters): number {
  const { a, b, c, scale } = params;
  const r2 = (x - a) * (x - a) + (y - b) * (y - b);
  return scale * Math.exp(-c * r2);
}

// Monkey saddle function
function monkeySaddle(x: number, y: number, params: SurfaceParameters): number {
  const { scale } = params;
  return scale * (x * x * x - 3 * x * y * y);
}

// Sine wave surface
function sineWave(x: number, y: number, params: SurfaceParameters): number {
  const { a, b, scale } = params;
  return scale * (Math.sin(a * x) * Math.cos(b * y));
}

// Torus function
function torus(x: number, y: number, params: SurfaceParameters): number {
  const { a, b, scale } = params; // a = major radius, b = minor radius
  const r = Math.sqrt(x * x + y * y);
  const h = Math.sqrt(Math.max(0, b * b - (r - a) * (r - a)));
  return scale * h;
}

// =============================================================================
// Types and Interfaces
// =============================================================================

interface SurfaceParameters {
  a: number;
  b: number;
  c: number;
  scale: number;
  offset: number;
}

type SurfaceFunction = (x: number, y: number, params: SurfaceParameters) => number;

interface SurfaceMeshProps {
  surfaceFunction: SurfaceFunction;
  parameters: SurfaceParameters;
  resolution: number;
  domain: number;
  animated: boolean;
  time: number;
  onStatsUpdate: (stats: { vertices: number; triangles: number; minZ: number; maxZ: number }) => void;
}

// =============================================================================
// Surface Functions Registry
// =============================================================================

const SURFACE_FUNCTIONS: Record<string, {
  func: SurfaceFunction;
  name: string;
  description: string;
  defaultParams: Partial<SurfaceParameters>;
}> = {
  hyperbolic: {
    func: hyperbolicBowl,
    name: "Hyperbolic Bowl",
    description: "Classic hyperbolic bowl surface with singularity control",
    defaultParams: { a: 1.0, b: 1.0, c: 1.0, offset: 0.02, scale: 0.2 }
  },
  paraboloid: {
    func: paraboloid,
    name: "Paraboloid",
    description: "Elliptic or circular paraboloid surface",
    defaultParams: { a: 0.5, b: 0.5, scale: 0.3 }
  },
  saddle: {
    func: saddle,
    name: "Saddle Surface",
    description: "Hyperbolic paraboloid (saddle point)",
    defaultParams: { a: 0.3, b: 0.3, scale: 0.4 }
  },
  ripple: {
    func: ripple,
    name: "Ripple Wave",
    description: "Damped oscillating wave pattern",
    defaultParams: { a: 4.0, b: 0.5, c: 0.0, scale: 0.2 }
  },
  gaussian: {
    func: gaussian,
    name: "Gaussian Peak",
    description: "Gaussian bell curve with adjustable center and width",
    defaultParams: { a: 0.0, b: 0.0, c: 2.0, scale: 0.5 }
  },
  monkeySaddle: {
    func: monkeySaddle,
    name: "Monkey Saddle",
    description: "Three-way saddle point surface",
    defaultParams: { scale: 0.15 }
  },
  sineWave: {
    func: sineWave,
    name: "Sine Wave",
    description: "Sinusoidal wave pattern in X and Y",
    defaultParams: { a: 2.0, b: 2.0, scale: 0.2 }
  },
  torus: {
    func: torus,
    name: "Torus Surface",
    description: "Toroidal surface with major and minor radii",
    defaultParams: { a: 0.6, b: 0.3, scale: 0.8 }
  }
};

// =============================================================================
// Animated Surface Mesh Component
// =============================================================================

function SurfaceMesh({
  surfaceFunction,
  parameters,
  resolution,
  domain,
  animated,
  time,
  onStatsUpdate
}: SurfaceMeshProps) {
  const geometryRef = useRef<THREE.PlaneGeometry>(null!);

  // Update geometry when parameters change
  useEffect(() => {
    if (!geometryRef.current) return;

    const geometry = geometryRef.current;
    const position = geometry.attributes.position as THREE.BufferAttribute;

    let minZ = Infinity;
    let maxZ = -Infinity;

    for (let j = 0; j < resolution; j++) {
      for (let i = 0; i < resolution; i++) {
        const x = -domain + 2 * domain * i / (resolution - 1);
        const y = -domain + 2 * domain * j / (resolution - 1);

        // Add time-based animation if enabled
        const animatedParams = animated ? {
          ...parameters,
          c: parameters.c + Math.sin(time * 2) * 0.5,
          a: parameters.a * (1 + Math.cos(time * 1.5) * 0.1),
          b: parameters.b * (1 + Math.sin(time * 1.2) * 0.1)
        } : parameters;

        const z = surfaceFunction(x, y, animatedParams);

        minZ = Math.min(minZ, z);
        maxZ = Math.max(maxZ, z);

        position.setZ(j * resolution + i, z);
      }
    }

    position.needsUpdate = true;
    geometry.computeVertexNormals();

    // Calculate statistics
    const vertices = position.count;
    const triangles = geometry.index ? geometry.index.count / 3 : vertices / 3;

    onStatsUpdate({
      vertices,
      triangles: Math.floor(triangles),
      minZ,
      maxZ
    });
  }, [surfaceFunction, parameters, resolution, domain, animated, time, onStatsUpdate]);

  return (
    <mesh rotation-x={-Math.PI / 2} castShadow receiveShadow>
      <planeGeometry
        ref={geometryRef}
        args={[2 * domain, 2 * domain, resolution - 1, resolution - 1]}
      />
      <meshStandardMaterial
        color="#ffcf66"
        metalness={0.1}
        roughness={0.85}
        side={THREE.DoubleSide}
        wireframe={false}
      />
    </mesh>
  );
}

// =============================================================================
// Grid Lines Component
// =============================================================================

function GridLines({ domain, resolution }: { domain: number; resolution: number }) {
  const points = useMemo(() => {
    const gridPoints: THREE.Vector3[] = [];

    // Horizontal lines
    for (let i = 0; i < resolution; i += Math.max(1, Math.floor(resolution / 20))) {
      const y = -domain + 2 * domain * i / (resolution - 1);
      gridPoints.push(new THREE.Vector3(-domain, y, 0.01));
      gridPoints.push(new THREE.Vector3(domain, y, 0.01));
    }

    // Vertical lines
    for (let i = 0; i < resolution; i += Math.max(1, Math.floor(resolution / 20))) {
      const x = -domain + 2 * domain * i / (resolution - 1);
      gridPoints.push(new THREE.Vector3(x, -domain, 0.01));
      gridPoints.push(new THREE.Vector3(x, domain, 0.01));
    }

    return gridPoints;
  }, [domain, resolution]);

  return (
    <group rotation-x={-Math.PI / 2}>
      {points.length > 0 && (
        <lineSegments>
          <bufferGeometry>
            <bufferAttribute
              attach="attributes-position"
              count={points.length}
              array={new Float32Array(points.flatMap(p => [p.x, p.y, p.z]))}
              itemSize={3}
            />
          </bufferGeometry>
          <lineBasicMaterial color="#444444" transparent opacity={0.3} />
        </lineSegments>
      )}
    </group>
  );
}

// =============================================================================
// Main HyperBowl Component
// =============================================================================

export default function HyperBowl() {
  const [stats, setStats] = useState({
    vertices: 0,
    triangles: 0,
    minZ: 0,
    maxZ: 0
  });
  const [time, setTime] = useState(0);

  useFrame(({ clock }) => {
    setTime(clock.getElapsedTime());
  });

  const {
    surfaceType,
    resolution,
    domain,
    a,
    b,
    c,
    scale,
    offset,
    animated,
    showGrid,
    wireframe
  } = useControls('HyperBowl Surface', {
    'Surface Type': folder({
      surfaceType: {
        value: 'hyperbolic',
        options: Object.fromEntries(
          Object.entries(SURFACE_FUNCTIONS).map(([key, { name }]) => [name, key])
        )
      }
    }),
    'Geometry': folder({
      resolution: { value: 160, min: 20, max: 300, step: 10 },
      domain: { value: 1.0, min: 0.5, max: 3.0, step: 0.1 },
      showGrid: false
    }),
    'Parameters': folder({
      a: { value: 1.0, min: -5.0, max: 5.0, step: 0.05 },
      b: { value: 1.0, min: -5.0, max: 5.0, step: 0.05 },
      c: { value: 1.0, min: -5.0, max: 5.0, step: 0.05 },
      scale: { value: 0.2, min: 0.01, max: 1.0, step: 0.01 },
      offset: { value: 0.02, min: 0.001, max: 0.1, step: 0.001 }
    }),
    'Animation': folder({
      animated: false,
      wireframe: false
    }),
    'Actions': folder({
      'Reset Parameters': button(() => resetToDefaults()),
      'Export Surface': button(() => exportSurfaceData()),
      'Capture Frame': button(() => captureFrame()),
      'Generate Random': button(() => generateRandomSurface())
    })
  });

  // Get current surface function
  const currentSurface = SURFACE_FUNCTIONS[surfaceType];

  // Combine parameters
  const parameters: SurfaceParameters = {
    a,
    b,
    c,
    scale,
    offset
  };

  const resetToDefaults = () => {
    const defaults = currentSurface.defaultParams;
    console.log(`Reset to defaults for ${currentSurface.name}:`, defaults);
  };

  const exportSurfaceData = () => {
    const data = {
      surfaceType,
      parameters,
      resolution,
      domain,
      stats,
      timestamp: Date.now(),
      description: currentSurface.description
    };

    console.log('Surface data exported:', data);

    // Create downloadable JSON file
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `hyperbowl-${surfaceType}-${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const captureFrame = () => {
    console.log(`Frame captured at t=${time.toFixed(2)}s`, {
      surface: currentSurface.name,
      parameters,
      stats
    });
  };

  const generateRandomSurface = () => {
    const randomParams = {
      a: (Math.random() - 0.5) * 4,
      b: (Math.random() - 0.5) * 4,
      c: Math.random() * 3,
      scale: Math.random() * 0.5 + 0.1,
      offset: Math.random() * 0.05 + 0.005
    };
    console.log('Random surface generated:', randomParams);
  };

  return (
    <div style={{ width: '100%', height: '100vh', display: 'flex', flexDirection: 'column' }}>
      {/* 3D Scene */}
      <div style={{ flex: 1, background: '#0a0a0a' }}>
        <Canvas camera={{ position: [3, 4, 3], fov: 60 }}>
          <ambientLight intensity={0.4} />
          <directionalLight position={[5, 8, 5]} intensity={0.8} castShadow />
          <pointLight position={[-3, 3, -3]} intensity={0.4} color="#ff6b6b" />
          <spotLight position={[0, 6, 0]} intensity={0.6} angle={0.4} penumbra={0.2} />

          <SurfaceMesh
            surfaceFunction={currentSurface.func}
            parameters={parameters}
            resolution={resolution}
            domain={domain}
            animated={animated}
            time={time}
            onStatsUpdate={setStats}
          />

          {showGrid && <GridLines domain={domain} resolution={resolution} />}

          <OrbitControls enablePan={true} enableZoom={true} enableRotate={true} />
        </Canvas>
      </div>

      {/* Status Panel */}
      <div style={{
        height: '140px',
        background: '#1a1a1a',
        color: '#fff',
        padding: '20px',
        display: 'flex',
        gap: '20px',
        overflow: 'auto'
      }}>
        <div style={{ flex: 1 }}>
          <h3>📊 Surface Analysis</h3>
          <div>Type: {currentSurface.name}</div>
          <div>Resolution: {resolution}×{resolution}</div>
          <div>Domain: [-{domain.toFixed(1)}, {domain.toFixed(1)}]²</div>
          <div>Animation: {animated ? 'ON' : 'OFF'}</div>
        </div>

        <div style={{ flex: 1 }}>
          <h3>🔢 Parameters</h3>
          <div>a: {a.toFixed(3)}, b: {b.toFixed(3)}</div>
          <div>c: {c.toFixed(3)}, scale: {scale.toFixed(3)}</div>
          <div>offset: {offset.toFixed(4)}</div>
          <div>Z-range: [{stats.minZ.toFixed(3)}, {stats.maxZ.toFixed(3)}]</div>
        </div>

        <div style={{ flex: 1 }}>
          <h3>⚙️ Geometry Stats</h3>
          <div>Vertices: {stats.vertices.toLocaleString()}</div>
          <div>Triangles: {stats.triangles.toLocaleString()}</div>
          <div>Memory: ~{((stats.vertices * 3 * 4) / 1024).toFixed(1)} KB</div>
          <div>Time: {time.toFixed(1)}s</div>
        </div>
      </div>
    </div>
  );
}
