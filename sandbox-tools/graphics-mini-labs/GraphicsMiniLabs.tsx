import React, { useRef, useMemo } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, Html } from '@react-three/drei';
import { Leva, useControls } from 'leva';
import * as THREE from 'three';

// =============================================================================
// Mini-lab A: Barycentric Vertex Paint
// =============================================================================

const BarycentricTriangle: React.FC = () => {
  const meshRef = useRef<THREE.Mesh>(null);

  const controls = useControls('Barycentric Paint', {
    showVertexLabels: { value: true, label: 'Show Vertex Labels' },
    showWeights: { value: true, label: 'Show Weight Values' },
    wireframe: { value: false, label: 'Wireframe Mode' },
    rotateTriangle: { value: false, label: 'Auto Rotate' }
  });

  // Create triangle geometry with barycentric coordinates as vertex attribute
  const geometry = useMemo(() => {
    const geom = new THREE.BufferGeometry();

    // Triangle vertices (A, B, C)
    const vertices = new Float32Array([
      -1, -0.5, 0,  // Vertex A
       1, -0.5, 0,  // Vertex B
       0,  1.0, 0   // Vertex C
    ]);

    // Barycentric coordinates for each vertex
    // A: (1,0,0) - red, B: (0,1,0) - green, C: (0,0,1) - blue
    const barycentrics = new Float32Array([
      1, 0, 0,  // A: pure red
      0, 1, 0,  // B: pure green
      0, 0, 1   // C: pure blue
    ]);

    const indices = new Uint16Array([0, 1, 2]);

    geom.setAttribute('position', new THREE.BufferAttribute(vertices, 3));
    geom.setAttribute('barycentric', new THREE.BufferAttribute(barycentrics, 3));
    geom.setIndex(new THREE.BufferAttribute(indices, 1));
    geom.computeVertexNormals();

    return geom;
  }, []);

  // Custom shader material
  const material = useMemo(() => {
    return new THREE.ShaderMaterial({
      uniforms: {
        time: { value: 0 }
      },
      vertexShader: `
        attribute vec3 barycentric;
        varying vec3 vBary;
        varying vec3 vPosition;

        void main() {
          vBary = barycentric;
          vPosition = position;
          gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
        }
      `,
      fragmentShader: `
        uniform float time;
        varying vec3 vBary;
        varying vec3 vPosition;

        void main() {
          // Use barycentric coordinates as RGB values
          // A(1,0,0) → red, B(0,1,0) → green, C(0,0,1) → blue
          vec3 color = vec3(vBary.x, vBary.y, vBary.z);

          // Add some subtle animation
          color *= 0.8 + 0.2 * sin(time * 2.0);

          gl_FragColor = vec4(color, 1.0);
        }
      `,
      wireframe: controls.wireframe,
      side: THREE.DoubleSide
    });
  }, [controls.wireframe]);

  useFrame((state) => {
    if (meshRef.current) {
      material.uniforms.time.value = state.clock.elapsedTime;

      if (controls.rotateTriangle) {
        meshRef.current.rotation.z = Math.sin(state.clock.elapsedTime * 0.5) * 0.3;
      }
    }
  });

  return (
    <>
      <mesh ref={meshRef} geometry={geometry} material={material} />

      {/* Vertex labels */}
      {controls.showVertexLabels && (
        <>
          <Html position={[-1, -0.5, 0]} style={{ color: '#ff4444', fontFamily: 'monospace', fontSize: '14px', fontWeight: 'bold' }}>
            A (1,0,0)
          </Html>
          <Html position={[1, -0.5, 0]} style={{ color: '#44ff44', fontFamily: 'monospace', fontSize: '14px', fontWeight: 'bold' }}>
            B (0,1,0)
          </Html>
          <Html position={[0, 1.0, 0]} style={{ color: '#4444ff', fontFamily: 'monospace', fontSize: '14px', fontWeight: 'bold' }}>
            C (0,0,1)
          </Html>
        </>
      )}

      {/* Weight explanation */}
      {controls.showWeights && (
        <Html position={[0, -1.5, 0]} style={{
          color: '#00ff88',
          fontFamily: 'monospace',
          fontSize: '12px',
          textAlign: 'center',
          background: 'rgba(0,0,0,0.8)',
          padding: '10px',
          borderRadius: '8px',
          border: '1px solid #00ff88'
        }}>
          <div>
            <strong>Barycentric Coordinates</strong><br/>
            • At vertex A: w = (1,0,0) → Pure Red<br/>
            • At midpoint AB: w ≈ (0.5,0.5,0) → Red + Green<br/>
            • At centroid: w ≈ (⅓,⅓,⅓) → Average of all three<br/>
            • Weights always sum to 1.0 inside triangle
          </div>
        </Html>
      )}
    </>
  );
};

// =============================================================================
// Mini-lab B: Catmull-Rom Curves with Parameterization
// =============================================================================

type V3 = [number, number, number];

// Vector operations
const add = (a: V3, b: V3): V3 => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
const sub = (a: V3, b: V3): V3 => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const mul = (a: V3, s: number): V3 => [a[0] * s, a[1] * s, a[2] * s];
const len = (a: V3) => Math.hypot(a[0], a[1], a[2]);

function parametrize(P: V3[], alpha: number): number[] {
  const t = [0];
  for (let i = 1; i < P.length; i++) {
    const d = len(sub(P[i], P[i - 1]));
    t.push(t[i - 1] + Math.pow(d, alpha));
  }
  return t;
}

function catmullRom(Pm1: V3, P0: V3, P1: V3, P2: V3, u: number, t: number[], i: number): V3 {
  const t0 = t[i - 1], t1 = t[i], t2 = t[i + 1], t3 = t[i + 2];

  // Local u maps [t1,t2] → [0,1]
  const u01 = (1 - u) * t1 + u * t2;

  // Tangents (finite difference with non-uniform spacing)
  const m1 = mul(sub(P1, Pm1), (t2 - t1) / (t1 - t0)) as V3;
  const m2 = mul(sub(P2, P0), (t2 - t1) / (t2 - t0)) as V3;

  // Hermite basis
  const s = (u01 - t1) / (t2 - t1);
  const h00 = 2 * s ** 3 - 3 * s ** 2 + 1;
  const h10 = s ** 3 - 2 * s ** 2 + s;
  const h01 = -2 * s ** 3 + 3 * s ** 2;
  const h11 = s ** 3 - s ** 2;

  return add(
    add(mul(P0, h00), mul(m1, h10 * (t2 - t1))),
    add(mul(P1, h01), mul(m2, h11 * (t2 - t1)))
  );
}

const CatmullRomCurves: React.FC = () => {
  const [controlPoints, setControlPoints] = React.useState<V3[]>([
    [-2, -1, 0],
    [-1, 1, 0],
    [1, -1, 0],
    [2, 1, 0]
  ]);

  const controls = useControls('Catmull-Rom', {
    alpha: { value: 0.5, min: 0, max: 1, step: 0.01, label: 'Parameterization (α)' },
    segments: { value: 100, min: 20, max: 200, step: 10, label: 'Curve Resolution' },
    showControlPoints: { value: true, label: 'Show Control Points' },
    showTangents: { value: false, label: 'Show Tangents' },
    parameterType: {
      value: 'centripetal',
      options: ['uniform', 'chordal', 'centripetal'],
      label: 'Parameter Type'
    }
  });

  // Convert parameter type to alpha value
  const alpha = controls.parameterType === 'uniform' ? 0 :
                controls.parameterType === 'chordal' ? 1 : 0.5;

  const curveGeometry = useMemo(() => {
    if (controlPoints.length < 4) return null;

    const t = parametrize(controlPoints, alpha);
    const curvePoints: THREE.Vector3[] = [];

    // Generate curve segments
    for (let i = 1; i < controlPoints.length - 2; i++) {
      for (let j = 0; j <= controls.segments; j++) {
        const u = j / controls.segments;
        const point = catmullRom(
          controlPoints[i - 1],
          controlPoints[i],
          controlPoints[i + 1],
          controlPoints[i + 2],
          u,
          t,
          i
        );
        curvePoints.push(new THREE.Vector3(...point));
      }
    }

    const geometry = new THREE.BufferGeometry().setFromPoints(curvePoints);
    return geometry;
  }, [controlPoints, alpha, controls.segments]);

  const handleCanvasClick = React.useCallback((event: any) => {
    // Convert mouse position to world coordinates
    const rect = event.target.getBoundingClientRect();
    const x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    const y = -((event.clientY - rect.top) / rect.height) * 2 + 1;

    const newPoint: V3 = [x * 3, y * 2, 0];
    setControlPoints(prev => [...prev, newPoint]);
  }, []);

  return (
    <>
      {/* Curve line */}
      {curveGeometry && (
        <line geometry={curveGeometry}>
          <lineBasicMaterial color="#00ff88" linewidth={3} />
        </line>
      )}

      {/* Control points */}
      {controls.showControlPoints && controlPoints.map((point, i) => (
        <mesh key={i} position={point}>
          <sphereGeometry args={[0.05]} />
          <meshBasicMaterial color={i === 0 || i === controlPoints.length - 1 ? "#ff4444" : "#ffff44"} />
        </mesh>
      ))}

      {/* Instructions */}
      <Html position={[0, 2.5, 0]} style={{
        color: '#00ff88',
        fontFamily: 'monospace',
        fontSize: '12px',
        textAlign: 'center',
        background: 'rgba(0,0,0,0.8)',
        padding: '15px',
        borderRadius: '8px',
        border: '1px solid #00ff88',
        maxWidth: '400px'
      }}>
        <div>
          <strong>Catmull-Rom Curve Parameterization</strong><br/>
          • <strong>Uniform (α=0)</strong>: Equal spacing, may overshoot<br/>
          • <strong>Chordal (α=1)</strong>: Distance-based, more tension<br/>
          • <strong>Centripetal (α=0.5)</strong>: Best balance, no loops<br/><br/>
          <button
            onClick={() => setControlPoints([])}
            style={{
              background: 'rgba(255,68,68,0.2)',
              border: '1px solid #ff4444',
              color: '#ff4444',
              padding: '5px 10px',
              borderRadius: '4px',
              cursor: 'pointer',
              fontSize: '11px'
            }}
          >
            Clear Points
          </button>
        </div>
      </Html>
    </>
  );
};

// =============================================================================
// Mini-lab C: Texture Paint via Barycentrics
// =============================================================================

const cross = (a: V3, b: V3): V3 => [
  a[1] * b[2] - a[2] * b[1],
  a[2] * b[0] - a[0] * b[2],
  a[0] * b[1] - a[1] * b[0]
];

function baryWeights(A: V3, B: V3, C: V3, P: V3): [number, number, number] {
  const areaABC = 0.5 * len(cross(sub(B, A), sub(C, A)));
  const w0 = 0.5 * len(cross(sub(B, P), sub(C, P))) / areaABC;
  const w1 = 0.5 * len(cross(sub(A, P), sub(C, P))) / areaABC;
  const w2 = 0.5 * len(cross(sub(A, P), sub(B, P))) / areaABC;
  return [w0, w1, w2];
}

const TexturePaintTriangle: React.FC = () => {
  const meshRef = useRef<THREE.Mesh>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [texture, setTexture] = React.useState<THREE.CanvasTexture | null>(null);
  const [isDrawing, setIsDrawing] = React.useState(false);

  const controls = useControls('Texture Paint', {
    brushSize: { value: 20, min: 5, max: 50, step: 1, label: 'Brush Size' },
    brushColor: { value: '#ff0000', label: 'Brush Color' },
    showUVGrid: { value: true, label: 'Show UV Grid' },
    clearTexture: { value: false, label: 'Clear Canvas' }
  });

  // Triangle vertices
  const triangleVertices = useMemo(() => ({
    A: [-1, -0.5, 0] as V3,
    B: [1, -0.5, 0] as V3,
    C: [0, 1.0, 0] as V3
  }), []);

  // Initialize canvas texture
  React.useEffect(() => {
    if (canvasRef.current) {
      const canvas = canvasRef.current;
      const ctx = canvas.getContext('2d')!;

      canvas.width = 256;
      canvas.height = 256;

      // Clear canvas
      ctx.fillStyle = '#222222';
      ctx.fillRect(0, 0, canvas.width, canvas.height);

      // Draw UV grid if enabled
      if (controls.showUVGrid) {
        ctx.strokeStyle = '#444444';
        ctx.lineWidth = 1;
        for (let i = 0; i <= 8; i++) {
          const x = (i / 8) * canvas.width;
          const y = (i / 8) * canvas.height;
          ctx.beginPath();
          ctx.moveTo(x, 0);
          ctx.lineTo(x, canvas.height);
          ctx.moveTo(0, y);
          ctx.lineTo(canvas.width, y);
          ctx.stroke();
        }
      }

      const canvasTexture = new THREE.CanvasTexture(canvas);
      canvasTexture.needsUpdate = true;
      setTexture(canvasTexture);
    }
  }, [controls.showUVGrid]);

  // Clear texture when button pressed
  React.useEffect(() => {
    if (controls.clearTexture && canvasRef.current) {
      const canvas = canvasRef.current;
      const ctx = canvas.getContext('2d')!;
      ctx.fillStyle = '#222222';
      ctx.fillRect(0, 0, canvas.width, canvas.height);

      if (texture) {
        texture.needsUpdate = true;
      }
    }
  }, [controls.clearTexture, texture]);

  const geometry = useMemo(() => {
    const geom = new THREE.BufferGeometry();

    const vertices = new Float32Array([
      -1, -0.5, 0,  // A
       1, -0.5, 0,  // B
       0,  1.0, 0   // C
    ]);

    // UV coordinates: A(0,0), B(1,0), C(0,1)
    const uvs = new Float32Array([
      0, 0,  // A
      1, 0,  // B
      0, 1   // C
    ]);

    const indices = new Uint16Array([0, 1, 2]);

    geom.setAttribute('position', new THREE.BufferAttribute(vertices, 3));
    geom.setAttribute('uv', new THREE.BufferAttribute(uvs, 2));
    geom.setIndex(new THREE.BufferAttribute(indices, 1));
    geom.computeVertexNormals();

    return geom;
  }, []);

  const handleTriangleClick = React.useCallback((event: any) => {
    if (!canvasRef.current || !texture) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d')!;

    // Get intersection point (this is simplified - in real implementation you'd raycast)
    const point = event.point;
    const P: V3 = [point.x, point.y, point.z];

    // Calculate barycentric coordinates
    const [w0, w1, w2] = baryWeights(
      triangleVertices.A,
      triangleVertices.B,
      triangleVertices.C,
      P
    );

    // Check if point is inside triangle
    if (w0 >= 0 && w1 >= 0 && w2 >= 0) {
      // Convert barycentric to UV coordinates
      // A(0,0), B(1,0), C(0,1)
      const u = w1 * 1 + w2 * 0; // B contributes to u
      const v = w2 * 1 + w0 * 0; // C contributes to v

      // Paint on canvas
      const x = u * canvas.width;
      const y = (1 - v) * canvas.height; // Flip Y for canvas

      ctx.fillStyle = controls.brushColor;
      ctx.beginPath();
      ctx.arc(x, y, controls.brushSize / 2, 0, Math.PI * 2);
      ctx.fill();

      texture.needsUpdate = true;
    }
  }, [triangleVertices, controls.brushColor, controls.brushSize, texture]);

  return (
    <>
      <mesh
        ref={meshRef}
        geometry={geometry}
        onClick={handleTriangleClick}
      >
        <meshBasicMaterial
          map={texture}
          side={THREE.DoubleSide}
          transparent
        />
      </mesh>

      {/* Hidden canvas element */}
      <canvas
        ref={canvasRef}
        style={{ display: 'none' }}
      />

      {/* Instructions */}
      <Html position={[0, -2, 0]} style={{
        color: '#00ff88',
        fontFamily: 'monospace',
        fontSize: '12px',
        textAlign: 'center',
        background: 'rgba(0,0,0,0.8)',
        padding: '15px',
        borderRadius: '8px',
        border: '1px solid #00ff88',
        maxWidth: '300px'
      }}>
        <div>
          <strong>Texture Paint with Barycentrics</strong><br/>
          • Click on triangle to paint<br/>
          • Barycentric coordinates → UV mapping<br/>
          • A(0,0), B(1,0), C(0,1) UV layout<br/>
          • Paint directly on 3D surface!
        </div>
      </Html>
    </>
  );
};

// =============================================================================
// Main Component
// =============================================================================

const GraphicsMiniLabs: React.FC = () => {
  const controls = useControls('Labs', {
    currentLab: {
      value: 'barycentric',
      options: ['barycentric', 'catmull-rom', 'texture-paint'],
      label: 'Active Mini-Lab'
    }
  });

  const renderCurrentLab = () => {
    switch (controls.currentLab) {
      case 'barycentric':
        return <BarycentricTriangle />;
      case 'catmull-rom':
        return <CatmullRomCurves />;
      case 'texture-paint':
        return <TexturePaintTriangle />;
      default:
        return <BarycentricTriangle />;
    }
  };

  return (
    <>
      <ambientLight intensity={0.6} />
      <directionalLight position={[10, 10, 5]} intensity={0.8} />

      {renderCurrentLab()}

      {/* Title */}
      <Html position={[0, 3, 0]} style={{
        color: '#00ff88',
        fontFamily: 'monospace',
        fontSize: '18px',
        fontWeight: 'bold',
        textAlign: 'center'
      }}>
        Graphics Mini-Labs Collection
      </Html>
    </>
  );
};

// =============================================================================
// Export
// =============================================================================

export default function GraphicsMiniLabsCanvas() {
  return (
    <div style={{ width: '100%', height: '100vh', background: '#000' }}>
      <Canvas camera={{ position: [0, 0, 4], fov: 75 }}>
        <OrbitControls enablePan enableZoom enableRotate />
        <GraphicsMiniLabs />
      </Canvas>
      <Leva collapsed={false} />
    </div>
  );
}
