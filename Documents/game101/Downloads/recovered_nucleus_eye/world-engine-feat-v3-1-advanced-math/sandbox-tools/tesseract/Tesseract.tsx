// Tesseract.tsx - Standalone 4D Hypercube Visualization
// Interactive 4D tesseract with real-time rotation and stereographic projection

import React, { useMemo, useRef } from "react";
import * as THREE from "three";
import { Canvas, useFrame } from "@react-three/fiber";
import { OrbitControls, Html, Line } from "@react-three/drei";
import { Leva, useControls } from "leva";

// 4D Vector class for tesseract calculations
class Vector4 {
  constructor(public x: number, public y: number, public z: number, public w: number) {}

  clone(): Vector4 {
    return new Vector4(this.x, this.y, this.z, this.w);
  }

  multiplyScalar(scalar: number): Vector4 {
    return new Vector4(this.x * scalar, this.y * scalar, this.z * scalar, this.w * scalar);
  }
}

// 4D Rotation matrices
function rotate4D_XY(v: Vector4, angle: number): Vector4 {
  const cos = Math.cos(angle);
  const sin = Math.sin(angle);
  return new Vector4(
    cos * v.x - sin * v.y,
    sin * v.x + cos * v.y,
    v.z,
    v.w
  );
}

function rotate4D_XZ(v: Vector4, angle: number): Vector4 {
  const cos = Math.cos(angle);
  const sin = Math.sin(angle);
  return new Vector4(
    cos * v.x - sin * v.z,
    v.y,
    sin * v.x + cos * v.z,
    v.w
  );
}

function rotate4D_XW(v: Vector4, angle: number): Vector4 {
  const cos = Math.cos(angle);
  const sin = Math.sin(angle);
  return new Vector4(
    cos * v.x - sin * v.w,
    v.y,
    v.z,
    sin * v.x + cos * v.w
  );
}

function rotate4D_YZ(v: Vector4, angle: number): Vector4 {
  const cos = Math.cos(angle);
  const sin = Math.sin(angle);
  return new Vector4(
    v.x,
    cos * v.y - sin * v.z,
    sin * v.y + cos * v.z,
    v.w
  );
}

function rotate4D_YW(v: Vector4, angle: number): Vector4 {
  const cos = Math.cos(angle);
  const sin = Math.sin(angle);
  return new Vector4(
    v.x,
    cos * v.y - sin * v.w,
    v.z,
    sin * v.y + cos * v.w
  );
}

function rotate4D_ZW(v: Vector4, angle: number): Vector4 {
  const cos = Math.cos(angle);
  const sin = Math.sin(angle);
  return new Vector4(
    v.x,
    v.y,
    cos * v.z - sin * v.w,
    sin * v.z + cos * v.w
  );
}

// 4D to 3D projection (stereographic from 4D to 3D)
function project4Dto3D(v: Vector4, distance: number = 3): THREE.Vector3 {
  // Stereographic projection: (x,y,z,w) -> (x,y,z) * scale
  const scale = distance / (distance - v.w);
  return new THREE.Vector3(v.x * scale, v.y * scale, v.z * scale);
}

// Generate tesseract vertices (16 vertices of 4D hypercube)
function generateTesseractVertices(): Vector4[] {
  const vertices: Vector4[] = [];

  // Generate all 16 combinations of (±1, ±1, ±1, ±1)
  for (let i = 0; i < 16; i++) {
    const x = (i & 1) ? 1 : -1;
    const y = (i & 2) ? 1 : -1;
    const z = (i & 4) ? 1 : -1;
    const w = (i & 8) ? 1 : -1;
    vertices.push(new Vector4(x, y, z, w));
  }

  return vertices;
}

// Generate tesseract edges (32 edges connecting adjacent vertices)
function generateTesseractEdges(): [number, number][] {
  const edges: [number, number][] = [];

  // Two vertices are connected if they differ in exactly one coordinate
  for (let i = 0; i < 16; i++) {
    for (let bit = 0; bit < 4; bit++) {
      const j = i ^ (1 << bit); // Flip one bit
      if (i < j) { // Avoid duplicate edges
        edges.push([i, j]);
      }
    }
  }

  return edges;
}

// Main Tesseract Component
function TesseractVisualization() {
  const lineRef = useRef<THREE.BufferGeometry>(null!);
  const vertexGroupRef = useRef<THREE.Group>(null!);

  const {
    scale,
    rotationSpeed,
    projectionDistance,
    showVertices,
    showEdges,
    vertexSize,
    edgeColor,
    vertexColor,
    rotateXY,
    rotateXZ,
    rotateXW,
    rotateYZ,
    rotateYW,
    rotateZW
  } = useControls("Tesseract", {
    scale: { value: 1.2, min: 0.5, max: 3, step: 0.1 },
    rotationSpeed: { value: 0.8, min: 0, max: 3, step: 0.1 },
    projectionDistance: { value: 3, min: 1.5, max: 8, step: 0.1 },
    showVertices: true,
    showEdges: true,
    vertexSize: { value: 0.05, min: 0.01, max: 0.15, step: 0.01 },
    edgeColor: { value: "#80c8ff" },
    vertexColor: { value: "#ff8080" },
    rotateXY: { value: 0.7, min: 0, max: 2, step: 0.1 },
    rotateXZ: { value: 0.0, min: 0, max: 2, step: 0.1 },
    rotateXW: { value: 0.6, min: 0, max: 2, step: 0.1 },
    rotateYZ: { value: 0.0, min: 0, max: 2, step: 0.1 },
    rotateYW: { value: 0.0, min: 0, max: 2, step: 0.1 },
    rotateZW: { value: 0.8, min: 0, max: 2, step: 0.1 }
  });

  // Generate static geometry
  const { vertices, edges } = useMemo(() => {
    return {
      vertices: generateTesseractVertices(),
      edges: generateTesseractEdges()
    };
  }, []);

  // Animation loop
  useFrame(({ clock }) => {
    const time = clock.getElapsedTime() * rotationSpeed;

    // Apply 4D rotations to vertices
    const rotatedVertices = vertices.map(vertex => {
      let v = vertex.clone();

      // Apply rotations in different 4D planes
      if (rotateXY > 0) v = rotate4D_XY(v, time * rotateXY);
      if (rotateXZ > 0) v = rotate4D_XZ(v, time * rotateXZ);
      if (rotateXW > 0) v = rotate4D_XW(v, time * rotateXW);
      if (rotateYZ > 0) v = rotate4D_YZ(v, time * rotateYZ);
      if (rotateYW > 0) v = rotate4D_YW(v, time * rotateYW);
      if (rotateZW > 0) v = rotate4D_ZW(v, time * rotateZW);

      return v;
    });

    // Project to 3D and update edge geometry
    if (showEdges && lineRef.current) {
      const positions: number[] = [];

      edges.forEach(([i, j]) => {
        const v1 = project4Dto3D(rotatedVertices[i].multiplyScalar(scale), projectionDistance);
        const v2 = project4Dto3D(rotatedVertices[j].multiplyScalar(scale), projectionDistance);

        positions.push(v1.x, v1.y, v1.z);
        positions.push(v2.x, v2.y, v2.z);
      });

      lineRef.current.setAttribute('position', new THREE.BufferAttribute(new Float32Array(positions), 3));
    }

    // Update vertex positions
    if (showVertices && vertexGroupRef.current) {
      vertexGroupRef.current.children.forEach((child, i) => {
        if (i < rotatedVertices.length) {
          const pos = project4Dto3D(rotatedVertices[i].multiplyScalar(scale), projectionDistance);
          child.position.set(pos.x, pos.y, pos.z);
        }
      });
    }
  });

  return (
    <group>
      {/* Edges */}
      {showEdges && (
        <lineSegments>
          <bufferGeometry ref={lineRef}>
            <bufferAttribute
              attach="attributes-position"
              array={new Float32Array(edges.length * 6)}
              count={edges.length * 2}
              itemSize={3}
            />
          </bufferGeometry>
          <lineBasicMaterial color={edgeColor} />
        </lineSegments>
      )}

      {/* Vertices */}
      {showVertices && (
        <group ref={vertexGroupRef}>
          {vertices.map((_, i) => (
            <mesh key={i}>
              <sphereGeometry args={[vertexSize, 8, 6]} />
              <meshBasicMaterial color={vertexColor} />
            </mesh>
          ))}
        </group>
      )}
    </group>
  );
}

// Information Panel
function InfoPanel() {
  return (
    <Html position={[0, 0, 0]} transform={false} prepend>
      <div
        style={{
          position: 'absolute',
          top: 20,
          left: 20,
          padding: 15,
          background: 'rgba(10, 15, 25, 0.9)',
          border: '1px solid rgba(255, 255, 255, 0.1)',
          borderRadius: 12,
          color: '#e0e7ff',
          fontSize: 14,
          minWidth: 300,
          maxWidth: 400,
          pointerEvents: 'none'
        }}
      >
        <div style={{ fontWeight: 700, marginBottom: 10, fontSize: 16 }}>
          4D Tesseract (Hypercube)
        </div>

        <div style={{ display: 'grid', gap: 6, fontSize: 13 }}>
          <div><strong>Vertices:</strong> 16 (all combinations of ±1 in 4D)</div>
          <div><strong>Edges:</strong> 32 (connecting vertices differing in 1 coordinate)</div>
          <div><strong>Projection:</strong> 4D → 3D stereographic projection</div>
          <div><strong>Rotation Planes:</strong> 6 possible (XY, XZ, XW, YZ, YW, ZW)</div>
        </div>

        <div style={{ marginTop: 12, padding: 8, background: 'rgba(255, 255, 255, 0.05)', borderRadius: 6 }}>
          <div style={{ fontSize: 12, opacity: 0.9 }}>
            <div><strong>Understanding 4D:</strong></div>
            <div>• Each vertex has 4 coordinates: (±1, ±1, ±1, ±1)</div>
            <div>• Edges connect vertices differing by exactly one coordinate</div>
            <div>• 4D rotation occurs in 2D planes within 4D space</div>
            <div>• Projection brings 4D structure into viewable 3D space</div>
          </div>
        </div>
      </div>
    </Html>
  );
}

// Demo Scene
function DemoScene() {
  return (
    <group>
      {/* Lighting */}
      <ambientLight intensity={0.4} />
      <directionalLight position={[5, 5, 5]} intensity={0.6} />

      {/* Tesseract */}
      <TesseractVisualization />

      {/* Reference grid */}
      <gridHelper args={[6, 20]} position={[0, -2, 0]} opacity={0.3} />

      {/* Info panel */}
      <InfoPanel />
    </group>
  );
}

// Main App
export default function TesseractApp() {
  return (
    <div style={{ width: '100%', height: '100vh', background: '#0a0f1a' }}>
      <Leva collapsed={false} />
      <Canvas camera={{ position: [3, 2, 5], fov: 60 }}>
        <color attach="background" args={["#0a0f1a"]} />
        <DemoScene />
        <OrbitControls enablePan={true} />
      </Canvas>
    </div>
  );
}
