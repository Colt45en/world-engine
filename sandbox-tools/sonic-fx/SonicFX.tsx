// SonicFX.tsx - Standalone Sonic Speed Overlay System
// Optimized cinematic speed perception overlay with motion streaks and radial effects

import React, { useRef } from "react";
import * as THREE from "three";
import { Canvas, useFrame, extend } from "@react-three/fiber";
import { OrbitControls, shaderMaterial } from "@react-three/drei";
import { Leva, useControls } from "leva";

// Optimized Sonic Shader Material
const SonicMaterial = shaderMaterial(
  {
    uTime: 0,
    uSpeed: 0,
    uIntensity: 0.9,
    uBands: 180.0,
    uRotation: 6.0,
    uFalloff: 1.1,
    uCore: 0.15,
    uColor: new THREE.Vector3(0.6, 0.85, 1.0)
  },
  /* vertex */ `
    varying vec2 vUv;
    void main() {
      vUv = uv;
      gl_Position = vec4(position, 1.0);
    }
  `,
  /* fragment */ `
    varying vec2 vUv;
    uniform float uTime;
    uniform float uSpeed;
    uniform float uIntensity;
    uniform float uBands;
    uniform float uRotation;
    uniform float uFalloff;
    uniform float uCore;
    uniform vec3 uColor;

    void main() {
      vec2 uv = vUv * 2.0 - 1.0;
      float r = length(uv);
      float angle = atan(uv.y, uv.x);

      // Dynamic band calculation based on speed
      float bands = uBands + 220.0 * uSpeed;

      // Rotating stripe pattern
      float stripe = smoothstep(0.60, 1.0, abs(sin(angle * bands + uTime * uRotation * uSpeed)));

      // Radial falloff
      float falloff = smoothstep(uFalloff, uCore, r);

      // Central spark effect
      float spark = pow(max(0.0, 1.0 - r), 3.0);

      // Combine effects
      float intensity = (stripe * falloff + spark) * uIntensity;

      // Color output with customizable tint
      gl_FragColor = vec4(uColor * intensity, intensity);
    }
  `
);

// Extend Three.js to recognize our material
extend({ SonicMaterial });

// TypeScript declaration for JSX
declare global {
  namespace JSX {
    interface IntrinsicElements {
      sonicMaterial: any;
    }
  }
}

// Sonic Overlay Component
function SonicOverlay() {
  const materialRef = useRef<THREE.ShaderMaterial>(null!);

  const {
    speed,
    intensity,
    bands,
    rotation,
    falloff,
    core,
    colorR,
    colorG,
    colorB,
    enabled
  } = useControls("Sonic FX", {
    enabled: true,
    speed: { value: 2.5, min: 0, max: 10, step: 0.1 },
    intensity: { value: 0.9, min: 0, max: 2, step: 0.05 },
    bands: { value: 180, min: 50, max: 500, step: 10 },
    rotation: { value: 6.0, min: 0, max: 20, step: 0.5 },
    falloff: { value: 1.1, min: 0.5, max: 2.0, step: 0.05 },
    core: { value: 0.15, min: 0.05, max: 0.5, step: 0.01 },
    colorR: { value: 0.6, min: 0, max: 1, step: 0.01 },
    colorG: { value: 0.85, min: 0, max: 1, step: 0.01 },
    colorB: { value: 1.0, min: 0, max: 1, step: 0.01 },
  });

  useFrame((state) => {
    if (materialRef.current && enabled) {
      materialRef.current.uniforms.uTime.value = state.clock.elapsedTime;
      materialRef.current.uniforms.uSpeed.value = speed;
      materialRef.current.uniforms.uIntensity.value = intensity;
      materialRef.current.uniforms.uBands.value = bands;
      materialRef.current.uniforms.uRotation.value = rotation;
      materialRef.current.uniforms.uFalloff.value = falloff;
      materialRef.current.uniforms.uCore.value = core;
      materialRef.current.uniforms.uColor.value.set(colorR, colorG, colorB);
    }
  });

  if (!enabled) return null;

  return (
    <mesh renderOrder={1000}>
      <planeGeometry args={[2, 2]} />
      <sonicMaterial
        ref={materialRef}
        transparent
        depthTest={false}
        depthWrite={false}
      />
    </mesh>
  );
}

// Demo Scene with Test Objects
function DemoScene() {
  const cubeRef = useRef<THREE.Mesh>(null!);
  const torusRef = useRef<THREE.Mesh>(null!);

  useFrame((state) => {
    if (cubeRef.current) {
      cubeRef.current.rotation.x = state.clock.elapsedTime * 0.5;
      cubeRef.current.rotation.y = state.clock.elapsedTime * 0.3;
    }
    if (torusRef.current) {
      torusRef.current.rotation.x = state.clock.elapsedTime * 0.7;
      torusRef.current.rotation.z = state.clock.elapsedTime * 0.4;
    }
  });

  return (
    <group>
      {/* Ambient lighting */}
      <ambientLight intensity={0.4} />
      <directionalLight position={[5, 5, 5]} intensity={0.8} />

      {/* Demo objects */}
      <mesh ref={cubeRef} position={[-1.5, 0, 0]}>
        <boxGeometry args={[1, 1, 1]} />
        <meshStandardMaterial color="#ff6b6b" />
      </mesh>

      <mesh ref={torusRef} position={[1.5, 0, 0]}>
        <torusGeometry args={[0.6, 0.3, 16, 32]} />
        <meshStandardMaterial color="#4ecdc4" />
      </mesh>

      <mesh position={[0, -1, 0]} rotation={[-Math.PI / 2, 0, 0]}>
        <planeGeometry args={[6, 6]} />
        <meshStandardMaterial color="#2c3e50" />
      </mesh>

      {/* Sonic Overlay */}
      <SonicOverlay />
    </group>
  );
}

// Main App Component
export default function SonicFXApp() {
  return (
    <div style={{ width: '100%', height: '100vh', background: '#1a1a1a' }}>
      <Leva collapsed={false} />
      <Canvas camera={{ position: [0, 0, 5], fov: 60 }}>
        <color attach="background" args={["#1a1a1a"]} />
        <DemoScene />
        <OrbitControls enablePan={false} />
      </Canvas>
    </div>
  );
}
