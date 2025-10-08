import React, { Suspense } from 'react'
import * as THREE from 'three'
import { Canvas } from '@react-three/fiber'
import { OrbitControls, Html } from '@react-three/drei'
import GeoHexFlowerScene from './GeoHexFlowerScene.jsx'

const wrapperStyle = {
  width: '100%',
  height: '100vh',
  margin: 0,
  background: 'radial-gradient(circle at 50% 10%, #14273f 0%, #061222 55%, #02060c 100%)',
  color: '#dff6ff',
  fontFamily: '"Inter", "Segoe UI", sans-serif',
  position: 'relative',
  overflow: 'hidden'
} as const

const overlayStyle = {
  pointerEvents: 'none',
  position: 'absolute',
  top: '2rem',
  left: '50%',
  transform: 'translateX(-50%)',
  textTransform: 'uppercase',
  letterSpacing: '0.4em',
  fontSize: '0.75rem',
  color: '#7fd3ff',
  opacity: 0.65
} as const

function LoadingOverlay() {
  return (
    <Html center>
      <div
        style={{
          padding: '1rem 1.5rem',
          background: 'rgba(7, 16, 27, 0.82)',
          border: '1px solid rgba(120, 220, 255, 0.35)',
          borderRadius: '999px',
          backdropFilter: 'blur(8px)',
          textTransform: 'uppercase',
          letterSpacing: '0.4em',
          fontSize: '0.65rem',
          color: '#9fe8ff',
          boxShadow: '0 0 24px rgba(90, 200, 255, 0.3)'
        }}
      >
        Stabilizing lattice
      </div>
    </Html>
  )
}

export default function App() {
  return (
    <div style={wrapperStyle}>
      <Canvas
        shadows
        camera={{ position: [0, 5.5, 11], fov: 45, near: 0.1, far: 60 }}
        dpr={[1, 2]}
        gl={{
          antialias: true,
          alpha: false,
          toneMapping: THREE.ACESFilmicToneMapping,
          outputColorSpace: THREE.SRGBColorSpace
        }}
      >
        <color attach="background" args={[0x020509]} />
        <Suspense fallback={<LoadingOverlay />}>
          <GeoHexFlowerScene />
          <OrbitControls
            enableDamping
            dampingFactor={0.08}
            enablePan={false}
            minDistance={6}
            maxDistance={14}
            rotateSpeed={0.8}
          />
        </Suspense>
      </Canvas>
      <div style={overlayStyle}>GeoHex Flower Observatory</div>
    </div>
  )
}
