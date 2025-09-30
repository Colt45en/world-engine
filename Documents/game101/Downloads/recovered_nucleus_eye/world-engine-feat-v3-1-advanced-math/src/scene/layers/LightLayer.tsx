import React from 'react'
import { useUI } from '../../state/ui'

export default function LightLayer() {
  const { reducedMotion } = useUI()

  // Adjust light intensity based on motion preferences
  // Reduced motion users may benefit from softer, more stable lighting
  const intensityMultiplier = reducedMotion ? 0.8 : 1.0

  return (
    <>
      {/* Ambient light for base illumination */}
      <ambientLight intensity={0.4 * intensityMultiplier} color="#ffffff" />

      {/* Main directional light */}
      <directionalLight
        position={[10, 10, 5]}
        intensity={1.2 * intensityMultiplier}
        color="#ffffff"
        castShadow
        shadow-mapSize-width={2048}
        shadow-mapSize-height={2048}
        shadow-camera-far={50}
        shadow-camera-left={-10}
        shadow-camera-right={10}
        shadow-camera-top={10}
        shadow-camera-bottom={-10}
      />

      {/* Fill light */}
      <directionalLight
        position={[-5, 5, -5]}
        intensity={0.6 * intensityMultiplier}
        color="#4a90e2"
      />

      {/* Rim light for depth */}
      <pointLight
        position={[0, 10, -10]}
        intensity={0.8 * intensityMultiplier}
        color="#ff6b6b"
        distance={20}
        decay={2}
      />
    </>
  )
}
