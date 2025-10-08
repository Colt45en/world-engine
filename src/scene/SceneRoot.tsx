import React from 'react'
import { OrbitControls } from '@react-three/drei'
import { useUI } from '../state/ui'
import EntityLayer from './layers/EntityLayer'
import LightLayer from './layers/LightLayer'
import HelpersLayer from './layers/HelpersLayer'

export default function SceneRoot() {
  const { reducedMotion } = useUI()

  // Adjust camera controls based on motion preferences
  const damping = reducedMotion ? 0.02 : 0.08
  const rotateSpeed = reducedMotion ? 0.2 : 0.8
  const zoomSpeed = reducedMotion ? 0.6 : 1
  const panSpeed = reducedMotion ? 0.6 : 1

  return (
    <>
      {/* Camera Controls */}
      <OrbitControls
        enableDamping
        dampingFactor={damping}
        rotateSpeed={rotateSpeed}
        zoomSpeed={zoomSpeed}
        panSpeed={panSpeed}
        enableRotate={true}
        enableZoom={true}
        enablePan={true}
        maxPolarAngle={Math.PI}
        minDistance={2}
        maxDistance={100}
        makeDefault
      />

      {/* Scene Layers */}
      <LightLayer />
      <EntityLayer />
      <HelpersLayer />
    </>
  )
}
