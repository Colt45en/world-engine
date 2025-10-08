import React, { Suspense } from 'react'
import { Canvas } from '@react-three/fiber'
import SceneRoot from './scene/SceneRoot'
import { useUI } from './state/ui'

function LoadingFallback() {
  return (
    <div className="flex items-center justify-center h-full bg-gray-900 text-white">
      <div className="text-center">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500 mx-auto mb-4"></div>
        <p className="text-sm text-gray-300">Loading 3D Scene...</p>
      </div>
    </div>
  )
}

export default function AppCanvas() {
  const { reducedMotion } = useUI()

  return (
    <div className="h-full w-full">
      <Suspense fallback={<LoadingFallback />}>
        <Canvas
          camera={{
            position: [5, 5, 5],
            fov: 75,
            near: 0.1,
            far: 1000
          }}
          shadows
          gl={{
            antialias: !reducedMotion, // Reduce GPU load for motion-sensitive users
            alpha: true,
            powerPreference: reducedMotion ? 'low-power' : 'high-performance'
          }}
          dpr={reducedMotion ? 1 : [1, 2]} // Lower DPR for reduced motion
        >
          <SceneRoot />
        </Canvas>
      </Suspense>
    </div>
  )
}
