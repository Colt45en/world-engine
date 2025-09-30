import React, { Suspense } from 'react';
import { Canvas } from '@react-three/fiber';
import SceneRoot from './scene/SceneRoot';

// Loading fallback component
const LoadingFallback = () => (
  <div className="flex items-center justify-center h-full bg-gray-900 text-white">
    <div className="text-center">
      <div className="animate-spin w-8 h-8 border-2 border-blue-400 border-t-transparent rounded-full mx-auto mb-2"></div>
      <div>Loading 3D Scene...</div>
    </div>
  </div>
);

export default function AppCanvas() {
  return (
    <Canvas
      shadows
      camera={{ position: [0, 1.5, 6], fov: 50 }}
      gl={{ antialias: true, alpha: false }}
      dpr={[1, 2]}
    >
      <Suspense fallback={null}>
        <SceneRoot />
      </Suspense>
    </Canvas>
  );
}
