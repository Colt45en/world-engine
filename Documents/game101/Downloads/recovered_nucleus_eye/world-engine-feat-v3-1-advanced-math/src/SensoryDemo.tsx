// Simple sensory overlay demo for testing integration
import * as React from 'react';
import * as THREE from 'three';
import { Canvas } from '@react-three/fiber';
import { OrbitControls, Environment } from '@react-three/drei';
import { SensoryOverlay } from './sensory/SensoryOverlay';
import { useSensoryMoment } from './sensory/useSensoryMoment';

export function SensoryDemo() {
  return (
    <div className="w-full h-screen bg-gray-900">
      <div className="flex h-full">
        {/* Controls Panel */}
        <div className="w-80 bg-black/50 p-4 overflow-y-auto">
          <h2 className="text-white text-lg font-semibold mb-4">Sensory Integration Demo</h2>
          <SensoryControls />
        </div>

        {/* 3D Scene */}
        <div className="flex-1">
          <Canvas camera={{ position: [4, 3, 6], fov: 45 }}>
            <color attach="background" args={['#0a0f1a']} />
            <Environment preset="night" />
            <SceneDemo />
            <OrbitControls makeDefault />
          </Canvas>
        </div>
      </div>
    </div>
  );
}

function SensoryControls() {
  const { moment, setPreset, setPerspective, availablePresets } = useSensoryMoment();

  return (
    <div className="space-y-4 text-white">
      <div>
        <label className="block text-sm font-medium text-gray-300 mb-2">
          Sensory Preset
        </label>
        <select
          value={moment.id.split('-')[0]}
          onChange={(e) => setPreset(e.target.value as any)}
          className="w-full px-3 py-2 bg-gray-800 border border-gray-600 rounded-md text-white"
        >
          {availablePresets.map(preset => (
            <option key={preset} value={preset}>
              {preset.charAt(0).toUpperCase() + preset.slice(1)}
            </option>
          ))}
        </select>
      </div>

      <div>
        <label className="block text-sm font-medium text-gray-300 mb-2">
          Perspective
        </label>
        <select
          value={moment.perspective}
          onChange={(e) => setPerspective(e.target.value as any)}
          className="w-full px-3 py-2 bg-gray-800 border border-gray-600 rounded-md text-white"
        >
          <option value="attuned">Attuned</option>
          <option value="oblivious">Oblivious</option>
          <option value="object">Object</option>
        </select>
      </div>

      <div className="mt-6">
        <h3 className="text-sm font-medium text-gray-300 mb-2">Current Moment</h3>
        <div className="text-xs text-gray-400 space-y-1">
          <div><strong>ID:</strong> {moment.id}</div>
          <div><strong>Label:</strong> {moment.label}</div>
          <div><strong>Channels:</strong> {moment.details.length}</div>
        </div>
      </div>

      <div className="mt-6">
        <h3 className="text-sm font-medium text-gray-300 mb-2">Channel Details</h3>
        <div className="space-y-2 max-h-60 overflow-y-auto">
          {moment.details.map((detail, i) => (
            <div key={i} className="text-xs bg-gray-800/50 p-2 rounded border border-gray-700">
              <div className="flex justify-between items-center mb-1">
                <span className="font-medium capitalize text-gray-300">
                  {detail.channel}
                </span>
                <span className="text-gray-400">
                  {Math.round((detail.strength ?? 0.5) * 100)}%
                </span>
              </div>
              <div className="text-gray-400 leading-relaxed">
                {detail.description}
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

function SceneDemo() {
  const meshRef = React.useRef<THREE.Mesh>(null);
  const { moment } = useSensoryMoment();

  return (
    <>
      {/* Test object */}
      <mesh ref={meshRef} position={[0, 0, 0]} castShadow receiveShadow>
        <icosahedronGeometry args={[0.8, 2]} />
        <meshStandardMaterial
          color="#4ecdc4"
          roughness={0.3}
          metalness={0.7}
          emissive="#1a4f4a"
          emissiveIntensity={0.1}
        />
      </mesh>

      {/* Ground plane */}
      <mesh position={[0, -1.5, 0]} rotation={[-Math.PI / 2, 0, 0]} receiveShadow>
        <planeGeometry args={[10, 10]} />
        <meshStandardMaterial color="#1a1a2e" roughness={0.9} metalness={0.0} />
      </mesh>

      {/* Lighting */}
      <ambientLight intensity={0.3} />
      <directionalLight
        position={[5, 8, 5]}
        intensity={1.2}
        castShadow
        shadow-mapSize-width={2048}
        shadow-mapSize-height={2048}
      />
      <pointLight position={[-3, 3, -3]} intensity={0.5} color="#6366f1" />

      {/* Sensory overlay */}
      <SensoryOverlay
        moment={moment}
        attachTo={meshRef.current ?? undefined}
        visible={true}
      />
    </>
  );
}
