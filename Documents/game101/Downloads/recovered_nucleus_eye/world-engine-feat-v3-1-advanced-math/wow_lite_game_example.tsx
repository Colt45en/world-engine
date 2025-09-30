import { Environment, OrbitControls } from '@react-three/drei';
import { Canvas } from '@react-three/fiber';
import React from 'react';
import { WoWLiteTier4Bridge, useWoWLiteTier4Bridge } from './wow_lite_tier4_bridge';

// Example usage of WoW Lite Tier-4 Bridge in a React Three Fiber scene
const WoWLiteGameScene: React.FC = () => {
  const { bridgeState, lastOperator, handleStateChange, handleOperatorApplied } = useWoWLiteTier4Bridge();

  return (
    <div style={{ width: '100vw', height: '100vh' }}>
      {/* Game UI Overlay */}
      <div style={{
        position: 'absolute',
        top: '20px',
        left: '20px',
        zIndex: 100,
        background: 'rgba(0, 0, 0, 0.8)',
        padding: '15px',
        borderRadius: '10px',
        color: 'white',
        fontFamily: 'monospace'
      }}>
        <h3>🎮 WoW Lite Game</h3>
        {bridgeState && (
          <div>
            <div>Current Kappa: {bridgeState.kappa.toFixed(3)}</div>
            <div>Level: {bridgeState.level}</div>
            <div>Last Operator: {lastOperator}</div>
          </div>
        )}
      </div>

      {/* 3D Scene */}
      <Canvas camera={{ position: [5, 5, 5], fov: 60 }}>
        <Environment preset="sunset" />

        {/* Lighting */}
        <ambientLight intensity={0.4} />
        <directionalLight position={[10, 10, 5]} intensity={1} />

        {/* WoW Lite Tier-4 Bridge */}
        <WoWLiteTier4Bridge
          websocketUrl="ws://localhost:8080/ws"
          onStateChange={handleStateChange}
          onOperatorApplied={handleOperatorApplied}
          position={[0, 0, 0]}
        />

        {/* Additional game elements can go here */}
        <mesh position={[3, 1, 0]}>
          <boxGeometry args={[1, 1, 1]} />
          <meshStandardMaterial color="#8b5cf6" />
        </mesh>

        <mesh position={[-3, 1, 0]}>
          <sphereGeometry args={[0.5]} />
          <meshStandardMaterial color="#f59e0b" />
        </mesh>

        {/* Camera controls */}
        <OrbitControls enablePan={true} enableZoom={true} enableRotate={true} />
      </Canvas>
    </div>
  );
};

// Alternative: Simple integration example
export const SimpleWoWLiteIntegration: React.FC = () => {
  return (
    <Canvas>
      <WoWLiteTier4Bridge position={[0, 0, 0]} />
    </Canvas>
  );
};

export default WoWLiteGameScene;
