import { useScene } from '../../state/store';
import { useState, useMemo } from 'react';
import { useGroundPointer } from '../hooks/useGroundPointer';
import * as THREE from 'three';

function SpotlightMarker() {
  const { spotlight } = useScene();
  const { enabled, position } = spotlight;

  if (!enabled) return null;

  return (
    <mesh position={position} castShadow>
      <sphereGeometry args={[0.12, 16, 16]} />
      <meshStandardMaterial emissive="#ffff99" color="#ffff99" />
    </mesh>
  );
}

function Spotlight() {
  const { spotlight } = useScene();
  const { enabled, position } = spotlight;

  if (!enabled) return null;

  return (
    <pointLight
      position={position}
      intensity={1.2}
      distance={30}
      decay={2}
      color="#fff6cc"
      castShadow
    />
  );
}

export default function LightLayer() {
  const { spotlight, setSpotlightPosition } = useScene();
  const [draggingSpot, setDraggingSpot] = useState(false);

  // Ground pointer logic for dragging spotlight
  const handleGroundPointer = useGroundPointer((position) => {
    setSpotlightPosition(position);
  });

  return (
    <group name="light-layer">
      <Spotlight />
      <SpotlightMarker />

      {/* Drag plane for spotlight */}
      <mesh
        rotation={[-Math.PI / 2, 0, 0]}
        onPointerDown={(e) => {
          if (spotlight.enabled) {
            setDraggingSpot(true);
            handleGroundPointer(e);
          }
        }}
        onPointerMove={(e) => {
          if (spotlight.enabled && draggingSpot) {
            handleGroundPointer(e);
          }
        }}
        onPointerUp={() => setDraggingSpot(false)}
      >
        <planeGeometry args={[200, 200]} />
        <meshBasicMaterial visible={false} />
      </mesh>
    </group>
  );
}
