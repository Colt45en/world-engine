import { useScene, Entity } from '../../state/store';
import { useMemo, useRef } from 'react';
import { useFrame } from '@react-three/fiber';
import { Text } from '@react-three/drei';
import * as THREE from 'three';

function CubeMesh({ entity }: { entity: Entity }) {
  const ref = useRef<THREE.Mesh>(null);
  const { selected, selectOnly, animationSpeed } = useScene();
  const isSelected = selected.includes(entity.id);

  const material = useMemo(() => {
    return new THREE.MeshStandardMaterial({
      color: isSelected ? '#66ccff' : entity.color,
      roughness: 0.7,
      metalness: 0.1
    });
  }, [isSelected, entity.color]);

  // Original cube animation logic
  useFrame((state) => {
    if (!ref.current) return;
    const t = state.clock.getElapsedTime();
    const bob = Math.sin(t + entity.offset) * 0.1 * animationSpeed;
    ref.current.position.set(
      entity.position[0],
      entity.position[1] + bob,
      entity.position[2]
    );
    ref.current.rotation.x += 0.01 * 0.3 * animationSpeed;
    ref.current.rotation.y += 0.01 * 0.5 * animationSpeed;
  });

  return (
    <group>
      <mesh
        ref={ref}
        castShadow
        receiveShadow
        onClick={(e) => {
          e.stopPropagation();
          selectOnly(entity.id);
        }}
        userData={{ entityId: entity.id }}
      >
        <boxGeometry args={[1, 1, 1]} />
        <primitive object={material} attach="material" />
      </mesh>
      <Text
        position={[entity.position[0], -0.8, entity.position[2]]}
        fontSize={0.18}
        color="#ffffff"
        anchorX="center"
        anchorY="middle"
      >
        {entity.name || entity.id}
      </Text>
    </group>
  );
}

function GroundPlane() {
  return (
    <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, 0, 0]} receiveShadow>
      <planeGeometry args={[200, 200]} />
      <meshStandardMaterial color="#111" roughness={0.8} metalness={0.2} />
    </mesh>
  );
}

export default function EntityLayer() {
  const { entities } = useScene();
  const cubes = Object.values(entities).filter(entity => entity.kind === 'cube');

  return (
    <group name="entity-layer">
      <GroundPlane />
      {cubes.map(entity => (
        <CubeMesh key={entity.id} entity={entity} />
      ))}
    </group>
  );
}
