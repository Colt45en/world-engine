import React, { useRef, useState, useEffect } from 'react';
import { Html, Text, Billboard, Sphere, Box, Plane } from '@react-three/drei';
import { useFrame, useThree } from '@react-three/fiber';
import * as THREE from 'three';
import { useSpring, MathUtils } from './utilities';

// =============================================================================
// UI Components
// =============================================================================

interface FloatingPanelProps {
  position?: [number, number, number];
  children: React.ReactNode;
  title?: string;
  width?: number;
  height?: number;
  background?: string;
  borderColor?: string;
  opacity?: number;
}

export const FloatingPanel: React.FC<FloatingPanelProps> = ({
  position = [0, 0, 0],
  children,
  title,
  width = 300,
  height = 200,
  background = 'rgba(0, 0, 0, 0.9)',
  borderColor = '#00ff88',
  opacity = 0.95
}) => {
  return (
    <Html
      position={position}
      style={{
        width: `${width}px`,
        minHeight: `${height}px`,
        background,
        border: `1px solid ${borderColor}`,
        borderRadius: '12px',
        padding: '20px',
        fontFamily: 'monospace',
        color: borderColor,
        backdropFilter: 'blur(10px)',
        opacity
      }}
    >
      {title && (
        <h3 style={{
          margin: '0 0 15px 0',
          fontSize: '16px',
          textAlign: 'center',
          borderBottom: `1px solid ${borderColor}`,
          paddingBottom: '10px'
        }}>
          {title}
        </h3>
      )}
      {children}
    </Html>
  );
};

interface InfoDisplayProps {
  info: Record<string, any>;
  position?: [number, number, number];
  color?: string;
  fontSize?: number;
}

export const InfoDisplay: React.FC<InfoDisplayProps> = ({
  info,
  position = [0, 2, 0],
  color = '#00ff88',
  fontSize = 0.15
}) => {
  return (
    <Billboard position={position}>
      <Text
        fontSize={fontSize}
        color={color}
        anchorX="left"
        anchorY="top"
        font="/fonts/monospace.woff"
        maxWidth={10}
      >
        {Object.entries(info)
          .map(([key, value]) => `${key}: ${typeof value === 'number' ? value.toFixed(2) : value}`)
          .join('\n')}
      </Text>
    </Billboard>
  );
};

// =============================================================================
// Visualization Components
// =============================================================================

interface AudioVisualizerProps {
  audioData: number[];
  position?: [number, number, number];
  scale?: number;
  color?: string;
}

export const AudioVisualizer: React.FC<AudioVisualizerProps> = ({
  audioData,
  position = [0, 0, 0],
  scale = 1,
  color = '#ff00ff'
}) => {
  const meshRefs = useRef<THREE.Mesh[]>([]);

  useFrame(() => {
    meshRefs.current.forEach((mesh, i) => {
      if (mesh && audioData[i] !== undefined) {
        mesh.scale.y = 0.1 + audioData[i] * scale;
        mesh.position.y = (mesh.scale.y / 2) - 0.5;
      }
    });
  });

  return (
    <group position={position}>
      {audioData.map((_, i) => (
        <Box
          key={i}
          ref={(ref) => {
            if (ref) meshRefs.current[i] = ref;
          }}
          position={[i * 0.1 - (audioData.length * 0.05), 0, 0]}
          args={[0.05, 1, 0.05]}
        >
          <meshStandardMaterial color={color} />
        </Box>
      ))}
    </group>
  );
};

interface ParticleSystemProps {
  count?: number;
  position?: [number, number, number];
  spread?: number;
  speed?: number;
  color?: string;
  size?: number;
}

export const ParticleSystem: React.FC<ParticleSystemProps> = ({
  count = 100,
  position = [0, 0, 0],
  spread = 5,
  speed = 0.01,
  color = '#ffffff',
  size = 0.1
}) => {
  const meshRef = useRef<THREE.InstancedMesh>(null);
  const particlesRef = useRef<Array<{
    position: THREE.Vector3;
    velocity: THREE.Vector3;
    life: number;
  }>>([]);

  // Initialize particles
  useEffect(() => {
    particlesRef.current = Array.from({ length: count }, () => ({
      position: new THREE.Vector3(
        (Math.random() - 0.5) * spread,
        (Math.random() - 0.5) * spread,
        (Math.random() - 0.5) * spread
      ),
      velocity: new THREE.Vector3(
        (Math.random() - 0.5) * speed,
        (Math.random() - 0.5) * speed,
        (Math.random() - 0.5) * speed
      ),
      life: Math.random()
    }));
  }, [count, spread, speed]);

  useFrame((_, delta) => {
    if (meshRef.current) {
      const dummy = new THREE.Object3D();

      particlesRef.current.forEach((particle, i) => {
        // Update particle
        particle.position.add(particle.velocity);
        particle.life += delta * 0.5;

        // Reset if out of bounds or old
        if (particle.life > 2 || particle.position.length() > spread) {
          particle.position.set(
            (Math.random() - 0.5) * spread,
            (Math.random() - 0.5) * spread,
            (Math.random() - 0.5) * spread
          );
          particle.life = 0;
        }

        // Update matrix
        dummy.position.copy(particle.position);
        dummy.scale.setScalar(Math.max(0, 1 - particle.life / 2));
        dummy.updateMatrix();

        meshRef.current!.setMatrixAt(i, dummy.matrix);
      });

      meshRef.current.instanceMatrix.needsUpdate = true;
    }
  });

  return (
    <instancedMesh
      ref={meshRef}
      args={[undefined, undefined, count]}
      position={position}
    >
      <sphereGeometry args={[size, 8, 8]} />
      <meshBasicMaterial color={color} transparent opacity={0.7} />
    </instancedMesh>
  );
};

// =============================================================================
// Interactive Components
// =============================================================================

interface DraggableProps {
  children: React.ReactNode;
  onDrag?: (position: THREE.Vector3) => void;
  onDragStart?: () => void;
  onDragEnd?: () => void;
  constraints?: {
    x?: [number, number];
    y?: [number, number];
    z?: [number, number];
  };
}

export const Draggable: React.FC<DraggableProps> = ({
  children,
  onDrag,
  onDragStart,
  onDragEnd,
  constraints
}) => {
  const [isDragging, setIsDragging] = useState(false);
  const { camera, raycaster } = useThree();
  const dragPlaneRef = useRef<THREE.Plane>(new THREE.Plane(new THREE.Vector3(0, 0, 1), 0));

  const handlePointerDown = (event: any) => {
    event.stopPropagation();
    setIsDragging(true);
    onDragStart?.();
  };

  const handlePointerMove = (event: any) => {
    if (!isDragging) return;

    const intersection = new THREE.Vector3();
    raycaster.ray.intersectPlane(dragPlaneRef.current, intersection);

    // Apply constraints
    if (constraints) {
      if (constraints.x) {
        intersection.x = MathUtils.clamp(intersection.x, constraints.x[0], constraints.x[1]);
      }
      if (constraints.y) {
        intersection.y = MathUtils.clamp(intersection.y, constraints.y[0], constraints.y[1]);
      }
      if (constraints.z) {
        intersection.z = MathUtils.clamp(intersection.z, constraints.z[0], constraints.z[1]);
      }
    }

    onDrag?.(intersection);
  };

  const handlePointerUp = () => {
    setIsDragging(false);
    onDragEnd?.();
  };

  return (
    <group
      onPointerDown={handlePointerDown}
      onPointerMove={handlePointerMove}
      onPointerUp={handlePointerUp}
    >
      {children}
    </group>
  );
};

interface SliderProps {
  value: number;
  onChange: (value: number) => void;
  min?: number;
  max?: number;
  position?: [number, number, number];
  length?: number;
  color?: string;
  handleColor?: string;
}

export const Slider3D: React.FC<SliderProps> = ({
  value,
  onChange,
  min = 0,
  max = 1,
  position = [0, 0, 0],
  length = 2,
  color = '#444',
  handleColor = '#00ff88'
}) => {
  const normalizedValue = (value - min) / (max - min);
  const handlePosition: [number, number, number] = [
    position[0] + (normalizedValue - 0.5) * length,
    position[1],
    position[2]
  ];

  return (
    <group position={position}>
      {/* Track */}
      <Box args={[length, 0.02, 0.02]} position={[0, 0, 0]}>
        <meshStandardMaterial color={color} />
      </Box>

      {/* Handle */}
      <Draggable
        constraints={{ x: [position[0] - length/2, position[0] + length/2] }}
        onDrag={(pos) => {
          const t = (pos.x - (position[0] - length/2)) / length;
          const newValue = min + t * (max - min);
          onChange(newValue);
        }}
      >
        <Sphere args={[0.05]} position={[handlePosition[0] - position[0], 0, 0]}>
          <meshStandardMaterial color={handleColor} />
        </Sphere>
      </Draggable>
    </group>
  );
};

// =============================================================================
// Effect Components
// =============================================================================

interface GridProps {
  size?: number;
  divisions?: number;
  color?: string;
  position?: [number, number, number];
  rotation?: [number, number, number];
  opacity?: number;
}

export const Grid3D: React.FC<GridProps> = ({
  size = 10,
  divisions = 10,
  color = '#444',
  position = [0, 0, 0],
  rotation = [0, 0, 0],
  opacity = 0.5
}) => {
  const meshRef = useRef<THREE.LineSegments>(null);

  useEffect(() => {
    if (meshRef.current) {
      const geometry = new THREE.BufferGeometry();
      const vertices = [];
      const step = size / divisions;
      const half = size / 2;

      // Horizontal lines
      for (let i = 0; i <= divisions; i++) {
        const y = -half + i * step;
        vertices.push(-half, y, 0, half, y, 0);
      }

      // Vertical lines
      for (let i = 0; i <= divisions; i++) {
        const x = -half + i * step;
        vertices.push(x, -half, 0, x, half, 0);
      }

      geometry.setAttribute('position', new THREE.Float32BufferAttribute(vertices, 3));
      meshRef.current.geometry = geometry;
    }
  }, [size, divisions]);

  return (
    <lineSegments ref={meshRef} position={position} rotation={rotation}>
      <bufferGeometry />
      <lineBasicMaterial color={color} transparent opacity={opacity} />
    </lineSegments>
  );
};

interface TrailProps {
  points: THREE.Vector3[];
  color?: string;
  opacity?: number;
  width?: number;
}

export const Trail3D: React.FC<TrailProps> = ({
  points,
  color = '#00ff88',
  opacity = 0.8,
  width = 0.01
}) => {
  const meshRef = useRef<THREE.Mesh>(null);

  useFrame(() => {
    if (meshRef.current && points.length > 1) {
      const curve = new THREE.CatmullRomCurve3(points);
      const tubeGeometry = new THREE.TubeGeometry(curve, points.length, width, 8, false);
      meshRef.current.geometry = tubeGeometry;
    }
  });

  if (points.length < 2) return null;

  return (
    <mesh ref={meshRef}>
      <tubeGeometry />
      <meshBasicMaterial color={color} transparent opacity={opacity} />
    </mesh>
  );
};

// =============================================================================
// Animation Components
// =============================================================================

interface AnimatedGroupProps {
  children: React.ReactNode;
  animation?: 'rotate' | 'bounce' | 'float' | 'pulse';
  speed?: number;
  amplitude?: number;
}

export const AnimatedGroup: React.FC<AnimatedGroupProps> = ({
  children,
  animation = 'rotate',
  speed = 1,
  amplitude = 1
}) => {
  const groupRef = useRef<THREE.Group>(null);

  useFrame((state) => {
    if (!groupRef.current) return;

    const time = state.clock.getElapsedTime() * speed;

    switch (animation) {
      case 'rotate':
        groupRef.current.rotation.y = time;
        break;
      case 'bounce':
        groupRef.current.position.y = Math.sin(time * 2) * amplitude;
        break;
      case 'float':
        groupRef.current.position.y = Math.sin(time) * amplitude * 0.5;
        groupRef.current.rotation.y = Math.sin(time * 0.5) * 0.2;
        break;
      case 'pulse':
        const scale = 1 + Math.sin(time * 3) * amplitude * 0.1;
        groupRef.current.scale.setScalar(scale);
        break;
    }
  });

  return <group ref={groupRef}>{children}</group>;
};

interface MorphingGeometryProps {
  geometries: THREE.BufferGeometry[];
  morphSpeed?: number;
  position?: [number, number, number];
  material?: THREE.Material;
}

export const MorphingGeometry: React.FC<MorphingGeometryProps> = ({
  geometries,
  morphSpeed = 1,
  position = [0, 0, 0],
  material
}) => {
  const meshRef = useRef<THREE.Mesh>(null);
  const [currentIndex, setCurrentIndex] = useState(0);

  useFrame((state) => {
    if (!meshRef.current || geometries.length < 2) return;

    const time = state.clock.getElapsedTime() * morphSpeed;
    const newIndex = Math.floor(time) % geometries.length;

    if (newIndex !== currentIndex) {
      setCurrentIndex(newIndex);
      meshRef.current.geometry = geometries[newIndex];
    }
  });

  if (geometries.length === 0) return null;

  return (
    <mesh ref={meshRef} position={position} geometry={geometries[0]}>
      {material || <meshStandardMaterial />}
    </mesh>
  );
};

// =============================================================================
// Utility Components
// =============================================================================

interface BoundingBoxProps {
  object: THREE.Object3D;
  color?: string;
  opacity?: number;
}

export const BoundingBox: React.FC<BoundingBoxProps> = ({
  object,
  color = '#ff0000',
  opacity = 0.3
}) => {
  const [box, setBox] = useState<THREE.Box3 | null>(null);

  useEffect(() => {
    if (object) {
      const bbox = new THREE.Box3().setFromObject(object);
      setBox(bbox);
    }
  }, [object]);

  if (!box) return null;

  const size = box.getSize(new THREE.Vector3());
  const center = box.getCenter(new THREE.Vector3());

  return (
    <Box args={[size.x, size.y, size.z]} position={[center.x, center.y, center.z]}>
      <meshBasicMaterial color={color} transparent opacity={opacity} wireframe />
    </Box>
  );
};

interface PerformanceMonitorProps {
  position?: [number, number, number];
  showFPS?: boolean;
  showTriangles?: boolean;
  showMemory?: boolean;
}

export const PerformanceMonitor: React.FC<PerformanceMonitorProps> = ({
  position = [0, 0, 0],
  showFPS = true,
  showTriangles = true,
  showMemory = false
}) => {
  const { gl } = useThree();
  const [stats, setStats] = useState({ fps: 0, triangles: 0, memory: 0 });

  useFrame(() => {
    if (Math.random() < 0.1) { // Update 10% of frames to reduce overhead
      setStats({
        fps: Math.round(1000 / performance.now()),
        triangles: gl.info.render.triangles,
        memory: (performance as any).memory?.usedJSHeapSize / 1048576 || 0
      });
    }
  });

  const info: Record<string, any> = {};
  if (showFPS) info.FPS = stats.fps;
  if (showTriangles) info.Triangles = stats.triangles;
  if (showMemory) info['Memory (MB)'] = stats.memory.toFixed(1);

  return <InfoDisplay info={info} position={position} />;
};
