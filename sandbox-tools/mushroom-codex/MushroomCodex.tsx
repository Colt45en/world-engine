import React, { useEffect, useMemo, useRef, useState } from "react";
import * as THREE from "three";
import { Canvas, useFrame } from "@react-three/fiber";
import { OrbitControls, Html } from "@react-three/drei";
import { useControls, button, folder } from "leva";
import { mathEngine } from '../shared/utils';

// =============================================================================
// Types and Interfaces
// =============================================================================

interface MushroomParams {
  capRadius?: number;
  capHeightScale?: number;
  stemHeight?: number;
  stemRadiusTop?: number;
  stemRadiusBottom?: number;
  capColor?: THREE.ColorRepresentation;
  stemColor?: THREE.ColorRepresentation;
  shades?: number;
  texture?: THREE.Texture | null;
  spotDensity?: number;
  spotScaleMin?: number;
  spotScaleMax?: number;
  outline?: boolean;
  outlineScale?: number;
  outlineColor?: THREE.ColorRepresentation;
  name?: string;
}

interface BuiltMushroom {
  group: THREE.Group;
  cap: THREE.Mesh;
  stem: THREE.Mesh;
  spots: THREE.Mesh[];
  outlineGroup?: THREE.Group;
}

interface MushroomProfile extends MushroomParams {
  seed?: number;
}

// =============================================================================
// Default Parameters and Profiles
// =============================================================================

function mushroomDefaults(p: MushroomParams): Required<MushroomParams> {
  return {
    capRadius: p.capRadius ?? 0.6,
    capHeightScale: p.capHeightScale ?? 0.9,
    stemHeight: p.stemHeight ?? 0.9,
    stemRadiusTop: p.stemRadiusTop ?? 0.18,
    stemRadiusBottom: p.stemRadiusBottom ?? 0.22,
    capColor: p.capColor ?? 0xd84b3c,
    stemColor: p.stemColor ?? 0xe9d5b5,
    shades: Math.max(2, p.shades ?? 5),
    texture: p.texture ?? null,
    spotDensity: p.spotDensity ?? 5.0,
    spotScaleMin: p.spotScaleMin ?? 0.04,
    spotScaleMax: p.spotScaleMax ?? 0.09,
    outline: p.outline ?? true,
    outlineScale: p.outlineScale ?? 1.02,
    outlineColor: p.outlineColor ?? 0x111111,
    name: p.name ?? "Mushroom",
  };
}

const MUSHROOM_PROFILES: MushroomProfile[] = [
  {
    name: "scarlet_tiny",
    seed: 101,
    capRadius: 0.55,
    capHeightScale: 0.9,
    stemHeight: 0.85,
    spotDensity: 4,
    capColor: "#d0302a",
    stemColor: "#ead9b9",
    outlineScale: 1.02
  },
  {
    name: "plump_fairy",
    seed: 202,
    capRadius: 0.8,
    capHeightScale: 0.8,
    stemHeight: 1.0,
    spotDensity: 7,
    capColor: "#e24a3b",
    stemColor: "#eddcc2",
    outlineScale: 1.03
  },
  {
    name: "tall_cap",
    seed: 303,
    capRadius: 0.6,
    capHeightScale: 1.1,
    stemHeight: 1.2,
    spotDensity: 5,
    capColor: "#c83f32",
    stemColor: "#e8d2b0",
    outlineScale: 1.02
  },
  {
    name: "ghost_white",
    seed: 404,
    capRadius: 0.7,
    capHeightScale: 0.95,
    stemHeight: 1.05,
    spotDensity: 0,
    capColor: "#f1f5f9",
    stemColor: "#dbe0e6",
    outlineScale: 1.015
  },
  {
    name: "speckled_amber",
    seed: 505,
    capRadius: 0.9,
    capHeightScale: 0.75,
    stemHeight: 0.9,
    spotDensity: 9,
    capColor: "#ff7f45",
    stemColor: "#f0dec2",
    outlineScale: 1.035
  },
  {
    name: "forest_king",
    seed: 606,
    capRadius: 1.1,
    capHeightScale: 1.2,
    stemHeight: 1.3,
    spotDensity: 3,
    capColor: "#8b3a2a",
    stemColor: "#e7d7b1",
    outlineScale: 1.028
  },
  {
    name: "violet_glow",
    seed: 707,
    capRadius: 0.65,
    capHeightScale: 0.85,
    stemHeight: 0.95,
    spotDensity: 6,
    capColor: "#7c3aed",
    stemColor: "#eadaff",
    outlineScale: 1.02
  },
  {
    name: "inky_blue",
    seed: 808,
    capRadius: 0.6,
    capHeightScale: 1.05,
    stemHeight: 1.15,
    spotDensity: 2,
    capColor: "#1f4fff",
    stemColor: "#d6e3ff",
    outlineScale: 1.018
  },
  {
    name: "sunset_peach",
    seed: 909,
    capRadius: 0.75,
    capHeightScale: 0.9,
    stemHeight: 1.0,
    spotDensity: 8,
    capColor: "#ff8c69",
    stemColor: "#ffe1c9",
    outlineScale: 1.03
  },
  {
    name: "midnight_teal",
    seed: 111,
    capRadius: 0.7,
    capHeightScale: 1.1,
    stemHeight: 1.25,
    spotDensity: 1,
    capColor: "#0f766e",
    stemColor: "#d3efe9",
    outlineScale: 1.022
  }
];

// =============================================================================
// Material Creation Functions
// =============================================================================

function createToonRamp(steps: number): THREE.DataTexture {
  const height = Math.max(2, steps | 0);
  const buffer = new Uint8Array(height * 4);

  for (let i = 0; i < height; i++) {
    const value = Math.round((i / (height - 1)) * 255);
    buffer[i * 4] = value;     // R
    buffer[i * 4 + 1] = value; // G
    buffer[i * 4 + 2] = value; // B
    buffer[i * 4 + 3] = 255;   // A
  }

  const texture = new THREE.DataTexture(buffer, 1, height);
  texture.needsUpdate = true;
  texture.minFilter = THREE.NearestFilter;
  texture.magFilter = THREE.NearestFilter;
  texture.generateMipmaps = false;
  texture.wrapS = THREE.ClampToEdgeWrapping;
  texture.wrapT = THREE.ClampToEdgeWrapping;

  return texture;
}

function createToonMaterial(
  color: THREE.ColorRepresentation,
  shades: number,
  map?: THREE.Texture
): THREE.MeshToonMaterial {
  const material = new THREE.MeshToonMaterial({ color });
  material.gradientMap = createToonRamp(shades);

  if (map) {
    material.map = map;
    (material.map as any).colorSpace = THREE.SRGBColorSpace;
    material.map.needsUpdate = true;
  }

  return material;
}

// =============================================================================
// Outline System
// =============================================================================

function createOutlineGroup(
  root: THREE.Object3D,
  scale = 1.02,
  color: THREE.ColorRepresentation = 0x111111
): THREE.Group {
  const group = new THREE.Group();
  const meshes: THREE.Mesh[] = [];

  root.traverse((obj) => {
    if ((obj as any).isMesh && (obj as THREE.Mesh).geometry) {
      meshes.push(obj as THREE.Mesh);
    }
  });

  for (const mesh of meshes) {
    const material = new THREE.MeshBasicMaterial({
      color,
      side: THREE.BackSide,
      depthWrite: false
    });

    const clone = new THREE.Mesh(mesh.geometry, material);
    clone.name = mesh.name + "_Outline";
    clone.position.copy(mesh.position);
    clone.quaternion.copy(mesh.quaternion);
    clone.scale.copy(mesh.scale).multiplyScalar(scale);

    group.add(clone);
  }

  return group;
}

// =============================================================================
// Spot Generation System
// =============================================================================

function randomNormal(): number {
  let u = 0, v = 0;
  while (u === 0) u = Math.random();
  while (v === 0) v = Math.random();
  return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
}

function generateSpots(cap: THREE.Mesh, params: Required<MushroomParams>): THREE.Mesh[] {
  const spots: THREE.Mesh[] = [];
  const radius = params.capRadius;
  const hemisphereArea = 2 * Math.PI * radius * radius;
  const spotCount = Math.max(0, Math.round(params.spotDensity * hemisphereArea));

  const baseGeometry = new THREE.CircleGeometry(1, 16);
  const zUpVector = new THREE.Vector3(0, 0, 1);
  const quaternion = new THREE.Quaternion();
  const normal = new THREE.Vector3();
  const position = new THREE.Vector3();

  for (let i = 0; i < spotCount; i++) {
    // Generate random direction on hemisphere (y >= 0)
    let direction: THREE.Vector3;
    do {
      direction = new THREE.Vector3(
        randomNormal(),
        Math.abs(randomNormal()),
        randomNormal()
      ).normalize();
    } while (!isFinite(direction.x));

    // Calculate spot position on cap surface
    normal.copy(direction).normalize();
    position.copy(normal).multiplyScalar(radius);
    position.y *= params.capHeightScale;

    // Orient spot to surface normal
    quaternion.setFromUnitVectors(zUpVector, normal.clone().normalize());

    // Random spot size
    const spotSize = THREE.MathUtils.lerp(
      params.spotScaleMin,
      params.spotScaleMax,
      Math.random()
    ) * radius;

    // Create spot mesh
    const spotMaterial = new THREE.MeshStandardMaterial({
      color: 0xffffff,
      roughness: 0.9,
      metalness: 0
    });

    const spot = new THREE.Mesh(baseGeometry, spotMaterial);
    spot.scale.setScalar(spotSize);
    spot.position.copy(position).addScaledVector(normal, -0.002 * radius);
    spot.quaternion.copy(quaternion);
    spot.name = `Spot_${i}`;

    cap.add(spot);
    spots.push(spot);
  }

  return spots;
}

// =============================================================================
// Main Mushroom Builder
// =============================================================================

function buildMushroom(params: MushroomParams = {}): BuiltMushroom {
  const p = mushroomDefaults(params);
  const group = new THREE.Group();
  group.name = p.name;

  // Create cap geometry and material
  const capGeometry = new THREE.SphereGeometry(
    p.capRadius,
    48,
    24,
    0,
    Math.PI * 2,
    0,
    Math.PI / 2
  );
  capGeometry.computeVertexNormals();

  const capMaterial = createToonMaterial(p.capColor, p.shades, p.texture);
  const cap = new THREE.Mesh(capGeometry, capMaterial);
  cap.scale.y *= p.capHeightScale;
  cap.name = "Cap";
  group.add(cap);

  // Create stem geometry and material
  const stemGeometry = new THREE.CylinderGeometry(
    p.stemRadiusTop,
    p.stemRadiusBottom,
    p.stemHeight,
    24,
    1,
    false
  );
  stemGeometry.computeVertexNormals();

  const stemMaterial = createToonMaterial(p.stemColor, p.shades);
  const stem = new THREE.Mesh(stemGeometry, stemMaterial);
  stem.position.y = -p.stemHeight * 0.5;
  stem.name = "Stem";
  group.add(stem);

  // Generate spots on cap
  const spots = generateSpots(cap, p);

  // Create outline group if enabled
  let outlineGroup: THREE.Group | undefined;
  if (p.outline && p.outlineScale > 1.0) {
    outlineGroup = createOutlineGroup(group, p.outlineScale, p.outlineColor);
    group.add(outlineGroup);
  }

  return {
    group,
    cap,
    stem,
    spots,
    outlineGroup
  };
}

// =============================================================================
// Mushroom Primitive Component
// =============================================================================

interface MushroomPrimitiveProps {
  params: MushroomParams;
  selected?: boolean;
  onPointerDown?: (event: any) => void;
  animated?: boolean;
}

function MushroomPrimitive({ params, selected, onPointerDown, animated }: MushroomPrimitiveProps) {
  const built = useMemo(() => buildMushroom(params), [JSON.stringify(params)]);
  const ref = useRef<THREE.Group>(null!);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      built.group.traverse((obj: any) => {
        if (obj.geometry) obj.geometry.dispose?.();
        if (obj.material) {
          if (Array.isArray(obj.material)) {
            obj.material.forEach((mat: any) => mat.dispose?.());
          } else {
            obj.material.dispose?.();
          }
        }
      });
    };
  }, [built]);

  // Animation
  useFrame(({ clock }) => {
    if (ref.current && animated) {
      ref.current.rotation.y = clock.getElapsedTime() * 0.5;
      ref.current.position.y = Math.sin(clock.getElapsedTime() * 2) * 0.02;
    }
  });

  return (
    <group onPointerDown={onPointerDown}>
      <primitive ref={ref} object={built.group} />
      {selected && (
        <Html center distanceFactor={8} sprite>
          <div style={{
            padding: '4px 8px',
            borderRadius: 999,
            background: 'rgba(255,255,255,0.15)',
            backdropFilter: 'blur(6px)',
            fontSize: 11,
            color: '#fff',
            fontWeight: 600,
            border: '1px solid rgba(255,255,255,0.2)'
          }}>
            Selected
          </div>
        </Html>
      )}
    </group>
  );
}

// =============================================================================
// Mushroom Garden Component
// =============================================================================

interface MushroomItem {
  id: string;
  params: MushroomParams;
}

function MushroomGarden() {
  const [items, setItems] = useState<MushroomItem[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [stats, setStats] = useState({ count: 0, selected: null as string | null });

  // Update stats
  useEffect(() => {
    const selected = items.find(item => item.id === selectedId);
    setStats({
      count: items.length,
      selected: selected?.params.name || null
    });
  }, [items, selectedId]);

  const spawnMushroom = (profile: MushroomProfile) => {
    const newItem: MushroomItem = {
      id: `${Date.now()}_${Math.random().toString(36).slice(2, 8)}`,
      params: { ...profile }
    };
    setItems(prev => [...prev, newItem]);
  };

  const clearGarden = () => {
    setItems([]);
    setSelectedId(null);
  };

  const exportSelected = () => {
    const selectedItem = items.find(item => item.id === selectedId);
    if (!selectedItem) return;

    const data = {
      name: selectedItem.params.name,
      parameters: selectedItem.params,
      timestamp: Date.now()
    };

    console.log('Mushroom exported:', data);

    // Create downloadable JSON file
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `mushroom-${selectedItem.params.name || 'unnamed'}-${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const generateRandom = () => {
    const randomParams: MushroomParams = {
      name: `random_${Math.random().toString(36).slice(2, 7)}`,
      capRadius: 0.4 + Math.random() * 0.8,
      capHeightScale: 0.6 + Math.random() * 0.8,
      stemHeight: 0.6 + Math.random() * 1.0,
      stemRadiusTop: 0.1 + Math.random() * 0.2,
      stemRadiusBottom: 0.15 + Math.random() * 0.25,
      spotDensity: Math.random() * 10,
      capColor: `hsl(${Math.random() * 360}, ${50 + Math.random() * 50}%, ${30 + Math.random() * 40}%)`,
      stemColor: `hsl(${Math.random() * 60 + 30}, ${20 + Math.random() * 30}%, ${60 + Math.random() * 30}%)`,
      outlineScale: 1.01 + Math.random() * 0.04
    };

    spawnMushroom(randomParams);
  };

  return (
    <>
      {/* Base platform */}
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, -0.02, 0]} receiveShadow>
        <cylinderGeometry args={[3.0, 3.0, 0.08, 72]} />
        <meshStandardMaterial
          color="#1a2636"
          roughness={0.95}
          metalness={0.05}
        />
      </mesh>

      {/* Mushroom grid */}
      <group position={[0, 0.6, 0]}>
        {items.map((item, index) => (
          <group
            key={item.id}
            position={[
              ((index % 6) - 2.5) * 1.2,
              0,
              -Math.floor(index / 6) * 1.2
            ]}
          >
            <MushroomPrimitive
              params={item.params}
              selected={selectedId === item.id}
              onPointerDown={(e) => {
                e.stopPropagation();
                setSelectedId(item.id);
              }}
              animated={true}
            />
          </group>
        ))}
      </group>

      {/* Control Panel */}
      <Html position={[0, 2.5, 0]} center>
        <div style={{
          display: 'flex',
          gap: 15,
          padding: 15,
          background: 'rgba(10,14,22,0.9)',
          border: '1px solid rgba(255,255,255,0.12)',
          borderRadius: 12,
          color: '#eaf2ff',
          fontSize: 13,
          backdropFilter: 'blur(10px)'
        }}>
          {/* Profiles Panel */}
          <div style={{ minWidth: 280 }}>
            <div style={{ fontWeight: 700, marginBottom: 8, color: '#64ffda' }}>
              🍄 Mushroom Codex Pro
            </div>
            <div style={{
              display: 'grid',
              gap: 4,
              maxHeight: 250,
              overflowY: 'auto',
              paddingRight: 8
            }}>
              {MUSHROOM_PROFILES.map((profile, index) => (
                <button
                  key={profile.name + index}
                  onClick={() => spawnMushroom(profile)}
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'space-between',
                    gap: 10,
                    padding: '8px 10px',
                    borderRadius: 8,
                    border: '1px solid rgba(255,255,255,0.15)',
                    background: 'rgba(255,255,255,0.08)',
                    color: '#fff',
                    fontSize: 12,
                    cursor: 'pointer',
                    transition: 'all 0.2s ease'
                  }}
                  onMouseEnter={(e) => {
                    e.currentTarget.style.background = 'rgba(255,255,255,0.15)';
                  }}
                  onMouseLeave={(e) => {
                    e.currentTarget.style.background = 'rgba(255,255,255,0.08)';
                  }}
                >
                  <span style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                    <span style={{
                      width: 16,
                      height: 16,
                      borderRadius: '50%',
                      border: '1px solid rgba(255,255,255,0.3)',
                      background: String(profile.capColor ?? '#d84b3c')
                    }} />
                    <span style={{ maxWidth: 120, overflow: 'hidden', textOverflow: 'ellipsis' }}>
                      {profile.name ?? 'profile'}
                    </span>
                  </span>
                  <span style={{ opacity: 0.7, fontSize: 10 }}>
                    R:{profile.capRadius?.toFixed(1) ?? '0.6'} H:{profile.stemHeight?.toFixed(1) ?? '0.9'}
                  </span>
                </button>
              ))}
            </div>
          </div>

          {/* Actions Panel */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: 8, minWidth: 150 }}>
            <div style={{ fontWeight: 600, marginBottom: 4 }}>Actions</div>

            <button
              onClick={generateRandom}
              style={{
                padding: '8px 12px',
                borderRadius: 8,
                background: 'rgba(123, 58, 237, 0.2)',
                color: '#c084fc',
                border: '1px solid rgba(123, 58, 237, 0.4)',
                cursor: 'pointer'
              }}
            >
              🎲 Generate Random
            </button>

            <button
              onClick={exportSelected}
              disabled={!selectedId}
              style={{
                padding: '8px 12px',
                borderRadius: 8,
                background: selectedId ? 'rgba(16, 185, 129, 0.2)' : 'rgba(100, 100, 100, 0.1)',
                color: selectedId ? '#6ee7b7' : '#9ca3af',
                border: selectedId ? '1px solid rgba(16, 185, 129, 0.4)' : '1px solid rgba(100, 100, 100, 0.2)',
                cursor: selectedId ? 'pointer' : 'not-allowed'
              }}
            >
              💾 Export Selected
            </button>

            <button
              onClick={clearGarden}
              style={{
                padding: '8px 12px',
                borderRadius: 8,
                background: 'rgba(239, 68, 68, 0.2)',
                color: '#fca5a5',
                border: '1px solid rgba(239, 68, 68, 0.4)',
                cursor: 'pointer'
              }}
            >
              🗑️ Clear Garden
            </button>

            <div style={{
              marginTop: 10,
              padding: '8px',
              background: 'rgba(255,255,255,0.05)',
              borderRadius: 6,
              fontSize: 11
            }}>
              <div>Total: {stats.count}</div>
              <div>Selected: {stats.selected || 'None'}</div>
            </div>
          </div>
        </div>
      </Html>
    </>
  );
}

// =============================================================================
// Main Mushroom Codex Component
// =============================================================================

export default function MushroomCodex() {
  return (
    <div style={{ width: '100%', height: '100vh', background: '#0a0a0a' }}>
      <Canvas
        camera={{ position: [4, 3, 4], fov: 60 }}
        shadows
      >
        <ambientLight intensity={0.3} />
        <directionalLight
          position={[5, 8, 5]}
          intensity={0.8}
          castShadow
          shadow-mapSize-width={2048}
          shadow-mapSize-height={2048}
        />
        <spotLight
          position={[-3, 4, 3]}
          intensity={0.4}
          angle={0.3}
          penumbra={0.2}
          color="#ff6b6b"
          castShadow
        />
        <pointLight
          position={[2, 1, -2]}
          intensity={0.3}
          color="#64ffda"
        />

        <MushroomGarden />

        <OrbitControls
          enablePan={true}
          enableZoom={true}
          enableRotate={true}
          maxPolarAngle={Math.PI * 0.75}
          minDistance={2}
          maxDistance={12}
        />
      </Canvas>
    </div>
  );
}
