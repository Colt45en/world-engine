// MaskMesher.tsx - Standalone Volume Mesh Generator from Silhouettes
// Creates 3D meshes from front/side mask images using volumetric intersection

import React, { useState, useRef, useCallback } from "react";
import * as THREE from "three";
import { Canvas, useFrame } from "@react-three/fiber";
import { OrbitControls, Html } from "@react-three/drei";
import { Leva, useControls } from "leva";

// Utility Functions
async function fileToImageData(file: File, width: number, height: number): Promise<ImageData> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => {
      const canvas = document.createElement('canvas');
      canvas.width = width;
      canvas.height = height;
      const ctx = canvas.getContext('2d')!;

      // Cover-fit scaling (preserve aspect ratio)
      const aspectRatio = img.width / img.height;
      const canvasRatio = width / height;
      let drawWidth = width;
      let drawHeight = height;
      let offsetX = 0;
      let offsetY = 0;

      if (aspectRatio > canvasRatio) {
        drawHeight = width / aspectRatio;
        offsetY = (height - drawHeight) / 2;
      } else {
        drawWidth = height * aspectRatio;
        offsetX = (width - drawWidth) / 2;
      }

      ctx.drawImage(img, offsetX, offsetY, drawWidth, drawHeight);
      const imageData = ctx.getImageData(0, 0, width, height);
      URL.revokeObjectURL(img.src);
      resolve(imageData);
    };
    img.onerror = reject;
    img.src = URL.createObjectURL(file);
  });
}

function imageDataToMask(imageData: ImageData, threshold: number = 0.5) {
  const { data, width, height } = imageData;
  const mask = new Uint8Array(width * height);

  for (let i = 0, p = 0; i < data.length; i += 4, p++) {
    const r = data[i];
    const g = data[i + 1];
    const b = data[i + 2];
    const luminance = (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255;
    mask[p] = luminance >= threshold ? 1 : 0;
  }

  return { data: mask, width, height };
}

function buildVolumeFromMasks(
  frontMask: { data: Uint8Array; width: number; height: number },
  sideMask: { data: Uint8Array; width: number; height: number } | null,
  volumeWidth: number,
  volumeHeight: number,
  volumeDepth: number
) {
  const volume = new Uint8Array(volumeWidth * volumeHeight * volumeDepth);

  for (let y = 0; y < volumeHeight; y++) {
    for (let x = 0; x < volumeWidth; x++) {
      // Sample from front mask
      const frontU = Math.min(frontMask.width - 1, Math.max(0, Math.round(x * frontMask.width / volumeWidth)));
      const frontV = Math.min(frontMask.height - 1, Math.max(0, Math.round(y * frontMask.height / volumeHeight)));
      const frontValue = frontMask.data[frontV * frontMask.width + frontU];

      if (!frontValue) continue;

      for (let z = 0; z < volumeDepth; z++) {
        let sideValue = 1; // Default if no side mask

        if (sideMask) {
          const sideU = Math.min(sideMask.width - 1, Math.max(0, Math.round(z * sideMask.width / volumeDepth)));
          const sideV = Math.min(sideMask.height - 1, Math.max(0, Math.round(y * sideMask.height / volumeHeight)));
          sideValue = sideMask.data[sideV * sideMask.width + sideU];
        }

        if (sideValue) {
          volume[y * volumeWidth * volumeDepth + z * volumeWidth + x] = 1;
        }
      }
    }
  }

  return { data: volume, width: volumeWidth, height: volumeHeight, depth: volumeDepth };
}

function createVoxelMesh(
  volume: { data: Uint8Array; width: number; height: number; depth: number },
  minBounds: THREE.Vector3,
  maxBounds: THREE.Vector3
): THREE.BufferGeometry {
  const { data, width, height, depth } = volume;
  const positions: number[] = [];
  const normals: number[] = [];
  const indices: number[] = [];

  const scale = new THREE.Vector3().subVectors(maxBounds, minBounds);

  const getVoxel = (x: number, y: number, z: number): number => {
    if (x < 0 || y < 0 || z < 0 || x >= width || y >= height || z >= depth) return 0;
    return data[y * width * depth + z * width + x];
  };

  const addFace = (
    corners: number[][],
    normal: number[]
  ) => {
    const startIndex = positions.length / 3;

    // Add vertices
    corners.forEach(([x, y, z]) => {
      positions.push(
        minBounds.x + scale.x * (x / width),
        minBounds.y + scale.y * (y / height),
        minBounds.z + scale.z * (z / depth)
      );
      normals.push(...normal);
    });

    // Add indices for two triangles
    indices.push(
      startIndex, startIndex + 1, startIndex + 2,
      startIndex, startIndex + 2, startIndex + 3
    );
  };

  // Generate faces for each voxel
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      for (let z = 0; z < depth; z++) {
        if (!getVoxel(x, y, z)) continue;

        // Check each face direction
        const faces = [
          { dir: [-1, 0, 0], corners: [[x, y, z], [x, y + 1, z], [x, y + 1, z + 1], [x, y, z + 1]] },
          { dir: [1, 0, 0], corners: [[x + 1, y, z], [x + 1, y, z + 1], [x + 1, y + 1, z + 1], [x + 1, y + 1, z]] },
          { dir: [0, -1, 0], corners: [[x, y, z], [x, y, z + 1], [x + 1, y, z + 1], [x + 1, y, z]] },
          { dir: [0, 1, 0], corners: [[x, y + 1, z], [x + 1, y + 1, z], [x + 1, y + 1, z + 1], [x, y + 1, z + 1]] },
          { dir: [0, 0, -1], corners: [[x, y, z], [x + 1, y, z], [x + 1, y + 1, z], [x, y + 1, z]] },
          { dir: [0, 0, 1], corners: [[x, y, z + 1], [x, y + 1, z + 1], [x + 1, y + 1, z + 1], [x + 1, y, z + 1]] }
        ];

        faces.forEach(({ dir, corners }) => {
          const [dx, dy, dz] = dir;
          if (!getVoxel(x + dx, y + dy, z + dz)) {
            addFace(corners, dir);
          }
        });
      }
    }
  }

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
  geometry.setAttribute('normal', new THREE.Float32BufferAttribute(normals, 3));
  geometry.setIndex(indices);

  return geometry;
}

// Main Mask Mesher Component
function MaskMesherTool() {
  const [frontFile, setFrontFile] = useState<File | null>(null);
  const [sideFile, setSideFile] = useState<File | null>(null);
  const [geometry, setGeometry] = useState<THREE.BufferGeometry | null>(null);
  const [isBuilding, setIsBuilding] = useState(false);
  const meshRef = useRef<THREE.Mesh>(null!);

  const {
    resolution,
    threshold,
    meshColor,
    wireframe,
    metalness,
    roughness
  } = useControls("Mesh Settings", {
    resolution: { value: 64, min: 16, max: 128, step: 8 },
    threshold: { value: 0.5, min: 0.1, max: 0.9, step: 0.01 },
    meshColor: { value: "#4ecdc4" },
    wireframe: false,
    metalness: { value: 0.2, min: 0, max: 1, step: 0.01 },
    roughness: { value: 0.45, min: 0, max: 1, step: 0.01 }
  });

  const buildMesh = useCallback(async () => {
    if (isBuilding || !frontFile) return;

    setIsBuilding(true);
    try {
      const frontImageData = await fileToImageData(frontFile, resolution, resolution);
      const frontMask = imageDataToMask(frontImageData, threshold);

      let sideMask = null;
      if (sideFile) {
        const sideImageData = await fileToImageData(sideFile, resolution, resolution);
        sideMask = imageDataToMask(sideImageData, threshold);
      }

      const volume = buildVolumeFromMasks(
        frontMask,
        sideMask,
        resolution,
        resolution,
        resolution
      );

      const newGeometry = createVoxelMesh(
        volume,
        new THREE.Vector3(-1.25, -1.25, -1.25),
        new THREE.Vector3(1.25, 1.25, 1.25)
      );

      newGeometry.computeVertexNormals();
      setGeometry(newGeometry);
    } catch (error) {
      console.error('Error building mesh:', error);
      alert('Error building mesh. Please check your image files.');
    } finally {
      setIsBuilding(false);
    }
  }, [frontFile, sideFile, resolution, threshold, isBuilding]);

  const exportMesh = useCallback(() => {
    if (!meshRef.current) return;

    const exporter = new THREE.BufferGeometryUtils.mergeBufferGeometries([meshRef.current.geometry] as THREE.BufferGeometry[]);
    // Here you would typically use your export utilities
    console.log('Mesh ready for export:', exporter);
  }, []);

  useFrame(() => {
    if (meshRef.current) {
      meshRef.current.rotation.y += 0.005;
    }
  });

  return (
    <group>
      {/* Base platform */}
      <mesh position={[0, 0, 0]} rotation={[-Math.PI / 2, 0, 0]}>
        <cylinderGeometry args={[2, 2, 0.02, 64]} />
        <meshStandardMaterial color="#1a2636" roughness={0.95} metalness={0.05} />
      </mesh>

      {/* Generated mesh */}
      {geometry && (
        <mesh
          ref={meshRef}
          geometry={geometry}
          position={[0, 0, 0]}
        >
          <meshStandardMaterial
            color={meshColor}
            metalness={metalness}
            roughness={roughness}
            wireframe={wireframe}
          />
        </mesh>
      )}

      {/* UI Panel */}
      <Html position={[0, 2.5, 0]} center>
        <div
          style={{
            display: 'flex',
            flexDirection: 'column',
            gap: 10,
            padding: 15,
            background: 'rgba(10, 14, 22, 0.9)',
            border: '1px solid rgba(255, 255, 255, 0.1)',
            borderRadius: 12,
            color: '#eaf2ff',
            fontSize: 14,
            minWidth: 350
          }}
        >
          <div style={{ fontWeight: 700, marginBottom: 10, textAlign: 'center' }}>
            Mask Mesher Tool
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10 }}>
            <div>
              <label style={{ display: 'block', marginBottom: 5, fontSize: 12 }}>
                Front Silhouette (Required)
              </label>
              <input
                type="file"
                accept="image/*"
                onChange={(e) => setFrontFile(e.target.files?.[0] || null)}
                style={{
                  padding: 5,
                  background: 'rgba(255, 255, 255, 0.1)',
                  border: '1px solid rgba(255, 255, 255, 0.2)',
                  borderRadius: 6,
                  color: '#eaf2ff',
                  fontSize: 12
                }}
              />
            </div>

            <div>
              <label style={{ display: 'block', marginBottom: 5, fontSize: 12 }}>
                Side Silhouette (Optional)
              </label>
              <input
                type="file"
                accept="image/*"
                onChange={(e) => setSideFile(e.target.files?.[0] || null)}
                style={{
                  padding: 5,
                  background: 'rgba(255, 255, 255, 0.1)',
                  border: '1px solid rgba(255, 255, 255, 0.2)',
                  borderRadius: 6,
                  color: '#eaf2ff',
                  fontSize: 12
                }}
              />
            </div>
          </div>

          <div style={{ display: 'flex', gap: 10, marginTop: 10 }}>
            <button
              onClick={buildMesh}
              disabled={!frontFile || isBuilding}
              style={{
                flex: 1,
                padding: '10px 15px',
                background: frontFile && !isBuilding ? '#2563eb' : 'rgba(100, 100, 100, 0.3)',
                color: '#fff',
                border: 'none',
                borderRadius: 8,
                cursor: frontFile && !isBuilding ? 'pointer' : 'not-allowed',
                fontWeight: 600
              }}
            >
              {isBuilding ? 'Building...' : 'Build Mesh'}
            </button>

            <button
              onClick={exportMesh}
              disabled={!geometry}
              style={{
                flex: 1,
                padding: '10px 15px',
                background: geometry ? '#059669' : 'rgba(100, 100, 100, 0.3)',
                color: '#fff',
                border: 'none',
                borderRadius: 8,
                cursor: geometry ? 'pointer' : 'not-allowed',
                fontWeight: 600
              }}
            >
              Export Mesh
            </button>
          </div>

          <div style={{ fontSize: 11, opacity: 0.8, marginTop: 5 }}>
            Resolution: {resolution}³ • Threshold: {threshold.toFixed(2)}
            {geometry && <div>✓ Mesh Generated</div>}
          </div>
        </div>
      </Html>
    </group>
  );
}

// Main App
export default function MaskMesherApp() {
  return (
    <div style={{ width: '100%', height: '100vh', background: '#0f1419' }}>
      <Leva collapsed={false} />
      <Canvas camera={{ position: [3, 2, 5], fov: 60 }} shadows>
        <color attach="background" args={["#0f1419"]} />
        <ambientLight intensity={0.4} />
        <directionalLight
          position={[5, 5, 5]}
          intensity={0.8}
          castShadow
          shadow-mapSize-width={2048}
          shadow-mapSize-height={2048}
        />
        <MaskMesherTool />
        <OrbitControls enablePan={true} />
      </Canvas>
    </div>
  );
}
