import React, { useEffect, useMemo, useRef, useState } from "react";
import * as THREE from "three";
import { Canvas, useFrame, useThree, extend } from "@react-three/fiber";
import { OrbitControls, Line, useFBO, shaderMaterial } from "@react-three/drei";
import { useControls, button, folder } from "leva";
import { mathEngine } from '../shared/utils';

// =============================================================================
// Avatar Impostor Layer Constant
// =============================================================================
const AVATAR_LAYER = 1;

// =============================================================================
// Hexagonal Platform Geometry
// =============================================================================
function Platform({ R = 3.6, r = 0.6, ring = 3 }) {
  const geometry = useMemo(() => {
    const dx = r * Math.sqrt(3);
    const dy = r * 1.5;
    const centers: THREE.Vector2[] = [];

    // Generate hexagonal grid centers
    for (let q = -ring; q <= ring; q++) {
      const minI = Math.max(-ring, -q - ring);
      const maxI = Math.min(ring, -q + ring);
      for (let i = minI; i <= maxI; i++) {
        centers.push(new THREE.Vector2(dx * (q + i / 2), dy * i));
      }
    }

    // Height function with hexagonal ridges and bowl shape
    const heightFunction = (x: number, y: number) => {
      let minDistance = Infinity;
      for (const center of centers) {
        const distance = Math.abs(Math.hypot(x - center.x, y - center.y) - r);
        minDistance = Math.min(minDistance, distance);
      }

      const ridge = Math.max(0, 1 - minDistance / (r * 0.2));
      const bowl = -0.15 * Math.exp(-(x * x + y * y) / (R * R));
      return 0.08 * (0.55 * ridge + bowl);
    };

    // Generate mesh
    const radialSegments = Math.max(24, Math.floor(120 * 0.6));
    const angularSegments = Math.max(48, 120);
    const positions: number[] = [];
    const indices: number[] = [];

    for (let i = 0; i <= radialSegments; i++) {
      const radius = R * i / radialSegments;
      for (let j = 0; j <= angularSegments; j++) {
        const theta = 2 * Math.PI * j / angularSegments;
        const x = radius * Math.cos(theta);
        const y = radius * Math.sin(theta);
        const z = heightFunction(x, y);
        positions.push(x, y, z);
      }
    }

    const columns = angularSegments + 1;
    for (let i = 0; i < radialSegments; i++) {
      for (let j = 0; j < angularSegments; j++) {
        const a = i * columns + j;
        const b = a + 1;
        const c = (i + 1) * columns + j;
        const d = c + 1;
        indices.push(a, c, b, b, c, d);
      }
    }

    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
    geometry.setIndex(indices);
    geometry.computeVertexNormals();
    geometry.toNonIndexed();
    geometry.computeVertexNormals();

    return geometry;
  }, [R, r, ring]);

  return (
    <mesh rotation={[-Math.PI / 2, 0, 0]} geometry={geometry}>
      <meshPhysicalMaterial
        color="#aee9ff"
        transmission={1}
        thickness={0.6}
        roughness={0.18}
        ior={1.52}
        reflectivity={0.35}
        clearcoat={0.6}
        clearcoatRoughness={0.16}
        flatShading
      />
    </mesh>
  );
}

// =============================================================================
// Hexagonal Grid Lines
// =============================================================================
function HexGridLines({ r = 0.6, ring = 3, steps = 192 }) {
  const dx = r * Math.sqrt(3);
  const dy = r * 1.5;

  const centers = useMemo(() => {
    const centerArray: THREE.Vector3[] = [];
    for (let q = -ring; q <= ring; q++) {
      const minI = Math.max(-ring, -q - ring);
      const maxI = Math.min(ring, -q + ring);
      for (let i = minI; i <= maxI; i++) {
        centerArray.push(new THREE.Vector3(dx * (q + i / 2), dy * i, 0));
      }
    }
    return centerArray;
  }, [r, ring]);

  const createCirclePoints = (center: THREE.Vector3) =>
    Array.from({ length: steps + 1 }, (_, i) => {
      const theta = 2 * Math.PI * i / steps;
      return new THREE.Vector3(
        center.x + r * Math.cos(theta),
        center.y + r * Math.sin(theta),
        0.004
      );
    });

  return (
    <group rotation={[-Math.PI / 2, 0, 0]}>
      {centers.map((center, index) => (
        <Line
          key={index}
          points={createCirclePoints(center)}
          color="#9ad1ff"
          lineWidth={1.2}
          depthTest={false}
        />
      ))}
    </group>
  );
}

// =============================================================================
// Multi-View Camera Ring System
// =============================================================================
interface MultiViewData {
  color: THREE.Texture[];
  depth: THREE.Texture[];
  pos: Float32Array;
  vp: Float32Array;
  texel: Float32Array;
  near: Float32Array;
  far: Float32Array;
}

interface CameraRingProps {
  onMultiView: (data: MultiViewData) => void;
  cameraCount?: number;
  textureSize?: number;
  rotationSpeed?: number;
  radius?: number;
  height?: number;
  captureInterval?: number;
  lookAtTarget?: THREE.Vector3;
}

function CameraRingLive({
  onMultiView,
  cameraCount = 9,
  textureSize = 512,
  rotationSpeed = 0.35,
  radius = 2.0,
  height = 2.4,
  captureInterval = 1,
  lookAtTarget = new THREE.Vector3(0, 1.7, 0)
}: CameraRingProps) {
  const { gl, scene } = useThree();

  const cameras = useMemo(() =>
    Array.from({ length: cameraCount }, () => {
      const camera = new THREE.PerspectiveCamera(60, 1, 0.5, 6.0);
      camera.layers.enableAll();
      camera.layers.disable(AVATAR_LAYER);
      return camera;
    }), [cameraCount]
  );

  const fbos = useMemo(() =>
    Array.from({ length: cameraCount }, () => useFBO({
      width: textureSize,
      height: textureSize,
      depthBuffer: true,
      depthTexture: true,
      samples: 0,
      stencilBuffer: false
    })), [cameraCount, textureSize]
  );

  const positions = useMemo(() => new Float32Array(cameraCount * 3), [cameraCount]);
  const viewProjectionMatrices = useMemo(() => new Float32Array(cameraCount * 16), [cameraCount]);
  const texelSize = useMemo(() => Float32Array.from(
    Array.from({ length: cameraCount * 2 }, () => 1 / textureSize)
  ), [cameraCount, textureSize]);
  const nearPlanes = useMemo(() => Float32Array.from({ length: cameraCount }, () => 0.5), [cameraCount]);
  const farPlanes = useMemo(() => Float32Array.from({ length: cameraCount }, () => 6.0), [cameraCount]);

  const frameRef = useRef(0);

  useFrame(({ clock }) => {
    frameRef.current++;
    const time = clock.getElapsedTime();

    for (let i = 0; i < cameraCount; i++) {
      const angle = rotationSpeed * time + 2 * Math.PI * i / cameraCount;
      const x = Math.cos(angle) * radius;
      const z = Math.sin(angle) * radius;
      const y = height;

      const camera = cameras[i];
      camera.position.set(x, y, z);
      camera.lookAt(lookAtTarget);

      if (frameRef.current % Math.max(1, captureInterval) === 0) {
        gl.setRenderTarget(fbos[i]);
        gl.render(scene, camera);
        gl.setRenderTarget(null);

        positions.set([x, y, z], i * 3);

        const viewMatrix = camera.matrixWorldInverse;
        const projectionMatrix = camera.projectionMatrix;
        const vpMatrix = new THREE.Matrix4().multiplyMatrices(projectionMatrix, viewMatrix);
        vpMatrix.toArray(viewProjectionMatrices, i * 16);
      }
    }

    if (frameRef.current % Math.max(1, captureInterval) === 0) {
      onMultiView({
        color: fbos.map(fbo => fbo.texture),
        depth: fbos.map(fbo => fbo.depthTexture!),
        pos: positions,
        vp: viewProjectionMatrices,
        texel: texelSize,
        near: nearPlanes,
        far: farPlanes
      });
    }
  });

  return null;
}

// =============================================================================
// Avatar Impostor Shader Material
// =============================================================================
const AvatarMaterial = shaderMaterial(
  // Uniforms
  {
    // Multi-view textures
    mvTex0: null, mvTex1: null, mvTex2: null, mvTex3: null, mvTex4: null,
    mvTex5: null, mvTex6: null, mvTex7: null, mvTex8: null,

    // Depth textures
    mvDepth0: null, mvDepth1: null, mvDepth2: null, mvDepth3: null, mvDepth4: null,
    mvDepth5: null, mvDepth6: null, mvDepth7: null, mvDepth8: null,

    // Camera data
    camPos: new Float32Array(9 * 3),
    camVP: new Float32Array(9 * 16),
    nearP: new Float32Array(9),
    farP: new Float32Array(9),
    texel: new Float32Array(9 * 2),

    // Rendering parameters
    viewCount: 0,
    bias: 0.02,
    gradLo: 0.02,
    gradHi: 0.12,
    occEps: 0.02,
    gradEnable: 1,
    diagMode: 0,
    diagMix: 0.75,
    lightDir: new THREE.Vector3(0.4, 0.6, 0.2).normalize()
  },
  // Vertex Shader
  /* glsl */`
    varying vec3 vWorldPos;
    varying vec3 vNormal;

    void main() {
      vec4 worldPos = modelMatrix * vec4(position * 1.9, 1.0);
      vWorldPos = worldPos.xyz;
      vNormal = normalize(mat3(modelMatrix) * normal);
      gl_Position = projectionMatrix * viewMatrix * worldPos;
    }
  `,
  // Fragment Shader
  /* glsl */`
    precision highp float;

    varying vec3 vWorldPos;
    varying vec3 vNormal;

    // Multi-view textures
    uniform sampler2D mvTex0, mvTex1, mvTex2, mvTex3, mvTex4, mvTex5, mvTex6, mvTex7, mvTex8;
    uniform sampler2D mvDepth0, mvDepth1, mvDepth2, mvDepth3, mvDepth4, mvDepth5, mvDepth6, mvDepth7, mvDepth8;

    // Camera data
    uniform vec3 camPos[9];
    uniform mat4 camVP[9];
    uniform int viewCount;
    uniform float nearP[9], farP[9];
    uniform vec2 texel[9];

    // Parameters
    uniform float bias, gradLo, gradHi, occEps;
    uniform int gradEnable, diagMode;
    uniform float diagMix;
    uniform vec3 lightDir;

    // Linearize depth
    float linearizeDepth(float depth, float near, float far) {
      float z = depth * 2.0 - 1.0;
      return (2.0 * near * far) / (far + near - z * (far - near));
    }

    // Get color from view
    vec3 getColor(int index, vec2 uv) {
      if (index == 0) return texture2D(mvTex0, uv).rgb;
      if (index == 1) return texture2D(mvTex1, uv).rgb;
      if (index == 2) return texture2D(mvTex2, uv).rgb;
      if (index == 3) return texture2D(mvTex3, uv).rgb;
      if (index == 4) return texture2D(mvTex4, uv).rgb;
      if (index == 5) return texture2D(mvTex5, uv).rgb;
      if (index == 6) return texture2D(mvTex6, uv).rgb;
      if (index == 7) return texture2D(mvTex7, uv).rgb;
      return texture2D(mvTex8, uv).rgb;
    }

    // Get depth from view
    float getDepth(int index, vec2 uv) {
      if (index == 0) return texture2D(mvDepth0, uv).x;
      if (index == 1) return texture2D(mvDepth1, uv).x;
      if (index == 2) return texture2D(mvDepth2, uv).x;
      if (index == 3) return texture2D(mvDepth3, uv).x;
      if (index == 4) return texture2D(mvDepth4, uv).x;
      if (index == 5) return texture2D(mvDepth5, uv).x;
      if (index == 6) return texture2D(mvDepth6, uv).x;
      if (index == 7) return texture2D(mvDepth7, uv).x;
      return texture2D(mvDepth8, uv).x;
    }

    // Get minimum depth in neighborhood
    float getMinDepth(int index, vec2 uv) {
      vec2 t = texel[index];
      float minDepth = 1.0;
      for (int x = -1; x <= 1; x++) {
        for (int y = -1; y <= 1; y++) {
          vec2 offset = vec2(float(x) * t.x, float(y) * t.y);
          minDepth = min(minDepth, getDepth(index, uv + offset));
        }
      }
      return minDepth;
    }

    // Gradient-based confidence
    float getGradientConfidence(int index, vec2 uv, float lo, float hi) {
      vec2 t = texel[index];
      float scale = length(t);
      float gradLow = lo * scale;
      float gradHigh = hi * scale;

      vec3 color = getColor(index, uv);
      float luminance = dot(color, vec3(0.299, 0.587, 0.114));

      float lumRight = dot(getColor(index, uv + vec2(t.x, 0.0)), vec3(0.299, 0.587, 0.114));
      float lumLeft = dot(getColor(index, uv + vec2(-t.x, 0.0)), vec3(0.299, 0.587, 0.114));
      float lumUp = dot(getColor(index, uv + vec2(0.0, t.y)), vec3(0.299, 0.587, 0.114));
      float lumDown = dot(getColor(index, uv + vec2(0.0, -t.y)), vec3(0.299, 0.587, 0.114));

      float gradX = lumRight - lumLeft;
      float gradY = lumUp - lumDown;
      float gradMagnitude = sqrt(gradX * gradX + gradY * gradY);

      return 1.0 - smoothstep(gradLow, gradHigh, gradMagnitude);
    }

    // Debug color palette
    vec3 getDebugColor(int index) {
      vec3 palette[9];
      palette[0] = vec3(1.0, 0.4, 0.4);
      palette[1] = vec3(1.0, 0.75, 0.4);
      palette[2] = vec3(0.98, 1.0, 0.4);
      palette[3] = vec3(0.6, 1.0, 0.4);
      palette[4] = vec3(0.4, 1.0, 0.8);
      palette[5] = vec3(0.4, 0.75, 1.0);
      palette[6] = vec3(0.6, 0.4, 1.0);
      palette[7] = vec3(0.9, 0.4, 1.0);
      palette[8] = vec3(1.0, 0.4, 0.7);
      return palette[int(mod(float(index), 9.0))];
    }

    void main() {
      vec3 accumulator = vec3(0.0);
      float weightSum = 1e-5;
      float maxWeight = -1.0;
      int maxIndex = -1;
      float maxCosTheta = 0.0;
      float occlusionCount = 0.0;
      float confidenceSum = 0.0;
      float validViews = 0.0;

      // Multi-view blending
      for (int i = 0; i < 9; i++) {
        if (i >= viewCount) break;

        // Project world position to view space
        vec4 clipSpacePos = camVP[i] * vec4(vWorldPos, 1.0);
        if (clipSpacePos.w <= 0.0) continue;

        vec3 ndcPos = clipSpacePos.xyz / clipSpacePos.w;
        vec2 screenUV = ndcPos.xy * 0.5 + 0.5;

        // Check if UV is in valid range
        if (screenUV.x < 0.0 || screenUV.x > 1.0 || screenUV.y < 0.0 || screenUV.y > 1.0) continue;

        // Depth test for occlusion
        float expectedLinearDepth = linearizeDepth(ndcPos.z * 0.5 + 0.5, nearP[i], farP[i]);
        float sampledLinearDepth = linearizeDepth(getMinDepth(i, screenUV), nearP[i], farP[i]);

        if (sampledLinearDepth + occEps < expectedLinearDepth) {
          occlusionCount += 1.0;
          continue;
        }

        // Calculate view direction and surface normal angle
        vec3 viewDirection = normalize(camPos[i] - vWorldPos);
        float cosTheta = max(0.0, dot(normalize(vNormal), viewDirection) - bias);

        // Gradient-based confidence
        float confidence = gradEnable == 1 ? getGradientConfidence(i, screenUV, gradLo, gradHi) : 1.0;

        // Combine factors for final weight
        float weight = cosTheta * confidence;

        accumulator += getColor(i, screenUV) * weight;
        weightSum += weight;

        if (weight > maxWeight) {
          maxWeight = weight;
          maxIndex = i;
          maxCosTheta = cosTheta;
        }

        confidenceSum += confidence;
        validViews += 1.0;
      }

      vec3 baseColor = accumulator / weightSum;
      float quality = maxWeight / max(weightSum, 1e-5);
      float occlusionRatio = viewCount > 0 ? occlusionCount / float(viewCount) : 0.0;
      float averageConfidence = validViews > 0.0 ? confidenceSum / validViews : 0.0;

      // Debug visualization overlays
      vec3 debugOverlay = vec3(0.0);
      if (diagMode == 1) {
        debugOverlay = maxIndex >= 0 ? getDebugColor(maxIndex) : vec3(0.0);
      } else if (diagMode == 2) {
        debugOverlay = mix(vec3(0.0), vec3(1.0, 0.25, 0.2), 1.0 - clamp(quality, 0.0, 1.0));
      } else if (diagMode == 3) {
        debugOverlay = vec3(0.2, 0.45, 1.0) * clamp(occlusionRatio, 0.0, 1.0);
      } else if (diagMode == 4) {
        debugOverlay = vec3(0.2, 1.0, 0.5) * clamp(averageConfidence, 0.0, 1.0);
      } else if (diagMode == 5) {
        debugOverlay = mix(vec3(0.0), vec3(0.4, 1.0, 0.6), clamp(maxCosTheta * 1.3, 0.0, 1.0));
      }

      gl_FragColor = vec4(mix(baseColor, debugOverlay, clamp(diagMix, 0.0, 1.0)), 1.0);
    }
  `
);

extend({ AvatarMaterial });
declare global {
  namespace JSX {
    interface IntrinsicElements {
      avatarMaterial: any;
    }
  }
}

// =============================================================================
// Avatar Mesh with Impostor Material
// =============================================================================
interface AvatarProps {
  multiViewData: MultiViewData | null;
  onMaterialRef: (material: THREE.ShaderMaterial) => void;
}

function Avatar({ multiViewData, onMaterialRef }: AvatarProps) {
  const meshRef = useRef<THREE.Mesh>(null!);
  const materialRef = useRef<THREE.ShaderMaterial>(null!);

  useEffect(() => {
    if (meshRef.current) {
      meshRef.current.layers.set(AVATAR_LAYER);
    }
  }, []);

  useEffect(() => {
    if (materialRef.current) {
      onMaterialRef(materialRef.current);
    }
  }, [onMaterialRef]);

  useEffect(() => {
    if (!multiViewData || !materialRef.current) return;

    const material = materialRef.current;
    const colorTextures = multiViewData.color;
    const depthTextures = multiViewData.depth;

    // Update texture uniforms
    for (let i = 0; i < 9; i++) {
      (material.uniforms as any)[`mvTex${i}`].value = colorTextures[i] || null;
      (material.uniforms as any)[`mvDepth${i}`].value = depthTextures[i] || null;
    }

    // Update camera data
    (material.uniforms.camPos.value as Float32Array).set(multiViewData.pos);
    (material.uniforms.camVP.value as Float32Array).set(multiViewData.vp);
    (material.uniforms.texel.value as Float32Array).set(multiViewData.texel);
    (material.uniforms.nearP.value as Float32Array).set(multiViewData.near);
    (material.uniforms.farP.value as Float32Array).set(multiViewData.far);

    (material.uniforms.viewCount as any).value = Math.min(9, colorTextures.filter(tex => !!tex).length);
    material.needsUpdate = true;
  }, [multiViewData]);

  return (
    <mesh ref={meshRef} position={[0, 1.7, 0]}>
      <icosahedronGeometry args={[1, 1]} />
      {/* @ts-ignore */}
      <avatarMaterial ref={materialRef} transparent={false} />
    </mesh>
  );
}

// =============================================================================
// Main Avatar Impostor Component
// =============================================================================
export default function AvatarImpostor() {
  const [multiViewData, setMultiViewData] = useState<MultiViewData | null>(null);
  const materialRef = useRef<THREE.ShaderMaterial | null>(null);

  const {
    cameraCount,
    textureSize,
    rotationSpeed,
    radius,
    height,
    captureInterval,
    bias,
    gradEnable,
    gradLo,
    gradHi,
    occEps,
    diagMode,
    diagMix
  } = useControls('Avatar Impostor', {
    'Camera System': folder({
      cameraCount: { value: 9, min: 3, max: 9, step: 1 },
      textureSize: { value: 512, options: [256, 512, 1024] },
      rotationSpeed: { value: 0.35, min: 0, max: 2, step: 0.05 },
      radius: { value: 2.0, min: 1, max: 4, step: 0.1 },
      height: { value: 2.4, min: 1, max: 4, step: 0.1 },
      captureInterval: { value: 1, min: 1, max: 10, step: 1 }
    }),
    'Rendering': folder({
      bias: { value: 0.02, min: 0, max: 0.1, step: 0.005 },
      gradEnable: true,
      gradLo: { value: 0.02, min: 0, max: 0.1, step: 0.005 },
      gradHi: { value: 0.12, min: 0, max: 0.3, step: 0.01 },
      occEps: { value: 0.02, min: 0, max: 0.1, step: 0.005 }
    }),
    'Debug': folder({
      diagMode: {
        value: 0,
        options: {
          'None': 0,
          'View ID': 1,
          'Quality': 2,
          'Occlusion': 3,
          'Confidence': 4,
          'Cos Theta': 5
        }
      },
      diagMix: { value: 0.65, min: 0, max: 1, step: 0.01 }
    }),
    'Actions': folder({
      'Export Data': button(() => exportImpostorData()),
      'Reset Camera': button(() => resetCameraSystem()),
      'Toggle Stats': button(() => togglePerformanceStats())
    })
  });

  const handleMaterialRef = (material: THREE.ShaderMaterial) => {
    materialRef.current = material;
  };

  // Update material uniforms when controls change
  useFrame(() => {
    if (!materialRef.current) return;

    const uniforms = materialRef.current.uniforms;
    (uniforms.diagMode as any).value = diagMode;
    (uniforms.diagMix as any).value = diagMix;
    (uniforms.gradEnable as any).value = gradEnable ? 1 : 0;
    (uniforms.bias as any).value = bias;
    (uniforms.gradLo as any).value = gradLo;
    (uniforms.gradHi as any).value = gradHi;
    (uniforms.occEps as any).value = occEps;
  });

  const exportImpostorData = () => {
    if (!multiViewData) {
      console.log('No multi-view data to export');
      return;
    }

    const data = {
      viewCount: multiViewData.color.filter(tex => !!tex).length,
      textureSize,
      cameraPositions: Array.from(multiViewData.pos),
      timestamp: Date.now()
    };

    console.log('Impostor data exported:', data);
  };

  const resetCameraSystem = () => {
    setMultiViewData(null);
    console.log('Camera system reset');
  };

  const togglePerformanceStats = () => {
    const stats = {
      activeViews: multiViewData?.color.filter(tex => !!tex).length || 0,
      textureMemory: (textureSize * textureSize * 4 * cameraCount) / (1024 * 1024),
      captureRate: 1000 / captureInterval
    };
    console.log('Performance stats:', stats);
  };

  return (
    <div style={{ width: '100%', height: '100vh', display: 'flex', flexDirection: 'column' }}>
      {/* 3D Scene */}
      <div style={{ flex: 1, background: '#1a1a2e' }}>
        <Canvas camera={{ position: [4, 3, 4], fov: 60 }}>
          <ambientLight intensity={0.3} />
          <directionalLight position={[5, 5, 5]} intensity={0.7} castShadow />

          <Platform />
          <HexGridLines />

          <CameraRingLive
            onMultiView={setMultiViewData}
            cameraCount={cameraCount}
            textureSize={textureSize}
            rotationSpeed={rotationSpeed}
            radius={radius}
            height={height}
            captureInterval={captureInterval}
          />

          <Avatar
            multiViewData={multiViewData}
            onMaterialRef={handleMaterialRef}
          />

          <OrbitControls enablePan={true} enableZoom={true} enableRotate={true} />
        </Canvas>
      </div>

      {/* Status Panel */}
      <div style={{
        height: '150px',
        background: '#1a1a1a',
        color: '#fff',
        padding: '20px',
        display: 'flex',
        gap: '20px',
        overflow: 'auto'
      }}>
        <div style={{ flex: 1 }}>
          <h3>📹 Camera System</h3>
          <div>Active Cameras: {cameraCount}</div>
          <div>Texture Size: {textureSize}x{textureSize}</div>
          <div>Rotation Speed: {rotationSpeed.toFixed(2)}</div>
          <div>Capture Rate: {(1000/captureInterval).toFixed(0)} fps</div>
        </div>

        <div style={{ flex: 1 }}>
          <h3>🎭 Impostor Status</h3>
          <div>Active Views: {multiViewData?.color.filter(tex => !!tex).length || 0}</div>
          <div>Debug Mode: {['None', 'View ID', 'Quality', 'Occlusion', 'Confidence', 'Cos Theta'][diagMode]}</div>
          <div>Gradient Filter: {gradEnable ? 'ON' : 'OFF'}</div>
          <div>Texture Memory: {((textureSize * textureSize * 4 * cameraCount) / (1024 * 1024)).toFixed(1)} MB</div>
        </div>

        <div style={{ flex: 1 }}>
          <h3>⚙️ Rendering Info</h3>
          <div>Surface Bias: {bias.toFixed(3)}</div>
          <div>Occlusion Epsilon: {occEps.toFixed(3)}</div>
          <div>Gradient Range: {gradLo.toFixed(3)} - {gradHi.toFixed(3)}</div>
          <div>Debug Mix: {(diagMix * 100).toFixed(0)}%</div>
        </div>
      </div>
    </div>
  );
}
