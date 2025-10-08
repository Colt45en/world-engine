import React, { useMemo, useState, useCallback } from "react";
import * as THREE from "three";
import { Canvas } from "@react-three/fiber";
import { OrbitControls, Html } from "@react-three/drei";
import { useControls, button, folder } from "leva";
import { mathEngine } from '../shared/utils';

// =============================================================================
// Core Types and Utilities
// =============================================================================
type MSPoint = [number, number];

interface MaskData {
  data: Uint8Array;
  width: number;
  height: number;
}

interface ContourExtrudeSettings {
  threshold: number;
  simplifyEps: number;
  extrudeDepth: number;
  bevelEnabled: boolean;
  bevelThickness: number;
  bevelSize: number;
  bevelSegments: number;
  steps: number;
}

// =============================================================================
// Image Processing and Masking
// =============================================================================
function imageDataToMask(img: ImageData, threshold = 0.5): MaskData {
  const { width: W, height: H, data } = img;
  const out = new Uint8Array(W * H);

  for (let i = 0, p = 0; i < data.length; i += 4, p++) {
    const r = data[i];
    const g = data[i + 1];
    const b = data[i + 2];

    // Convert to grayscale using luminance formula
    const luminance = (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255;
    out[p] = luminance >= threshold ? 1 : 0;
  }

  return { data: out, width: W, height: H };
}

// =============================================================================
// Marching Squares Contour Detection
// =============================================================================
function marchingSquares(mask: MaskData, simplifyEps = 1.0): MSPoint[] {
  const { data, width: W, height: H } = mask;
  const visited = new Uint8Array(W * H);
  const contours: MSPoint[][] = [];

  // 8-directional neighbors
  const dirs: [number, number][] = [
    [1, 0], [1, 1], [0, 1], [-1, 1],
    [-1, 0], [-1, -1], [0, -1], [1, -1]
  ];

  const inside = (x: number, y: number) => x >= 0 && y >= 0 && x < W && y < H;
  const at = (x: number, y: number) => inside(x, y) ? data[y * W + x] : 0;

  // Check if a pixel is on the boundary
  const isBoundary = (x: number, y: number) => {
    if (!at(x, y)) return false;

    for (let k = 0; k < 8; k++) {
      const nx = x + dirs[k][0];
      const ny = y + dirs[k][1];
      if (!inside(nx, ny) || at(nx, ny) === 0) return true;
    }
    return false;
  };

  // Find contours using boundary following
  for (let y = 1; y < H - 1; y++) {
    for (let x = 1; x < W - 1; x++) {
      const idx = y * W + x;
      if (visited[idx] || !isBoundary(x, y)) continue;

      let cx = x, cy = y, dir = 6; // Starting direction
      const contour: MSPoint[] = [];
      const sx = x, sy = y; // Starting point
      let safety = W * H * 4; // Safety counter to prevent infinite loops

      do {
        visited[cy * W + cx] = 1;
        contour.push([cx + 0.5, cy + 0.5]); // Offset for sub-pixel precision

        let found = false;
        // Look for next boundary pixel in clockwise order
        for (let turn = 0; turn < 8; turn++) {
          const ndir = (dir + 7 + turn) % 8;
          const nx = cx + dirs[ndir][0];
          const ny = cy + dirs[ndir][1];

          if (inside(nx, ny) && isBoundary(nx, ny)) {
            cx = nx;
            cy = ny;
            dir = ndir;
            found = true;
            break;
          }
        }

        if (!found) break;
        if (--safety <= 0) break;
      } while (!(cx === sx && cy === sy));

      if (contour.length > 2) {
        // Close the contour if not already closed
        if (contour[0] !== contour[contour.length - 1]) {
          contour.push(contour[0]);
        }
        contours.push(contour);
      }
    }
  }

  if (!contours.length) return [];

  // Find the longest contour (main shape)
  let bestContour = contours[0];
  let bestLength = polygonLength(bestContour);

  for (let i = 1; i < contours.length; i++) {
    const length = polygonLength(contours[i]);
    if (length > bestLength) {
      bestContour = contours[i];
      bestLength = length;
    }
  }

  return simplifyPolygon(bestContour, simplifyEps, true);
}

// =============================================================================
// Polygon Utilities
// =============================================================================
function polygonLength(points: MSPoint[]): number {
  let length = 0;
  for (let i = 1; i < points.length; i++) {
    const dx = points[i][0] - points[i - 1][0];
    const dy = points[i][1] - points[i - 1][1];
    length += Math.hypot(dx, dy);
  }
  return length;
}

// Ramer-Douglas-Peucker algorithm for polygon simplification
function simplifyPolygon(points: MSPoint[], epsilon: number, closed: boolean): MSPoint[] {
  if (points.length <= 2 || epsilon <= 0) return points.slice();

  let pts = points.slice();
  if (closed && (pts[0][0] !== pts[pts.length - 1][0] || pts[0][1] !== pts[pts.length - 1][1])) {
    pts.push(pts[0]);
  }

  const keep = new Uint8Array(pts.length);
  keep[0] = 1;
  keep[pts.length - 1] = 1;

  // Distance from point to line segment
  const distanceToSegment = (p: MSPoint, a: MSPoint, b: MSPoint): number => {
    const vx = b[0] - a[0];
    const vy = b[1] - a[1];
    const wx = p[0] - a[0];
    const wy = p[1] - a[1];

    const c1 = vx * wx + vy * wy;
    const c2 = vx * vx + vy * vy;
    let t = c2 ? c1 / c2 : 0;
    t = Math.max(0, Math.min(1, t));

    const dx = a[0] + t * vx - p[0];
    const dy = a[1] + t * vy - p[1];
    return Math.hypot(dx, dy);
  };

  // Recursive simplification
  const simplify = (i: number, j: number) => {
    let maxDistance = -1;
    let maxIndex = -1;

    for (let k = i + 1; k < j; k++) {
      const distance = distanceToSegment(pts[k], pts[i], pts[j]);
      if (distance > maxDistance) {
        maxDistance = distance;
        maxIndex = k;
      }
    }

    if (maxDistance > epsilon) {
      keep[maxIndex] = 1;
      simplify(i, maxIndex);
      simplify(maxIndex, j);
    }
  };

  simplify(0, pts.length - 1);

  const result: MSPoint[] = [];
  for (let i = 0; i < pts.length; i++) {
    if (keep[i]) result.push(pts[i]);
  }

  if (closed && (result[0][0] !== result[result.length - 1][0] || result[0][1] !== result[result.length - 1][1])) {
    result.push(result[0]);
  }

  return result;
}

function polygonToShape(points: MSPoint[]): THREE.Shape {
  const shape = new THREE.Shape();
  if (!points.length) return shape;

  shape.moveTo(points[0][0], points[0][1]);
  for (let i = 1; i < points.length; i++) {
    shape.lineTo(points[i][0], points[i][1]);
  }
  shape.lineTo(points[0][0], points[0][1]);

  return shape;
}

// =============================================================================
// Canvas Drawing Functions
// =============================================================================
function createTextCanvas(
  text: string,
  width: number,
  height: number,
  fontSize: number = 160,
  font: string = 'bold'
): ImageData {
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d')!;

  // Black background
  ctx.fillStyle = '#000';
  ctx.fillRect(0, 0, width, height);

  // White text
  ctx.fillStyle = '#fff';
  ctx.font = `${font} ${fontSize}px system-ui,Segoe UI,Arial`;
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.translate(width * 0.5, height * 0.5);
  ctx.rotate(-0.08); // Slight rotation
  ctx.fillText(text, 0, 10);

  return ctx.getImageData(0, 0, width, height);
}

function createShapeCanvas(width: number, height: number, shapeType: string): ImageData {
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d')!;

  // Black background
  ctx.fillStyle = '#000';
  ctx.fillRect(0, 0, width, height);

  // White shape
  ctx.fillStyle = '#fff';
  ctx.translate(width * 0.5, height * 0.5);

  switch (shapeType) {
    case 'circle':
      ctx.beginPath();
      ctx.arc(0, 0, Math.min(width, height) * 0.3, 0, Math.PI * 2);
      ctx.fill();
      break;

    case 'star':
      const points = 5;
      const outer = Math.min(width, height) * 0.3;
      const inner = outer * 0.5;
      ctx.beginPath();
      for (let i = 0; i < points * 2; i++) {
        const angle = (i * Math.PI) / points;
        const radius = i % 2 === 0 ? outer : inner;
        const x = Math.cos(angle) * radius;
        const y = Math.sin(angle) * radius;
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.closePath();
      ctx.fill();
      break;

    case 'heart':
      const scale = Math.min(width, height) * 0.01;
      ctx.beginPath();
      for (let t = 0; t < Math.PI * 2; t += 0.1) {
        const x = 16 * Math.pow(Math.sin(t), 3) * scale;
        const y = -(13 * Math.cos(t) - 5 * Math.cos(2 * t) - 2 * Math.cos(3 * t) - Math.cos(4 * t)) * scale;
        if (t === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.closePath();
      ctx.fill();
      break;

    default:
      // Rectangle
      ctx.fillRect(-width * 0.25, -height * 0.25, width * 0.5, height * 0.5);
  }

  return ctx.getImageData(0, 0, width, height);
}

// =============================================================================
// Contour Extrude Mesh Component
// =============================================================================
interface ContourMeshProps {
  imageData: ImageData | null;
  settings: ContourExtrudeSettings;
  onStatsUpdate: (stats: { contourPoints: number; triangles: number; vertices: number }) => void;
}

function ContourMesh({ imageData, settings, onStatsUpdate }: ContourMeshProps) {
  const geometry = useMemo(() => {
    if (!imageData) {
      onStatsUpdate({ contourPoints: 0, triangles: 0, vertices: 0 });
      return new THREE.BoxGeometry(1, 1, 1);
    }

    const mask = imageDataToMask(imageData, settings.threshold);
    const contour = marchingSquares(mask, settings.simplifyEps);

    if (!contour.length) {
      onStatsUpdate({ contourPoints: 0, triangles: 0, vertices: 0 });
      return new THREE.BoxGeometry(1, 1, 1);
    }

    // Normalize coordinates to [-1, 1] range
    const minX = 0, minY = 0;
    const maxX = mask.width, maxY = mask.height;
    const scaleX = 2 / (maxX - minX);
    const scaleY = 2 / (maxY - minY);

    const normalizedContour: MSPoint[] = contour.map(([x, y]) => [
      (x - minX) * scaleX - 1,
      (y - minY) * scaleY - 1
    ]);

    const shape = polygonToShape(normalizedContour);

    const extrudeGeometry = new THREE.ExtrudeGeometry(shape, {
      depth: settings.extrudeDepth,
      bevelEnabled: settings.bevelEnabled,
      bevelThickness: settings.bevelThickness,
      bevelSize: settings.bevelSize,
      bevelSegments: settings.bevelSegments,
      steps: settings.steps
    });

    extrudeGeometry.center();

    // Calculate statistics
    const vertices = extrudeGeometry.attributes.position.count;
    const triangles = extrudeGeometry.index ? extrudeGeometry.index.count / 3 : vertices / 3;

    onStatsUpdate({
      contourPoints: contour.length,
      triangles: Math.floor(triangles),
      vertices
    });

    return extrudeGeometry;
  }, [imageData, settings, onStatsUpdate]);

  return (
    <mesh geometry={geometry} castShadow receiveShadow rotation-y={Math.PI * 0.1}>
      <meshStandardMaterial
        color="#8be8d8"
        metalness={0.25}
        roughness={0.35}
        flatShading={false}
      />
    </mesh>
  );
}

// =============================================================================
// Main Contour Extrude Component
// =============================================================================
export default function ContourExtrude() {
  const [imageData, setImageData] = useState<ImageData | null>(null);
  const [stats, setStats] = useState({ contourPoints: 0, triangles: 0, vertices: 0 });
  const [processingTime, setProcessingTime] = useState(0);

  const settings = useControls('Contour Detection', {
    'Image Source': folder({
      sourceType: {
        value: 'text',
        options: { 'Text': 'text', 'Circle': 'circle', 'Star': 'star', 'Heart': 'heart', 'Rectangle': 'rectangle' }
      },
      customText: { value: 'NF', hint: 'Custom text to render' },
      threshold: { value: 0.5, min: 0.1, max: 0.9, step: 0.01 }
    }),
    'Contour Processing': folder({
      simplifyEps: { value: 1.2, min: 0.1, max: 5.0, step: 0.1 },
      extrudeDepth: { value: 0.3, min: 0.05, max: 1.0, step: 0.01 }
    }),
    'Bevel Settings': folder({
      bevelEnabled: true,
      bevelThickness: { value: 0.04, min: 0.005, max: 0.1, step: 0.005 },
      bevelSize: { value: 0.04, min: 0.005, max: 0.1, step: 0.005 },
      bevelSegments: { value: 4, min: 1, max: 12, step: 1 }
    }),
    'Advanced': folder({
      steps: { value: 1, min: 1, max: 10, step: 1 },
      'Generate New': button(() => generateImage()),
      'Export Contour': button(() => exportContourData()),
      'Load Image': button(() => loadCustomImage())
    })
  });

  const generateImage = useCallback(() => {
    const startTime = performance.now();

    let newImageData: ImageData;
    const width = 256, height = 256;

    if (settings.sourceType === 'text') {
      newImageData = createTextCanvas(settings.customText, width, height);
    } else {
      newImageData = createShapeCanvas(width, height, settings.sourceType);
    }

    setImageData(newImageData);
    setProcessingTime(performance.now() - startTime);
  }, [settings.sourceType, settings.customText]);

  const exportContourData = () => {
    if (!imageData) return;

    const mask = imageDataToMask(imageData, settings.threshold);
    const contour = marchingSquares(mask, settings.simplifyEps);

    const data = {
      contour: contour,
      settings: settings,
      stats: stats,
      timestamp: Date.now()
    };

    console.log('Contour data exported:', data);

    // Create downloadable JSON file
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `contour-${settings.sourceType}-${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const loadCustomImage = () => {
    const input = document.createElement('input');
    input.type = 'file';
    input.accept = 'image/*';
    input.onchange = (e) => {
      const file = (e.target as HTMLInputElement).files?.[0];
      if (!file) return;

      const img = new Image();
      img.onload = () => {
        const canvas = document.createElement('canvas');
        canvas.width = 256;
        canvas.height = 256;
        const ctx = canvas.getContext('2d')!;

        // Draw image scaled to fit canvas
        ctx.fillStyle = '#000';
        ctx.fillRect(0, 0, 256, 256);

        const scale = Math.min(256 / img.width, 256 / img.height);
        const width = img.width * scale;
        const height = img.height * scale;
        const x = (256 - width) / 2;
        const y = (256 - height) / 2;

        ctx.drawImage(img, x, y, width, height);

        const imageData = ctx.getImageData(0, 0, 256, 256);
        setImageData(imageData);
        setProcessingTime(performance.now() - performance.now());
      };
      img.src = URL.createObjectURL(file);
    };
    input.click();
  };

  // Generate initial image
  React.useEffect(() => {
    generateImage();
  }, [generateImage]);

  return (
    <div style={{ width: '100%', height: '100vh', display: 'flex', flexDirection: 'column' }}>
      {/* 3D Scene */}
      <div style={{ flex: 1, background: '#0f0f0f' }}>
        <Canvas camera={{ position: [3, 2, 3], fov: 60 }}>
          <ambientLight intensity={0.4} />
          <directionalLight position={[5, 5, 5]} intensity={0.8} castShadow />
          <spotLight position={[-3, 4, 3]} intensity={0.6} angle={0.3} penumbra={0.2} />

          <ContourMesh
            imageData={imageData}
            settings={settings}
            onStatsUpdate={setStats}
          />

          <OrbitControls enablePan={true} enableZoom={true} enableRotate={true} />
        </Canvas>
      </div>

      {/* Status Panel */}
      <div style={{
        height: '120px',
        background: '#1a1a1a',
        color: '#fff',
        padding: '15px 20px',
        display: 'flex',
        gap: '20px',
        overflow: 'auto'
      }}>
        <div style={{ flex: 1 }}>
          <h3>🔍 Contour Detection</h3>
          <div>Source: {settings.sourceType === 'text' ? `"${settings.customText}"` : settings.sourceType}</div>
          <div>Threshold: {(settings.threshold * 100).toFixed(0)}%</div>
          <div>Simplification: {settings.simplifyEps.toFixed(1)}px</div>
        </div>

        <div style={{ flex: 1 }}>
          <h3>📐 Geometry Stats</h3>
          <div>Contour Points: {stats.contourPoints}</div>
          <div>Triangles: {stats.triangles.toLocaleString()}</div>
          <div>Vertices: {stats.vertices.toLocaleString()}</div>
        </div>

        <div style={{ flex: 1 }}>
          <h3>🏗️ Extrusion Settings</h3>
          <div>Depth: {settings.extrudeDepth.toFixed(2)}</div>
          <div>Bevel: {settings.bevelEnabled ? 'ON' : 'OFF'}</div>
          <div>Bevel Size: {settings.bevelSize.toFixed(3)}</div>
          <div>Processing: {processingTime.toFixed(1)}ms</div>
        </div>
      </div>
    </div>
  );
}
