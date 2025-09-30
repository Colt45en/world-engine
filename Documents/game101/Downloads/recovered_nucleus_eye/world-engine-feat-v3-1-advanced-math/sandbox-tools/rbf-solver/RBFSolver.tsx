// RBFSolver.tsx - Standalone Radial Basis Function Poisson Solver
// Meshless numerical PDE solver using RBF interpolation with multiple kernel types

import React, { useEffect, useMemo, useRef, useState } from "react";
import * as THREE from "three";
import { Canvas, useFrame } from "@react-three/fiber";
import { OrbitControls, Html, Line } from "@react-three/drei";
import { Leva, useControls } from "leva";

// Mathematical Utilities
function seedRandom(seed: number) {
  let state = seed >>> 0;
  return () => (state = (state * 1664525 + 1013904223) >>> 0) / 2 ** 32;
}

function gaussianNoise(rand: () => number, sigma: number): number {
  const u = 1 - rand();
  const v = 1 - rand();
  return sigma * Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

// Linear System Solver (Gaussian Elimination with Partial Pivoting)
function solveLinearSystem(A: number[][], b: number[]): number[] {
  const n = A.length;

  // Create copies to avoid modifying originals
  const matrix = A.map(row => row.slice());
  const rhs = b.slice();

  // Forward elimination with partial pivoting
  for (let k = 0; k < n; k++) {
    // Find pivot
    let pivotRow = k;
    let maxValue = Math.abs(matrix[k][k]);

    for (let i = k + 1; i < n; i++) {
      const value = Math.abs(matrix[i][k]);
      if (value > maxValue) {
        maxValue = value;
        pivotRow = i;
      }
    }

    // Swap rows if needed
    if (pivotRow !== k) {
      [matrix[k], matrix[pivotRow]] = [matrix[pivotRow], matrix[k]];
      [rhs[k], rhs[pivotRow]] = [rhs[pivotRow], rhs[k]];
    }

    // Eliminate column
    for (let i = k + 1; i < n; i++) {
      if (Math.abs(matrix[k][k]) < 1e-12) continue;
      const factor = matrix[i][k] / matrix[k][k];
      for (let j = k; j < n; j++) {
        matrix[i][j] -= factor * matrix[k][j];
      }
      rhs[i] -= factor * rhs[k];
    }
  }

  // Back substitution
  const solution = new Array(n).fill(0);
  for (let i = n - 1; i >= 0; i--) {
    let sum = rhs[i];
    for (let j = i + 1; j < n; j++) {
      sum -= matrix[i][j] * solution[j];
    }
    solution[i] = sum / matrix[i][i];
  }

  return solution;
}

// RBF Kernel Functions
type KernelType = 'IMQ' | 'Gaussian' | 'Multiquadric' | 'ThinPlate';

interface RBFKernel {
  phi: (r: number) => number;
  laplacian: (r: number) => number;
}

function createKernel(type: KernelType, params: { c?: number; beta?: number; eps?: number }): RBFKernel {
  switch (type) {
    case 'IMQ': // Inverse Multiquadric
      const { c = 0.25, beta = -0.5 } = params;
      return {
        phi: (r: number) => Math.pow(c * c + r * r, beta),
        laplacian: (r: number) => {
          const s = c * c + r * r;
          return 4 * beta * Math.pow(s, beta - 1) + 4 * beta * (beta - 1) * r * r * Math.pow(s, beta - 2);
        }
      };

    case 'Gaussian':
      const { eps = 2.0 } = params;
      return {
        phi: (r: number) => Math.exp(-(eps * r) * (eps * r)),
        laplacian: (r: number) => {
          const phi = Math.exp(-(eps * r) * (eps * r));
          return 4 * eps * eps * phi * ((eps * eps) * r * r - 1);
        }
      };

    case 'Multiquadric':
      const c_mq = params.c || 0.25;
      return {
        phi: (r: number) => Math.sqrt(c_mq * c_mq + r * r),
        laplacian: (r: number) => {
          const s = c_mq * c_mq + r * r;
          return 1 / Math.sqrt(s) + r * r / Math.pow(s, 1.5);
        }
      };

    case 'ThinPlate':
      return {
        phi: (r: number) => r === 0 ? 0 : r * r * Math.log(r),
        laplacian: (r: number) => r === 0 ? 0 : 4 * Math.log(r) + 4
      };

    default:
      throw new Error(`Unknown kernel type: ${type}`);
  }
}

// Test Problem: (1-x²)(1-y²) with Poisson equation
function analyticalSolution(x: number, y: number): number {
  return (1 - x * x) * (1 - y * y);
}

function poissonSource(x: number, y: number, b: number): number {
  const r2 = x * x + y * y;
  const negativeLaplacian = -(-4 + 2 * r2); // Analytical Laplacian of u_true
  return negativeLaplacian + b * analyticalSolution(x, y);
}

// Center Point Generation
interface RBFCenter {
  x: number;
  y: number;
  isBoundary: boolean;
}

function generateCenters(
  interiorCount: number,
  boundaryPerEdge: number,
  seed: number
): RBFCenter[] {
  const rand = seedRandom(seed);
  const centers: RBFCenter[] = [];

  // Interior points (scattered)
  const gridSize = Math.ceil(Math.sqrt(interiorCount));
  const margin = 0.05;

  for (let i = 0; i < gridSize && centers.length < interiorCount; i++) {
    for (let j = 0; j < gridSize && centers.length < interiorCount; j++) {
      const x = -1 + margin + (i + 0.5 + 0.3 * (rand() - 0.5)) * ((2 - 2 * margin) / gridSize);
      const y = -1 + margin + (j + 0.5 + 0.3 * (rand() - 0.5)) * ((2 - 2 * margin) / gridSize);

      // Ensure point is within bounds
      if (x > -1 + margin && x < 1 - margin && y > -1 + margin && y < 1 - margin) {
        centers.push({ x, y, isBoundary: false });
      }
    }
  }

  // Boundary points (4 edges)
  const edges = [
    (t: number) => ({ x: -1, y: -1 + 2 * t }),       // left edge
    (t: number) => ({ x: 1, y: -1 + 2 * t }),        // right edge
    (t: number) => ({ x: -1 + 2 * t, y: -1 }),       // bottom edge
    (t: number) => ({ x: -1 + 2 * t, y: 1 })         // top edge
  ];

  edges.forEach(edgeFunc => {
    for (let i = 0; i < boundaryPerEdge; i++) {
      const t = (i + 0.5) / boundaryPerEdge;
      const point = edgeFunc(t);
      centers.push({ x: point.x, y: point.y, isBoundary: true });
    }
  });

  return centers;
}

// Main RBF Solver
function solveRBFProblem(
  kernelType: KernelType,
  kernelParams: { c?: number; beta?: number; eps?: number },
  interiorCount: number,
  boundaryPerEdge: number,
  b: number,
  noiseLevel: number,
  seed: number
) {
  const centers = generateCenters(interiorCount, boundaryPerEdge, seed);
  const n = centers.length;
  const kernel = createKernel(kernelType, kernelParams);
  const rand = seedRandom(seed + 1000); // Different seed for noise

  // Build system matrix
  const A = Array.from({ length: n }, () => Array(n).fill(0));
  const rhs = Array(n).fill(0);

  for (let i = 0; i < n; i++) {
    const center_i = centers[i];

    for (let j = 0; j < n; j++) {
      const center_j = centers[j];
      const distance = Math.hypot(center_i.x - center_j.x, center_i.y - center_j.y);

      if (center_i.isBoundary) {
        // Boundary condition: u = u_true
        A[i][j] = kernel.phi(distance);
      } else {
        // Interior condition: -∆u + b*u = f
        A[i][j] = -kernel.laplacian(distance) + b * kernel.phi(distance);
      }
    }

    // Right-hand side
    if (center_i.isBoundary) {
      rhs[i] = analyticalSolution(center_i.x, center_i.y);
    } else {
      rhs[i] = poissonSource(center_i.x, center_i.y, b) + gaussianNoise(rand, noiseLevel);
    }
  }

  // Solve system
  const coefficients = solveLinearSystem(A, rhs);

  return { centers, coefficients, kernel };
}

// RBF Solver Component
function RBFSolverTool() {
  const [solution, setSolution] = useState<any>(null);
  const [error, setError] = useState<number>(0);
  const [isComputing, setIsComputing] = useState(false);
  const meshRef = useRef<THREE.Mesh>(null!);

  const {
    kernelType,
    c,
    beta,
    eps,
    interiorCount,
    boundaryPerEdge,
    gridResolution,
    b,
    noiseLevel,
    seed,
    showCenters,
    showError
  } = useControls("RBF Solver", {
    kernelType: { value: 'IMQ' as KernelType, options: ['IMQ', 'Gaussian', 'Multiquadric', 'ThinPlate'] },
    c: { value: 0.25, min: 0.05, max: 1, step: 0.01 },
    beta: { value: -0.5, min: -1.5, max: 0, step: 0.01 },
    eps: { value: 2.0, min: 0.2, max: 5, step: 0.05 },
    interiorCount: { value: 196, min: 36, max: 625, step: 1 },
    boundaryPerEdge: { value: 24, min: 8, max: 64, step: 1 },
    gridResolution: { value: 96, min: 48, max: 192, step: 4 },
    b: { value: 1.0, min: 0, max: 5, step: 0.05 },
    noiseLevel: { value: 0.002, min: 0, max: 0.02, step: 0.0005 },
    seed: { value: 42, min: 1, max: 9999, step: 1 },
    showCenters: false,
    showError: false
  });

  const kernelParams = useMemo(() => ({ c, beta, eps }), [c, beta, eps]);

  const { geometry, centerPoints, errorField } = useMemo(() => {
    if (!solution) return { geometry: null, centerPoints: [], errorField: null };

    const { centers, coefficients, kernel } = solution;
    const res = gridResolution;

    // Evaluate solution on grid
    const positions: number[] = [];
    const colors: number[] = [];
    const indices: number[] = [];
    let totalError = 0;
    let errorCount = 0;

    const evaluateAt = (x: number, y: number): number => {
      let value = 0;
      for (let j = 0; j < centers.length; j++) {
        const distance = Math.hypot(x - centers[j].x, y - centers[j].y);
        value += coefficients[j] * kernel.phi(distance);
      }
      return value;
    };

    // Generate grid geometry
    for (let j = 0; j < res; j++) {
      for (let i = 0; i < res; i++) {
        const x = -1 + (2 * i) / (res - 1);
        const y = -1 + (2 * j) / (res - 1);
        const z = evaluateAt(x, y) * 0.5; // Scale for visualization

        positions.push(x, y, z);

        // Error coloring
        const trueValue = analyticalSolution(x, y);
        const error = Math.abs(z / 0.5 - trueValue);
        totalError += error * error;
        errorCount++;

        const errorColor = Math.min(1, error * 10); // Scale error for visibility
        colors.push(1 - errorColor, 1 - errorColor * 0.5, 1 - errorColor * 0.2);
      }
    }

    // Generate indices
    for (let j = 0; j < res - 1; j++) {
      for (let i = 0; i < res - 1; i++) {
        const a = j * res + i;
        const b = j * res + i + 1;
        const c = (j + 1) * res + i;
        const d = (j + 1) * res + i + 1;

        indices.push(a, c, b, b, c, d);
      }
    }

    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
    geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
    geometry.setIndex(indices);
    geometry.computeVertexNormals();

    // Center points for visualization
    const centerPoints = centers.map(center => [center.x, center.y, 0.01]);

    setError(Math.sqrt(totalError / errorCount));

    return { geometry, centerPoints, errorField: null };
  }, [solution, gridResolution]);

  // Compute solution
  useEffect(() => {
    setIsComputing(true);

    // Use timeout to allow UI to update
    const timer = setTimeout(() => {
      try {
        const newSolution = solveRBFProblem(
          kernelType,
          kernelParams,
          interiorCount,
          boundaryPerEdge,
          b,
          noiseLevel,
          seed
        );
        setSolution(newSolution);
      } catch (error) {
        console.error('RBF computation failed:', error);
        setSolution(null);
      } finally {
        setIsComputing(false);
      }
    }, 100);

    return () => clearTimeout(timer);
  }, [kernelType, kernelParams, interiorCount, boundaryPerEdge, b, noiseLevel, seed]);

  return (
    <group>
      {/* Solution surface */}
      {geometry && (
        <mesh
          ref={meshRef}
          geometry={geometry}
          rotation={[-Math.PI / 2, 0, 0]}
          castShadow
          receiveShadow
        >
          <meshStandardMaterial
            vertexColors={showError}
            color={showError ? 0xffffff : "#ffd080"}
            metalness={0.1}
            roughness={0.85}
          />
        </mesh>
      )}

      {/* Center points visualization */}
      {showCenters && centerPoints.length > 0 && (
        <group>
          {centerPoints.map((point, i) => (
            <mesh key={i} position={point}>
              <sphereGeometry args={[0.01, 8, 6]} />
              <meshBasicMaterial color={solution?.centers[i]?.isBoundary ? "#ff4444" : "#44ff44"} />
            </mesh>
          ))}
        </group>
      )}

      {/* Info Panel */}
      <Html position={[1.2, 0.8, 0]}>
        <div
          style={{
            padding: 12,
            background: "rgba(20, 24, 36, 0.9)",
            border: "1px solid rgba(255, 255, 255, 0.08)",
            borderRadius: 10,
            color: "#e7eefc",
            fontSize: 13,
            minWidth: 280,
            maxWidth: 320
          }}
        >
          <div style={{ fontWeight: 700, marginBottom: 8 }}>
            RBF Poisson Solver
          </div>

          <div style={{ display: 'grid', gap: 4, fontSize: 12 }}>
            <div>
              <strong>Kernel:</strong> {kernelType}
              {kernelType === 'IMQ' && ` (c=${c.toFixed(2)}, β=${beta.toFixed(2)})`}
              {kernelType === 'Gaussian' && ` (ε=${eps.toFixed(2)})`}
            </div>

            <div>
              <strong>Centers:</strong> {interiorCount} interior + {4 * boundaryPerEdge} boundary
            </div>

            <div>
              <strong>Grid:</strong> {gridResolution}²
            </div>

            <div>
              <strong>L² Error:</strong> {error.toExponential(3)}
            </div>

            <div>
              <strong>Problem:</strong> -∆u + {b}u = f, u|∂Ω = g
            </div>

            <div style={{ marginTop: 8, fontSize: 11, opacity: 0.8 }}>
              {isComputing ? "⏳ Computing..." : "✅ Solution ready"}
            </div>
          </div>
        </div>
      </Html>
    </group>
  );
}

// Main App
export default function RBFSolverApp() {
  return (
    <div style={{ width: '100%', height: '100vh', background: '#0a0e16' }}>
      <Leva collapsed={false} />
      <Canvas camera={{ position: [2, 1.5, 2], fov: 60 }} shadows>
        <color attach="background" args={["#0a0e16"]} />
        <ambientLight intensity={0.4} />
        <directionalLight
          position={[4, 6, 3]}
          intensity={0.8}
          castShadow
          shadow-mapSize-width={2048}
          shadow-mapSize-height={2048}
        />
        <RBFSolverTool />
        <OrbitControls enablePan={true} />
      </Canvas>
    </div>
  );
}
