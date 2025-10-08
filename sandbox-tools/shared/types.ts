// SharedTypes.ts - Common type definitions used across sandbox tools

import * as THREE from "three";

// Common 3D Vector types
export interface Vector3D {
    x: number;
    y: number;
    z: number;
}

export interface Vector4D {
    x: number;
    y: number;
    z: number;
    w: number;
}

// Material configuration
export interface MaterialConfig {
    color: THREE.ColorRepresentation;
    metalness?: number;
    roughness?: number;
    wireframe?: boolean;
    transparent?: boolean;
    opacity?: number;
}

// Export configuration
export interface ExportConfig {
    format: 'glb' | 'obj' | 'stl';
    name: string;
    scale?: number;
    optimize?: boolean;
}

// Kernel types for mathematical operations
export type KernelType = 'IMQ' | 'Gaussian' | 'Multiquadric' | 'ThinPlate';

export interface KernelParams {
    c?: number;      // Shape parameter for IMQ, Multiquadric
    beta?: number;   // Power parameter for IMQ
    eps?: number;    // Shape parameter for Gaussian
}

// Image processing types
export interface ImageMask {
    data: Uint8Array;
    width: number;
    height: number;
}

export interface VolumeData {
    data: Uint8Array;
    width: number;
    height: number;
    depth: number;
}

// Bounds and regions
export interface BoundingBox {
    min: Vector3D;
    max: Vector3D;
}

export interface Region2D {
    minX: number;
    maxX: number;
    minY: number;
    maxY: number;
}

// Animation and control types
export interface AnimationConfig {
    enabled: boolean;
    speed: number;
    loop: boolean;
    autoReverse?: boolean;
}

export interface CameraConfig {
    position: Vector3D;
    target?: Vector3D;
    fov?: number;
    near?: number;
    far?: number;
}

// UI component props
export interface ControlPanelProps {
    title: string;
    position?: Vector3D;
    collapsed?: boolean;
    style?: React.CSSProperties;
    children: React.ReactNode;
}

// Tool-specific base interfaces
export interface BaseTool {
    id: string;
    name: string;
    description: string;
    version: string;
    author?: string;
}

export interface RenderableTool extends BaseTool {
    render(): JSX.Element;
    dispose?(): void;
}

// Mathematical function types
export type UnaryFunction = (x: number) => number;
export type BinaryFunction = (x: number, y: number) => number;
export type TripleFunction = (x: number, y: number, z: number) => number;

// Solver configuration
export interface SolverConfig {
    tolerance: number;
    maxIterations: number;
    verbose?: boolean;
}

// Geometry processing
export interface MeshData {
    vertices: Float32Array;
    normals: Float32Array;
    indices: Uint32Array;
    uvs?: Float32Array;
}

export interface GeometryStats {
    vertexCount: number;
    faceCount: number;
    boundingBox: BoundingBox;
    volume?: number;
    surfaceArea?: number;
}

// Event types
export interface ToolEvent {
    type: string;
    timestamp: number;
    data?: any;
}

export interface ProgressEvent extends ToolEvent {
    type: 'progress';
    data: {
        current: number;
        total: number;
        message?: string;
    };
}

export interface ErrorEvent extends ToolEvent {
    type: 'error';
    data: {
        message: string;
        code?: string;
        stack?: string;
    };
}

export interface CompleteEvent extends ToolEvent {
    type: 'complete';
    data?: any;
}
