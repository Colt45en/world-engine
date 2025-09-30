// SharedUtils.ts - Common utility functions used across sandbox tools

import * as THREE from "three";
import { Vector3D, Vector4D, ImageMask, VolumeData, BoundingBox } from "./types";

// =============================================================================
// Enhanced Mathematical Core
// =============================================================================

/**
 * MathCore: Robust mathematical operations with validation
 */
class MathCore {
    static _ensureFinite(n: number, label = 'number'): void {
        if (typeof n !== 'number' || !Number.isFinite(n)) {
            throw new TypeError(`Expected finite ${label}, got ${n}`);
        }
    }
    static add(a: number, b: number): number { this._ensureFinite(a, 'a'); this._ensureFinite(b, 'b'); return a + b; }
    static subtract(a: number, b: number): number { this._ensureFinite(a, 'a'); this._ensureFinite(b, 'b'); return a - b; }
    static multiply(a: number, b: number): number { this._ensureFinite(a, 'a'); this._ensureFinite(b, 'b'); return a * b; }
    static divide(a: number, b: number): number {
        this._ensureFinite(a, 'a'); this._ensureFinite(b, 'b');
        if (b === 0) throw new RangeError('Division by zero');
        return a / b;
    }
    static _ensureVector(v: number[], len: number | null = null, label = 'vector'): void {
        if (!Array.isArray(v) || v.some(x => typeof x !== 'number' || !Number.isFinite(x))) {
            throw new TypeError(`Expected numeric ${label}`);
        }
        if (len != null && v.length !== len) {
            throw new RangeError(`Expected ${label} length ${len}, got ${v.length}`);
        }
    }
    static dotProduct(v1: number[], v2: number[]): number {
        this._ensureVector(v1, null, 'v1'); this._ensureVector(v2, null, 'v2');
        if (v1.length !== v2.length) throw new RangeError('Dot product requires equal-length vectors');
        return v1.reduce((sum, val, i) => sum + val * v2[i], 0);
    }
    static crossProduct(a: number[], b: number[]): number[] {
        this._ensureVector(a, 3, 'a (3D)'); this._ensureVector(b, 3, 'b (3D)');
        const [x1, y1, z1] = a, [x2, y2, z2] = b;
        return [y1 * z2 - z1 * y2, z1 * x2 - x1 * z2, x1 * y2 - y1 * x2];
    }
    static normalize(v: number[]): number[] {
        this._ensureVector(v, null, 'vector');
        const mag = Math.hypot(...v);
        return mag === 0 ? [...v] : v.map(x => x / mag);
    }
}

/**
 * MatrixCore: Matrix operations with validation
 */
class MatrixCore {
    static _isRect(m: number[][]): boolean {
        return Array.isArray(m) && m.length > 0 && m.every(r => Array.isArray(r) && r.length === m[0].length);
    }
    static _ensureMatrix(m: number[][], label = 'matrix'): void {
        if (!this._isRect(m) || m.some(r => r.some(x => typeof x !== 'number' || !Number.isFinite(x)))) {
            throw new TypeError(`Expected numeric rectangular ${label}`);
        }
    }
    static shape(m: number[][]): [number, number] { this._ensureMatrix(m); return [m.length, m[0].length]; }
    static transpose(m: number[][]): number[][] {
        this._ensureMatrix(m);
        const [r, c] = this.shape(m);
        const out = Array.from({ length: c }, () => Array(r));
        for (let i = 0; i < r; i++) for (let j = 0; j < c; j++) out[j][i] = m[i][j];
        return out;
    }
    static identity(n: number): number[][] {
        if (!Number.isInteger(n) || n <= 0) throw new RangeError('identity(n) needs positive integer n');
        return Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => i === j ? 1 : 0));
    }
    static matMul(A: number[][], B: number[][]): number[][] {
        this._ensureMatrix(A, 'A'); this._ensureMatrix(B, 'B');
        const [rA, cA] = this.shape(A); const [rB, cB] = this.shape(B);
        if (cA !== rB) throw new RangeError(`matMul dimension mismatch: ${rA}x${cA} · ${rB}x${cB}`);
        const out = Array.from({ length: rA }, () => Array(cB).fill(0));
        for (let i = 0; i < rA; i++) {
            for (let k = 0; k < cA; k++) {
                const aik = A[i][k];
                for (let j = 0; j < cB; j++) out[i][j] += aik * B[k][j];
            }
        }
        return out;
    }
    static det2([[a, b], [c, d]]: [[number, number], [number, number]]): number { return a * d - b * c; }
    static inv2(M: number[][]): number[][] {
        this._ensureMatrix(M, '2x2'); const [r, c] = this.shape(M);
        if (r !== 2 || c !== 2) throw new RangeError('inv2 expects 2x2');
        const [[a, b], [c_, d]] = M;
        const det = this.det2([[a, b], [c_, d]]);
        if (det === 0) throw new RangeError('Matrix is singular (2x2)');
        const invDet = 1 / det;
        return [[d * invDet, -b * invDet], [-c_ * invDet, a * invDet]];
    }
    static det3([[a, b, c], [d, e, f], [g, h, i]]: [[number, number, number], [number, number, number], [number, number, number]]): number {
        return a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
    }
    static inv3(M: number[][]): number[][] {
        this._ensureMatrix(M, '3x3'); const [r, c] = this.shape(M);
        if (r !== 3 || c !== 3) throw new RangeError('inv3 expects 3x3');
        const [[a, b, c1], [d, e, f], [g, h, i]] = M;
        const det = this.det3([[a, b, c1], [d, e, f], [g, h, i]]);
        if (det === 0) throw new RangeError('Matrix is singular (3x3)');
        const A = [
            [(e * i - f * h), -(b * i - c1 * h), (b * f - c1 * e)],
            [-(d * i - f * g), (a * i - c1 * g), -(a * f - c1 * d)],
            [(d * h - e * g), -(a * h - b * g), (a * e - b * d)]
        ];
        // inverse = adj(M) / det = transpose(cofactor matrix)/det; A above is cofactor^T already.
        const invDet = 1 / det;
        for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++) A[r][c] *= invDet;
        return A;
    }
}

/**
 * MathParser: Safe mathematical expression evaluation
 */
class MathParser {
    static readonly #ALLOWED_MATH = Object.freeze({
        PI: Math.PI, E: Math.E, LN2: Math.LN2, LN10: Math.LN10,
        LOG2E: Math.LOG2E, LOG10E: Math.LOG10E, SQRT1_2: Math.SQRT1_2, SQRT2: Math.SQRT2,
        abs: Math.abs, acos: Math.acos, acosh: Math.acosh, asin: Math.asin, asinh: Math.asinh,
        atan: Math.atan, atan2: Math.atan2, atanh: Math.atanh, cbrt: Math.cbrt, ceil: Math.ceil,
        cos: Math.cos, cosh: Math.cosh, exp: Math.exp, expm1: Math.expm1, floor: Math.floor,
        fround: Math.fround, hypot: Math.hypot, log: Math.log, log10: Math.log10, log1p: Math.log1p,
        log2: Math.log2, max: Math.max, min: Math.min, pow: Math.pow, round: Math.round, sign: Math.sign,
        sin: Math.sin, sinh: Math.sinh, sqrt: Math.sqrt, tan: Math.tan, tanh: Math.tanh, trunc: Math.trunc
    });

    static parseExpression(expr: string, vars: Record<string, number> = {}): number {
        if (typeof expr !== 'string') throw new TypeError('Expression must be a string');
        if (vars == null || typeof vars !== 'object') throw new TypeError('vars must be an object');
        // Normalize "^" to JS exponentiation
        const sanitized = expr.replace(/\^/g, '**');

        // Build param list: whitelist math + user vars (checked names)
        const mathNames = Object.keys(this.#ALLOWED_MATH);
        const varNames = Object.keys(vars);
        // Validate variable names (simple identifier rule)
        for (const name of varNames) {
            if (!/^[A-Za-z_]\w*$/.test(name)) throw new SyntaxError(`Illegal variable name "${name}"`);
            const v = vars[name];
            if (typeof v !== 'number' || !Number.isFinite(v)) throw new TypeError(`Variable ${name} must be a finite number`);
        }

        // Ensure only allowed identifiers appear (math names + var names)
        const identifiers = sanitized.match(/[A-Za-z_]\w*/g) || [];
        const allowed = new Set([...mathNames, ...varNames]);
        for (const id of identifiers) {
            if (!allowed.has(id)) throw new SyntaxError(`Identifier "${id}" is not allowed`);
        }

        const paramList = [...mathNames, ...varNames].join(',');
        const argValues = [...mathNames.map(n => (this.#ALLOWED_MATH as any)[n]), ...varNames.map(n => vars[n])];

        // eslint-disable-next-line no-new-func
        const fn = new Function(paramList, `"use strict"; return (${sanitized});`);
        const result = fn(...argValues);
        if (typeof result !== 'number' || !Number.isFinite(result)) {
            throw new Error('Expression did not evaluate to a finite number');
        }
        return result;
    }

    static parseVector(input: string | number[]): number[] {
        if (Array.isArray(input)) {
            const v = input.map(Number);
            if (v.some(n => !Number.isFinite(n))) throw new TypeError('Vector contains non-finite values');
            return v;
        }
        if (typeof input !== 'string') throw new TypeError('Vector input must be string or array');
        const cleaned = input.trim().replace(/^\[|\]$/g, '').replace(/;/g, ',').replace(/\s+/g, ',');
        const parts = cleaned.split(',').filter(Boolean);
        const v = parts.map(Number);
        if (!v.length || v.some(n => !Number.isFinite(n))) throw new TypeError(`Could not parse vector from "${input}"`);
        return v;
    }
}

/**
 * CalculationEngine: Unified mathematical computation registry
 */
export class CalculationEngine {
    public core: typeof MathCore;
    public mcore: typeof MatrixCore;
    public parser: typeof MathParser;
    public ops: Map<string, (...args: any[]) => any>;

    constructor(core = MathCore, parser = MathParser, mcore = MatrixCore) {
        this.core = core;
        this.mcore = mcore;
        this.parser = parser;
        this.ops = new Map([
            // scalars/vectors
            ['add', (a: number, b: number) => this.core.add(a, b)],
            ['subtract', (a: number, b: number) => this.core.subtract(a, b)],
            ['multiply', (a: number, b: number) => this.core.multiply(a, b)],
            ['divide', (a: number, b: number) => this.core.divide(a, b)],
            ['dotProduct', (v1: number[], v2: number[]) => this.core.dotProduct(v1, v2)],
            ['crossProduct', (v1: number[], v2: number[]) => this.core.crossProduct(v1, v2)],
            ['normalize', (v: number[]) => this.core.normalize(v)],
            // matrices
            ['transpose', (M: number[][]) => this.mcore.transpose(M)],
            ['matMul', (A: number[][], B: number[][]) => this.mcore.matMul(A, B)],
            ['identity', (n: number) => this.mcore.identity(n)],
            ['inv2', (M: number[][]) => this.mcore.inv2(M)],
            ['inv3', (M: number[][]) => this.mcore.inv3(M)]
        ]);
    }

    register(name: string, fn: (...args: any[]) => any): void {
        if (typeof name !== 'string' || typeof fn !== 'function') throw new TypeError('register(name, fn) expects (string, function)');
        if (this.ops.has(name)) throw new Error(`Operation "${name}" already exists`);
        this.ops.set(name, fn);
    }

    compute(operation: string, ...operands: any[]): any {
        const op = this.ops.get(operation);
        if (!op) throw new Error(`Unknown operation "${operation}"`);
        return op(...operands);
    }

    evaluate(expression: string, vars?: Record<string, number>): number {
        return this.parser.parseExpression(expression, vars);
    }

    parseVector(input: string | number[]): number[] {
        return this.parser.parseVector(input);
    }
}

// Export the calculation engine instance for convenience
export const mathEngine = new CalculationEngine();

// =============================================================================
// Legacy Mathematical Utilities (for compatibility)
// =============================================================================

/**
 * Clamp a value between min and max
 */
export function clamp(value: number, min: number, max: number): number {
    return Math.max(min, Math.min(max, value));
}

/**
 * Linear interpolation between two values
 */
export function lerp(a: number, b: number, t: number): number {
    return a + (b - a) * clamp(t, 0, 1);
}

/**
 * Smooth step interpolation (S-curve)
 */
export function smoothstep(edge0: number, edge1: number, x: number): number {
    const t = clamp((x - edge0) / (edge1 - edge0), 0, 1);
    return t * t * (3 - 2 * t);
}

/**
 * Map value from one range to another
 */
export function mapRange(value: number, inMin: number, inMax: number, outMin: number, outMax: number): number {
    return outMin + (value - inMin) * (outMax - outMin) / (inMax - inMin);
}

export function inverseLerp(a: number, b: number, value: number): number {
    return (value - a) / (b - a);
}

export function map(value: number, fromMin: number, fromMax: number, toMin: number, toMax: number): number {
    return lerp(toMin, toMax, inverseLerp(fromMin, fromMax, value));
}

/**
 * Degrees to radians conversion
 */
export function degToRad(degrees: number): number {
    return degrees * Math.PI / 180;
}

/**
 * Radians to degrees conversion
 */
export function radToDeg(radians: number): number {
    return radians * 180 / Math.PI;
}

// =============================================================================
// Random Number Generation
// =============================================================================

/**
 * Seeded random number generator (LCG)
 */
export function createSeededRandom(seed: number) {
    let state = seed >>> 0;
    return () => (state = (state * 1664525 + 1013904223) >>> 0) / 2 ** 32;
}

/**
 * Generate Gaussian-distributed random number (Box-Muller)
 */
export function gaussianRandom(mean: number = 0, stdDev: number = 1, rand: () => number = Math.random): number {
    const u = 1 - rand(); // Avoid log(0)
    const v = rand();
    const z = Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
    return mean + z * stdDev;
}

/**
 * Random point on unit sphere
 */
export function randomUnitSphere(rand: () => number = Math.random): Vector3D {
    const u = rand();
    const v = rand();
    const theta = 2 * Math.PI * u;
    const phi = Math.acos(2 * v - 1);

    return {
        x: Math.sin(phi) * Math.cos(theta),
        y: Math.sin(phi) * Math.sin(theta),
        z: Math.cos(phi)
    };
}

// =============================================================================
// Vector Operations
// =============================================================================

/**
 * 3D vector operations
 */
export const Vec3 = {
    create: (x: number = 0, y: number = 0, z: number = 0): Vector3D => ({ x, y, z }),

    add: (a: Vector3D, b: Vector3D): Vector3D => ({
        x: a.x + b.x,
        y: a.y + b.y,
        z: a.z + b.z
    }),

    subtract: (a: Vector3D, b: Vector3D): Vector3D => ({
        x: a.x - b.x,
        y: a.y - b.y,
        z: a.z - b.z
    }),

    multiply: (v: Vector3D, scalar: number): Vector3D => ({
        x: v.x * scalar,
        y: v.y * scalar,
        z: v.z * scalar
    }),

    dot: (a: Vector3D, b: Vector3D): number => a.x * b.x + a.y * b.y + a.z * b.z,

    cross: (a: Vector3D, b: Vector3D): Vector3D => ({
        x: a.y * b.z - a.z * b.y,
        y: a.z * b.x - a.x * b.z,
        z: a.x * b.y - a.y * b.x
    }),

    length: (v: Vector3D): number => Math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z),

    normalize: (v: Vector3D): Vector3D => {
        const len = Vec3.length(v);
        return len > 0 ? Vec3.multiply(v, 1 / len) : { x: 0, y: 0, z: 0 };
    },

    distance: (a: Vector3D, b: Vector3D): number => Vec3.length(Vec3.subtract(a, b))
};

/**
 * 4D vector operations
 */
export const Vec4 = {
    create: (x: number = 0, y: number = 0, z: number = 0, w: number = 0): Vector4D => ({ x, y, z, w }),

    add: (a: Vector4D, b: Vector4D): Vector4D => ({
        x: a.x + b.x,
        y: a.y + b.y,
        z: a.z + b.z,
        w: a.w + b.w
    }),

    multiply: (v: Vector4D, scalar: number): Vector4D => ({
        x: v.x * scalar,
        y: v.y * scalar,
        z: v.z * scalar,
        w: v.w * scalar
    }),

    length: (v: Vector4D): number => Math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w)
};

// =============================================================================
// Image Processing Utilities
// =============================================================================

/**
 * Convert image file to ImageData
 */
export async function fileToImageData(file: File, width: number, height: number): Promise<ImageData> {
    return new Promise((resolve, reject) => {
        const img = new Image();
        img.onload = () => {
            const canvas = document.createElement('canvas');
            canvas.width = width;
            canvas.height = height;
            const ctx = canvas.getContext('2d')!;

            // Preserve aspect ratio (cover-fit)
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

/**
 * Convert ImageData to binary mask based on luminance threshold
 */
export function imageDataToMask(imageData: ImageData, threshold: number = 0.5): ImageMask {
    const { data, width, height } = imageData;
    const mask = new Uint8Array(width * height);

    for (let i = 0, p = 0; i < data.length; i += 4, p++) {
        const r = data[i];
        const g = data[i + 1];
        const b = data[i + 2];
        // Standard luminance calculation
        const luminance = (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255;
        mask[p] = luminance >= threshold ? 1 : 0;
    }

    return { data: mask, width, height };
}

// =============================================================================
// Geometry Processing
// =============================================================================

/**
 * Calculate bounding box from vertex array
 */
export function calculateBoundingBox(vertices: Float32Array): BoundingBox {
    if (vertices.length === 0) {
        return { min: { x: 0, y: 0, z: 0 }, max: { x: 0, y: 0, z: 0 } };
    }

    let minX = Infinity, minY = Infinity, minZ = Infinity;
    let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;

    for (let i = 0; i < vertices.length; i += 3) {
        const x = vertices[i];
        const y = vertices[i + 1];
        const z = vertices[i + 2];

        minX = Math.min(minX, x);
        minY = Math.min(minY, y);
        minZ = Math.min(minZ, z);
        maxX = Math.max(maxX, x);
        maxY = Math.max(maxY, y);
        maxZ = Math.max(maxZ, z);
    }

    return {
        min: { x: minX, y: minY, z: minZ },
        max: { x: maxX, y: maxY, z: maxZ }
    };
}

/**
 * Generate vertex normals from triangle faces
 */
export function computeVertexNormals(vertices: Float32Array, indices: Uint32Array): Float32Array {
    const normals = new Float32Array(vertices.length);
    const counts = new Float32Array(vertices.length / 3);

    // Accumulate face normals for each vertex
    for (let i = 0; i < indices.length; i += 3) {
        const i1 = indices[i] * 3;
        const i2 = indices[i + 1] * 3;
        const i3 = indices[i + 2] * 3;

        const v1: Vector3D = { x: vertices[i1], y: vertices[i1 + 1], z: vertices[i1 + 2] };
        const v2: Vector3D = { x: vertices[i2], y: vertices[i2 + 1], z: vertices[i2 + 2] };
        const v3: Vector3D = { x: vertices[i3], y: vertices[i3 + 1], z: vertices[i3 + 2] };

        const edge1 = Vec3.subtract(v2, v1);
        const edge2 = Vec3.subtract(v3, v1);
        const faceNormal = Vec3.normalize(Vec3.cross(edge1, edge2));

        // Add to each vertex
        [i1, i2, i3].forEach(idx => {
            normals[idx] += faceNormal.x;
            normals[idx + 1] += faceNormal.y;
            normals[idx + 2] += faceNormal.z;
            counts[idx / 3]++;
        });
    }

    // Normalize accumulated normals
    for (let i = 0; i < normals.length; i += 3) {
        const count = counts[i / 3];
        if (count > 0) {
            const normal = Vec3.normalize({
                x: normals[i] / count,
                y: normals[i + 1] / count,
                z: normals[i + 2] / count
            });
            normals[i] = normal.x;
            normals[i + 1] = normal.y;
            normals[i + 2] = normal.z;
        }
    }

    return normals;
}

// =============================================================================
// Performance and Optimization
// =============================================================================

/**
 * Debounce function calls
 */
export function debounce<T extends (...args: any[]) => any>(
    func: T,
    delay: number
): (...args: Parameters<T>) => void {
    let timeoutId: NodeJS.Timeout;
    return (...args: Parameters<T>) => {
        clearTimeout(timeoutId);
        timeoutId = setTimeout(() => func(...args), delay);
    };
}

/**
 * Throttle function calls
 */
export function throttle<T extends (...args: any[]) => any>(
    func: T,
    delay: number
): (...args: Parameters<T>) => void {
    let lastCall = 0;
    return (...args: Parameters<T>) => {
        const now = Date.now();
        if (now - lastCall >= delay) {
            lastCall = now;
            func(...args);
        }
    };
}

/**
 * Simple object pool for reusing objects
 */
export class ObjectPool<T> {
    private pool: T[] = [];

    constructor(private factory: () => T, private reset?: (obj: T) => void) { }

    acquire(): T {
        if (this.pool.length > 0) {
            const obj = this.pool.pop()!;
            if (this.reset) this.reset(obj);
            return obj;
        }
        return this.factory();
    }

    release(obj: T): void {
        this.pool.push(obj);
    }

    clear(): void {
        this.pool.length = 0;
    }
}

// =============================================================================
// Format and Export Utilities
// =============================================================================

/**
 * Format number with appropriate precision
 */
export function formatNumber(value: number, precision: number = 3): string {
    if (Math.abs(value) < 0.001) {
        return value.toExponential(precision);
    }
    return value.toFixed(precision);
}

/**
 * Download data as file
 */
export function downloadFile(data: string | Uint8Array, filename: string, mimeType: string = 'text/plain'): void {
    const blob = new Blob([data], { type: mimeType });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
}

/**
 * Copy text to clipboard
 */
export async function copyToClipboard(text: string): Promise<boolean> {
    try {
        await navigator.clipboard.writeText(text);
        return true;
    } catch (err) {
        console.warn('Failed to copy to clipboard:', err);
        return false;
    }
}
