import GraphicsMiniLabs from './GraphicsMiniLabs';

export default GraphicsMiniLabs;

// Export individual mini-lab components for advanced usage
export {
    BarycentricTriangle,
    CatmullRomCurves,
    TexturePaintTriangle
} from './GraphicsMiniLabs';

// Export mathematical utility functions
export const MathUtils = {
    // Vector operations
    add: (a: [number, number, number], b: [number, number, number]): [number, number, number] =>
        [a[0] + b[0], a[1] + b[1], a[2] + b[2]],

    sub: (a: [number, number, number], b: [number, number, number]): [number, number, number] =>
        [a[0] - b[0], a[1] - b[1], a[2] - b[2]],

    mul: (a: [number, number, number], s: number): [number, number, number] =>
        [a[0] * s, a[1] * s, a[2] * s],

    len: (a: [number, number, number]) => Math.hypot(a[0], a[1], a[2]),

    cross: (a: [number, number, number], b: [number, number, number]): [number, number, number] => [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0]
    ],

    // Barycentric coordinate calculation
    baryWeights: (A: [number, number, number], B: [number, number, number], C: [number, number, number], P: [number, number, number]): [number, number, number] => {
        const { add, sub, cross, len } = MathUtils;
        const areaABC = 0.5 * len(cross(sub(B, A), sub(C, A)));
        const w0 = 0.5 * len(cross(sub(B, P), sub(C, P))) / areaABC;
        const w1 = 0.5 * len(cross(sub(A, P), sub(C, P))) / areaABC;
        const w2 = 0.5 * len(cross(sub(A, P), sub(B, P))) / areaABC;
        return [w0, w1, w2];
    },

    // Parameterization for curves
    parametrize: (points: [number, number, number][], alpha: number): number[] => {
        const { sub, len } = MathUtils;
        const t = [0];
        for (let i = 1; i < points.length; i++) {
            const d = len(sub(points[i], points[i - 1]));
            t.push(t[i - 1] + Math.pow(d, alpha));
        }
        return t;
    },

    // Catmull-Rom spline evaluation
    catmullRom: (Pm1: [number, number, number], P0: [number, number, number], P1: [number, number, number], P2: [number, number, number], u: number, t: number[], i: number): [number, number, number] => {
        const { add, sub, mul } = MathUtils;
        const t0 = t[i - 1], t1 = t[i], t2 = t[i + 1], t3 = t[i + 2];

        // Local u maps [t1,t2] → [0,1]
        const u01 = (1 - u) * t1 + u * t2;

        // Tangents (finite difference with non-uniform spacing)
        const m1 = mul(sub(P1, Pm1), (t2 - t1) / (t1 - t0));
        const m2 = mul(sub(P2, P0), (t2 - t1) / (t2 - t0));

        // Hermite basis
        const s = (u01 - t1) / (t2 - t1);
        const h00 = 2 * s ** 3 - 3 * s ** 2 + 1;
        const h10 = s ** 3 - 2 * s ** 2 + s;
        const h01 = -2 * s ** 3 + 3 * s ** 2;
        const h11 = s ** 3 - s ** 2;

        return add(
            add(mul(P0, h00), mul(m1, h10 * (t2 - t1))),
            add(mul(P1, h01), mul(m2, h11 * (t2 - t1)))
        );
    }
};

// Export parameterization types
export const ParametrizationTypes = {
    UNIFORM: 0,
    CENTRIPETAL: 0.5,
    CHORDAL: 1
} as const;

// Export educational constants
export const EducationalConstants = {
    BARYCENTRIC_VERTICES: {
        A: { position: [-1, -0.5, 0], color: [1, 0, 0], coords: [1, 0, 0] },
        B: { position: [1, -0.5, 0], color: [0, 1, 0], coords: [0, 1, 0] },
        C: { position: [0, 1.0, 0], color: [0, 0, 1], coords: [0, 0, 1] }
    },

    UV_MAPPING: {
        A: [0, 0],
        B: [1, 0],
        C: [0, 1]
    },

    CURVE_TYPES: {
        uniform: { alpha: 0, description: 'Equal spacing, may overshoot' },
        centripetal: { alpha: 0.5, description: 'Best balance, no loops' },
        chordal: { alpha: 1, description: 'Distance-based, more tension' }
    }
} as const;

// Type definitions for external usage
export type Vector3 = [number, number, number];
export type BarycentricCoords = [number, number, number];
export type UVCoords = [number, number];

export interface MiniLabConfig {
    showLabels?: boolean;
    showHelp?: boolean;
    interactive?: boolean;
    educationalMode?: boolean;
}

export interface CurveConfig extends MiniLabConfig {
    parameterization?: 'uniform' | 'centripetal' | 'chordal';
    resolution?: number;
    showControlPoints?: boolean;
    showTangents?: boolean;
}

export interface PaintConfig extends MiniLabConfig {
    brushSize?: number;
    brushColor?: string;
    showUVGrid?: boolean;
    canvasSize?: number;
}
