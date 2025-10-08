export type Vec = number[];         // column vector
export type Mat = number[][];       // row-major

export function zeros(n: number): Vec {
    return Array(n).fill(0);
}

export function eye(n: number): Mat {
    return Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => i === j ? 1 : 0));
}

export function cloneM(A: Mat): Mat {
    return A.map(r => r.slice());
}

export function T(A: Mat): Mat {
    const m = A.length, n = A[0].length;
    const R: Array<Vec> = Array.from({ length: n }, () => Array(m));
    for (let i = 0; i < m; i++) for (let j = 0; j < n; j++) R[j][i] = A[i][j];
    return R;
}

export function mm(A: Mat, B: Mat): Mat {
    const m = A.length, n = B[0].length, k = B.length;
    const R: Mat = Array.from({ length: m }, () => Array(n).fill(0));
    for (let i = 0; i < m; i++) for (let j = 0; j < n; j++) {
        let s = 0;
        for (let t = 0; t < k; t++) s += A[i][t] * B[t][j];
        R[i][j] = s;
    }
    return R;
}

export function mv(A: Mat, x: Vec): Vec {
    const m = A.length, k = x.length;
    const r = Array(m).fill(0);
    for (let i = 0; i < m; i++) {
        let s = 0;
        for (let j = 0; j < k; j++) s += A[i][j] * x[j];
        r[i] = s;
    }
    return r;
}

export function vvAdd(a: Vec, b: Vec): Vec {
    return a.map((v, i) => v + b[i]);
}

export function mmAdd(A: Mat, B: Mat): Mat {
    return A.map((r, i) => r.map((v, j) => v + B[i][j]));
}

export function diag(vals: number[]): Mat {
    const n = vals.length;
    const M = eye(n);
    for (let i = 0; i < n; i++) M[i][i] = vals[i];
    return M;
}

export function proj(indices: number[], dim: number): Mat {
    // projection keeping given indices
    const P: Mat = Array.from({ length: dim }, () => Array(dim).fill(0));
    for (const i of indices) P[i][i] = 1;
    return P;
}

export function rot2D(i: number, j: number, theta: number, dim: number): Mat {
    const R = eye(dim);
    const c = Math.cos(theta), s = Math.sin(theta);
    R[i][i] = c; R[j][j] = c; R[i][j] = -s; R[j][i] = s;
    return R;
}
