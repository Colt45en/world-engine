/*
 *  Math Pro — numerically safer math core for your meta-fractal engine
 * - Strict shape checks
 * - Cached-efficient mat·vec/mat·mat multiply
 * - Cholesky solve, Gaussian elimination fallback
 * - Moore–Penrose pseudoinverse (regularized)
 * - PSD utilities (ensureSPD), Mahalanobis distance
 * - Stable softmax, norms, lerp
 * - Random Gaussian sampler (Box–Muller + Cholesky)
 *
 * Drop-in: import { LLEMath, Morpheme, Button, StationaryUnit, ScalingOperations, ButtonFactory } from './lle-math-pro.js'
 */

class LLEMath {
    // ---------- Type guards & safety ----------
    static _isMatrix(A) { return Array.isArray(A) && Array.isArray(A[0]); }
    static _isVector(v) { return Array.isArray(v) && !Array.isArray(v[0]); }
    static _shape(A) { return this._isMatrix(A) ? [A.length, A[0].length] : [A.length]; }
    static _assertMatrix(A, name = 'A') { if (!this._isMatrix(A)) throw new Error(`${name} must be matrix`); const r = A.length, c = A[0].length; for (const row of A) { if (!Array.isArray(row) || row.length !== c) throw new Error(`${name} rows must have equal length`); } return [r, c]; }
    static _assertVector(v, name = 'v') { if (!this._isVector(v)) throw new Error(`${name} must be vector`); return v.length; }
    static validateFinite(vecOrMat, name = 'value') {
        if (this._isMatrix(vecOrMat)) {
            for (const row of vecOrMat) for (const x of row) if (!Number.isFinite(x)) throw new Error(`${name} contains non-finite`);
        } else if (this._isVector(vecOrMat)) {
            for (const x of vecOrMat) if (!Number.isFinite(x)) throw new Error(`${name} contains non-finite`);
        } else throw new Error(`${name} must be vector or matrix`);
        return true;
    }

    // ---------- Core ops ----------
    static multiply(A, B) {
        const [r, k] = this._assertMatrix(A, 'A');
        if (this._isMatrix(B)) {
            const [k2, c] = this._assertMatrix(B, 'B');
            if (k !== k2) throw new Error(`multiply: shape mismatch (${r}×${k})·(${k2}×${c})`);
            const out = Array.from({ length: r }, () => Array(c).fill(0));
            for (let i = 0; i < r; i++) {
                for (let t = 0; t < k; t++) {
                    const a = A[i][t];
                    for (let j = 0; j < c; j++) out[i][j] += a * B[t][j];
                }
            }
            return out;
        }
        const n = this._assertVector(B, 'B'); if (k !== n) throw new Error(`multiply: shape mismatch (${r}×${k})·(${n})`);
        const vOut = Array(r).fill(0); for (let i = 0; i < r; i++) { let s = 0; for (let j = 0; j < k; j++) s += A[i][j] * B[j]; vOut[i] = s; } return vOut;
    }
    static transpose(A) { this._assertMatrix(A, 'A'); const r = A.length, c = A[0].length; const T = Array.from({ length: c }, (_, j) => Array(r).fill(0)); for (let i = 0; i < r; i++) for (let j = 0; j < c; j++) T[j][i] = A[i][j]; return T; }
    static identity(n) { return Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => i === j ? 1 : 0)); }
    static diagonal(vals) { const n = vals.length; const M = this.identity(n); for (let i = 0; i < n; i++) M[i][i] = vals[i]; return M; }
    static vectorAdd(a, b) { const n = this._assertVector(a, 'a'); const m = this._assertVector(b, 'b'); if (n !== m) throw new Error('vectorAdd: length mismatch'); const out = new Array(n); for (let i = 0; i < n; i++) out[i] = a[i] + b[i]; return out; }
    static vectorSub(a, b) { const n = this._assertVector(a, 'a'); if (b.length !== n) throw new Error('vectorSub: length mismatch'); const out = new Array(n); for (let i = 0; i < n; i++) out[i] = a[i] - b[i]; return out; }
    static vectorScale(v, s) { const n = this._assertVector(v, 'v'); const out = new Array(n); for (let i = 0; i < n; i++) out[i] = v[i] * s; return out; }
    static hadamard(A, B) { const [r, c] = this._assertMatrix(A, 'A'); const [r2, c2] = this._assertMatrix(B, 'B'); if (r !== r2 || c !== c2) throw new Error('hadamard: shape mismatch'); const out = Array.from({ length: r }, () => Array(c).fill(0)); for (let i = 0; i < r; i++) for (let j = 0; j < c; j++) out[i][j] = A[i][j] * B[i][j]; return out; }
    static clamp(v, min, max) { return Math.max(min, Math.min(max, v)); }

    // ---------- Norms & numerics ----------
    static dot(a, b) { const n = this._assertVector(a, 'a'); if (b.length !== n) throw new Error('dot: length mismatch'); let s = 0; for (let i = 0; i < n; i++) s += a[i] * b[i]; return s; }
    static norm2(v) { return Math.sqrt(this.dot(v, v)); }
    static normalize(v, eps = 1e-12) { const nrm = this.norm2(v); return nrm < eps ? v.slice() : v.map(x => x / nrm); }
    static softmax(v) { const m = Math.max(...v); const ex = v.map(x => Math.exp(x - m)); const s = ex.reduce((a, b) => a + b, 0) || 1; return ex.map(x => x / s); }

    // ---------- Decompositions & solves ----------
    static cholesky(S) { const [n, n2] = this._assertMatrix(S, 'S'); if (n !== n2) throw new Error('cholesky: S must be square'); const L = Array.from({ length: n }, () => Array(n).fill(0)); for (let i = 0; i < n; i++) { for (let j = 0; j <= i; j++) { let sum = S[i][j]; for (let k = 0; k < j; k++) sum -= L[i][k] * L[j][k]; if (i === j) { if (sum <= 0) throw new Error('cholesky: not SPD'); L[i][j] = Math.sqrt(sum); } else { L[i][j] = sum / L[j][j]; } } } return L; }
    static solveCholesky(S, b) { const L = this.cholesky(S); const n = L.length; const y = new Array(n).fill(0); for (let i = 0; i < n; i++) { let s = b[i]; for (let k = 0; k < i; k++) s -= L[i][k] * y[k]; y[i] = s / L[i][i]; } const x = new Array(n).fill(0); for (let i = n - 1; i >= 0; i--) { let s = y[i]; for (let k = i + 1; k < n; k++) s -= L[k][i] * x[k]; x[i] = s / L[i][i]; } return x; }
    static gaussianElimSolve(A, b) { const [r, c] = this._assertMatrix(A, 'A'); if (r !== c) throw new Error('gaussianElimSolve: A must be square'); if (b.length !== r) throw new Error('gaussianElimSolve: b length mismatch'); const M = A.map(row => row.slice()); const v = b.slice(); for (let i = 0; i < r; i++) { let p = i; for (let k = i + 1; k < r; k++) if (Math.abs(M[k][i]) > Math.abs(M[p][i])) p = k;[M[i], M[p]] = [M[p], M[i]];[v[i], v[p]] = [v[p], v[i]]; const piv = M[i][i]; if (Math.abs(piv) < 1e-12) throw new Error('gaussianElimSolve: singular'); for (let j = i; j < r; j++) M[i][j] /= piv; v[i] /= piv; for (let k = 0; k < r; k++) { if (k === i) continue; const f = M[k][i]; for (let j = i; j < r; j++) M[k][j] -= f * M[i][j]; v[k] -= f * v[i]; } } return v; }
    static inverseSmall(M) { const [n, n2] = this._assertMatrix(M, 'M'); if (n !== n2) throw new Error('inverseSmall: square only'); const A = M.map((row, i) => [...row, ...this.identity(n)[i]]); for (let i = 0; i < n; i++) { let p = i; for (let r = i + 1; r < n; r++) if (Math.abs(A[r][i]) > Math.abs(A[p][i])) p = r;[A[i], A[p]] = [A[p], A[i]]; let piv = A[i][i]; if (Math.abs(piv) < 1e-12) throw new Error('inverseSmall: singular'); const inv = 1 / piv; for (let j = 0; j < 2 * n; j++) A[i][j] *= inv; for (let r = 0; r < n; r++) { if (r === i) continue; const f = A[r][i]; for (let j = 0; j < 2 * n; j++) A[r][j] -= f * A[i][j]; } } return A.map(row => row.slice(n)); }
    static pseudoInverse(A, lambda = 1e-6) { const AT = this.transpose(A); const m = A.length, n = A[0].length; if (m >= n) { const AAT = this.multiply(A, AT); const reg = AAT.map((row, i) => row.map((v, j) => v + (i === j ? lambda : 0))); const inv = this.inverseSmall(reg); return this.multiply(AT, inv); } else { const ATA = this.multiply(AT, A); const reg = ATA.map((row, i) => row.map((v, j) => v + (i === j ? lambda : 0))); const inv = this.inverseSmall(reg); return this.multiply(inv, AT); } }

    // ---------- Statistics & geometry ----------
    static mahalanobis(x, mu, Sigma) { const v = this.vectorSub(x, mu); const inv = this.inverseSmall(Sigma); const tmp = this.multiply(inv, v); return Math.sqrt(this.dot(v, tmp)); }
    static ensureSPD(S, jitter = 1e-6, maxTries = 6) { let J = jitter; for (let t = 0; t < maxTries; t++) { try { this.cholesky(S); return S; } catch { const n = S.length; for (let i = 0; i < n; i++) S[i][i] += J; J *= 10; } } return S; }
    static lerp(a, b, t) { const n = this._assertVector(a, 'a'); if (b.length !== n) throw new Error('lerp: length mismatch'); const out = new Array(n); for (let i = 0; i < n; i++) out[i] = a[i] * (1 - t) + b[i] * t; return out; }
    static rotationMatrix2D(theta) { const c = Math.cos(theta), s = Math.sin(theta); return [[c, -s, 0], [s, c, 0], [0, 0, 1]]; }
    static projectionMatrix(dims, keep) { const P = Array.from({ length: keep.length }, () => Array(dims).fill(0)); keep.forEach((k, i) => { if (k >= dims) throw new Error('projection: OOB'); P[i][k] = 1; }); return P; }

    // ---------- Randoms ----------
    static randn() { let u = 0, v = 0; while (u === 0) u = Math.random(); while (v === 0) v = Math.random(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); }
    static sampleGaussian(mean, Sigma) { const n = mean.length; this.ensureSPD(Sigma); const L = this.cholesky(Sigma); const z = Array.from({ length: n }, () => this.randn()); const y = new Array(n).fill(0); for (let i = 0; i < n; i++) { let s = 0; for (let k = 0; k <= i; k++) s += L[i][k] * z[k]; y[i] = s; } return this.vectorAdd(mean, y); }

    // ---------- Safe transform ----------
    static safeTransform(M, x, b = null) { this.validateFinite(x, 'input vector'); if (b) this.validateFinite(b, 'bias vector'); const y = this.multiply(M, x); if (b) { const f = this.vectorAdd(y, b); this.validateFinite(f, 'result vector'); return f; } this.validateFinite(y, 'result vector'); return y; }
}

/************************************
 * Morphemes & Operators
 ************************************/
class Morpheme {
    constructor(symbol, M, b, effects = {}) {
        this.symbol = symbol;
        this.M = M; // Linear transformation matrix
        this.b = b; // Bias vector
        this.effects = effects; // Additional effects (C, deltaLevel, alpha, beta)
    }

    static createBuiltInMorphemes(dim = 3) {
        const morphemes = new Map();

        // Prefix morphemes
        morphemes.set('re', new Morpheme('re',
            [[0.9, 0, 0], [0, 1.1, 0], [0, 0, 1]],
            [0.1, 0, 0],
            { deltaLevel: -1, alpha: 0.95, description: 'repetition, restoration' }
        ));

        morphemes.set('un', new Morpheme('un',
            [[-1, 0, 0], [0, 1, 0], [0, 0, 1]],
            [0, 0, 0],
            { alpha: 0.8, description: 'negation, reversal' }
        ));

        morphemes.set('counter', new Morpheme('counter',
            [[-0.8, 0.2, 0], [0.2, -0.8, 0], [0, 0, 1]],
            [0, 0, 0],
            { deltaLevel: 1, description: 'opposition, counteraction' }
        ));

        morphemes.set('multi', new Morpheme('multi',
            [[1.5, 0, 0], [0, 1.5, 0], [0, 0, 1.2]],
            [0, 0, 0.1],
            { deltaLevel: 1, description: 'multiplication, many' }
        ));

        // Suffix morphemes
        morphemes.set('ize', new Morpheme('ize',
            [[1, 0.1, 0], [0, 1, 0.1], [0, 0, 1.1]],
            [0, 0, 0],
            { deltaLevel: 1, description: 'make into, cause to become' }
        ));

        morphemes.set('ness', new Morpheme('ness',
            [[1, 0, 0], [0, 1, 0], [0.1, 0.1, 0.9]],
            [0, 0, 0.1],
            { C: [[1.1, 0, 0], [0, 1.1, 0], [0, 0, 0.9]], description: 'quality, state' }
        ));

        morphemes.set('ment', new Morpheme('ment',
            [[1.1, 0, 0], [0, 0.9, 0], [0, 0, 1]],
            [0, 0.1, 0],
            { deltaLevel: 1, description: 'result, action' }
        ));

        morphemes.set('ing', new Morpheme('ing',
            [[1, 0.2, 0], [0, 1.2, 0], [0, 0, 1]],
            [0, 0, 0],
            { alpha: 1.1, description: 'ongoing action' }
        ));

        // Root morphemes
        morphemes.set('build', new Morpheme('build',
            [[1.2, 0, 0], [0, 1.2, 0], [0, 0, 1.1]],
            [0.1, 0.1, 0],
            { deltaLevel: 1, description: 'construct, create' }
        ));

        morphemes.set('move', new Morpheme('move',
            [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            [0.2, 0, 0],
            { description: 'change position' }
        ));

        morphemes.set('scale', new Morpheme('scale',
            [[1.1, 0, 0], [0, 1.1, 0], [0, 0, 1.1]],
            [0, 0, 0],
            { description: 'change size' }
        ));

        return morphemes;
    }
}

class Button {
    constructor(label, abbr, wordClass, morphemes, options = {}, registry = null) {
        this.label = label;
        this.abbr = abbr;
        this.wordClass = wordClass;
        this.morphemes = morphemes;

        const baseDim = options.dimensions || 3;
        let M = LLEMath.identity(baseDim);
        let b = Array(baseDim).fill(0);
        let C = LLEMath.identity(baseDim);
        let alpha = 1.0;
        let beta = 0.0;
        let delta = 0;

        if (registry) {
            for (const sym of morphemes) {
                const m = registry.get(sym);
                if (!m) continue;
                const newM = LLEMath.multiply(m.M, M);
                const newB = LLEMath.vectorAdd(LLEMath.multiply(m.M, b), m.b);
                M = newM; b = newB;
                if (m.effects?.C) C = LLEMath.multiply(m.effects.C, C);
                if (m.effects?.deltaLevel) delta += m.effects.deltaLevel;
                if (typeof m.effects?.alpha === 'number') alpha *= m.effects.alpha;
                if (typeof m.effects?.beta === 'number') beta += m.effects.beta;
            }
        }

        // Overrides
        this.M = options.M || M;
        this.b = options.b || b;
        this.C = options.C || C;
        this.alpha = options.alpha ?? alpha;
        this.beta = options.beta ?? beta;
        this.deltaLevel = options.deltaLevel ?? delta;

        this.inputType = options.inputType || 'State';
        this.outputType = options.outputType || 'State';
        this.description = options.description || '';
    }

    apply(su) {
        const dim = su.d;
        if (this.M.length !== dim || this.M[0].length !== dim) throw new Error(`Button ${this.abbr}: M shape ${this.M.length}×${this.M[0].length} != ${dim}×${dim}`);
        if (this.b.length !== dim) throw new Error(`Button ${this.abbr}: b length ${this.b.length} != ${dim}`);
        if (this.C.length !== dim || this.C[0].length !== dim) throw new Error(`Button ${this.abbr}: C shape ${this.C.length}×${this.C[0].length} != ${dim}×${dim}`);

        const newSU = su.copy();
        newSU.x = LLEMath.safeTransform(this.M, su.x, this.b);

        // Covariance: Σ' = C Σ C^T (with SPD guard)
        let CS = LLEMath.multiply(this.C, su.Sigma);
        let SigmaNext = LLEMath.multiply(CS, LLEMath.transpose(this.C));
        SigmaNext = LLEMath.ensureSPD(SigmaNext);
        newSU.Sigma = SigmaNext;

        newSU.kappa = LLEMath.clamp((this.alpha * su.kappa) + this.beta, 0, 1);
        newSU.level = su.level + this.deltaLevel;
        LLEMath.validateFinite(newSU.x, `Button ${this.abbr} result`);
        return newSU;
    }

    canComposeWith(other) { return other.outputType === this.inputType; }
    toString() { return `[${this.abbr}] ${this.label} (${this.wordClass}, δℓ=${this.deltaLevel})`; }
    toJSON() { return { label: this.label, abbr: this.abbr, class: this.wordClass, morphemes: this.morphemes, delta_level: this.deltaLevel, M: this.M, b: this.b, C: this.C, alpha: this.alpha, beta: this.beta, description: this.description }; }
}

class StationaryUnit {
    constructor(dimensions = 3, x = null, Sigma = null, kappa = 1.0, level = 0) {
        this.d = dimensions;
        this.x = x || Array(dimensions).fill(0);
        this.Sigma = Sigma || LLEMath.identity(dimensions);
        this.kappa = kappa;
        this.level = level;
        this.timestamp = Date.now();
    }
    copy() { return new StationaryUnit(this.d, [...this.x], this.Sigma.map(r => [...r]), this.kappa, this.level); }
    toString() { const pos = this.x.map(v => v.toFixed(3)).join(','); return `SU(x=[${pos}], κ=${this.kappa.toFixed(3)}, ℓ=${this.level}, d=${this.d})`; }
}

class ScalingOperations {
    static createProjectionMatrix(fromDim, keepIndices) { const P = Array.from({ length: keepIndices.length }, () => Array(fromDim).fill(0)); keepIndices.forEach((idx, i) => { if (idx >= fromDim) throw new Error('projection: index out of bounds'); P[i][idx] = 1; }); return P; }

    static downscale(su, keepIndices = [0, 2]) {
        const newSU = su.copy();
        const P = this.createProjectionMatrix(su.d, keepIndices);
        newSU.x = LLEMath.multiply(P, su.x);
        const PS = LLEMath.multiply(P, su.Sigma);
        newSU.Sigma = LLEMath.multiply(PS, LLEMath.transpose(P));
        newSU.level = Math.max(0, su.level - 1);
        newSU.d = keepIndices.length;
        return newSU;
    }

    static upscale(su, abstractionMatrix = null, toDim = 3) {
        const newSU = su.copy();
        const A = abstractionMatrix || [[0.5, 0.5, 0], [0, 0.3, 0.7], [0.2, 0.2, 0.6]];
        const Aplus = LLEMath.pseudoInverse(A, 1e-6);
        const abstract = su.d === A.length ? su.x : LLEMath.multiply(A, su.x);
        const recon = LLEMath.multiply(Aplus, abstract);
        const xUp = recon.length < toDim ? [...recon, ...Array(toDim - recon.length).fill(0)] : recon.slice(0, toDim);
        newSU.x = xUp;
        const AplusT = LLEMath.transpose(Aplus);
        const AS = LLEMath.multiply(Aplus, su.Sigma);
        let SigmaUp = LLEMath.multiply(AS, AplusT);
        SigmaUp = LLEMath.ensureSPD(SigmaUp);

        if (SigmaUp.length < toDim) {
            const pad = LLEMath.identity(toDim);
            for (let i = 0; i < SigmaUp.length; i++) for (let j = 0; j < SigmaUp[0].length; j++) pad[i][j] = SigmaUp[i][j];
            SigmaUp = pad;
        }
        newSU.Sigma = SigmaUp;
        newSU.level = su.level + 1;
        newSU.d = toDim;
        return newSU;
    }
}

class ButtonFactory {
    static createStandardButtons(dim = 3) {
        const buttons = new Map();
        const morphemes = Morpheme.createBuiltInMorphemes(dim);
        const B = (label, abbr, wordClass, morphemeList, options = {}) =>
            new Button(label, abbr, wordClass, morphemeList, { ...options, dimensions: dim }, morphemes);

        buttons.set('RB', B('Rebuild', 'RB', 'Action', ['re', 'build'], { description: 'Recompose from parts (concretize)' }));
        buttons.set('UP', B('Upscale', 'UP', 'Action', ['multi'], { deltaLevel: 1, description: 'Scale up dimensions and complexity' }));
        buttons.set('CV', B('Convert', 'CV', 'Action', ['ize'], { description: 'Transform into different form' }));
        buttons.set('TL', B('Translucent', 'TL', 'Property', ['ness'], { deltaLevel: 1, beta: 0.1, description: 'Increase observability' }));
        buttons.set('MV', B('Move', 'MV', 'Action', ['move'], { description: 'Change position in space' }));
        buttons.set('SC', B('Scale', 'SC', 'Action', ['scale'], { description: 'Uniform scaling transformation' }));
        buttons.set('NG', B('Negate', 'NG', 'Action', ['un'], { description: 'Reverse or negate current state' }));
        buttons.set('CN', B('Counter', 'CN', 'Action', ['counter'], { description: 'Apply counteracting force' }));
        return buttons;
    }
}

export { LLEMath, Morpheme, Button, StationaryUnit, ScalingOperations, ButtonFactory };
