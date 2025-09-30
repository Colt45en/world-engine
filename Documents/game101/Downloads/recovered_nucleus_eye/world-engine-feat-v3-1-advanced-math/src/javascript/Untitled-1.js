// figure8-breakdown-reconstruction.js
// Explicit atomic breakdown (downward) and reconstruction (upward)
// Drop-in helper for your MetaFractal engine. Pure data first; audio/visual hooks optional.

/********************
 * Core data types
 ********************/

/** Stationary Unit (SU): linear state + meta */
export class StationaryUnit {
    constructor(dim = 3) {
        this.x = new Array(dim).fill(1);     // latent state vector
        this.level = 0;                      // reconstruction depth
        this.kappa = 1.0;                    // damping/consistency (0..1)
    }
    copy() { const s = new StationaryUnit(this.x.length); s.x = [...this.x]; s.level = this.level; s.kappa = this.kappa; return s; }
}

/** Linear transform bundle */
export function applyLinear({ M, b }, x) {
    const y = new Array(M.length).fill(0);
    for (let i = 0; i < M.length; i++) {
        let acc = 0;
        for (let j = 0; j < M[i].length; j++) acc += M[i][j] * (x[j] ?? 0);
        y[i] = acc + (b?.[i] ?? 0);
    }
    return y;
}

/********************
 * Morpheme registry (english)
 ********************/

class Morpheme {
    constructor(key, { M, b, note = "", audio = {} }) {
        this.type = "morpheme";
        this.key = key;        // 're', 'build', '-ness', etc.
        this.M = M;            // matrix (dim x dim)
        this.b = b;            // bias (dim)
        this.note = note;      // description
        this.audio = audio;    // optional AM/FM mapping
    }
}

export function createMorphemeRegistry(dim = 3) {
    const I = (d) => Array.from({ length: d }, (_, i) => Array.from({ length: d }, (_, j) => i === j ? 1 : 0));
    const R = new Map();

    R.set("re", new Morpheme("re", {
        M: [[0.95, 0, 0], [0, 1.05, 0], [0, 0, 1]],
        b: [0, 0, 0],
        note: "again/restore",
        audio: { amDepth: 0.2, fmDepth: 0.05, carrierShift: -20, filterCut: 700 }
    }));

    R.set("build", new Morpheme("build", {
        M: [[1.15, 0, 0], [0, 1.15, 0], [0, 0, 1.05]],
        b: [0.05, 0.05, 0],
        note: "construct",
        audio: { amDepth: 0.15, fmDepth: 0.15, carrierShift: 0, filterCut: 1600 }
    }));

    // Examples
    R.set("multi", new Morpheme("multi", {
        M: [[1.4, 0, 0], [0, 1.4, 0], [0, 0, 1.1]], b: [0, 0, 0.05], note: "many/scale",
        audio: { amDepth: 0.3, fmDepth: 0.35, carrierShift: 40, filterCut: 2000 }
    }));
    R.set("-ize", new Morpheme("-ize", {
        M: [[1, 0.08, 0], [0, 1, 0.08], [0, 0, 1.08]], b: [0, 0, 0], note: "make into",
        audio: { amDepth: 0.1, fmDepth: 0.3, carrierShift: 20, filterCut: 1800 }
    }));
    R.set("-ness", new Morpheme("-ness", {
        M: [[1.05, 0, 0], [0, 1.05, 0], [0, 0, 0.95]], b: [0, 0, 0.05], note: "state/quality",
        audio: { amDepth: 0.25, fmDepth: 0.15, carrierShift: 10, filterCut: 1000 }
    }));

    // identity fallback
    R.set("<id>", new Morpheme("<id>", { M: I(dim), b: new Array(dim).fill(0), note: "identity" }));
    return R;
}

/********************
 * English: atomic breakdown → reconstruction
 ********************/

/** crude longest-match prefix/root/suffix splitter */
export function englishBreakdown(word) {
    const w = (word || "").toLowerCase();
    // toy rules; swap with your longest-match splitter
    const prefixes = [];
    let root = w; const suffixes = [];
    if (root.startsWith("re")) { prefixes.push("re"); root = root.slice(2); }
    if (root.startsWith("multi")) { prefixes.push("multi"); root = root.slice(5); }
    if (root.endsWith("ness")) { suffixes.push("-ness"); root = root.slice(0, -4); }
    if (root.endsWith("ize")) { suffixes.push("-ize"); root = root.slice(0, -3); }

    const morphemes = [
        ...prefixes.map(p => ({ type: "prefix", text: p })),
        { type: "root", text: root },
        ...suffixes.map(s => ({ type: "suffix", text: s }))
    ];

    return {
        token: w,
        prefixes, root, suffixes, morphemes
    };
}

export function englishReconstruct(atom, registry = createMorphemeRegistry(3)) {
    const su = new StationaryUnit(3);
    let kappa = 1.0; let level = 0;
    for (const m of atom.morphemes) {
        const key = m.text;
        const mor = registry.get(key) || registry.get("<id>");
        su.x = applyLinear({ M: mor.M, b: mor.b }, su.x);
        level += 1; kappa *= 0.95; // simple attenuation per application
    }
    su.level = level; su.kappa = kappa;
    return { word: atom.token, su };
}

/********************
 * Math: atomic breakdown → reconstruction
 ********************/

const isNum = (t) => /^\d+(?:\.\d+)?$/.test(t);
const isOp = (t) => /^[+\-*/]$/.test(t);

export function mathBreakdown(expr) {
    const tokens = expr.replace(/\s+/g, "").match(/\d+\.?\d*|[+\-*/()]/g) || [];
    const atoms = tokens.map(t => isNum(t) ? { type: "number", value: Number(t) }
        : isOp(t) ? { type: "operator", op: t, effect: "combine" }
            : { type: "paren", value: t });
    return atoms;
}

export function mathReconstruct(atoms) {
    // very small safe evaluator: left-to-right with * and / precedence
    const arr = atoms.filter(a => a.type !== "paren").map(a => a.type === "number" ? a.value : a.op);
    const collapse = (list, ops) => {
        const out = [];
        for (let i = 0; i < list.length; i++) {
            const t = list[i];
            if (typeof t === "string" && ops.includes(t)) {
                const a = out.pop(); const b = list[++i];
                out.push(t === "*" ? a * b : a / b);
            } else out.push(t);
        }
        return out;
    };
    let tmp = collapse(arr, ["*", "/"]);
    let val = tmp[0];
    for (let i = 1; i < tmp.length; i += 2) { const op = tmp[i]; const b = tmp[i + 1]; val = op === "+" ? val + b : val - b; }

    // Pack as SU-like record for symmetry
    return { expression: atoms.map(a => a.type === "number" ? String(a.value) : a.op || a.value).join(""), value: val, M: [[1, 0], [0, 1]], b: [val], level: 1 };
}

/********************
 * Unified figure-8 API
 ********************/

export function breakdown(input) {
    const isMath = /[+\-*/()]/.test(input) && /\d/.test(input);
    if (isMath) {
        return { mode: "math", downward: mathBreakdown(input) };
    } else {
        const e = englishBreakdown(input);
        const atoms = e.morphemes.map(m => ({ type: m.type, text: m.text, effect: "linear" }));
        return { mode: "english", downward: atoms, meta: e };
    }
}

export function reconstruct(input, registry) {
    const { mode } = breakdown(input);
    if (mode === "math") {
        const atoms = mathBreakdown(input);
        const up = mathReconstruct(atoms);
        return { mode, upward: up };
    } else {
        const e = englishBreakdown(input);
        const up = englishReconstruct(e, registry);
        return { mode, upward: up };
    }
}

export function debugTrace(input, registry) {
    const d = breakdown(input);
    const r = reconstruct(input, registry);
    return { downward: d.downward ?? d.meta?.morphemes, upward: r.upward, mode: d.mode, meta: d.meta };
}

/********************
 * Optional: audio/visual control lattice hooks
 ********************/

/** Map morphemes → audio control vector θ (AM/FM/etc.) */
export function toControlTheta(atom, registry = createMorphemeRegistry(3)) {
    let D_AM = 0, D_FM = 0, f0 = 180, fc = 1200, Q = 0.9;
    let N = 12; // partials
    for (const m of atom.morphemes) {
        const mor = registry.get(m.text) || registry.get("<id>");
        const a = mor.audio || {};
        D_AM += a.amDepth || 0; D_FM += (a.fmDepth || 0) * 6; f0 += a.carrierShift || 0; fc = a.filterCut || fc;
    }
    D_AM = Math.max(0, Math.min(0.95, D_AM));
    return { D_AM, D_FM, f0, fc, Q, N };
}

/********************
 * Demo helpers
 ********************/

export function exampleEnglish() {
    const e = englishBreakdown("rebuild");
    const up = englishReconstruct(e);
    return { downward: e.morphemes, upward: up };
}

export function exampleMath() {
    const a = mathBreakdown("2 + 3");
    const up = mathReconstruct(a);
    return { downward: a, upward: up };
}
