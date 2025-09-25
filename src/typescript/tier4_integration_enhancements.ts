/*
  Tier-4 IDE Integration Enhancements
  ===================================

  Post-integration enhancements for your Tier-4 Meta System:
  - Determinism validation
  - Dimension loss prevention
  - Developer HUD for debugging
  - Three Ides macros (no more 2D collapse)
  - Auto-planner (Tier-4 autonomous decisions)
  - Session persistence
  - Debug hotkeys
*/

// Types (minimal definitions to avoid import issues)
interface StationaryUnit {
    x: number[];
    kappa: number;
    level: number;
}

interface Event {
    inputCid: string;
    outputCid: string;
    operator: string;
    timestamp: number;
}

interface Snapshot {
    state: StationaryUnit;
    cid: string;
    timestamp: number;
}

// ============================= Validation & Guardrails =============================

export function cidOf(obj: any): string {
    const s = JSON.stringify(obj, Object.keys(obj).sort());
    let h = 2166136261 >>> 0; // FNV-1a seed
    for (let i = 0; i < s.length; i++) {
        h ^= s.charCodeAt(i);
        h = Math.imul(h, 16777619);
    }
    return "cid_" + (h >>> 0).toString(16);
}

// Determinism check: same inputs → same CIDs
export function assertDeterministic(
    applyFn: (s: StationaryUnit) => StationaryUnit,
    state: StationaryUnit
): void {
    const resultA = applyFn(state);
    const resultB = applyFn(state);
    const cidA = cidOf(resultA);
    const cidB = cidOf(resultB);

    if (cidA !== cidB) {
        throw new Error(`Non-deterministic transform: ${cidA} ≠ ${cidB}`);
    }
}

// Snapshot integrity: CIDs must match content
export function assertReload(snapshots: Record<string, Snapshot>): void {
    for (const cid in snapshots) {
        const reCid = cidOf(snapshots[cid].state);
        if (reCid !== cid) {
            throw new Error(`Snapshot CID mismatch @ ${cid}: expected ${cid}, got ${reCid}`);
        }
    }
}

// Lineage integrity: every event must link existing snapshots
export function assertLineage(events: Event[], snaps: Record<string, Snapshot>): void {
    for (const e of events) {
        if (!snaps[e.inputCid]) {
            throw new Error(`Missing input snapshot: ${e.inputCid}`);
        }
        if (!snaps[e.outputCid]) {
            throw new Error(`Missing output snapshot: ${e.outputCid}`);
        }
    }
}

// Prevent silent dimension loss (unless PR was applied)
export function assertNoSilentLoss(
    prev: StationaryUnit,
    next: StationaryUnit,
    lastButton: string
): void {
    const dropped = prev.x.map((v: number, i: number) => (v !== 0 && next.x[i] === 0) ? i : -1)
        .filter((i: number) => i >= 0);

    if (dropped.length && lastButton !== "PR") {
        throw new Error(`Axis dropped without Prevent: ${dropped.join(",")}`);
    }
}

// ============================= Enhanced Mathematical Safety =============================

// NaN/Infinity guards for all numeric operations
export function assertFinite(value: number | number[], context: string = ""): void {
    const check = (v: number, ctx: string) => {
        if (!isFinite(v)) {
            throw new Error(`Non-finite value in ${ctx}: ${v}`);
        }
    };

    if (Array.isArray(value)) {
        value.forEach((v, i) => check(v, `${context}[${i}]`));
    } else {
        check(value, context);
    }
}

// Matrix condition number checking
export function assertMatrixCondition(M: number[][], maxCondition: number = 1e12): void {
    const n = M.length;
    if (n === 0 || !M[0] || M[0].length !== n) {
        throw new Error("Matrix must be square and non-empty");
    }

    // Simple condition check: trace vs determinant ratio
    const trace = M.reduce((sum, row, i) => sum + row[i], 0);
    const det = computeDeterminant(M);

    if (Math.abs(det) < 1e-12) {
        throw new Error("Matrix is singular or near-singular");
    }

    const conditionEstimate = Math.abs(trace / det);
    if (conditionEstimate > maxCondition) {
        throw new Error(`Matrix condition too high: ${conditionEstimate.toExponential(2)}`);
    }
}

// Fast determinant computation for small matrices
function computeDeterminant(M: number[][]): number {
    const n = M.length;

    if (n === 1) return M[0][0];
    if (n === 2) return M[0][0] * M[1][1] - M[0][1] * M[1][0];
    if (n === 3) {
        return (
            M[0][0] * (M[1][1] * M[2][2] - M[1][2] * M[2][1]) -
            M[0][1] * (M[1][0] * M[2][2] - M[1][2] * M[2][0]) +
            M[0][2] * (M[1][0] * M[2][1] - M[1][1] * M[2][0])
        );
    }

    // For n > 3, use LU decomposition or throw error
    throw new Error(`Determinant computation not implemented for ${n}x${n} matrices`);
}

// Enhanced state validation with numerical bounds
export function assertValidState(state: StationaryUnit): void {
    // Check vector components
    assertFinite(state.x, "state.x");

    // Check bounds
    if (state.x.some((v: number) => Math.abs(v) > 1e6)) {
        throw new Error("State vector components exceeded safe bounds");
    }

    // Check scalar properties
    assertFinite(state.kappa, "state.kappa");
    if (state.kappa < 0 || state.kappa > 1) {
        throw new Error(`Invalid kappa value: ${state.kappa} (must be in [0,1])`);
    }

    // Check level bounds
    if (!Number.isInteger(state.level) || Math.abs(state.level) > 100) {
        throw new Error(`Invalid level: ${state.level} (must be integer in [-100,100])`);
    }
}

// Safe matrix-vector multiplication with overflow protection
export function safeMatrixVectorMult(M: number[][], v: number[]): number[] {
    const n = M.length;
    const m = v.length;

    if (n === 0 || !M[0] || M[0].length !== m) {
        throw new Error("Matrix-vector dimension mismatch");
    }

    assertMatrixCondition(M);
    assertFinite(v, "input vector");

    const result = new Array(n);
    for (let i = 0; i < n; i++) {
        let sum = 0;
        for (let j = 0; j < m; j++) {
            sum += M[i][j] * v[j];
        }
        assertFinite(sum, `result[${i}]`);
        result[i] = sum;
    }

    return result;
}

// Safe operator application with all checks
export function safeApplyOperator(
    state: StationaryUnit,
    M: number[][],
    b: number[],
    alpha: number = 1.0
): StationaryUnit {
    // Pre-validation
    assertValidState(state);
    assertMatrixCondition(M);
    assertFinite(b, "bias vector");
    assertFinite(alpha, "alpha");

    // Dimension checks
    if (M.length !== state.x.length || b.length !== state.x.length) {
        throw new Error("Operator dimensions don't match state");
    }

    // Apply transformation
    const Mx = safeMatrixVectorMult(M, state.x);
    const newX = Mx.map((mx, i) => alpha * mx + b[i]);

    // Post-validation
    assertFinite(newX, "transformed state");

    const newState = {
        ...state,
        x: newX
    };

    assertValidState(newState);
    return newState;
}

// ============================= Developer HUD =============================

export interface HUDState {
    op: string;
    dt: number;
    dx: number;
    mu: number;
    level: number;
    kappa: number;
    lastError?: string;
}

export function createHUD(): HUDState {
    return {
        op: "-",
        dt: 0,
        dx: 0,
        mu: 0,
        level: 0,
        kappa: 0
    };
}

export function updateHUD(
    hud: HUDState,
    abbr: string,
    prev: StationaryUnit,
    next: StationaryUnit,
    dt: number
): HUDState {
    const dx = Math.hypot(...next.x.map((v: number, i: number) => v - prev.x[i]));
    const mu = Math.abs(next.x[0]) + next.x[1] + next.x[2] + next.x[3];

    return {
        ...hud,
        op: abbr,
        dt: +dt.toFixed(2),
        dx: +dx.toFixed(4),
        mu: +mu.toFixed(3),
        level: next.level,
        kappa: +next.kappa.toFixed(3),
        lastError: undefined
    };
}

// ============================= Three Ides Macros =============================

export const MACROS: Record<string, string[]> = {
    // Your original three ides as first-class macros
    IDE_A: ["ST", "SL", "CP"],           // Analysis path
    IDE_B: ["CV", "PR", "RC"],           // Constraint path
    IDE_C: ["TL", "RB", "MD"],           // Build path

    // Alignment step to prevent 2D collapse
    ALIGN_IDES: ["CV", "CV"],            // Double rotation for dimensional stability

    // Explicit merge with no dimension loss
    MERGE_ABC: ["ALIGN_IDES", "IDE_A", "IDE_B", "IDE_C"].flatMap(k => MACROS[k] || [k]),

    // Common development workflows
    OPTIMIZE: ["ST", "CP", "PR", "RC"],  // Status → Factor → Constrain → Recompute
    DEBUG: ["TL", "SL", "ED", "RS"],     // Make visible → Select → Edit → Restore
    REFACTOR: ["ST", "CP", "CV", "MD"],  // Status → Factor → Convert → Package

    // Confidence building sequences
    STABILIZE: ["PR", "RC", "TL"],       // Constrain → Recompute → Make visible
    GROUND: ["BD", "CP", "RB"],          // Ground → Factor → Rebuild
};

export function runMacro(
    name: string,
    applyFn: (abbr: string) => void
): string[] {
    const sequence = MACROS[name];
    if (!sequence) {
        throw new Error(`Unknown macro: ${name}`);
    }

    // Expand nested macros
    const expanded: string[] = [];
    for (const step of sequence) {
        if (MACROS[step]) {
            expanded.push(...MACROS[step]);
        } else {
            expanded.push(step);
        }
    }

    // Execute sequence
    for (const abbr of expanded) {
        applyFn(abbr);
    }

    return expanded;
}

// ============================= Structural Diff =============================

export interface StateDiff {
    dx: number[];
    dκ: number;
    dL: number;
    magnitude: number;
}

export function computeDiff(a: StationaryUnit, b: StationaryUnit): StateDiff {
    const dx = a.x.map((v: number, i: number) => +(b.x[i] - v).toFixed(5));
    const dκ = +(b.kappa - a.kappa).toFixed(5);
    const dL = b.level - a.level;
    const magnitude = Math.hypot(...dx) + Math.abs(dκ) + Math.abs(dL);

    return { dx, dκ, dL, magnitude: +magnitude.toFixed(4) };
}

// ============================= Auto-Planner (Tier-4) =============================

export function scoreState(s: StationaryUnit): number {
    // Balance confidence and abstraction level
    return 0.6 * s.kappa + 0.4 * Math.tanh(s.level / 3.0);
}

export function planNextMove(
    currentState: StationaryUnit,
    operators: Record<string, ButtonOperator>
): { abbr: string; expectedScore: number; reasoning: string } | null {
    const candidates = ["ST", "CP", "RB", "RC", "CV", "PR", "TL", "UP", "RS"];
    let best = { abbr: "", val: -Infinity, reasoning: "" };

    const currentScore = scoreState(currentState);

    for (const abbr of candidates) {
        const operator = operators[abbr];
        if (!operator) continue;

        try {
            const futureState = operator.apply(currentState);
            const futureScore = scoreState(futureState);
            const improvement = futureScore - currentScore;
            const val = improvement - (operator.cost || 0.03); // Small action cost

            let reasoning = `${abbr}: score ${currentScore.toFixed(3)} → ${futureScore.toFixed(3)}`;

            if (val > best.val) {
                best = { abbr, val, reasoning };
            }
        } catch (error) {
            // Skip operators that would fail
            continue;
        }
    }

    return best.abbr ? {
        abbr: best.abbr,
        expectedScore: best.val,
        reasoning: best.reasoning
    } : null;
}

// ============================= Session Persistence =============================

export interface SessionData {
    su: StationaryUnit;
    events: Event[];
    snapshots: Record<string, Snapshot>;
    macroHistory: string[];
    timestamp: number;
    version: string;
}

export function saveSession(
    su: StationaryUnit,
    events: Event[],
    snapshots: Record<string, Snapshot>,
    macroHistory: string[] = []
): void {
    const sessionData: SessionData = {
        su,
        events,
        snapshots,
        macroHistory,
        timestamp: Date.now(),
        version: "1.0.0"
    };

    try {
        localStorage.setItem("tier4_session", JSON.stringify(sessionData));
        console.log("✅ Session saved to localStorage");
    } catch (error) {
        console.error("❌ Failed to save session:", error);
    }
}

export function loadSession(): SessionData | null {
    try {
        const saved = localStorage.getItem("tier4_session");
        if (!saved) return null;

        const data = JSON.parse(saved) as SessionData;

        // Validate loaded data
        if (!data.su || !data.events || !data.snapshots) {
            throw new Error("Invalid session data structure");
        }

        console.log(`✅ Session loaded: ${data.events.length} events, ${Object.keys(data.snapshots).length} snapshots`);
        return data;
    } catch (error) {
        console.error("❌ Failed to load session:", error);
        return null;
    }
}

// ============================= Debug Hotkeys =============================

export interface HotkeyConfig {
    onReset: () => void;
    onPlan: () => void;
    onUpscale: () => void;
    onDownscale: () => void;
    onSave: () => void;
    onLoad: () => void;
}

export function setupHotkeys(config: HotkeyConfig): () => void {
    const handleKeyDown = (event: KeyboardEvent) => {
        // Only trigger if no input is focused
        if (event.target instanceof HTMLInputElement || event.target instanceof HTMLTextAreaElement) {
            return;
        }

        switch (event.key.toLowerCase()) {
            case 'r':
                if (event.ctrlKey || event.metaKey) return; // Don't override browser refresh
                config.onReset();
                console.log("🔄 Reset triggered");
                break;

            case 'p':
                config.onPlan();
                console.log("🧠 Auto-planner triggered");
                break;

            case 'u':
                config.onUpscale();
                console.log("📈 Upscale triggered");
                break;

            case 'd':
                config.onDownscale();
                console.log("📉 Downscale triggered");
                break;

            case 's':
                if (event.ctrlKey || event.metaKey) {
                    event.preventDefault();
                    config.onSave();
                    console.log("💾 Save triggered");
                }
                break;

            case 'l':
                if (event.ctrlKey || event.metaKey) {
                    event.preventDefault();
                    config.onLoad();
                    console.log("📂 Load triggered");
                }
                break;
        }
    };

    window.addEventListener('keydown', handleKeyDown);

    // Return cleanup function
    return () => {
        window.removeEventListener('keydown', handleKeyDown);
    };
}

// ============================= Regression Testing =============================

export function runRegressionTests(operators: Record<string, ButtonOperator>): {
    passed: number;
    failed: number;
    results: Array<{ test: string; passed: boolean; error?: string }>;
} {
    const results: Array<{ test: string; passed: boolean; error?: string }> = [];

    const baseState: StationaryUnit = {
        x: [0.6, 0.4, 0.2, 0.5],
        sigma: [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
        kappa: 0.62,
        level: 0
    };

    // Test 1: RB then ST reproducibility
    try {
        const s1 = operators.ST.apply(operators.RB.apply(baseState));
        const s2 = operators.ST.apply(operators.RB.apply(baseState));
        const cidA = cidOf(s1);
        const cidB = cidOf(s2);

        results.push({
            test: "RB→ST reproducibility",
            passed: cidA === cidB,
            error: cidA !== cidB ? `CID mismatch: ${cidA} ≠ ${cidB}` : undefined
        });
    } catch (error) {
        results.push({
            test: "RB→ST reproducibility",
            passed: false,
            error: String(error)
        });
    }

    // Test 2: Operator determinism
    for (const [abbr, op] of Object.entries(operators)) {
        try {
            assertDeterministic(op.apply, baseState);
            results.push({
                test: `${abbr} determinism`,
                passed: true
            });
        } catch (error) {
            results.push({
                test: `${abbr} determinism`,
                passed: false,
                error: String(error)
            });
        }
    }

    // Test 3: Macro consistency
    try {
        // Run OPTIMIZE macro twice
        let state1 = baseState;
        for (const op of MACROS.OPTIMIZE) {
            state1 = operators[op].apply(state1);
        }

        let state2 = baseState;
        for (const op of MACROS.OPTIMIZE) {
            state2 = operators[op].apply(state2);
        }

        const consistent = cidOf(state1) === cidOf(state2);
        results.push({
            test: "OPTIMIZE macro consistency",
            passed: consistent,
            error: consistent ? undefined : "Macro produced different results"
        });
    } catch (error) {
        results.push({
            test: "OPTIMIZE macro consistency",
            passed: false,
            error: String(error)
        });
    }

    const passed = results.filter(r => r.passed).length;
    const failed = results.filter(r => !r.passed).length;

    return { passed, failed, results };
}

// ============================= Usage Examples =============================

/*
// Integration example for your React component:

import {
    assertDeterministic,
    MACROS,
    runMacro,
    createHUD,
    updateHUD,
    setupHotkeys,
    saveSession,
    loadSession,
    planNextMove
} from './tier4_integration_enhancements';

// In your component:
const [hud, setHUD] = useState(createHUD());

const applyWithGuards = (abbr: string) => {
    const t0 = performance.now();
    const prevState = {...su};

    try {
        // Apply operator
        const newState = operators[abbr].apply(su);

        // Validate
        assertDeterministic(operators[abbr].apply, su);
        assertNoSilentLoss(prevState, newState, abbr);

        // Update state
        setSU(newState);

        // Update HUD
        const dt = performance.now() - t0;
        setHUD(updateHUD(hud, abbr, prevState, newState, dt));

        // Auto-save
        saveSession(newState, events, snapshots);

    } catch (error) {
        setHUD({...hud, lastError: String(error)});
        console.error("Operator failed:", error);
    }
};

// Setup hotkeys
useEffect(() => {
    return setupHotkeys({
        onReset: () => setSU(initialState),
        onPlan: () => {
            const plan = planNextMove(su, operators);
            if (plan) applyWithGuards(plan.abbr);
        },
        onSave: () => saveSession(su, events, snapshots),
        onLoad: () => {
            const session = loadSession();
            if (session) {
                setSU(session.su);
                setEvents(session.events);
                setSnapshots(session.snapshots);
            }
        },
        onUpscale: () => applyWithGuards('ST'),
        onDownscale: () => applyWithGuards('BD')
    });
}, [su, events, snapshots]);
*/
