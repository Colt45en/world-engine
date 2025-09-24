/**
 * Lexical Logic Engine (LLE)
 * A math-first engine where words behave like buttons, and each "click" applies
 * a precise transform that can scale up (more abstract) or down (more concrete).
 * Think Lego bricks for thought.
 */

// Core mathematical utilities
class LLEMath {
    static multiply(A, B) {
        if (Array.isArray(B[0])) {
            // Matrix multiplication
            return A.map(row =>
                B[0].map((_, j) =>
                    row.reduce((sum, val, k) => sum + val * B[k][j], 0)
                )
            );
        } else {
            // Matrix-vector multiplication
            return A.map(row =>
                row.reduce((sum, val, i) => sum + val * B[i], 0)
            );
        }
    }

    static transpose(A) {
        return A[0].map((_, j) => A.map(row => row[j]));
    }

    static identity(n) {
        return Array(n).fill(0).map((_, i) =>
            Array(n).fill(0).map((_, j) => i === j ? 1 : 0)
        );
    }

    static diagonal(values) {
        const n = values.length;
        return Array(n).fill(0).map((_, i) =>
            Array(n).fill(0).map((_, j) => i === j ? values[i] : 0)
        );
    }

    static vectorAdd(a, b) {
        return a.map((val, i) => val + b[i]);
    }

    static vectorScale(v, scalar) {
        return v.map(val => val * scalar);
    }

    static clamp(value, min, max) {
        return Math.max(min, Math.min(max, value));
    }

    static rotationMatrix2D(theta) {
        const cos = Math.cos(theta);
        const sin = Math.sin(theta);
        return [
            [cos, -sin, 0],
            [sin, cos, 0],
            [0, 0, 1]
        ];
    }

    static projectionMatrix(dims, keepDims) {
        const P = Array(keepDims.length).fill(0).map(() => Array(dims).fill(0));
        keepDims.forEach((dim, i) => {
            P[i][dim] = 1;
        });
        return P;
    }
}

/**
 * Stationary Unit (SU) - The atomic state you manipulate
 * SU = ⟨x, Σ, κ, ℓ⟩
 */
class StationaryUnit {
    constructor(d = 3) {
        this.d = d; // dimension
        this.x = Array(d).fill(0); // meaning vector (position in concept space)
        this.x[0] = 1; // default initial state

        this.Sigma = LLEMath.identity(d); // structure/constraint matrix
        this.kappa = 0.7; // confidence/consistency scalar [0,1]
        this.level = 0; // level-of-abstraction (0=concrete, higher=meta)
    }

    copy() {
        const su = new StationaryUnit(this.d);
        su.x = [...this.x];
        su.Sigma = this.Sigma.map(row => [...row]);
        su.kappa = this.kappa;
        su.level = this.level;
        return su;
    }

    toString() {
        return `SU⟨x:[${this.x.map(v => v.toFixed(2)).join(',')}], κ:${this.kappa.toFixed(2)}, ℓ:${this.level}⟩`;
    }

    toJSON() {
        return {
            x: this.x,
            Sigma: this.Sigma,
            kappa: this.kappa,
            level: this.level
        };
    }

    static fromJSON(data) {
        const su = new StationaryUnit(data.x.length);
        su.x = data.x;
        su.Sigma = data.Sigma;
        su.kappa = data.kappa;
        su.level = data.level;
        return su;
    }
}

/**
 * Morpheme - Building blocks for word operators
 */
class Morpheme {
    constructor(symbol, M = null, b = null, effects = {}) {
        this.symbol = symbol;
        this.M = M || LLEMath.identity(3); // transformation matrix
        this.b = b || [0, 0, 0]; // bias vector
        this.effects = effects; // additional effects on Σ, κ, ℓ
    }

    static createBuiltInMorphemes() {
        const morphemes = new Map();

        // Prefixes
        morphemes.set('re', new Morpheme('re', LLEMath.identity(3), [0, 0, 0], {
            description: 'again, backward',
            allowHistory: true
        }));

        morphemes.set('up', new Morpheme('up', [
            [1, 0, 0],
            [0, 0.9, 0],
            [0, 0, 0.9]
        ], [0, 0.1, 0.1], {
            deltaLevel: 1,
            description: 'toward higher'
        }));

        morphemes.set('trans', new Morpheme('trans', LLEMath.rotationMatrix2D(Math.PI/6), [0, 0, 0], {
            description: 'across, transformation'
        }));

        morphemes.set('pre', new Morpheme('pre', LLEMath.identity(3), [0, 0, 0], {
            deltaLevel: -1,
            description: 'before, preparation'
        }));

        // Suffixes
        morphemes.set('or', new Morpheme('or', LLEMath.identity(3), [0, 0, 0], {
            description: 'agent, doer',
            allowEdits: true
        }));

        morphemes.set('ent', new Morpheme('ent', LLEMath.identity(3), [0, 0, 0], {
            deltaLevel: 1,
            description: 'state, quality'
        }));

        morphemes.set('ing', new Morpheme('ing', [
            [1.1, 0, 0],
            [0, 1.1, 0],
            [0, 0, 1]
        ], [0, 0, 0], {
            description: 'continuous action'
        }));

        // Roots
        morphemes.set('build', new Morpheme('build', [
            [1, 0, 0],
            [0, 1.05, 0],
            [0, 0, 1.05]
        ], [0, 0, 0], {
            description: 'construct, create'
        }));

        morphemes.set('vent', new Morpheme('vent', LLEMath.diagonal([1, 0, 1]), [0, 0, 0], {
            description: 'prevent, block',
            constraintType: 'block'
        }));

        return morphemes;
    }
}

/**
 * Button - A clickable word that applies transforms to the Stationary Unit
 */
class Button {
    constructor(label, abbr, wordClass, morphemes, options = {}) {
        this.label = label;
        this.abbr = abbr;
        this.wordClass = wordClass;
        this.morphemes = morphemes; // array of morpheme symbols
        this.deltaLevel = options.deltaLevel || 0;

        // Core transform matrices (calculated from morphemes)
        this.M = options.M || LLEMath.identity(3);
        this.b = options.b || [0, 0, 0];
        this.C = options.C || LLEMath.identity(3);
        this.alpha = options.alpha || 1.0;
        this.beta = options.beta || 0.0;

        // Type information for logical composition
        this.inputType = options.inputType || 'State';
        this.outputType = options.outputType || 'State';

        this.description = options.description || '';
    }

    // Apply this button to a Stationary Unit
    apply(su) {
        const newSU = su.copy();

        // x' = M*x + b
        newSU.x = LLEMath.vectorAdd(LLEMath.multiply(this.M, su.x), this.b);

        // Σ' = C*Σ*C^T
        const CSigma = LLEMath.multiply(this.C, su.Sigma);
        newSU.Sigma = LLEMath.multiply(CSigma, LLEMath.transpose(this.C));

        // κ' = min(1, α*κ + β)
        newSU.kappa = LLEMath.clamp(this.alpha * su.kappa + this.beta, 0, 1);

        // ℓ' = ℓ + δ
        newSU.level = su.level + this.deltaLevel;

        return newSU;
    }

    // Check if this button can be applied after another (type safety)
    canComposeWith(other) {
        return other.outputType === this.inputType;
    }

    toString() {
        return `[${this.abbr}] ${this.label} (${this.wordClass}, δℓ=${this.deltaLevel})`;
    }

    toJSON() {
        return {
            label: this.label,
            abbr: this.abbr,
            class: this.wordClass,
            morphemes: this.morphemes,
            delta_level: this.deltaLevel,
            M: this.M,
            b: this.b,
            C: this.C,
            alpha: this.alpha,
            beta: this.beta,
            description: this.description
        };
    }
}

/**
 * Button Factory - Creates the standard button set
 */
class ButtonFactory {
    static createStandardButtons() {
        const buttons = new Map();

        // Actions
        buttons.set('RB', new Button('Rebuild', 'RB', 'Action', ['re', 'build'], {
            deltaLevel: -1,
            M: [[1, 0, 0], [0, 1.05, 0], [0, 0, 1.05]],
            alpha: 0.98,
            description: 'Recompose from parts (concretize)'
        }));

        buttons.set('UP', new Button('Update', 'UP', 'Action', ['up'], {
            deltaLevel: 0,
            M: [[1, 0, 0], [0, 0.95, 0.05], [0, 0, 1]],
            description: 'Move along current manifold'
        }));

        buttons.set('RS', new Button('Restore', 'RS', 'Action', ['re'], {
            deltaLevel: -1,
            M: [[0.9, 0, 0], [0, 0.9, 0], [0, 0, 0.9]],
            alpha: 1.1,
            beta: 0.05,
            description: 'Revert toward prior fixed point'
        }));

        buttons.set('CV', new Button('Convert', 'CV', 'Action', ['trans'], {
            deltaLevel: 0,
            M: LLEMath.rotationMatrix2D(Math.PI/4),
            description: 'Change representation basis'
        }));

        buttons.set('CH', new Button('Change', 'CH', 'Delta', [], {
            deltaLevel: 0,
            M: [[1, 0.1, 0], [0, 1, 0.1], [0, 0, 1]],
            description: 'Apply difference update'
        }));

        buttons.set('RC', new Button('Recompute', 'RC', 'Action', ['re'], {
            deltaLevel: 0,
            M: [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            alpha: 0.9,
            beta: 0.1,
            description: 'Re-evaluate derived fields'
        }));

        // States and Properties
        buttons.set('ST', new Button('Status', 'ST', 'State/Identity', [], {
            deltaLevel: 1,
            M: [[1, 0, 0], [0, 0.8, 0], [0, 0, 0.8]],
            alpha: 1.0,
            beta: 0.1,
            description: 'Read/expose current invariants'
        }));

        buttons.set('TL', new Button('Translucent', 'TL', 'Property', ['ent'], {
            deltaLevel: 1,
            C: [[0.9, 0, 0], [0, 1.1, 0], [0, 0, 1.1]],
            alpha: 1.0,
            beta: 0.1,
            description: 'Increase observability'
        }));

        // Structure
        buttons.set('CP', new Button('Component', 'CP', 'Structure', [], {
            deltaLevel: -1,
            M: [[1, 0, 0], [0, 1.1, 0], [0, 0, 1.1]],
            description: 'Factor into atoms'
        }));

        buttons.set('MD', new Button('Module', 'MD', 'Structure', [], {
            deltaLevel: 1,
            M: [[1, 0, 0], [0, 0.8, 0], [0, 0, 0.8]],
            description: 'Encapsulate subgraph'
        }));

        buttons.set('SL', new Button('Selection', 'SL', 'Filter', [], {
            deltaLevel: -1,
            C: LLEMath.diagonal([1, 0.5, 1]),
            description: 'Subset by predicate'
        }));

        // Constraints and Modifiers
        buttons.set('PR', new Button('Prevent', 'PR', 'Constraint', ['pre', 'vent'], {
            deltaLevel: 0,
            C: LLEMath.diagonal([1, 0, 1]),
            description: 'Remove disallowed states'
        }));

        buttons.set('ED', new Button('Editor', 'ED', 'Agent/Tool', ['or'], {
            deltaLevel: 0,
            alpha: 1.0,
            beta: 0.05,
            description: 'Enable structural edits'
        }));

        buttons.set('EX', new Button('Extra', 'EX', 'Modifier', [], {
            deltaLevel: 0,
            M: [[1.05, 0, 0], [0, 1.05, 0], [0, 0, 1.05]],
            description: 'Add slack/degree of freedom'
        }));

        buttons.set('BD', new Button('Based', 'BD', 'Grounding', [], {
            deltaLevel: -1,
            M: [[0.95, 0, 0], [0, 0.95, 0], [0, 0, 0.95]],
            alpha: 1.1,
            description: 'Bind to reference frame'
        }));

        return buttons;
    }
}

/**
 * Scaling Operations - Up/Down abstraction transforms
 */
class ScalingOperations {
    static createProjectionMatrix(fromDim, keepIndices) {
        const P = Array(keepIndices.length).fill(0).map(() => Array(fromDim).fill(0));
        keepIndices.forEach((idx, i) => {
            P[i][idx] = 1;
        });
        return P;
    }

    static downscale(su, keepIndices = [0, 2]) {
        const newSU = su.copy();
        const P = this.createProjectionMatrix(su.d, keepIndices);
        newSU.x = LLEMath.multiply(P, su.x);
        newSU.level = Math.max(0, su.level - 1);
        return newSU;
    }

    static upscale(su, abstractionMatrix = null) {
        const newSU = su.copy();

        if (!abstractionMatrix) {
            // Default upscaling: feature pooling
            abstractionMatrix = [
                [0.5, 0.5, 0],
                [0, 0.3, 0.7],
                [0.2, 0.2, 0.6]
            ];
        }

        // α: concrete → abstract
        const abstract = LLEMath.multiply(abstractionMatrix, su.x);

        // γ: abstract → concrete (pseudo-inverse reconstruction)
        const pseudoInv = LLEMath.transpose(abstractionMatrix);
        newSU.x = LLEMath.multiply(pseudoInv, abstract);

        newSU.level = su.level + 1;
        return newSU;
    }
}

/**
 * Lexical Logic Engine - Main orchestrator
 */
class LexicalLogicEngine {
    constructor(dimensions = 3) {
        this.dimensions = dimensions;
        this.su = new StationaryUnit(dimensions);
        this.buttons = ButtonFactory.createStandardButtons();
        this.morphemes = Morpheme.createBuiltInMorphemes();
        this.history = []; // Track operations for undo/analysis

        // Event system
        this.listeners = new Map();
    }

    addEventListener(event, callback) {
        if (!this.listeners.has(event)) {
            this.listeners.set(event, []);
        }
        this.listeners.get(event).push(callback);
    }

    emit(event, data) {
        if (this.listeners.has(event)) {
            this.listeners.get(event).forEach(callback => callback(data));
        }
    }

    // Click a button (core operation)
    click(buttonId) {
        if (!this.buttons.has(buttonId)) {
            throw new Error(`Unknown button: ${buttonId}`);
        }

        const button = this.buttons.get(buttonId);
        const oldSU = this.su.copy();

        this.su = button.apply(this.su);

        const operation = {
            type: 'click',
            button: buttonId,
            before: oldSU,
            after: this.su.copy(),
            timestamp: Date.now()
        };

        this.history.push(operation);
        this.emit('stateChanged', {
            operation,
            currentState: this.su,
            button
        });

        return this.su;
    }

    // Click sequence
    clickSequence(buttonIds) {
        const results = [];
        buttonIds.forEach(id => {
            results.push(this.click(id));
        });
        return results;
    }

    // Scale operations
    downscale(keepIndices) {
        const oldSU = this.su.copy();
        this.su = ScalingOperations.downscale(this.su, keepIndices);

        this.history.push({
            type: 'downscale',
            before: oldSU,
            after: this.su.copy(),
            keepIndices,
            timestamp: Date.now()
        });

        this.emit('stateChanged', {
            operation: { type: 'downscale' },
            currentState: this.su
        });

        return this.su;
    }

    upscale(abstractionMatrix) {
        const oldSU = this.su.copy();
        this.su = ScalingOperations.upscale(this.su, abstractionMatrix);

        this.history.push({
            type: 'upscale',
            before: oldSU,
            after: this.su.copy(),
            abstractionMatrix,
            timestamp: Date.now()
        });

        this.emit('stateChanged', {
            operation: { type: 'upscale' },
            currentState: this.su
        });

        return this.su;
    }

    // Reset to initial state
    reset() {
        this.su = new StationaryUnit(this.dimensions);
        this.history = [];
        this.emit('stateChanged', {
            operation: { type: 'reset' },
            currentState: this.su
        });
    }

    // Get current state
    getState() {
        return {
            su: this.su,
            buttons: Array.from(this.buttons.entries()).map(([id, btn]) => ({
                id,
                ...btn.toJSON()
            })),
            history: this.history.slice(-10) // Last 10 operations
        };
    }

    // Compose operations (for advanced workflows)
    compose(operations) {
        return operations.reduce((su, op) => {
            if (typeof op === 'string') {
                return this.buttons.get(op).apply(su);
            } else if (op.type === 'downscale') {
                return ScalingOperations.downscale(su, op.keepIndices);
            } else if (op.type === 'upscale') {
                return ScalingOperations.upscale(su, op.abstractionMatrix);
            }
            return su;
        }, this.su);
    }
}

// Export for use in other modules
export {
    LexicalLogicEngine,
    StationaryUnit,
    Button,
    ButtonFactory,
    ScalingOperations,
    LLEMath
};
