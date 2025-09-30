/**
 * LLE Stable Math - Enhanced Mathematical Operations with Safety and Pseudo-Inverse
 * Strict shape checking, Moore-Penrose pseudo-inverse, dimension-agnostic operations
 */

// @syn:example
// name: matrix-multiply-basic
// input:
//   args: [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]
// expect: [[19, 22], [43, 50]]
// ---

// @syn:property
// name: multiply-associative
// gens:
//   a: [[1, 2], [3, 4]]
//   b: [[2, 0], [1, 3]]
//   c: [[1, 1], [0, 2]]
// prop: JSON.stringify(LLEMath.multiply(LLEMath.multiply(a,b),c)) === JSON.stringify(LLEMath.multiply(a,LLEMath.multiply(b,c)))
// trials: 50
// ---

// @syn:contract
// name: multiply-shape-safety
// pre: Array.isArray(a) && Array.isArray(b) && Array.isArray(a[0]) && Array.isArray(b[0])
// post: Array.isArray(out) && Array.isArray(out[0])
// fn: multiply
// ---

class LLEMath {
  /**
   * Matrix multiplication with strict shape validation
   */
  static multiply(A, B) {
    const isMatA = Array.isArray(A[0]);
    const isMatB = Array.isArray(B[0]);

    if (!isMatA) throw new Error('multiply: A must be matrix');

    // Matrix × Matrix
    if (isMatB) {
      const r = A.length, k = A[0].length, k2 = B.length, c = B[0].length;
      if (k !== k2) throw new Error(`multiply: shape mismatch (${r}×${k})·(${k2}×${c})`);

      return Array.from({length: r}, (_, i) =>
        Array.from({length: c}, (_, j) =>
          A[i].reduce((sum, aij, t) => sum + aij * B[t][j], 0)
        )
      );
    }

    // Matrix × Vector
    const r = A.length, k = A[0].length, n = B.length;
    if (k !== n) throw new Error(`multiply: shape mismatch (${r}×${k})·(${n})`);

    return Array.from({length: r}, (_, i) =>
      A[i].reduce((sum, aij, j) => sum + aij * B[j], 0)
    );
  }

  /**
   * Matrix transpose with validation
   */
  static transpose(A) {
    if (!Array.isArray(A[0])) throw new Error('transpose: A must be matrix');
    const r = A.length, c = A[0].length;
    return Array.from({length: c}, (_, j) =>
      Array.from({length: r}, (_, i) => A[i][j])
    );
  }

  /**
   * Identity matrix generator
   */
  static identity(n) {
    return Array.from({length: n}, (_, i) =>
      Array.from({length: n}, (_, j) => (i === j ? 1 : 0))
    );
  }

  /**
   * Diagonal matrix from values
   */
  static diagonal(values) {
    const n = values.length;
    return Array.from({length: n}, (_, i) =>
      Array.from({length: n}, (_, j) => (i === j ? values[i] : 0))
    );
  }

  /**
   * Vector addition with length validation
   */
  // @syn:example
  // name: vector-add-simple
  // input:
  //   args: [[1, 2, 3], [4, 5, 6]]
  // expect: [5, 7, 9]
  // ---
  // @syn:property
  // name: vector-add-commutative
  // gens:
  //   a: [1, 2, 3]
  //   b: [4, 5, 6]
  // prop: JSON.stringify(LLEMath.vectorAdd(a,b)) === JSON.stringify(LLEMath.vectorAdd(b,a))
  // trials: 100
  // ---
  static vectorAdd(a, b) {
    if (a.length !== b.length) throw new Error('vectorAdd: length mismatch');
    return a.map((v, i) => v + b[i]);
  }

  /**
   * Vector scaling
   */
  static vectorScale(v, s) {
    return v.map(val => val * s);
  }

  /**
   * Value clamping
   */
  static clamp(v, min, max) {
    return Math.max(min, Math.min(max, v));
  }

  /**
   * 2D rotation in homogeneous 3D (x,y,meta) coordinates
   */
  static rotationMatrix2D(theta) {
    const c = Math.cos(theta), s = Math.sin(theta);
    return [
      [c, -s, 0],
      [s,  c, 0],
      [0,  0, 1]
    ];
  }

  /**
   * Projection matrix for dimension reduction
   */
  static projectionMatrix(dims, keepDims) {
    const P = Array.from({length: keepDims.length}, () => Array(dims).fill(0));
    keepDims.forEach((dim, i) => {
      if (dim >= dims) throw new Error('projection: index out of range');
      P[i][dim] = 1;
    });
    return P;
  }

  /**
   * Moore-Penrose pseudo-inverse for stable reconstruction
   * A^+ = A^T·(A·A^T + λI)^-1 for tall matrices
   * A^+ = (A^T·A + λI)^-1·A^T for wide matrices
   */
  static pseudoInverse(A, lambda = 1e-6) {
    const AT = this.transpose(A);
    const m = A.length, n = A[0].length;

    if (m >= n) {
      // Tall matrix: A^+ = A^T·(A·A^T + λI)^-1
      const AAT = this.multiply(A, AT); // m×m
      const reg = AAT.map((row, i) =>
        row.map((v, j) => v + (i === j ? lambda : 0))
      );
      const inv = this.inverseSmall(reg);
      return this.multiply(AT, inv); // n×m
    } else {
      // Wide matrix: A^+ = (A^T·A + λI)^-1·A^T
      const ATA = this.multiply(AT, A); // n×n
      const reg = ATA.map((row, i) =>
        row.map((v, j) => v + (i === j ? lambda : 0))
      );
      const inv = this.inverseSmall(reg);
      return this.multiply(inv, AT); // n×m
    }
  }

  /**
   * Gauss-Jordan inverse for small symmetric matrices
   */
  static inverseSmall(M) {
    const n = M.length;
    // Augment with identity
    const A = M.map((row, i) => [...row, ...this.identity(n)[i]]);

    // Gauss-Jordan elimination
    for (let i = 0; i < n; i++) {
      // Find pivot
      const piv = A[i][i];
      if (Math.abs(piv) < 1e-12) {
        throw new Error('inverseSmall: singular matrix');
      }

      const invPiv = 1 / piv;
      for (let j = 0; j < 2 * n; j++) {
        A[i][j] *= invPiv;
      }

      // Eliminate column
      for (let r = 0; r < n; r++) {
        if (r === i) continue;
        const f = A[r][i];
        for (let j = 0; j < 2 * n; j++) {
          A[r][j] -= f * A[i][j];
        }
      }
    }

    // Extract right half
    return A.map(row => row.slice(n));
  }

  /**
   * Validate that all elements are finite numbers
   */
  // @syn:example
  // name: validate-finite-pass
  // input:
  //   args: [[1, 2, 3, 4.5], "test-vector"]
  // expect: true
  // ---
  // @syn:contract
  // name: validate-finite-throws-on-infinite
  // pre: vector.some(x => !Number.isFinite(x))
  // post: false
  // fn: validateFinite
  // ---
  static validateFinite(vector, name = 'vector') {
    if (!vector.every(Number.isFinite)) {
      throw new Error(`${name} contains non-finite values: ${vector}`);
    }
    return true;
  }

  /**
   * Safe matrix-vector operation with validation
   */
  static safeTransform(M, x, b = null) {
    // Validate inputs
    this.validateFinite(x, 'input vector');
    if (b) this.validateFinite(b, 'bias vector');

    const result = this.multiply(M, x);
    if (b) {
      const final = this.vectorAdd(result, b);
      this.validateFinite(final, 'result vector');
      return final;
    }

    this.validateFinite(result, 'result vector');
    return result;
  }
}

/**
 * Enhanced Morpheme System with Mathematical Transformations
 */
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

/**
 * Enhanced Button with Morpheme-Driven Composition
 */
class Button {
  constructor(label, abbr, wordClass, morphemes, options = {}, registry = null) {
    this.label = label;
    this.abbr = abbr;
    this.wordClass = wordClass;
    this.morphemes = morphemes;
    this.deltaLevel = options.deltaLevel || 0;

    // Compose from morphemes if registry provided
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

        // Compose transformations: x' = M_m * (M * x + b) + b_m = (M_m * M) * x + (M_m * b + b_m)
        const newM = LLEMath.multiply(m.M, M);
        const newB = LLEMath.vectorAdd(LLEMath.multiply(m.M, b), m.b);

        M = newM;
        b = newB;

        if (m.effects?.C) {
          C = LLEMath.multiply(m.effects.C, C);
        }
        if (m.effects?.deltaLevel) {
          delta += m.effects.deltaLevel;
        }
        if (typeof m.effects?.alpha === 'number') {
          alpha *= m.effects.alpha;
        }
        if (typeof m.effects?.beta === 'number') {
          beta += m.effects.beta;
        }
      }
    }

    // Allow explicit overrides
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

  /**
   * Apply button transformation with strict validation
   */
  apply(su) {
    const dim = su.d;

    // Shape checks
    if (this.M.length !== dim || this.M[0].length !== dim) {
      throw new Error(`Button ${this.abbr}: M shape ${this.M.length}×${this.M[0].length} != ${dim}×${dim}`);
    }
    if (this.b.length !== dim) {
      throw new Error(`Button ${this.abbr}: b length ${this.b.length} != ${dim}`);
    }
    if (this.C.length !== dim || this.C[0].length !== dim) {
      throw new Error(`Button ${this.abbr}: C shape ${this.C.length}×${this.C[0].length} != ${dim}×${dim}`);
    }

    const newSU = su.copy();

    // Apply linear transformation with validation
    newSU.x = LLEMath.safeTransform(this.M, su.x, this.b);

    // Apply covariance transformation: Σ' = C * Σ * C^T
    const CS = LLEMath.multiply(this.C, su.Sigma);
    newSU.Sigma = LLEMath.multiply(CS, LLEMath.transpose(this.C));

    // Update parameters
    newSU.kappa = LLEMath.clamp((this.alpha * su.kappa) + this.beta, 0, 1);
    newSU.level = su.level + this.deltaLevel;

    // Validate result
    LLEMath.validateFinite(newSU.x, `Button ${this.abbr} result`);

    return newSU;
  }

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
 * Enhanced Stationary Unit with dimension flexibility
 */
class StationaryUnit {
  constructor(dimensions = 3, x = null, Sigma = null, kappa = 1.0, level = 0) {
    this.d = dimensions;
    this.x = x || Array(dimensions).fill(0);
    this.Sigma = Sigma || LLEMath.identity(dimensions);
    this.kappa = kappa;
    this.level = level;
    this.timestamp = Date.now();
  }

  copy() {
    return new StationaryUnit(
      this.d,
      [...this.x],
      this.Sigma.map(row => [...row]),
      this.kappa,
      this.level
    );
  }

  toString() {
    const pos = this.x.map(v => v.toFixed(3)).join(',');
    return `SU(x=[${pos}], κ=${this.kappa.toFixed(3)}, ℓ=${this.level}, d=${this.d})`;
  }
}

/**
 * Enhanced Scaling Operations with Pseudo-Inverse Recovery
 */
class ScalingOperations {
  static createProjectionMatrix(fromDim, keepIndices) {
    const P = Array.from({length: keepIndices.length}, () => Array(fromDim).fill(0));
    keepIndices.forEach((idx, i) => {
      if (idx >= fromDim) throw new Error('projection: index out of bounds');
      P[i][idx] = 1;
    });
    return P;
  }

  /**
   * Downscale with proper covariance projection
   */
  static downscale(su, keepIndices = [0, 2]) {
    const newSU = su.copy();
    const P = this.createProjectionMatrix(su.d, keepIndices); // m×d

    newSU.x = LLEMath.multiply(P, su.x);       // m-dimensional

    // Project covariance: Σ' = P * Σ * P^T
    const PS = LLEMath.multiply(P, su.Sigma);
    newSU.Sigma = LLEMath.multiply(PS, LLEMath.transpose(P));

    newSU.level = Math.max(0, su.level - 1);
    newSU.d = keepIndices.length;

    return newSU;
  }

  /**
   * Upscale with pseudo-inverse reconstruction
   */
  static upscale(su, abstractionMatrix = null, toDim = 3) {
    const newSU = su.copy();

    // Default abstraction matrix for 3D case
    const A = abstractionMatrix || [
      [0.5, 0.5, 0],     // Pool first two dimensions
      [0, 0.3, 0.7],     // Weighted combination
      [0.2, 0.2, 0.6]    // Another combination
    ];

    // Use pseudo-inverse for stable reconstruction
    const Aplus = LLEMath.pseudoInverse(A, 1e-6);

    // If input is already abstracted, use it; otherwise abstract first
    const abstract = su.d === A.length ? su.x : LLEMath.multiply(A, su.x);
    const recon = LLEMath.multiply(Aplus, abstract);

    // Pad or truncate to target dimensions
    const xUp = recon.length < toDim ?
      [...recon, ...Array(toDim - recon.length).fill(0)] :
      recon.slice(0, toDim);

    newSU.x = xUp;

    // Reconstruct covariance: Σ' ≈ A^+ * Σ * (A^+)^T
    const AplusT = LLEMath.transpose(Aplus);
    const AS = LLEMath.multiply(Aplus, su.Sigma);
    newSU.Sigma = LLEMath.multiply(AS, AplusT);

    // Ensure proper dimensions for covariance
    if (newSU.Sigma.length < toDim) {
      const padded = LLEMath.identity(toDim);
      for (let i = 0; i < newSU.Sigma.length; i++) {
        for (let j = 0; j < newSU.Sigma[0].length; j++) {
          padded[i][j] = newSU.Sigma[i][j];
        }
      }
      newSU.Sigma = padded;
    }

    newSU.level = su.level + 1;
    newSU.d = toDim;

    return newSU;
  }
}

/**
 * Enhanced Button Factory with Morpheme-Driven Composition
 */
class ButtonFactory {
  static createStandardButtons(dim = 3) {
    const buttons = new Map();
    const morphemes = Morpheme.createBuiltInMorphemes(dim);

    // Helper function to create button with morpheme registry
    const B = (label, abbr, wordClass, morphemeList, options = {}) =>
      new Button(label, abbr, wordClass, morphemeList, { ...options, dimensions: dim }, morphemes);

    // Morpheme-driven buttons
    buttons.set('RB', B('Rebuild', 'RB', 'Action', ['re', 'build'], {
      description: 'Recompose from parts (concretize)'
    }));

    buttons.set('UP', B('Upscale', 'UP', 'Action', ['multi'], {
      deltaLevel: 1,
      description: 'Scale up dimensions and complexity'
    }));

    buttons.set('CV', B('Convert', 'CV', 'Action', ['ize'], {
      description: 'Transform into different form'
    }));

    buttons.set('TL', B('Translucent', 'TL', 'Property', ['ness'], {
      deltaLevel: 1,
      beta: 0.1,
      description: 'Increase observability'
    }));

    buttons.set('MV', B('Move', 'MV', 'Action', ['move'], {
      description: 'Change position in space'
    }));

    buttons.set('SC', B('Scale', 'SC', 'Action', ['scale'], {
      description: 'Uniform scaling transformation'
    }));

    buttons.set('NG', B('Negate', 'NG', 'Action', ['un'], {
      description: 'Reverse or negate current state'
    }));

    buttons.set('CN', B('Counter', 'CN', 'Action', ['counter'], {
      description: 'Apply counteracting force'
    }));

    return buttons;
  }
}

export {
  LLEMath,
  Morpheme,
  Button,
  StationaryUnit,
  ScalingOperations,
  ButtonFactory
};
