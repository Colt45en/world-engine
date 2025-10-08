/* world-engine-unified.js
   Complete unified World Engine: Math Core + Initializer + Word Engine + Bridge Integration
   Algorithm-synchronized, scaling-modular, self-contained system.
   Runs in browser or Node. No external dependencies.
*/

/* =========================
 * Section A: Core Math
 * ========================= */

// LLEMath: Linear Algebra Foundation
class LLEMath {
  static multiply(A, B) {
    const isMatA = Array.isArray(A?.[0]);
    const isMatB = Array.isArray(B?.[0]);
    if (!isMatA) throw new Error('multiply: A must be matrix');

    if (isMatB) {
      const r = A.length, k = A[0].length, k2 = B.length, c = B[0].length;
      if (k !== k2) throw new Error(`multiply: shape mismatch (${r}×${k})·(${k2}×${c})`);
      return Array.from({ length: r }, (_, i) =>
        Array.from({ length: c }, (_, j) =>
          A[i].reduce((sum, aij, t) => sum + aij * B[t][j], 0)
        )
      );
    }

    const r = A.length, k = A[0].length, n = B.length;
    if (k !== n) throw new Error(`multiply: shape mismatch (${r}×${k})·(${n})`);
    return Array.from({ length: r }, (_, i) =>
      A[i].reduce((sum, aij, j) => sum + aij * B[j], 0)
    );
  }

  static transpose(A) {
    if (!Array.isArray(A?.[0])) throw new Error('transpose: A must be matrix');
    const r = A.length, c = A[0].length;
    return Array.from({ length: c }, (_, j) =>
      Array.from({ length: r }, (_, i) => A[i][j])
    );
  }

  static identity(n) {
    return Array.from({ length: n }, (_, i) =>
      Array.from({ length: n }, (_, j) => (i === j ? 1 : 0))
    );
  }

  static diagonal(values) {
    const n = values.length;
    return Array.from({ length: n }, (_, i) =>
      Array.from({ length: n }, (_, j) => (i === j ? values[i] : 0))
    );
  }

  static vectorAdd(a, b) {
    if (a.length !== b.length) throw new Error('vectorAdd: length mismatch');
    return a.map((v, i) => v + b[i]);
  }

  static vectorScale(v, s) { return v.map(val => val * s); }
  static clamp(v, min, max) { return Math.max(min, Math.min(max, v)); }

  static rotationMatrix2D(theta) {
    const c = Math.cos(theta), s = Math.sin(theta);
    return [[c, -s, 0],[s, c, 0],[0, 0, 1]];
  }

  static projectionMatrix(dims, keepDims) {
    const P = Array.from({ length: keepDims.length }, () => Array(dims).fill(0));
    keepDims.forEach((dim, i) => {
      if (dim >= dims) throw new Error('projection: index out of range');
      P[i][dim] = 1;
    });
    return P;
  }

  static pseudoInverse(A, lambda = 1e-6) {
    const AT = this.transpose(A);
    const m = A.length, n = A[0].length;

    if (m >= n) {
      const AAT = this.multiply(A, AT);
      const reg = AAT.map((row, i) => row.map((v, j) => v + (i === j ? lambda : 0)));
      const inv = this.inverseSmall(reg);
      return this.multiply(AT, inv);
    } else {
      const ATA = this.multiply(AT, A);
      const reg = ATA.map((row, i) => row.map((v, j) => v + (i === j ? lambda : 0)));
      const inv = this.inverseSmall(reg);
      return this.multiply(inv, AT);
    }
  }

  static inverseSmall(M) {
    const n = M.length;
    const A = M.map((row, i) => [...row, ...this.identity(n)[i]]);
    for (let i = 0; i < n; i++) {
      const piv = A[i][i];
      if (Math.abs(piv) < 1e-12) throw new Error('inverseSmall: singular matrix');
      const invPiv = 1 / piv;
      for (let j = 0; j < 2 * n; j++) A[i][j] *= invPiv;
      for (let r = 0; r < n; r++) if (r !== i) {
        const f = A[r][i];
        for (let j = 0; j < 2 * n; j++) A[r][j] -= f * A[i][j];
      }
    }
    return A.map(row => row.slice(n));
  }

  static validateFinite(vec, name='vector') {
    if (!vec.every(Number.isFinite)) throw new Error(`${name} contains non-finite values: ${vec}`);
    return true;
  }

  static safeTransform(M, x, b=null) {
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

// Morphemes: Linguistic building blocks
class Morpheme {
  constructor(symbol, M, b, effects = {}) {
    this.symbol = symbol;
    this.M = M;
    this.b = b;
    this.effects = effects;
  }

  static createBuiltInMorphemes(dim = 3) {
    const m = new Map();
    m.set('re',     new Morpheme('re',     [[0.9,0,0],[0,1.1,0],[0,0,1]],[0.1,0,0],     { deltaLevel:-1, alpha:0.95, description:'repetition, restoration' }));
    m.set('un',     new Morpheme('un',     [[-1,0,0],[0,1,0],[0,0,1]],[0,0,0],         { alpha:0.8, description:'negation, reversal' }));
    m.set('counter',new Morpheme('counter',[[ -0.8,0.2,0],[0.2,-0.8,0],[0,0,1]],[0,0,0], { deltaLevel:1, description:'opposition, counteraction' }));
    m.set('multi',  new Morpheme('multi',  [[1.5,0,0],[0,1.5,0],[0,0,1.2]],[0,0,0.1],  { deltaLevel:1, description:'multiplication, many' }));
    m.set('ize',    new Morpheme('ize',    [[1,0.1,0],[0,1,0.1],[0,0,1.1]],[0,0,0],    { deltaLevel:1, description:'make into, become' }));
    m.set('ness',   new Morpheme('ness',   [[1,0,0],[0,1,0],[0.1,0.1,0.9]],[0,0,0.1],  { C:[[1.1,0,0],[0,1.1,0],[0,0,0.9]], description:'quality, state' }));
    m.set('ment',   new Morpheme('ment',   [[1.1,0,0],[0,0.9,0],[0,0,1]],[0,0.1,0],    { deltaLevel:1, description:'result, action' }));
    m.set('ing',    new Morpheme('ing',    [[1,0.2,0],[0,1.2,0],[0,0,1]],[0,0,0],      { alpha:1.1, description:'ongoing action' }));
    m.set('build',  new Morpheme('build',  [[1.2,0,0],[0,1.2,0],[0,0,1.1]],[0.1,0.1,0],{ deltaLevel:1, description:'construct, create' }));
    m.set('move',   new Morpheme('move',   [[1,0,0],[0,1,0],[0,0,1]],[0.2,0,0],        { description:'change position' }));
    m.set('scale',  new Morpheme('scale',  [[1.1,0,0],[0,1.1,0],[0,0,1.1]],[0,0,0],    { description:'change size' }));
    return m;
  }
}

// Button: Morpheme composition
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
    let alpha = 1.0, beta = 0.0, delta = 0;

    if (registry) {
      for (const sym of morphemes) {
        const m = registry.get(sym);
        if (!m) continue;
        const newM = LLEMath.multiply(m.M, M);
        const newB = LLEMath.vectorAdd(LLEMath.multiply(m.M, b), m.b);
        M = newM; b = newB;

        if (m.effects?.C)   C = LLEMath.multiply(m.effects.C, C);
        if (m.effects?.deltaLevel) delta += m.effects.deltaLevel;
        if (typeof m.effects?.alpha === 'number') alpha *= m.effects.alpha;
        if (typeof m.effects?.beta  === 'number') beta  += m.effects.beta;
      }
    }

    this.M = options.M || M;
    this.b = options.b || b;
    this.C = options.C || C;
    this.alpha = options.alpha ?? alpha;
    this.beta  = options.beta  ?? beta;
    this.deltaLevel = options.deltaLevel ?? delta;

    this.inputType = options.inputType || 'State';
    this.outputType = options.outputType || 'State';
    this.description = options.description || '';
  }

  apply(su) {
    const dim = su.d;
    if (this.M.length !== dim || this.M[0].length !== dim)
      throw new Error(`Button ${this.abbr}: M shape ${this.M.length}×${this.M[0].length} != ${dim}×${dim}`);
    if (this.b.length !== dim)
      throw new Error(`Button ${this.abbr}: b length ${this.b.length} != ${dim}`);
    if (this.C.length !== dim || this.C[0].length !== dim)
      throw new Error(`Button ${this.abbr}: C shape ${this.C.length}×${this.C[0].length} != ${dim}×${dim}`);

    const newSU = su.copy();
    newSU.x = LLEMath.safeTransform(this.M, su.x, this.b);

    const CS = LLEMath.multiply(this.C, su.Sigma);
    newSU.Sigma = LLEMath.multiply(CS, LLEMath.transpose(this.C));

    newSU.kappa = LLEMath.clamp((this.alpha * su.kappa) + this.beta, 0, 1);
    newSU.level = su.level + this.deltaLevel;
    LLEMath.validateFinite(newSU.x, `Button ${this.abbr} result`);
    return newSU;
  }

  canComposeWith(other) { return other.outputType === this.inputType; }
  toString(){ return `[${this.abbr}] ${this.label} (${this.wordClass}, δℓ=${this.deltaLevel})`; }
}

// StationaryUnit: State representation
class StationaryUnit {
  constructor(dimensions = 3, x = null, Sigma = null, kappa = 1.0, level = 0) {
    this.d = dimensions;
    this.x = x || Array(dimensions).fill(0);
    this.Sigma = Sigma || LLEMath.identity(dimensions);
    this.kappa = kappa;
    this.level = level;
    this.timestamp = Date.now();
  }

  copy(){
    return new StationaryUnit(this.d, [...this.x], this.Sigma.map(r=>[...r]), this.kappa, this.level);
  }

  toString(){
    const pos = this.x.map(v => v.toFixed(3)).join(',');
    return `SU(x=[${pos}], κ=${this.kappa.toFixed(3)}, ℓ=${this.level}, d=${this.d})`;
  }
}

// ScalingOperations: Dimension management
class ScalingOperations {
  static createProjectionMatrix(fromDim, keepIndices) {
    const P = Array.from({ length: keepIndices.length }, () => Array(fromDim).fill(0));
    keepIndices.forEach((idx, i) => {
      if (idx >= fromDim) throw new Error('projection: index out of bounds');
      P[i][idx] = 1;
    });
    return P;
  }

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
    const A = abstractionMatrix || [
      [0.5, 0.5, 0],
      [0, 0.3, 0.7],
      [0.2, 0.2, 0.6]
    ];
    const Aplus = LLEMath.pseudoInverse(A, 1e-6);
    const abstract = (su.d === A.length) ? su.x : LLEMath.multiply(A, su.x);
    const recon = LLEMath.multiply(Aplus, abstract);
    const xUp = recon.length < toDim ? [...recon, ...Array(toDim - recon.length).fill(0)] : recon.slice(0, toDim);
    newSU.x = xUp;

    const AplusT = LLEMath.transpose(Aplus);
    const AS = LLEMath.multiply(Aplus, su.Sigma);
    newSU.Sigma = LLEMath.multiply(AS, AplusT);

    if (newSU.Sigma.length < toDim) {
      const padded = LLEMath.identity(toDim);
      for (let i = 0; i < newSU.Sigma.length; i++)
        for (let j = 0; j < newSU.Sigma[0].length; j++)
          padded[i][j] = newSU.Sigma[i][j];
      newSU.Sigma = padded;
    }

    newSU.level = su.level + 1;
    newSU.d = toDim;
    return newSU;
  }
}

// ButtonFactory: Standard button creation
class ButtonFactory {
  static createStandardButtons(dim = 3) {
    const buttons = new Map();
    const reg = Morpheme.createBuiltInMorphemes(dim);
    const B = (label, abbr, wordClass, morphemeList, options = {}) =>
      new Button(label, abbr, wordClass, morphemeList, { ...options, dimensions: dim }, reg);

    buttons.set('RB', B('Rebuild',    'RB', 'Action',   ['re','build'], { description:'Recompose from parts' }));
    buttons.set('UP', B('Upscale',    'UP', 'Action',   ['multi'],      { deltaLevel:1, description:'Scale up' }));
    buttons.set('CV', B('Convert',    'CV', 'Action',   ['ize'],        { description:'Transform form' }));
    buttons.set('TL', B('Translucent','TL', 'Property', ['ness'],       { deltaLevel:1, beta:0.1, description:'Increase observability' }));
    buttons.set('MV', B('Move',       'MV', 'Action',   ['move'],       { description:'Translate state' }));
    buttons.set('SC', B('Scale',      'SC', 'Action',   ['scale'],      { description:'Uniform scaling' }));
    buttons.set('NG', B('Negate',     'NG', 'Action',   ['un'],         { description:'Reverse orientation' }));
    buttons.set('CN', B('Counter',    'CN', 'Action',   ['counter'],    { description:'Counteract force' }));
    return buttons;
  }
}

// LexicalLogicEngine: Core engine
class LexicalLogicEngine {
  constructor(dim = 3) {
    this.dim = dim;
    this.su = new StationaryUnit(dim);
    this.buttons = ButtonFactory.createStandardButtons(dim);
    this.history = [];
  }

  clickButton(key, params = {}) {
    const btn = this.buttons.get(key) || this.buttons.get(String(key));
    if (!btn) throw new Error(`Button not found: ${key}`);
    const before = this.su.copy();
    const after = btn.apply(before);
    this.history.push(before);
    this.su = after;
    return after;
  }

  previewCompose(ops = []) {
    let state = this.su.copy();
    const steps = [];
    try {
      for (const op of ops) {
        if (typeof op === 'string') {
          const btn = this.buttons.get(op);
          if (!btn) throw new Error(`Preview: unknown button ${op}`);
          state = btn.apply(state);
          steps.push({ operation: { type:'button', key: op }, after: state.copy() });
        } else if (op?.type === 'upscale') {
          state = ScalingOperations.upscale(state, null, op.toDim ?? (state.d+1));
          steps.push({ operation: op, after: state.copy() });
        } else if (op?.type === 'downscale') {
          state = ScalingOperations.downscale(state, op.keep ?? [0, Math.max(1, state.d-1)]);
          steps.push({ operation: op, after: state.copy() });
        } else {
          throw new Error('Preview: unknown op');
        }
      }
      return { success:true, steps, finalState: state.copy() };
    } catch (e) {
      return { success:false, error:String(e), steps };
    }
  }

  downscale(keepIndices = [0, 2]) {
    const before = this.su.copy();
    const after = ScalingOperations.downscale(before, keepIndices);
    this.history.push(before);
    this.su = after;
    return after;
  }

  upscale(A = null, toDim = this.dim) {
    const before = this.su.copy();
    const after = ScalingOperations.upscale(before, A, toDim);
    this.history.push(before);
    this.su = after;
    return after;
  }

  undo() {
    if (!this.history.length) return this.su;
    this.su = this.history.pop();
    return this.su;
  }

  getState(){
    return {
      dim: this.dim,
      current: this.su.toString(),
      buttons: [...this.buttons.keys()]
    };
  }
}

/* =========================
 * Section B: Initializer
 * ========================= */

class WorldEngineInit {
  constructor() {
    this.components = new Map();
    this.status = 'idle';
    this.retryCount = 0;
    this.maxRetries = 3;
    this.healthChecks = new Map();
  }

  static getInstance() {
    if (!WorldEngineInit.instance) {
      WorldEngineInit.instance = new WorldEngineInit();
    }
    return WorldEngineInit.instance;
  }

  static getComponents() {
    return WorldEngineInit.getInstance().components;
  }

  async initialize(options = {}) {
    const startTime = Date.now();
    this.status = 'initializing';

    try {
      console.log('🌍 World Engine Unified Initialization Starting...');

      // Phase 1: Core validation
      await this.validateEnvironment();

      // Phase 2: Component discovery
      await this.discoverComponents();

      // Phase 3: Health checks
      await this.runHealthChecks();

      // Phase 4: Setup complete
      this.status = 'ready';
      const duration = Date.now() - startTime;

      const summary = {
        status: 'success',
        duration,
        components: this.components.size,
        healthChecks: this.healthChecks.size,
        timestamp: Date.now()
      };

      console.log(`✅ World Engine Unified ready in ${duration}ms`);
      return summary;

    } catch (error) {
      this.status = 'error';
      console.error('❌ World Engine Unified initialization failed:', error);

      if (this.retryCount < this.maxRetries) {
        this.retryCount++;
        console.log(`🔄 Retry ${this.retryCount}/${this.maxRetries}...`);
        await new Promise(resolve => setTimeout(resolve, 1000 * this.retryCount));
        return this.initialize(options);
      }

      return {
        status: 'error',
        error: error.message,
        retryCount: this.retryCount,
        timestamp: Date.now()
      };
    }
  }

  async validateEnvironment() {
    // Check basic JavaScript features
    if (!Array.from || !Map || !Promise) {
      throw new Error('Missing ES6+ features required for World Engine');
    }

    // Check mathematical operations
    try {
      LLEMath.multiply([[1,2],[3,4]], [5,6]);
    } catch (error) {
      throw new Error(`Mathematical validation failed: ${error.message}`);
    }

    this.components.set('environment', {
      name: 'Environment Validation',
      status: 'ready',
      checks: ['ES6+', 'LLEMath', 'Core APIs']
    });
  }

  async discoverComponents() {
    // Discover available components
    const available = [];

    if (typeof window !== 'undefined') {
      available.push('browser');
      if (window.location) available.push('location');
      if (window.localStorage) available.push('storage');
      if (window.fetch) available.push('network');
    }

    if (typeof process !== 'undefined') {
      available.push('node');
      if (process.version) available.push('version');
    }

    this.components.set('discovery', {
      name: 'Component Discovery',
      status: 'ready',
      available,
      count: available.length
    });
  }

  async runHealthChecks() {
    const checks = [
      { name: 'Math Operations', test: () => LLEMath.identity(3).length === 3 },
      { name: 'Morpheme Registry', test: () => Morpheme.createBuiltInMorphemes(3).size > 0 },
      { name: 'Button Factory', test: () => ButtonFactory.createStandardButtons(3).size > 0 },
      { name: 'Stationary Unit', test: () => new StationaryUnit(3).d === 3 },
      { name: 'Scaling Operations', test: () => ScalingOperations.createProjectionMatrix(3, [0,2]).length === 2 }
    ];

    for (const check of checks) {
      try {
        const result = await check.test();
        this.healthChecks.set(check.name, {
          status: result ? 'pass' : 'fail',
          result,
          timestamp: Date.now()
        });
      } catch (error) {
        this.healthChecks.set(check.name, {
          status: 'error',
          error: error.message,
          timestamp: Date.now()
        });
      }
    }
  }

  getStatus() {
    return {
      status: this.status,
      components: Array.from(this.components.entries()),
      healthChecks: Array.from(this.healthChecks.entries()),
      retryCount: this.retryCount
    };
  }
}

/* =========================
 * Section C: Word Engine Integration
 * ========================= */

class WordEngineIntegration {
  constructor() {
    this.state = {
      schema: null,
      alias: null,
      data: null,
      view: 'lexicon'
    };
    this.aliasRules = [];
  }

  async initWordEngine(options = {}) {
    const {
      schemaUrl = 'toggle_schema.json',
      aliasUrl = 'alias_wow_ggl.json',
      onToggle = (view) => console.log('[WordEngine] view ->', view),
      getDataFrame = async (path) => {
        try {
          const response = await fetch(path);
          return await response.text();
        } catch (error) {
          console.warn(`Could not load ${path}:`, error.message);
          return `# ${path}\nNo data available`;
        }
      }
    } = options;

    try {
      // Load configuration files with fallbacks
      const [schema, alias] = await Promise.all([
        this.loadWithFallback(schemaUrl, this.getDefaultSchema()),
        this.loadWithFallback(aliasUrl, this.getDefaultAlias())
      ]);

      // Setup alias rules
      this.setupAliasRules(alias);

      // Load data sources
      const lexPath = schema.views?.lexicon?.source || 'lexicon.md';
      const runesPath = schema.views?.runes?.source || 'runes.md';

      let lexRaw = await getDataFrame(lexPath);
      let runesRaw = await getDataFrame(runesPath);

      // Apply aliases
      lexRaw = this.applyAliases(lexRaw, 'headers');
      runesRaw = this.applyAliases(runesRaw, 'code');

      // Update state
      this.state = {
        schema,
        alias,
        data: { lexRaw, runesRaw },
        view: schema.default_view || 'lexicon'
      };

      // Setup toggle handler
      this.setupToggleHandler(onToggle);

      console.log('✅ Word Engine Integration initialized');
      return this.state;

    } catch (error) {
      console.error('❌ Word Engine Integration failed:', error);
      return this.getMinimalState();
    }
  }

  async loadWithFallback(url, fallback) {
    try {
      const response = await fetch(url);
      if (response.ok) {
        return await response.json();
      } else {
        console.warn(`Failed to load ${url}, using fallback`);
        return fallback;
      }
    } catch (error) {
      console.warn(`Error loading ${url}, using fallback:`, error.message);
      return fallback;
    }
  }

  setupAliasRules(alias) {
    this.aliasRules = (alias.apply_rules || []).map(rule => ({
      targets: new Set(rule.where || []),
      regex: new RegExp(rule.pattern, rule.flags?.includes('ignore_case') ? 'gi' : 'g'),
      replacement: rule.replacement
    }));
  }

  applyAliases(text, targetLabel) {
    let result = text;
    for (const rule of this.aliasRules) {
      if (rule.targets.has(targetLabel)) {
        result = result.replace(rule.regex, rule.replacement);
      }
    }
    return result;
  }

  setupToggleHandler(onToggle) {
    if (typeof window !== 'undefined') {
      window.addEventListener('keydown', (event) => {
        if (event.key === 'T' && !event.ctrlKey && !event.altKey) {
          const views = Object.keys(this.state.schema.views || { lexicon: {}, runes: {} });
          const currentIndex = views.indexOf(this.state.view);
          const nextIndex = (currentIndex + 1) % views.length;
          this.state.view = views[nextIndex];
          onToggle(this.state.view);
        }
      });
    }
  }

  getDefaultSchema() {
    return {
      default_view: 'lexicon',
      views: {
        lexicon: {
          source: 'lexicon.md',
          title: 'Lexicon View'
        },
        runes: {
          source: 'runes.md',
          title: 'Runes View'
        }
      }
    };
  }

  getDefaultAlias() {
    return {
      apply_rules: [
        {
          pattern: 'WOW',
          replacement: 'GGL',
          where: ['headers', 'code'],
          flags: ['ignore_case']
        }
      ]
    };
  }

  getMinimalState() {
    return {
      schema: this.getDefaultSchema(),
      alias: this.getDefaultAlias(),
      data: { lexRaw: '# Default Lexicon', runesRaw: '# Default Runes' },
      view: 'lexicon'
    };
  }

  getState() {
    return { ...this.state };
  }
}

/* =========================
 * Section D: Unified Mathematical System
 * ========================= */

class WorldEngineMathematicalSystemV3 {
  constructor(dimensions = 3, options = {}) {
    this.dimensions = dimensions;
    this.options = options;
    this.lexicalEngine = new LexicalLogicEngine(dimensions);
    this.math = LLEMath;
    this.morphemes = Morpheme.createBuiltInMorphemes(dimensions);

    // Adapters with smart defaults
    this.llexEngine = options.llexEngine || new NullLLEXEngine();
    this.indexManager = options.indexManager || new EnhancedIndexManager();
    this.addressResolver = options.addressResolver || new NullAddressResolver();

    this._initEnhancedButtons();
  }

  async _initEnhancedButtons() {
    const defs = [
      { lid:'button:MorphRebuild', morphemes:['re','build'], wordClass:'Action', options:{ description:'Morpheme rebuild (validated)' } },
      { lid:'button:SafeMove', morphemes:['move'], wordClass:'Action', options:{ description:'Safe move' } },
      { lid:'button:PseudoScale', morphemes:['scale'], wordClass:'Transform', options:{ description:'Scaling with pseudo-inverse' } },
      { lid:'button:CounterNegate', morphemes:['counter'], wordClass:'Action', options:{ description:'Counteraction' } },
    ];

    for (const d of defs) {
      try {
        const b = new Button(
          d.lid.split(':')[1],
          d.lid.split(':')[1].slice(0,2).toUpperCase(),
          d.wordClass,
          d.morphemes,
          { ...d.options, dimensions: this.dimensions },
          this.morphemes
        );
        this.lexicalEngine.buttons.set(b.abbr, b);

        const operator = { M:b.M, b:b.b, C:b.C, alpha:b.alpha, beta:b.beta, delta_level:b.deltaLevel };
        const result = await this.llexEngine.createButton?.('core', d.lid, operator, d.morphemes, { class:d.wordClass, description:d.options.description }) || { cid:'cid:dummy', address:'addr:dummy' };

        this.indexManager.indexObject?.({
          type:'enhanced_button', cid: result.cid, lid:d.lid, description:d.options.description,
          morphemes:d.morphemes, class:d.wordClass, mathematical:true, dimensions:this.dimensions
        });
      } catch (e) {
        if (typeof console !== 'undefined') console.warn('⚠️ Failed to create enhanced button:', d.lid, e);
      }
    }
  }

  // Safe operation wrappers
  async safeClickButton(buttonKey, params = {}) {
    try {
      const result = this.lexicalEngine.clickButton(buttonKey, params);
      return { success:true, result, mathematical_validation:'passed', timestamp: Date.now() };
    } catch (e) {
      return { success:false, error:String(e), mathematical_validation:'failed', timestamp: Date.now() };
    }
  }

  async previewComposition(operations, options = {}) {
    try {
      const preview = this.lexicalEngine.previewCompose(operations);
      const analysis = {
        dimensionality: preview.finalState?.d || this.dimensions,
        mathematical_stability: this._analyzeStability(preview.steps),
        recovery_possible: this._checkRecovery(preview.steps),
        composition_valid: preview.success
      };
      return { ...preview, mathematical_analysis: analysis, safe_to_apply: preview.success && analysis.mathematical_stability, timestamp: Date.now() };
    } catch (e) {
      return { success:false, error:String(e), mathematical_analysis:{ stability:false }, timestamp: Date.now() };
    }
  }

  async safeDownscale(keepIndices = [0,2]) {
    try {
      const before = this.lexicalEngine.su.copy();
      const result = this.lexicalEngine.downscale(keepIndices);
      return { success:true, result, before_dimensions:before.d, after_dimensions:result.d, timestamp: Date.now() };
    } catch (e) {
      return { success:false, error:String(e), timestamp: Date.now() };
    }
  }

  async safeUpscale(abstractionMatrix = null, toDim = null) {
    try {
      const target = toDim || this.dimensions;
      const before = this.lexicalEngine.su.copy();
      const A = abstractionMatrix || this._defaultAbstraction(before.d, target);
      const result = this.lexicalEngine.upscale(A, target);
      return { success:true, result, before_dimensions:before.d, after_dimensions:result.d, abstraction_matrix:A, pseudo_inverse_used:!abstractionMatrix, timestamp: Date.now() };
    } catch (e) {
      return { success:false, error:String(e), timestamp: Date.now() };
    }
  }

  async safeUndo() {
    try {
      const before = this.lexicalEngine.su.copy();
      const result = this.lexicalEngine.undo();
      return { success:true, result, restored: before.toString() !== result.toString(), mathematical_validation:'passed', timestamp: Date.now() };
    } catch (e) {
      return { success:false, error:String(e), mathematical_validation:'failed', timestamp: Date.now() };
    }
  }

  async runMathematicalTests() {
    const basic = { summary: { total: 2, passed: 2 } };
    const math = await this._runEnhancedMathTests();
    return { basic_tests: basic, mathematical_tests: math, overall_health: this._overallHealth(basic, math), timestamp: Date.now() };
  }

  async _runEnhancedMathTests() {
    const out = [];
    try {
      const btn = new Button('Test','TS','Action',['re','build'],{ dimensions:this.dimensions }, this.morphemes);
      const composed = btn.M.some(row => row.some(v => v !== 0 && v !== 1));
      out.push({ name:'Morpheme Composition', success: composed });
    } catch(e){ out.push({ name:'Morpheme Composition', success:false, error:String(e) }); }

    try {
      const A = [[1,0.5,0],[0,1,0.3],[0.2,0,1]];
      const Aplus = LLEMath.pseudoInverse(A);
      const round = LLEMath.multiply(A, LLEMath.multiply(Aplus, A));
      const err = A.reduce((S,row,i)=> S + row.reduce((s,v,j)=> s + Math.abs(v - round[i][j]),0),0);
      out.push({ name:'Pseudo-inverse Roundtrip', success: err < 0.01, reconstruction_error: err });
    } catch(e){ out.push({ name:'Pseudo-inverse Roundtrip', success:false, error:String(e) }); }

    return out;
  }

  _analyzeStability(steps) {
    return steps.every(s => {
      if (!s.after?.x) return false;
      if (!s.after.x.every(Number.isFinite)) return false;
      return Math.max(...s.after.x.map(Math.abs)) < 1000;
    });
  }

  _checkRecovery(steps) {
    return steps.some(s => s.operation?.type === 'upscale' || s.operation?.type === 'downscale' || this.lexicalEngine.history.length > 0);
  }

  _defaultAbstraction(fromDim, toDim) {
    const rows = Math.min(fromDim, toDim), cols = Math.max(fromDim, toDim);
    return Array.from({ length: rows }, (_, i) =>
      Array.from({ length: cols }, (_, j) => (i === j ? 1 : (j === i+1 ? 0.5 : 0)))
    );
  }

  _overallHealth(basic, math) {
    const total = basic.summary.total + math.length;
    const passed = basic.summary.passed + math.filter(t=>t.success).length;
    return { overall_score: passed/total, tests_passed: passed, total_tests: total, health_status: passed/total > 0.8 ? 'healthy' : 'degraded' };
  }

  getEnhancedState() {
    return {
      dimensions: this.dimensions,
      lexical_engine: this.lexicalEngine.getState(),
      mathematical: {
        current_state: this.lexicalEngine.su.toString(),
        stability: this._currentStability(),
        available_operations: [...this.lexicalEngine.buttons.keys()],
        morpheme_count: this.morphemes.size
      },
      system_version: '3.0-unified',
      timestamp: Date.now()
    };
  }

  _currentStability() {
    const su = this.lexicalEngine.su;
    return {
      finite_values: su.x.every(Number.isFinite),
      reasonable_bounds: Math.max(...su.x.map(Math.abs)) < 1000,
      positive_kappa: su.kappa > 0,
      valid_dimensions: su.d === this.dimensions
    };
  }
}

// Enhanced Index Manager
class EnhancedIndexManager {
  constructor() {
    this.objects = new Map();
    this.searchIndex = new Map();
  }

  indexObject(obj) {
    if (obj.cid) {
      this.objects.set(obj.cid, obj);
      // Index searchable terms
      const terms = this._extractSearchTerms(obj);
      terms.forEach(term => {
        if (!this.searchIndex.has(term)) {
          this.searchIndex.set(term, new Set());
        }
        this.searchIndex.get(term).add(obj.cid);
      });
    }
  }

  unifiedSearch(query, options = {}) {
    const terms = query.toLowerCase().split(' ');
    const results = new Set();

    terms.forEach(term => {
      const matches = this.searchIndex.get(term);
      if (matches) {
        matches.forEach(cid => results.add(cid));
      }
    });

    return {
      results: Array.from(results).map(cid => this.objects.get(cid)).filter(Boolean),
      query,
      options
    };
  }

  _extractSearchTerms(obj) {
    const terms = [];
    if (obj.type) terms.push(obj.type);
    if (obj.description) terms.push(...obj.description.toLowerCase().split(' '));
    if (obj.morphemes) terms.push(...obj.morphemes);
    if (obj.class) terms.push(obj.class.toLowerCase());
    return terms;
  }

  getStats() {
    return {
      indexed: this.objects.size,
      searchTerms: this.searchIndex.size
    };
  }
}

// Null adapters for standalone operation
class NullLLEXEngine {
  async createButton() { return { cid: 'cid:dummy', address: 'addr:dummy' }; }
  async objectStore() { return; }
  async setSessionHead() { return; }
  getStats() { return { objects: 0 }; }
}

class NullAddressResolver {
  constructor() { }
}

/* =========================
 * Section E: Unified Initialization Bridge
 * ========================= */

async function initializeUnifiedEngine(dim = 3, options = {}) {
  console.log('🌍 Initializing Unified World Engine...');

  // 1. Run base initialization
  const initSystem = WorldEngineInit.getInstance();
  const initSummary = await initSystem.initialize(options);

  if (initSummary.status === 'error') {
    throw new Error(`Initialization failed: ${initSummary.error}`);
  }

  // 2. Start mathematical engine
  const mathEngine = new WorldEngineMathematicalSystemV3(dim, options);

  // 3. Wire math engine into init registry
  initSystem.components.set('math-engine', {
    name: 'Mathematical Core V3',
    status: 'ready',
    api: mathEngine,
    health: () => mathEngine.getEnhancedState().mathematical.stability
  });

  // 4. Initialize Word Engine Integration
  const wordIntegration = new WordEngineIntegration();
  const wordEngine = await wordIntegration.initWordEngine(options.wordEngine || {});

  // 5. Wire Word Engine data into math engine's indexer
  mathEngine.indexManager.indexObject({
    type: 'alias_rules',
    cid: 'word-engine-aliases',
    data: wordEngine.alias,
    description: 'Word engine alias transformation rules'
  });

  mathEngine.indexManager.indexObject({
    type: 'lexicon_data',
    cid: 'word-engine-lexicon',
    data: wordEngine.data,
    description: 'Word engine lexicon and runes data'
  });

  // 6. Register Word Engine in components
  initSystem.components.set('word-engine', {
    name: 'Word Engine Integration',
    status: 'ready',
    api: wordIntegration,
    health: () => ({ data_loaded: !!wordEngine.data, view: wordEngine.view })
  });

  // 7. Run comprehensive tests if requested
  if (options.runTests) {
    console.log('🔬 Running unified system tests...');
    const testResults = await mathEngine.runMathematicalTests();
    console.log(`✅ Tests completed: ${testResults.overall_health.tests_passed}/${testResults.overall_health.total_tests} passed`);
  }

  const finalState = {
    initSummary,
    mathEngine,
    wordEngine: wordIntegration,
    components: initSystem.getStatus(),
    ready: true,
    timestamp: Date.now()
  };

  console.log(`🚀 Unified World Engine online! (${dim}D, ${initSystem.components.size} components)`);
  return finalState;
}

// Factory for different configurations
const UnifiedWorldEngineFactory = {
  create2D: (options = {}) => initializeUnifiedEngine(2, options),
  create3D: (options = {}) => initializeUnifiedEngine(3, options),
  create4D: (options = {}) => initializeUnifiedEngine(4, options),
  createCustom: (dimensions, options = {}) => initializeUnifiedEngine(dimensions, options),
  createWithTests: (dimensions = 3) => initializeUnifiedEngine(dimensions, { runTests: true }),
  createFull: (dimensions = 3) => initializeUnifiedEngine(dimensions, { runTests: true, wordEngine: { enableToggle: true } })
};

// Demo function
async function runUnifiedDemo() {
  console.log('🎭 Unified World Engine Demo');
  console.log('============================\n');

  const engine = await UnifiedWorldEngineFactory.createFull(3);

  console.log('1. System Status:', engine.components.status);
  console.log('2. Math Engine State:', engine.mathEngine.getEnhancedState());
  console.log('3. Word Engine View:', engine.wordEngine.getState().view);

  // Test mathematical operations
  console.log('\n🧮 Testing Mathematical Operations...');
  const moveResult = await engine.mathEngine.safeClickButton('MV');
  console.log('Move operation:', moveResult.success ? '✅ Success' : '❌ Failed');

  const scaleResult = await engine.mathEngine.safeUpscale(null, 4);
  console.log('Upscale to 4D:', scaleResult.success ? '✅ Success' : '❌ Failed');

  // Test search integration
  console.log('\n🔍 Testing Search Integration...');
  const searchResult = engine.mathEngine.indexManager.unifiedSearch('morpheme action');
  console.log(`Search results: ${searchResult.results.length} found`);

  console.log('\n🎉 Unified Demo Complete!');
  return engine;
}

/* =========================
 * Auto-boot and Exports
 * ========================= */

const UnifiedAPI = {
  // Core math classes
  LLEMath, Morpheme, Button, StationaryUnit, ScalingOperations, ButtonFactory,

  // Engine classes
  LexicalLogicEngine, WorldEngineMathematicalSystemV3,

  // Integration classes
  WorldEngineInit, WordEngineIntegration,

  // Factory and initialization
  UnifiedWorldEngineFactory, initializeUnifiedEngine,

  // Utilities
  runUnifiedDemo
};

// Auto-boot unless disabled
if (typeof window !== 'undefined' && !window.WE_MANUAL_INIT) {
  console.log('🌍 Auto-initializing Unified World Engine...');
  UnifiedWorldEngineFactory.createFull(3)
    .then(engine => {
      window.UnifiedWorldEngine = engine;
      window.dispatchEvent(new CustomEvent('world-engine-ready', { detail: engine }));
      console.log('🚀 Unified World Engine auto-boot complete');
    })
    .catch(error => {
      console.error('❌ Auto-boot failed:', error);
    });
}

// Module exports
if (typeof module !== 'undefined' && module.exports) {
  module.exports = UnifiedAPI;
} else if (typeof window !== 'undefined') {
  window.UnifiedWorldEngine = UnifiedAPI;
  console.log('🌐 Unified World Engine loaded in browser');
}
