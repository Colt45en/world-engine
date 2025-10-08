/**
 * World Engine Types Lattice - JavaScript Edition
 * Complete type system with composition checking and lattice operations
 * Version: 1.0.0
 */

// ===== Core Type System =====

/**
 * @typedef {'void'|'any'|'unknown'|'string'|'number'|'boolean'|'array'|'object'|'function'|'morpheme'|'lexeme'|'token'|'vector2d'|'vector3d'|'matrix'|'analysis'|'result'|'composite'} TypeName
 */

/**
 * @typedef {Object} TypeNode
 * @property {TypeName} name
 * @property {TypeName[]} supertypes
 * @property {TypeName[]} subtypes
 * @property {Object} properties
 * @property {Object} meta
 * @property {number} meta.created
 * @property {number} meta.rank
 * @property {TypeName[]} meta.compatibility
 */

// ===== Type Lattice Implementation =====

class WorldEngineTypeLattice {
  constructor() {
    /** @type {Map<TypeName, TypeNode>} */
    this.nodes = new Map();

    this.initializeBasicTypes();
    this.initializeWorldEngineTypes();
    this.initializeMathematicalTypes();
    this.initializeCompositeTypes();
  }

  initializeBasicTypes() {
    // Foundation types
    this.addType('void', [], ['any'], { primitive: true });
    this.addType('any', ['void'], [], { top: true });
    this.addType('unknown', ['void'], ['any'], { safe: true });

    // Basic data types
    this.addType('string', ['unknown'], ['token', 'lexeme'], { serializable: true });
    this.addType('number', ['unknown'], ['vector2d', 'vector3d'], { numeric: true });
    this.addType('boolean', ['unknown'], [], { logical: true });
    this.addType('array', ['unknown'], ['matrix', 'vector2d', 'vector3d'], { container: true });
    this.addType('object', ['unknown'], ['morpheme', 'analysis', 'result'], { structured: true });
    this.addType('function', ['unknown'], [], { executable: true });
  }

  initializeWorldEngineTypes() {
    // Linguistic types
    this.addType('token', ['string'], ['lexeme'], { linguistic: true, atomic: true });
    this.addType('lexeme', ['token'], ['morpheme'], { linguistic: true, meaningful: true });
    this.addType('morpheme', ['object', 'lexeme'], ['composite'], {
      linguistic: true,
      transformable: true,
      components: ['symbol', 'matrix', 'bias']
    });
  }

  initializeMathematicalTypes() {
    // Mathematical types
    this.addType('vector2d', ['array', 'number'], ['vector3d'], {
      mathematical: true,
      dimensions: 2,
      operations: ['add', 'scale', 'dot']
    });
    this.addType('vector3d', ['vector2d'], ['matrix'], {
      mathematical: true,
      dimensions: 3,
      operations: ['add', 'scale', 'dot', 'cross']
    });
    this.addType('matrix', ['array', 'vector3d'], [], {
      mathematical: true,
      dimensions: 'variable',
      operations: ['multiply', 'transpose', 'inverse']
    });
  }

  initializeCompositeTypes() {
    // Analysis types
    this.addType('analysis', ['object'], ['result'], {
      worldengine: true,
      components: ['input', 'process', 'output']
    });
    this.addType('result', ['object', 'analysis'], ['composite'], {
      worldengine: true,
      components: ['outcome', 'confidence', 'metadata']
    });
    this.addType('composite', ['object', 'morpheme', 'result'], [], {
      worldengine: true,
      complex: true,
      combinable: true
    });
  }

  /**
   * @param {TypeName} name
   * @param {TypeName[]} supertypes
   * @param {TypeName[]} subtypes
   * @param {Object} properties
   */
  addType(name, supertypes, subtypes, properties) {
    this.nodes.set(name, {
      name,
      supertypes,
      subtypes,
      properties,
      meta: {
        created: Date.now(),
        rank: this.calculateRank(supertypes),
        compatibility: this.calculateCompatibility(name, supertypes, properties)
      }
    });
  }

  /**
   * @param {TypeName[]} supertypes
   * @returns {number}
   */
  calculateRank(supertypes) {
    if (supertypes.length === 0) return 0;
    return Math.max(...supertypes.map(t => (this.nodes.get(t)?.meta.rank ?? 0) + 1));
  }

  /**
   * @param {TypeName} name
   * @param {TypeName[]} supertypes
   * @param {Object} properties
   * @returns {TypeName[]}
   */
  calculateCompatibility(name, supertypes, properties) {
    const compatible = new Set([name, ...supertypes]);

    // Add mathematical compatibility
    if (properties.mathematical) {
      compatible.add('number');
      compatible.add('array');
    }

    // Add linguistic compatibility
    if (properties.linguistic) {
      compatible.add('string');
      compatible.add('object');
    }

    return Array.from(compatible);
  }

  /**
   * Check if two types can be composed (outA -> inB)
   * @param {TypeName} outA
   * @param {TypeName} inB
   * @returns {boolean}
   */
  checkCompose(outA, inB) {
    const nodeA = this.nodes.get(outA);
    const nodeB = this.nodes.get(inB);

    if (!nodeA || !nodeB) return false;

    // Direct compatibility check
    if (nodeA.meta.compatibility.includes(inB) ||
        nodeB.meta.compatibility.includes(outA)) {
      return true;
    }

    // Subtyping check
    if (this.isSubtype(outA, inB) || this.isSubtype(inB, outA)) {
      return true;
    }

    // Mathematical composition rules
    if (this.checkMathematicalComposition(nodeA, nodeB)) {
      return true;
    }

    // World Engine specific rules
    if (this.checkWorldEngineComposition(nodeA, nodeB)) {
      return true;
    }

    return false;
  }

  /**
   * @param {TypeNode} nodeA
   * @param {TypeNode} nodeB
   * @returns {boolean}
   */
  checkMathematicalComposition(nodeA, nodeB) {
    const mathA = nodeA.properties.mathematical;
    const mathB = nodeB.properties.mathematical;

    if (!mathA && !mathB) return false;

    // Vector-Matrix compositions
    if (nodeA.name === 'matrix' && (nodeB.name === 'vector2d' || nodeB.name === 'vector3d')) {
      return true;
    }

    // Vector-Vector compositions
    if ((nodeA.name === 'vector2d' || nodeA.name === 'vector3d') &&
        (nodeB.name === 'vector2d' || nodeB.name === 'vector3d')) {
      return true;
    }

    return false;
  }

  /**
   * @param {TypeNode} nodeA
   * @param {TypeNode} nodeB
   * @returns {boolean}
   */
  checkWorldEngineComposition(nodeA, nodeB) {
    const weA = nodeA.properties.worldengine;
    const weB = nodeB.properties.worldengine;
    const lingA = nodeA.properties.linguistic;
    const lingB = nodeB.properties.linguistic;

    // World Engine analysis can compose with linguistic types
    if (weA && lingB) return true;
    if (lingA && weB) return true;

    // Morphemes can compose with mathematical types
    if (nodeA.name === 'morpheme' && nodeB.properties.mathematical) {
      return true;
    }

    return false;
  }

  /**
   * @param {TypeName} type
   * @returns {TypeName[]}
   */
  getSupertypes(type) {
    const node = this.nodes.get(type);
    if (!node) return [];

    const supertypes = new Set(node.supertypes);

    // Recursively add supertypes of supertypes
    for (const supertype of node.supertypes) {
      this.getSupertypes(supertype).forEach(s => supertypes.add(s));
    }

    return Array.from(supertypes);
  }

  /**
   * @param {TypeName} type
   * @returns {TypeName[]}
   */
  getSubtypes(type) {
    const node = this.nodes.get(type);
    if (!node) return [];

    const subtypes = new Set(node.subtypes);

    // Recursively add subtypes of subtypes
    for (const subtype of node.subtypes) {
      this.getSubtypes(subtype).forEach(s => subtypes.add(s));
    }

    return Array.from(subtypes);
  }

  /**
   * @param {TypeName} source
   * @param {TypeName} target
   * @returns {boolean}
   */
  isCompatible(source, target) {
    return this.checkCompose(source, target);
  }

  /**
   * @param {TypeName} subtype
   * @param {TypeName} supertype
   * @returns {boolean}
   */
  isSubtype(subtype, supertype) {
    return this.getSupertypes(subtype).includes(supertype);
  }

  /**
   * @param {TypeName[]} types
   * @returns {TypeName | null}
   */
  findCommonSupertype(types) {
    if (types.length === 0) return null;
    if (types.length === 1) return types[0];

    // Get all supertypes of the first type
    const commonSupertypes = new Set(this.getSupertypes(types[0]));
    commonSupertypes.add(types[0]);

    // Intersect with supertypes of other types
    for (let i = 1; i < types.length; i++) {
      const typeSupertypes = new Set(this.getSupertypes(types[i]));
      typeSupertypes.add(types[i]);

      // Keep only common supertypes
      for (const supertype of commonSupertypes) {
        if (!typeSupertypes.has(supertype)) {
          commonSupertypes.delete(supertype);
        }
      }
    }

    if (commonSupertypes.size === 0) return null;

    // Return the most specific common supertype (highest rank)
    let bestSupertype = null;
    let bestRank = -1;

    for (const supertype of commonSupertypes) {
      const rank = this.nodes.get(supertype)?.meta.rank ?? 0;
      if (rank > bestRank) {
        bestRank = rank;
        bestSupertype = supertype;
      }
    }

    return bestSupertype;
  }

  /**
   * @param {TypeName} from
   * @param {TypeName} to
   * @returns {boolean}
   */
  canTransform(from, to) {
    if (from === to) return true;

    const fromNode = this.nodes.get(from);
    const toNode = this.nodes.get(to);

    if (!fromNode || !toNode) return false;

    // Check if transformation is possible through subtyping
    if (this.isSubtype(from, to) || this.isSubtype(to, from)) {
      return true;
    }

    // Check if both types have transformable properties
    if (fromNode.properties.transformable && toNode.properties.transformable) {
      return true;
    }

    // Mathematical transformations
    if (fromNode.properties.mathematical && toNode.properties.mathematical) {
      return true;
    }

    return false;
  }

  // ===== Utility Methods =====

  /**
   * @param {TypeName} type
   * @returns {TypeNode | null}
   */
  getTypeInfo(type) {
    return this.nodes.get(type) ?? null;
  }

  /**
   * @returns {TypeName[]}
   */
  getAllTypes() {
    return Array.from(this.nodes.keys());
  }

  /**
   * @param {TypeName[]} chain
   * @returns {{ valid: boolean; error?: string }}
   */
  validateTypeChain(chain) {
    if (chain.length < 2) return { valid: true };

    for (let i = 0; i < chain.length - 1; i++) {
      if (!this.checkCompose(chain[i], chain[i + 1])) {
        return {
          valid: false,
          error: `Cannot compose ${chain[i]} -> ${chain[i + 1]} at position ${i}`
        };
      }
    }

    return { valid: true };
  }

  /**
   * @param {TypeName} type
   * @returns {TypeName[]}
   */
  suggestCompositions(type) {
    const suggestions = [];

    for (const [candidateType] of this.nodes) {
      if (candidateType !== type && this.checkCompose(type, candidateType)) {
        suggestions.push(candidateType);
      }
    }

    return suggestions;
  }
}

// ===== Create global instance =====
const typeLattice = new WorldEngineTypeLattice();

// ===== Enhanced Store with Type Safety =====
const TypedStore = {
  /**
   * @param {string} key
   * @param {any} value
   * @param {TypeName} [type]
   * @returns {Promise<boolean>}
   */
  async save(key, value, type) {
    try {
      if (window.externalStore?.upsert) {
        const payload = type ? { value, type, timestamp: Date.now() } : value;
        await window.externalStore.upsert(key, payload);
        return true;
      } else if (window.localStorage) {
        const payload = type ? { value, type, timestamp: Date.now() } : value;
        window.localStorage.setItem(key, JSON.stringify(payload));
        return true;
      }
      throw new Error('No storage available');
    } catch {
      return false;
    }
  },

  /**
   * @param {string} key
   * @param {TypeName} [expectedType]
   * @returns {Promise<any>}
   */
  async load(key, expectedType) {
    try {
      if (window.externalStore?.get) {
        const result = await window.externalStore.get(key);
        return this.validateAndExtract(result, expectedType);
      } else if (window.localStorage) {
        const item = window.localStorage.getItem(key);
        if (item) {
          const parsed = JSON.parse(item);
          return this.validateAndExtract(parsed, expectedType);
        }
      }
      return null;
    } catch {
      return null;
    }
  },

  /**
   * @param {any} data
   * @param {TypeName} [expectedType]
   * @returns {any}
   */
  validateAndExtract(data, expectedType) {
    if (!expectedType) return data;

    if (data && typeof data === 'object' && 'type' in data && 'value' in data) {
      if (typeLattice.isCompatible(data.type, expectedType)) {
        return data.value;
      } else {
        console.warn(`Type mismatch: expected ${expectedType}, got ${data.type}`);
        return null;
      }
    }

    return data;
  }
};

// ===== Enhanced Transport with Type Safety =====
function setupTypedEngineTransport(engineFrame) {
  const isSameOrigin = () => {
    try {
      void engineFrame.contentWindow.document;
      return true;
    } catch {
      return false;
    }
  };

  const withEngine = (fn) => {
    if (isSameOrigin()) {
      const doc = engineFrame.contentWindow.document;
      return fn(doc);
    } else {
      throw new Error('Cross-origin detected, use postMessage transport');
    }
  };

  const sendTypedMessage = (message, type) => {
    const typedMessage = {
      ...message,
      __type: type,
      __timestamp: Date.now()
    };

    if (isSameOrigin()) {
      // Direct access for same-origin
      withEngine(doc => {
        const event = new CustomEvent('studio:typed:message', { detail: typedMessage });
        doc.dispatchEvent(event);
      });
    } else {
      // PostMessage for cross-origin
      engineFrame.contentWindow?.postMessage(typedMessage, '*');
    }
  };

  return { isSameOrigin, withEngine, sendTypedMessage, typeLattice };
}

// ===== Enhanced Utilities with Type Support =====
const TypedUtils = {
  generateId: () => String(Date.now()) + Math.random().toString(36).slice(2, 11),

  /**
   * @param {string} line
   * @returns {{ type: string; args: string; dataType?: TypeName }}
   */
  parseCommand: (line) => {
    const t = line.trim();

    // Enhanced parsing with type hints
    if (t.startsWith('/run ')) {
      const args = t.slice(5);
      const typeMatch = args.match(/--type=(\w+)/);
      const dataType = typeMatch ? typeMatch[1] : 'string';
      return {
        type: 'run',
        args: args.replace(/--type=\w+\s*/, ''),
        dataType
      };
    }

    if (t.startsWith('/test ')) return { type: 'test', args: t.slice(6), dataType: 'analysis' };
    if (t === '/rec start') return { type: 'rec', args: 'start', dataType: 'void' };
    if (t === '/rec stop') return { type: 'rec', args: 'stop', dataType: 'result' };
    if (t.startsWith('/mark ')) return { type: 'mark', args: t.slice(6), dataType: 'string' };

    return { type: 'run', args: t, dataType: 'string' };
  },

  /**
   * @param {string} message
   * @param {'info'|'warn'|'error'|'debug'} level
   * @param {TypeName} [context]
   */
  log: (message, level = 'info', context) => {
    const timestamp = new Date().toISOString();
    const prefix = context ? `[Studio:${level}:${context}]` : `[Studio:${level}]`;
    (console[level] ?? console.log)(`${prefix} ${timestamp} - ${message}`);
  },

  /**
   * @param {any} value
   * @param {TypeName} expectedType
   * @returns {boolean}
   */
  validateType: (value, expectedType) => {
    return typeLattice.isCompatible('unknown', expectedType); // Simplified for now
  },

  /**
   * @param {any} value
   * @returns {TypeName}
   */
  inferType: (value) => {
    if (value === null || value === undefined) return 'void';
    if (typeof value === 'string') return 'string';
    if (typeof value === 'number') return 'number';
    if (typeof value === 'boolean') return 'boolean';
    if (Array.isArray(value)) {
      if (value.length === 2 && value.every(v => typeof v === 'number')) return 'vector2d';
      if (value.length === 3 && value.every(v => typeof v === 'number')) return 'vector3d';
      return 'array';
    }
    if (typeof value === 'object') {
      if ('symbol' in value && 'M' in value && 'b' in value) return 'morpheme';
      if ('outcome' in value) return 'result';
      return 'object';
    }
    return 'unknown';
  }
};

// ===== Integration with Studio Bridge =====
if (typeof window !== 'undefined') {
  // Enhanced Studio Bridge with type system
  window.StudioBridge = {
    ...window.StudioBridge,
    typeLattice,
    TypedStore,
    TypedUtils,
    setupTypedEngineTransport,

    // Enhanced versions of existing APIs
    onBus: window.StudioBridge?.onBus || (() => {}),
    sendBus: window.StudioBridge?.sendBus || (() => {}),

    // Type-aware message handling
    sendTypedBus: (msg, type) => {
      const typedMsg = { ...msg, __type: type, __timestamp: Date.now() };
      if (window.StudioBridge?.sendBus) {
        window.StudioBridge.sendBus(typedMsg);
      }
    },

    onTypedBus: (fn, expectedType) => {
      if (window.StudioBridge?.onBus) {
        return window.StudioBridge.onBus((msg) => {
          if (expectedType && msg.__type && !typeLattice.isCompatible(msg.__type, expectedType)) {
            console.warn(`Type mismatch in message: expected ${expectedType}, got ${msg.__type}`);
            return;
          }
          fn(msg);
        });
      }
      return () => {};
    }
  };
}

// ===== Debug Helper with Type Information =====
if (typeof window !== 'undefined' && window.location?.search.includes('debug=types')) {
  console.log('🔬 World Engine Type Lattice Debug Mode');
  console.log('Available types:', typeLattice.getAllTypes());
  console.log('Type lattice:', typeLattice);

  // Test some compositions
  console.log('Composition tests:');
  console.log('string -> token:', typeLattice.checkCompose('string', 'token'));
  console.log('morpheme -> matrix:', typeLattice.checkCompose('morpheme', 'matrix'));
  console.log('vector3d -> matrix:', typeLattice.checkCompose('vector3d', 'matrix'));

  // Add global debugging helpers
  window.debugTypes = {
    lattice: typeLattice,
    testComposition: (a, b) => typeLattice.checkCompose(a, b),
    getSupertypes: (type) => typeLattice.getSupertypes(type),
    getSubtypes: (type) => typeLattice.getSubtypes(type),
    inferType: (value) => TypedUtils.inferType(value)
  };
}

// ===== Smoke Test for Type System =====
function runTypeLatticeTest() {
  console.log('🧪 Running Type Lattice Smoke Test');

  const tests = [
    { name: 'string -> token', from: 'string', to: 'token', expected: true },
    { name: 'token -> lexeme', from: 'token', to: 'lexeme', expected: true },
    { name: 'lexeme -> morpheme', from: 'lexeme', to: 'morpheme', expected: true },
    { name: 'number -> vector2d', from: 'number', to: 'vector2d', expected: true },
    { name: 'vector2d -> vector3d', from: 'vector2d', to: 'vector3d', expected: true },
    { name: 'morpheme -> matrix', from: 'morpheme', to: 'matrix', expected: true },
    { name: 'boolean -> number', from: 'boolean', to: 'number', expected: false }
  ];

  let passed = 0;
  let failed = 0;

  for (const test of tests) {
    const result = typeLattice.checkCompose(test.from, test.to);
    if (result === test.expected) {
      console.log(`✅ ${test.name}: ${result}`);
      passed++;
    } else {
      console.log(`❌ ${test.name}: expected ${test.expected}, got ${result}`);
      failed++;
    }
  }

  console.log(`🎯 Test Results: ${passed} passed, ${failed} failed`);
  return { passed, failed, total: tests.length };
}

// Auto-run test if debug mode is enabled
if (typeof window !== 'undefined' && window.location?.search.includes('debug=types')) {
  setTimeout(() => runTypeLatticeTest(), 100);
}

// ===== Export for Module Systems =====
if (typeof module !== 'undefined' && module.exports) {
  module.exports = {
    WorldEngineTypeLattice,
    typeLattice,
    TypedStore,
    TypedUtils,
    setupTypedEngineTransport,
    runTypeLatticeTest
  };
}
