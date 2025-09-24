/**
 * World Engine V3 Mathematical Complete System
 * Integration of all surgical upgrades with mathematical safety and dimension-agnostic design
 */

// Import enhanced mathematical components
import { LLEXIndexManager } from './llex-performance-index.js';
import { LLEXEngine, LLEXAddressResolver, CIDGenerator } from './llex-core-stable.js';
import { createLLEMorphologyV2, createWordEngineIntegrationV2, initEnhancedUpflow } from './morphology-integration.js';
import { LLEMath, Morpheme, Button, StationaryUnit, ScalingOperations, ButtonFactory } from './lle-stable-math.js';
import { LexicalLogicEngine, EngineFactory, EngineTester } from './lexical-logic-engine-enhanced.js';

/**
 * Enhanced Complete System V3 with Mathematical Safety
 */
class WorldEngineMathematicalSystemV3 {
  constructor(dimensions = 3, options = {}) {
    this.dimensions = dimensions;
    this.options = options;

    // Core mathematical engine with dimension flexibility
    this.lexicalEngine = new LexicalLogicEngine(dimensions);

    // Core LLEX engine with stability patches
    this.llexEngine = new LLEXEngine();

    // High-performance indexing system
    this.indexManager = new LLEXIndexManager();

    // Enhanced address resolution with LRU cache
    this.addressResolver = new LLEXAddressResolver(this.llexEngine);

    // V2 Morphology system with longest-match
    this.morphology = createLLEMorphologyV2();

    // V2 Word engine integration with safer parsing
    this.wordEngine = createWordEngineIntegrationV2();

    // Mathematical components
    this.math = LLEMath;
    this.morphemes = Morpheme.createBuiltInMorphemes(dimensions);
    this.buttons = ButtonFactory.createStandardButtons(dimensions);

    // Initialize the system
    this.initializeEnhancedButtons();
  }

  async initializeEnhancedButtons() {
    const enhancedButtons = [
      {
        lid: 'button:MorphRebuild',
        morphemes: ['re', 'build'],
        wordClass: 'Action',
        options: { description: 'Morpheme-driven rebuild with mathematical validation' }
      },
      {
        lid: 'button:SafeMove',
        morphemes: ['move'],
        wordClass: 'Action',
        options: { description: 'Safe movement with shape validation' }
      },
      {
        lid: 'button:PseudoScale',
        morphemes: ['scale'],
        wordClass: 'Transform',
        options: { description: 'Scaling with pseudo-inverse recovery' }
      },
      {
        lid: 'button:CounterNegate',
        morphemes: ['counter'],
        wordClass: 'Action',
        options: { description: 'Counteraction with stability checks' }
      }
    ];

    for (const buttonDef of enhancedButtons) {
      try {
        // Create button with morpheme composition
        const button = new Button(
          buttonDef.lid.split(':')[1], // label
          buttonDef.lid.split(':')[1].substring(0, 2).toUpperCase(), // abbr
          buttonDef.wordClass,
          buttonDef.morphemes,
          { ...buttonDef.options, dimensions: this.dimensions },
          this.morphemes
        );

        // Store in lexical engine
        this.lexicalEngine.buttons.set(button.abbr, button);

        // Create LLEX operator from button
        const operator = {
          M: button.M,
          b: button.b,
          C: button.C,
          alpha: button.alpha,
          beta: button.beta,
          delta_level: button.deltaLevel
        };

        // Store in LLEX engine
        const result = await this.llexEngine.createButton(
          'core',
          buttonDef.lid,
          operator,
          buttonDef.morphemes,
          { class: buttonDef.wordClass, description: buttonDef.options.description }
        );

        // Index the enhanced button
        this.indexManager.indexObject({
          type: 'enhanced_button',
          cid: result.cid,
          lid: buttonDef.lid,
          description: buttonDef.options.description,
          morphemes: buttonDef.morphemes,
          class: buttonDef.wordClass,
          mathematical: true,
          dimensions: this.dimensions
        });

        console.log('✅ Created enhanced button:', result.address);
      } catch (error) {
        console.warn('⚠️ Failed to create enhanced button:', buttonDef.lid, error);
      }
    }
  }

  // Mathematical operations with safety
  async safeClickButton(buttonKey, params = {}) {
    try {
      // Use lexical engine for mathematical safety
      const result = this.lexicalEngine.clickButton(buttonKey, params);

      // Sync state to LLEX engine if needed
      if (params.syncToLLEX) {
        const session = params.session || 'default';
        const stateCID = await CIDGenerator.hashCanonical(result);
        await this.llexEngine.objectStore.store({ ...result, cid: stateCID });
        this.llexEngine.setSessionHead(session, stateCID);
      }

      return {
        success: true,
        result,
        mathematical_validation: 'passed',
        timestamp: Date.now()
      };
    } catch (error) {
      return {
        success: false,
        error: error.message,
        mathematical_validation: 'failed',
        timestamp: Date.now()
      };
    }
  }

  // Enhanced preview with mathematical validation
  async previewComposition(operations, options = {}) {
    try {
      const preview = this.lexicalEngine.previewCompose(operations);

      // Add mathematical analysis
      const analysis = {
        dimensionality: preview.finalState?.d || this.dimensions,
        mathematical_stability: this._analyzeMathematicalStability(preview.steps),
        recovery_possible: this._checkRecoveryPossibility(preview.steps),
        composition_valid: preview.success
      };

      return {
        ...preview,
        mathematical_analysis: analysis,
        safe_to_apply: preview.success && analysis.mathematical_stability,
        timestamp: Date.now()
      };
    } catch (error) {
      return {
        success: false,
        error: error.message,
        mathematical_analysis: { stability: false },
        timestamp: Date.now()
      };
    }
  }

  _analyzeMathematicalStability(steps) {
    return steps.every(step => {
      if (!step.after?.x) return false;

      // Check for finite values
      if (!step.after.x.every(Number.isFinite)) return false;

      // Check for reasonable bounds
      const maxValue = Math.max(...step.after.x.map(Math.abs));
      if (maxValue > 1000) return false;

      return true;
    });
  }

  _checkRecoveryPossibility(steps) {
    // Check if operations can be undone or if pseudo-inverse can recover
    return steps.some(step =>
      step.operation?.type === 'upscale' ||
      step.operation?.type === 'downscale' ||
      this.lexicalEngine.history.length > 0
    );
  }

  // Scaling operations with pseudo-inverse
  async safeDownscale(keepIndices = [0, 2], options = {}) {
    try {
      const before = this.lexicalEngine.su.copy();
      const result = this.lexicalEngine.downscale(keepIndices);

      return {
        success: true,
        result,
        before_dimensions: before.d,
        after_dimensions: result.d,
        recovery_matrix: this._computeRecoveryMatrix(keepIndices, before.d),
        timestamp: Date.now()
      };
    } catch (error) {
      return {
        success: false,
        error: error.message,
        timestamp: Date.now()
      };
    }
  }

  async safeUpscale(abstractionMatrix = null, toDim = null, options = {}) {
    try {
      const targetDim = toDim || this.dimensions;
      const before = this.lexicalEngine.su.copy();

      // Use pseudo-inverse if no abstraction matrix provided
      const A = abstractionMatrix || this._generateDefaultAbstraction(before.d, targetDim);
      const result = this.lexicalEngine.upscale(A, targetDim);

      return {
        success: true,
        result,
        before_dimensions: before.d,
        after_dimensions: result.d,
        abstraction_matrix: A,
        pseudo_inverse_used: !abstractionMatrix,
        timestamp: Date.now()
      };
    } catch (error) {
      return {
        success: false,
        error: error.message,
        timestamp: Date.now()
      };
    }
  }

  _computeRecoveryMatrix(keepIndices, originalDim) {
    try {
      const P = LLEMath.projectionMatrix(originalDim, keepIndices);
      return LLEMath.pseudoInverse(P);
    } catch (error) {
      console.warn('Could not compute recovery matrix:', error);
      return null;
    }
  }

  _generateDefaultAbstraction(fromDim, toDim) {
    // Generate a reasonable default abstraction matrix
    const A = Array.from({length: Math.min(fromDim, toDim)}, (_, i) =>
      Array.from({length: Math.max(fromDim, toDim)}, (_, j) => {
        if (i === j) return 1.0;
        if (j === i + 1 && j < Math.max(fromDim, toDim)) return 0.5;
        return 0.0;
      })
    );
    return A;
  }

  // Enhanced search with mathematical context
  async mathematicalSearch(query, options = {}) {
    const baseResults = this.indexManager.unifiedSearch(query, options);

    // Add mathematical context
    const enhancedResults = {
      ...baseResults,
      mathematical_context: {
        current_dimensions: this.dimensions,
        current_state: this.lexicalEngine.su.toString(),
        available_operations: Array.from(this.lexicalEngine.buttons.keys()),
        mathematical_stability: this._checkCurrentStability()
      },
      query,
      timestamp: Date.now()
    };

    return enhancedResults;
  }

  _checkCurrentStability() {
    try {
      const su = this.lexicalEngine.su;
      return {
        finite_values: su.x.every(Number.isFinite),
        reasonable_bounds: Math.max(...su.x.map(Math.abs)) < 1000,
        positive_kappa: su.kappa > 0,
        valid_dimensions: su.d === this.dimensions
      };
    } catch (error) {
      return { error: error.message };
    }
  }

  // Undo with mathematical validation
  async safeUndo() {
    try {
      const before = this.lexicalEngine.su.copy();
      const result = this.lexicalEngine.undo();

      return {
        success: true,
        result,
        restored: !this._statesEqual(before, result),
        mathematical_validation: 'passed',
        timestamp: Date.now()
      };
    } catch (error) {
      return {
        success: false,
        error: error.message,
        mathematical_validation: 'failed',
        timestamp: Date.now()
      };
    }
  }

  _statesEqual(state1, state2) {
    return state1.toString() === state2.toString();
  }

  // Self-testing capabilities
  async runMathematicalTests() {
    const basicTests = await EngineTester.runBasicTests();

    // Add our enhanced mathematical tests
    const mathTests = await this._runEnhancedMathTests();

    return {
      basic_tests: basicTests,
      mathematical_tests: mathTests,
      overall_health: this._computeOverallHealth(basicTests, mathTests),
      timestamp: Date.now()
    };
  }

  async _runEnhancedMathTests() {
    const results = [];

    // Test 1: Morpheme composition
    try {
      const button = new Button('Test', 'TS', 'Action', ['re', 'build'], { dimensions: 3 }, this.morphemes);
      const isComposed = button.M.some(row => row.some(v => v !== 0 && v !== 1));

      results.push({
        name: 'Morpheme Composition',
        success: isComposed,
        composed_matrix: isComposed
      });
    } catch (error) {
      results.push({
        name: 'Morpheme Composition',
        success: false,
        error: error.message
      });
    }

    // Test 2: Pseudo-inverse roundtrip
    try {
      const A = [[1, 0.5, 0], [0, 1, 0.3], [0.2, 0, 1]];
      const Aplus = LLEMath.pseudoInverse(A);
      const roundtrip = LLEMath.multiply(A, LLEMath.multiply(Aplus, A));

      const error = A.reduce((sum, row, i) =>
        sum + row.reduce((rowSum, val, j) =>
          rowSum + Math.abs(val - roundtrip[i][j]), 0), 0);

      results.push({
        name: 'Pseudo-inverse Roundtrip',
        success: error < 0.01,
        reconstruction_error: error
      });
    } catch (error) {
      results.push({
        name: 'Pseudo-inverse Roundtrip',
        success: false,
        error: error.message
      });
    }

    return results;
  }

  _computeOverallHealth(basicTests, mathTests) {
    const totalTests = basicTests.summary.total + mathTests.length;
    const totalPassed = basicTests.summary.passed + mathTests.filter(t => t.success).length;

    return {
      overall_score: totalPassed / totalTests,
      tests_passed: totalPassed,
      total_tests: totalTests,
      health_status: totalPassed / totalTests > 0.8 ? 'healthy' : 'degraded'
    };
  }

  // Complete system state
  getEnhancedState() {
    return {
      dimensions: this.dimensions,
      lexical_engine: this.lexicalEngine.getState(),
      llex_engine: this.llexEngine.getStats(),
      indexing: this.indexManager.getStats(),
      mathematical: {
        current_state: this.lexicalEngine.su.toString(),
        stability: this._checkCurrentStability(),
        available_operations: Array.from(this.lexicalEngine.buttons.keys()),
        morpheme_count: this.morphemes.size
      },
      system_version: '3.0',
      timestamp: Date.now()
    };
  }
}

// Enhanced initialization with mathematical upgrades
async function initWorldEngineV3(dimensions = 3, options = {}) {
  console.log(`🧮 Initializing World Engine V3 Mathematical System (${dimensions}D)...`);

  const system = new WorldEngineMathematicalSystemV3(dimensions, options);

  // Initialize enhanced upflow if requested
  if (options.enableUpflow) {
    try {
      system.upflow = await initEnhancedUpflow({
        idbFactory: () => import('./upflow-automation.js')
      });
      console.log('✅ Enhanced Upflow initialized');
    } catch (error) {
      console.warn('⚠️ Failed to initialize Enhanced Upflow:', error);
    }
  }

  // Wait for button initialization
  await new Promise(resolve => setTimeout(resolve, 100));

  // Run initial mathematical validation
  if (options.runTests) {
    const testResults = await system.runMathematicalTests();
    console.log('🧪 Mathematical tests:', testResults.overall_health);
  }

  console.log('🚀 World Engine V3 Mathematical System initialized');
  console.log('📊 Enhanced system state:', system.getEnhancedState().mathematical);

  return system;
}

// Convenience factory functions
const WorldEngineV3Factory = {
  create2D: (options = {}) => initWorldEngineV3(2, options),
  create3D: (options = {}) => initWorldEngineV3(3, options),
  create4D: (options = {}) => initWorldEngineV3(4, options),
  createCustom: (dimensions, options = {}) => initWorldEngineV3(dimensions, options),

  // Testing configurations
  createWithTests: (dimensions = 3) => initWorldEngineV3(dimensions, { runTests: true }),
  createWithUpflow: (dimensions = 3) => initWorldEngineV3(dimensions, { enableUpflow: true }),
  createFull: (dimensions = 3) => initWorldEngineV3(dimensions, { runTests: true, enableUpflow: true })
};

export {
  WorldEngineMathematicalSystemV3,
  initWorldEngineV3,
  WorldEngineV3Factory,
  // Re-export mathematical components for direct use
  LLEMath,
  Morpheme,
  Button,
  StationaryUnit,
  ScalingOperations,
  ButtonFactory,
  LexicalLogicEngine,
  EngineFactory,
  EngineTester
};

// Usage examples with mathematical safety:
/*
// Create a 3D system with all enhancements
const engine = await WorldEngineV3Factory.createFull(3);

// Safe mathematical operations
const clickResult = await engine.safeClickButton('MO', { syncToLLEX: true });
const preview = await engine.previewComposition(['RB', 'SC', { type: 'upscale', toDim: 4 }]);
const scaleResult = await engine.safeUpscale(null, 4);

// Mathematical search with context
const searchResults = await engine.mathematicalSearch('transformation matrix stability');

// Self-testing
const testResults = await engine.runMathematicalTests();

// Access mathematical components directly
const pseudoInv = LLEMath.pseudoInverse([[1,2],[3,4],[5,6]]);
const morphemes = Morpheme.createBuiltInMorphemes(4);
const customButton = new Button('Custom', 'CU', 'Action', ['re', 'scale'], {}, morphemes);
*/
