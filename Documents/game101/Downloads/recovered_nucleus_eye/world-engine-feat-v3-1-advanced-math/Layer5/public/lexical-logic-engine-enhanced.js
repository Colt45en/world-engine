/**
 * Enhanced Dimension-Agnostic Lexical Logic Engine
 * Features: Undo/Redo, Preview Composition, Type Safety, Mathematical Validation
 */

import { LLEMath, Morpheme, Button, StationaryUnit, ScalingOperations, ButtonFactory } from './lle-stable-math.js';

/**
 * Enhanced Lexical Logic Engine with full dimensional flexibility
 */
class LexicalLogicEngine {
  constructor(dimensions = 3) {
    this.dimensions = dimensions;
    this.su = new StationaryUnit(dimensions);
    this.morphemes = Morpheme.createBuiltInMorphemes(dimensions);
    this.buttons = ButtonFactory.createStandardButtons(dimensions);
    this.history = [];
    this.listeners = new Map();

    // Track operation statistics
    this.stats = {
      totalOperations: 0,
      undoOperations: 0,
      previewRequests: 0,
      compositionErrors: 0,
      mathErrors: 0
    };
  }

  /**
   * Enhanced button click with type checking and validation
   */
  clickButton(buttonKey, params = {}) {
    const button = this.buttons.get(buttonKey);
    if (!button) {
      throw new Error(`Button not found: ${buttonKey}`);
    }

    // Type composition check if we have a previous operation
    if (this.history.length > 0) {
      const lastOp = this.history[this.history.length - 1];
      if (lastOp.button && !button.canComposeWith(lastOp.button)) {
        this.stats.compositionErrors++;
        throw new Error(`Type mismatch: ${lastOp.button.outputType} → ${button.inputType} not compatible`);
      }
    }

    try {
      const before = this.su.copy();
      const after = button.apply(this.su);

      // Validate result for NaN/Infinity
      LLEMath.validateFinite(after.x, `Button ${buttonKey} result`);

      this.su = after;
      this.history.push({
        type: 'button',
        button,
        buttonKey,
        params,
        before,
        after: after.copy(),
        timestamp: Date.now()
      });

      this.stats.totalOperations++;
      this.emit('stateChanged', {
        operation: { type: 'button', key: buttonKey },
        currentState: this.su
      });

      return this.su;
    } catch (error) {
      this.stats.mathErrors++;
      throw error;
    }
  }

  /**
   * Execute a sequence of button clicks with composition validation
   */
  clickSequence(buttonKeys, params = {}) {
    const results = [];
    const initialState = this.su.copy();

    try {
      for (const key of buttonKeys) {
        const result = this.clickButton(key, params[key] || {});
        results.push(result.copy());
      }
      return results;
    } catch (error) {
      // Rollback on error
      this.su = initialState;
      this.history.push({
        type: 'rollback',
        reason: error.message,
        restoredTo: initialState.copy(),
        timestamp: Date.now()
      });
      throw error;
    }
  }

  /**
   * Scaling operations with enhanced validation
   */
  downscale(keepIndices = [0, 2]) {
    const before = this.su.copy();
    const after = ScalingOperations.downscale(this.su, keepIndices);

    this.su = after;
    this.history.push({
      type: 'downscale',
      keepIndices,
      before,
      after: after.copy(),
      timestamp: Date.now()
    });

    this.stats.totalOperations++;
    this.emit('stateChanged', {
      operation: { type: 'downscale', keepIndices },
      currentState: this.su
    });

    return this.su;
  }

  upscale(abstractionMatrix = null, toDim = null) {
    const targetDim = toDim || this.dimensions;
    const before = this.su.copy();
    const after = ScalingOperations.upscale(this.su, abstractionMatrix, targetDim);

    this.su = after;
    this.history.push({
      type: 'upscale',
      abstractionMatrix,
      toDim: targetDim,
      before,
      after: after.copy(),
      timestamp: Date.now()
    });

    this.stats.totalOperations++;
    this.emit('stateChanged', {
      operation: { type: 'upscale', toDim: targetDim },
      currentState: this.su
    });

    return this.su;
  }

  /**
   * Undo last operation that has a 'before' state
   */
  undo() {
    for (let i = this.history.length - 1; i >= 0; i--) {
      const h = this.history[i];
      if (h.before) {
        this.su = h.before.copy();
        this.history.push({
          type: 'undo',
          restoredFrom: i,
          restoredState: this.su.copy(),
          timestamp: Date.now()
        });

        this.stats.undoOperations++;
        this.emit('stateChanged', {
          operation: { type: 'undo' },
          currentState: this.su
        });

        return this.su;
      }
    }

    // No undoable operation found
    return this.su;
  }

  /**
   * Preview composition without mutating current state
   */
  previewCompose(operations) {
    this.stats.previewRequests++;
    let su = this.su.copy();
    const steps = [];

    try {
      for (const op of operations) {
        const stepBefore = su.copy();

        if (typeof op === 'string') {
          // Button operation
          const button = this.buttons.get(op);
          if (!button) throw new Error(`Button not found: ${op}`);
          su = button.apply(su);
        } else if (op.type === 'downscale') {
          su = ScalingOperations.downscale(su, op.keepIndices || [0, 2]);
        } else if (op.type === 'upscale') {
          su = ScalingOperations.upscale(su, op.abstractionMatrix, op.toDim || this.dimensions);
        } else {
          throw new Error(`Unknown operation type: ${op.type}`);
        }

        steps.push({
          operation: op,
          before: stepBefore,
          after: su.copy()
        });
      }

      return {
        success: true,
        finalState: su,
        steps,
        initialState: this.su.copy()
      };

    } catch (error) {
      return {
        success: false,
        error: error.message,
        steps,
        finalState: su,
        initialState: this.su.copy()
      };
    }
  }

  /**
   * Apply a previewed composition if it was successful
   */
  applyPreview(previewResult) {
    if (!previewResult.success) {
      throw new Error(`Cannot apply failed preview: ${previewResult.error}`);
    }

    const before = this.su.copy();
    this.su = previewResult.finalState.copy();

    this.history.push({
      type: 'preview_apply',
      before,
      after: this.su.copy(),
      previewSteps: previewResult.steps.length,
      timestamp: Date.now()
    });

    this.stats.totalOperations++;
    this.emit('stateChanged', {
      operation: { type: 'preview_apply' },
      currentState: this.su
    });

    return this.su;
  }

  /**
   * Event listener management
   */
  addEventListener(event, callback) {
    if (!this.listeners.has(event)) {
      this.listeners.set(event, []);
    }
    this.listeners.get(event).push(callback);
  }

  removeEventListener(event, callback) {
    const arr = this.listeners.get(event) || [];
    const i = arr.indexOf(callback);
    if (i >= 0) arr.splice(i, 1);
  }

  emit(event, data) {
    const callbacks = this.listeners.get(event) || [];
    callbacks.forEach(cb => {
      try {
        cb(data);
      } catch (error) {
        console.warn(`Event listener error for ${event}:`, error);
      }
    });
  }

  /**
   * Get current engine state
   */
  getState() {
    return {
      su: this.su.copy(),
      dimensions: this.dimensions,
      history: this.history.length,
      stats: { ...this.stats },
      buttons: Array.from(this.buttons.keys())
    };
  }

  /**
   * Get detailed button information
   */
  getButtonInfo(buttonKey) {
    const button = this.buttons.get(buttonKey);
    if (!button) return null;

    return {
      key: buttonKey,
      label: button.label,
      abbr: button.abbr,
      wordClass: button.wordClass,
      morphemes: button.morphemes,
      inputType: button.inputType,
      outputType: button.outputType,
      description: button.description,
      deltaLevel: button.deltaLevel,
      canApply: true // Could add more sophisticated checks here
    };
  }

  /**
   * Reset engine to initial state
   */
  reset(dimensions = null) {
    const newDim = dimensions || this.dimensions;
    this.su = new StationaryUnit(newDim);

    if (newDim !== this.dimensions) {
      this.dimensions = newDim;
      this.morphemes = Morpheme.createBuiltInMorphemes(newDim);
      this.buttons = ButtonFactory.createStandardButtons(newDim);
    }

    this.history.push({
      type: 'reset',
      dimensions: newDim,
      timestamp: Date.now()
    });

    this.emit('stateChanged', {
      operation: { type: 'reset' },
      currentState: this.su
    });

    return this.su;
  }

  /**
   * Export engine state for serialization
   */
  exportState() {
    return {
      dimensions: this.dimensions,
      currentState: {
        x: this.su.x,
        Sigma: this.su.Sigma,
        kappa: this.su.kappa,
        level: this.su.level,
        d: this.su.d
      },
      stats: this.stats,
      historyLength: this.history.length,
      timestamp: Date.now()
    };
  }

  /**
   * Import engine state from serialized data
   */
  importState(stateData) {
    this.dimensions = stateData.dimensions;
    this.su = new StationaryUnit(
      stateData.currentState.d,
      stateData.currentState.x,
      stateData.currentState.Sigma,
      stateData.currentState.kappa,
      stateData.currentState.level
    );

    this.morphemes = Morpheme.createBuiltInMorphemes(this.dimensions);
    this.buttons = ButtonFactory.createStandardButtons(this.dimensions);

    this.history.push({
      type: 'import',
      importedFrom: stateData.timestamp,
      timestamp: Date.now()
    });

    this.emit('stateChanged', {
      operation: { type: 'import' },
      currentState: this.su
    });

    return this.su;
  }
}

/**
 * Engine Factory for convenient creation
 */
class EngineFactory {
  static create2D() {
    return new LexicalLogicEngine(2);
  }

  static create3D() {
    return new LexicalLogicEngine(3);
  }

  static create4D() {
    return new LexicalLogicEngine(4);
  }

  static createCustom(dimensions) {
    if (dimensions < 1 || dimensions > 10) {
      throw new Error('Dimensions must be between 1 and 10');
    }
    return new LexicalLogicEngine(dimensions);
  }
}

/**
 * Self-test suite for validation
 */
class EngineTester {
  static async runBasicTests() {
    const results = [];

    // Test 1: Basic button operations
    try {
      const eng = new LexicalLogicEngine(3);
      const before = eng.getState().su.toString();

      eng.clickSequence(['RB', 'UP', 'CV', 'TL']);
      const after = eng.getState().su.toString();

      results.push({
        name: 'Basic Operations',
        success: true,
        before,
        after,
        operations: 4
      });
    } catch (error) {
      results.push({
        name: 'Basic Operations',
        success: false,
        error: error.message
      });
    }

    // Test 2: Down→Up roundtrip
    try {
      const eng = new LexicalLogicEngine(3);
      const original = eng.su.copy();

      const preview = eng.previewCompose([
        { type: 'downscale', keepIndices: [0, 2] },
        { type: 'upscale', toDim: 3 }
      ]);

      const drift = preview.finalState.x.map((v, i) =>
        Math.abs(v - original.x[i])
      );
      const maxDrift = Math.max(...drift);

      results.push({
        name: 'Roundtrip Test',
        success: maxDrift < 0.1,
        maxDrift,
        acceptable: maxDrift < 0.1
      });
    } catch (error) {
      results.push({
        name: 'Roundtrip Test',
        success: false,
        error: error.message
      });
    }

    // Test 3: Undo functionality
    try {
      const eng = new LexicalLogicEngine(3);
      const initial = eng.su.copy();

      eng.clickButton('RB');
      eng.clickButton('UP');
      const afterOps = eng.su.copy();

      eng.undo();
      const afterUndo = eng.su.copy();

      const undoWorks = Math.abs(afterUndo.level - initial.level) <
                       Math.abs(afterOps.level - initial.level);

      results.push({
        name: 'Undo Functionality',
        success: undoWorks,
        levelRestore: undoWorks
      });
    } catch (error) {
      results.push({
        name: 'Undo Functionality',
        success: false,
        error: error.message
      });
    }

    // Test 4: Dimension flexibility
    try {
      const eng2D = EngineFactory.create2D();
      const eng4D = EngineFactory.create4D();

      eng2D.clickButton('MV');
      eng4D.clickButton('SC');

      const dims2D = eng2D.su.d;
      const dims4D = eng4D.su.d;

      results.push({
        name: 'Dimension Flexibility',
        success: dims2D === 2 && dims4D === 4,
        dims2D,
        dims4D
      });
    } catch (error) {
      results.push({
        name: 'Dimension Flexibility',
        success: false,
        error: error.message
      });
    }

    return {
      testResults: results,
      summary: {
        total: results.length,
        passed: results.filter(r => r.success).length,
        failed: results.filter(r => !r.success).length
      },
      timestamp: Date.now()
    };
  }
}

export {
  LexicalLogicEngine,
  EngineFactory,
  EngineTester
};
