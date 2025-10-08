/**
 * World Engine V3.1 Integration Layer
 * Integrates Type Lattice, Jacobian Tracing, Morpheme Discovery, and Distributed Sync
 * with existing LexicalLogicEngine and WorldEngine classes
 */

import { LexicalLogicEngine } from './lexical-logic-engine-enhanced.js';
import { TypeLattice } from './type-lattice.js';
import { JacobianTracer, TraceManager } from './jacobian-tracer.js';
import { MorphemeDiscovery, SemanticMorphemeDiscovery } from './morpheme-discovery.js';
import { DistributedSyncManager, ConflictResolver } from './distributed-sync.js';
import { LLEMath, Button, ButtonFactory } from './lle-stable-math.js';

/**
 * Enhanced Lexical Logic Engine with V3.1 Features
 */
export class LexicalLogicEngineV31 extends LexicalLogicEngine {
  constructor(dimensions = 3, options = {}) {
    super(dimensions);

    // Initialize V3.1 components
    this.typeLattice = new TypeLattice();
    this.traceManager = new TraceManager(options.maxTraceHistory || 100);
    this.morphemeDiscovery = options.useSemanticDiscovery
      ? new SemanticMorphemeDiscovery(options.discoveryOptions)
      : new MorphemeDiscovery(options.discoveryOptions);
    this.syncManager = options.enableSync
      ? new DistributedSyncManager(options.syncOptions)
      : null;

    // Enhanced state tracking
    this.v31Features = {
      typeLatticeEnabled: options.typeLatticeEnabled !== false,
      jacobianTracingEnabled: options.jacobianTracingEnabled !== false,
      morphemeDiscoveryEnabled: options.morphemeDiscoveryEnabled !== false,
      distributedSyncEnabled: options.enableSync || false
    };

    this.setupV31Integration();
  }

  /**
   * Setup integration between V3.1 components
   */
  setupV31Integration() {
    // Connect morpheme discovery to button interactions
    this.morphemeDiscovery.on('morphemePromoted', (morpheme) => {
      this.handleMorphemePromotion(morpheme);
    });

    // Connect trace manager to state changes
    this.traceManager.on('traceAdded', (trace) => {
      this.emit('jacobianTrace', trace);
    });

    // Connect sync manager if enabled
    if (this.syncManager) {
      this.syncManager.on('operation-applied', (operation) => {
        this.handleRemoteOperation(operation);
      });

      this.syncManager.on('conflict-resolved', ({ original, resolved }) => {
        this.emit('syncConflictResolved', { original, resolved });
      });
    }

    // Enhanced error handling with V3.1 context
    this.on('error', (error) => {
      this.handleV31Error(error);
    });
  }

  /**
   * Enhanced button click with full V3.1 integration
   */
  clickButton(buttonKey, params = {}) {
    const button = this.buttons.get(buttonKey);
    if (!button) {
      throw new Error(`Button not found: ${buttonKey}`);
    }

    // V3.1 Type lattice composition check
    if (this.v31Features.typeLatticeEnabled && this.history.length > 0) {
      const lastOp = this.history[this.history.length - 1];
      if (lastOp.button && !this.typeLattice.checkCompose(lastOp.button.outputType, button.inputType)) {
        this.stats.compositionErrors++;
        const suggestion = this.typeLattice.suggestComposition(lastOp.button.outputType, button.inputType);
        throw new Error(`Type lattice violation: ${lastOp.button.outputType} ⊄ ${button.inputType}. ${suggestion}`);
      }
    }

    try {
      const before = this.su.copy();
      const after = button.apply(this.su);

      // Validate result
      LLEMath.validateFinite(after.x, `Button ${buttonKey} result`);

      // V3.1 Jacobian tracing
      if (this.v31Features.jacobianTracingEnabled) {
        const trace = JacobianTracer.log({
          button,
          before: { ...before, x: [...before.x] },
          after: { ...after, x: [...after.x] },
          metadata: {
            buttonKey,
            params,
            sessionId: this.syncManager?.sessionId,
            peerId: this.syncManager?.peerId
          }
        });
        this.traceManager.addTrace(trace);
      }

      this.su = after;

      // Enhanced history entry with V3.1 metadata
      const historyEntry = {
        type: 'button',
        button,
        buttonKey,
        params,
        before,
        after: after.copy(),
        timestamp: Date.now(),
        v31_metadata: {
          jacobian_available: this.v31Features.jacobianTracingEnabled,
          type_checked: this.v31Features.typeLatticeEnabled,
          sync_distributed: this.v31Features.distributedSyncEnabled
        }
      };

      this.history.push(historyEntry);
      this.stats.totalOperations++;

      // V3.1 Morpheme discovery learning
      if (this.v31Features.morphemeDiscoveryEnabled) {
        this.recordInteractionForDiscovery([buttonKey], params);
      }

      // V3.1 Distributed sync broadcast
      if (this.syncManager && this.v31Features.distributedSyncEnabled) {
        this.syncManager.applyOperation({
          type: 'button-press',
          data: {
            button: this.serializeButton(button),
            buttonKey,
            previousState: before,
            newState: after
          }
        });
      }

      this.emit('stateChanged', {
        operation: { type: 'button', key: buttonKey },
        currentState: this.su,
        v31_enhancements: {
          trace_id: this.traceManager.traces.length - 1,
          type_validation_passed: true
        }
      });

      return this.su;
    } catch (error) {
      this.stats.mathErrors++;
      this.handleV31Error(error, { buttonKey, params });
      throw error;
    }
  }

  /**
   * Apply discovered morphemes as executable button sequences
   */
  applyMorpheme(morphemeId, params = {}) {
    const morpheme = this.morphemeDiscovery.morphemes.get(morphemeId);
    if (!morpheme) {
      throw new Error(`Morpheme not found: ${morphemeId}`);
    }

    console.log(`🧬 Applying morpheme: ${morpheme.suggested_label} (${morpheme.pattern.join('→')})`);

    const results = [];
    const initialState = this.su.copy();
    const sequenceTrace = [];

    try {
      // Apply each button in the morpheme sequence
      for (const buttonKey of morpheme.pattern) {
        const before = this.su.copy();
        const result = this.clickButton(buttonKey, params[buttonKey] || {});
        const after = result.copy();

        sequenceTrace.push({
          buttonKey,
          before: before.x,
          after: after.x,
          effect: JacobianTracer.effect(before.x, after.x)
        });

        results.push(result);
      }

      // Record morpheme application
      this.history.push({
        type: 'morpheme',
        morpheme,
        morphemeId,
        params,
        initialState,
        finalState: this.su.copy(),
        sequenceTrace,
        timestamp: Date.now()
      });

      // Update morpheme usage statistics
      morpheme.usageCount = (morpheme.usageCount || 0) + 1;
      this.morphemeDiscovery.morphemes.set(morphemeId, morpheme);

      // Distributed sync for morpheme application
      if (this.syncManager) {
        this.syncManager.applyOperation({
          type: 'morpheme-application',
          data: {
            morpheme: this.serializeMorpheme(morpheme),
            morphemeId,
            initialState,
            finalState: this.su
          }
        });
      }

      this.emit('morphemeApplied', {
        morpheme,
        results,
        sequenceTrace,
        effectiveness: morpheme.effectiveness
      });

      return results;
    } catch (error) {
      // Rollback on error
      this.su = initialState;
      this.stats.compositionErrors++;
      this.handleV31Error(error, { morphemeId, morpheme: morpheme.pattern });
      throw error;
    }
  }

  /**
   * Record interaction sequence for morpheme discovery
   */
  recordInteractionForDiscovery(sequence, context = {}) {
    if (!this.v31Features.morphemeDiscoveryEnabled) return;

    const enrichedContext = {
      ...context,
      currentState: this.su.x,
      stateLevel: this.su.level,
      kappa: this.su.kappa,
      sessionId: this.syncManager?.sessionId,
      improvement: this.calculateStateImprovement()
    };

    this.morphemeDiscovery.recordInteraction(sequence, enrichedContext);
  }

  /**
   * Calculate state improvement heuristic for learning
   */
  calculateStateImprovement() {
    if (this.history.length < 2) return 0.5;

    const current = this.history[this.history.length - 1];
    const previous = this.history[this.history.length - 2];

    // Simple heuristic: improvement based on magnitude change
    const currentMagnitude = Math.sqrt(current.after.x.reduce((sum, xi) => sum + xi * xi, 0));
    const previousMagnitude = Math.sqrt(previous.after.x.reduce((sum, xi) => sum + xi * xi, 0));

    // Normalize improvement to [0, 1] range
    const improvement = Math.max(0, Math.min(1, (currentMagnitude - previousMagnitude + 1) / 2));
    return improvement;
  }

  /**
   * Handle morpheme promotion from discovery system
   */
  handleMorphemePromotion(morpheme) {
    console.log(`🎯 New morpheme promoted: "${morpheme.suggested_label}" with pattern [${morpheme.pattern.join('→')}]`);

    // Create a button for the morpheme if it's highly effective
    if (morpheme.effectiveness > 0.7 && morpheme.strength > 0.6) {
      this.createMorphemeButton(morpheme);
    }

    this.emit('morphemeDiscovered', morpheme);
  }

  /**
   * Create an executable button from a discovered morpheme
   */
  createMorphemeButton(morpheme) {
    const buttonKey = `morph_${morpheme.pattern.join('_')}`;

    // Generate a composition matrix from the morpheme pattern
    const composedMatrix = this.calculateMorphemeComposition(morpheme.pattern);

    const morphemeButton = new Button(
      buttonKey,
      morpheme.suggested_label,
      composedMatrix,
      morpheme.strength,
      0 // deltaLevel
    );

    morphemeButton.morphemes = morpheme.pattern;
    morphemeButton.discoveredFrom = morpheme.id;
    morphemeButton.effectiveness = morpheme.effectiveness;

    this.buttons.set(buttonKey, morphemeButton);

    console.log(`🔘 Created morpheme button: ${buttonKey} -> "${morpheme.suggested_label}"`);
    this.emit('morphemeButtonCreated', { button: morphemeButton, morpheme });
  }

  /**
   * Calculate composed transformation matrix from button sequence
   */
  calculateMorphemeComposition(buttonSequence) {
    let composedMatrix = LLEMath.identityMatrix(this.dimensions);

    for (const buttonKey of buttonSequence) {
      const button = this.buttons.get(buttonKey);
      if (button && button.M) {
        composedMatrix = LLEMath.multiplyMatrices(button.M, composedMatrix);
      }
    }

    return composedMatrix;
  }

  /**
   * Handle remote operations from distributed sync
   */
  handleRemoteOperation(operation) {
    if (!this.v31Features.distributedSyncEnabled) return;

    try {
      switch (operation.type) {
      case 'button-press':
        this.applyRemoteButtonPress(operation.data);
        break;
      case 'morpheme-application':
        this.applyRemoteMorpheme(operation.data);
        break;
      case 'state-update':
        this.applyRemoteStateUpdate(operation.data);
        break;
      default:
        console.warn('Unknown remote operation type:', operation.type);
      }
    } catch (error) {
      console.error('Failed to apply remote operation:', error);
      this.emit('remoteOperationError', { operation, error });
    }
  }

  /**
   * Apply button press received from remote peer
   */
  applyRemoteButtonPress(data) {
    const { buttonKey, previousState, newState } = data;

    // Update local state
    this.su.x = [...newState.x];
    this.su.kappa = newState.kappa;
    this.su.level = newState.level;

    // Add to history with remote flag
    this.history.push({
      type: 'remote-button',
      buttonKey,
      before: previousState,
      after: newState,
      timestamp: Date.now(),
      source: 'distributed-sync'
    });

    this.emit('remoteStateChanged', {
      operation: { type: 'remote-button', key: buttonKey },
      currentState: this.su
    });
  }

  /**
   * Get comprehensive V3.1 status and analytics
   */
  getV31Status() {
    const status = {
      engine: {
        dimensions: this.dimensions,
        historyLength: this.history.length,
        stats: { ...this.stats }
      },
      features: { ...this.v31Features },
      components: {}
    };

    // Type Lattice status
    if (this.v31Features.typeLatticeEnabled) {
      status.components.typeLattice = {
        enabled: true,
        latticeType: this.typeLattice.constructor.name
      };
    }

    // Jacobian Tracing status
    if (this.v31Features.jacobianTracingEnabled) {
      status.components.jacobianTracing = {
        enabled: true,
        traceCount: this.traceManager.traces.length,
        summary: this.traceManager.getSummary()
      };
    }

    // Morpheme Discovery status
    if (this.v31Features.morphemeDiscoveryEnabled) {
      status.components.morphemeDiscovery = {
        enabled: true,
        discoveredCount: this.morphemeDiscovery.morphemes.size,
        analytics: this.morphemeDiscovery.getAnalytics(),
        topMorphemes: this.morphemeDiscovery.getTopMorphemes(5).map(m => ({
          id: m.id,
          label: m.suggested_label,
          pattern: m.pattern,
          effectiveness: m.effectiveness
        }))
      };
    }

    // Distributed Sync status
    if (this.v31Features.distributedSyncEnabled && this.syncManager) {
      status.components.distributedSync = {
        enabled: true,
        status: this.syncManager.getStatus(),
        connectedPeers: this.syncManager.getStatus().connectedPeers.length
      };
    }

    return status;
  }

  /**
   * Enhanced error handling with V3.1 context
   */
  handleV31Error(error, context = {}) {
    const errorInfo = {
      error: error.message,
      context,
      timestamp: Date.now(),
      v31_components: {
        typeLattice: this.v31Features.typeLatticeEnabled,
        jacobianTracing: this.v31Features.jacobianTracingEnabled,
        morphemeDiscovery: this.v31features.morphemeDiscoveryEnabled,
        distributedSync: this.v31Features.distributedSyncEnabled
      },
      state: {
        currentX: this.su.x,
        historyLength: this.history.length,
        activeTraces: this.v31Features.jacobianTracingEnabled ? this.traceManager.traces.length : 0,
        discoveredMorphemes: this.v31Features.morphemeDiscoveryEnabled ? this.morphemeDiscovery.morphemes.size : 0
      }
    };

    console.error('V3.1 Enhanced Error:', errorInfo);
    this.emit('v31Error', errorInfo);
  }

  /**
   * Serialize button for network transmission
   */
  serializeButton(button) {
    return {
      abbr: button.abbr,
      label: button.label,
      M: button.M,
      alpha: button.alpha,
      deltaLevel: button.deltaLevel,
      morphemes: button.morphemes,
      inputType: button.inputType,
      outputType: button.outputType
    };
  }

  /**
   * Serialize morpheme for network transmission
   */
  serializeMorpheme(morpheme) {
    return {
      id: morpheme.id,
      pattern: morpheme.pattern,
      suggested_label: morpheme.suggested_label,
      effectiveness: morpheme.effectiveness,
      strength: morpheme.strength,
      frequency: morpheme.frequency
    };
  }

  /**
   * Clean up V3.1 resources
   */
  dispose() {
    if (this.syncManager) {
      this.syncManager.disconnect();
    }

    this.morphemeDiscovery.clear();
    this.traceManager.clearTraces();
    this.listeners.clear();
  }
}

/**
 * Factory for creating V3.1 enhanced engines with different configurations
 */
export class EngineV31Factory {
  /**
   * Create a basic V3.1 engine with essential features
   */
  static createBasic(dimensions = 3) {
    return new LexicalLogicEngineV31(dimensions, {
      typeLatticeEnabled: true,
      jacobianTracingEnabled: true,
      morphemeDiscoveryEnabled: false,
      enableSync: false
    });
  }

  /**
   * Create a full-featured V3.1 engine
   */
  static createFullFeatured(dimensions = 3, syncOptions = {}) {
    return new LexicalLogicEngineV31(dimensions, {
      typeLatticeEnabled: true,
      jacobianTracingEnabled: true,
      morphemeDiscoveryEnabled: true,
      useSemanticDiscovery: true,
      enableSync: true,
      syncOptions: {
        isHost: false,
        ...syncOptions
      },
      discoveryOptions: {
        minFrequency: 3,
        promotionThreshold: 5
      }
    });
  }

  /**
   * Create a research-focused engine with advanced analytics
   */
  static createResearch(dimensions = 3) {
    return new LexicalLogicEngineV31(dimensions, {
      typeLatticeEnabled: true,
      jacobianTracingEnabled: true,
      morphemeDiscoveryEnabled: true,
      useSemanticDiscovery: true,
      enableSync: false,
      maxTraceHistory: 1000,
      discoveryOptions: {
        minFrequency: 2,
        promotionThreshold: 3,
        clusterThreshold: 0.6
      }
    });
  }

  /**
   * Create a collaborative engine for multi-user sessions
   */
  static createCollaborative(dimensions = 3, sessionId, isHost = false) {
    return new LexicalLogicEngineV31(dimensions, {
      typeLatticeEnabled: true,
      jacobianTracingEnabled: true,
      morphemeDiscoveryEnabled: true,
      enableSync: true,
      syncOptions: {
        sessionId,
        isHost,
        websocketUrl: 'ws://localhost:8080/sync'
      }
    });
  }
}
