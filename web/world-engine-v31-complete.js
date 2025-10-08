/**
 * World Engine V3.1 Mathematical System with Advanced Features
 * Integrates Type Lattice, Jacobian Tracing, Morpheme Discovery, and Distributed Sync
 * with the complete V3 mathematical foundation
 */

// Import V3 Mathematical System
import { WorldEngineMathematicalSystemV3 } from './world-engine-v3-mathematical.js';

// Import V3.1 Advanced Components
import { LexicalLogicEngineV31, EngineV31Factory } from './engine-v31-integration.js';
import { TypeLattice } from './type-lattice.js';
import { JacobianTracer, TraceManager } from './jacobian-tracer.js';
import { MorphemeDiscovery, SemanticMorphemeDiscovery } from './morpheme-discovery.js';
import { DistributedSyncManager } from './distributed-sync.js';
import { SyncMessageFactory, StateSerializer, NetworkTopology } from './sync-protocol.js';

// Import Mathematical Foundation
import { LLEMath, Button, ButtonFactory } from './lle-stable-math.js';

/**
 * World Engine V3.1 - Complete Mathematical System with Advanced Features
 */
export class WorldEngineV31 extends WorldEngineMathematicalSystemV3 {
  constructor(dimensions = 3, options = {}) {
    super(dimensions, options);

    // Replace base lexical engine with V3.1 enhanced version
    this.lexicalEngine = new LexicalLogicEngineV31(dimensions, {
      typeLatticeEnabled: options.typeLatticeEnabled !== false,
      jacobianTracingEnabled: options.jacobianTracingEnabled !== false,
      morphemeDiscoveryEnabled: options.morphemeDiscoveryEnabled !== false,
      useSemanticDiscovery: options.useSemanticDiscovery || false,
      enableSync: options.enableSync || false,
      ...options
    });

    // V3.1 Advanced Components
    this.v31Components = {
      typeLattice: this.lexicalEngine.typeLattice,
      traceManager: this.lexicalEngine.traceManager,
      morphemeDiscovery: this.lexicalEngine.morphemeDiscovery,
      syncManager: this.lexicalEngine.syncManager
    };

    // V3.1 Analytics and Monitoring
    this.analytics = {
      sessionStartTime: Date.now(),
      totalInteractions: 0,
      morphemesDiscovered: 0,
      conflictsResolved: 0,
      peersConnected: 0,
      typeViolationsPrevented: 0
    };

    // Enhanced error tracking
    this.errorLog = [];
    this.maxErrorLogSize = options.maxErrorLogSize || 100;

    this.initializeV31Features();
  }

  /**
   * Initialize V3.1 specific features and integrations
   */
  async initializeV31Features() {
    // Setup component event integration
    this.setupV31EventHandling();

    // Initialize type lattice with mathematical button types
    this.initializeTypeLattice();

    // Setup morpheme discovery with mathematical context
    this.initializeMorphemeDiscovery();

    // Initialize distributed sync if enabled
    await this.initializeDistributedSync();

    // Setup periodic analytics collection
    this.setupAnalyticsCollection();

    console.log('🧠 World Engine V3.1 initialized with advanced features');
    this.logSystemStatus();
  }

  /**
   * Setup V3.1 component event handling
   */
  setupV31EventHandling() {
    // Morpheme discovery events
    if (this.v31Components.morphemeDiscovery) {
      this.v31Components.morphemeDiscovery.on('morphemePromoted', (morpheme) => {
        this.analytics.morphemesDiscovered++;
        this.emit('v31:morpheme-discovered', morpheme);
        console.log(`🔬 Morpheme discovered: ${morpheme.suggested_label} (effectiveness: ${morpheme.effectiveness.toFixed(3)})`);
      });

      this.v31Components.morphemeDiscovery.on('patternDiscovered', (pattern) => {
        this.emit('v31:pattern-discovered', pattern);
      });
    }

    // Jacobian tracing events
    if (this.v31Components.traceManager) {
      this.v31Components.traceManager.on('traceAdded', (trace) => {
        this.emit('v31:jacobian-trace', trace);

        // Log significant transformations
        if (trace.transformation.effect_magnitude > 1.0) {
          console.log(`📊 Significant transformation: ${trace.explanation}`);
        }
      });
    }

    // Distributed sync events
    if (this.v31Components.syncManager) {
      this.v31Components.syncManager.on('peer-connected', ({ peerId }) => {
        this.analytics.peersConnected++;
        this.emit('v31:peer-connected', { peerId });
        console.log(`👥 Peer connected: ${peerId}`);
      });

      this.v31Components.syncManager.on('conflict-resolved', ({ original, resolved }) => {
        this.analytics.conflictsResolved++;
        this.emit('v31:conflict-resolved', { original, resolved });
        console.log('⚖️ Conflict resolved between operations');
      });
    }

    // Enhanced error handling
    this.lexicalEngine.on('v31Error', (errorInfo) => {
      this.logError(errorInfo);
      this.emit('v31:error', errorInfo);
    });
  }

  /**
   * Initialize type lattice with mathematical button classifications
   */
  initializeTypeLattice() {
    if (!this.v31Components.typeLattice) return;

    const lattice = this.v31Components.typeLattice;

    // Define mathematical type relationships for buttons
    const buttonTypes = new Map([
      ['linear', 'Property'],     // Linear transformations
      ['rotation', 'Property'],   // Rotations
      ['scaling', 'Property'],    // Scaling operations
      ['translation', 'State'],   // Position changes
      ['composition', 'Structure'], // Complex compositions
      ['morpheme', 'Concept'],    // Morpheme-based operations
      ['pseudo', 'Structure']     // Pseudo-inverse operations
    ]);

    // Classify existing buttons by mathematical properties
    for (const [key, button] of this.lexicalEngine.buttons.entries()) {
      if (button.M) {
        const det = this.math.determinant(button.M);
        const isOrthogonal = this.isOrthogonalMatrix(button.M);

        let buttonType = 'Property'; // Default

        if (Math.abs(det - 1) < 1e-6) {
          buttonType = isOrthogonal ? 'rotation' : 'linear';
        } else if (det > 1) {
          buttonType = 'scaling';
        } else {
          buttonType = 'composition';
        }

        button.inputType = buttonTypes.get(buttonType) || 'State';
        button.outputType = button.inputType; // Same type for mathematical operations
      }
    }

    console.log(`🏗️ Type lattice initialized with ${buttonTypes.size} mathematical categories`);
  }

  /**
   * Initialize morpheme discovery with mathematical context
   */
  initializeMorphemeDiscovery() {
    if (!this.v31Components.morphemeDiscovery) return;

    // Set mathematical improvement calculation
    const originalCalculateImprovement = this.lexicalEngine.calculateStateImprovement.bind(this.lexicalEngine);

    this.lexicalEngine.calculateStateImprovement = () => {
      const basicImprovement = originalCalculateImprovement();

      // Enhanced improvement calculation considering mathematical properties
      if (this.lexicalEngine.history.length > 1) {
        const lastTrace = this.v31Components.traceManager.getRecentTraces(1)[0];
        if (lastTrace) {
          // Factor in stability and mathematical soundness
          const stabilityScore = lastTrace.operation.stability === 'stable' ? 0.2 : -0.1;
          const magnitudeScore = Math.min(0.2, lastTrace.transformation.effect_magnitude / 5);
          return Math.max(0, Math.min(1, basicImprovement + stabilityScore + magnitudeScore));
        }
      }

      return basicImprovement;
    };

    console.log('🧬 Morpheme discovery initialized with mathematical context');
  }

  /**
   * Initialize distributed synchronization
   */
  async initializeDistributedSync() {
    if (!this.v31Components.syncManager) return;

    // Setup mathematical state serialization
    const originalApplyOperation = this.v31Components.syncManager.applyOperation.bind(this.v31Components.syncManager);

    this.v31Components.syncManager.applyOperation = (operation) => {
      // Add mathematical validation before sync
      if (operation.type === 'button-press' && operation.data.newState) {
        try {
          this.math.validateFinite(operation.data.newState.x, 'Sync operation state');
        } catch (error) {
          console.error('Invalid state in sync operation:', error);
          return;
        }
      }

      return originalApplyOperation(operation);
    };

    console.log('🌐 Distributed sync initialized with mathematical validation');
  }

  /**
   * Enhanced button click with V3.1 mathematical integration
   */
  async clickButton(buttonKey, params = {}) {
    try {
      this.analytics.totalInteractions++;

      // Use V3.1 enhanced engine
      const result = await this.lexicalEngine.clickButton(buttonKey, params);

      // Update LLEX state synchronously
      this.updateLLEXFromMathematicalState(result);

      // Trigger morpheme discovery pattern analysis
      this.triggerPatternAnalysis(buttonKey, result);

      this.emit('v31:button-clicked', {
        buttonKey,
        result,
        analytics: this.getInteractionAnalytics()
      });

      return result;
    } catch (error) {
      this.logError({
        type: 'button-click-error',
        buttonKey,
        error: error.message,
        params
      });
      throw error;
    }
  }

  /**
   * Apply discovered morpheme with V3.1 enhancements
   */
  async applyMorpheme(morphemeId, params = {}) {
    try {
      const result = await this.lexicalEngine.applyMorpheme(morphemeId, params);

      // Update LLEX state with morpheme result
      this.updateLLEXFromMathematicalState(result[result.length - 1]);

      this.emit('v31:morpheme-applied', {
        morphemeId,
        result,
        sequenceLength: result.length
      });

      return result;
    } catch (error) {
      this.logError({
        type: 'morpheme-application-error',
        morphemeId,
        error: error.message,
        params
      });
      throw error;
    }
  }

  /**
   * Update LLEX system from mathematical engine state
   */
  updateLLEXFromMathematicalState(mathematicalState) {
    try {
      // Convert mathematical state to LLEX format
      const llexState = {
        x: mathematicalState.x,
        kappa: mathematicalState.kappa,
        level: mathematicalState.level,
        timestamp: Date.now()
      };

      // Store in LLEX engine
      this.llexEngine.state = llexState;

      // Update index with new state
      this.indexManager.updateIndex('current_state', llexState);

    } catch (error) {
      console.error('Failed to update LLEX state:', error);
    }
  }

  /**
   * Trigger pattern analysis for morpheme discovery
   */
  triggerPatternAnalysis(buttonKey, currentState) {
    if (!this.v31Components.morphemeDiscovery) return;

    // Build recent button sequence for pattern analysis
    const recentButtons = this.lexicalEngine.history
      .filter(entry => entry.type === 'button')
      .slice(-5)
      .map(entry => entry.buttonKey);

    recentButtons.push(buttonKey);

    if (recentButtons.length >= 2) {
      // Record for morpheme discovery
      this.v31Components.morphemeDiscovery.recordInteraction(recentButtons, {
        currentState: currentState.x,
        stateLevel: currentState.level,
        kappa: currentState.kappa,
        improvement: this.lexicalEngine.calculateStateImprovement(),
        sessionId: this.v31Components.syncManager?.sessionId
      });
    }
  }

  /**
   * Check if matrix is orthogonal (for type classification)
   */
  isOrthogonalMatrix(matrix) {
    const n = matrix.length;
    const transpose = this.math.transposeMatrix(matrix);
    const product = this.math.multiplyMatrices(matrix, transpose);
    const identity = this.math.identityMatrix(n);

    // Check if M * M^T ≈ I
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        if (Math.abs(product[i][j] - identity[i][j]) > 1e-6) {
          return false;
        }
      }
    }
    return true;
  }

  /**
   * Get comprehensive interaction analytics
   */
  getInteractionAnalytics() {
    const sessionDuration = Date.now() - this.analytics.sessionStartTime;
    const v31Status = this.lexicalEngine.getV31Status();

    return {
      session: {
        duration_ms: sessionDuration,
        total_interactions: this.analytics.totalInteractions,
        interactions_per_minute: (this.analytics.totalInteractions / (sessionDuration / 60000)).toFixed(2)
      },
      v31_features: {
        morphemes_discovered: this.analytics.morphemesDiscovered,
        conflicts_resolved: this.analytics.conflictsResolved,
        peers_connected: this.analytics.peersConnected,
        type_violations_prevented: this.analytics.typeViolationsPrevented
      },
      engine_status: v31Status,
      mathematical_health: this.getMathematicalHealthMetrics()
    };
  }

  /**
   * Get mathematical health metrics
   */
  getMathematicalHealthMetrics() {
    const recentTraces = this.v31Components.traceManager?.getRecentTraces(10) || [];
    const errorRate = this.errorLog.length / Math.max(1, this.analytics.totalInteractions);

    return {
      error_rate: errorRate,
      recent_stability: recentTraces.filter(t => t.operation.stability === 'stable').length / Math.max(1, recentTraces.length),
      average_effect_magnitude: recentTraces.length > 0
        ? recentTraces.reduce((sum, t) => sum + t.transformation.effect_magnitude, 0) / recentTraces.length
        : 0,
      state_vector_health: {
        current_magnitude: Math.sqrt(this.lexicalEngine.su.x.reduce((sum, xi) => sum + xi * xi, 0)),
        has_nan: this.lexicalEngine.su.x.some(xi => !isFinite(xi)),
        kappa_stable: Math.abs(this.lexicalEngine.su.kappa - 1.0) < 10
      }
    };
  }

  /**
   * Setup periodic analytics collection
   */
  setupAnalyticsCollection() {
    setInterval(() => {
      const analytics = this.getInteractionAnalytics();
      this.emit('v31:analytics-update', analytics);

      // Auto-cleanup old traces and errors
      this.cleanup();
    }, 30000); // Every 30 seconds
  }

  /**
   * Log error with V3.1 context
   */
  logError(errorInfo) {
    const enhancedError = {
      ...errorInfo,
      timestamp: Date.now(),
      system_state: {
        dimensions: this.dimensions,
        current_x: this.lexicalEngine.su.x,
        history_length: this.lexicalEngine.history.length
      },
      v31_context: {
        morphemes_active: this.v31Components.morphemeDiscovery?.morphemes.size || 0,
        traces_available: this.v31Components.traceManager?.traces.length || 0,
        sync_connected: this.v31Components.syncManager?.isConnected || false
      }
    };

    this.errorLog.push(enhancedError);

    // Maintain error log size
    if (this.errorLog.length > this.maxErrorLogSize) {
      this.errorLog.shift();
    }
  }

  /**
   * Cleanup old data to maintain performance
   */
  cleanup() {
    // Cleanup trace manager
    if (this.v31Components.traceManager && this.v31Components.traceManager.traces.length > 200) {
      const oldTraces = this.v31Components.traceManager.traces.splice(0, 50);
      console.log(`🧹 Cleaned up ${oldTraces.length} old Jacobian traces`);
    }

    // Cleanup morpheme discovery patterns
    if (this.v31Components.morphemeDiscovery) {
      this.v31Components.morphemeDiscovery.applyDecay();
    }

    // Cleanup LLEX index
    this.indexManager.cleanup();
  }

  /**
   * Log comprehensive system status
   */
  logSystemStatus() {
    const status = {
      engine: 'World Engine V3.1',
      dimensions: this.dimensions,
      features: {
        mathematical_safety: '✓ LLEMath with NaN guards',
        type_lattice: this.v31Components.typeLattice ? '✓ Hierarchical type system' : '✗ Disabled',
        jacobian_tracing: this.v31Components.traceManager ? '✓ Mathematical transparency' : '✗ Disabled',
        morpheme_discovery: this.v31Components.morphemeDiscovery ? '✓ Learning system' : '✗ Disabled',
        distributed_sync: this.v31Components.syncManager ? '✓ Multi-user ready' : '✗ Disabled'
      },
      components: {
        buttons: this.lexicalEngine.buttons.size,
        morphemes: this.morphemes.size,
        llex_operators: this.llexEngine.operators?.size || 0
      }
    };

    console.log('🌟 World Engine V3.1 Status:', status);
  }

  /**
   * Export complete system state for persistence
   */
  exportState() {
    return {
      version: 'v3.1',
      timestamp: Date.now(),
      dimensions: this.dimensions,
      mathematical_state: {
        current_x: this.lexicalEngine.su.x,
        kappa: this.lexicalEngine.su.kappa,
        level: this.lexicalEngine.su.level
      },
      history: this.lexicalEngine.history.slice(-50), // Recent history
      discovered_morphemes: this.v31Components.morphemeDiscovery?.exportMorphemes() || [],
      jacobian_traces: this.v31Components.traceManager?.getRecentTraces(20) || [],
      analytics: this.analytics,
      llex_state: this.llexEngine.state,
      error_summary: {
        total_errors: this.errorLog.length,
        recent_errors: this.errorLog.slice(-5)
      }
    };
  }

  /**
   * Import and restore system state
   */
  async importState(exportedState) {
    if (exportedState.version !== 'v3.1') {
      console.warn('State version mismatch, attempting compatibility mode');
    }

    // Restore mathematical state
    this.lexicalEngine.su.x = exportedState.mathematical_state.current_x;
    this.lexicalEngine.su.kappa = exportedState.mathematical_state.kappa;
    this.lexicalEngine.su.level = exportedState.mathematical_state.level;

    // Restore discovered morphemes
    if (exportedState.discovered_morphemes && this.v31Components.morphemeDiscovery) {
      this.v31Components.morphemeDiscovery.importMorphemes(exportedState.discovered_morphemes);
    }

    // Restore analytics
    this.analytics = { ...this.analytics, ...exportedState.analytics };

    console.log(`🔄 State restored from ${new Date(exportedState.timestamp).toISOString()}`);
  }

  /**
   * Dispose of all V3.1 resources
   */
  dispose() {
    this.lexicalEngine.dispose();
    this.cleanup();
    this.removeAllListeners();

    console.log('🗑️ World Engine V3.1 disposed');
  }
}

/**
 * Factory for creating different V3.1 system configurations
 */
export class WorldEngineV31Factory {
  /**
   * Create a standard V3.1 system with essential features
   */
  static createStandard(dimensions = 3) {
    return new WorldEngineV31(dimensions, {
      typeLatticeEnabled: true,
      jacobianTracingEnabled: true,
      morphemeDiscoveryEnabled: false,
      enableSync: false
    });
  }

  /**
   * Create a full-featured V3.1 system
   */
  static createFullFeatured(dimensions = 3) {
    return new WorldEngineV31(dimensions, {
      typeLatticeEnabled: true,
      jacobianTracingEnabled: true,
      morphemeDiscoveryEnabled: true,
      useSemanticDiscovery: true,
      enableSync: true
    });
  }

  /**
   * Create a research configuration with extended analytics
   */
  static createResearch(dimensions = 3) {
    return new WorldEngineV31(dimensions, {
      typeLatticeEnabled: true,
      jacobianTracingEnabled: true,
      morphemeDiscoveryEnabled: true,
      useSemanticDiscovery: true,
      enableSync: false,
      maxErrorLogSize: 500,
      discoveryOptions: {
        minFrequency: 2,
        promotionThreshold: 3
      }
    });
  }

  /**
   * Create a collaborative system for multi-user sessions
   */
  static createCollaborative(dimensions = 3, sessionId, isHost = false) {
    return new WorldEngineV31(dimensions, {
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

/**
 * Initialize World Engine V3.1 with auto-detection of optimal configuration
 */
export async function initWorldEngineV31(dimensions = 3, options = {}) {
  console.log('🚀 Initializing World Engine V3.1...');

  const autoOptions = {
    // Auto-enable features based on environment
    typeLatticeEnabled: true,
    jacobianTracingEnabled: true,
    morphemeDiscoveryEnabled: options.learningMode !== false,
    enableSync: options.multiUser || false,
    useSemanticDiscovery: options.learningMode === 'advanced',
    ...options
  };

  const engine = new WorldEngineV31(dimensions, autoOptions);
  await engine.initializeV31Features();

  console.log('✨ World Engine V3.1 ready with advanced mathematical features');
  return engine;
}
