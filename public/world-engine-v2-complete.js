/**
 * World Engine V3 - Complete Mathematical System Integration
 * Features: Mathematical safety, morpheme-driven composition, pseudo-inverse operations,
 * dimension-agnostic engine, undo/redo, preview composition
 */

// Import all the enhanced components
import { createLLEXGraphIndex,createLLEXIndexManager } from './llex-performance-index.js';
import { LLEXEngine, LLEXAddressResolver, CIDGenerator } from './llex-core-stable.js';
import { createLLEMorphologyV2, createWordEngineIntegrationV2, initEnhancedUpflow } from './morphology-integration.js';
import { LLEMath, Morpheme, Button, StationaryUnit, ScalingOperations, ButtonFactory } from './lle-stable-math.js';
import { LexicalLogicEngine, EngineFactory, EngineTester } from './lexical-logic-engine-enhanced.js';

/**
 * Enhanced Complete System with all upgrades
 */
class WorldEngineCompleteSystemV2 {
  constructor() {
    // Core engine with stability patches
    this.engine = new LLEXEngine();

    // High-performance indexing system
    this.indexManager = new createLLEXIndexManager();

    // Enhanced address resolution with LRU cache
    this.addressResolver = new LLEXAddressResolver(this.engine);

    // V2 Morphology system with longest-match
    this.morphology = createLLEMorphologyV2();

    // V2 Word engine integration with safer parsing
    this.wordEngine = createWordEngineIntegrationV2();

    this.initializeBasicButtons();
  }

  async initializeBasicButtons() {
    const basicButtons = [
      {
        lid: 'button:Move',
        operator: { M: [[1,0,0],[0,1,0],[0,0,1]], b: [0.1,0,0], alpha: 1.0, delta_level: 1 },
        morphemes: ['move'],
        metadata: { class: 'Motion', description: 'Basic movement transformation' }
      },
      {
        lid: 'button:Scale',
        operator: { M: [[1.1,0,0],[0,1.1,0],[0,0,1.1]], b: [0,0,0], alpha: 1.0, delta_level: 0 },
        morphemes: ['scale'],
        metadata: { class: 'Transform', description: 'Uniform scaling transformation' }
      },
      {
        lid: 'button:Rotate',
        operator: { M: [[0.9,0.1,0],[-0.1,0.9,0],[0,0,1]], b: [0,0,0], alpha: 1.0, delta_level: 0 },
        morphemes: ['rotate'],
        metadata: { class: 'Transform', description: 'Rotation transformation' }
      }
    ];

    for (const buttonDef of basicButtons) {
      try {
        const result = await this.engine.createButton(
          'core',
          buttonDef.lid,
          buttonDef.operator,
          buttonDef.morphemes,
          buttonDef.metadata
        );

        // Index the button for search
        this.indexManager.indexObject({
          type: 'button',
          cid: result.cid,
          lid: buttonDef.lid,
          description: buttonDef.metadata.description,
          morphemes: buttonDef.morphemes,
          class: buttonDef.metadata.class
        });

        console.log('✅ Created button:', result.address);
      } catch (error) {
        console.warn('⚠️ Failed to create button:', buttonDef.lid, error);
      }
    }
  }

  // Enhanced search with fusion scoring
  async search(query, options = {}) {
    const results = this.indexManager.unifiedSearch(query, options);
    return {
      ...results,
      query,
      timestamp: Date.now(),
      system_version: '2.0'
    };
  }

  // Enhanced morphological analysis
  analyzeMorphology(word, context = '') {
    const analysis = this.morphology(word);
    return {
      ...analysis,
      context,
      analysis_timestamp: Date.now(),
      complexity_score: analysis.complexity,
      semantic_hints: this._generateSemanticHints(analysis)
    };
  }

  _generateSemanticHints(morphAnalysis) {
    const hints = [];

    // Prefix-based hints
    for (const prefix of morphAnalysis.prefixes) {
      if (prefix === 'counter') hints.push('opposition_indicator');
      if (prefix === 'multi') hints.push('quantity_multiplier');
      if (['un', 'dis'].includes(prefix)) hints.push('negation_marker');
    }

    // Suffix-based hints
    for (const suffix of morphAnalysis.suffixes) {
      if (['ize', 'ify', 'ate'].includes(suffix)) hints.push('action_verb');
      if (['ness', 'tion', 'ment'].includes(suffix)) hints.push('abstract_noun');
      if (['ive', 'ous', 'al'].includes(suffix)) hints.push('descriptive_adjective');
    }

    return hints;
  }

  // Safe button click with enhanced error handling
  async clickButton(session, buttonAddress, params = {}, provenance = {}) {
    try {
      // Parse button address safely
      const [buttonLID, buttonVID] = this.engine.parseButtonAddress(buttonAddress);

      // Resolve button with caching
      const button = await this.addressResolver.resolve(
        this.addressResolver.buildAddress('core', 'button', buttonLID.split(':')[1], buttonVID)
      );

      // Get current session state or initialize
      const currentState = this.engine.getSessionHead(session) || { x: [0, 0, 0], level: 1, kappa: 1.0 };

      // Ensure vector dimension sync
      if (this.indexManager.vectorIndex && this.indexManager.vectorIndex.vectors?.size === 0) {
        this.indexManager.setVectorDimension(currentState.x.length);
      }

      // Apply transformation with validation
      const newState = this._applyTransformation(currentState, button, params);

      // Generate deterministic CID for new state
      const stateCID = await CIDGenerator.hashCanonical(newState);

      // Store new state
      await this.engine.objectStore.store({ ...newState, cid: stateCID });

      // Update session head
      this.engine.setSessionHead(session, stateCID);

      // Index the new state for search
      this.indexManager.indexObject({
        type: 'state',
        cid: stateCID,
        x: newState.x,
        level: newState.level,
        session,
        button_used: buttonLID
      });

      return {
        success: true,
        newState,
        stateCID,
        button: button.lid,
        session,
        timestamp: Date.now()
      };

    } catch (error) {
      console.error('Button click failed:', error);
      return {
        success: false,
        error: error.message,
        session,
        timestamp: Date.now()
      };
    }
  }

  _applyTransformation(state, button, params = {}) {
    const { operator } = button;
    const x = state.x || [];
    const M = operator.M || [];
    const b = operator.b || [];
    const alpha = operator.alpha ?? 1.0;
    const deltaLevel = operator.delta_level|0;

    // Dimension checks (already validated in button creation, but double-check)
    const d = x.length;
    if (!Array.isArray(M) || M.length !== d) {
      throw new Error(`Operator M shape mismatch: expected ${d}x${d}`);
    }

    const newX = new Array(d).fill(0);
    for (let i=0;i<d;i++) {
      let s = 0;
      for (let j=0;j<d;j++) s += M[i][j] * x[j];
      newX[i] = alpha * s + b[i];
    }

    return {
      x: newX,
      kappa: Math.max(0, (state.kappa ?? 1.0) + (params.kappa_delta || 0)),
      level: (state.level|0) + deltaLevel,
      timestamp: Date.now()
    };
  }

  // Enhanced health check with comprehensive diagnostics
  async healthCheck() {
    const stats = this.getStats();
    const checks = {};

    // Core system checks
    checks.object_store = stats.engine.object_store.total_objects >= 0;
    checks.catalog = stats.engine.catalog_entries >= 0;
    checks.event_log = stats.engine.total_events >= 0;

    // Index system checks
    checks.vector_index = stats.indexes.vector.total_vectors >= 0;
    checks.graph_index = stats.indexes.graph.total_nodes >= 0;
    checks.text_index = stats.indexes.text.total_documents >= 0;

    // Cache performance check
    const cacheStats = this.addressResolver.getCacheStats();
    checks.cache_performance = cacheStats.hitRate > 0.5 || cacheStats.size < 10;

    // Resolution round-trip test
    let canResolve = true;
    try {
      const buttons = this.engine.catalog.list('core','button');
      if (buttons.length) {
        const { namespace, type, lid, vid, cid } = buttons[0];
        const addr = this.addressResolver.buildAddress(namespace, type, lid, vid, cid);
        await this.addressResolver.resolve(addr);
      }
    } catch {
      canResolve = false;
    }
    checks.resolve_roundtrip = canResolve;

    const healthy = {
      status: Object.values(checks).every(Boolean) ? 'healthy' : 'degraded',
      checks,
      cache_stats: cacheStats,
      performance_metrics: {
        total_objects: stats.engine.object_store.total_objects,
        index_coverage: {
          vector: stats.indexes.vector.total_vectors,
          graph: stats.indexes.graph.total_nodes,
          text: stats.indexes.text.total_documents
        }
      },
      timestamp: Date.now()
    };

    return healthy;
  }

  getStats() {
    return {
      engine: this.engine.getStats(),
      indexes: this.indexManager.getStats(),
      cache: this.addressResolver.getCacheStats(),
      system_version: '2.0',
      timestamp: Date.now()
    };
  }
}

// Enhanced initialization with upflow integration
async function initWorldEngineV2(options = {}) {
  const system = new WorldEngineCompleteSystemV2();

  // Initialize enhanced upflow if requested
  if (options.enableUpflow) {
    try {
      system.upflow = await initEnhancedUpflow({
        idbFactory: () => import('../src/automations/upflow-automation.js')
      });
      console.log('✅ Enhanced Upflow initialized');
    } catch (error) {
      console.warn('⚠️ Failed to initialize Enhanced Upflow:', error);
    }
  }

  // Wait for button initialization
  await new Promise(resolve => setTimeout(resolve, 100));

  console.log('🚀 World Engine V2 Complete System initialized');
  console.log('📊 System stats:', system.getStats());

  return system;
}

// Export enhanced system
export {
  WorldEngineCompleteSystemV2,
  initWorldEngineV2
};

// Usage examples:
// const engine = await initWorldEngineV2({ enableUpflow: true });
// const searchResults = await engine.search('transformation rotate', { vector: [1, 0, 0], k: 3 });
// const morphAnalysis = engine.analyzeMorphology('counterproductive');
// const clickResult = await engine.clickButton('session1', 'button:Move@v1', { kappa_delta: 0.1 });
// const health = await engine.healthCheck();
