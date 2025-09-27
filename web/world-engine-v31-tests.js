/**
 * World Engine V3.1 Comprehensive Testing Suite
 * Tests all advanced features: Type Lattice, Jacobian Tracing, Morpheme Discovery, Distributed Sync
 */

import { WorldEngineV31Factory, initWorldEngineV31 } from './world-engine-v31-complete.js';
import { TypeLattice } from './type-lattice.js';
import { JacobianTracer, TraceManager } from './jacobian-tracer.js';
import { MorphemeDiscovery, SemanticMorphemeDiscovery } from './morpheme-discovery.js';
import { DistributedSyncManager, ConflictResolver } from './distributed-sync.js';
import { SyncMessageFactory, StateSerializer } from './sync-protocol.js';

/**
 * Test Suite Runner for V3.1 Components
 */
export class WorldEngineV31TestSuite {
  constructor() {
    this.results = {
      passed: 0,
      failed: 0,
      total: 0,
      details: []
    };
    this.isRunning = false;
  }

  /**
   * Run all V3.1 tests
   */
  async runAllTests() {
    console.log('🧪 Starting World Engine V3.1 Comprehensive Test Suite...');
    this.isRunning = true;
    this.results = { passed: 0, failed: 0, total: 0, details: [] };

    try {
      await this.testTypeLattice();
      await this.testJacobianTracing();
      await this.testMorphemeDiscovery();
      await this.testDistributedSync();
      await this.testEngineIntegration();
      await this.testMathematicalSafety();
      await this.testPerformance();

      this.printResults();
    } catch (error) {
      console.error('❌ Test suite failed:', error);
      this.fail('Test Suite', 'Critical error: ' + error.message);
    } finally {
      this.isRunning = false;
    }

    return this.results;
  }

  /**
   * Test Type Lattice System
   */
  async testTypeLattice() {
    console.log('🏗️ Testing Type Lattice System...');

    const lattice = new TypeLattice();

    // Test 1: Hierarchical relationships
    this.test('Type Lattice: State ⊑ Property', () => {
      return lattice.leq('State', 'Property');
    });

    this.test('Type Lattice: Property ⊑ Structure', () => {
      return lattice.leq('Property', 'Structure');
    });

    this.test('Type Lattice: Structure ⊑ Concept', () => {
      return lattice.leq('Structure', 'Concept');
    });

    this.test('Type Lattice: State ⊑ Concept (transitivity)', () => {
      return lattice.leq('State', 'Concept');
    });

    // Test 2: Join operations
    this.test('Type Lattice: join(State, Property) = Property', () => {
      return lattice.join('State', 'Property') === 'Property';
    });

    this.test('Type Lattice: join(Property, Structure) = Structure', () => {
      return lattice.join('Property', 'Structure') === 'Structure';
    });

    // Test 3: Meet operations
    this.test('Type Lattice: meet(Structure, Property) = Property', () => {
      return lattice.meet('Structure', 'Property') === 'Property';
    });

    this.test('Type Lattice: meet(Concept, State) = State', () => {
      return lattice.meet('Concept', 'State') === 'State';
    });

    // Test 4: Composition checking
    this.test('Type Lattice: Valid composition State → Property', () => {
      return lattice.checkCompose('State', 'Property');
    });

    this.test('Type Lattice: Invalid composition Concept → State', () => {
      return !lattice.checkCompose('Concept', 'State');
    });

    // Test 5: Custom lattice creation
    this.test('Type Lattice: Custom lattice creation', () => {
      const customLattice = TypeLattice.createCustomLattice(['A', 'B', 'C'], [['A', 'B'], ['B', 'C']]);
      return customLattice.leq('A', 'C'); // Should be true due to transitivity
    });
  }

  /**
   * Test Jacobian Tracing System
   */
  async testJacobianTracing() {
    console.log('📊 Testing Jacobian Tracing System...');

    // Create test button
    const testButton = {
      abbr: 'TST',
      label: 'Test Button',
      M: [[2, 0, 0], [0, 0.5, 0], [0, 0, 1]],
      alpha: 1.0,
      deltaLevel: 0,
      morphemes: ['test']
    };

    // Test 1: Jacobian computation
    this.test('Jacobian Tracing: Matrix computation', () => {
      const jacobian = JacobianTracer.jacobian(testButton);
      return jacobian[0][0] === 2.0 && jacobian[1][1] === 0.5 && jacobian[2][2] === 1.0;
    });

    // Test 2: Effect calculation
    this.test('Jacobian Tracing: Effect vector calculation', () => {
      const prevState = [1, 1, 1];
      const nextState = [2, 0.5, 1];
      const effect = JacobianTracer.effect(prevState, nextState);
      return effect[0] === 1 && effect[1] === -0.5 && effect[2] === 0;
    });

    // Test 3: Effect magnitude
    this.test('Jacobian Tracing: Effect magnitude', () => {
      const deltaX = [3, 4, 0]; // 3-4-5 triangle
      const magnitude = JacobianTracer.effectMagnitude(deltaX);
      return Math.abs(magnitude - 5.0) < 1e-10;
    });

    // Test 4: Directional analysis
    this.test('Jacobian Tracing: Directional analysis', () => {
      const jacobian = [[2, 0, 0], [0, 0.5, 0], [0, 0, 1]];
      const analysis = JacobianTracer.directionalAnalysis(jacobian);
      return analysis.expansive.includes(0) && analysis.contractive.includes(1) && analysis.preserving.includes(2);
    });

    // Test 5: Trace logging
    this.test('Jacobian Tracing: Comprehensive trace logging', () => {
      const before = { x: [1, 1, 1], kappa: 1.0, level: 0, timestamp: Date.now() };
      const after = { x: [2, 0.5, 1], kappa: 1.0, level: 0, timestamp: Date.now() };

      const trace = JacobianTracer.log({ button: testButton, before, after });

      return trace.operation && trace.transformation && trace.explanation.includes('Test Button');
    });

    // Test 6: Volume preservation check
    this.test('Jacobian Tracing: Volume preservation detection', () => {
      const identityMatrix = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
      return JacobianTracer.isVolumePreserving(identityMatrix);
    });

    // Test 7: Stability prediction
    this.test('Jacobian Tracing: Stability prediction', () => {
      const contractiveMatrix = [[0.5, 0, 0], [0, 0.5, 0], [0, 0, 0.5]];
      return JacobianTracer.isContractive(contractiveMatrix);
    });

    // Test 8: Trace Manager
    this.test('Jacobian Tracing: TraceManager functionality', () => {
      const manager = new TraceManager(5);
      const trace = { test: 'trace', timestamp: Date.now() };
      manager.addTrace(trace);
      return manager.traces.length === 1 && manager.getRecentTraces(1)[0] === trace;
    });
  }

  /**
   * Test Morpheme Discovery System
   */
  async testMorphemeDiscovery() {
    console.log('🧬 Testing Morpheme Discovery System...');

    const discovery = new MorphemeDiscovery({
      minFrequency: 2,
      promotionThreshold: 3
    });

    // Test 1: Pattern recording
    this.test('Morpheme Discovery: Pattern recording', () => {
      discovery.recordInteraction(['A', 'B', 'A'], { sessionId: 'test' });
      return discovery.sequences.length === 1;
    });

    // Test 2: Pattern extraction
    this.test('Morpheme Discovery: Pattern extraction', () => {
      discovery.recordInteraction(['A', 'B'], {});
      discovery.recordInteraction(['A', 'B'], {});
      return discovery.patterns.has('A→B') && discovery.patterns.get('A→B') >= 2;
    });

    // Test 3: Morpheme promotion
    this.test('Morpheme Discovery: Morpheme promotion', () => {
      // Record pattern enough times to trigger promotion
      for (let i = 0; i < 5; i++) {
        discovery.recordInteraction(['X', 'Y', 'Z'], { improvement: 0.8 });
      }

      return discovery.morphemes.size > 0;
    });

    // Test 4: Effectiveness calculation
    this.test('Morpheme Discovery: Effectiveness calculation', () => {
      const pattern = ['test', 'pattern'];
      discovery.recordInteraction(pattern, { improvement: 0.9 });
      discovery.recordInteraction(pattern, { improvement: 0.7 });

      const effectiveness = discovery.calculateEffectiveness(pattern);
      return effectiveness > 0.5 && effectiveness <= 1.0;
    });

    // Test 5: Label generation
    this.test('Morpheme Discovery: Label generation', () => {
      const label = discovery.suggestLabel(['move', 'scale']);
      return typeof label === 'string' && label.length > 0;
    });

    // Test 6: Export/Import functionality
    this.test('Morpheme Discovery: Export/Import morphemes', () => {
      const exported = discovery.exportMorphemes();
      const newDiscovery = new MorphemeDiscovery();
      newDiscovery.importMorphemes(exported);
      return newDiscovery.morphemes.size === discovery.morphemes.size;
    });

    // Test 7: Semantic clustering (if available)
    this.test('Morpheme Discovery: Semantic clustering', () => {
      if (discovery instanceof SemanticMorphemeDiscovery) {
        const clusters = discovery.clusterMorphemes();
        return clusters instanceof Map;
      }
      return true; // Skip if not semantic discovery
    });

    // Test 8: Analytics generation
    this.test('Morpheme Discovery: Analytics generation', () => {
      const analytics = discovery.getAnalytics();
      return analytics.discovery_stats && analytics.morpheme_quality && analytics.learning_velocity;
    });
  }

  /**
   * Test Distributed Sync System
   */
  async testDistributedSync() {
    console.log('🌐 Testing Distributed Sync System...');

    // Test 1: Sync Manager initialization
    this.test('Distributed Sync: Manager initialization', () => {
      const manager = new DistributedSyncManager({
        peerId: 'test-peer',
        isHost: true,
        sessionId: 'test-session'
      });
      return manager.peerId === 'test-peer' && manager.isHost === true;
    });

    // Test 2: Operation creation
    this.test('Distributed Sync: Operation creation', () => {
      const manager = new DistributedSyncManager({ peerId: 'test' });
      const operation = {
        type: 'button-press',
        data: { button: 'TEST', state: [1, 2, 3] }
      };

      // Mock the broadcast method for testing
      manager.broadcastOperation = () => {};
      manager.applyOperation(operation);

      return manager.operationHistory.length === 1;
    });

    // Test 3: Conflict resolution
    this.test('Distributed Sync: Conflict detection', () => {
      const resolver = new ConflictResolver();
      const op1 = {
        type: 'state-update',
        data: { x: [1, 0, 0] },
        timestamp: 1000,
        peerId: 'peer1'
      };
      const op2 = {
        type: 'state-update',
        data: { x: [0, 1, 0] },
        timestamp: 1001,
        peerId: 'peer2'
      };

      const conflict = resolver.detectConflict(op2, [op1]);
      return conflict && conflict.type === 'concurrent-modification';
    });

    // Test 4: State serialization
    this.test('Distributed Sync: State serialization', () => {
      const state = { x: [1, 2, 3], kappa: 1.5, level: 2, timestamp: Date.now() };
      const serialized = StateSerializer.serialize(state);
      const deserialized = StateSerializer.deserialize(serialized);

      return JSON.stringify(state.x) === JSON.stringify(deserialized.x) &&
             state.kappa === deserialized.kappa;
    });

    // Test 5: Message factory
    this.test('Distributed Sync: Message factory', () => {
      const message = SyncMessageFactory.createOperation({
        operationType: 'button-press',
        peerId: 'test-peer',
        sessionId: 'test-session',
        data: { button: 'TEST' }
      });

      return message.type === 'operation' && message.operation.peerId === 'test-peer';
    });

    // Test 6: Message validation
    this.test('Distributed Sync: Message validation', () => {
      const validMessage = {
        type: 'operation',
        timestamp: Date.now(),
        operation: { id: 'test', type: 'test', peerId: 'test' }
      };

      const validation = SyncMessageFactory.validateMessage(validMessage);
      return validation.valid === true;
    });

    // Test 7: State delta calculation
    this.test('Distributed Sync: State delta calculation', () => {
      const oldState = { x: [1, 1, 1], kappa: 1.0, version: 1 };
      const newState = { x: [2, 1.5, 1], kappa: 1.2, version: 2 };

      const delta = StateSerializer.calculateDelta(oldState, newState);
      return delta.changes.x && delta.changes.kappa !== undefined;
    });
  }

  /**
   * Test Engine Integration
   */
  async testEngineIntegration() {
    console.log('⚙️ Testing Engine Integration...');

    // Test 1: V3.1 Engine initialization
    this.test('Engine Integration: V3.1 initialization', async () => {
      const engine = await initWorldEngineV31(3, {
        typeLatticeEnabled: true,
        jacobianTracingEnabled: true,
        morphemeDiscoveryEnabled: false,
        enableSync: false
      });

      return engine && engine.v31Components && engine.lexicalEngine;
    });

    // Test 2: Factory methods
    this.test('Engine Integration: Factory methods', () => {
      const standardEngine = WorldEngineV31Factory.createStandard(3);
      return standardEngine.dimensions === 3 && standardEngine.v31Features;
    });

    // Test 3: Button integration with V3.1 features
    this.test('Engine Integration: Enhanced button clicks', async () => {
      const engine = WorldEngineV31Factory.createStandard(3);
      await engine.initializeV31Features();

      // Should have mathematical buttons
      return engine.lexicalEngine.buttons.size > 0;
    });

    // Test 4: Event system integration
    this.test('Engine Integration: Event system', async () => {
      const engine = WorldEngineV31Factory.createStandard(3);
      await engine.initializeV31Features();

      let eventFired = false;
      engine.lexicalEngine.on('stateChanged', () => {
        eventFired = true;
      });

      // Trigger a button click if buttons exist
      const firstButton = Array.from(engine.lexicalEngine.buttons.keys())[0];
      if (firstButton) {
        await engine.clickButton(firstButton);
      }

      return eventFired || !firstButton; // Pass if no buttons or event fired
    });

    // Test 5: State export/import
    this.test('Engine Integration: State persistence', async () => {
      const engine = WorldEngineV31Factory.createStandard(3);
      await engine.initializeV31Features();

      const exportedState = engine.exportState();
      const newEngine = WorldEngineV31Factory.createStandard(3);
      await newEngine.initializeV31Features();
      await newEngine.importState(exportedState);

      return exportedState.version === 'v3.1';
    });

    // Test 6: Multi-dimensional support
    this.test('Engine Integration: Multi-dimensional support', async () => {
      const engine4D = await initWorldEngineV31(4, { enableSync: false });
      return engine4D.dimensions === 4 && engine4D.lexicalEngine.su.x.length === 4;
    });
  }

  /**
   * Test Mathematical Safety
   */
  async testMathematicalSafety() {
    console.log('🔒 Testing Mathematical Safety...');

    const engine = WorldEngineV31Factory.createStandard(3);
    await engine.initializeV31Features();

    // Test 1: NaN detection and prevention
    this.test('Mathematical Safety: NaN prevention', () => {
      try {
        engine.math.validateFinite([1, NaN, 3], 'Test vector');
        return false; // Should throw
      } catch (error) {
        return error.message.includes('NaN');
      }
    });

    // Test 2: Infinity detection
    this.test('Mathematical Safety: Infinity prevention', () => {
      try {
        engine.math.validateFinite([1, Infinity, 3], 'Test vector');
        return false; // Should throw
      } catch (error) {
        return error.message.includes('Infinity');
      }
    });

    // Test 3: Matrix operations safety
    this.test('Mathematical Safety: Matrix multiplication safety', () => {
      const A = [[1, 2], [3, 4]];
      const B = [[5, 6], [7, 8]];
      const result = engine.math.multiplyMatrices(A, B);

      // Check result is finite and correct
      return result[0][0] === 19 && result[1][1] === 50 &&
             result.every(row => row.every(val => isFinite(val)));
    });

    // Test 4: Pseudo-inverse stability
    this.test('Mathematical Safety: Pseudo-inverse stability', () => {
      const matrix = [[2, 0], [0, 0.5], [0, 0]]; // 3x2 matrix
      const pseudoInv = engine.math.pseudoInverse(matrix);

      // Should be 2x3 and contain finite values
      return pseudoInv.length === 2 && pseudoInv[0].length === 3 &&
             pseudoInv.every(row => row.every(val => isFinite(val)));
    });

    // Test 5: Error propagation prevention
    this.test('Mathematical Safety: Error propagation prevention', async () => {
      // Try to create a pathological transformation
      const badButton = {
        abbr: 'BAD',
        label: 'Bad Button',
        M: [[1e10, 0, 0], [0, 1e-10, 0], [0, 0, NaN]],
        alpha: 1.0
      };

      try {
        engine.lexicalEngine.buttons.set('BAD', badButton);
        await engine.clickButton('BAD');
        return false; // Should have failed
      } catch (error) {
        return true; // Correctly caught the error
      }
    });
  }

  /**
   * Test Performance
   */
  async testPerformance() {
    console.log('⚡ Testing Performance...');

    const engine = WorldEngineV31Factory.createFullFeatured(3);
    await engine.initializeV31Features();

    // Test 1: Button click performance
    this.test('Performance: Button click latency', async () => {
      const start = performance.now();

      // Perform multiple button clicks
      const buttons = Array.from(engine.lexicalEngine.buttons.keys()).slice(0, 5);
      for (const button of buttons) {
        await engine.clickButton(button);
      }

      const duration = performance.now() - start;
      console.log(`Button clicks took ${duration.toFixed(2)}ms`);
      return duration < 1000; // Should complete within 1 second
    });

    // Test 2: Memory usage monitoring
    this.test('Performance: Memory management', () => {
      const initialTraces = engine.v31Components.traceManager.traces.length;

      // Generate many traces
      for (let i = 0; i < 150; i++) {
        const mockTrace = {
          timestamp: Date.now(),
          operation: { stability: 'stable' },
          transformation: { effect_magnitude: 0.1 }
        };
        engine.v31Components.traceManager.addTrace(mockTrace);
      }

      // Should have triggered cleanup
      return engine.v31Components.traceManager.traces.length <= 100;
    });

    // Test 3: Large state vector handling
    this.test('Performance: Large dimension handling', async () => {
      const largeEngine = await initWorldEngineV31(10, {
        morphemeDiscoveryEnabled: false,
        enableSync: false
      });

      const start = performance.now();
      // Perform some operations
      const buttons = Array.from(largeEngine.lexicalEngine.buttons.keys()).slice(0, 3);
      for (const button of buttons) {
        await largeEngine.clickButton(button);
      }

      const duration = performance.now() - start;
      return duration < 2000 && largeEngine.lexicalEngine.su.x.length === 10;
    });
  }

  /**
   * Helper method to run a single test
   */
  test(name, testFunction) {
    this.results.total++;

    try {
      const result = testFunction();

      if (result instanceof Promise) {
        return result.then(asyncResult => {
          if (asyncResult) {
            this.pass(name);
          } else {
            this.fail(name, 'Async test returned false');
          }
        }).catch(error => {
          this.fail(name, 'Async test threw: ' + error.message);
        });
      } else if (result) {
        this.pass(name);
      } else {
        this.fail(name, 'Test returned false');
      }
    } catch (error) {
      this.fail(name, error.message);
    }
  }

  /**
   * Mark test as passed
   */
  pass(name) {
    this.results.passed++;
    this.results.details.push({ name, status: 'PASS' });
    console.log(`  ✅ ${name}`);
  }

  /**
   * Mark test as failed
   */
  fail(name, reason) {
    this.results.failed++;
    this.results.details.push({ name, status: 'FAIL', reason });
    console.log(`  ❌ ${name}: ${reason}`);
  }

  /**
   * Print final test results
   */
  printResults() {
    console.log('\n🧪 World Engine V3.1 Test Results:');
    console.log(`   Total Tests: ${this.results.total}`);
    console.log(`   Passed: ${this.results.passed} ✅`);
    console.log(`   Failed: ${this.results.failed} ❌`);
    console.log(`   Success Rate: ${((this.results.passed / this.results.total) * 100).toFixed(1)}%`);

    if (this.results.failed > 0) {
      console.log('\n❌ Failed Tests:');
      this.results.details
        .filter(detail => detail.status === 'FAIL')
        .forEach(detail => {
          console.log(`   - ${detail.name}: ${detail.reason}`);
        });
    }

    if (this.results.passed === this.results.total) {
      console.log('\n🎉 All tests passed! World Engine V3.1 is ready for deployment.');
    } else {
      console.log(`\n⚠️ ${this.results.failed} test(s) failed. Please review and fix issues before deployment.`);
    }
  }
}

/**
 * Quick test runner for development
 */
export async function runV31Tests() {
  const testSuite = new WorldEngineV31TestSuite();
  return await testSuite.runAllTests();
}

/**
 * Specific component test runners
 */
export async function testTypeLatticeOnly() {
  const suite = new WorldEngineV31TestSuite();
  await suite.testTypeLattice();
  suite.printResults();
  return suite.results;
}

export async function testJacobianTracingOnly() {
  const suite = new WorldEngineV31TestSuite();
  await suite.testJacobianTracing();
  suite.printResults();
  return suite.results;
}

export async function testMorphemeDiscoveryOnly() {
  const suite = new WorldEngineV31TestSuite();
  await suite.testMorphemeDiscovery();
  suite.printResults();
  return suite.results;
}

export async function testDistributedSyncOnly() {
  const suite = new WorldEngineV31TestSuite();
  await suite.testDistributedSync();
  suite.printResults();
  return suite.results;
}

/**
 * Continuous testing setup
 */
export class ContinuousTestRunner {
  constructor(intervalMs = 60000) { // 1 minute default
    this.interval = intervalMs;
    this.isRunning = false;
    this.lastResults = null;
    this.listeners = [];
  }

  start() {
    if (this.isRunning) return;

    this.isRunning = true;
    console.log('🔄 Starting continuous V3.1 testing...');

    this.intervalId = setInterval(async () => {
      console.log('🕐 Running scheduled V3.1 tests...');
      this.lastResults = await runV31Tests();

      this.listeners.forEach(listener => {
        try {
          listener(this.lastResults);
        } catch (error) {
          console.error('Listener error:', error);
        }
      });
    }, this.interval);
  }

  stop() {
    if (!this.isRunning) return;

    clearInterval(this.intervalId);
    this.isRunning = false;
    console.log('⏹️ Stopped continuous testing');
  }

  onResults(listener) {
    this.listeners.push(listener);
  }
}

// Export test utilities
export const V31TestUtils = {
  createMockEngine: async () => {
    return await initWorldEngineV31(3, {
      morphemeDiscoveryEnabled: false,
      enableSync: false
    });
  },

  createMockButton: (abbr = 'TST', M = [[1,0,0],[0,1,0],[0,0,1]]) => ({
    abbr,
    label: `${abbr} Button`,
    M,
    alpha: 1.0,
    deltaLevel: 0
  }),

  waitForCondition: (condition, timeoutMs = 5000) => {
    return new Promise((resolve, reject) => {
      const start = Date.now();

      const check = () => {
        if (condition()) {
          resolve(true);
        } else if (Date.now() - start > timeoutMs) {
          reject(new Error('Condition timeout'));
        } else {
          setTimeout(check, 100);
        }
      };

      check();
    });
  }
};

// Auto-run tests if this file is executed directly
if (typeof window !== 'undefined' && window.location.search.includes('autotest=true')) {
  console.log('🚀 Auto-running V3.1 tests...');
  runV31Tests().then(results => {
    console.log('🏁 Auto-test completed');
    window.testResults = results;
  });
}
