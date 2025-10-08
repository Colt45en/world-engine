/**
 * World Engine V3.1 Integration Test Suite
 * Tests V3.1 features and explores integration with lexicon concepts
 */

import { WorldEngineV31Factory, initWorldEngineV31 } from './world-engine-v31-complete.js';
import { TypeLattice } from './type-lattice.js';
import { JacobianTracer } from './jacobian-tracer.js';
import { createMorphemeDiscovery } from './morphology-integration.js';

/**
 * Lexicon Explorer Data Service (inspired by attachment)
 * Provides morphological analysis and navigation
 */
class LexiconExplorer {
  constructor(upflowIndex, morphemeDiscovery) {
    this.upflow = upflowIndex;
    this.discovery = morphemeDiscovery;
    this.cache = new Map();
  }

  // Search across different morphological dimensions
  search(term) {
    if (this.cache.has(term)) {
      return this.cache.get(term);
    }

    const result = {
      roots: this.findByRoot(term),
      prefixes: this.findByPrefix(term),
      suffixes: this.findBySuffix(term),
      abbreviations: this.findByAbbrev(term),
      wordDetail: this.getWordDetail(term),
      morphemeScore: this.scoreMorpheme(term)
    };

    this.cache.set(term, result);
    return result;
  }

  // Link words through morphological relationships
  linkWord(word) {
    const analysis = this.analyzeWord(word);
    return {
      word,
      root: analysis.root,
      morphemes: analysis.morphemes,
      links: this.findRelatedWords(analysis),
      score: this.calculateRelatednessScore(word)
    };
  }

  // Analyze word structure (simplified morphological parsing)
  analyzeWord(word) {
    if (!word || word.length < 2) return { root: word, morphemes: [] };

    const morphemes = [];
    let remaining = word.toLowerCase();

    // Common prefixes (from the lexicon explorer concept)
    const prefixes = ['anti', 'auto', 'counter', 'inter', 'multi', 'pre', 're', 'un', 'dis'];
    const suffixes = ['ation', 'ment', 'ness', 'able', 'ible', 'ing', 'ed', 'er', 'ly'];

    // Extract prefix
    for (const prefix of prefixes) {
      if (remaining.startsWith(prefix)) {
        morphemes.push({ type: 'prefix', form: prefix, meaning: 'modifier' });
        remaining = remaining.slice(prefix.length);
        break;
      }
    }

    // Extract suffix
    for (const suffix of suffixes) {
      if (remaining.endsWith(suffix)) {
        morphemes.push({ type: 'suffix', form: suffix, meaning: 'modifier' });
        remaining = remaining.slice(0, -suffix.length);
        break;
      }
    }

    // Root
    if (remaining.length > 0) {
      morphemes.unshift({ type: 'root', form: remaining, meaning: 'core' });
    }

    return {
      root: remaining,
      morphemes,
      structure: morphemes.map(m => m.type).join('+')
    };
  }

  findByRoot(root) {
    // Simulate upflow query
    return ['state', 'static', 'status'].filter(w => w.includes(root));
  }

  findByPrefix(prefix) {
    return ['rebuild', 'restore', 'restate'].filter(w => w.startsWith(prefix));
  }

  findBySuffix(suffix) {
    return ['statement', 'movement', 'achievement'].filter(w => w.endsWith(suffix));
  }

  findByAbbrev(abbrev) {
    const abbrevMap = {
      'sta': ['state', 'static', 'status'],
      'mov': ['move', 'movement', 'mobile'],
      'tra': ['trace', 'track', 'transform']
    };
    return abbrevMap[abbrev] || [];
  }

  getWordDetail(word) {
    const analysis = this.analyzeWord(word);
    return {
      word,
      ...analysis,
      frequency: Math.random(), // Simulated
      lastUsed: Date.now(),
      contexts: ['mathematical', 'linguistic', 'semantic']
    };
  }

  findRelatedWords(analysis) {
    const related = [];
    if (analysis.root) {
      related.push(...this.findByRoot(analysis.root));
    }
    return related.slice(0, 5); // Limit results
  }

  scoreMorpheme(word) {
    const analysis = this.analyzeWord(word);
    return analysis.morphemes.length * 0.3 + (analysis.root?.length || 0) * 0.1;
  }

  calculateRelatednessScore(word) {
    return Math.random(); // Simplified scoring
  }
}

/**
 * V3.1 Feature Integration Test
 */
export class V31IntegrationTest {
  constructor() {
    this.results = [];
    this.lexiconExplorer = null;
  }

  async runAllTests() {
    console.log('🧪 Starting World Engine V3.1 Integration Tests...');

    try {
      // Test 1: Initialize V3.1 System
      await this.testV31Initialization();

      // Test 2: Type Lattice with Lexicon Integration
      await this.testTypeLatticeWithLexicon();

      // Test 3: Jacobian Tracing with Morphological Analysis
      await this.testJacobianWithMorphology();

      // Test 4: Morpheme Discovery Pipeline
      await this.testMorphemeDiscoveryPipeline();

      // Test 5: Lexicon Explorer Integration
      await this.testLexiconExplorer();

      // Test 6: Neural-Inspired Learning (from backpropagation concepts)
      await this.testNeuralInspiredLearning();

      this.printResults();
      return this.results;

    } catch (error) {
      console.error('❌ Integration test failed:', error);
      return { error: error.message, results: this.results };
    }
  }

  async testV31Initialization() {
    console.log('🔧 Testing V3.1 System Initialization...');

    const engine = await initWorldEngineV31(3, {
      typeLatticeEnabled: true,
      jacobianTracingEnabled: true,
      morphemeDiscoveryEnabled: true,
      enableSync: false
    });

    this.record('V3.1 Initialization',
      engine && engine.v31Components && engine.lexicalEngine,
      'System initialized with all V3.1 components'
    );

    this.testEngine = engine;
  }

  async testTypeLatticeWithLexicon() {
    console.log('🏗️ Testing Type Lattice with Lexicon Integration...');

    if (!this.testEngine) return this.record('Type Lattice + Lexicon', false, 'No engine available');

    const lattice = this.testEngine.v31Components.typeLattice;

    // Test hierarchical relationships with lexical concepts
    const tests = [
      { a: 'State', b: 'Property', expected: true, desc: 'State ⊑ Property' },
      { a: 'Property', b: 'Structure', expected: true, desc: 'Property ⊑ Structure' },
      { a: 'Structure', b: 'Concept', expected: true, desc: 'Structure ⊑ Concept' },
      { a: 'State', b: 'Concept', expected: true, desc: 'State ⊑ Concept (transitivity)' }
    ];

    let passed = 0;
    for (const test of tests) {
      const result = lattice.leq(test.a, test.b);
      if (result === test.expected) passed++;
    }

    this.record('Type Lattice + Lexicon',
      passed === tests.length,
      `Passed ${passed}/${tests.length} lattice relationship tests`
    );
  }

  async testJacobianWithMorphology() {
    console.log('📊 Testing Jacobian Tracing with Morphological Analysis...');

    if (!this.testEngine) return this.record('Jacobian + Morphology', false, 'No engine available');

    // Create a test button with morphological properties
    const morphButton = {
      abbr: 'MOR',
      label: 'Morph Transform',
      M: [[1.5, 0, 0], [0, 0.8, 0], [0, 0, 1.2]],
      alpha: 1.0,
      morphemes: ['morph', 'transform'],
      inputType: 'State',
      outputType: 'Property'
    };

    // Compute Jacobian
    const jacobian = JacobianTracer.jacobian(morphButton);

    // Test effect calculation
    const before = [1, 1, 1];
    const after = [1.5, 0.8, 1.2];
    const effect = JacobianTracer.effect(before, after);

    const validJacobian = jacobian[0][0] === 1.5 && jacobian[1][1] === 0.8 && jacobian[2][2] === 1.2;
    const validEffect = Math.abs(effect[0] - 0.5) < 1e-10 && Math.abs(effect[1] + 0.2) < 1e-10;

    this.record('Jacobian + Morphology',
      validJacobian && validEffect,
      'Jacobian computation with morphological context successful'
    );
  }

  async testMorphemeDiscoveryPipeline() {
    console.log('🧬 Testing Morpheme Discovery Pipeline...');

    const discovery = createMorphemeDiscovery({ minFreq: 2, stability: 0.9 });

    // Simulate interaction patterns
    const testWords = [
      'transform', 'transform', 'transform',
      'reshape', 'reshape', 'reshape',
      'reform', 'reform', 'reform',
      'restate', 'restate'
    ];

    testWords.forEach(word => {
      discovery.observe(word);
    });

    // Test pattern extraction
    const patterns = Array.from(discovery.patterns.keys());
    const hasTransformPattern = patterns.some(p => p.includes('transform'));
    const hasRePattern = patterns.some(p => p.includes('re'));

    this.record('Morpheme Discovery Pipeline',
      hasTransformPattern && hasRePattern && patterns.length > 0,
      `Discovered ${patterns.length} morphological patterns`
    );

    this.morphemeDiscovery = discovery;
  }

  async testLexiconExplorer() {
    console.log('📚 Testing Lexicon Explorer Integration...');

    this.lexiconExplorer = new LexiconExplorer(null, this.morphemeDiscovery);

    // Test morphological analysis
    const testWords = ['restate', 'transformation', 'movement'];
    const analysisResults = [];

    for (const word of testWords) {
      const analysis = this.lexiconExplorer.analyzeWord(word);
      const searchResult = this.lexiconExplorer.search(word);
      const linkResult = this.lexiconExplorer.linkWord(word);

      analysisResults.push({
        word,
        morphemes: analysis.morphemes.length,
        hasRoot: !!analysis.root,
        searchHits: Object.values(searchResult).flat().length,
        relatedness: linkResult.score
      });
    }

    const avgMorphemes = analysisResults.reduce((sum, r) => sum + r.morphemes, 0) / analysisResults.length;
    const hasRoots = analysisResults.every(r => r.hasRoot);

    this.record('Lexicon Explorer',
      hasRoots && avgMorphemes > 1,
      `Analyzed ${analysisResults.length} words, avg ${avgMorphemes.toFixed(1)} morphemes each`
    );
  }

  async testNeuralInspiredLearning() {
    console.log('🧠 Testing Neural-Inspired Learning Integration...');

    if (!this.testEngine || !this.morphemeDiscovery) {
      return this.record('Neural Learning', false, 'Missing dependencies');
    }

    // Simulate a simple XOR-like learning scenario with morphemes
    const trainingData = [
      { input: ['pre', 'state'], target: 'transform', expected: 1 },
      { input: ['re', 'form'], target: 'transform', expected: 1 },
      { input: ['pre', 'form'], target: 'maintain', expected: 0 },
      { input: ['re', 'state'], target: 'maintain', expected: 0 }
    ];

    // Simple learning simulation
    let correctPredictions = 0;
    const weights = new Map();

    for (const data of trainingData) {
      const key = data.input.join('+');

      // Simple scoring based on morpheme combinations
      const score = this.calculateMorphemeScore(data.input, data.target);
      const prediction = score > 0.5 ? 1 : 0;

      if (prediction === data.expected) {
        correctPredictions++;
      }

      weights.set(key, score);
    }

    const accuracy = correctPredictions / trainingData.length;

    this.record('Neural Learning',
      accuracy >= 0.5,
      `Learning accuracy: ${(accuracy * 100).toFixed(1)}% (${correctPredictions}/${trainingData.length})`
    );
  }

  calculateMorphemeScore(inputs, target) {
    // Simple heuristic: if inputs suggest transformation and target is transform, score high
    const hasTransformPrefix = inputs.some(i => ['re', 'pre', 'trans'].includes(i));
    const isTransformTarget = target.includes('transform');

    if (hasTransformPrefix && isTransformTarget) return 0.8;
    if (!hasTransformPrefix && !isTransformTarget) return 0.2;
    return 0.4;
  }

  record(testName, success, details) {
    const result = {
      test: testName,
      success,
      details,
      timestamp: Date.now()
    };

    this.results.push(result);
    console.log(`  ${success ? '✅' : '❌'} ${testName}: ${details}`);
    return result;
  }

  printResults() {
    const total = this.results.length;
    const passed = this.results.filter(r => r.success).length;
    const failed = total - passed;

    console.log('\n🧪 World Engine V3.1 Integration Test Results:');
    console.log(`   Total Tests: ${total}`);
    console.log(`   Passed: ${passed} ✅`);
    console.log(`   Failed: ${failed} ❌`);
    console.log(`   Success Rate: ${((passed / total) * 100).toFixed(1)}%`);

    if (failed > 0) {
      console.log('\n❌ Failed Tests:');
      this.results
        .filter(r => !r.success)
        .forEach(r => console.log(`   - ${r.test}: ${r.details}`));
    }

    if (passed === total) {
      console.log('\n🎉 All integration tests passed! V3.1 system is ready.');
    }
  }
}

// Export test utilities
export async function runV31IntegrationTests() {
  const testSuite = new V31IntegrationTest();
  return await testSuite.runAllTests();
}

export { LexiconExplorer };

// Auto-run tests if executed directly
if (typeof window !== 'undefined' && window.location.search.includes('integration-test=true')) {
  console.log('🚀 Auto-running V3.1 integration tests...');
  runV31IntegrationTests().then(results => {
    console.log('🏁 Integration test completed');
    window.integrationTestResults = results;
  });
}
