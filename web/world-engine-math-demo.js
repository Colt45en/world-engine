/**
 * World Engine V3 Mathematical Demo
 * Demonstrates all surgical mathematical upgrades in action
 */

import { WorldEngineV3Factory, LLEMath, Morpheme, Button } from './world-engine-v3-mathematical.js';

async function runMathematicalDemo() {
  console.log('🧮 World Engine V3 Mathematical Demo');
  console.log('=====================================\n');

  // 1. Create dimension-flexible engines
  console.log('1. Creating dimension-flexible engines...');
  const engine2D = await WorldEngineV3Factory.create2D();
  const engine3D = await WorldEngineV3Factory.create3D();
  const engine4D = await WorldEngineV3Factory.create4D();

  console.log(`✅ 2D Engine: ${engine2D.dimensions}D, buttons: ${engine2D.lexicalEngine.buttons.size}`);
  console.log(`✅ 3D Engine: ${engine3D.dimensions}D, buttons: ${engine3D.lexicalEngine.buttons.size}`);
  console.log(`✅ 4D Engine: ${engine4D.dimensions}D, buttons: ${engine4D.lexicalEngine.buttons.size}\n`);

  // 2. Demonstrate mathematical safety
  console.log('2. Testing mathematical safety...');

  // Shape validation
  try {
    const badMatrix = [[1, 0], [0, 1], [1, 1]]; // Wrong shape
    const vector = [1, 1, 1];
    LLEMath.multiply(badMatrix, vector);
  } catch (error) {
    console.log(`✅ Shape validation caught: ${error.message}`);
  }

  // Pseudo-inverse demonstration
  const tallMatrix = [[1, 0, 0.5], [0, 1, 0.3], [0.2, 0.1, 1], [0.1, 0.2, 0.8]];
  const pseudoInv = LLEMath.pseudoInverse(tallMatrix);
  console.log(`✅ Pseudo-inverse computed for ${tallMatrix.length}×${tallMatrix[0].length} matrix`);
  console.log(`   Result dimensions: ${pseudoInv.length}×${pseudoInv[0].length}\n`);

  // 3. Morpheme-driven button composition
  console.log('3. Demonstrating morpheme-driven composition...');
  const morphemes = Morpheme.createBuiltInMorphemes(3);
  console.log(`✅ Created ${morphemes.size} morphemes`);

  const customButton = new Button(
    'Rebuild-Multi', 'RM', 'Action',
    ['re', 'build', 'multi'],
    { dimensions: 3 },
    morphemes
  );

  console.log(`✅ Custom button composed from morphemes: ${customButton.morphemes.join('+')}`);
  console.log(`   Matrix diagonal: [${customButton.M.map((row, i) => row[i].toFixed(2)).join(', ')}]`);
  console.log(`   Bias vector: [${customButton.b.map(v => v.toFixed(2)).join(', ')}]\n`);

  // 4. Safe operations with validation
  console.log('4. Testing safe operations...');
  const initial3D = engine3D.lexicalEngine.su.copy();
  console.log(`Initial 3D state: ${initial3D.toString()}`);

  // Safe button click
  const clickResult = await engine3D.safeClickButton('MO');
  console.log(`✅ Safe click result: ${clickResult.success ? 'SUCCESS' : 'FAILED'}`);
  if (clickResult.success) {
    console.log(`   New state: ${clickResult.result.toString()}`);
  }

  // Preview composition without mutation
  const previewOps = ['RB', 'SC', { type: 'upscale', toDim: 4 }];
  const preview = await engine3D.previewComposition(previewOps);
  console.log(`✅ Preview composition: ${preview.success ? 'SUCCESS' : 'FAILED'}`);
  console.log(`   Current state unchanged: ${engine3D.lexicalEngine.su.toString()}`);
  if (preview.mathematical_analysis) {
    console.log(`   Mathematical stability: ${preview.mathematical_analysis.mathematical_stability}`);
    console.log(`   Recovery possible: ${preview.mathematical_analysis.recovery_possible}\n`);
  }

  // 5. Scaling with pseudo-inverse recovery
  console.log('5. Testing scaling with pseudo-inverse recovery...');

  // Downscale 3D → 2D
  const downResult = await engine3D.safeDownscale([0, 2]);
  if (downResult.success) {
    console.log(`✅ Downscale: ${downResult.before_dimensions}D → ${downResult.after_dimensions}D`);
    console.log(`   State: ${downResult.result.toString()}`);
  }

  // Upscale back with pseudo-inverse
  const upResult = await engine3D.safeUpscale(null, 3);
  if (upResult.success) {
    console.log(`✅ Upscale: ${upResult.before_dimensions}D → ${upResult.after_dimensions}D`);
    console.log(`   Pseudo-inverse used: ${upResult.pseudo_inverse_used}`);
    console.log(`   Final state: ${upResult.result.toString()}\n`);
  }

  // 6. Undo/Redo functionality
  console.log('6. Testing undo functionality...');
  const beforeUndo = engine3D.lexicalEngine.su.copy();
  const undoResult = await engine3D.safeUndo();

  if (undoResult.success && undoResult.restored) {
    console.log('✅ Undo successful, state restored');
    console.log(`   Current: ${undoResult.result.toString()}`);
  } else {
    console.log('⚠️  No undoable operation or undo failed\n');
  }

  // 7. Mathematical search with context
  console.log('7. Testing mathematical search...');
  const searchResult = await engine3D.mathematicalSearch('transformation stability matrix');
  console.log('✅ Mathematical search completed');
  console.log(`   Results found: ${searchResult.results?.length || 0}`);
  if (searchResult.mathematical_context) {
    console.log(`   Current dimensions: ${searchResult.mathematical_context.current_dimensions}`);
    console.log(`   Available operations: ${searchResult.mathematical_context.available_operations.length}`);
    console.log(`   Mathematical stability: ${JSON.stringify(searchResult.mathematical_context.mathematical_stability)}\n`);
  }

  // 8. Comprehensive testing
  console.log('8. Running comprehensive mathematical tests...');
  const testResults = await engine3D.runMathematicalTests();

  console.log('✅ Test Results:');
  console.log(`   Basic tests: ${testResults.basic_tests.summary.passed}/${testResults.basic_tests.summary.total} passed`);
  console.log(`   Mathematical tests: ${testResults.mathematical_tests.filter(t => t.success).length}/${testResults.mathematical_tests.length} passed`);
  console.log(`   Overall health: ${testResults.overall_health.health_status} (${(testResults.overall_health.overall_score * 100).toFixed(1)}%)`);

  // Print individual test results
  testResults.mathematical_tests.forEach(test => {
    console.log(`   ${test.success ? '✅' : '❌'} ${test.name}: ${test.success ? 'PASS' : test.error}`);
  });

  console.log('\n🎉 Mathematical Demo Complete!');
  console.log('=====================================');

  return {
    engines: { engine2D, engine3D, engine4D },
    testResults,
    demonstrations: [
      'dimension-flexibility',
      'mathematical-safety',
      'morpheme-composition',
      'safe-operations',
      'pseudo-inverse-scaling',
      'undo-redo',
      'mathematical-search',
      'comprehensive-testing'
    ]
  };
}

// Quick self-test for critical operations
async function quickSelfTest() {
  console.log('🔬 Quick Self-Test Suite');
  console.log('========================');

  const tests = [
    {
      name: 'Matrix Math Safety',
      test: () => {
        const A = [[1, 2], [3, 4]];
        const b = [5, 6];
        const result = LLEMath.multiply(A, b);
        return result.length === 2 && result.every(Number.isFinite);
      }
    },
    {
      name: 'Pseudo-Inverse Stability',
      test: () => {
        const A = [[1, 0], [0, 1], [1, 1]];
        const Aplus = LLEMath.pseudoInverse(A);
        return Aplus.length === 2 && Aplus[0].length === 3;
      }
    },
    {
      name: 'Morpheme Composition',
      test: () => {
        const morphemes = Morpheme.createBuiltInMorphemes(3);
        const button = new Button('Test', 'TS', 'Action', ['re'], {}, morphemes);
        return button.M.length === 3 && button.b.length === 3;
      }
    },
    {
      name: 'Engine Creation',
      test: async () => {
        const engine = await WorldEngineV3Factory.create2D();
        return engine.dimensions === 2 && engine.lexicalEngine.buttons.size > 0;
      }
    }
  ];

  const results = [];
  for (const test of tests) {
    try {
      const result = await test.test();
      results.push({ name: test.name, success: result });
      console.log(`${result ? '✅' : '❌'} ${test.name}`);
    } catch (error) {
      results.push({ name: test.name, success: false, error: error.message });
      console.log(`❌ ${test.name}: ${error.message}`);
    }
  }

  const passed = results.filter(r => r.success).length;
  console.log(`\n📊 Self-Test Results: ${passed}/${results.length} passed`);

  return results;
}

// Export for use in other modules
export { runMathematicalDemo, quickSelfTest };

// Auto-run if this is the main module
if (typeof window !== 'undefined') {
  // Browser environment - add to window
  window.WorldEngineMathDemo = { runMathematicalDemo, quickSelfTest };
  console.log('🌐 World Engine Mathematical Demo loaded in browser');
} else if (import.meta.url === `file://${process.argv[1]}`) {
  // Node.js environment - run directly
  runMathematicalDemo().catch(console.error);
}
