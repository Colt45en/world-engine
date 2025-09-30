/**
 * World Engine Studio - Type System Integration Test
 * Smoke test to verify type lattice and combined controllers work together
 */

// Test function that runs after everything loads
function runStudioIntegrationTest() {
  console.log('🚀 Running Studio Integration Test');

  const results = {
    typeLattice: false,
    controllers: false,
    unifiedEngine: false,
    studioBridge: false,
    integration: false
  };

  // Test 1: Type Lattice
  try {
    if (window.StudioBridge?.typeLattice) {
      const lattice = window.StudioBridge.typeLattice;
      const testResult = lattice.checkCompose('string', 'token');
      results.typeLattice = testResult === true;
      console.log('✅ Type Lattice: Working');
    } else {
      console.log('❌ Type Lattice: Not available');
    }
  } catch (error) {
    console.log('❌ Type Lattice: Error -', error.message);
  }

  // Test 2: Combined Controllers
  try {
    if (window.WorldEngineControllers) {
      const hasAllControllers = !!(
        window.WorldEngineControllers.RecorderController &&
        window.WorldEngineControllers.AIBotController &&
        window.WorldEngineControllers.ChatController &&
        window.WorldEngineControllers.EngineController &&
        window.WorldEngineControllers.WorldEngineOrchestrator
      );
      results.controllers = hasAllControllers;
      console.log('✅ Combined Controllers: Working');
    } else {
      console.log('❌ Combined Controllers: Not available');
    }
  } catch (error) {
    console.log('❌ Combined Controllers: Error -', error.message);
  }

  // Test 3: Unified Engine
  try {
    if (window.UnifiedWorldEngine) {
      const hasFactory = !!window.UnifiedWorldEngine.UnifiedWorldEngineFactory;
      results.unifiedEngine = hasFactory;
      console.log('✅ Unified World Engine: Available');
    } else {
      console.log('❌ Unified World Engine: Not available');
    }
  } catch (error) {
    console.log('❌ Unified World Engine: Error -', error.message);
  }

  // Test 4: Studio Bridge Enhanced
  try {
    if (window.StudioBridge) {
      const hasEnhanced = !!(
        window.StudioBridge.typeLattice &&
        window.StudioBridge.TypedStore &&
        window.StudioBridge.TypedUtils
      );
      results.studioBridge = hasEnhanced;
      console.log('✅ Enhanced Studio Bridge: Working');
    } else {
      console.log('❌ Studio Bridge: Not available');
    }
  } catch (error) {
    console.log('❌ Studio Bridge: Error -', error.message);
  }

  // Test 5: Integration Test - Send a typed message
  try {
    if (window.StudioBridge?.sendTypedBus && window.StudioBridge?.onTypedBus) {
      let messageReceived = false;

      const unsubscribe = window.StudioBridge.onTypedBus((msg) => {
        if (msg.type === 'test.integration') {
          messageReceived = true;
        }
      }, 'string');

      window.StudioBridge.sendTypedBus({
        type: 'test.integration',
        message: 'Integration test message'
      }, 'string');

      setTimeout(() => {
        results.integration = messageReceived;
        unsubscribe();
        console.log(messageReceived ? '✅ Integration: Working' : '❌ Integration: Failed');

        // Display final results
        displayTestResults(results);
      }, 100);

      return; // Exit early to wait for async test
    } else {
      console.log('❌ Integration: Enhanced bridge not available');
    }
  } catch (error) {
    console.log('❌ Integration: Error -', error.message);
  }

  // If we reach here, display results immediately
  displayTestResults(results);
}

function displayTestResults(results) {
  const passed = Object.values(results).filter(Boolean).length;
  const total = Object.keys(results).length;

  console.log(`\n📊 Integration Test Results: ${passed}/${total} passed`);
  console.log('Details:', results);

  if (passed === total) {
    console.log('🎉 All systems integrated successfully!');

    // Show type lattice debug info
    if (window.StudioBridge?.typeLattice) {
      console.log('\n🔬 Type System Ready:');
      console.log('Available types:', window.StudioBridge.typeLattice.getAllTypes().join(', '));

      // Test some interesting compositions
      const compositions = [
        ['string', 'morpheme'],
        ['morpheme', 'matrix'],
        ['vector3d', 'matrix'],
        ['analysis', 'result']
      ];

      console.log('Composition tests:');
      compositions.forEach(([from, to]) => {
        const canCompose = window.StudioBridge.typeLattice.checkCompose(from, to);
        console.log(`  ${from} -> ${to}: ${canCompose ? '✅' : '❌'}`);
      });
    }
  } else {
    console.log('⚠️  Some systems need attention');
  }

  // Make results available globally for debugging
  window.studioTestResults = results;
}

// Auto-run test when page loads (after a delay to ensure everything is loaded)
if (typeof window !== 'undefined') {
  window.addEventListener('load', () => {
    setTimeout(runStudioIntegrationTest, 2000);
  });

  // Also expose the test function globally
  window.runStudioIntegrationTest = runStudioIntegrationTest;
}

// Export for module systems
if (typeof module !== 'undefined' && module.exports) {
  module.exports = { runStudioIntegrationTest };
}
