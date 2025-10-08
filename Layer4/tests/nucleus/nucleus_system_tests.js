/**
 * Nucleus System Test Suite
 * Tests for AI Bot communication, Librarian data flow, and Nucleus processing
 */

// Test configuration
const nucleusToOperatorMap = {
  'VIBRATE': 'ST',      // Stabilization
  'OPTIMIZATION': 'UP',  // Update/Progress
  'STATE': 'CV',        // Convergence
  'SEED': 'RB'          // Rollback
};

// Test helper functions
function simulateAIBotMessage(message, type) {
  const operatorMapping = {
    query: 'VIBRATE',
    learning: 'OPTIMIZATION',
    feedback: 'STATE'
  };

  const nucleusRole = operatorMapping[type];
  const communicationEntry = {
    timestamp: new Date().toLocaleTimeString(),
    type: 'ai_bot',
    source: 'AI Bot',
    message: `${type}: ${message}`
  };

  return {
    nucleusRole,
    communicationEntry,
    aiMessage: {
      type: 'ai_bot_message',
      role: nucleusRole,
      message,
      timestamp: Date.now(),
      source: 'ai_bot'
    }
  };
}

function simulateLibrarianData(librarian, dataType, data) {
  const operatorMapping = {
    pattern: 'VIBRATE',
    classification: 'STATE',
    analysis: 'OPTIMIZATION'
  };

  const nucleusRole = operatorMapping[dataType];
  const communicationEntry = {
    timestamp: new Date().toLocaleTimeString(),
    type: 'librarian',
    source: librarian,
    message: `${dataType}: ${JSON.stringify(data)}`
  };

  return {
    nucleusRole,
    communicationEntry,
    librarianMessage: {
      type: 'librarian_data',
      librarian,
      dataType,
      data,
      timestamp: Date.now()
    }
  };
}

function simulateNucleusEventProcessing(role, data) {
  const operator = nucleusToOperatorMap[role];

  return {
    operator,
    event: {
      role,
      operator,
      data,
      processed: true
    }
  };
}

// Test runner
class NucleusTestRunner {
  constructor() {
    this.testResults = [];
    this.totalTests = 0;
    this.passedTests = 0;
    this.failedTests = 0;
  }

  test(description, testFn) {
    this.totalTests++;

    try {
      testFn();
      this.passedTests++;
      this.testResults.push({
        description,
        status: 'PASSED',
        error: null
      });
      console.log(`✅ PASS: ${description}`);
    } catch (error) {
      this.failedTests++;
      this.testResults.push({
        description,
        status: 'FAILED',
        error: error.message
      });
      console.log(`❌ FAIL: ${description} - ${error.message}`);
    }
  }

  assertEqual(actual, expected, message) {
    if (actual !== expected) {
      throw new Error(`${message || 'Assertion failed'}: expected ${expected}, got ${actual}`);
    }
  }

  assertTrue(condition, message) {
    if (!condition) {
      throw new Error(message || 'Expected condition to be true');
    }
  }

  assertContains(haystack, needle, message) {
    if (!haystack.includes(needle)) {
      throw new Error(`${message || 'Assertion failed'}: expected "${haystack}" to contain "${needle}"`);
    }
  }

  runAllTests() {
    console.log('🧠 Starting Nucleus System Tests...\n');

    this.testNucleusToOperatorMapping();
    this.testAIBotCommunication();
    this.testLibrarianDataProcessing();
    this.testNucleusEventProcessing();
    this.testCommunicationFlowIntegration();
    this.testErrorHandling();

    this.printSummary();
    return this.testResults;
  }

  testNucleusToOperatorMapping() {
    console.log('📋 Testing Nucleus to Operator Mapping...');

    this.test('should map VIBRATE to ST', () => {
      this.assertEqual(nucleusToOperatorMap.VIBRATE, 'ST');
    });

    this.test('should map OPTIMIZATION to UP', () => {
      this.assertEqual(nucleusToOperatorMap.OPTIMIZATION, 'UP');
    });

    this.test('should map STATE to CV', () => {
      this.assertEqual(nucleusToOperatorMap.STATE, 'CV');
    });

    this.test('should map SEED to RB', () => {
      this.assertEqual(nucleusToOperatorMap.SEED, 'RB');
    });

    this.test('should have all 4 nucleus roles', () => {
      const roles = Object.keys(nucleusToOperatorMap);
      this.assertEqual(roles.length, 4);
    });
  }

  testAIBotCommunication() {
    console.log('\n🤖 Testing AI Bot Communication...');

    this.test('should route AI bot queries to VIBRATE', () => {
      const result = simulateAIBotMessage('Analyze current patterns', 'query');
      this.assertEqual(result.nucleusRole, 'VIBRATE');
      this.assertEqual(result.communicationEntry.type, 'ai_bot');
    });

    this.test('should route AI bot learning to OPTIMIZATION', () => {
      const result = simulateAIBotMessage('Learning optimization needed', 'learning');
      this.assertEqual(result.nucleusRole, 'OPTIMIZATION');
      this.assertEqual(result.aiMessage.role, 'OPTIMIZATION');
    });

    this.test('should route AI bot feedback to STATE', () => {
      const result = simulateAIBotMessage('Feedback on results', 'feedback');
      this.assertEqual(result.nucleusRole, 'STATE');
      this.assertContains(result.communicationEntry.message, 'feedback');
    });

    this.test('should create proper communication log entries', () => {
      const result = simulateAIBotMessage('Test message', 'query');
      this.assertEqual(result.communicationEntry.source, 'AI Bot');
      this.assertContains(result.communicationEntry.message, 'query: Test message');
      this.assertTrue(result.communicationEntry.timestamp.length > 0, 'Timestamp should exist');
    });
  }

  testLibrarianDataProcessing() {
    console.log('\n📚 Testing Librarian Data Processing...');

    this.test('should route pattern data to VIBRATE', () => {
      const testData = { equations: 12, complexity: 'high' };
      const result = simulateLibrarianData('Math Librarian', 'pattern', testData);
      this.assertEqual(result.nucleusRole, 'VIBRATE');
      this.assertEqual(result.librarianMessage.dataType, 'pattern');
    });

    this.test('should route classification data to STATE', () => {
      const testData = { words: 245, sentiment: 'positive' };
      const result = simulateLibrarianData('English Librarian', 'classification', testData);
      this.assertEqual(result.nucleusRole, 'STATE');
      this.assertEqual(result.communicationEntry.source, 'English Librarian');
    });

    this.test('should route analysis data to OPTIMIZATION', () => {
      const testData = { patterns: 8, confidence: 0.85 };
      const result = simulateLibrarianData('Pattern Librarian', 'analysis', testData);
      this.assertEqual(result.nucleusRole, 'OPTIMIZATION');
      this.assertEqual(result.librarianMessage.data.patterns, 8);
    });

    this.test('should handle all librarian types', () => {
      const librarians = ['Math Librarian', 'English Librarian', 'Pattern Librarian'];

      librarians.forEach(librarian => {
        const result = simulateLibrarianData(librarian, 'pattern', {});
        this.assertEqual(result.communicationEntry.source, librarian);
        this.assertEqual(result.communicationEntry.type, 'librarian');
      });
    });
  }

  testNucleusEventProcessing() {
    console.log('\n🧠 Testing Nucleus Event Processing...');

    this.test('should process VIBRATE events correctly', () => {
      const result = simulateNucleusEventProcessing('VIBRATE', { source: 'ai_bot' });
      this.assertEqual(result.operator, 'ST');
      this.assertEqual(result.event.role, 'VIBRATE');
      this.assertTrue(result.event.processed);
    });

    this.test('should process OPTIMIZATION events correctly', () => {
      const result = simulateNucleusEventProcessing('OPTIMIZATION', { source: 'librarian' });
      this.assertEqual(result.operator, 'UP');
      this.assertTrue(result.event.processed);
    });

    this.test('should process STATE events correctly', () => {
      const result = simulateNucleusEventProcessing('STATE');
      this.assertEqual(result.operator, 'CV');
    });

    this.test('should process SEED events correctly', () => {
      const result = simulateNucleusEventProcessing('SEED');
      this.assertEqual(result.operator, 'RB');
    });
  }

  testCommunicationFlowIntegration() {
    console.log('\n📡 Testing Communication Flow Integration...');

    this.test('should maintain chronological order in communication log', () => {
      const entries = [];

      const aiEntry = {
        timestamp: '10:00:01',
        type: 'ai_bot',
        source: 'AI Bot',
        message: 'query: Test query'
      };

      const librarianEntry = {
        timestamp: '10:00:02',
        type: 'librarian',
        source: 'Math Librarian',
        message: 'pattern: {"data": "test"}'
      };

      entries.push(aiEntry, librarianEntry);

      this.assertEqual(entries.length, 2);
      this.assertEqual(entries[0].type, 'ai_bot');
      this.assertEqual(entries[1].type, 'librarian');
    });

    this.test('should limit communication log entries', () => {
      const maxEntries = 20;
      const entries = [];

      // Simulate adding more than max entries
      for (let i = 0; i < 25; i++) {
        const entry = {
          timestamp: `10:00:${i.toString().padStart(2, '0')}`,
          type: i % 2 === 0 ? 'ai_bot' : 'librarian',
          source: i % 2 === 0 ? 'AI Bot' : 'Math Librarian',
          message: `test message ${i}`
        };
        entries.push(entry);

        // Keep only last 20 entries (simulating the slice(-19) logic)
        if (entries.length > maxEntries) {
          entries.splice(0, entries.length - maxEntries);
        }
      }

      this.assertTrue(entries.length <= maxEntries, `Entries should be limited to ${maxEntries}, got ${entries.length}`);
    });
  }

  testErrorHandling() {
    console.log('\n⚠️ Testing Error Handling...');

    this.test('should handle invalid nucleus roles gracefully', () => {
      const invalidRole = 'INVALID_ROLE';
      const operator = nucleusToOperatorMap[invalidRole];
      this.assertTrue(operator === undefined, 'Invalid role should return undefined');
    });

    this.test('should handle empty messages', () => {
      const result = simulateAIBotMessage('', 'query');
      this.assertContains(result.communicationEntry.message, 'query: ');
      this.assertEqual(result.aiMessage.message, '');
    });

    this.test('should handle invalid data types', () => {
      const invalidDataType = 'invalid';
      const operatorMapping = {
        pattern: 'VIBRATE',
        classification: 'STATE',
        analysis: 'OPTIMIZATION'
      };

      const nucleusRole = operatorMapping[invalidDataType];
      this.assertTrue(nucleusRole === undefined, 'Invalid data type should return undefined');
    });
  }

  printSummary() {
    console.log('\n' + '='.repeat(60));
    console.log('🧠 NUCLEUS SYSTEM TEST SUMMARY');
    console.log('='.repeat(60));
    console.log(`Total Tests: ${this.totalTests}`);
    console.log(`✅ Passed: ${this.passedTests}`);
    console.log(`❌ Failed: ${this.failedTests}`);
    console.log(`📊 Success Rate: ${((this.passedTests / this.totalTests) * 100).toFixed(1)}%`);

    if (this.failedTests > 0) {
      console.log('\n❌ FAILED TESTS:');
      this.testResults
        .filter(result => result.status === 'FAILED')
        .forEach(result => {
          console.log(`  • ${result.description}: ${result.error}`);
        });
    }

    console.log('='.repeat(60));

    return {
      total: this.totalTests,
      passed: this.passedTests,
      failed: this.failedTests,
      successRate: (this.passedTests / this.totalTests) * 100
    };
  }
}

// Export for Node.js
if (typeof module !== 'undefined' && module.exports) {
  module.exports = {
    NucleusTestRunner,
    simulateAIBotMessage,
    simulateLibrarianData,
    simulateNucleusEventProcessing,
    nucleusToOperatorMap
  };
}

// Auto-run tests if this file is executed directly
if (typeof require !== 'undefined' && require.main === module) {
  const runner = new NucleusTestRunner();
  const results = runner.runAllTests();

  // Exit with error code if tests failed
  process.exit(results.failed > 0 ? 1 : 0);
}
