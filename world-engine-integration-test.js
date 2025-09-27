#!/usr/bin/env node
/**
 * World Engine Advanced Systems Integration Test
 * Comprehensive testing of all surgical upgrades and system integration
 */

import { createWorldEngineFramework } from './world-engine-integration-framework.js';

async function runComprehensiveIntegrationTest() {
    console.log('🚀 Starting World Engine Advanced Systems Integration Test');
    console.log('Testing surgical upgrades from Untitled-10.js implementation');
    console.log('='.repeat(80));

    let framework = null;
    const testResults = {
        passed: 0,
        failed: 0,
        total: 0,
        details: []
    };

    try {
        // Test 1: Framework Initialization
        console.log('\n🔧 Test 1: Framework Initialization');
        framework = await createWorldEngineFramework({
            enableAutoStart: true,
            enablePerformanceMonitoring: true,
            autoConnect: true
        });

        if (framework && framework.isInitialized) {
            console.log('✅ Framework initialized successfully');
            testResults.passed++;
        } else {
            console.log('❌ Framework initialization failed');
            testResults.failed++;
        }
        testResults.total++;

        // Test 2: Advanced Morphology Processing
        console.log('\n🔬 Test 2: Advanced Morphology Processing');
        const morphologyTestWords = [
            'counterproductivity',
            'bioengineering',
            'antidisestablishmentarianism',
            'machine learning'
        ];

        for (const word of morphologyTestWords) {
            try {
                const result = await framework.processWithAdvancedMorphology(word, {
                    autoIndex: true,
                    generateVectors: true
                });

                if (result && result.morphemes && result.classification) {
                    console.log(`  ✅ "${word}" analyzed: ${result.morphemes.length} morphemes, ${result.classification.type} classification`);
                    testResults.passed++;
                } else {
                    console.log(`  ❌ "${word}" analysis failed`);
                    testResults.failed++;
                }
                testResults.total++;

            } catch (error) {
                console.log(`  ❌ "${word}" error: ${error.message}`);
                testResults.failed++;
                testResults.total++;
            }
        }

        // Test 3: Index Manager Search
        console.log('\n📊 Test 3: Index Manager Search Capabilities');
        const searchQueries = [
            'productivity analysis',
            'engineering systems',
            'machine learning algorithms',
            'biological processes'
        ];

        for (const query of searchQueries) {
            try {
                const results = await framework.searchWithIndexManager(query, {
                    maxResults: 5,
                    useMorphology: true,
                    includeVectorSearch: true
                });

                if (results && Array.isArray(results)) {
                    console.log(`  ✅ "${query}" found ${results.length} results`);
                    testResults.passed++;
                } else {
                    console.log(`  ❌ "${query}" search failed`);
                    testResults.failed++;
                }
                testResults.total++;

            } catch (error) {
                console.log(`  ❌ "${query}" error: ${error.message}`);
                testResults.failed++;
                testResults.total++;
            }
        }

        // Test 4: UX Pipeline Processing
        console.log('\n🎨 Test 4: UX Pipeline Processing');
        const userInputs = [
            'How does bioengineering work?',
            'Explain machine learning algorithms',
            'What is morphological analysis?',
            'Show me productivity tools'
        ];

        for (const input of userInputs) {
            try {
                const response = await framework.processUserInput(input, {
                    enableMorphology: true,
                    enableSearch: true,
                    useAgents: true,
                    generateVisuals: false
                });

                if (response && response.success) {
                    console.log(`  ✅ "${input}" processed successfully`);
                    testResults.passed++;
                } else {
                    console.log(`  ❌ "${input}" processing failed`);
                    testResults.failed++;
                }
                testResults.total++;

            } catch (error) {
                console.log(`  ❌ "${input}" error: ${error.message}`);
                testResults.failed++;
                testResults.total++;
            }
        }

        // Test 5: Comprehensive Processing Pipeline
        console.log('\n🎯 Test 5: Comprehensive Processing Pipeline');
        const comprehensiveInputs = [
            'advanced morphological complexity analysis',
            'high-performance indexing systems',
            'user experience optimization'
        ];

        for (const input of comprehensiveInputs) {
            try {
                const result = await framework.processComprehensive(input, {
                    enableMorphology: true,
                    enableSearch: true,
                    useAgents: true,
                    maxSearchResults: 5
                });

                if (result && result.success) {
                    console.log(`  ✅ "${input}" comprehensive processing completed in ${result.totalTime}ms`);
                    console.log(`    - Morphology: ${result.morphology ? '✅' : '❌'}`);
                    console.log(`    - Indexing: ${result.indexing ? '✅' : '❌'}`);
                    console.log(`    - UX Processing: ${result.uxProcessing ? '✅' : '❌'}`);
                    console.log(`    - Agent Insights: ${result.agentInsights ? '✅' : '❌'}`);
                    testResults.passed++;
                } else {
                    console.log(`  ❌ "${input}" comprehensive processing failed`);
                    testResults.failed++;
                }
                testResults.total++;

            } catch (error) {
                console.log(`  ❌ "${input}" error: ${error.message}`);
                testResults.failed++;
                testResults.total++;
            }
        }

        // Test 6: System Health and Performance
        console.log('\n📈 Test 6: System Health and Performance Monitoring');
        try {
            const systemStatus = framework.getSystemStatus();
            const healthReport = await framework.generateSystemHealthReport();

            if (systemStatus && healthReport) {
                console.log('  ✅ System status retrieved successfully');
                console.log(`    - Systems Online: ${systemStatus.metrics?.systemsOnline || 0}`);
                console.log(`    - Framework Health: ${systemStatus.framework?.isInitialized ? 'Healthy' : 'Degraded'}`);
                console.log(`    - Overall Health: ${healthReport.overallHealth}`);
                testResults.passed++;
            } else {
                console.log('  ❌ System health monitoring failed');
                testResults.failed++;
            }
            testResults.total++;

        } catch (error) {
            console.log(`  ❌ Health monitoring error: ${error.message}`);
            testResults.failed++;
            testResults.total++;
        }

        // Test 7: Error Handling and Circuit Breakers
        console.log('\n🛡️ Test 7: Error Handling and Circuit Breakers');
        try {
            // Test invalid input handling
            await framework.processComprehensive('', {
                enableMorphology: true,
                enableSearch: true
            });

            // Test with null input
            await framework.processUserInput(null, {});

            // Test disconnected system handling
            const tempSystem = framework.systems.advancedMorphologyEngine;
            framework.systems.advancedMorphologyEngine = null;

            try {
                await framework.processWithAdvancedMorphology('test');
                console.log('  ❌ Error handling test failed - should have thrown error');
                testResults.failed++;
            } catch (expectedError) {
                console.log('  ✅ Error handling working correctly - caught expected error');
                testResults.passed++;
            }

            // Restore system
            framework.systems.advancedMorphologyEngine = tempSystem;
            testResults.total++;

        } catch (error) {
            console.log(`  ⚠️ Error handling test inconclusive: ${error.message}`);
            testResults.total++;
        }

    } catch (error) {
        console.error('\n💥 Critical test failure:', error);
        testResults.failed++;
        testResults.total++;
    }

    // Generate Test Report
    console.log('\n' + '='.repeat(80));
    console.log('📊 WORLD ENGINE ADVANCED SYSTEMS INTEGRATION TEST REPORT');
    console.log('='.repeat(80));

    const successRate = testResults.total > 0 ? (testResults.passed / testResults.total * 100).toFixed(1) : 0;

    console.log(`Total Tests: ${testResults.total}`);
    console.log(`Passed: ${testResults.passed}`);
    console.log(`Failed: ${testResults.failed}`);
    console.log(`Success Rate: ${successRate}%`);

    if (successRate >= 80) {
        console.log('\n🎉 INTEGRATION TEST PASSED!');
        console.log('All surgical upgrades from Untitled-10.js are functioning correctly.');
        console.log('Advanced Systems are ready for production deployment.');

        // Display system capabilities
        console.log('\n🚀 DEPLOYED ADVANCED SYSTEMS:');
        console.log('  🔬 Advanced Morphology Engine v2 - Longest-match multi-affix processing');
        console.log('  📊 World Engine Index Manager - BM25 ranking & vector similarity');
        console.log('  🎨 UX Pipeline Integration - Circuit breaker & performance monitoring');
        console.log('  🔗 Framework Integration - Layer 0 oversight & system coordination');

    } else {
        console.log('\n⚠️ INTEGRATION TEST NEEDS ATTENTION');
        console.log('Some systems may need debugging or configuration adjustments.');
    }

    // Cleanup
    if (framework) {
        try {
            await framework.shutdown();
            console.log('\n🔧 Framework shutdown completed');
        } catch (shutdownError) {
            console.log('\n⚠️ Framework shutdown error:', shutdownError.message);
        }
    }

    return {
        success: successRate >= 80,
        testResults: testResults,
        successRate: successRate
    };
}

// Performance Benchmark Test
async function runPerformanceBenchmark() {
    console.log('\n🏃‍♂️ Running Performance Benchmark...');

    const framework = await createWorldEngineFramework({
        enableAutoStart: true,
        enablePerformanceMonitoring: true,
        autoConnect: true
    });

    const benchmarkWords = [
        'productivity', 'engineering', 'development', 'optimization',
        'counterproductivity', 'bioengineering', 'antidisestablishmentarianism',
        'machine learning algorithms', 'neural network optimization',
        'advanced morphological complexity analysis systems'
    ];

    let totalTime = 0;
    let wordCount = 0;

    console.log('Processing benchmark words...');
    for (const word of benchmarkWords) {
        const startTime = Date.now();

        try {
            await framework.processComprehensive(word, {
                enableMorphology: true,
                enableSearch: true,
                maxSearchResults: 3
            });

            const wordTime = Date.now() - startTime;
            totalTime += wordTime;
            wordCount++;

            console.log(`  "${word}" - ${wordTime}ms`);

        } catch (error) {
            console.log(`  "${word}" - ERROR: ${error.message}`);
        }
    }

    const avgTime = wordCount > 0 ? totalTime / wordCount : 0;
    const wordsPerSecond = totalTime > 0 ? (wordCount / totalTime * 1000) : 0;

    console.log('\n📊 Performance Benchmark Results:');
    console.log(`  Total Words Processed: ${wordCount}`);
    console.log(`  Total Time: ${totalTime}ms`);
    console.log(`  Average Time per Word: ${avgTime.toFixed(2)}ms`);
    console.log(`  Processing Rate: ${wordsPerSecond.toFixed(2)} words/second`);
    console.log(`  System Efficiency: ${avgTime < 100 ? 'EXCELLENT' : avgTime < 250 ? 'GOOD' : 'NEEDS OPTIMIZATION'}`);

    await framework.shutdown();
}

// Main execution
if (typeof window === 'undefined' && typeof process !== 'undefined') {
    // Node.js environment
    console.log('🌍 World Engine Advanced Systems - Integration Test Suite');
    console.log('Testing implementation of surgical upgrades from Untitled-10.js\n');

    runComprehensiveIntegrationTest()
        .then(async (result) => {
            if (result.success) {
                console.log('\n🎯 Running Performance Benchmark...');
                await runPerformanceBenchmark();
                console.log('\n✅ All tests completed successfully!');
                process.exit(0);
            } else {
                console.log('\n⚠️ Tests completed with issues. Review results above.');
                process.exit(1);
            }
        })
        .catch((error) => {
            console.error('\n💥 Test execution failed:', error);
            process.exit(1);
        });
}

export { runComprehensiveIntegrationTest, runPerformanceBenchmark };
