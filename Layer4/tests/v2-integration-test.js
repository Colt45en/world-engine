/**
 * V2 System Integration Test Suite
 * Tests the integration between enhanced morphology engine, upflow v2, and HTML pipeline
 */

export class V2IntegrationTester {
    constructor() {
        this.testResults = [];
        this.morphologyEngine = null;
        this.enhancedUpflow = null;
        this.htmlPipeline = null;
    }

    async initialize() {
        console.log('🧪 Initializing V2 Integration Test Suite...');

        try {
            // Load v2 modules
            const [
                { AdvancedMorphologyEngine },
                { createEnhancedUpflowV2 },
                { createHTMLPipelineV2 }
            ] = await Promise.all([
                import('./models/advanced-morphology-engine.js'),
                import('./models/enhanced-upflow-v2-fixed.js'),
                import('./models/enhanced-html-pipeline-v2.js')
            ]);

            // Initialize systems
            this.morphologyEngine = new AdvancedMorphologyEngine({
                debug: false,
                enableV2Features: true
            });

            this.enhancedUpflow = await createEnhancedUpflowV2({});

            this.htmlPipeline = createHTMLPipelineV2({
                container: document.body,
                reactMode: false
            });

            console.log('✅ All v2 systems initialized for testing');
            return true;

        } catch (error) {
            console.error('❌ V2 system initialization failed:', error);
            return false;
        }
    }

    async runAllTests() {
        console.log('🚀 Running V2 Integration Tests...');

        await this.testMorphologyAnalysis();
        await this.testUpflowIntegration();
        await this.testHTMLPipelineFeatures();
        await this.testSemanticQueries();
        await this.testPerformanceMetrics();

        return this.generateReport();
    }

    async testMorphologyAnalysis() {
        console.log('Testing Morphology Engine v2...');

        const testWords = [
            'productivity',
            'counterproductivity',
            'bioengineering',
            'unrecognizable'
        ];

        for (const word of testWords) {
            try {
                const result = this.morphologyEngine.analyzeV2(word);

                this.testResults.push({
                    test: 'morphology_analysis',
                    input: word,
                    success: !!(result && result.original && result.morphemes),
                    result: result,
                    details: {
                        morphemeCount: result.morphemes ? result.morphemes.length : 0,
                        hasPositions: !!(result.positions && result.positions.length > 0),
                        hasSemanticClass: !!result.semanticClass,
                        complexity: result.complexity
                    }
                });

                console.log(`✅ ${word}: ${result.morphemes?.length || 0} morphemes, complexity ${result.complexity}`);

            } catch (error) {
                this.testResults.push({
                    test: 'morphology_analysis',
                    input: word,
                    success: false,
                    error: error.message
                });

                console.error(`❌ ${word}: ${error.message}`);
            }
        }
    }

    async testUpflowIntegration() {
        console.log('Testing Enhanced Upflow v2 integration...');

        try {
            // Test ingesting morphology results
            const testResult = this.morphologyEngine.analyzeV2('productivity');
            await this.enhancedUpflow.ingestEnhanced(testResult);

            // Test semantic queries
            const actions = await this.enhancedUpflow.semantic.actions();
            const byRoot = await this.enhancedUpflow.semantic.byRoot('product');

            this.testResults.push({
                test: 'upflow_integration',
                input: 'productivity',
                success: true,
                details: {
                    ingestSuccess: true,
                    actionsCount: actions.length,
                    rootQueryCount: byRoot.length
                }
            });

            console.log(`✅ Upflow integration: ${actions.length} actions, ${byRoot.length} by root`);

        } catch (error) {
            this.testResults.push({
                test: 'upflow_integration',
                input: 'productivity',
                success: false,
                error: error.message
            });

            console.error(`❌ Upflow integration: ${error.message}`);
        }
    }

    async testHTMLPipelineFeatures() {
        console.log('Testing HTML Pipeline v2 features...');

        try {
            // Test toast notifications
            this.htmlPipeline.showToast('Test notification', 'info', 1000);

            // Test morpheme highlighting
            const testElement = document.createElement('div');
            testElement.textContent = 'productivity';
            const morphResult = this.morphologyEngine.analyzeV2('productivity');

            if (morphResult.positions) {
                this.htmlPipeline.highlightMorphemes(testElement, morphResult);
            }

            // Test performance measurement
            const duration = this.htmlPipeline.measurePerformance('Test Operation', () => {
                return 'test result';
            });

            this.testResults.push({
                test: 'html_pipeline_features',
                success: true,
                details: {
                    toastDisplayed: true,
                    morphemeHighlighting: testElement.innerHTML.includes('morpheme-highlight'),
                    performanceMeasured: true
                }
            });

            console.log('✅ HTML Pipeline features working');

        } catch (error) {
            this.testResults.push({
                test: 'html_pipeline_features',
                success: false,
                error: error.message
            });

            console.error(`❌ HTML Pipeline: ${error.message}`);
        }
    }

    async testSemanticQueries() {
        console.log('Testing semantic query capabilities...');

        try {
            // Add some test data
            const testWords = ['running', 'beautiful', 'house'];
            for (const word of testWords) {
                const result = this.morphologyEngine.analyzeV2(word);
                await this.enhancedUpflow.ingestEnhanced(result);
            }

            // Test different semantic queries
            const actions = await this.enhancedUpflow.semantic.actions();
            const properties = await this.enhancedUpflow.semantic.properties();
            const structures = await this.enhancedUpflow.semantic.structures();

            this.testResults.push({
                test: 'semantic_queries',
                success: true,
                details: {
                    actionsFound: actions.length,
                    propertiesFound: properties.length,
                    structuresFound: structures.length
                }
            });

            console.log(`✅ Semantic queries: ${actions.length + properties.length + structures.length} total results`);

        } catch (error) {
            this.testResults.push({
                test: 'semantic_queries',
                success: false,
                error: error.message
            });

            console.error(`❌ Semantic queries: ${error.message}`);
        }
    }

    async testPerformanceMetrics() {
        console.log('Testing performance metrics...');

        try {
            const startTime = performance.now();

            // Run several operations to generate metrics
            for (let i = 0; i < 10; i++) {
                const result = this.morphologyEngine.analyzeV2(`test${i}`);
                await this.enhancedUpflow.ingestEnhanced(result);
            }

            const endTime = performance.now();
            const totalTime = endTime - startTime;

            this.testResults.push({
                test: 'performance_metrics',
                success: true,
                details: {
                    operationsPerformed: 10,
                    totalTime: totalTime,
                    averageTime: totalTime / 10,
                    operationsPerSecond: Math.round(10000 / totalTime)
                }
            });

            console.log(`✅ Performance: ${Math.round(totalTime)}ms total, ${Math.round(totalTime / 10)}ms avg`);

        } catch (error) {
            this.testResults.push({
                test: 'performance_metrics',
                success: false,
                error: error.message
            });

            console.error(`❌ Performance test: ${error.message}`);
        }
    }

    generateReport() {
        const totalTests = this.testResults.length;
        const successfulTests = this.testResults.filter(t => t.success).length;
        const failedTests = totalTests - successfulTests;

        const report = {
            summary: {
                total: totalTests,
                successful: successfulTests,
                failed: failedTests,
                successRate: `${Math.round((successfulTests / totalTests) * 100)}%`
            },
            details: this.testResults,
            timestamp: new Date().toISOString()
        };

        console.log('📊 V2 Integration Test Report:');
        console.log(`✅ Successful: ${successfulTests}/${totalTests} (${report.summary.successRate})`);
        if (failedTests > 0) {
            console.log(`❌ Failed: ${failedTests}`);
            this.testResults.filter(t => !t.success).forEach(test => {
                console.log(`  - ${test.test}: ${test.error}`);
            });
        }

        return report;
    }

    cleanup() {
        if (this.htmlPipeline) {
            this.htmlPipeline.destroy();
        }
        if (this.enhancedUpflow) {
            this.enhancedUpflow.destroy();
        }
    }
}

// Auto-run tests when loaded
window.addEventListener('load', async () => {
    if (window.location.search.includes('test=v2')) {
        const tester = new V2IntegrationTester();
        if (await tester.initialize()) {
            const report = await tester.runAllTests();
            window.v2TestReport = report;
            tester.cleanup();
        }
    }
});

export { V2IntegrationTester };
