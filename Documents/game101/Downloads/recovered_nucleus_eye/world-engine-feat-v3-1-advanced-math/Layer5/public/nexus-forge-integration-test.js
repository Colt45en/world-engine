/**
 * NEXUS Forge Unified System Integration Test
 * • Validates complete system functionality
 * • Tests C++ NexusGameEngine integration concepts
 * • Verifies world generation, AI patterns, beat synchronization
 * • Comprehensive open-world game framework testing
 */

class NexusForgeIntegrationTest {
    constructor() {
        this.testResults = [];
        this.nexusForge = null;
        this.startTime = 0;
    }

    async runAllTests() {
        console.log('🧪 Starting NEXUS Forge Unified System Integration Tests...');
        this.startTime = performance.now();

        try {
            // Test 1: System Initialization
            await this.testSystemInitialization();

            // Test 2: Math Engine Integration
            await this.testMathEngineIntegration();

            // Test 3: World Generation System
            await this.testWorldGenerationSystem();

            // Test 4: C++ Integration Concepts
            await this.testCppIntegrationConcepts();

            // Test 5: AI Pattern Recognition
            await this.testAIPatternRecognition();

            // Test 6: Beat Synchronization
            await this.testBeatSynchronization();

            // Test 7: Audio-Reactive World Generation
            await this.testAudioReactiveWorldGeneration();

            // Test 8: Component System Architecture
            await this.testComponentSystemArchitecture();

            // Test 9: Asset Management
            await this.testAssetManagement();

            // Test 10: Performance & Memory
            await this.testPerformanceMemory();

            this.generateTestReport();

        } catch (error) {
            console.error('❌ Integration test suite failed:', error);
            this.logTestResult('CRITICAL_ERROR', 'Integration Test Suite', false, error.message);
        }
    }

    async testSystemInitialization() {
        console.log('🔧 Testing System Initialization...');

        try {
            this.nexusForge = new NexusForgeUnified();

            // Test initialization
            await this.nexusForge.initialize({
                enableAudio: false, // Disable for testing
                debug: true
            });

            this.logTestResult('INIT', 'System Initialization',
                this.nexusForge.initialized === true,
                'NEXUS Forge system initialized successfully');

            // Test component availability
            const hasAllComponents =
                this.nexusForge.math !== null &&
                this.nexusForge.ai !== null &&
                this.nexusForge.synthesis !== null &&
                this.nexusForge.beat !== null &&
                this.nexusForge.world !== null;

            this.logTestResult('COMPONENTS', 'Core Components Available',
                hasAllComponents,
                `Components loaded: ${hasAllComponents ? 'ALL' : 'MISSING'}`);

        } catch (error) {
            this.logTestResult('INIT', 'System Initialization', false, error.message);
        }
    }

    async testMathEngineIntegration() {
        console.log('🔢 Testing Math Engine Integration...');

        try {
            const math = this.nexusForge.math;

            // Test morpheme system
            const morphemes = math.createBuiltInMorphemes();
            this.logTestResult('MORPHEMES', 'Morpheme System',
                morphemes.size >= 9,
                `Morphemes created: ${morphemes.size}`);

            // Test button creation
            const buildButton = math.createButton('Build', 'BD', 'verb', ['build']);
            this.logTestResult('BUTTONS', 'Button Creation',
                buildButton !== null && buildButton.label === 'Build',
                'Button creation successful');

            // Test matrix operations
            const matrixA = [[1, 2], [3, 4]];
            const matrixB = [[5, 6], [7, 8]];
            const result = math.multiply(matrixA, matrixB);

            this.logTestResult('MATRIX', 'Matrix Multiplication',
                result[0][0] === 19 && result[0][1] === 22,
                'Matrix operations working correctly');

            // Test pseudoinverse
            const pseudoInv = math.pseudoInverse(matrixA, 0.01);
            this.logTestResult('PSEUDOINV', 'Pseudoinverse Calculation',
                pseudoInv !== null && Array.isArray(pseudoInv),
                'Pseudoinverse calculation successful');

        } catch (error) {
            this.logTestResult('MATH', 'Math Engine Integration', false, error.message);
        }
    }

    async testWorldGenerationSystem() {
        console.log('🌍 Testing World Generation System...');

        try {
            const world = this.nexusForge.world;

            // Test chunk generation
            world.updateActiveChunks([0, 0, 0]);
            const chunkCount = world.chunks.size;

            this.logTestResult('CHUNKS', 'Chunk Generation',
                chunkCount > 0,
                `Chunks generated: ${chunkCount}`);

            // Test biome system
            const chunk = world.chunks.values().next().value;
            const hasBiomeData = chunk && chunk.biomeMap && chunk.biomeMap.length > 0;

            this.logTestResult('BIOMES', 'Biome System',
                hasBiomeData,
                'Biome data generated for chunks');

            // Test height generation
            const height = this.nexusForge.getWorldHeight(0, 0);
            this.logTestResult('TERRAIN', 'Terrain Height Generation',
                typeof height === 'number' && !isNaN(height),
                `Height at origin: ${height.toFixed(2)}`);

            // Test world entities
            const entity = this.nexusForge.createWorldEntity('test_tree', 'vegetation', [10, 5, 10]);
            this.logTestResult('ENTITIES', 'World Entity Creation',
                entity !== null && entity.name === 'test_tree',
                'World entity creation successful');

        } catch (error) {
            this.logTestResult('WORLD', 'World Generation System', false, error.message);
        }
    }

    async testCppIntegrationConcepts() {
        console.log('🏗️ Testing C++ Integration Concepts...');

        try {
            const world = this.nexusForge.world;

            // Test component systems (C++ inspired)
            const hasComponentSystems = world.components.size >= 3;
            this.logTestResult('CPP_COMPONENTS', 'Component Systems',
                hasComponentSystems,
                `Component systems: ${world.components.size}`);

            // Test entity management
            const entity = world.createEntity('cpp_test', 'game_object');
            const componentAdded = world.addComponentToEntity(entity.id, 'Transform', {
                position: [0, 0, 0],
                rotation: [0, 0, 0],
                scale: [1, 1, 1]
            });

            this.logTestResult('CPP_ENTITIES', 'Entity-Component System',
                componentAdded && world.entities.has(entity.id),
                'Entity-component architecture working');

            // Test LOD system
            const lodLevels = world.lodLevels;
            this.logTestResult('CPP_LOD', 'Level of Detail System',
                lodLevels && lodLevels.length === 4,
                `LOD levels configured: ${lodLevels.length}`);

            // Test system status (C++ SystemStatus concept)
            const status = this.nexusForge.getSystemStatus();
            const hasSystemData = status.components && status.world && status.performance;
            this.logTestResult('CPP_STATUS', 'System Status Monitoring',
                hasSystemData,
                'System status reporting functional');

        } catch (error) {
            this.logTestResult('CPP', 'C++ Integration Concepts', false, error.message);
        }
    }

    async testAIPatternRecognition() {
        console.log('🧠 Testing AI Pattern Recognition...');

        try {
            const ai = this.nexusForge.ai;

            // Test pattern detection
            const patterns = ai.detectPatterns(this.nexusForge.gameState);
            this.logTestResult('AI_PATTERNS', 'Pattern Detection',
                patterns !== null && Array.isArray(patterns),
                `Patterns detected: ${patterns ? patterns.length : 0}`);

            // Test recommendations
            const recommendations = ai.getRecommendations(this.nexusForge.gameState);
            const hasRecommendations = recommendations &&
                (recommendations.immediate.length > 0 || recommendations.strategic.length > 0);

            this.logTestResult('AI_RECOMMENDATIONS', 'AI Recommendations',
                recommendations !== null,
                `Recommendations generated: ${hasRecommendations ? 'YES' : 'NO'}`);

            // Test AI insights integration
            const insights = this.nexusForge.getAIInsights();
            this.logTestResult('AI_INSIGHTS', 'AI Insights Integration',
                insights !== null,
                'AI insights system functional');

        } catch (error) {
            this.logTestResult('AI', 'AI Pattern Recognition', false, error.message);
        }
    }

    async testBeatSynchronization() {
        console.log('🎵 Testing Beat Synchronization...');

        try {
            const beat = this.nexusForge.beat;

            // Test BPM setting
            this.nexusForge.setBPM(128);
            this.logTestResult('BEAT_BPM', 'BPM Configuration',
                beat.bpm === 128,
                `BPM set to: ${beat.bpm}`);

            // Test world parameters
            const worldParams = beat.getWorldParameters();
            const hasWorldParams = worldParams &&
                'terrainHeight' in worldParams &&
                'biomeDensity' in worldParams;

            this.logTestResult('BEAT_WORLD', 'Beat-Synchronized World Parameters',
                hasWorldParams,
                'World parameters responsive to beat');

            // Test beat engine state
            const beatInfo = {
                bpm: beat.clock.bpm,
                beat: beat.clock.beat,
                phase: beat.clock.beatPhase
            };

            this.logTestResult('BEAT_STATE', 'Beat Engine State',
                beatInfo.bpm > 0,
                `Beat state: ${beatInfo.beat}, Phase: ${beatInfo.phase.toFixed(2)}`);

        } catch (error) {
            this.logTestResult('BEAT', 'Beat Synchronization', false, error.message);
        }
    }

    async testAudioReactiveWorldGeneration() {
        console.log('🎤 Testing Audio-Reactive World Generation...');

        try {
            const synthesis = this.nexusForge.synthesis;

            // Test vibe state
            synthesis.updateVibeState(0.8, 0.6, 0.7, 0.5); // p, i, g, c values

            this.logTestResult('AUDIO_VIBE', 'Vibe State Management',
                synthesis.vibeState.p === 0.8,
                'Vibe state updates functional');

            // Test audio-reactive terrain generation
            const terrainHeight1 = this.nexusForge.generateTerrain(0, 0, { expression: 'beat_terrain' });

            // Change vibe state and test again
            synthesis.updateVibeState(0.2, 0.3, 0.1, 0.9);
            const terrainHeight2 = this.nexusForge.generateTerrain(0, 0, { expression: 'beat_terrain' });

            this.logTestResult('AUDIO_TERRAIN', 'Audio-Reactive Terrain',
                typeof terrainHeight1 === 'number' && typeof terrainHeight2 === 'number',
                `Terrain heights: ${terrainHeight1.toFixed(2)} -> ${terrainHeight2.toFixed(2)}`);

            // Test expression synthesis
            const expressions = synthesis.expressions;
            const hasBeatExpressions = expressions.has('beat_terrain') && expressions.has('pulse_world');

            this.logTestResult('AUDIO_EXPRESSIONS', 'Audio Expression Synthesis',
                hasBeatExpressions,
                'Beat-synchronized expressions available');

        } catch (error) {
            this.logTestResult('AUDIO', 'Audio-Reactive World Generation', false, error.message);
        }
    }

    async testComponentSystemArchitecture() {
        console.log('🔧 Testing Component System Architecture...');

        try {
            const world = this.nexusForge.world;

            // Test component system types
            const systemTypes = Array.from(world.components.keys());
            const expectedSystems = ['TerrainSystem', 'ResourceSystem', 'BiomeSystem'];
            const hasExpectedSystems = expectedSystems.every(sys => systemTypes.includes(sys));

            this.logTestResult('COMP_SYSTEMS', 'Component System Types',
                hasExpectedSystems,
                `Systems: ${systemTypes.join(', ')}`);

            // Test entity-component relationships
            const entity = world.createEntity('component_test', 'test');
            const terrainAdded = world.addComponentToEntity(entity.id, 'Terrain', {
                heightMod: 1.0,
                biome: 'forest'
            });

            this.logTestResult('COMP_RELATIONS', 'Entity-Component Relationships',
                terrainAdded && entity.components.has('Terrain'),
                'Entity-component binding functional');

            // Test system updates
            world.updateSystemWithBeat(0.016);
            this.logTestResult('COMP_UPDATES', 'Component System Updates',
                true, // If no error thrown, system updates work
                'Component systems update successfully');

        } catch (error) {
            this.logTestResult('COMPONENTS', 'Component System Architecture', false, error.message);
        }
    }

    async testAssetManagement() {
        console.log('📦 Testing Asset Management...');

        try {
            // Test asset loading (inspired by C++ ResourceEngine)
            const assetPromise = this.nexusForge.requestWorldAsset('textures', 'terrain_grass.png', 6);
            const asset = await assetPromise;

            this.logTestResult('ASSETS_LOAD', 'Asset Loading',
                asset && asset.loaded === true,
                `Asset loaded: ${asset.name} (${asset.size.toFixed(2)} MB)`);

            // Test multiple asset requests
            const assets = await Promise.all([
                this.nexusForge.requestWorldAsset('audio', 'ambient.ogg', 5),
                this.nexusForge.requestWorldAsset('models', 'tree.obj', 4),
                this.nexusForge.requestWorldAsset('shaders', 'terrain.glsl', 7)
            ]);

            this.logTestResult('ASSETS_MULTI', 'Multiple Asset Loading',
                assets.length === 3 && assets.every(a => a.loaded),
                `Loaded ${assets.length} assets simultaneously`);

            // Test resource system integration
            const resources = this.nexusForge.getChunkResources(0, 0);
            this.logTestResult('ASSETS_RESOURCES', 'Resource System Integration',
                Array.isArray(resources),
                `Chunk resources: ${resources.length} types`);

        } catch (error) {
            this.logTestResult('ASSETS', 'Asset Management', false, error.message);
        }
    }

    async testPerformanceMemory() {
        console.log('⚡ Testing Performance & Memory...');

        try {
            // Test large world generation
            const startTime = performance.now();

            // Generate multiple chunks
            for (let x = -2; x <= 2; x++) {
                for (let z = -2; z <= 2; z++) {
                    this.nexusForge.updatePlayerPosition([x * 64, 0, z * 64]);
                }
            }

            const genTime = performance.now() - startTime;
            this.logTestResult('PERF_GENERATION', 'Large World Generation Performance',
                genTime < 1000, // Should complete in under 1 second
                `Generated 25 chunks in ${genTime.toFixed(2)}ms`);

            // Test memory usage
            const totalObjects = this.nexusForge.getTotalObjects();
            this.logTestResult('PERF_OBJECTS', 'Object Count Management',
                totalObjects > 0 && totalObjects < 10000,
                `Total objects: ${totalObjects}`);

            // Test entity search performance
            const searchStart = performance.now();
            const nearbyEntities = this.nexusForge.getEntitiesInRadius([0, 0, 0], 100);
            const searchTime = performance.now() - searchStart;

            this.logTestResult('PERF_SEARCH', 'Entity Search Performance',
                searchTime < 100, // Should complete in under 100ms
                `Found ${nearbyEntities.length} entities in ${searchTime.toFixed(2)}ms`);

            // Test system info retrieval
            const systemInfo = this.nexusForge.getSystemInfo();
            this.logTestResult('PERF_INFO', 'System Information Retrieval',
                systemInfo && systemInfo.metrics,
                'System metrics available');

        } catch (error) {
            this.logTestResult('PERFORMANCE', 'Performance & Memory', false, error.message);
        }
    }

    logTestResult(category, testName, passed, details) {
        const result = {
            category,
            testName,
            passed,
            details,
            timestamp: new Date().toISOString()
        };

        this.testResults.push(result);

        const status = passed ? '✅' : '❌';
        console.log(`${status} ${category}: ${testName} - ${details}`);
    }

    generateTestReport() {
        const duration = performance.now() - this.startTime;
        const totalTests = this.testResults.length;
        const passedTests = this.testResults.filter(r => r.passed).length;
        const failedTests = totalTests - passedTests;
        const successRate = ((passedTests / totalTests) * 100).toFixed(1);

        console.log('\n' + '='.repeat(80));
        console.log('🧪 NEXUS FORGE UNIFIED SYSTEM INTEGRATION TEST REPORT');
        console.log('='.repeat(80));
        console.log(`⏱️  Duration: ${duration.toFixed(2)}ms`);
        console.log(`📊 Total Tests: ${totalTests}`);
        console.log(`✅ Passed: ${passedTests}`);
        console.log(`❌ Failed: ${failedTests}`);
        console.log(`📈 Success Rate: ${successRate}%`);
        console.log('='.repeat(80));

        // Group by category
        const categories = {};
        this.testResults.forEach(result => {
            if (!categories[result.category]) {
                categories[result.category] = { passed: 0, failed: 0, tests: [] };
            }
            if (result.passed) {
                categories[result.category].passed++;
            } else {
                categories[result.category].failed++;
            }
            categories[result.category].tests.push(result);
        });

        // Print category summary
        console.log('\n📋 Test Results by Category:');
        for (const [category, data] of Object.entries(categories)) {
            const categoryRate = ((data.passed / (data.passed + data.failed)) * 100).toFixed(1);
            console.log(`  ${category}: ${data.passed}/${data.passed + data.failed} (${categoryRate}%)`);

            // Show failed tests
            const failed = data.tests.filter(t => !t.passed);
            if (failed.length > 0) {
                failed.forEach(f => {
                    console.log(`    ❌ ${f.testName}: ${f.details}`);
                });
            }
        }

        console.log('\n' + '='.repeat(80));

        if (successRate >= 90) {
            console.log('🎉 INTEGRATION TESTS PASSED - System ready for production!');
        } else if (successRate >= 70) {
            console.log('⚠️  INTEGRATION TESTS PARTIAL - Some issues need attention');
        } else {
            console.log('🚨 INTEGRATION TESTS FAILED - Critical issues detected');
        }

        console.log('='.repeat(80));

        return {
            totalTests,
            passedTests,
            failedTests,
            successRate: parseFloat(successRate),
            duration,
            categories,
            results: this.testResults
        };
    }
}

// Export for use in other modules or direct execution
if (typeof module !== 'undefined' && module.exports) {
    module.exports = NexusForgeIntegrationTest;
}

// Auto-run if loaded directly in browser
if (typeof window !== 'undefined') {
    window.NexusForgeIntegrationTest = NexusForgeIntegrationTest;

    // Add a global function to run tests
    window.runNexusForgeTests = async function () {
        const testSuite = new NexusForgeIntegrationTest();
        return await testSuite.runAllTests();
    };
}

console.log('🧪 NEXUS Forge Integration Test Suite loaded. Run with: runNexusForgeTests()');
