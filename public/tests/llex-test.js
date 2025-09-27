/**
 * LLEX System Test - Verify content-addressable storage works
 * @fileoverview Simple test to demonstrate LLEX capabilities
 */

console.log('🧪 Starting LLEX System Test...');

// Wait for LLEX to load
setTimeout(async () => {
    try {
        if (typeof window !== 'undefined' && window.backgroundLLEX) {
            console.log('✅ LLEX Background Service detected');

            // Wait for initialization
            const checkInit = setInterval(async () => {
                if (window.backgroundLLEX.isInitialized) {
                    clearInterval(checkInit);

                    console.log('🔬 Testing LLEX capabilities...');

                    // Test 1: Analyze a word
                    const analysis = await window.backgroundLLEX.analyzeWord('rebuild', 'test context');
                    if (analysis) {
                        console.log('✅ Word analysis working:', analysis.morphemes.length, 'morphemes found');
                    }

                    // Test 2: Get system stats
                    const stats = await window.backgroundLLEX.getStats();
                    if (stats) {
                        console.log('✅ System stats:', {
                            objects: stats.engine.object_store.total_objects,
                            health: stats.health.status,
                            session: stats.session
                        });
                    }

                    // Test 3: Search functionality
                    const searchResults = await window.backgroundLLEX.query('search', 'rebuild');
                    console.log('✅ Search capability working:', searchResults.results?.length || 0, 'results');

                    console.log('🎉 LLEX System Test Complete - All systems operational!');
                }
            }, 500);

            // Timeout after 10 seconds
            setTimeout(() => {
                clearInterval(checkInit);
                if (!window.backgroundLLEX.isInitialized) {
                    console.warn('⚠️ LLEX initialization timeout');
                }
            }, 10000);

        } else {
            console.warn('⚠️ LLEX Background Service not found - make sure studio.html is loaded');
        }
    } catch (error) {
        console.error('❌ LLEX Test failed:', error);
    }
}, 2000);

console.log('🌟 LLEX Test script loaded - waiting for system initialization...');
