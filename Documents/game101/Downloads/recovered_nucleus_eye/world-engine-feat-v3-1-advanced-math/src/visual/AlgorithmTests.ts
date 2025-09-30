// Algorithm Validation Tests for Visual Bleedway
import * as THREE from 'three';
import {
    otsu,
    morph,
    bbox,
    cleanMask,
    buildField,
    meshFromField,
    BUILTIN_PRESETS
} from './SilhouetteProcessing';

export async function runAlgorithmTests(): Promise<TestResults> {
    const results: TestResults = {
        passed: 0,
        failed: 0,
        tests: []
    };

    console.log('🧪 Running Visual Bleedway Algorithm Tests...\n');

    // Test 1: Otsu Auto-Thresholding
    await runTest(results, 'Otsu Auto-Thresholding', () => {
        const testData = new Uint8Array([0, 50, 100, 150, 200, 255]);
        const threshold = otsu(testData);

        if (threshold < 0 || threshold > 255) {
            throw new Error(`Invalid threshold: ${threshold}`);
        }

        console.log(`  ✓ Otsu threshold: ${threshold}`);
        return true;
    });

    // Test 2: Morphological Operations
    await runTest(results, 'Morphological Operations', () => {
        const testMask = createTestMask(32, 32);
        const eroded = morph(testMask, 32, 32, 'erode', 3);
        const dilated = morph(testMask, 32, 32, 'dilate', 3);

        if (eroded.length !== testMask.length || dilated.length !== testMask.length) {
            throw new Error('Morphology output size mismatch');
        }

        console.log(`  ✓ Morphology: erode/dilate operations completed`);
        return true;
    });

    // Test 3: Bounding Box Calculation
    await runTest(results, 'Bounding Box Calculation', () => {
        const testMask = createTestMask(64, 64, { centerSquare: true });
        const bounds = bbox(testMask, 64, 64);

        if (bounds.x1 >= bounds.x2 || bounds.y1 >= bounds.y2) {
            throw new Error(`Invalid bounding box: ${JSON.stringify(bounds)}`);
        }

        console.log(`  ✓ BBox: [${bounds.x1},${bounds.y1}] to [${bounds.x2},${bounds.y2}]`);
        return true;
    });

    // Test 4: Clean Mask Processing
    await runTest(results, 'Clean Mask Processing', () => {
        const testImage = createTestImageData(128, 256);
        const cleaned = cleanMask(testImage, {
            auto: true,
            threshold: 128,
            kernel: 3,
            maxSide: 256,
            flipX: false
        });

        if (!cleaned || cleaned.length === 0) {
            throw new Error('Clean mask returned empty result');
        }

        console.log(`  ✓ Clean mask: ${cleaned.length} pixels processed`);
        return true;
    });

    // Test 5: 3D Field Building
    await runTest(results, '3D Volume Field Building', () => {
        const frontMask = createTestMask(32, 32, { centerSquare: true });
        const sideMask = createTestMask(32, 32, { centerSquare: true });
        const field = buildField(frontMask, sideMask, 16);

        const expectedSize = 16 * 16 * 16;
        if (field.length !== expectedSize) {
            throw new Error(`Field size mismatch: got ${field.length}, expected ${expectedSize}`);
        }

        const nonZeroVoxels = field.filter(v => v > 0).length;
        console.log(`  ✓ 3D Field: ${expectedSize} voxels, ${nonZeroVoxels} non-zero`);
        return true;
    });

    // Test 6: Mesh Generation
    await runTest(results, 'Mesh Generation', () => {
        const frontMask = createTestMask(16, 16, { centerSquare: true });
        const sideMask = createTestMask(16, 16, { centerSquare: true });
        const field = buildField(frontMask, sideMask, 8);

        const mesh = meshFromField(field, 8, {
            iso: 0.5,
            height: 1.0,
            subs: 0,
            lap: 0,
            dec: 0,
            color: '#ffffff'
        });

        if (!mesh || !(mesh instanceof THREE.Mesh)) {
            throw new Error('Mesh generation failed');
        }

        const vertices = (mesh.geometry as THREE.BufferGeometry).getAttribute('position');
        console.log(`  ✓ Mesh: ${vertices.count} vertices generated`);
        return true;
    });

    // Test 7: Preset Validation
    await runTest(results, 'Preset Validation', () => {
        const presets = Object.keys(BUILTIN_PRESETS);
        if (presets.length === 0) {
            throw new Error('No presets found');
        }

        for (const preset of presets) {
            const config = BUILTIN_PRESETS[preset];
            if (!config.hasOwnProperty('auto') || !config.hasOwnProperty('thr')) {
                throw new Error(`Invalid preset: ${preset}`);
            }
        }

        console.log(`  ✓ Presets: ${presets.length} valid configurations`);
        return true;
    });

    // Test 8: Performance Benchmark
    await runTest(results, 'Performance Benchmark', async () => {
        const start = performance.now();

        const testImage = createTestImageData(256, 256);
        const cleaned = cleanMask(testImage, BUILTIN_PRESETS['Auto (Otsu)']);
        const field = buildField(cleaned, cleaned, 32);
        const mesh = meshFromField(field, 32, {
            iso: 0.5,
            height: 1.0,
            subs: 0,
            lap: 0,
            dec: 0,
            color: '#4ecdc4'
        });

        const elapsed = performance.now() - start;

        if (elapsed > 5000) { // 5 second threshold
            throw new Error(`Performance too slow: ${elapsed}ms`);
        }

        console.log(`  ✓ Performance: Complete pipeline in ${elapsed.toFixed(1)}ms`);
        return true;
    });

    // Summary
    const successRate = results.passed / (results.passed + results.failed) * 100;
    console.log(`\n🎯 Test Results: ${results.passed}/${results.passed + results.failed} passed (${successRate.toFixed(1)}%)`);

    if (results.failed === 0) {
        console.log('🎉 All Visual Bleedway algorithms are working correctly!\n');
    } else {
        console.log(`❌ ${results.failed} tests failed. Check console for details.\n`);
    }

    return results;
}

// Test utilities
async function runTest(results: TestResults, name: string, testFn: () => boolean | Promise<boolean>) {
    try {
        console.log(`Running: ${name}`);
        await testFn();
        results.passed++;
        results.tests.push({ name, status: 'passed', error: null });
        console.log(`✅ ${name}\n`);
    } catch (error) {
        results.failed++;
        results.tests.push({ name, status: 'failed', error: error.message });
        console.error(`❌ ${name}: ${error.message}\n`);
    }
}

function createTestMask(width: number, height: number, options: { centerSquare?: boolean } = {}): Uint8Array {
    const mask = new Uint8Array(width * height);

    if (options.centerSquare) {
        const centerX = Math.floor(width / 2);
        const centerY = Math.floor(height / 2);
        const size = Math.min(width, height) / 4;

        for (let y = centerY - size; y < centerY + size; y++) {
            for (let x = centerX - size; x < centerX + size; x++) {
                if (x >= 0 && x < width && y >= 0 && y < height) {
                    mask[y * width + x] = 255;
                }
            }
        }
    }

    return mask;
}

function createTestImageData(width: number, height: number): ImageData {
    const data = new Uint8ClampedArray(width * height * 4);

    // Create a simple gradient with some white areas
    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            const idx = (y * width + x) * 4;
            const centerDist = Math.sqrt(
                Math.pow(x - width / 2, 2) + Math.pow(y - height / 2, 2)
            );
            const maxDist = Math.sqrt(width * width + height * height) / 2;
            const value = centerDist < maxDist * 0.3 ? 255 : 0; // White center, black edges

            data[idx] = value;     // R
            data[idx + 1] = value; // G
            data[idx + 2] = value; // B
            data[idx + 3] = 255;   // A
        }
    }

    return new ImageData(data, width, height);
}

export type TestResults = {
    passed: number;
    failed: number;
    tests: Array<{
        name: string;
        status: 'passed' | 'failed';
        error: string | null;
    }>;
};

// Auto-run tests in development
if (process.env.NODE_ENV === 'development') {
    // Run tests after a brief delay to avoid blocking initial load
    setTimeout(() => {
        runAlgorithmTests().catch(console.error);
    }, 1000);
}
