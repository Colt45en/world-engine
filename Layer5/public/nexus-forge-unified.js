/**
 * NEXUS FORGE UNIFIED SYSTEM
 * ==========================
 *
 * Complete unified open-world game development framework combining:
 * • LLEMath linear algebra foundation (from world-engine-unified.js)
 * • AI Pattern Recognition Engine (from nexus_forge_primordial.js)
 * • Mathematical Synthesis Engine (from nexus_synthesis_engine.js)
 * • Holy Beat Mathematical Engine (from nexus_holy_beat_math_engine.js)
 * • World Generation Systems (from C++ NexusGameEngine concepts)
 * • Audio-Visual Synchronization
 * • Procedural Content Generation
 *
 * No duplicates - each system serves a specific purpose in the unified architecture.
 */

// =============================================================================
// SECTION A: CORE MATHEMATICAL FOUNDATION (LLEMath Integration)
// =============================================================================

class UnifiedLLEMath {
    constructor() {
        this.precision = 1e-10;
        this.maxIterations = 1000;
        this.validationEnabled = true;

        // Initialize morpheme registry for linguistic transformations
        this.morphemeRegistry = this.createBuiltInMorphemes();
    }

    // Enhanced matrix operations from world-engine-unified.js and world-engine-math-unified.js
    multiply(A, B) {
        const isMatA = Array.isArray(A?.[0]);
        const isMatB = Array.isArray(B?.[0]);
        if (!isMatA) throw new Error('multiply: A must be matrix');

        if (isMatB) {
            const r = A.length, k = A[0].length, k2 = B.length, c = B[0].length;
            if (k !== k2) throw new Error(`multiply: shape mismatch (${r}×${k})·(${k2}×${c})`);
            return Array.from({ length: r }, (_, i) =>
                Array.from({ length: c }, (_, j) =>
                    A[i].reduce((sum, aij, t) => sum + aij * B[t][j], 0)
                )
            );
        }

        const r = A.length, k = A[0].length, n = B.length;
        if (k !== n) throw new Error(`multiply: shape mismatch (${r}×${k})·(${n})`);
        return Array.from({ length: r }, (_, i) =>
            A[i].reduce((sum, aij, j) => sum + aij * B[j], 0)
        );
    }

    transpose(A) {
        if (!Array.isArray(A?.[0])) throw new Error('transpose: A must be matrix');
        const r = A.length, c = A[0].length;
        return Array.from({ length: c }, (_, j) =>
            Array.from({ length: r }, (_, i) => A[i][j])
        );
    }

    identity(n) {
        return Array.from({ length: n }, (_, i) =>
            Array.from({ length: n }, (_, j) => (i === j ? 1 : 0))
        );
    }

    diagonal(values) {
        const n = values.length;
        return Array.from({ length: n }, (_, i) =>
            Array.from({ length: n }, (_, j) => (i === j ? values[i] : 0))
        );
    }

    // Pseudoinverse for advanced linear algebra
    pseudoInverse(A, lambda = 1e-6) {
        const AT = this.transpose(A);
        const m = A.length, n = A[0].length;

        if (m >= n) {
            const AAT = this.multiply(A, AT);
            const reg = AAT.map((row, i) => row.map((v, j) => v + (i === j ? lambda : 0)));
            const inv = this.inverseSmall(reg);
            return this.multiply(AT, inv);
        } else {
            const ATA = this.multiply(AT, A);
            const reg = ATA.map((row, i) => row.map((v, j) => v + (i === j ? lambda : 0)));
            const inv = this.inverseSmall(reg);
            return this.multiply(inv, AT);
        }
    }

    inverseSmall(M) {
        const n = M.length;
        const A = M.map((row, i) => [...row, ...this.identity(n)[i]]);
        for (let i = 0; i < n; i++) {
            const piv = A[i][i];
            if (Math.abs(piv) < 1e-12) throw new Error('inverseSmall: singular matrix');
            const invPiv = 1 / piv;
            for (let j = 0; j < 2 * n; j++) A[i][j] *= invPiv;
            for (let r = 0; r < n; r++) if (r !== i) {
                const f = A[r][i];
                for (let j = 0; j < 2 * n; j++) A[r][j] -= f * A[i][j];
            }
        }
        return A.map(row => row.slice(n));
    }

    // Enhanced matrix operations for game development
    rotationMatrix2D(angle) {
        const cos = Math.cos(angle);
        const sin = Math.sin(angle);
        return [
            [cos, -sin],
            [sin, cos]
        ];
    }

    rotationMatrix3D(axis, angle) {
        const [x, y, z] = axis;
        const cos = Math.cos(angle);
        const sin = Math.sin(angle);
        const oneMinusCos = 1 - cos;

        return [
            [cos + x * x * oneMinusCos, x * y * oneMinusCos - z * sin, x * z * oneMinusCos + y * sin],
            [y * x * oneMinusCos + z * sin, cos + y * y * oneMinusCos, y * z * oneMinusCos - x * sin],
            [z * x * oneMinusCos - y * sin, z * y * oneMinusCos + x * sin, cos + z * z * oneMinusCos]
        ];
    }

    scaleMatrix(sx, sy, sz = 1) {
        return [
            [sx, 0, 0],
            [0, sy, 0],
            [0, 0, sz]
        ];
    }

    translationMatrix(tx, ty, tz = 0) {
        return [
            [1, 0, 0, tx],
            [0, 1, 0, ty],
            [0, 0, 1, tz],
            [0, 0, 0, 1]
        ];
    }

    isValidMatrix(matrix) {
        if (!Array.isArray(matrix) || matrix.length === 0) return false;
        if (!Array.isArray(matrix[0])) return false;

        const cols = matrix[0].length;
        return matrix.every(row => Array.isArray(row) && row.length === cols);
    }

    // Vector operations for 3D world systems
    vectorAdd(a, b) {
        return a.map((val, i) => val + b[i]);
    }

    vectorSubtract(a, b) {
        return a.map((val, i) => val - b[i]);
    }

    vectorScale(vector, scalar) {
        return vector.map(val => val * scalar);
    }

    dotProduct(a, b) {
        return a.reduce((sum, val, i) => sum + val * b[i], 0);
    }

    crossProduct(a, b) {
        if (a.length !== 3 || b.length !== 3) {
            throw new Error('Cross product requires 3D vectors');
        }
        return [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0]
        ];
    }

    vectorMagnitude(vector) {
        return Math.sqrt(vector.reduce((sum, val) => sum + val * val, 0));
    }

    vectorNormalize(vector) {
        const magnitude = this.vectorMagnitude(vector);
        return magnitude > 0 ? vector.map(val => val / magnitude) : vector;
    }

    clamp(v, min, max) {
        return Math.max(min, Math.min(max, v));
    }

    // Safety and validation methods
    validateFinite(vec, name = 'vector') {
        if (!vec.every(Number.isFinite)) throw new Error(`${name} contains non-finite values: ${vec}`);
        return true;
    }

    safeTransform(M, x, b = null) {
        this.validateFinite(x, 'input vector');
        if (b) this.validateFinite(b, 'bias vector');
        const result = this.multiply(M, x);
        if (b) {
            const final = this.vectorAdd(result, b);
            this.validateFinite(final, 'result vector');
            return final;
        }
        this.validateFinite(result, 'result vector');
        return result;
    }

    projectionMatrix(dims, keepDims) {
        const P = Array.from({ length: keepDims.length }, () => Array(dims).fill(0));
        keepDims.forEach((dim, i) => {
            if (dim >= dims) throw new Error('projection: index out of range');
            P[i][dim] = 1;
        });
        return P;
    }

    // =============================================================================
    // MORPHEME-TO-BUTTON TRANSFORMATION SYSTEM (from world-engine-math-unified.js)
    // =============================================================================

    createBuiltInMorphemes(dim = 3) {
        const morphemes = new Map();

        morphemes.set('re', {
            symbol: 're',
            M: [[0.9, 0, 0], [0, 1.1, 0], [0, 0, 1]],
            b: [0.1, 0, 0],
            effects: { deltaLevel: -1, alpha: 0.95, description: 'repetition, restoration' }
        });

        morphemes.set('un', {
            symbol: 'un',
            M: [[-1, 0, 0], [0, 1, 0], [0, 0, 1]],
            b: [0, 0, 0],
            effects: { alpha: 0.8, description: 'negation, reversal' }
        });

        morphemes.set('counter', {
            symbol: 'counter',
            M: [[-0.8, 0.2, 0], [0.2, -0.8, 0], [0, 0, 1]],
            b: [0, 0, 0],
            effects: { deltaLevel: 1, description: 'opposition, counteraction' }
        });

        morphemes.set('multi', {
            symbol: 'multi',
            M: [[1.5, 0, 0], [0, 1.5, 0], [0, 0, 1.2]],
            b: [0, 0, 0.1],
            effects: { deltaLevel: 1, description: 'multiplication, many' }
        });

        morphemes.set('ize', {
            symbol: 'ize',
            M: [[1, 0.1, 0], [0, 1, 0.1], [0, 0, 1.1]],
            b: [0, 0, 0],
            effects: { deltaLevel: 1, description: 'make into, become' }
        });

        morphemes.set('ness', {
            symbol: 'ness',
            M: [[1, 0, 0], [0, 1, 0], [0.1, 0.1, 0.9]],
            b: [0, 0, 0.1],
            effects: {
                C: [[1.1, 0, 0], [0, 1.1, 0], [0, 0, 0.9]],
                description: 'quality, state'
            }
        });

        morphemes.set('ment', {
            symbol: 'ment',
            M: [[1.1, 0, 0], [0, 0.9, 0], [0, 0, 1]],
            b: [0, 0.1, 0],
            effects: { deltaLevel: 1, description: 'result, action' }
        });

        morphemes.set('ing', {
            symbol: 'ing',
            M: [[1, 0.2, 0], [0, 1.2, 0], [0, 0, 1]],
            b: [0, 0, 0],
            effects: { alpha: 1.1, description: 'ongoing action' }
        });

        morphemes.set('build', {
            symbol: 'build',
            M: [[1.2, 0, 0], [0, 1.2, 0], [0, 0, 1.1]],
            b: [0.1, 0.1, 0],
            effects: { deltaLevel: 1, description: 'construct, create' }
        });

        morphemes.set('move', {
            symbol: 'move',
            M: [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            b: [0.2, 0, 0],
            effects: { description: 'change position' }
        });

        morphemes.set('scale', {
            symbol: 'scale',
            M: [[1.1, 0, 0], [0, 1.1, 0], [0, 0, 1.1]],
            b: [0, 0, 0],
            effects: { description: 'change size' }
        });

        return morphemes;
    }

    createButton(label, abbr, wordClass, morphemes, options = {}) {
        const baseDim = options.dimensions || 3;

        let M = this.identity(baseDim);
        let b = Array(baseDim).fill(0);
        let C = this.identity(baseDim);
        let alpha = 1.0, beta = 0.0, delta = 0;

        // Apply morphemes in sequence
        for (const morphemeSymbol of morphemes) {
            const morpheme = this.morphemeRegistry.get(morphemeSymbol);
            if (!morpheme) continue;

            const newM = this.multiply(morpheme.M, M);
            const newB = this.vectorAdd(this.multiply(morpheme.M, b), morpheme.b);
            M = newM;
            b = newB;

            if (morpheme.effects?.C) C = this.multiply(morpheme.effects.C, C);
            if (morpheme.effects?.deltaLevel) delta += morpheme.effects.deltaLevel;
            if (typeof morpheme.effects?.alpha === 'number') alpha *= morpheme.effects.alpha;
            if (typeof morpheme.effects?.beta === 'number') beta += morpheme.effects.beta;
        }

        return {
            label,
            abbr,
            wordClass,
            morphemes,
            M: options.M || M,
            b: options.b || b,
            C: options.C || C,
            alpha: options.alpha ?? alpha,
            beta: options.beta ?? beta,
            deltaLevel: options.deltaLevel ?? delta,
            inputType: options.inputType || 'State',
            outputType: options.outputType || 'State',
            description: options.description || '',

            // Apply button transformation to state
            apply: (gameState) => {
                if (!gameState.vector) gameState.vector = [0, 0, 0];
                const transformed = this.safeTransform(M, gameState.vector, b);
                return {
                    ...gameState,
                    vector: transformed,
                    level: (gameState.level || 0) + delta,
                    alpha: (gameState.alpha || 1) * alpha,
                    beta: (gameState.beta || 0) + beta
                };
            }
        };
    }

    // Create some common game development buttons
    createGameButtons() {
        const buttons = new Map();

        // Movement buttons
        buttons.set('moveForward', this.createButton('Move Forward', 'MF', 'verb', ['move'], {
            M: [[1, 0, 0], [0, 1, 0], [0, 1, 1]],
            description: 'Move player forward'
        }));

        buttons.set('build', this.createButton('Build Structure', 'BS', 'verb', ['build'], {
            description: 'Construct buildings and structures'
        }));

        buttons.set('rebuild', this.createButton('Rebuild', 'RB', 'verb', ['re', 'build'], {
            description: 'Reconstruct existing structures'
        }));

        buttons.set('multiScale', this.createButton('Multi Scale', 'MS', 'verb', ['multi', 'scale'], {
            description: 'Scale multiple objects simultaneously'
        }));

        buttons.set('counterMove', this.createButton('Counter Move', 'CM', 'verb', ['counter', 'move'], {
            description: 'Move in opposition to current direction'
        }));

        return buttons;
    }
}

// =============================================================================
// SECTION B: AI PATTERN RECOGNITION ENGINE (from nexus_forge_primordial.js)
// =============================================================================

class UnifiedAIPatternEngine {
    constructor() {
        this.patterns = new Map();
        this.painThreshold = 0.7;
        this.opportunityThreshold = 0.6;
        this.clusteringEnabled = true;
        this.learningRate = 0.1;
        this.memorySize = 1000;
        this.patternHistory = [];

        this.initializeBuiltInPatterns();
        console.log('🧠 Unified AI Pattern Engine initialized');
    }

    initializeBuiltInPatterns() {
        // Pain detection patterns
        this.addPattern('performance_bottleneck', {
            indicators: ['slow_render', 'frame_drop', 'memory_spike', 'cpu_peak'],
            painScore: 0.8,
            solutions: [
                'Optimize rendering pipeline',
                'Implement LOD system',
                'Add object pooling',
                'Profile and optimize hotspots'
            ],
            category: 'performance'
        });

        this.addPattern('world_generation_inconsistency', {
            indicators: ['terrain_gaps', 'biome_conflicts', 'chunk_loading_errors'],
            painScore: 0.7,
            solutions: [
                'Implement seamless chunk transitions',
                'Add biome blending algorithms',
                'Validate terrain generation parameters'
            ],
            category: 'world_generation'
        });

        // Opportunity patterns
        this.addPattern('audio_visual_sync_opportunity', {
            indicators: ['beat_detection_active', 'visual_elements_present', 'low_cpu_usage'],
            opportunityScore: 0.8,
            solutions: [
                'Implement beat-synchronized particle effects',
                'Add rhythm-based gameplay mechanics',
                'Create audio-reactive world elements'
            ],
            category: 'enhancement'
        });

        this.addPattern('procedural_content_expansion', {
            indicators: ['player_exploration_high', 'content_density_low', 'generation_resources_available'],
            opportunityScore: 0.7,
            solutions: [
                'Generate additional world chunks',
                'Add procedural quest generation',
                'Implement dynamic NPC placement'
            ],
            category: 'content'
        });
    }

    addPattern(name, pattern) {
        this.patterns.set(name, {
            ...pattern,
            detectionCount: 0,
            lastDetected: null,
            confidence: 0
        });
    }

    detectPatterns(gameState) {
        const detectedPatterns = [];
        const currentTime = Date.now();

        for (const [name, pattern] of this.patterns) {
            const confidence = this.calculatePatternConfidence(pattern, gameState);

            if (confidence > (pattern.painScore || pattern.opportunityScore || 0.5)) {
                pattern.detectionCount++;
                pattern.lastDetected = currentTime;
                pattern.confidence = confidence;

                detectedPatterns.push({
                    name,
                    type: pattern.category,
                    confidence,
                    solutions: pattern.solutions,
                    painScore: pattern.painScore || 0,
                    opportunityScore: pattern.opportunityScore || 0
                });
            }
        }

        // Store in pattern history for learning
        this.patternHistory.push({
            timestamp: currentTime,
            gameState: { ...gameState },
            detectedPatterns: [...detectedPatterns]
        });

        // Trim history to prevent memory issues
        if (this.patternHistory.length > this.memorySize) {
            this.patternHistory.shift();
        }

        return detectedPatterns;
    }

    calculatePatternConfidence(pattern, gameState) {
        let matchScore = 0;
        let totalIndicators = pattern.indicators.length;

        for (const indicator of pattern.indicators) {
            if (this.evaluateIndicator(indicator, gameState)) {
                matchScore++;
            }
        }

        return totalIndicators > 0 ? matchScore / totalIndicators : 0;
    }

    evaluateIndicator(indicator, gameState) {
        // Smart indicator evaluation based on game state
        switch (indicator) {
            case 'slow_render':
                return gameState.fps < 30;
            case 'frame_drop':
                return gameState.frameDrops > 5;
            case 'memory_spike':
                return gameState.memoryUsage > 0.8;
            case 'cpu_peak':
                return gameState.cpuUsage > 0.9;
            case 'terrain_gaps':
                return gameState.terrainErrors > 0;
            case 'biome_conflicts':
                return gameState.biomeInconsistencies > 0;
            case 'beat_detection_active':
                return gameState.audioAnalysis?.beatDetected || false;
            case 'visual_elements_present':
                return gameState.renderableObjects > 0;
            case 'low_cpu_usage':
                return gameState.cpuUsage < 0.6;
            case 'player_exploration_high':
                return gameState.playerMovement?.speed > 0.5;
            case 'content_density_low':
                return gameState.worldDensity < 0.3;
            case 'generation_resources_available':
                return gameState.memoryUsage < 0.7 && gameState.cpuUsage < 0.7;
            default:
                return gameState[indicator] || false;
        }
    }

    // Pattern clustering for optimization insights
    clusterPatterns() {
        if (!this.clusteringEnabled || this.patternHistory.length < 10) {
            return [];
        }

        // Simple k-means clustering on pattern frequency
        const patternFrequency = new Map();

        for (const historyEntry of this.patternHistory) {
            for (const pattern of historyEntry.detectedPatterns) {
                const count = patternFrequency.get(pattern.name) || 0;
                patternFrequency.set(pattern.name, count + 1);
            }
        }

        const clusters = [];
        const sorted = Array.from(patternFrequency.entries())
            .sort((a, b) => b[1] - a[1]);

        // Group patterns by frequency ranges
        const highFreq = sorted.filter(([, count]) => count > this.patternHistory.length * 0.3);
        const medFreq = sorted.filter(([, count]) => count > this.patternHistory.length * 0.1 && count <= this.patternHistory.length * 0.3);
        const lowFreq = sorted.filter(([, count]) => count <= this.patternHistory.length * 0.1);

        if (highFreq.length > 0) clusters.push({ type: 'critical', patterns: highFreq });
        if (medFreq.length > 0) clusters.push({ type: 'moderate', patterns: medFreq });
        if (lowFreq.length > 0) clusters.push({ type: 'occasional', patterns: lowFreq });

        return clusters;
    }

    getRecommendations(gameState) {
        const detectedPatterns = this.detectPatterns(gameState);
        const clusters = this.clusterPatterns();

        return {
            immediate: detectedPatterns
                .filter(p => p.painScore > this.painThreshold)
                .map(p => ({ pattern: p.name, solutions: p.solutions, priority: 'high' })),
            opportunities: detectedPatterns
                .filter(p => p.opportunityScore > this.opportunityThreshold)
                .map(p => ({ pattern: p.name, solutions: p.solutions, priority: 'medium' })),
            strategic: clusters
                .filter(c => c.type === 'critical')
                .map(c => ({
                    cluster: c.type,
                    patterns: c.patterns.map(p => p[0]),
                    priority: 'long-term'
                }))
        };
    }
}

// =============================================================================
// SECTION C: MATHEMATICAL SYNTHESIS ENGINE (from nexus_synthesis_engine.js)
// =============================================================================

class UnifiedMathSynthesisEngine {
    constructor() {
        this.vibeState = { p: 0, i: 0, g: 0, c: 0 };
        this.frameCount = 0;
        this.isActive = false;
        this.expressions = new Map();
        this.functions = new Map();

        this.initializeMathematicalFunctions();
        this.loadDefaultExpressions();
        console.log('🌊 Unified Mathematical Synthesis Engine initialized');
    }

    initializeMathematicalFunctions() {
        // Enhanced noise functions for world generation
        this.defineFunction('noise', (x, y = 0, z = 0) => {
            const hash = Math.sin(x * 12.9898 + y * 78.233 + z * 37.719) * 43758.5453;
            return (hash - Math.floor(hash)) * 2 - 1;
        });

        this.defineFunction('fbm', (x, y, octaves = 4) => {
            let value = 0;
            let amplitude = 1;
            for (let i = 0; i < octaves; i++) {
                value += this.functions.get('noise')(x, y) * amplitude;
                x *= 2;
                y *= 2;
                amplitude *= 0.5;
            }
            return value;
        });

        // Vibe state functions for audio-reactive world
        this.defineFunction('vibe_p', () => this.vibeState.p);
        this.defineFunction('vibe_i', () => this.vibeState.i);
        this.defineFunction('vibe_g', () => this.vibeState.g);
        this.defineFunction('vibe_c', () => this.vibeState.c);
        this.defineFunction('vibe_time', () => this.frameCount * 0.016);

        // Specialized synthesis functions
        this.defineFunction('heart', (x, y) => {
            const left = Math.pow(x * x + y * y - 1, 3);
            const right = x * x * y * y * y;
            return Math.abs(left - right) < 0.1 ? 1 : 0;
        });

        this.defineFunction('rose', (t, n = 5) => Math.cos(n * t));
        this.defineFunction('spiral', (t, a = 1, b = 0.1) => a + b * t);

        // World generation functions
        this.defineFunction('terrain', (x, y) => {
            return this.functions.get('noise')(x * 0.1, y * 0.1) * 10 +
                this.functions.get('noise')(x * 0.05, y * 0.05) * 20 +
                this.functions.get('noise')(x * 0.02, y * 0.02) * 40;
        });

        this.defineFunction('biome', (x, y, temperature, humidity) => {
            const temp = temperature + this.functions.get('noise')(x * 0.01, y * 0.01) * 0.3;
            const hum = humidity + this.functions.get('noise')(x * 0.013, y * 0.007) * 0.3;

            if (temp < -0.5) return 0; // Ice
            if (temp > 0.5 && hum < -0.3) return 1; // Desert
            if (hum > 0.3) return 2; // Forest
            return 3; // Plains
        });
    }

    defineFunction(name, fn) {
        this.functions.set(name, fn);
    }

    loadDefaultExpressions() {
        // Terrain generation expressions
        this.expressions.set('mountains', 'terrain(x, y) * (1 + vibe_i() * 0.5)');
        this.expressions.set('valleys', 'terrain(x, y) * -0.5 + vibe_p() * 10');
        this.expressions.set('plateaus', 'clamp(terrain(x, y), -20, 20) + vibe_c() * 5');

        // Audio-reactive expressions
        this.expressions.set('beat_terrain', 'terrain(x, y) + sin(vibe_time() * 4) * vibe_g() * 15');
        this.expressions.set('pulse_world', 'terrain(x, y) * (1 + sin(vibe_time() * 2) * 0.3)');

        // Artistic expressions
        this.expressions.set('heart_world', 'heart(x * 0.1, y * 0.1) * 50 + terrain(x, y) * 0.5');
        this.expressions.set('spiral_paths', 'spiral(atan2(y, x), sqrt(x*x + y*y) * 0.1) * 5');
    }

    updateVibeState(p, i, g, c) {
        this.vibeState = { p, i, g, c };
    }

    evaluate(expression, variables = {}) {
        try {
            // Simple expression evaluation with variable substitution
            let expr = expression;

            // Replace function calls
            for (const [name, fn] of this.functions) {
                const regex = new RegExp(`${name}\\(([^)]+)\\)`, 'g');
                expr = expr.replace(regex, (match, args) => {
                    const argValues = args.split(',').map(arg => this.evaluateVariable(arg.trim(), variables));
                    return fn(...argValues);
                });
            }

            // Replace variables
            for (const [varName, value] of Object.entries(variables)) {
                const regex = new RegExp(`\\b${varName}\\b`, 'g');
                expr = expr.replace(regex, value);
            }

            // Basic math expression evaluation (simplified)
            return this.evaluateMathExpression(expr);
        } catch (error) {
            console.warn(`Expression evaluation failed: ${expression}`, error);
            return 0;
        }
    }

    evaluateVariable(varStr, variables) {
        const trimmed = varStr.trim();
        if (!isNaN(trimmed)) return parseFloat(trimmed);
        if (variables[trimmed] !== undefined) return variables[trimmed];
        if (this.functions.has(trimmed)) return this.functions.get(trimmed)();
        return 0;
    }

    evaluateMathExpression(expr) {
        // Simplified math expression evaluator
        // In production, use a proper expression parser
        try {
            // Replace common math functions
            expr = expr.replace(/sin\(/g, 'Math.sin(');
            expr = expr.replace(/cos\(/g, 'Math.cos(');
            expr = expr.replace(/tan\(/g, 'Math.tan(');
            expr = expr.replace(/sqrt\(/g, 'Math.sqrt(');
            expr = expr.replace(/abs\(/g, 'Math.abs(');
            expr = expr.replace(/pi/g, 'Math.PI');
            expr = expr.replace(/atan2\(/g, 'Math.atan2(');
            expr = expr.replace(/clamp\(([^,]+),([^,]+),([^)]+)\)/g, 'Math.max($2, Math.min($1, $3))');

            return eval(expr);
        } catch (error) {
            return 0;
        }
    }

    update() {
        this.frameCount++;
        this.isActive = true;
    }

    generateTerrain(x, y, options = {}) {
        const expression = options.expression || 'terrain';
        const variables = { x, y, ...options.variables };

        if (this.expressions.has(expression)) {
            return this.evaluate(this.expressions.get(expression), variables);
        } else {
            return this.evaluate(expression, variables);
        }
    }
}

// =============================================================================
// SECTION D: HOLY BEAT MATHEMATICAL ENGINE (from nexus_holy_beat_math_engine.js)
// =============================================================================

class UnifiedHolyBeatEngine {
    constructor() {
        this.clock = {
            bpm: 120,
            beatsPerBar: 4,
            startTime: 0,
            running: false,
            currentTime: 0,
            beatPhase: 0,
            beat: 0,
            bar: 0
        };

        this.lfo = {
            amDivision: 4,
            fmDivision: 8,
            amDepth: 0.2,
            fmDepth: 6.0,
            amPhase: 0,
            fmPhase: 0
        };

        this.synth = {
            baseFreq: 220,
            harmonics: 6,
            partialGains: [],
            partialPhases: [],
            kappa: 0.1,
            filterCutoff: 4000,
            noiseLevel: 0.05,
            masterGain: 0.5
        };

        this.features = {
            spectralCentroid: 440,
            rmsEnergy: 0.1,
            strokeDensity: 0.5,
            terrainRoughness: 0.3
        };

        this.updateQueue = [];
        this.pendingUpdates = new Map();

        this.initializeHarmonicStack();
        console.log('🎵 Unified Holy Beat Mathematical Engine initialized');
    }

    initializeHarmonicStack() {
        this.synth.partialGains = [];
        this.synth.partialPhases = [];

        for (let n = 1; n <= this.synth.harmonics; n++) {
            this.synth.partialGains.push(1.0 / n);
            this.synth.partialPhases.push(Math.random() * 2 * Math.PI);
        }
    }

    updateClockState(currentTime) {
        if (!this.clock.running) return;

        this.clock.currentTime = currentTime;
        const elapsed = currentTime - this.clock.startTime;
        const beatPeriod = 60000 / this.clock.bpm; // ms per beat

        const totalBeats = elapsed / beatPeriod;
        this.clock.beat = Math.floor(totalBeats);
        this.clock.beatPhase = totalBeats - this.clock.beat;
        this.clock.bar = Math.floor(this.clock.beat / this.clock.beatsPerBar);

        // Update LFO phases
        this.lfo.amPhase = Math.sin(2 * Math.PI * this.clock.beatPhase / this.lfo.amDivision);
        this.lfo.fmPhase = Math.sin(2 * Math.PI * this.clock.beatPhase / this.lfo.fmDivision);
    }

    synthesizeHarmonicStack(t) {
        let signal = 0;
        const am = 1 + this.lfo.amDepth * this.lfo.amPhase;

        for (let n = 1; n <= this.synth.harmonics; n++) {
            const freq = this.synth.baseFreq * n;
            const fmMod = this.lfo.fmDepth * this.lfo.fmPhase * this.synth.kappa;
            const phase = 2 * Math.PI * freq * t + fmMod + this.synth.partialPhases[n - 1];

            signal += this.synth.partialGains[n - 1] * Math.sin(phase);
        }

        return signal * am * this.synth.masterGain;
    }

    getBeatSyncedValue(baseValue, modulation = 0.2) {
        const beatMod = Math.sin(2 * Math.PI * this.clock.beatPhase) * modulation;
        return baseValue * (1 + beatMod);
    }

    getWorldParameters() {
        return {
            terrainHeight: this.getBeatSyncedValue(1.0, 0.3),
            biomeDensity: this.getBeatSyncedValue(0.5, 0.2),
            particleDensity: this.getBeatSyncedValue(100, 0.5),
            lightIntensity: this.getBeatSyncedValue(0.8, 0.4),
            audioReactivity: this.lfo.amPhase * 0.5 + 0.5
        };
    }

    start() {
        this.clock.running = true;
        this.clock.startTime = performance.now();
    }

    stop() {
        this.clock.running = false;
    }

    setBPM(bpm) {
        this.clock.bpm = Math.max(60, Math.min(200, bpm));
    }
}

// =============================================================================
// SECTION E: WORLD GENERATION SYSTEM
// =============================================================================

class UnifiedWorldGenerator {
    constructor(mathEngine, synthEngine, beatEngine) {
        this.mathEngine = mathEngine;
        this.synthEngine = synthEngine;
        this.beatEngine = beatEngine;

        // C++ NexusGameEngine inspired architecture
        this.chunks = new Map();
        this.activeChunks = new Set();
        this.entities = new Map(); // Game entities (from C++ GameEntity concept)
        this.components = new Map(); // Component systems
        this.resourceEngine = null; // Asset/resource management

        // World parameters (from C++ SystemParameters)
        this.chunkSize = 64;
        this.renderDistance = 3;
        this.heightScale = 100;
        this.terrainRoughness = 0.4;

        // Biome system with enhanced properties
        this.biomes = {
            MOUNTAINS: { id: 0, name: 'Mountains', color: '#8B7355', heightMod: 1.5, resources: ['stone', 'metals'], vegetation: 0.2 },
            DESERT: { id: 1, name: 'Desert', color: '#D2B48C', heightMod: 0.3, resources: ['sand', 'crystals'], vegetation: 0.1 },
            FOREST: { id: 2, name: 'Forest', color: '#228B22', heightMod: 0.8, resources: ['wood', 'herbs'], vegetation: 0.9 },
            PLAINS: { id: 3, name: 'Plains', color: '#90EE90', heightMod: 0.5, resources: ['grain', 'wildlife'], vegetation: 0.6 },
            OCEAN: { id: 4, name: 'Ocean', color: '#1E90FF', heightMod: -0.5, resources: ['fish', 'pearls'], vegetation: 0.3 }
        };

        // LOD system (from C++ chunking concepts)
        this.lodLevels = [
            { distance: 1, detail: 1.0, maxObjects: 100 },
            { distance: 2, detail: 0.7, maxObjects: 50 },
            { distance: 3, detail: 0.4, maxObjects: 20 },
            { distance: 4, detail: 0.2, maxObjects: 5 }
        ];

        // Initialize component systems
        this.initializeComponentSystems();

        console.log('🌍 Unified World Generator initialized with C++ architecture');
    }

    generateChunk(chunkX, chunkZ) {
        const chunkKey = `${chunkX},${chunkZ}`;
        if (this.chunks.has(chunkKey)) {
            return this.chunks.get(chunkKey);
        }

        const chunk = {
            x: chunkX,
            z: chunkZ,
            heightMap: [],
            biomeMap: [],
            objects: [],
            generated: false
        };

        // Generate heightmap using synthesis engine
        for (let x = 0; x < this.chunkSize; x++) {
            chunk.heightMap[x] = [];
            chunk.biomeMap[x] = [];

            for (let z = 0; z < this.chunkSize; z++) {
                const worldX = chunkX * this.chunkSize + x;
                const worldZ = chunkZ * this.chunkSize + z;

                // Use synthesis engine for terrain generation
                const baseHeight = this.synthEngine.generateTerrain(worldX * 0.01, worldZ * 0.01);
                const beatMod = this.beatEngine.getWorldParameters().terrainHeight;
                const height = baseHeight * this.heightScale * beatMod;

                chunk.heightMap[x][z] = height;

                // Determine biome
                const temperature = this.synthEngine.functions.get('noise')(worldX * 0.005, worldZ * 0.005);
                const humidity = this.synthEngine.functions.get('noise')(worldX * 0.007, worldZ * 0.003);
                const biomeId = this.synthEngine.functions.get('biome')(worldX, worldZ, temperature, humidity);

                chunk.biomeMap[x][z] = Math.floor(Math.abs(biomeId)) % Object.keys(this.biomes).length;
            }
        }

        // Generate objects based on biome
        this.generateChunkObjects(chunk);

        chunk.generated = true;
        this.chunks.set(chunkKey, chunk);
        return chunk;
    }

    generateChunkObjects(chunk) {
        const beatParams = this.beatEngine.getWorldParameters();
        const objectDensity = beatParams.particleDensity * 0.01;

        for (let i = 0; i < this.chunkSize * objectDensity; i++) {
            const x = Math.floor(Math.random() * this.chunkSize);
            const z = Math.floor(Math.random() * this.chunkSize);
            const height = chunk.heightMap[x][z];
            const biome = chunk.biomeMap[x][z];

            if (height > 0) { // Above water level
                chunk.objects.push({
                    type: this.getObjectTypeForBiome(biome),
                    position: [x, height, z],
                    scale: 0.8 + Math.random() * 0.4,
                    rotation: Math.random() * Math.PI * 2
                });
            }
        }
    }

    getObjectTypeForBiome(biomeId) {
        switch (biomeId) {
            case 0: return 'rock'; // Mountains
            case 1: return 'cactus'; // Desert
            case 2: return 'tree'; // Forest
            case 3: return 'grass'; // Plains
            default: return 'rock';
        }
    }

    // C++ NexusGameEngine inspired methods
    initializeComponentSystems() {
        this.components.set('TerrainSystem', {
            update: (deltaTime) => this.updateTerrainSystem(deltaTime),
            initialize: () => console.log('🏔️ Terrain System initialized'),
            entities: new Set()
        });

        this.components.set('ResourceSystem', {
            update: (deltaTime) => this.updateResourceSystem(deltaTime),
            initialize: () => console.log('⚡ Resource System initialized'),
            entities: new Set()
        });

        this.components.set('BiomeSystem', {
            update: (deltaTime) => this.updateBiomeSystem(deltaTime),
            initialize: () => console.log('🌿 Biome System initialized'),
            entities: new Set()
        });

        // Initialize all component systems
        for (const [, system] of this.components) {
            system.initialize();
        }
    }

    createEntity(name, type = 'generic') {
        const entity = {
            id: `${type}_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
            name: name,
            type: type,
            position: [0, 0, 0],
            components: new Map(),
            active: true,
            created: Date.now()
        };

        this.entities.set(entity.id, entity);
        return entity;
    }

    addComponentToEntity(entityId, componentType, componentData) {
        const entity = this.entities.get(entityId);
        if (!entity) return false;

        entity.components.set(componentType, componentData);

        // Register entity with component system
        const system = this.components.get(componentType + 'System');
        if (system) {
            system.entities.add(entityId);
        }

        return true;
    }

    updateSystemWithBeat(deltaTime) {
        const beatParams = this.beatEngine.getWorldParameters();

        // Update terrain roughness from beat
        this.terrainRoughness = 0.4 + (beatParams.terrainHeight - 1.0) * 0.2;

        // Update all component systems
        for (const [, system] of this.components) {
            system.update(deltaTime);
        }
    }

    updateTerrainSystem(deltaTime) {
        // Update terrain entities based on beat parameters
        const beatParams = this.beatEngine.getWorldParameters();

        for (const entityId of this.components.get('TerrainSystem').entities) {
            const entity = this.entities.get(entityId);
            if (!entity || !entity.active) continue;

            const terrainComp = entity.components.get('Terrain');
            if (terrainComp) {
                // Apply beat modulation to terrain
                terrainComp.heightMod = 1.0 + Math.sin(Date.now() * 0.001) * beatParams.terrainHeight * 0.1;
            }
        }
    }

    updateResourceSystem(deltaTime) {
        // Update resource spawning based on biomes and beat
        const beatParams = this.beatEngine.getWorldParameters();

        if (Math.random() < beatParams.particleDensity * 0.001) {
            // Spawn resources in active chunks
            for (const chunkKey of this.activeChunks) {
                const chunk = this.chunks.get(chunkKey);
                if (!chunk) continue;

                // Create resource entity
                const resource = this.createEntity(`resource_${chunkKey}`, 'resource');
                this.addComponentToEntity(resource.id, 'Resource', {
                    type: this.getRandomResourceForChunk(chunk),
                    amount: Math.floor(Math.random() * 10) + 1,
                    respawnTime: 30000 // 30 seconds
                });
            }
        }
    }

    updateBiomeSystem(deltaTime) {
        // Update biome transitions and vegetation
        for (const [chunkKey, chunk] of this.chunks) {
            if (!this.activeChunks.has(chunkKey)) continue;

            // Biome-based environmental updates
            for (let x = 0; x < this.chunkSize; x++) {
                for (let z = 0; z < this.chunkSize; z++) {
                    const biomeId = chunk.biomeMap[x][z];
                    const biome = Object.values(this.biomes)[biomeId];

                    if (biome && Math.random() < biome.vegetation * 0.001) {
                        // Spawn vegetation
                        const vegEntity = this.createEntity(`vegetation_${x}_${z}`, 'vegetation');
                        vegEntity.position = [chunk.x * this.chunkSize + x, chunk.heightMap[x][z], chunk.z * this.chunkSize + z];
                    }
                }
            }
        }
    }

    getRandomResourceForChunk(chunk) {
        const biomeResources = [];

        // Collect resources from all biomes in chunk
        for (let x = 0; x < this.chunkSize; x++) {
            for (let z = 0; z < this.chunkSize; z++) {
                const biomeId = chunk.biomeMap[x][z];
                const biome = Object.values(this.biomes)[biomeId];
                if (biome && biome.resources) {
                    biomeResources.push(...biome.resources);
                }
            }
        }

        return biomeResources.length > 0
            ? biomeResources[Math.floor(Math.random() * biomeResources.length)]
            : 'generic';
    }

    // LOD system integration
    updateChunkLOD(chunkKey, playerPosition) {
        const chunk = this.chunks.get(chunkKey);
        if (!chunk) return;

        const [playerX, , playerZ] = playerPosition;
        const chunkCenterX = (chunk.x * this.chunkSize) + (this.chunkSize / 2);
        const chunkCenterZ = (chunk.z * this.chunkSize) + (this.chunkSize / 2);

        const distance = Math.sqrt(
            Math.pow(playerX - chunkCenterX, 2) + Math.pow(playerZ - chunkCenterZ, 2)
        ) / this.chunkSize;

        // Find appropriate LOD level
        let lodLevel = this.lodLevels[this.lodLevels.length - 1];
        for (const lod of this.lodLevels) {
            if (distance <= lod.distance) {
                lodLevel = lod;
                break;
            }
        }

        chunk.lodLevel = lodLevel;
        chunk.maxObjects = Math.min(chunk.objects.length, lodLevel.maxObjects);
    }

    updateActiveChunks(playerPosition) {
        const [playerX, , playerZ] = playerPosition;
        const centerChunkX = Math.floor(playerX / this.chunkSize);
        const centerChunkZ = Math.floor(playerZ / this.chunkSize);

        const newActiveChunks = new Set();

        for (let dx = -this.renderDistance; dx <= this.renderDistance; dx++) {
            for (let dz = -this.renderDistance; dz <= this.renderDistance; dz++) {
                const chunkX = centerChunkX + dx;
                const chunkZ = centerChunkZ + dz;
                const chunkKey = `${chunkX},${chunkZ}`;

                newActiveChunks.add(chunkKey);

                if (!this.chunks.has(chunkKey)) {
                    this.generateChunk(chunkX, chunkZ);
                }

                // Update LOD for chunk
                this.updateChunkLOD(chunkKey, playerPosition);
            }
        }

        // Update active chunks
        this.activeChunks = newActiveChunks;

        // Update systems with new chunk data
        this.updateSystemWithBeat(0.016); // Assume 60fps for systems update
    }

    getChunkAt(x, z) {
        const chunkX = Math.floor(x / this.chunkSize);
        const chunkZ = Math.floor(z / this.chunkSize);
        return this.chunks.get(`${chunkX},${chunkZ}`);
    }

    getHeightAt(x, z) {
        const chunk = this.getChunkAt(x, z);
        if (!chunk) return 0;

        const localX = Math.floor(x) % this.chunkSize;
        const localZ = Math.floor(z) % this.chunkSize;

        if (localX >= 0 && localX < this.chunkSize && localZ >= 0 && localZ < this.chunkSize) {
            return chunk.heightMap[localX][localZ];
        }

        return 0;
    }
}

// =============================================================================
// SECTION F: UNIFIED NEXUS FORGE SYSTEM
// =============================================================================

class NexusForgeUnified {
    constructor() {
        this.version = '1.0.0-unified';
        this.initialized = false;
        this.components = new Map();

        // Core engines
        this.math = new UnifiedLLEMath();
        this.ai = new UnifiedAIPatternEngine();
        this.synthesis = new UnifiedMathSynthesisEngine();
        this.beat = new UnifiedHolyBeatEngine();
        this.world = null; // Initialize after other engines

        // Game state
        this.gameState = {
            fps: 60,
            frameDrops: 0,
            memoryUsage: 0.3,
            cpuUsage: 0.2,
            terrainErrors: 0,
            biomeInconsistencies: 0,
            renderableObjects: 0,
            playerMovement: { speed: 0, direction: [0, 0, 1] },
            worldDensity: 0.5,
            audioAnalysis: { beatDetected: false, energy: 0 }
        };

        // Performance metrics
        this.metrics = {
            frameTime: 0,
            updateTime: 0,
            renderTime: 0,
            totalObjects: 0,
            activeChunks: 0
        };

        console.log('🚀 NEXUS FORGE UNIFIED SYSTEM initialized');
    }

    async initialize(options = {}) {
        try {
            console.log('🌟 Initializing Unified NEXUS Forge System...');
            const startTime = performance.now();

            // Initialize world generator with all engines
            this.world = new UnifiedWorldGenerator(this.math, this.synthesis, this.beat);

            // Register components
            this.components.set('math', this.math);
            this.components.set('ai', this.ai);
            this.components.set('synthesis', this.synthesis);
            this.components.set('beat', this.beat);
            this.components.set('world', this.world);

            // Start beat engine if audio is enabled
            if (options.enableAudio !== false) {
                this.beat.start();
            }

            // Initial world generation around origin
            this.world.updateActiveChunks([0, 0, 0]);

            const duration = performance.now() - startTime;
            this.initialized = true;

            console.log(`✅ NEXUS Forge Unified System ready in ${duration.toFixed(2)}ms`);
            console.log(`📊 Components: ${this.components.size}`);

            return true;
        } catch (error) {
            console.error('❌ NEXUS Forge Unified initialization failed:', error);
            return false;
        }
    }

    update(deltaTime) {
        if (!this.initialized) return;

        const updateStart = performance.now();

        // Update beat engine
        this.beat.updateClockState(performance.now());

        // Update synthesis engine with beat state
        this.synthesis.updateVibeState(
            this.beat.lfo.amPhase,
            this.beat.lfo.fmPhase,
            this.beat.features.strokeDensity,
            this.beat.features.terrainRoughness
        );
        this.synthesis.update();

        // Update game state
        this.updateGameState(deltaTime);

        // Run AI pattern detection
        const patterns = this.ai.detectPatterns(this.gameState);

        // Apply AI recommendations
        if (patterns.length > 0) {
            this.applyAIRecommendations(patterns);
        }

        this.metrics.updateTime = performance.now() - updateStart;
    }

    updateGameState(deltaTime) {
        // Simulate basic game state updates
        this.gameState.fps = Math.round(1000 / (deltaTime || 16.67));
        this.gameState.frameDrops = this.gameState.fps < 30 ? this.gameState.frameDrops + 1 : Math.max(0, this.gameState.frameDrops - 1);
        this.gameState.audioAnalysis.beatDetected = this.beat.clock.beatPhase < 0.1;
        this.gameState.audioAnalysis.energy = this.beat.features.rmsEnergy;

        // Update metrics
        this.metrics.totalObjects = this.getTotalObjects();
        this.metrics.activeChunks = this.world.activeChunks.size;
    }

    applyAIRecommendations(patterns) {
        for (const pattern of patterns) {
            if (pattern.painScore > 0.7) {
                console.log(`🔧 Applying fix for: ${pattern.name}`);
                // Apply performance optimizations
                if (pattern.name === 'performance_bottleneck') {
                    this.optimizePerformance();
                }
            }

            if (pattern.opportunityScore > 0.6) {
                console.log(`💡 Implementing enhancement: ${pattern.name}`);
                // Apply enhancements
                if (pattern.name === 'audio_visual_sync_opportunity') {
                    this.enhanceAudioVisualSync();
                }
            }
        }
    }

    optimizePerformance() {
        // Reduce render distance if performance is poor
        if (this.gameState.fps < 30) {
            this.world.renderDistance = Math.max(1, this.world.renderDistance - 1);
            console.log(`📉 Reduced render distance to ${this.world.renderDistance}`);
        }
    }

    enhanceAudioVisualSync() {
        // Increase audio reactivity
        this.beat.lfo.amDepth = Math.min(0.8, this.beat.lfo.amDepth + 0.1);
        this.beat.lfo.fmDepth = Math.min(20, this.beat.lfo.fmDepth + 2);
        console.log('🎵 Enhanced audio-visual synchronization');
    }

    getTotalObjects() {
        let total = 0;
        for (const [, chunk] of this.world.chunks) {
            total += chunk.objects.length;
        }
        return total;
    }

    // Public API methods - Enhanced with C++ integration
    generateTerrain(x, z, options = {}) {
        return this.synthesis.generateTerrain(x, z, options);
    }

    getWorldHeight(x, z) {
        return this.world.getHeightAt(x, z);
    }

    updatePlayerPosition(position) {
        this.gameState.playerMovement.speed = this.math.vectorMagnitude(
            this.math.vectorSubtract(position, this.gameState.playerMovement.lastPosition || position)
        );
        this.gameState.playerMovement.lastPosition = position;
        this.world.updateActiveChunks(position);
    }

    setBPM(bpm) {
        this.beat.setBPM(bpm);
    }

    // C++ NexusGameEngine inspired API extensions
    createWorldEntity(name, type, position = [0, 0, 0]) {
        if (!this.world) return null;

        const entity = this.world.createEntity(name, type);
        entity.position = [...position];
        return entity;
    }

    addEntityComponent(entityId, componentType, componentData) {
        if (!this.world) return false;
        return this.world.addComponentToEntity(entityId, componentType, componentData);
    }

    getEntitiesInRadius(position, radius) {
        if (!this.world) return [];

        const [centerX, centerY, centerZ] = position;
        const entities = [];

        for (const [, entity] of this.world.entities) {
            const [entityX, entityY, entityZ] = entity.position;
            const distance = Math.sqrt(
                Math.pow(centerX - entityX, 2) +
                Math.pow(centerY - entityY, 2) +
                Math.pow(centerZ - entityZ, 2)
            );

            if (distance <= radius) {
                entities.push(entity);
            }
        }

        return entities;
    }

    getChunkResources(chunkX, chunkZ) {
        if (!this.world) return [];

        const chunkKey = `${chunkX},${chunkZ}`;
        const chunk = this.world.chunks.get(chunkKey);
        if (!chunk) return [];

        const biome = Object.values(this.world.biomes)[chunk.biomeMap[0][0]];
        return biome ? biome.resources || [] : [];
    }

    getSystemStatus() {
        return {
            components: {
                math: this.math !== null,
                ai: this.ai !== null,
                synthesis: this.synthesis !== null,
                beat: this.beat !== null && this.beat.isRunning,
                world: this.world !== null
            },
            world: {
                chunksLoaded: this.world ? this.world.chunks.size : 0,
                activeChunks: this.world ? this.world.activeChunks.size : 0,
                entitiesCount: this.world ? this.world.entities.size : 0,
                componentSystems: this.world ? this.world.components.size : 0
            },
            performance: {
                fps: this.gameState.fps,
                memoryUsage: this.gameState.memoryUsage,
                totalObjects: this.getTotalObjects()
            },
            beat: {
                bpm: this.beat ? this.beat.bpm : 0,
                isRunning: this.beat ? this.beat.isRunning : false,
                beatCount: this.beat ? this.beat.beatCount : 0
            }
        };
    }

    // Asset management inspired by C++ ResourceEngine
    requestWorldAsset(type, name, priority = 5) {
        console.log(`🎯 Asset requested: ${type}/${name} (priority: ${priority})`);

        // Simulate async asset loading
        return new Promise((resolve) => {
            setTimeout(() => {
                const asset = {
                    type: type,
                    name: name,
                    loaded: true,
                    priority: priority,
                    size: Math.random() * 10 + 1 // MB
                };
                console.log(`✅ Asset loaded: ${type}/${name}`);
                resolve(asset);
            }, Math.random() * 100 + 50);
        });
    }

    getSystemInfo() {
        return {
            version: this.version,
            initialized: this.initialized,
            components: Array.from(this.components.keys()),
            metrics: this.metrics,
            gameState: this.gameState,
            beatInfo: {
                bpm: this.beat.clock.bpm,
                beat: this.beat.clock.beat,
                bar: this.beat.clock.bar,
                phase: this.beat.clock.beatPhase
            }
        };
    }

    getAIInsights() {
        return this.ai.getRecommendations(this.gameState);
    }
}

// =============================================================================
// GLOBAL INITIALIZATION AND EXPORT
// =============================================================================

// Global instance
window.NexusForgeUnified = NexusForgeUnified;

// Auto-initialize if requested
if (window.location.search.includes('auto-init')) {
    window.addEventListener('DOMContentLoaded', async () => {
        window.nexusForge = new NexusForgeUnified();
        await window.nexusForge.initialize();
        console.log('🚀 NEXUS Forge Unified auto-initialized');
    });
}

// Export for module systems
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { NexusForgeUnified };
}

console.log('📦 NEXUS Forge Unified System loaded successfully');
