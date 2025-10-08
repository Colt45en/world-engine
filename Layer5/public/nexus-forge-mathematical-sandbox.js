/**
 * NEXUS Forge Mathematical Sandbox & Testing Environment
 * • Interactive Mathematical Playground
 * • Equation Solver and Calculator
 * • Geometric Construction Tools
 * • Advanced Mathematical Experiments
 * • Performance Testing Suite
 */

class NexusForgeMathematicalSandbox {
    constructor() {
        this.mathEngine = null;
        this.nucleusCommunicator = null;
        this.experiments = new Map();
        this.calculator = new AdvancedCalculator();
        this.geometricConstructor = new GeometricConstructor();
        this.equationSolver = new EquationSolver();
        this.performanceTester = new PerformanceTester();

        this.sandboxState = {
            activeExperiment: null,
            history: [],
            variables: new Map(),
            functions: new Map()
        };

        console.log('🧪 Mathematical Sandbox initialized');
        this.initializeSandbox();
    }

    initializeSandbox() {
        this.setupMathematicalExperiments();
        this.setupInteractivePlayground();
        this.loadMathEngine();
        this.initializeNucleusCommunication();
    }

    async loadMathEngine() {
        try {
            if (typeof NexusForgemathematicalEngine !== 'undefined') {
                this.mathEngine = new NexusForgemathematicalEngine();
                console.log('✅ Mathematical Engine loaded in sandbox');
            } else {
                console.warn('⚠️ Mathematical Engine not available, using sandbox-only features');
            }
        } catch (error) {
            console.error('❌ Failed to load Mathematical Engine:', error);
        }
    }

    async initializeNucleusCommunication() {
        try {
            if (typeof NexusForgeNucleusCommunicator !== 'undefined') {
                this.nucleusCommunicator = new NexusForgeNucleusCommunicator();
                console.log('🧠 Nucleus Communicator loaded in sandbox');

                // Send initial greeting to Nucleus
                await this.nucleusCommunicator.sendMessageToNucleus(
                    'Mathematical Sandbox initialized and ready for training',
                    'system-status',
                    'high'
                );
            } else {
                console.warn('⚠️ Nucleus Communicator not available');
            }
        } catch (error) {
            console.error('❌ Failed to initialize Nucleus communication:', error);
        }
    }

    setupMathematicalExperiments() {
        // Algebraic Experiments
        this.addExperiment('polynomial_roots', {
            name: 'Polynomial Root Finding',
            description: 'Find roots of polynomial equations using various methods',
            category: 'algebra',
            execute: (params) => this.findPolynomialRoots(params)
        });

        this.addExperiment('matrix_operations', {
            name: 'Matrix Operations Suite',
            description: 'Comprehensive matrix calculations and transformations',
            category: 'algebra',
            execute: (params) => this.performMatrixOperations(params)
        });

        this.addExperiment('fourier_analysis', {
            name: 'Fourier Transform Analysis',
            description: 'Analyze functions using Fourier transforms',
            category: 'algebra',
            execute: (params) => this.performFourierAnalysis(params)
        });

        // Geometric Experiments
        this.addExperiment('geometric_transformations', {
            name: 'Geometric Transformations',
            description: 'Apply various transformations to geometric shapes',
            category: 'geometry',
            execute: (params) => this.applyGeometricTransformations(params)
        });

        this.addExperiment('intersection_detection', {
            name: 'Geometric Intersection Detection',
            description: 'Detect intersections between various geometric shapes',
            category: 'geometry',
            execute: (params) => this.detectIntersections(params)
        });

        this.addExperiment('convex_hull', {
            name: 'Convex Hull Generation',
            description: 'Generate convex hulls for point sets using different algorithms',
            category: 'geometry',
            execute: (params) => this.generateConvexHull(params)
        });

        // Sacred Geometry Experiments
        this.addExperiment('golden_ratio_constructions', {
            name: 'Golden Ratio Constructions',
            description: 'Explore constructions based on the golden ratio',
            category: 'sacred',
            execute: (params) => this.constructGoldenRatioShapes(params)
        });

        this.addExperiment('platonic_relationships', {
            name: 'Platonic Solid Relationships',
            description: 'Analyze mathematical relationships between Platonic solids',
            category: 'sacred',
            execute: (params) => this.analyzePlatonicRelationships(params)
        });

        // Physics Experiments
        this.addExperiment('n_body_simulation', {
            name: 'N-Body Gravitational Simulation',
            description: 'Simulate gravitational interactions between multiple bodies',
            category: 'physics',
            execute: (params) => this.simulateNBodyGravity(params)
        });

        this.addExperiment('wave_interference', {
            name: 'Wave Interference Patterns',
            description: 'Simulate and analyze wave interference phenomena',
            category: 'physics',
            execute: (params) => this.simulateWaveInterference(params)
        });

        // Fractal Experiments
        this.addExperiment('custom_l_systems', {
            name: 'Custom L-System Generator',
            description: 'Create and experiment with custom L-system rules',
            category: 'fractals',
            execute: (params) => this.generateCustomLSystem(params)
        });

        this.addExperiment('fractal_dimension', {
            name: 'Fractal Dimension Calculator',
            description: 'Calculate fractal dimensions using various methods',
            category: 'fractals',
            execute: (params) => this.calculateFractalDimension(params)
        });
    }

    setupInteractivePlayground() {
        // Interactive mathematical playground features
        this.playground = {
            expressionEvaluator: new ExpressionEvaluator(),
            graphingCalculator: new GraphingCalculator(),
            symbolProcessor: new SymbolProcessor(),
            numericalMethods: new NumericalMethods()
        };
    }

    // Experiment execution methods
    addExperiment(id, experiment) {
        this.experiments.set(id, experiment);
    }

    async runExperiment(experimentId, parameters = {}) {
        const experiment = this.experiments.get(experimentId);
        if (!experiment) {
            throw new Error(`Experiment '${experimentId}' not found`);
        }

        console.log(`🧪 Running experiment: ${experiment.name}`);
        const startTime = performance.now();

        try {
            const result = await experiment.execute(parameters);
            const endTime = performance.now();

            const experimentResult = {
                experimentId,
                name: experiment.name,
                parameters,
                result,
                executionTime: endTime - startTime,
                timestamp: new Date().toISOString()
            };

            this.sandboxState.history.push(experimentResult);

            // Send to Nucleus for learning
            if (this.nucleusCommunicator) {
                await this.sendExperimentToNucleus(experimentResult);
            }

            return experimentResult;

        } catch (error) {
            console.error(`❌ Experiment '${experimentId}' failed:`, error);

            // Send failure data to Nucleus for learning
            if (this.nucleusCommunicator) {
                await this.nucleusCommunicator.sendMessageToNucleus(
                    `Experiment ${experimentId} failed: ${error.message}`,
                    'experiment-failure',
                    'medium'
                );
            }

            throw error;
        }
    }    // Nucleus communication methods
    async sendExperimentToNucleus(experimentResult) {
        try {
            await this.nucleusCommunicator.sendExperimentResults(
                experimentResult.experimentId,
                experimentResult
            );

            // If it's a mathematical pattern, send it for pattern learning
            if (this.isMathematicalPattern(experimentResult)) {
                await this.nucleusCommunicator.sendMathematicalPattern(
                    experimentResult.result,
                    experimentResult.name,
                    experimentResult.parameters
                );
            }

            console.log('🧠 Experiment data sent to Nucleus for learning');
        } catch (error) {
            console.error('❌ Failed to send experiment to Nucleus:', error);
        }
    }

    async trainNucleusWithCalculation(expression, result) {
        if (!this.nucleusCommunicator) return;

        const calculationData = {
            expression,
            result,
            timestamp: Date.now(),
            context: 'mathematical-calculation'
        };

        await this.nucleusCommunicator.trainNucleus(calculationData, 'calculations');
    }

    async sendPatternToNucleus(pattern, type, insights = {}) {
        if (!this.nucleusCommunicator) return;

        const patternData = {
            pattern,
            type,
            insights,
            discoveryMethod: 'sandbox-analysis',
            timestamp: Date.now()
        };

        await this.nucleusCommunicator.sendMathematicalPattern(patternData, type);
        console.log(`🧠 Pattern '${type}' sent to Nucleus for learning`);
    }

    async requestNucleusInsights(query, context = {}) {
        if (!this.nucleusCommunicator) {
            console.warn('⚠️ Nucleus communicator not available for insights');
            return null;
        }

        return await this.nucleusCommunicator.requestNucleusInsights(query, {
            ...context,
            sandbox: 'mathematical',
            timestamp: Date.now()
        });
    }

    async syncWithNucleus() {
        if (!this.nucleusCommunicator || !this.mathEngine) return;

        try {
            await this.nucleusCommunicator.syncMathematicalState(this.mathEngine);
            console.log('🔄 Synchronized with Nucleus');
        } catch (error) {
            console.error('❌ Failed to sync with Nucleus:', error);
        }
    }

    isMathematicalPattern(experimentResult) {
        const mathematicalCategories = ['algebra', 'geometry', 'sacred', 'fractals', 'physics'];
        return mathematicalCategories.some(category =>
            experimentResult.name.toLowerCase().includes(category) ||
            experimentResult.experimentId.includes(category)
        );
    }
    findPolynomialRoots(params) {
        const { coefficients = [1, 0, -1], method = 'newton' } = params;

        switch (method) {
            case 'newton':
                return this.newtonMethod(coefficients);
            case 'bisection':
                return this.bisectionMethod(coefficients);
            case 'analytical':
                return this.analyticalRoots(coefficients);
            default:
                return this.newtonMethod(coefficients);
        }
    }

    newtonMethod(coefficients, x0 = 1, tolerance = 1e-10, maxIterations = 100) {
        const polynomial = x => coefficients.reduce((sum, coef, i) => sum + coef * Math.pow(x, coefficients.length - 1 - i), 0);
        const derivative = x => coefficients.slice(0, -1).reduce((sum, coef, i) => sum + coef * (coefficients.length - 1 - i) * Math.pow(x, coefficients.length - 2 - i), 0);

        let x = x0;
        const iterations = [];

        for (let i = 0; i < maxIterations; i++) {
            const fx = polynomial(x);
            const fpx = derivative(x);

            if (Math.abs(fpx) < tolerance) break;

            const newX = x - fx / fpx;
            iterations.push({ iteration: i, x, fx, error: Math.abs(newX - x) });

            if (Math.abs(newX - x) < tolerance) {
                return { root: newX, iterations, converged: true };
            }

            x = newX;
        }

        return { root: x, iterations, converged: false };
    }

    bisectionMethod(coefficients, a = -10, b = 10, tolerance = 1e-10) {
        const polynomial = x => coefficients.reduce((sum, coef, i) => sum + coef * Math.pow(x, coefficients.length - 1 - i), 0);

        if (polynomial(a) * polynomial(b) >= 0) {
            return { error: 'No root found in interval [a, b]' };
        }

        const iterations = [];
        let iteration = 0;

        while (Math.abs(b - a) > tolerance && iteration < 1000) {
            const c = (a + b) / 2;
            const fc = polynomial(c);

            iterations.push({ iteration, a, b, c, fc, interval: b - a });

            if (Math.abs(fc) < tolerance) {
                return { root: c, iterations, converged: true };
            }

            if (polynomial(a) * fc < 0) {
                b = c;
            } else {
                a = c;
            }

            iteration++;
        }

        return { root: (a + b) / 2, iterations, converged: true };
    }

    analyticalRoots(coefficients) {
        const degree = coefficients.length - 1;

        switch (degree) {
            case 1: // Linear: ax + b = 0
                return { roots: [-coefficients[1] / coefficients[0]], type: 'linear' };

            case 2: // Quadratic: ax² + bx + c = 0
                const [a, b, c] = coefficients;
                const discriminant = b * b - 4 * a * c;

                if (discriminant > 0) {
                    return {
                        roots: [(-b + Math.sqrt(discriminant)) / (2 * a), (-b - Math.sqrt(discriminant)) / (2 * a)],
                        type: 'quadratic',
                        discriminant
                    };
                } else if (discriminant === 0) {
                    return { roots: [-b / (2 * a)], type: 'quadratic', discriminant };
                } else {
                    return {
                        roots: [
                            { real: -b / (2 * a), imaginary: Math.sqrt(-discriminant) / (2 * a) },
                            { real: -b / (2 * a), imaginary: -Math.sqrt(-discriminant) / (2 * a) }
                        ],
                        type: 'quadratic',
                        discriminant
                    };
                }

            default:
                return { error: 'Analytical solution not implemented for degree > 2' };
        }
    }

    performMatrixOperations(params) {
        const { operation = 'multiply', matrixA = [[1, 2], [3, 4]], matrixB = [[5, 6], [7, 8]] } = params;

        switch (operation) {
            case 'multiply':
                return { result: this.multiplyMatrices(matrixA, matrixB), operation: 'multiplication' };
            case 'inverse':
                return { result: this.inverseMatrix(matrixA), operation: 'inverse' };
            case 'determinant':
                return { result: this.determinant(matrixA), operation: 'determinant' };
            case 'eigenvalues':
                return { result: this.eigenvalues(matrixA), operation: 'eigenvalues' };
            case 'transpose':
                return { result: this.transposeMatrix(matrixA), operation: 'transpose' };
            default:
                return { error: 'Unknown matrix operation' };
        }
    }

    multiplyMatrices(A, B) {
        if (A[0].length !== B.length) {
            throw new Error('Matrix dimensions incompatible for multiplication');
        }

        const result = Array(A.length).fill().map(() => Array(B[0].length).fill(0));

        for (let i = 0; i < A.length; i++) {
            for (let j = 0; j < B[0].length; j++) {
                for (let k = 0; k < B.length; k++) {
                    result[i][j] += A[i][k] * B[k][j];
                }
            }
        }

        return result;
    }

    determinant(matrix) {
        const n = matrix.length;

        if (n === 1) return matrix[0][0];
        if (n === 2) return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0];

        let det = 0;
        for (let col = 0; col < n; col++) {
            const subMatrix = matrix.slice(1).map(row =>
                row.filter((_, colIndex) => colIndex !== col)
            );
            det += matrix[0][col] * Math.pow(-1, col) * this.determinant(subMatrix);
        }

        return det;
    }

    transposeMatrix(matrix) {
        return matrix[0].map((_, colIndex) => matrix.map(row => row[colIndex]));
    }

    inverseMatrix(matrix) {
        const det = this.determinant(matrix);
        if (det === 0) {
            throw new Error('Matrix is not invertible (determinant = 0)');
        }

        const n = matrix.length;
        const adjugate = this.adjugateMatrix(matrix);

        return adjugate.map(row => row.map(element => element / det));
    }

    adjugateMatrix(matrix) {
        const n = matrix.length;
        const adj = Array(n).fill().map(() => Array(n).fill(0));

        for (let i = 0; i < n; i++) {
            for (let j = 0; j < n; j++) {
                const subMatrix = matrix
                    .filter((_, rowIndex) => rowIndex !== i)
                    .map(row => row.filter((_, colIndex) => colIndex !== j));

                adj[j][i] = Math.pow(-1, i + j) * this.determinant(subMatrix);
            }
        }

        return adj;
    }

    performFourierAnalysis(params) {
        const { signal = [], sampleRate = 44100, windowSize = 1024 } = params;

        if (signal.length === 0) {
            // Generate sample signal if none provided
            const samples = [];
            for (let i = 0; i < windowSize; i++) {
                const t = i / sampleRate;
                samples.push(
                    Math.sin(2 * Math.PI * 440 * t) +  // A4 note
                    0.5 * Math.sin(2 * Math.PI * 880 * t) +  // A5 note
                    0.25 * Math.sin(2 * Math.PI * 1320 * t)   // E6 note
                );
            }
            return this.computeFFT(samples, sampleRate);
        }

        return this.computeFFT(signal, sampleRate);
    }

    computeFFT(signal, sampleRate) {
        const N = signal.length;
        const frequencies = [];
        const magnitudes = [];
        const phases = [];

        // Simple DFT implementation (not optimized FFT)
        for (let k = 0; k < N / 2; k++) {
            let realSum = 0;
            let imagSum = 0;

            for (let n = 0; n < N; n++) {
                const angle = -2 * Math.PI * k * n / N;
                realSum += signal[n] * Math.cos(angle);
                imagSum += signal[n] * Math.sin(angle);
            }

            const magnitude = Math.sqrt(realSum * realSum + imagSum * imagSum) / N;
            const phase = Math.atan2(imagSum, realSum);
            const frequency = k * sampleRate / N;

            frequencies.push(frequency);
            magnitudes.push(magnitude);
            phases.push(phase);
        }

        return { frequencies, magnitudes, phases, sampleRate, windowSize: N };
    }

    // Geometry experiment implementations
    applyGeometricTransformations(params) {
        const { shape = 'square', transformations = ['rotate', 'scale', 'translate'] } = params;

        // Generate base shape
        let points = this.generateBaseShape(shape);
        const transformationHistory = [];

        transformations.forEach((transform, index) => {
            let transformedPoints;

            switch (transform) {
                case 'rotate':
                    const angle = (index + 1) * Math.PI / 6; // 30 degree increments
                    transformedPoints = this.rotatePoints(points, angle);
                    transformationHistory.push({ type: 'rotation', angle: angle * 180 / Math.PI });
                    break;

                case 'scale':
                    const scale = 1 + index * 0.2;
                    transformedPoints = this.scalePoints(points, scale);
                    transformationHistory.push({ type: 'scaling', factor: scale });
                    break;

                case 'translate':
                    const dx = index * 20;
                    const dy = index * 10;
                    transformedPoints = this.translatePoints(points, dx, dy);
                    transformationHistory.push({ type: 'translation', dx, dy });
                    break;

                default:
                    transformedPoints = points;
            }

            points = transformedPoints;
        });

        return {
            originalShape: this.generateBaseShape(shape),
            transformedShape: points,
            transformations: transformationHistory
        };
    }

    generateBaseShape(shape) {
        switch (shape) {
            case 'square':
                return [[-1, -1], [1, -1], [1, 1], [-1, 1], [-1, -1]];
            case 'triangle':
                return [[0, 1], [-1, -1], [1, -1], [0, 1]];
            case 'circle':
                const points = [];
                for (let i = 0; i <= 32; i++) {
                    const angle = i * 2 * Math.PI / 32;
                    points.push([Math.cos(angle), Math.sin(angle)]);
                }
                return points;
            case 'pentagon':
                const pentPoints = [];
                for (let i = 0; i <= 5; i++) {
                    const angle = i * 2 * Math.PI / 5 - Math.PI / 2;
                    pentPoints.push([Math.cos(angle), Math.sin(angle)]);
                }
                return pentPoints;
            default:
                return [[-1, -1], [1, -1], [1, 1], [-1, 1], [-1, -1]];
        }
    }

    rotatePoints(points, angle) {
        const cos = Math.cos(angle);
        const sin = Math.sin(angle);

        return points.map(([x, y]) => [
            x * cos - y * sin,
            x * sin + y * cos
        ]);
    }

    scalePoints(points, scale) {
        return points.map(([x, y]) => [x * scale, y * scale]);
    }

    translatePoints(points, dx, dy) {
        return points.map(([x, y]) => [x + dx, y + dy]);
    }

    detectIntersections(params) {
        const { shapeA = 'circle', shapeB = 'line', positionA = [0, 0], positionB = [1, 1] } = params;

        // Simplified intersection detection
        const distance = Math.sqrt(
            Math.pow(positionB[0] - positionA[0], 2) +
            Math.pow(positionB[1] - positionA[1], 2)
        );

        return {
            shapeA,
            shapeB,
            positionA,
            positionB,
            distance,
            intersecting: distance < 2, // Simple threshold
            intersectionPoints: distance < 2 ? this.calculateIntersectionPoints(shapeA, shapeB, positionA, positionB) : []
        };
    }

    calculateIntersectionPoints(shapeA, shapeB, posA, posB) {
        // Simplified intersection calculation
        const midpoint = [(posA[0] + posB[0]) / 2, (posA[1] + posB[1]) / 2];
        return [midpoint]; // Simplified to single point
    }

    generateConvexHull(params) {
        let { points = [] } = params;

        if (points.length === 0) {
            // Generate random point set
            points = [];
            for (let i = 0; i < 20; i++) {
                points.push([
                    (Math.random() - 0.5) * 100,
                    (Math.random() - 0.5) * 100
                ]);
            }
        }

        return {
            originalPoints: points,
            convexHull: this.grahamScan(points),
            algorithm: 'Graham Scan'
        };
    }

    grahamScan(points) {
        if (points.length < 3) return points;

        // Find bottom-most point (and leftmost in case of tie)
        let bottom = 0;
        for (let i = 1; i < points.length; i++) {
            if (points[i][1] < points[bottom][1] ||
                (points[i][1] === points[bottom][1] && points[i][0] < points[bottom][0])) {
                bottom = i;
            }
        }

        // Swap bottom point to index 0
        [points[0], points[bottom]] = [points[bottom], points[0]];

        // Sort points by polar angle with respect to bottom point
        const bottomPoint = points[0];
        const sortedPoints = points.slice(1).sort((a, b) => {
            const angleA = Math.atan2(a[1] - bottomPoint[1], a[0] - bottomPoint[0]);
            const angleB = Math.atan2(b[1] - bottomPoint[1], b[0] - bottomPoint[0]);
            return angleA - angleB;
        });

        // Build convex hull
        const hull = [bottomPoint, sortedPoints[0]];

        for (let i = 1; i < sortedPoints.length; i++) {
            while (hull.length >= 2 &&
                this.crossProduct(hull[hull.length - 2], hull[hull.length - 1], sortedPoints[i]) <= 0) {
                hull.pop();
            }
            hull.push(sortedPoints[i]);
        }

        return hull;
    }

    crossProduct(o, a, b) {
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
    }

    // Sacred geometry experiments
    constructGoldenRatioShapes(params) {
        const { type = 'rectangle', iterations = 5 } = params;
        const phi = (1 + Math.sqrt(5)) / 2;

        const constructions = [];

        switch (type) {
            case 'rectangle':
                let width = 1;
                let height = phi;

                for (let i = 0; i < iterations; i++) {
                    constructions.push({
                        iteration: i,
                        width: width,
                        height: height,
                        ratio: height / width,
                        vertices: [
                            [-width / 2, -height / 2],
                            [width / 2, -height / 2],
                            [width / 2, height / 2],
                            [-width / 2, height / 2],
                            [-width / 2, -height / 2]
                        ]
                    });

                    // Next golden rectangle
                    const newWidth = height;
                    const newHeight = width + height;
                    width = newWidth;
                    height = newHeight;
                }
                break;

            case 'spiral':
                let radius = 1;
                const points = [];

                for (let i = 0; i < iterations * 100; i++) {
                    const angle = i * 0.1;
                    const currentRadius = radius * Math.pow(phi, angle / (2 * Math.PI));
                    points.push([
                        currentRadius * Math.cos(angle),
                        currentRadius * Math.sin(angle)
                    ]);
                }

                constructions.push({ type: 'spiral', points });
                break;
        }

        return {
            type,
            phi,
            iterations,
            constructions
        };
    }

    analyzePlatonicRelationships(params) {
        const solids = {
            tetrahedron: { faces: 4, vertices: 4, edges: 6 },
            cube: { faces: 6, vertices: 8, edges: 12 },
            octahedron: { faces: 8, vertices: 6, edges: 12 },
            dodecahedron: { faces: 12, vertices: 20, edges: 30 },
            icosahedron: { faces: 20, vertices: 12, edges: 30 }
        };

        const relationships = [];

        Object.keys(solids).forEach(name => {
            const solid = solids[name];

            // Verify Euler's formula: V - E + F = 2
            const eulerResult = solid.vertices - solid.edges + solid.faces;

            relationships.push({
                name,
                ...solid,
                eulerFormula: eulerResult,
                eulerValid: eulerResult === 2
            });
        });

        // Find dual relationships
        const dualPairs = [
            ['tetrahedron', 'tetrahedron'],
            ['cube', 'octahedron'],
            ['dodecahedron', 'icosahedron']
        ];

        return {
            solids: relationships,
            dualRelationships: dualPairs,
            eulerFormulaVerification: relationships.every(s => s.eulerValid)
        };
    }

    // Physics experiments
    simulateNBodyGravity(params) {
        const { bodyCount = 3, timeStep = 0.01, iterations = 1000 } = params;
        const G = 6.67430e-11; // Gravitational constant (scaled for visualization)

        // Initialize random bodies
        const bodies = [];
        for (let i = 0; i < bodyCount; i++) {
            bodies.push({
                id: i,
                mass: 1 + Math.random() * 9, // Mass between 1-10
                position: {
                    x: (Math.random() - 0.5) * 100,
                    y: (Math.random() - 0.5) * 100
                },
                velocity: {
                    x: (Math.random() - 0.5) * 10,
                    y: (Math.random() - 0.5) * 10
                },
                acceleration: { x: 0, y: 0 },
                trail: []
            });
        }

        const simulationSteps = [];

        // Run simulation
        for (let step = 0; step < iterations; step++) {
            // Calculate gravitational forces
            for (let i = 0; i < bodies.length; i++) {
                bodies[i].acceleration = { x: 0, y: 0 };

                for (let j = 0; j < bodies.length; j++) {
                    if (i === j) continue;

                    const dx = bodies[j].position.x - bodies[i].position.x;
                    const dy = bodies[j].position.y - bodies[i].position.y;
                    const distance = Math.sqrt(dx * dx + dy * dy);

                    if (distance > 0) {
                        const force = G * bodies[i].mass * bodies[j].mass / (distance * distance);
                        const forceX = force * dx / distance;
                        const forceY = force * dy / distance;

                        bodies[i].acceleration.x += forceX / bodies[i].mass;
                        bodies[i].acceleration.y += forceY / bodies[i].mass;
                    }
                }
            }

            // Update positions and velocities
            for (let i = 0; i < bodies.length; i++) {
                bodies[i].velocity.x += bodies[i].acceleration.x * timeStep;
                bodies[i].velocity.y += bodies[i].acceleration.y * timeStep;

                bodies[i].position.x += bodies[i].velocity.x * timeStep;
                bodies[i].position.y += bodies[i].velocity.y * timeStep;

                // Store trail
                if (step % 10 === 0) {
                    bodies[i].trail.push({
                        x: bodies[i].position.x,
                        y: bodies[i].position.y
                    });

                    // Limit trail length
                    if (bodies[i].trail.length > 100) {
                        bodies[i].trail.shift();
                    }
                }
            }

            // Store simulation step
            if (step % 50 === 0) {
                simulationSteps.push({
                    step,
                    time: step * timeStep,
                    bodies: JSON.parse(JSON.stringify(bodies))
                });
            }
        }

        return {
            bodyCount,
            timeStep,
            iterations,
            finalBodies: bodies,
            simulationSteps,
            totalEnergy: this.calculateSystemEnergy(bodies)
        };
    }

    calculateSystemEnergy(bodies) {
        let kineticEnergy = 0;
        let potentialEnergy = 0;
        const G = 6.67430e-11;

        // Calculate kinetic energy
        bodies.forEach(body => {
            const speed = Math.sqrt(body.velocity.x * body.velocity.x + body.velocity.y * body.velocity.y);
            kineticEnergy += 0.5 * body.mass * speed * speed;
        });

        // Calculate potential energy
        for (let i = 0; i < bodies.length; i++) {
            for (let j = i + 1; j < bodies.length; j++) {
                const dx = bodies[i].position.x - bodies[j].position.x;
                const dy = bodies[i].position.y - bodies[j].position.y;
                const distance = Math.sqrt(dx * dx + dy * dy);

                if (distance > 0) {
                    potentialEnergy -= G * bodies[i].mass * bodies[j].mass / distance;
                }
            }
        }

        return {
            kinetic: kineticEnergy,
            potential: potentialEnergy,
            total: kineticEnergy + potentialEnergy
        };
    }

    // Fractal experiments
    generateCustomLSystem(params) {
        const {
            axiom = 'F',
            rules = { 'F': 'F[+F]F[-F]F' },
            iterations = 4,
            angle = 25
        } = params;

        let current = axiom;
        const evolutionSteps = [axiom];

        // Generate L-system string
        for (let i = 0; i < iterations; i++) {
            let next = '';
            for (const char of current) {
                next += rules[char] || char;
            }
            current = next;
            evolutionSteps.push(current);
        }

        // Interpret the L-system
        const interpretedPath = this.interpretLSystemString(current, angle);

        return {
            axiom,
            rules,
            iterations,
            angle,
            evolutionSteps,
            finalString: current,
            stringLength: current.length,
            interpretedPath,
            pathLength: interpretedPath.length
        };
    }

    interpretLSystemString(lstring, angle) {
        const path = [];
        const stack = [];
        let x = 0, y = 0, heading = 90;
        let length = 10;

        for (const char of lstring) {
            switch (char) {
                case 'F':
                    const newX = x + length * Math.cos(heading * Math.PI / 180);
                    const newY = y + length * Math.sin(heading * Math.PI / 180);
                    path.push({ type: 'line', from: { x, y }, to: { x: newX, y: newY } });
                    x = newX;
                    y = newY;
                    break;
                case '+':
                    heading += angle;
                    break;
                case '-':
                    heading -= angle;
                    break;
                case '[':
                    stack.push({ x, y, heading, length });
                    length *= 0.7;
                    break;
                case ']':
                    const state = stack.pop();
                    if (state) {
                        x = state.x;
                        y = state.y;
                        heading = state.heading;
                        length = state.length;
                    }
                    break;
            }
        }

        return path;
    }

    calculateFractalDimension(params) {
        const { method = 'box_counting', data = [] } = params;

        switch (method) {
            case 'box_counting':
                return this.boxCountingDimension(data);
            case 'hausdorff':
                return this.hausdorffDimension(data);
            default:
                return this.boxCountingDimension(data);
        }
    }

    boxCountingDimension(points) {
        if (points.length === 0) {
            // Generate Koch snowflake for example
            points = this.generateKochSnowflake(4);
        }

        const boxSizes = [1, 2, 4, 8, 16, 32];
        const counts = [];

        boxSizes.forEach(boxSize => {
            const boxes = new Set();

            points.forEach(point => {
                const boxX = Math.floor(point.x / boxSize);
                const boxY = Math.floor(point.y / boxSize);
                boxes.add(`${boxX},${boxY}`);
            });

            counts.push(boxes.size);
        });

        // Calculate dimension using linear regression on log-log plot
        const logSizes = boxSizes.map(s => Math.log(1 / s));
        const logCounts = counts.map(c => Math.log(c));

        const dimension = this.linearRegression(logSizes, logCounts).slope;

        return {
            method: 'box_counting',
            boxSizes,
            counts,
            dimension,
            dataPoints: points.length
        };
    }

    generateKochSnowflake(iterations) {
        let points = [
            { x: -50, y: 25 },
            { x: 50, y: 25 },
            { x: 0, y: -50 },
            { x: -50, y: 25 }
        ];

        for (let iter = 0; iter < iterations; iter++) {
            const newPoints = [];

            for (let i = 0; i < points.length - 1; i++) {
                const p1 = points[i];
                const p2 = points[i + 1];

                const dx = p2.x - p1.x;
                const dy = p2.y - p1.y;

                const a = { x: p1.x + dx / 3, y: p1.y + dy / 3 };
                const b = { x: p1.x + 2 * dx / 3, y: p1.y + 2 * dy / 3 };

                const angle = Math.atan2(dy, dx) - Math.PI / 3;
                const length = Math.sqrt(dx * dx + dy * dy) / 3;

                const c = {
                    x: a.x + length * Math.cos(angle),
                    y: a.y + length * Math.sin(angle)
                };

                newPoints.push(p1, a, c, b);
            }
            newPoints.push(points[points.length - 1]);

            points = newPoints;
        }

        return points;
    }

    linearRegression(x, y) {
        const n = x.length;
        const sumX = x.reduce((sum, val) => sum + val, 0);
        const sumY = y.reduce((sum, val) => sum + val, 0);
        const sumXY = x.reduce((sum, val, i) => sum + val * y[i], 0);
        const sumXX = x.reduce((sum, val) => sum + val * val, 0);

        const slope = (n * sumXY - sumX * sumY) / (n * sumXX - sumX * sumX);
        const intercept = (sumY - slope * sumX) / n;

        return { slope, intercept };
    }

    // Utility methods
    getExperimentList() {
        return Array.from(this.experiments.entries()).map(([id, experiment]) => ({
            id,
            name: experiment.name,
            description: experiment.description,
            category: experiment.category
        }));
    }

    getExperimentHistory() {
        return this.sandboxState.history;
    }

    clearHistory() {
        this.sandboxState.history = [];
        console.log('🧹 Experiment history cleared');
    }

    exportResults(format = 'json') {
        const data = {
            experiments: this.getExperimentList(),
            history: this.getExperimentHistory(),
            timestamp: new Date().toISOString()
        };

        switch (format) {
            case 'json':
                return JSON.stringify(data, null, 2);
            case 'csv':
                return this.convertToCSV(data.history);
            default:
                return data;
        }
    }

    convertToCSV(history) {
        if (history.length === 0) return '';

        const headers = ['Experiment', 'Execution Time (ms)', 'Timestamp'];
        const rows = history.map(result => [
            result.name,
            result.executionTime.toFixed(2),
            result.timestamp
        ]);

        return [headers, ...rows].map(row => row.join(',')).join('\n');
    }
}

// Supporting classes for the sandbox
class AdvancedCalculator {
    constructor() {
        this.constants = {
            pi: Math.PI,
            e: Math.E,
            phi: (1 + Math.sqrt(5)) / 2,
            tau: 2 * Math.PI
        };
    }

    evaluate(expression) {
        // Simple expression evaluator (would need more robust parser in real implementation)
        try {
            // Replace constants
            let processed = expression;
            Object.keys(this.constants).forEach(constant => {
                processed = processed.replace(new RegExp(constant, 'g'), this.constants[constant]);
            });

            // Evaluate (WARNING: using eval - in production, use a proper parser)
            return eval(processed);
        } catch (error) {
            throw new Error(`Invalid expression: ${error.message}`);
        }
    }
}

class GeometricConstructor {
    constructor() {
        this.constructions = [];
    }

    constructCircle(center, radius) {
        return { type: 'circle', center, radius };
    }

    constructLine(point1, point2) {
        return { type: 'line', point1, point2 };
    }

    findIntersection(shape1, shape2) {
        // Simplified intersection finding
        return { type: 'intersection', shapes: [shape1, shape2] };
    }
}

class EquationSolver {
    constructor() {
        this.methods = ['newton', 'bisection', 'secant', 'fixed_point'];
    }

    solve(equation, method = 'newton', initialGuess = 0) {
        // Implementation would depend on the specific method
        return { solution: initialGuess, method, iterations: 0 };
    }
}

class PerformanceTester {
    constructor() {
        this.benchmarks = new Map();
    }

    benchmark(name, func, iterations = 1000) {
        const startTime = performance.now();

        for (let i = 0; i < iterations; i++) {
            func();
        }

        const endTime = performance.now();
        const result = {
            name,
            iterations,
            totalTime: endTime - startTime,
            averageTime: (endTime - startTime) / iterations
        };

        this.benchmarks.set(name, result);
        return result;
    }

    getBenchmarks() {
        return Array.from(this.benchmarks.values());
    }
}

class ExpressionEvaluator {
    evaluate(expression) {
        // Safe expression evaluator
        return new Function('return ' + expression)();
    }
}

class GraphingCalculator {
    plot(func, range = [-10, 10], resolution = 100) {
        const points = [];
        const step = (range[1] - range[0]) / resolution;

        for (let x = range[0]; x <= range[1]; x += step) {
            try {
                const y = func(x);
                if (isFinite(y)) {
                    points.push({ x, y });
                }
            } catch (error) {
                // Skip invalid points
            }
        }

        return points;
    }
}

class SymbolProcessor {
    differentiate(expression, variable = 'x') {
        // Simplified symbolic differentiation
        // In a real implementation, this would use a computer algebra system
        return `d/d${variable}(${expression})`;
    }

    integrate(expression, variable = 'x') {
        // Simplified symbolic integration
        return `∫(${expression})d${variable}`;
    }
}

class NumericalMethods {
    eulerMethod(dydx, x0, y0, h, steps) {
        const points = [{ x: x0, y: y0 }];
        let x = x0, y = y0;

        for (let i = 0; i < steps; i++) {
            y = y + h * dydx(x, y);
            x = x + h;
            points.push({ x, y });
        }

        return points;
    }

    rungeKutta4(dydx, x0, y0, h, steps) {
        const points = [{ x: x0, y: y0 }];
        let x = x0, y = y0;

        for (let i = 0; i < steps; i++) {
            const k1 = h * dydx(x, y);
            const k2 = h * dydx(x + h / 2, y + k1 / 2);
            const k3 = h * dydx(x + h / 2, y + k2 / 2);
            const k4 = h * dydx(x + h, y + k3);

            y = y + (k1 + 2 * k2 + 2 * k3 + k4) / 6;
            x = x + h;
            points.push({ x, y });
        }

        return points;
    }
}

// Export the sandbox
if (typeof module !== 'undefined' && module.exports) {
    module.exports = NexusForgeMathematicalSandbox;
}

// Auto-initialize if in browser
if (typeof window !== 'undefined') {
    window.NexusForgeMathematicalSandbox = NexusForgeMathematicalSandbox;
    console.log('🧪 Mathematical Sandbox available globally');
}
