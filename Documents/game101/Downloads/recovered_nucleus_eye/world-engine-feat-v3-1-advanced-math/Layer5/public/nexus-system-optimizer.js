// NEXUS Forge System Optimizer v5.0.0
// Advanced performance optimization and system monitoring

class NexusSystemOptimizer {
    constructor() {
        this.performanceCache = new Map();
        this.systemMetrics = new Map();
        this.optimizationRules = new Map();
        this.isOptimizing = false;
        this.debugMode = true;

        this.initializeOptimizer();
    }

    initializeOptimizer() {
        console.log('🚀 NEXUS System Optimizer v5.0.0 - Initializing...');

        // Register optimization rules
        this.registerOptimizationRules();

        // Start performance monitoring
        this.startPerformanceMonitoring();

        // Initialize system health checks
        this.startSystemHealthChecks();

        // Setup memory management
        this.setupMemoryManagement();

        console.log('✅ System Optimizer ready');
    }

    registerOptimizationRules() {
        // WebGL optimization rules
        this.optimizationRules.set('webgl', {
            maxTextureSize: 2048,
            maxVertices: 65536,
            enableDepthTest: true,
            enableCulling: true,
            optimizeShaders: true,
            useVAOs: true
        });

        // Mathematical computation rules
        this.optimizationRules.set('math', {
            precisionLevel: 8,
            cacheResults: true,
            useFastMath: true,
            vectorizeOperations: true,
            parallelCompute: true
        });

        // AI training optimization rules
        this.optimizationRules.set('ai', {
            batchSize: 32,
            learningRate: 0.001,
            useGradientClipping: true,
            enableEarlyStopping: true,
            cacheEmbeddings: true
        });

        // UI/UX performance rules
        this.optimizationRules.set('ui', {
            maxFPS: 60,
            reduceMotion: false,
            useRequestAnimationFrame: true,
            debounceInputs: true,
            virtualizeList: true
        });
    }

    // Performance monitoring system
    startPerformanceMonitoring() {
        let frameCount = 0;
        let lastTime = performance.now();

        const monitorLoop = (currentTime) => {
            frameCount++;

            if (currentTime - lastTime >= 1000) {
                const fps = Math.round((frameCount * 1000) / (currentTime - lastTime));
                this.updateMetric('fps', fps);

                frameCount = 0;
                lastTime = currentTime;
            }

            // Monitor memory usage
            if (performance.memory) {
                this.updateMetric('memoryUsed', performance.memory.usedJSHeapSize);
                this.updateMetric('memoryTotal', performance.memory.totalJSHeapSize);
                this.updateMetric('memoryLimit', performance.memory.jsHeapSizeLimit);
            }

            // Monitor CPU usage (approximate)
            const cpuUsage = this.estimateCPUUsage(currentTime);
            this.updateMetric('cpuUsage', cpuUsage);

            requestAnimationFrame(monitorLoop);
        };

        requestAnimationFrame(monitorLoop);
    }

    estimateCPUUsage(currentTime) {
        const interval = 100;
        const startTime = performance.now();

        // Busy wait for a short interval to measure CPU responsiveness
        while (performance.now() - startTime < 1) {
            // CPU-intensive operation
            Math.random() * Math.random();
        }

        const endTime = performance.now();
        const actualTime = endTime - startTime;

        // Estimate CPU usage based on how long the operation took
        const usage = Math.min(100, Math.max(0, (actualTime - 1) * 20));
        return Math.round(usage);
    }

    updateMetric(name, value) {
        if (!this.systemMetrics.has(name)) {
            this.systemMetrics.set(name, []);
        }

        const metrics = this.systemMetrics.get(name);
        metrics.push({ value, timestamp: Date.now() });

        // Keep only last 100 measurements
        if (metrics.length > 100) {
            metrics.shift();
        }
    }

    getMetric(name) {
        const metrics = this.systemMetrics.get(name);
        if (!metrics || metrics.length === 0) return null;

        return metrics[metrics.length - 1].value;
    }

    getAverageMetric(name, duration = 10000) {
        const metrics = this.systemMetrics.get(name);
        if (!metrics || metrics.length === 0) return null;

        const cutoffTime = Date.now() - duration;
        const recentMetrics = metrics.filter(m => m.timestamp > cutoffTime);

        if (recentMetrics.length === 0) return null;

        const sum = recentMetrics.reduce((acc, m) => acc + m.value, 0);
        return sum / recentMetrics.length;
    }

    // System health monitoring
    startSystemHealthChecks() {
        setInterval(() => {
            this.checkSystemHealth();
        }, 5000);
    }

    checkSystemHealth() {
        const health = {
            fps: this.getMetric('fps'),
            memory: this.getMemoryUsagePercentage(),
            cpu: this.getMetric('cpuUsage'),
            timestamp: Date.now()
        };

        // Check for performance issues
        if (health.fps < 30) {
            this.handlePerformanceIssue('low_fps', health.fps);
        }

        if (health.memory > 80) {
            this.handlePerformanceIssue('high_memory', health.memory);
        }

        if (health.cpu > 90) {
            this.handlePerformanceIssue('high_cpu', health.cpu);
        }

        // Update global health status
        this.updateSystemHealth(health);
    }

    getMemoryUsagePercentage() {
        const used = this.getMetric('memoryUsed');
        const limit = this.getMetric('memoryLimit');

        if (!used || !limit) return 0;

        return Math.round((used / limit) * 100);
    }

    handlePerformanceIssue(type, value) {
        console.warn(`⚠️ Performance issue detected: ${type} = ${value}`);

        switch (type) {
            case 'low_fps':
                this.optimizeFPS();
                break;
            case 'high_memory':
                this.optimizeMemory();
                break;
            case 'high_cpu':
                this.optimizeCPU();
                break;
        }
    }

    // Performance optimization methods
    optimizeFPS() {
        console.log('🔧 Optimizing FPS performance...');

        // Reduce render quality temporarily
        this.broadcastOptimization({
            type: 'reduce_quality',
            settings: {
                renderScale: 0.8,
                shadowQuality: 'low',
                particleCount: 0.5
            }
        });
    }

    optimizeMemory() {
        console.log('🔧 Optimizing memory usage...');

        // Clear caches and unused resources
        this.performanceCache.clear();

        // Suggest garbage collection
        if (window.gc) {
            window.gc();
        }

        // Reduce texture resolution
        this.broadcastOptimization({
            type: 'reduce_memory',
            settings: {
                textureResolution: 0.5,
                cacheSize: 'small',
                preloadAssets: false
            }
        });
    }

    optimizeCPU() {
        console.log('🔧 Optimizing CPU usage...');

        // Reduce computation frequency
        this.broadcastOptimization({
            type: 'reduce_cpu',
            settings: {
                updateFrequency: 0.5,
                mathPrecision: 'fast',
                enableLOD: true
            }
        });
    }

    broadcastOptimization(optimization) {
        // Broadcast to all NEXUS systems
        window.postMessage({
            type: 'nexus_optimization',
            data: optimization
        }, '*');

        // Also try to communicate with child windows
        if (window.nexusWindows) {
            window.nexusWindows.forEach(win => {
                if (win && !win.closed) {
                    try {
                        win.postMessage({
                            type: 'nexus_optimization',
                            data: optimization
                        }, '*');
                    } catch (e) {
                        // Cross-origin error, ignore
                    }
                }
            });
        }
    }

    // Memory management
    setupMemoryManagement() {
        // Periodic cleanup
        setInterval(() => {
            this.performMemoryCleanup();
        }, 30000);

        // Listen for memory pressure events
        if ('memory' in navigator) {
            navigator.memory.addEventListener('memorychange', (event) => {
                this.handleMemoryPressure(event.level);
            });
        }
    }

    performMemoryCleanup() {
        // Clear old performance data
        for (const [key, metrics] of this.systemMetrics.entries()) {
            if (metrics.length > 50) {
                metrics.splice(0, metrics.length - 50);
            }
        }

        // Clear old cache entries
        if (this.performanceCache.size > 1000) {
            const entries = Array.from(this.performanceCache.entries());
            entries.slice(0, 500).forEach(([key]) => {
                this.performanceCache.delete(key);
            });
        }

        if (this.debugMode) {
            console.log('🧹 Memory cleanup performed');
        }
    }

    handleMemoryPressure(level) {
        console.warn(`⚠️ Memory pressure detected: ${level}`);

        switch (level) {
            case 'critical':
                this.performanceCache.clear();
                this.broadcastOptimization({
                    type: 'emergency_memory_cleanup',
                    settings: {
                        clearAll: true,
                        minimalMode: true
                    }
                });
                break;
            case 'moderate':
                this.optimizeMemory();
                break;
        }
    }

    // WebGL optimization utilities
    optimizeWebGL(gl) {
        if (!gl) return null;

        const rules = this.optimizationRules.get('webgl');

        // Enable depth testing
        if (rules.enableDepthTest) {
            gl.enable(gl.DEPTH_TEST);
            gl.depthFunc(gl.LEQUAL);
        }

        // Enable face culling
        if (rules.enableCulling) {
            gl.enable(gl.CULL_FACE);
            gl.cullFace(gl.BACK);
        }

        // Optimize viewport
        const canvas = gl.canvas;
        const devicePixelRatio = window.devicePixelRatio || 1;
        const displayWidth = Math.floor(canvas.clientWidth * devicePixelRatio);
        const displayHeight = Math.floor(canvas.clientHeight * devicePixelRatio);

        if (canvas.width !== displayWidth || canvas.height !== displayHeight) {
            canvas.width = displayWidth;
            canvas.height = displayHeight;
            gl.viewport(0, 0, displayWidth, displayHeight);
        }

        return {
            optimized: true,
            width: displayWidth,
            height: displayHeight,
            pixelRatio: devicePixelRatio
        };
    }

    // Mathematical optimization utilities
    createOptimizedMathFunctions() {
        const rules = this.optimizationRules.get('math');
        const functions = {};

        // Fast square root using bit manipulation
        functions.fastSqrt = rules.useFastMath ?
            (x) => {
                if (x < 0) return NaN;
                if (x === 0) return 0;

                let result = x;
                let temp = 0;

                // Newton-Raphson method with good initial guess
                while (result !== temp) {
                    temp = result;
                    result = (result + x / result) * 0.5;
                }

                return result;
            } : Math.sqrt;

        // Fast inverse square root (Quake algorithm inspired)
        functions.fastInvSqrt = rules.useFastMath ?
            (x) => {
                const halfx = x * 0.5;
                let y = x;
                const view = new DataView(new ArrayBuffer(4));
                view.setFloat32(0, y);
                let i = view.getUint32(0);
                i = 0x5f3759df - (i >> 1);
                view.setUint32(0, i);
                y = view.getFloat32(0);
                y = y * (1.5 - (halfx * y * y));
                return y;
            } : (x) => 1 / Math.sqrt(x);

        // Vectorized operations for arrays
        functions.vectorAdd = (a, b) => {
            const result = new Float32Array(a.length);
            for (let i = 0; i < a.length; i++) {
                result[i] = a[i] + b[i];
            }
            return result;
        };

        functions.vectorMultiply = (a, scalar) => {
            const result = new Float32Array(a.length);
            for (let i = 0; i < a.length; i++) {
                result[i] = a[i] * scalar;
            }
            return result;
        };

        functions.dotProduct = (a, b) => {
            let sum = 0;
            for (let i = 0; i < a.length; i++) {
                sum += a[i] * b[i];
            }
            return sum;
        };

        return functions;
    }

    // AI optimization utilities
    optimizeAITraining(config = {}) {
        const rules = this.optimizationRules.get('ai');

        return {
            batchSize: config.batchSize || rules.batchSize,
            learningRate: config.learningRate || rules.learningRate,
            useGradientClipping: config.useGradientClipping ?? rules.useGradientClipping,
            enableEarlyStopping: config.enableEarlyStopping ?? rules.enableEarlyStopping,
            optimizedForPerformance: true,

            // Optimized training functions
            processBatch: (data) => {
                // Process in chunks for better memory usage
                const chunkSize = rules.batchSize;
                const results = [];

                for (let i = 0; i < data.length; i += chunkSize) {
                    const chunk = data.slice(i, i + chunkSize);
                    results.push(this.processTrainingChunk(chunk));
                }

                return results.flat();
            },

            adaptiveLearningRate: (epoch, loss) => {
                // Adaptive learning rate based on performance
                if (loss > this.lastLoss) {
                    return rules.learningRate * 0.9;
                } else {
                    return Math.min(rules.learningRate * 1.1, 0.01);
                }
            }
        };
    }

    processTrainingChunk(chunk) {
        // Optimized chunk processing
        return chunk.map(item => ({
            ...item,
            processed: true,
            timestamp: Date.now()
        }));
    }

    // System diagnostics
    generateDiagnosticsReport() {
        const report = {
            timestamp: new Date().toISOString(),
            system: {
                userAgent: navigator.userAgent,
                platform: navigator.platform,
                language: navigator.language,
                hardwareConcurrency: navigator.hardwareConcurrency,
                deviceMemory: navigator.deviceMemory || 'unknown'
            },
            performance: {
                fps: this.getAverageMetric('fps'),
                memory: {
                    used: this.getMetric('memoryUsed'),
                    total: this.getMetric('memoryTotal'),
                    limit: this.getMetric('memoryLimit'),
                    percentage: this.getMemoryUsagePercentage()
                },
                cpu: this.getAverageMetric('cpuUsage')
            },
            optimization: {
                rules: Object.fromEntries(this.optimizationRules),
                cacheSize: this.performanceCache.size,
                metricsCount: this.systemMetrics.size
            },
            health: this.calculateSystemHealth()
        };

        return report;
    }

    calculateSystemHealth() {
        const fps = this.getMetric('fps') || 0;
        const memory = this.getMemoryUsagePercentage();
        const cpu = this.getMetric('cpuUsage') || 0;

        let score = 100;

        // FPS penalties
        if (fps < 30) score -= 30;
        else if (fps < 45) score -= 15;
        else if (fps < 55) score -= 5;

        // Memory penalties
        if (memory > 90) score -= 40;
        else if (memory > 80) score -= 25;
        else if (memory > 70) score -= 10;

        // CPU penalties
        if (cpu > 95) score -= 35;
        else if (cpu > 85) score -= 20;
        else if (cpu > 75) score -= 10;

        return Math.max(0, Math.min(100, score));
    }

    updateSystemHealth(health) {
        // Broadcast health update to all systems
        window.dispatchEvent(new CustomEvent('nexusHealthUpdate', {
            detail: health
        }));
    }

    // Public API methods
    getSystemStatus() {
        return {
            isOptimizing: this.isOptimizing,
            health: this.calculateSystemHealth(),
            metrics: {
                fps: this.getMetric('fps'),
                memory: this.getMemoryUsagePercentage(),
                cpu: this.getMetric('cpuUsage')
            }
        };
    }

    enableDebugMode(enabled = true) {
        this.debugMode = enabled;
        console.log(`🔍 Debug mode ${enabled ? 'enabled' : 'disabled'}`);
    }

    exportPerformanceData() {
        const data = {
            metrics: Object.fromEntries(this.systemMetrics),
            diagnostics: this.generateDiagnosticsReport(),
            timestamp: Date.now()
        };

        return JSON.stringify(data, null, 2);
    }

    // Event handlers for system communication
    setupEventHandlers() {
        // Listen for optimization requests
        window.addEventListener('message', (event) => {
            if (event.data.type === 'nexus_optimization_request') {
                this.handleOptimizationRequest(event.data.request);
            }
        });

        // Listen for performance reports from child systems
        window.addEventListener('nexusPerformanceReport', (event) => {
            this.handlePerformanceReport(event.detail);
        });
    }

    handleOptimizationRequest(request) {
        console.log('📥 Optimization request received:', request);

        switch (request.type) {
            case 'webgl':
                return this.optimizeWebGL(request.context);
            case 'math':
                return this.createOptimizedMathFunctions();
            case 'ai':
                return this.optimizeAITraining(request.config);
            default:
                console.warn('Unknown optimization request:', request.type);
                return null;
        }
    }

    handlePerformanceReport(report) {
        // Update metrics from external systems
        if (report.fps) this.updateMetric('external_fps', report.fps);
        if (report.memory) this.updateMetric('external_memory', report.memory);
        if (report.computations) this.updateMetric('computations_per_second', report.computations);
    }
}

// Global instance
window.nexusOptimizer = new NexusSystemOptimizer();

// Setup event handlers
window.nexusOptimizer.setupEventHandlers();

// Expose optimization utilities globally
window.NexusOptimizer = {
    getInstance: () => window.nexusOptimizer,
    optimizeWebGL: (gl) => window.nexusOptimizer.optimizeWebGL(gl),
    createMathFunctions: () => window.nexusOptimizer.createOptimizedMathFunctions(),
    optimizeAI: (config) => window.nexusOptimizer.optimizeAITraining(config),
    getStatus: () => window.nexusOptimizer.getSystemStatus(),
    generateReport: () => window.nexusOptimizer.generateDiagnosticsReport(),
    exportData: () => window.nexusOptimizer.exportPerformanceData()
};

console.log('🚀 NEXUS System Optimizer loaded and ready');
console.log('💡 Access via window.NexusOptimizer or window.nexusOptimizer');
