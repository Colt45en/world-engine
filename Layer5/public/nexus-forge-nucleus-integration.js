/**
 * NEXUS Forge Nucleus Integration Engine
 * Comprehensive connection and training system for the Nucleus AI core
 * Automatically sends all mathematical operations, discoveries, and insights to Nucleus for learning
 * Version: 3.1.0 - Advanced Nucleus Communication & Training
 */

class NexusForgeNucleusIntegration {
    constructor() {
        this.version = "3.1.0";
        this.status = "INITIALIZING";
        this.nucleusCommunicator = null;
        this.mathEngine = null;
        this.sandbox = null;

        // Learning analytics
        this.learningAnalytics = {
            totalMessages: 0,
            trainingSessions: 0,
            insightsReceived: 0,
            patterns: new Map(),
            experiments: new Map(),
            discoveries: new Map()
        };

        // Training configuration
        this.trainingConfig = {
            autoTraining: true,
            smartBatching: true,
            adaptiveLearning: true,
            insightGeneration: true,
            patternRecognition: true,
            continuousSync: true
        };

        // Real-time communication
        this.realtimeConfig = {
            batchSize: 10,
            batchTimeout: 5000,
            maxRetries: 3,
            adaptiveDelay: 1000,
            priorityQueue: []
        };

        this.messageQueue = [];
        this.trainingQueue = [];
        this.isProcessing = false;
        this.connectionHealth = {
            status: 'disconnected',
            lastPing: null,
            latency: 0,
            reliability: 100
        };

        console.log("🚀 NEXUS Nucleus Integration Engine v3.1.0 - Initializing...");
        this.initialize();
    }

    async initialize() {
        try {
            // Initialize all core systems
            await this.initializeCoreSystems();

            // Set up communication protocols
            await this.setupCommunicationProtocols();

            // Start real-time processing
            this.startRealtimeProcessing();

            // Begin health monitoring
            this.startHealthMonitoring();

            // Initialize training systems
            await this.initializeTrainingSystems();

            this.status = "ACTIVE";
            console.log("✅ Nucleus Integration Engine - FULLY OPERATIONAL");

            // Send initialization message to Nucleus
            await this.sendNucleusMessage(
                "Nucleus Integration Engine v3.1.0 fully operational and ready for advanced AI training",
                "system-initialization",
                "high"
            );

        } catch (error) {
            console.error("❌ Failed to initialize Nucleus Integration:", error);
            this.status = "ERROR";
        }
    }

    async initializeCoreSystems() {
        console.log("🔧 Initializing core systems...");

        // Initialize Nucleus Communicator
        if (typeof NexusForgeNucleusCommunicator !== 'undefined') {
            this.nucleusCommunicator = new NexusForgeNucleusCommunicator();
            console.log("✅ Nucleus Communicator initialized");
        }

        // Initialize Mathematical Engine
        if (typeof NexusForgemathematicalEngine !== 'undefined') {
            this.mathEngine = new NexusForgemathematicalEngine();
            console.log("✅ Mathematical Engine initialized");
        }

        // Initialize Sandbox
        if (typeof NexusForgeMathematicalSandbox !== 'undefined') {
            this.sandbox = new NexusForgeMathematicalSandbox();
            console.log("✅ Mathematical Sandbox initialized");
        }
    }

    async setupCommunicationProtocols() {
        console.log("📡 Setting up advanced communication protocols...");

        // Hook into mathematical engine operations
        if (this.mathEngine) {
            this.hookMathEngineOperations();
        }

        // Hook into sandbox experiments
        if (this.sandbox) {
            this.hookSandboxExperiments();
        }

        // Set up automatic pattern detection
        this.setupPatternDetection();

        // Initialize smart batching system
        this.setupSmartBatching();
    }

    hookMathEngineOperations() {
        if (!this.mathEngine) return;

        console.log("🔗 Hooking mathematical engine operations...");

        // Intercept algebra operations
        const originalAlgebra = this.mathEngine.solveEquation;
        this.mathEngine.solveEquation = async (equation, method) => {
            const result = originalAlgebra.call(this.mathEngine, equation, method);

            // Send to Nucleus for learning
            this.queueTrainingData({
                type: 'algebraic-operation',
                equation,
                method,
                result,
                timestamp: Date.now()
            });

            return result;
        };

        // Intercept geometry calculations
        const originalGeometry = this.mathEngine.calculateArea;
        this.mathEngine.calculateArea = async (shape, parameters) => {
            const result = originalGeometry.call(this.mathEngine, shape, parameters);

            this.queueTrainingData({
                type: 'geometric-calculation',
                shape,
                parameters,
                result,
                timestamp: Date.now()
            });

            return result;
        };

        // Intercept fractal generations
        const originalFractal = this.mathEngine.generateFractal;
        this.mathEngine.generateFractal = async (type, iterations, parameters) => {
            const result = originalFractal.call(this.mathEngine, type, iterations, parameters);

            this.queueTrainingData({
                type: 'fractal-generation',
                fractalType: type,
                iterations,
                parameters,
                result,
                complexity: this.calculateComplexity(result),
                timestamp: Date.now()
            });

            return result;
        };
    }

    hookSandboxExperiments() {
        if (!this.sandbox) return;

        console.log("🧪 Hooking sandbox experiments...");

        // Intercept experiment runs
        const originalRunExperiment = this.sandbox.runExperiment;
        this.sandbox.runExperiment = async (experimentId) => {
            const result = await originalRunExperiment.call(this.sandbox, experimentId);

            // Enhanced experiment tracking
            const experimentData = {
                id: experimentId,
                type: result.type || 'unknown',
                parameters: result.parameters || {},
                results: result,
                performance: this.analyzePerformance(result),
                insights: this.extractInsights(result),
                timestamp: Date.now()
            };

            // Queue for Nucleus training
            this.queueTrainingData({
                type: 'experiment-result',
                experiment: experimentData
            });

            // Request insights from Nucleus
            this.requestNucleusAnalysis(experimentData);

            return result;
        };
    }

    setupPatternDetection() {
        console.log("🔍 Setting up intelligent pattern detection...");

        this.patternDetector = {
            mathematicalPatterns: new Map(),
            sequencePatterns: new Map(),
            geometricPatterns: new Map(),
            behaviorPatterns: new Map()
        };

        // Continuous pattern analysis
        setInterval(() => {
            this.analyzePatterns();
        }, 10000);
    }

    setupSmartBatching() {
        console.log("📦 Initializing smart batching system...");

        // Process queues intelligently
        setInterval(() => {
            if (!this.isProcessing) {
                this.processBatchedOperations();
            }
        }, this.realtimeConfig.batchTimeout);
    }

    async processBatchedOperations() {
        if (this.messageQueue.length === 0 && this.trainingQueue.length === 0) return;

        this.isProcessing = true;

        try {
            // Process message queue
            if (this.messageQueue.length > 0) {
                await this.processBatchedMessages();
            }

            // Process training queue
            if (this.trainingQueue.length > 0) {
                await this.processBatchedTraining();
            }

        } catch (error) {
            console.error("❌ Batch processing failed:", error);
        }

        this.isProcessing = false;
    }

    async processBatchedMessages() {
        const batch = this.messageQueue.splice(0, this.realtimeConfig.batchSize);

        for (const message of batch) {
            try {
                await this.nucleusCommunicator.sendMessageToNucleus(
                    message.content,
                    message.type,
                    message.priority
                );

                this.learningAnalytics.totalMessages++;

            } catch (error) {
                console.error("❌ Failed to send message to Nucleus:", error);
                // Re-queue with lower priority
                message.retries = (message.retries || 0) + 1;
                if (message.retries < this.realtimeConfig.maxRetries) {
                    this.messageQueue.push({ ...message, priority: 'low' });
                }
            }
        }
    }

    async processBatchedTraining() {
        const batch = this.trainingQueue.splice(0, this.realtimeConfig.batchSize);

        // Group by type for efficient processing
        const groupedData = this.groupTrainingData(batch);

        for (const [type, data] of Object.entries(groupedData)) {
            try {
                await this.nucleusCommunicator.trainNucleus(data, type);
                this.learningAnalytics.trainingSessions++;

                // Store patterns for analysis
                this.updatePatternDatabase(type, data);

            } catch (error) {
                console.error(`❌ Failed to train Nucleus with ${type} data:`, error);
            }
        }
    }

    groupTrainingData(batch) {
        const grouped = {};

        for (const item of batch) {
            const type = item.type || 'general';
            if (!grouped[type]) {
                grouped[type] = [];
            }
            grouped[type].push(item);
        }

        return grouped;
    }

    updatePatternDatabase(type, data) {
        const patterns = this.patternDetector.mathematicalPatterns;

        if (!patterns.has(type)) {
            patterns.set(type, {
                count: 0,
                samples: [],
                insights: [],
                lastUpdate: Date.now()
            });
        }

        const patternData = patterns.get(type);
        patternData.count += data.length;
        patternData.samples.push(...data);
        patternData.lastUpdate = Date.now();

        // Keep only recent samples
        if (patternData.samples.length > 100) {
            patternData.samples = patternData.samples.slice(-50);
        }
    }

    async analyzePatterns() {
        console.log("🔍 Running pattern analysis...");

        for (const [type, data] of this.patternDetector.mathematicalPatterns.entries()) {
            if (data.samples.length < 5) continue;

            const analysis = this.performPatternAnalysis(data.samples);

            if (analysis.significance > 0.7) {
                await this.reportPatternDiscovery(type, analysis);
            }
        }
    }

    performPatternAnalysis(samples) {
        // Advanced pattern analysis
        const analysis = {
            frequency: samples.length,
            complexity: this.calculateAverageComplexity(samples),
            coherence: this.calculateCoherence(samples),
            novelty: this.calculateNovelty(samples),
            significance: 0
        };

        // Calculate overall significance
        analysis.significance = (analysis.complexity * 0.3 +
            analysis.coherence * 0.4 +
            analysis.novelty * 0.3);

        return analysis;
    }

    calculateAverageComplexity(samples) {
        const complexities = samples.map(sample => this.calculateComplexity(sample));
        return complexities.reduce((a, b) => a + b, 0) / complexities.length;
    }

    calculateComplexity(data) {
        // Simple complexity metric based on data structure
        const jsonStr = JSON.stringify(data);
        return Math.min(1.0, jsonStr.length / 1000);
    }

    calculateCoherence(samples) {
        // Measure how similar samples are to each other
        let totalSimilarity = 0;
        let comparisons = 0;

        for (let i = 0; i < samples.length - 1; i++) {
            for (let j = i + 1; j < samples.length; j++) {
                totalSimilarity += this.calculateSimilarity(samples[i], samples[j]);
                comparisons++;
            }
        }

        return comparisons > 0 ? totalSimilarity / comparisons : 0;
    }

    calculateSimilarity(sample1, sample2) {
        // Simple similarity metric
        const keys1 = Object.keys(sample1 || {});
        const keys2 = Object.keys(sample2 || {});
        const commonKeys = keys1.filter(key => keys2.includes(key));

        return commonKeys.length / Math.max(keys1.length, keys2.length, 1);
    }

    calculateNovelty(samples) {
        // Measure how different recent samples are from historical data
        const recent = samples.slice(-10);
        const historical = samples.slice(0, -10);

        if (historical.length === 0) return 1.0;

        let noveltySum = 0;
        for (const recentSample of recent) {
            let maxSimilarity = 0;
            for (const historicalSample of historical) {
                const similarity = this.calculateSimilarity(recentSample, historicalSample);
                maxSimilarity = Math.max(maxSimilarity, similarity);
            }
            noveltySum += (1 - maxSimilarity);
        }

        return recent.length > 0 ? noveltySum / recent.length : 0;
    }

    async reportPatternDiscovery(type, analysis) {
        console.log(`🎯 Significant pattern discovered: ${type}`, analysis);

        const discoveryReport = {
            type: `pattern-discovery-${type}`,
            analysis,
            significance: analysis.significance,
            timestamp: Date.now(),
            details: {
                frequency: analysis.frequency,
                complexity: analysis.complexity,
                coherence: analysis.coherence,
                novelty: analysis.novelty
            }
        };

        // Send to Nucleus
        await this.sendNucleusMessage(
            `Pattern Discovery: ${type} - Significance: ${analysis.significance.toFixed(2)}`,
            'pattern-discovery',
            'high'
        );

        // Train Nucleus with the discovery
        this.queueTrainingData(discoveryReport);

        // Update analytics
        this.learningAnalytics.discoveries.set(type, discoveryReport);
    }

    async requestNucleusAnalysis(experimentData) {
        if (!this.nucleusCommunicator) return;

        try {
            const query = `Analyze experiment: ${experimentData.id}.
                          Type: ${experimentData.type}.
                          Performance: ${JSON.stringify(experimentData.performance)}.
                          Provide insights for optimization.`;

            const response = await this.nucleusCommunicator.requestNucleusInsights(query);

            if (response && response.data) {
                this.learningAnalytics.insightsReceived++;

                // Store insights for future reference
                experimentData.nucleusInsights = response.data;

                console.log(`💡 Nucleus insights for ${experimentData.id}:`, response.data);
            }

        } catch (error) {
            console.error("❌ Failed to request Nucleus analysis:", error);
        }
    }

    analyzePerformance(result) {
        return {
            executionTime: result.executionTime || 0,
            accuracy: result.accuracy || 1.0,
            efficiency: result.efficiency || 1.0,
            resourceUsage: result.resourceUsage || 0.5
        };
    }

    extractInsights(result) {
        const insights = [];

        if (result.accuracy && result.accuracy > 0.95) {
            insights.push("High accuracy achieved");
        }

        if (result.executionTime && result.executionTime < 100) {
            insights.push("Fast execution time");
        }

        if (result.patterns && result.patterns.length > 0) {
            insights.push(`${result.patterns.length} patterns detected`);
        }

        return insights;
    }

    queueTrainingData(data) {
        if (this.trainingConfig.smartBatching) {
            this.trainingQueue.push({
                ...data,
                priority: this.calculatePriority(data),
                timestamp: Date.now()
            });
        } else {
            // Send immediately
            this.sendTrainingDataToNucleus(data);
        }
    }

    calculatePriority(data) {
        // Priority based on data type and significance
        const typeWeights = {
            'pattern-discovery': 0.9,
            'experiment-result': 0.8,
            'fractal-generation': 0.7,
            'geometric-calculation': 0.6,
            'algebraic-operation': 0.5
        };

        return typeWeights[data.type] || 0.5;
    }

    async sendTrainingDataToNucleus(data) {
        if (!this.nucleusCommunicator) return;

        try {
            await this.nucleusCommunicator.trainNucleus(data, data.type);
            this.learningAnalytics.trainingSessions++;
        } catch (error) {
            console.error("❌ Failed to send training data:", error);
        }
    }

    async sendNucleusMessage(content, type = 'general', priority = 'normal') {
        if (this.realtimeConfig.smartBatching) {
            this.messageQueue.push({
                content,
                type,
                priority,
                timestamp: Date.now()
            });
        } else {
            // Send immediately
            try {
                await this.nucleusCommunicator.sendMessageToNucleus(content, type, priority);
                this.learningAnalytics.totalMessages++;
            } catch (error) {
                console.error("❌ Failed to send message:", error);
            }
        }
    }

    startRealtimeProcessing() {
        console.log("⚡ Starting real-time processing...");

        // Adaptive processing based on queue size
        setInterval(() => {
            const queueSize = this.messageQueue.length + this.trainingQueue.length;

            if (queueSize > 50) {
                // High load - process more frequently
                this.realtimeConfig.batchTimeout = 2000;
                this.realtimeConfig.batchSize = 15;
            } else if (queueSize > 20) {
                // Medium load
                this.realtimeConfig.batchTimeout = 3000;
                this.realtimeConfig.batchSize = 10;
            } else {
                // Low load - normal processing
                this.realtimeConfig.batchTimeout = 5000;
                this.realtimeConfig.batchSize = 5;
            }
        }, 15000);
    }

    startHealthMonitoring() {
        console.log("🏥 Starting health monitoring...");

        setInterval(async () => {
            await this.checkConnectionHealth();
        }, 10000);
    }

    async checkConnectionHealth() {
        if (!this.nucleusCommunicator) return;

        const startTime = Date.now();

        try {
            const isConnected = this.nucleusCommunicator.isConnected();
            const latency = Date.now() - startTime;

            this.connectionHealth = {
                status: isConnected ? 'connected' : 'disconnected',
                lastPing: Date.now(),
                latency,
                reliability: isConnected ? Math.min(100, this.connectionHealth.reliability + 1) :
                    Math.max(0, this.connectionHealth.reliability - 5)
            };

            if (!isConnected) {
                console.warn("⚠️ Nucleus connection lost - attempting reconnection...");
                await this.attemptReconnection();
            }

        } catch (error) {
            console.error("❌ Health check failed:", error);
            this.connectionHealth.status = 'error';
            this.connectionHealth.reliability = Math.max(0, this.connectionHealth.reliability - 10);
        }
    }

    async attemptReconnection() {
        try {
            // Re-initialize communicator
            this.nucleusCommunicator = new NexusForgeNucleusCommunicator();
            console.log("✅ Reconnection successful");
        } catch (error) {
            console.error("❌ Reconnection failed:", error);
        }
    }

    async initializeTrainingSystems() {
        console.log("🎓 Initializing advanced training systems...");

        // Set up continuous learning
        if (this.trainingConfig.continuousSync) {
            setInterval(() => {
                this.performContinuousLearning();
            }, 30000);
        }

        // Set up insight generation
        if (this.trainingConfig.insightGeneration) {
            setInterval(() => {
                this.generateInsights();
            }, 60000);
        }
    }

    async performContinuousLearning() {
        if (!this.sandbox || !this.nucleusCommunicator) return;

        try {
            // Run a background experiment for learning
            const randomExperiment = this.selectRandomExperiment();
            if (randomExperiment) {
                const result = await this.sandbox.runExperiment(randomExperiment);
                console.log(`🎯 Continuous learning: ${randomExperiment} completed`);
            }
        } catch (error) {
            console.error("❌ Continuous learning failed:", error);
        }
    }

    selectRandomExperiment() {
        const experiments = [
            'algebraic-patterns',
            'geometric-sequences',
            'fractal-analysis',
            'sacred-geometry',
            'physics-simulation',
            'number-theory'
        ];

        return experiments[Math.floor(Math.random() * experiments.length)];
    }

    async generateInsights() {
        console.log("💡 Generating insights from accumulated data...");

        const insights = [];

        // Analyze patterns
        for (const [type, data] of this.patternDetector.mathematicalPatterns.entries()) {
            if (data.count > 10) {
                insights.push(`${type}: Processed ${data.count} operations with high efficiency`);
            }
        }

        // Analyze performance trends
        const totalOperations = this.learningAnalytics.totalMessages + this.learningAnalytics.trainingSessions;
        if (totalOperations > 100) {
            insights.push(`System processed ${totalOperations} operations with ${this.connectionHealth.reliability}% reliability`);
        }

        // Send insights to Nucleus
        for (const insight of insights) {
            await this.sendNucleusMessage(
                `System Insight: ${insight}`,
                'system-insight',
                'normal'
            );
        }
    }

    // Public API methods

    getStatus() {
        return {
            status: this.status,
            version: this.version,
            analytics: this.learningAnalytics,
            connectionHealth: this.connectionHealth,
            queueSizes: {
                messages: this.messageQueue.length,
                training: this.trainingQueue.length
            }
        };
    }

    async sendDirectMessage(message, type = 'user-input', priority = 'normal') {
        return await this.sendNucleusMessage(message, type, priority);
    }

    async trainDirectly(data, category) {
        return await this.sendTrainingDataToNucleus({ data, type: category });
    }

    getAnalytics() {
        return {
            ...this.learningAnalytics,
            patterns: Object.fromEntries(this.patternDetector.mathematicalPatterns),
            discoveries: Object.fromEntries(this.learningAnalytics.discoveries)
        };
    }

    getConnectionHealth() {
        return this.connectionHealth;
    }

    updateConfiguration(config) {
        this.trainingConfig = { ...this.trainingConfig, ...config.training };
        this.realtimeConfig = { ...this.realtimeConfig, ...config.realtime };
        console.log("⚙️ Configuration updated", { training: this.trainingConfig, realtime: this.realtimeConfig });
    }

    async performFullSync() {
        console.log("🔄 Performing full system synchronization...");

        try {
            // Sync mathematical engine state
            if (this.mathEngine && this.nucleusCommunicator) {
                await this.nucleusCommunicator.syncMathematicalState(this.mathEngine);
            }

            // Sync sandbox state
            if (this.sandbox) {
                await this.sandbox.syncWithNucleus();
            }

            // Process all queued operations
            await this.processBatchedOperations();

            console.log("✅ Full synchronization completed");

        } catch (error) {
            console.error("❌ Full sync failed:", error);
        }
    }

    // Advanced capabilities

    async enableAdvancedLearning() {
        console.log("🧠 Enabling advanced learning capabilities...");

        this.trainingConfig.adaptiveLearning = true;
        this.trainingConfig.patternRecognition = true;

        // Start advanced pattern recognition
        this.advancedPatternRecognition();
    }

    advancedPatternRecognition() {
        setInterval(() => {
            // Analyze cross-domain patterns
            this.analyzeCrossDomainPatterns();

            // Detect emerging patterns
            this.detectEmergingPatterns();

            // Optimize learning parameters
            this.optimizeLearningParameters();

        }, 45000);
    }

    analyzeCrossDomainPatterns() {
        const domains = ['algebraic', 'geometric', 'fractal', 'physics'];
        const crossDomainPatterns = [];

        for (let i = 0; i < domains.length - 1; i++) {
            for (let j = i + 1; j < domains.length; j++) {
                const pattern1 = this.patternDetector.mathematicalPatterns.get(domains[i]);
                const pattern2 = this.patternDetector.mathematicalPatterns.get(domains[j]);

                if (pattern1 && pattern2) {
                    const correlation = this.calculateCorrelation(pattern1.samples, pattern2.samples);
                    if (correlation > 0.7) {
                        crossDomainPatterns.push({
                            domains: [domains[i], domains[j]],
                            correlation,
                            significance: correlation * 0.8
                        });
                    }
                }
            }
        }

        if (crossDomainPatterns.length > 0) {
            console.log("🔗 Cross-domain patterns detected:", crossDomainPatterns);
            this.sendNucleusMessage(
                `Cross-domain patterns detected: ${crossDomainPatterns.length} correlations found`,
                'cross-domain-analysis',
                'high'
            );
        }
    }

    calculateCorrelation(samples1, samples2) {
        // Simplified correlation calculation
        const min = Math.min(samples1.length, samples2.length);
        if (min < 3) return 0;

        let correlation = 0;
        for (let i = 0; i < min; i++) {
            correlation += this.calculateSimilarity(samples1[i], samples2[i]);
        }

        return correlation / min;
    }

    detectEmergingPatterns() {
        const recentWindow = 300000; // 5 minutes
        const currentTime = Date.now();

        for (const [type, data] of this.patternDetector.mathematicalPatterns.entries()) {
            const recentSamples = data.samples.filter(
                sample => (currentTime - sample.timestamp) < recentWindow
            );

            if (recentSamples.length >= 5) {
                const emergingPattern = this.analyzeEmergingPattern(recentSamples);
                if (emergingPattern.strength > 0.8) {
                    console.log(`🌟 Emerging pattern in ${type}:`, emergingPattern);

                    this.sendNucleusMessage(
                        `Emerging pattern detected in ${type} with strength ${emergingPattern.strength.toFixed(2)}`,
                        'emerging-pattern',
                        'high'
                    );
                }
            }
        }
    }

    analyzeEmergingPattern(samples) {
        // Analyze trend strength
        const timeStamps = samples.map(s => s.timestamp).sort();
        const timeDiff = timeStamps[timeStamps.length - 1] - timeStamps[0];

        const frequency = samples.length / (timeDiff / 1000); // per second
        const consistency = this.calculateConsistency(samples);

        return {
            frequency,
            consistency,
            strength: (frequency * 0.6 + consistency * 0.4),
            samples: samples.length
        };
    }

    calculateConsistency(samples) {
        if (samples.length < 2) return 0;

        let consistencySum = 0;
        for (let i = 1; i < samples.length; i++) {
            const similarity = this.calculateSimilarity(samples[i - 1], samples[i]);
            consistencySum += similarity;
        }

        return consistencySum / (samples.length - 1);
    }

    optimizeLearningParameters() {
        const performance = this.connectionHealth.reliability / 100;
        const queueLoad = (this.messageQueue.length + this.trainingQueue.length) / 100;

        // Adaptive batch size based on performance
        if (performance > 0.9 && queueLoad < 0.3) {
            this.realtimeConfig.batchSize = Math.min(20, this.realtimeConfig.batchSize + 1);
        } else if (performance < 0.7 || queueLoad > 0.7) {
            this.realtimeConfig.batchSize = Math.max(5, this.realtimeConfig.batchSize - 1);
        }

        // Adaptive timeout based on load
        this.realtimeConfig.batchTimeout = Math.max(2000, 5000 - (queueLoad * 3000));

        console.log(`⚙️ Learning parameters optimized - Batch: ${this.realtimeConfig.batchSize}, Timeout: ${this.realtimeConfig.batchTimeout}ms`);
    }
}

// Global integration instance
let nucleusIntegration = null;

// Initialize when page loads
window.addEventListener('load', async () => {
    console.log("🚀 Initializing NEXUS Nucleus Integration...");
    nucleusIntegration = new NexusForgeNucleusIntegration();

    // Make available globally
    window.nucleusIntegration = nucleusIntegration;

    // Set up global event listeners for mathematical operations
    setupGlobalMathListeners();
});

function setupGlobalMathListeners() {
    // Listen for custom mathematical events
    document.addEventListener('mathematical-operation', (event) => {
        if (nucleusIntegration) {
            nucleusIntegration.queueTrainingData({
                type: 'mathematical-operation',
                operation: event.detail.operation,
                parameters: event.detail.parameters,
                result: event.detail.result,
                source: event.detail.source || 'global-listener',
                timestamp: Date.now()
            });
        }
    });

    // Listen for experiment events
    document.addEventListener('experiment-completed', (event) => {
        if (nucleusIntegration) {
            nucleusIntegration.queueTrainingData({
                type: 'experiment-result',
                experiment: event.detail,
                timestamp: Date.now()
            });
        }
    });

    // Listen for discovery events
    document.addEventListener('pattern-discovered', (event) => {
        if (nucleusIntegration) {
            nucleusIntegration.sendNucleusMessage(
                `Pattern discovered: ${event.detail.type} - ${event.detail.description}`,
                'pattern-discovery',
                'high'
            );
        }
    });
}

// Export for module systems
if (typeof module !== 'undefined' && module.exports) {
    module.exports = NexusForgeNucleusIntegration;
}
