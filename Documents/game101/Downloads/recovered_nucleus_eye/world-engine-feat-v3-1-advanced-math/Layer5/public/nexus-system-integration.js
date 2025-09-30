// NEXUS Forge Complete System Integration v5.0.0
// Master integration hub for all NEXUS systems

class NexusSystemIntegration {
    constructor() {
        this.version = '5.0.0';
        this.systems = new Map();
        this.messageQueue = [];
        this.eventBus = new EventTarget();
        this.isInitialized = false;
        this.debugMode = true;
        this.connectionStatus = new Map();

        this.initializeIntegration();
    }

    initializeIntegration() {
        console.log('🚀 NEXUS System Integration v5.0.0 - Initializing...');

        // Register all known systems
        this.registerSystems();

        // Setup communication channels
        this.setupCommunication();

        // Initialize cross-system data sharing
        this.setupDataSharing();

        // Start system monitoring
        this.startSystemMonitoring();

        // Setup automatic optimization
        this.setupAutoOptimization();

        this.isInitialized = true;
        console.log('✅ NEXUS System Integration ready');
    }

    registerSystems() {
        // Core NEXUS systems
        this.systems.set('nucleus', {
            name: 'Nucleus AI Training System',
            url: 'nucleus-training-interface.html',
            apis: ['sendMessage', 'trainAI', 'getInsights', 'monitorProgress'],
            status: 'offline',
            priority: 1,
            dependencies: []
        });

        this.systems.set('playground', {
            name: '3D Mathematical Playground',
            url: 'nexus-3d-mathematical-playground.html',
            apis: ['calculateDistance', 'renderScene', 'updateMath', 'exportData'],
            status: 'offline',
            priority: 2,
            dependencies: ['optimizer']
        });

        this.systems.set('education', {
            name: 'Educational Bridge System',
            url: 'nexus-3d-mathematical-playground.html',
            apis: ['startLearning', 'trackProgress', 'generateCurriculum', 'assessLearning'],
            status: 'offline',
            priority: 3,
            dependencies: ['playground']
        });

        this.systems.set('optimizer', {
            name: 'System Performance Optimizer',
            url: 'nexus-system-optimizer.js',
            apis: ['optimizeWebGL', 'createMathFunctions', 'optimizeAI', 'getStatus'],
            status: 'loading',
            priority: 0,
            dependencies: []
        });

        this.systems.set('control', {
            name: 'Master Control Center',
            url: 'nexus-master-control-center.html',
            apis: ['launchSystem', 'runDiagnostics', 'exportData', 'showDocumentation'],
            status: 'loading',
            priority: 0,
            dependencies: []
        });
    }

    setupCommunication() {
        // Cross-window messaging
        window.addEventListener('message', (event) => {
            this.handleMessage(event);
        });

        // Custom event system for internal communication
        this.eventBus.addEventListener('systemMessage', (event) => {
            this.processSystemMessage(event.detail);
        });

        // WebSocket for real-time communication (if available)
        this.setupWebSocket();

        // SharedArrayBuffer for high-performance data sharing (if supported)
        this.setupSharedMemory();
    }

    setupWebSocket() {
        // This would connect to a local WebSocket server for real-time communication
        // For now, we'll simulate it with localStorage events
        window.addEventListener('storage', (event) => {
            if (event.key && event.key.startsWith('nexus_')) {
                this.handleStorageMessage(event);
            }
        });
    }

    setupSharedMemory() {
        // Check for SharedArrayBuffer support
        if (typeof SharedArrayBuffer !== 'undefined') {
            try {
                this.sharedBuffer = new SharedArrayBuffer(1024 * 1024); // 1MB shared buffer
                this.sharedView = new Float32Array(this.sharedBuffer);
                console.log('✅ SharedArrayBuffer initialized for high-performance data sharing');
            } catch (error) {
                console.warn('⚠️ SharedArrayBuffer not available:', error.message);
            }
        }
    }

    setupDataSharing() {
        // Shared data store for cross-system information
        this.sharedData = {
            mathematicalResults: new Map(),
            trainingData: new Map(),
            performanceMetrics: new Map(),
            userProgress: new Map(),
            systemState: new Map()
        };

        // Periodic data synchronization
        setInterval(() => {
            this.synchronizeData();
        }, 5000);
    }

    startSystemMonitoring() {
        // Monitor system health and connectivity
        setInterval(() => {
            this.checkSystemHealth();
        }, 3000);

        // Monitor cross-system dependencies
        setInterval(() => {
            this.checkDependencies();
        }, 10000);
    }

    setupAutoOptimization() {
        // Listen for performance issues and auto-optimize
        this.eventBus.addEventListener('performanceIssue', (event) => {
            this.handlePerformanceIssue(event.detail);
        });

        // Proactive optimization based on usage patterns
        setInterval(() => {
            this.proactiveOptimization();
        }, 30000);
    }

    // Message handling
    handleMessage(event) {
        const { type, data, source, target } = event.data;

        if (!type || !type.startsWith('nexus_')) return;

        if (this.debugMode) {
            console.log('📨 Message received:', { type, source, target });
        }

        switch (type) {
            case 'nexus_system_register':
                this.registerExternalSystem(data);
                break;
            case 'nexus_data_request':
                this.handleDataRequest(data, event.source);
                break;
            case 'nexus_data_update':
                this.handleDataUpdate(data);
                break;
            case 'nexus_performance_report':
                this.handlePerformanceReport(data);
                break;
            case 'nexus_error_report':
                this.handleErrorReport(data);
                break;
            default:
                this.routeMessage(event.data);
        }
    }

    handleStorageMessage(event) {
        try {
            const data = JSON.parse(event.newValue);
            this.processSystemMessage({
                type: 'storage_update',
                key: event.key,
                data: data,
                timestamp: Date.now()
            });
        } catch (error) {
            console.warn('Failed to parse storage message:', error);
        }
    }

    processSystemMessage(message) {
        // Process internal system messages
        switch (message.type) {
            case 'math_calculation_complete':
                this.propagateMathResults(message.data);
                break;
            case 'ai_training_update':
                this.propagateTrainingUpdate(message.data);
                break;
            case 'learning_progress_update':
                this.propagateLearningProgress(message.data);
                break;
            case 'performance_optimization':
                this.applyOptimization(message.data);
                break;
        }
    }

    // Data synchronization
    synchronizeData() {
        // Sync mathematical results across systems
        this.syncMathematicalData();

        // Sync AI training data
        this.syncTrainingData();

        // Sync user progress
        this.syncUserProgress();

        // Sync performance metrics
        this.syncPerformanceData();
    }

    syncMathematicalData() {
        const mathData = {
            timestamp: Date.now(),
            results: Array.from(this.sharedData.mathematicalResults.entries()),
            activeCalculations: this.getActiveCalculations()
        };

        this.broadcastData('math_sync', mathData);
    }

    syncTrainingData() {
        const trainingData = {
            timestamp: Date.now(),
            sessions: Array.from(this.sharedData.trainingData.entries()),
            progress: this.getTrainingProgress()
        };

        this.broadcastData('training_sync', trainingData);
    }

    syncUserProgress() {
        const progressData = {
            timestamp: Date.now(),
            progress: Array.from(this.sharedData.userProgress.entries()),
            achievements: this.getUserAchievements()
        };

        this.broadcastData('progress_sync', progressData);
    }

    syncPerformanceData() {
        const performanceData = {
            timestamp: Date.now(),
            metrics: Array.from(this.sharedData.performanceMetrics.entries()),
            systemHealth: this.calculateOverallHealth()
        };

        this.broadcastData('performance_sync', performanceData);
    }

    // System health monitoring
    checkSystemHealth() {
        for (const [systemId, system] of this.systems.entries()) {
            this.pingSystem(systemId);
        }
    }

    async pingSystem(systemId) {
        const system = this.systems.get(systemId);
        if (!system) return;

        try {
            const startTime = Date.now();

            // Try to communicate with the system
            const response = await this.sendSystemMessage(systemId, {
                type: 'health_check',
                timestamp: startTime
            });

            const responseTime = Date.now() - startTime;

            if (response) {
                system.status = 'online';
                system.lastSeen = Date.now();
                system.responseTime = responseTime;
                this.connectionStatus.set(systemId, true);
            }
        } catch (error) {
            system.status = 'offline';
            system.lastError = error.message;
            this.connectionStatus.set(systemId, false);
        }
    }

    checkDependencies() {
        for (const [systemId, system] of this.systems.entries()) {
            if (system.dependencies.length > 0) {
                const missingDeps = system.dependencies.filter(dep => {
                    const depSystem = this.systems.get(dep);
                    return !depSystem || depSystem.status !== 'online';
                });

                if (missingDeps.length > 0) {
                    console.warn(`⚠️ System ${systemId} has missing dependencies:`, missingDeps);
                    this.handleMissingDependencies(systemId, missingDeps);
                }
            }
        }
    }

    handleMissingDependencies(systemId, missingDeps) {
        // Attempt to start missing dependencies
        missingDeps.forEach(depId => {
            this.startSystem(depId);
        });

        // Notify the system about dependency issues
        this.sendSystemMessage(systemId, {
            type: 'dependency_warning',
            missing: missingDeps,
            timestamp: Date.now()
        });
    }

    // System management
    async startSystem(systemId) {
        const system = this.systems.get(systemId);
        if (!system) {
            console.error(`❌ Unknown system: ${systemId}`);
            return false;
        }

        console.log(`🚀 Starting system: ${system.name}`);

        try {
            // If it's a JavaScript module, load it
            if (system.url.endsWith('.js')) {
                await this.loadJavaScriptSystem(system);
            } else {
                // If it's an HTML page, open it
                await this.launchHTMLSystem(system);
            }

            system.status = 'starting';
            system.lastStarted = Date.now();

            return true;
        } catch (error) {
            console.error(`❌ Failed to start ${systemId}:`, error);
            system.status = 'error';
            system.lastError = error.message;
            return false;
        }
    }

    async loadJavaScriptSystem(system) {
        return new Promise((resolve, reject) => {
            const script = document.createElement('script');
            script.src = system.url;
            script.onload = () => {
                console.log(`✅ JavaScript system loaded: ${system.name}`);
                resolve();
            };
            script.onerror = () => {
                reject(new Error(`Failed to load JavaScript: ${system.url}`));
            };
            document.head.appendChild(script);
        });
    }

    async launchHTMLSystem(system) {
        const windowFeatures = this.getWindowFeatures(system);
        const newWindow = window.open(system.url, `nexus_${system.name}`, windowFeatures);

        if (!newWindow) {
            throw new Error('Failed to open window - popup blocked?');
        }

        // Monitor the window
        this.monitorSystemWindow(newWindow, system);

        return newWindow;
    }

    getWindowFeatures(system) {
        // Default window features based on system type
        const defaultFeatures = 'width=1200,height=800,scrollbars=yes,resizable=yes';

        const customFeatures = {
            'nucleus': 'width=1400,height=900,scrollbars=yes,resizable=yes',
            'playground': 'width=1600,height=1000,scrollbars=no,resizable=yes',
            'education': 'width=1600,height=1000,scrollbars=no,resizable=yes',
            'control': 'width=1800,height=1200,scrollbars=no,resizable=yes'
        };

        return customFeatures[system.name] || defaultFeatures;
    }

    monitorSystemWindow(windowRef, system) {
        const checkInterval = setInterval(() => {
            if (windowRef.closed) {
                system.status = 'offline';
                clearInterval(checkInterval);
                console.log(`📝 System window closed: ${system.name}`);
            }
        }, 1000);
    }

    // Communication methods
    async sendSystemMessage(systemId, message) {
        const system = this.systems.get(systemId);
        if (!system) return null;

        // Add metadata
        const fullMessage = {
            ...message,
            source: 'integration_hub',
            target: systemId,
            timestamp: Date.now(),
            id: this.generateMessageId()
        };

        // Try different communication methods
        try {
            // Try direct window communication first
            const response = await this.sendWindowMessage(system, fullMessage);
            if (response) return response;

            // Try localStorage communication
            const storageResponse = await this.sendStorageMessage(systemId, fullMessage);
            if (storageResponse) return storageResponse;

            // Try custom event system
            return this.sendEventMessage(systemId, fullMessage);
        } catch (error) {
            console.warn(`Failed to send message to ${systemId}:`, error);
            return null;
        }
    }

    async sendWindowMessage(system, message) {
        // This would send to specific system windows
        // For now, broadcast to all windows
        if (window.nexusWindows) {
            window.nexusWindows.forEach(win => {
                if (win && !win.closed) {
                    try {
                        win.postMessage({
                            type: 'nexus_system_message',
                            data: message
                        }, '*');
                    } catch (error) {
                        // Cross-origin error, ignore
                    }
                }
            });
        }

        return new Promise(resolve => {
            setTimeout(() => resolve(null), 1000);
        });
    }

    async sendStorageMessage(systemId, message) {
        const key = `nexus_message_${systemId}_${Date.now()}`;
        localStorage.setItem(key, JSON.stringify(message));

        // Clean up after a delay
        setTimeout(() => {
            localStorage.removeItem(key);
        }, 5000);

        return message;
    }

    sendEventMessage(systemId, message) {
        this.eventBus.dispatchEvent(new CustomEvent('systemMessage', {
            detail: {
                ...message,
                targetSystem: systemId
            }
        }));

        return message;
    }

    broadcastData(type, data) {
        const message = {
            type: `nexus_${type}`,
            data: data,
            timestamp: Date.now(),
            source: 'integration_hub'
        };

        // Broadcast to all systems
        for (const systemId of this.systems.keys()) {
            this.sendSystemMessage(systemId, message);
        }

        // Also use localStorage for persistent communication
        localStorage.setItem(`nexus_broadcast_${type}`, JSON.stringify(message));
    }

    // Data management
    storeSharedData(category, key, value) {
        if (!this.sharedData[category]) {
            this.sharedData[category] = new Map();
        }

        this.sharedData[category].set(key, {
            value: value,
            timestamp: Date.now(),
            source: 'integration_hub'
        });

        // Broadcast update
        this.broadcastData('data_update', {
            category,
            key,
            value,
            action: 'store'
        });
    }

    getSharedData(category, key) {
        if (!this.sharedData[category]) return null;

        const data = this.sharedData[category].get(key);
        return data ? data.value : null;
    }

    // Utility methods
    generateMessageId() {
        return `msg_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    }

    getActiveCalculations() {
        // Return list of active mathematical calculations
        return Array.from(this.sharedData.mathematicalResults.keys())
            .filter(key => {
                const data = this.sharedData.mathematicalResults.get(key);
                return data && (Date.now() - data.timestamp) < 30000; // Active in last 30 seconds
            });
    }

    getTrainingProgress() {
        // Calculate overall AI training progress
        const sessions = Array.from(this.sharedData.trainingData.values());
        if (sessions.length === 0) return 0;

        const totalProgress = sessions.reduce((sum, session) => {
            return sum + (session.value.progress || 0);
        }, 0);

        return totalProgress / sessions.length;
    }

    getUserAchievements() {
        // Generate user achievement data
        const achievements = [];
        const progress = this.sharedData.userProgress;

        if (progress.has('mathematics_mastery') && progress.get('mathematics_mastery').value > 0.8) {
            achievements.push('Mathematical Mastery');
        }

        if (progress.has('ai_training_sessions') && progress.get('ai_training_sessions').value > 10) {
            achievements.push('AI Training Specialist');
        }

        return achievements;
    }

    calculateOverallHealth() {
        const systems = Array.from(this.systems.values());
        const onlineSystems = systems.filter(s => s.status === 'online').length;
        const totalSystems = systems.length;

        const healthPercentage = (onlineSystems / totalSystems) * 100;

        return {
            percentage: healthPercentage,
            onlineSystems,
            totalSystems,
            status: healthPercentage > 80 ? 'healthy' : healthPercentage > 50 ? 'warning' : 'critical'
        };
    }

    // Event handlers
    handleDataRequest(data, source) {
        const requestedData = this.getSharedData(data.category, data.key);

        if (source && source.postMessage) {
            source.postMessage({
                type: 'nexus_data_response',
                data: requestedData,
                requestId: data.requestId
            }, '*');
        }
    }

    handleDataUpdate(data) {
        this.storeSharedData(data.category, data.key, data.value);
    }

    handlePerformanceReport(data) {
        this.sharedData.performanceMetrics.set(data.systemId, {
            value: data,
            timestamp: Date.now(),
            source: data.systemId
        });
    }

    handleErrorReport(data) {
        console.error(`❌ Error from ${data.systemId}:`, data.error);

        // Store error for analysis
        if (!this.sharedData.errorLog) {
            this.sharedData.errorLog = new Map();
        }

        this.sharedData.errorLog.set(`${data.systemId}_${Date.now()}`, {
            value: data,
            timestamp: Date.now(),
            source: data.systemId
        });
    }

    handlePerformanceIssue(issue) {
        console.warn('⚠️ Performance issue detected:', issue);

        // Trigger optimization
        this.applyOptimization({
            type: 'auto_optimization',
            trigger: issue,
            timestamp: Date.now()
        });
    }

    applyOptimization(optimization) {
        // Apply optimization across all systems
        this.broadcastData('optimization', optimization);

        console.log('🔧 Applied optimization:', optimization.type);
    }

    proactiveOptimization() {
        const health = this.calculateOverallHealth();

        if (health.percentage < 80) {
            this.applyOptimization({
                type: 'proactive_optimization',
                reason: 'low_system_health',
                healthPercentage: health.percentage,
                timestamp: Date.now()
            });
        }
    }

    // Public API
    getSystemStatus() {
        return {
            isInitialized: this.isInitialized,
            systems: Array.from(this.systems.entries()),
            health: this.calculateOverallHealth(),
            sharedDataSize: this.getSharedDataSize(),
            messageQueueSize: this.messageQueue.length
        };
    }

    getSharedDataSize() {
        let size = 0;
        for (const [, dataMap] of Object.entries(this.sharedData)) {
            if (dataMap instanceof Map) {
                size += dataMap.size;
            }
        }
        return size;
    }

    enableDebugMode(enabled = true) {
        this.debugMode = enabled;
        console.log(`🔍 Integration debug mode ${enabled ? 'enabled' : 'disabled'}`);
    }

    exportSystemData() {
        return {
            version: this.version,
            timestamp: new Date().toISOString(),
            systems: Object.fromEntries(this.systems),
            sharedData: this.serializeSharedData(),
            health: this.calculateOverallHealth(),
            connectionStatus: Object.fromEntries(this.connectionStatus)
        };
    }

    serializeSharedData() {
        const serialized = {};

        for (const [category, dataMap] of Object.entries(this.sharedData)) {
            if (dataMap instanceof Map) {
                serialized[category] = Object.fromEntries(dataMap);
            } else {
                serialized[category] = dataMap;
            }
        }

        return serialized;
    }
}

// Global instance
window.nexusIntegration = new NexusSystemIntegration();

// Expose public API
window.NexusIntegration = {
    getInstance: () => window.nexusIntegration,
    startSystem: (systemId) => window.nexusIntegration.startSystem(systemId),
    sendMessage: (systemId, message) => window.nexusIntegration.sendSystemMessage(systemId, message),
    storeData: (category, key, value) => window.nexusIntegration.storeSharedData(category, key, value),
    getData: (category, key) => window.nexusIntegration.getSharedData(category, key),
    getStatus: () => window.nexusIntegration.getSystemStatus(),
    exportData: () => window.nexusIntegration.exportSystemData(),
    enableDebug: (enabled) => window.nexusIntegration.enableDebugMode(enabled)
};

console.log('🚀 NEXUS System Integration loaded and ready');
console.log('💡 Access via window.NexusIntegration or window.nexusIntegration');
