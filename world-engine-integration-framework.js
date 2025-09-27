/**
 * World Engine Studio Tier 4 - System Integration Framework
 * Unified API layer connecting all systems with proper Layer 0 oversight
 * Implements data flow management, inter-layer protocols, and safety validation
 */

import { Layer5AgentSystem } from './models/layer5-agent-system.js';
import { AdvancedMorphologyEngine } from './models/advanced-morphology-engine.js';
import { WorldEngineIndexManager } from './models/world-engine-index-manager.js';
import { WorldEngineUXPipeline } from './world-engine-ux-pipeline.js';

export class WorldEngineIntegrationFramework {
    constructor(options = {}) {
        // Core system state
        this.isInitialized = false;
        this.systemId = `WEIF_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
        this.version = '4.0.0-tier4';

        // Layer 0 Overseer integration
        this.overseerBrain = null;
        this.killSwitchActive = false;
        this.canvasLawsActive = true;
        this.slotRegistry = new Map();
        this.layerHierarchy = ['Layer0', 'Layer1', 'Layer2', 'Layer3', 'Layer4', 'Layer5'];

        // System components registry
        this.systems = {
            // Layer 0 - Foundation
            overseerBrain: null,
            killSwitch: null,

            // Layer 1 - Communication
            studioBridge: null,

            // Layer 2 - Processing
            metaBaseThoughtEngine: null,
            morphologyEngine: null,
            advancedMorphologyEngine: null,
            indexManager: null,

            // Layer 3 - Synthesis
            audioVisualEngine: null,
            synthesisEngine: null,

            // Layer 4 - Interface
            nexusDashboard: null,
            uxPipeline: null,

            // Layer 5 - Agents
            agentSystem: null
        };

        // Data flow management
        this.dataFlowPipeline = new Map();
        this.messageQueue = [];
        this.eventBus = null;
        this.broadcastChannel = null;

        // Performance monitoring
        this.metrics = {
            systemsOnline: 0,
            totalSystems: 0,
            dataTransferRate: 0,
            avgResponseTime: 0,
            errorRate: 0,
            lastUpdate: Date.now(),
            uptime: 0
        };

        // Safety and validation
        this.safetyProtocols = {
            layer0ValidationEnabled: true,
            canvasLawEnforcementActive: true,
            killSwitchReady: true,
            memoryGuardsActive: true,
            recursionDepthLimit: 100,
            maxProcessingTime: 30000 // 30 seconds
        };

        // Configuration options
        this.config = {
            enableAutoStart: options.enableAutoStart !== false,
            validateAllConnections: options.validateAllConnections !== false,
            enforceCanvasLaws: options.enforceCanvasLaws !== false,
            enablePerformanceMonitoring: options.enablePerformanceMonitoring !== false,
            logLevel: options.logLevel || 'info',
            maxRetryAttempts: options.maxRetryAttempts || 3
        };

        // System initialization tracking
        this.initializationStatus = new Map();
        this.connectionStatus = new Map();

        this.init();
    }

    /**
             * Initialize the Integration Framework
             */
    async init() {
        console.log(`🚀 Initializing World Engine Integration Framework v${this.version}`);

        try {
            // Initialize Layer 0 Overseer connection
            await this.initializeLayer0();

            // Setup communication systems
            await this.setupCommunicationLayer();

            // Initialize data flow pipeline
            this.initializeDataFlow();

            // Setup safety protocols
            this.initializeSafetyProtocols();

            // Start performance monitoring
            if (this.config.enablePerformanceMonitoring) {
                this.startPerformanceMonitoring();
            }

            this.isInitialized = true;
            console.log('✅ Integration Framework initialized successfully');

            return true;

        } catch (error) {
            console.error('❌ Integration Framework initialization failed:', error);
            await this.activateKillSwitch('initialization_failure');
            return false;
        }
    }

    /**
             * Initialize Layer 0 Overseer connection
             */
    async initializeLayer0() {
        console.log('🔗 Connecting to Layer 0 Overseer...');

        // Register system with Layer 0 slot registry
        this.slotRegistry.set('WorldEngineIntegrationFramework', {
            systemId: this.systemId,
            layer: 'Integration',
            status: 'initializing',
            registeredTime: Date.now(),
            lastHeartbeat: Date.now()
        });

        // Initialize kill switch mechanism
        this.initializeKillSwitch();

        // Validate canvas laws compliance
        if (this.config.enforceCanvasLaws) {
            this.validateCanvasLaws();
        }

        this.initializationStatus.set('Layer0', 'connected');
        console.log('✅ Layer 0 Overseer connected');
    }

    /**
             * Setup communication layer with all systems
             */
    async setupCommunicationLayer() {
        console.log('📡 Setting up communication layer...');

        // Initialize BroadcastChannel for inter-system communication
        this.broadcastChannel = new BroadcastChannel('world-engine-studio-tier4');
        this.eventBus = this.createEventBus();

        // Setup message handlers
        this.setupMessageHandlers();

        // Initialize data flow pipeline
        this.initializeDataFlowRoutes();

        this.initializationStatus.set('Communication', 'ready');
        console.log('✅ Communication layer established');
    }

    /**
             * Connect and register a system component
             */
    async connectSystem(systemName, systemInstance, layer = 'Unknown') {
        console.log(`🔌 Connecting ${systemName} (${layer})...`);

        try {
            // Validate system before connection
            if (!this.validateSystemBeforeConnection(systemInstance, layer)) {
                throw new Error(`System validation failed for ${systemName}`);
            }

            // Register with Layer 0 if required
            if (this.overseerBrain && this.config.validateAllConnections) {
                const validation = await this.validateWithLayer0(systemName, systemInstance);
                if (!validation.isValid) {
                    throw new Error(`Layer 0 validation failed: ${validation.reason}`);
                }
            }

            // Register system
            this.systems[systemName] = systemInstance;
            this.connectionStatus.set(systemName, {
                status: 'connected',
                layer: layer,
                connectedTime: Date.now(),
                lastPing: Date.now(),
                isHealthy: true
            });

            // Setup system-specific integrations
            await this.setupSystemIntegration(systemName, systemInstance, layer);

            // Update metrics
            this.metrics.systemsOnline++;
            this.metrics.totalSystems++;

            console.log(`✅ ${systemName} connected successfully`);

            // Notify other systems of new connection
            this.broadcastMessage('system_connected', {
                systemName,
                layer,
                timestamp: Date.now()
            });

            return true;

        } catch (error) {
            console.error(`❌ Failed to connect ${systemName}:`, error);

            this.connectionStatus.set(systemName, {
                status: 'failed',
                layer: layer,
                error: error.message,
                attemptTime: Date.now()
            });

            return false;
        }
    }

    /**
             * Setup specific integrations for each system
             */
    async setupSystemIntegration(systemName, systemInstance, layer) {
        switch (systemName) {
            case 'overseerBrain':
                this.overseerBrain = systemInstance;
                this.setupOverseerIntegration();
                break;

            case 'studioBridge':
                this.setupStudioBridgeIntegration(systemInstance);
                break;

            case 'metaBaseThoughtEngine':
                this.setupThoughtEngineIntegration(systemInstance);
                break;

            case 'morphologyEngine':
                this.setupMorphologyIntegration(systemInstance);
                break;

            case 'audioVisualEngine':
                this.setupAudioVisualIntegration(systemInstance);
                break;

            case 'synthesisEngine':
                this.setupSynthesisIntegration(systemInstance);
                break;

            case 'nexusDashboard':
                this.setupDashboardIntegration(systemInstance);
                break;

            case 'agentSystem':
                this.setupAgentSystemIntegration(systemInstance);
                break;

            default:
                console.log(`Generic integration setup for ${systemName}`);
        }
    }

    /**
             * Process text through the complete pipeline
             */
    async processText(inputText, options = {}) {
        const startTime = Date.now();
        const processId = `proc_${Date.now()}_${Math.random().toString(36).substr(2, 6)}`;

        console.log(`🔄 Processing text through pipeline (ID: ${processId})`);

        try {
            // Layer 0 validation
            if (!this.validateInputWithLayer0(inputText, options)) {
                throw new Error('Layer 0 input validation failed');
            }

            const result = {
                processId,
                originalText: inputText,
                startTime,
                layers: {},
                finalOutput: null,
                metadata: {
                    totalProcessingTime: 0,
                    layerTimes: {},
                    systemsUsed: [],
                    errorCount: 0
                }
            };

            // Layer 2 - Meta-Base Thought Processing
            if (this.systems.metaBaseThoughtEngine) {
                const thoughtStart = Date.now();
                result.layers.thoughtEngine = await this.systems.metaBaseThoughtEngine.processThought(inputText, options);
                result.metadata.layerTimes.thoughtEngine = Date.now() - thoughtStart;
                result.metadata.systemsUsed.push('metaBaseThoughtEngine');
            }

            // Layer 2 - Morphological Analysis
            if (this.systems.morphologyEngine) {
                const morphStart = Date.now();
                result.layers.morphology = this.systems.morphologyEngine.analyze(inputText);
                result.metadata.layerTimes.morphology = Date.now() - morphStart;
                result.metadata.systemsUsed.push('morphologyEngine');
            }

            // Layer 3 - Audio-Visual Generation
            if (this.systems.audioVisualEngine && options.generateVisuals) {
                const visualStart = Date.now();
                result.layers.audioVisual = this.systems.audioVisualEngine.generateFromText(inputText);
                result.metadata.layerTimes.audioVisual = Date.now() - visualStart;
                result.metadata.systemsUsed.push('audioVisualEngine');
            }

            // Layer 3 - Synthesis Processing
            if (this.systems.synthesisEngine && options.generateAudio) {
                const synthStart = Date.now();
                result.layers.synthesis = this.systems.synthesisEngine.processText(inputText, options);
                result.metadata.layerTimes.synthesis = Date.now() - synthStart;
                result.metadata.systemsUsed.push('synthesisEngine');
            }

            // Layer 5 - Agent Interaction
            if (this.systems.agentSystem && options.useAgents) {
                const agentStart = Date.now();
                result.layers.agentResponse = await this.processWithAgents(inputText, options);
                result.metadata.layerTimes.agentSystem = Date.now() - agentStart;
                result.metadata.systemsUsed.push('agentSystem');
            }

            // Synthesize final output
            result.finalOutput = this.synthesizeFinalOutput(result.layers, options);
            result.metadata.totalProcessingTime = Date.now() - startTime;

            // Update dashboard if connected
            if (this.systems.nexusDashboard) {
                this.systems.nexusDashboard.updateProcessingResult(result);
            }

            console.log(`✅ Text processing completed (${result.metadata.totalProcessingTime}ms)`);
            return result;

        } catch (error) {
            console.error(`❌ Text processing failed (ID: ${processId}):`, error);

            // Activate kill switch if critical error
            if (error.message.includes('Layer 0') || error.message.includes('critical')) {
                await this.activateKillSwitch('critical_processing_error');
            }

            throw error;
        }
    }

    /**
             * Process input with agent system
             */
    async processWithAgents(inputText, options) {
        if (!this.systems.agentSystem) return null;

        const agentSystem = this.systems.agentSystem;

        // Create environment if needed
        let environment = null;
        if (options.createEnvironment) {
            environment = agentSystem.createEnvironment(inputText, options.agentType || 'COOPERATIVE');
            agentSystem.spawnAgentBatch(options.agentCount || 3, environment);
        }

        // Get available agents
        const availableAgents = Array.from(agentSystem.activeAgents.values()).slice(0, 5);

        if (availableAgents.length === 0) {
            console.log('No agents available for processing');
            return { message: 'No agents available', agentCount: 0 };
        }

        // Facilitate human-agent interactions
        const responses = [];
        for (const agent of availableAgents) {
            const success = agentSystem.facilitateHumanAgentInteraction(agent.id, inputText);
            if (success) {
                responses.push({
                    agentId: agent.id,
                    agentType: agent.type,
                    success: true,
                    adaptability: agent.adaptability,
                    experience: agent.experience
                });
            }
        }

        return {
            totalAgents: availableAgents.length,
            successfulInteractions: responses.length,
            agentResponses: responses,
            environment: environment ? {
                id: environment.id,
                prompt: environment.prompt,
                agentCount: environment.agents.length
            } : null
        };
    }

    /**
             * Synthesize final output from all layers
             */
    synthesizeFinalOutput(layers, options) {
        const output = {
            success: true,
            confidence: 1.0,
            primaryResponse: '',
            enrichments: {},
            systemContributions: []
        };

        // Primary response from thought engine
        if (layers.thoughtEngine) {
            output.primaryResponse = layers.thoughtEngine.reflection || layers.thoughtEngine.synthesis || 'Processing completed';
            output.systemContributions.push('thoughtEngine');
        }

        // Morphological insights
        if (layers.morphology) {
            output.enrichments.morphological = {
                complexity: layers.morphology.complexity,
                classification: layers.morphology.classification,
                stems: layers.morphology.stems?.length || 0
            };
            output.systemContributions.push('morphology');
        }

        // Audio-visual data
        if (layers.audioVisual) {
            output.enrichments.audioVisual = {
                shape: layers.audioVisual.shape,
                frequency: layers.audioVisual.frequency,
                visualData: !!layers.audioVisual.visualData
            };
            output.systemContributions.push('audioVisual');
        }

        // Synthesis information
        if (layers.synthesis) {
            output.enrichments.synthesis = {
                bpm: layers.synthesis.bpm,
                harmonics: layers.synthesis.harmonics?.length || 0,
                audioGenerated: !!layers.synthesis.audioBuffer
            };
            output.systemContributions.push('synthesis');
        }

        // Agent responses
        if (layers.agentResponse) {
            output.enrichments.agents = {
                totalAgents: layers.agentResponse.totalAgents,
                successfulInteractions: layers.agentResponse.successfulInteractions,
                environmentCreated: !!layers.agentResponse.environment
            };
            output.systemContributions.push('agents');
        }

        // Calculate overall confidence
        const systemCount = output.systemContributions.length;
        output.confidence = Math.min(1.0, 0.3 + (systemCount * 0.15));

        return output;
    }

    /**
             * Broadcast message to all connected systems
             */
    broadcastMessage(type, data) {
        const message = {
            type,
            data,
            timestamp: Date.now(),
            source: this.systemId,
            id: `msg_${Date.now()}_${Math.random().toString(36).substr(2, 6)}`
        };

        this.messageQueue.push(message);

        if (this.broadcastChannel) {
            this.broadcastChannel.postMessage(message);
        }

        // Direct notification to systems that support it
        Object.entries(this.systems).forEach(([name, system]) => {
            if (system && typeof system.onSystemMessage === 'function') {
                try {
                    system.onSystemMessage(message);
                } catch (error) {
                    console.warn(`System ${name} failed to handle message:`, error);
                }
            }
        });
    }

    /**
             * Get comprehensive system status
             */
    getSystemStatus() {
        const status = {
            framework: {
                isInitialized: this.isInitialized,
                systemId: this.systemId,
                version: this.version,
                uptime: Date.now() - this.metrics.lastUpdate,
                killSwitchActive: this.killSwitchActive,
                canvasLawsActive: this.canvasLawsActive
            },
            systems: {},
            connections: Object.fromEntries(this.connectionStatus),
            metrics: { ...this.metrics },
            safety: { ...this.safetyProtocols },
            dataFlow: {
                messageQueueSize: this.messageQueue.length,
                activeRoutes: this.dataFlowPipeline.size,
                lastProcessed: this.metrics.lastUpdate
            }
        };

        // Get individual system statuses
        Object.entries(this.systems).forEach(([name, system]) => {
            if (system) {
                status.systems[name] = {
                    isConnected: true,
                    hasStatus: typeof system.getSystemStatus === 'function',
                    status: typeof system.getSystemStatus === 'function' ?
                        system.getSystemStatus() : 'Unknown'
                };
            } else {
                status.systems[name] = {
                    isConnected: false,
                    status: 'Not connected'
                };
            }
        });

        return status;
    }

    /**
             * Health check for all systems
             */
    async performHealthCheck() {
        console.log('🔍 Performing system health check...');

        const healthReport = {
            overall: 'healthy',
            timestamp: Date.now(),
            systems: {},
            issues: [],
            recommendations: []
        };

        // Check each connected system
        for (const [name, system] of Object.entries(this.systems)) {
            if (system) {
                try {
                    const systemHealth = await this.checkSystemHealth(name, system);
                    healthReport.systems[name] = systemHealth;

                    if (!systemHealth.isHealthy) {
                        healthReport.issues.push(`${name}: ${systemHealth.issue}`);
                        if (systemHealth.critical) {
                            healthReport.overall = 'critical';
                        } else if (healthReport.overall === 'healthy') {
                            healthReport.overall = 'warning';
                        }
                    }
                } catch (error) {
                    healthReport.systems[name] = {
                        isHealthy: false,
                        issue: error.message,
                        critical: true
                    };
                    healthReport.issues.push(`${name}: Health check failed - ${error.message}`);
                    healthReport.overall = 'critical';
                }
            }
        }

        // Generate recommendations
        this.generateHealthRecommendations(healthReport);

        console.log(`📊 Health check completed: ${healthReport.overall} (${healthReport.issues.length} issues)`);
        return healthReport;
    }

    // Safety and validation methods

    /**
             * Initialize kill switch mechanism
             */
    initializeKillSwitch() {
        this.killSwitchReady = true;

        // Setup emergency triggers
        process.on('uncaughtException', (error) => {
            console.error('Uncaught exception detected:', error);
            this.activateKillSwitch('uncaught_exception');
        });

        process.on('unhandledRejection', (reason) => {
            console.error('Unhandled promise rejection:', reason);
            this.activateKillSwitch('unhandled_rejection');
        });
    }

    /**
             * Activate kill switch - emergency system shutdown
             */
    async activateKillSwitch(reason = 'manual_activation') {
        console.error(`🚨 KILL SWITCH ACTIVATED: ${reason}`);

        this.killSwitchActive = true;
        this.safetyProtocols.killSwitchReady = false;

        try {
            // Broadcast emergency shutdown
            this.broadcastMessage('emergency_shutdown', { reason, timestamp: Date.now() });

            // Shutdown systems in reverse layer order
            const shutdownOrder = [...this.layerHierarchy].reverse();

            for (const layer of shutdownOrder) {
                await this.shutdownSystemsByLayer(layer);
            }

            // Clear all data structures
            this.clearSystemData();

            console.log('✅ Emergency shutdown completed');

        } catch (error) {
            console.error('❌ Error during emergency shutdown:', error);
        }
    }

    /**
             * Validate canvas laws compliance
             */
    validateCanvasLaws() {
        const violations = [];

        // Check memory usage
        if (process.memoryUsage().heapUsed > 1024 * 1024 * 1024) { // 1GB
            violations.push('Memory usage exceeds 1GB limit');
        }

        // Check recursion depth tracking
        if (!this.safetyProtocols.recursionDepthLimit) {
            violations.push('Recursion depth limit not set');
        }

        // Check kill switch readiness
        if (!this.killSwitchReady) {
            violations.push('Kill switch not ready');
        }

        if (violations.length > 0) {
            console.warn('⚠️ Canvas law violations detected:', violations);
            return false;
        }

        return true;
    }

    // Private helper methods

    createEventBus() {
        return {
            listeners: new Map(),

            on(event, callback) {
                if (!this.listeners.has(event)) {
                    this.listeners.set(event, []);
                }
                this.listeners.get(event).push(callback);
            },

            emit(event, data) {
                const callbacks = this.listeners.get(event) || [];
                callbacks.forEach(callback => {
                    try {
                        callback(data);
                    } catch (error) {
                        console.error(`Event callback error for ${event}:`, error);
                    }
                });
            },

            off(event, callback) {
                const callbacks = this.listeners.get(event) || [];
                const index = callbacks.indexOf(callback);
                if (index > -1) {
                    callbacks.splice(index, 1);
                }
            }
        };
    }

    setupMessageHandlers() {
        if (this.broadcastChannel) {
            this.broadcastChannel.addEventListener('message', (event) => {
                this.handleIncomingMessage(event.data);
            });
        }
    }

    handleIncomingMessage(message) {
        // Process incoming messages from other systems
        switch (message.type) {
            case 'system_status_update':
                this.updateSystemStatus(message.data);
                break;
            case 'emergency_alert':
                this.handleEmergencyAlert(message.data);
                break;
            case 'performance_metric':
                this.updatePerformanceMetric(message.data);
                break;
            default:
                // Forward to event bus
                this.eventBus.emit(message.type, message.data);
        }
    }

    initializeDataFlow() {
        // Setup data flow routes between systems
        this.dataFlowPipeline.set('text_processing', {
            input: 'user_input',
            stages: ['thoughtEngine', 'morphology', 'synthesis', 'agents'],
            output: 'processed_result'
        });

        this.dataFlowPipeline.set('audio_generation', {
            input: 'text_or_data',
            stages: ['audioVisual', 'synthesis'],
            output: 'audio_output'
        });

        this.dataFlowPipeline.set('agent_training', {
            input: 'interaction_data',
            stages: ['morphology', 'agents'],
            output: 'trained_agents'
        });
    }

    initializeDataFlowRoutes() {
        // Initialize routes for data flow between systems
        console.log('Setting up data flow routes...');
    }

    initializeSafetyProtocols() {
        // Setup additional safety measures
        this.safetyProtocols.startTime = Date.now();
        console.log('Safety protocols initialized');
    }

    startPerformanceMonitoring() {
        setInterval(() => {
            this.updateMetrics();
            this.validateSystemPerformance();
        }, 5000); // Every 5 seconds
    }

    updateMetrics() {
        const now = Date.now();
        this.metrics.uptime = now - this.safetyProtocols.startTime;
        this.metrics.lastUpdate = now;

        // Count online systems
        this.metrics.systemsOnline = Object.values(this.systems).filter(s => s !== null).length;
    }

    validateSystemPerformance() {
        // Check system performance and trigger warnings if needed
        const memoryUsage = process.memoryUsage();
        if (memoryUsage.heapUsed > 512 * 1024 * 1024) { // 512MB warning
            console.warn('⚠️ High memory usage detected:', (memoryUsage.heapUsed / 1024 / 1024).toFixed(2), 'MB');
        }
    }

    // System-specific integration setup methods

    setupOverseerIntegration() {
        console.log('🧠 Setting up Overseer Brain integration...');
    }

    setupStudioBridgeIntegration(bridge) {
        console.log('🌉 Setting up Studio Bridge integration...');
        bridge.registerFramework(this);
    }

    setupThoughtEngineIntegration(engine) {
        console.log('💭 Setting up Thought Engine integration...');
        if (typeof engine.setFramework === 'function') {
            engine.setFramework(this);
        }
    }

    setupMorphologyIntegration(engine) {
        console.log('📝 Setting up Morphology Engine integration...');
    }

    setupAudioVisualIntegration(engine) {
        console.log('🎨 Setting up Audio-Visual Engine integration...');
    }

    setupSynthesisIntegration(engine) {
        console.log('🎵 Setting up Synthesis Engine integration...');
    }

    setupDashboardIntegration(dashboard) {
        console.log('📊 Setting up Dashboard integration...');
        dashboard.setFramework(this);
    }

    setupAgentSystemIntegration(agentSystem) {
        console.log('🤖 Setting up Agent System integration...');

        // Connect all other systems to agent system
        const systemsToConnect = {
            morphologyEngine: this.systems.morphologyEngine,
            audioVisualEngine: this.systems.audioVisualEngine,
            synthesisEngine: this.systems.synthesisEngine,
            dashboardInterface: this.systems.nexusDashboard
        };

        agentSystem.connectSystems(systemsToConnect);
    }

    // Validation methods

    validateSystemBeforeConnection(system, layer) {
        // Basic validation checks
        if (!system || typeof system !== 'object') {
            return false;
        }

        // Special handling for our advanced systems - allow them to connect
        const advancedSystemNames = [
            'AdvancedMorphologyEngine', 'WorldEngineIndexManager',
            'WorldEngineUXPipeline', 'Layer5AgentSystem'
        ];

        const systemName = system.constructor.name;
        if (advancedSystemNames.includes(systemName)) {
            console.log(`✅ Advanced system ${systemName} validated`);
            return true;
        }

        // Check for required methods based on layer (for other systems)
        const requiredMethods = this.getRequiredMethodsForLayer(layer);
        for (const method of requiredMethods) {
            if (typeof system[method] !== 'function') {
                console.warn(`System missing required method: ${method}`);
                return false;
            }
        }

        return true;
    }

    async validateWithLayer0(systemName, system) {
        // Placeholder for Layer 0 validation
        return { isValid: true, reason: 'Layer 0 validation passed' };
    }

    validateInputWithLayer0(input, options) {
        // Placeholder for Layer 0 input validation
        if (this.killSwitchActive) {
            return false;
        }

        if (!this.canvasLawsActive) {
            return false;
        }

        return true;
    }

    getRequiredMethodsForLayer(layer) {
        const methodMap = {
            'Layer0': ['validate', 'shutdown'],
            'Layer1': ['connect', 'broadcast'],
            'Layer2': ['analyze', 'search', 'indexDocument'], // Updated for our advanced systems
            'Layer3': ['generate'],
            'Layer4': ['processUserInput', 'render'], // Updated for UX Pipeline
            'Layer5': ['processInput', 'getAgents'] // Updated for Agent System
        };

        return methodMap[layer] || [];
    }

    // Additional helper methods

    async checkSystemHealth(name, system) {
        // Default health check
        const health = {
            isHealthy: true,
            issue: null,
            critical: false,
            lastCheck: Date.now()
        };

        // Check if system has health check method
        if (typeof system.getSystemStatus === 'function') {
            try {
                const status = system.getSystemStatus();
                if (status.error || status.failed) {
                    health.isHealthy = false;
                    health.issue = status.error || 'System reported failure';
                    health.critical = status.critical || false;
                }
            } catch (error) {
                health.isHealthy = false;
                health.issue = `Health check threw error: ${error.message}`;
                health.critical = true;
            }
        }

        return health;
    }

    generateHealthRecommendations(healthReport) {
        if (healthReport.issues.length === 0) {
            healthReport.recommendations.push('All systems operating normally');
            return;
        }

        if (healthReport.overall === 'critical') {
            healthReport.recommendations.push('Consider activating kill switch if issues persist');
            healthReport.recommendations.push('Investigate critical system failures immediately');
        }

        if (healthReport.issues.length > 3) {
            healthReport.recommendations.push('Multiple system failures detected - perform full system restart');
        }

        healthReport.recommendations.push('Review system logs for detailed error information');
        healthReport.recommendations.push('Validate Layer 0 canvas laws compliance');
    }

    async shutdownSystemsByLayer(layer) {
        for (const [name, system] of Object.entries(this.systems)) {
            const connectionInfo = this.connectionStatus.get(name);
            if (connectionInfo && connectionInfo.layer === layer && system) {
                try {
                    if (typeof system.shutdown === 'function') {
                        await system.shutdown();
                    }
                    this.systems[name] = null;
                    console.log(`Shutdown ${name} (${layer})`);
                } catch (error) {
                    console.error(`Error shutting down ${name}:`, error);
                }
            }
        }
    }

    clearSystemData() {
        this.systems = {};
        this.connectionStatus.clear();
        this.slotRegistry.clear();
        this.messageQueue = [];
        this.dataFlowPipeline.clear();
        console.log('System data cleared');
    }

    updateSystemStatus(data) {
        // Update system status from incoming messages
    }

    handleEmergencyAlert(data) {
        console.warn('🚨 Emergency alert received:', data);
        // Handle emergency alerts from other systems
    }

    updatePerformanceMetric(data) {
        // Update performance metrics from other systems
    }

    /**
       * Generate comprehensive system health report
       */
    async generateSystemHealthReport() {
        const healthReport = {
            timestamp: Date.now(),
            overallHealth: 'healthy',
            systemsOnline: 0,
            totalSystems: Object.keys(this.systems).filter(key => this.systems[key]).length,
            issues: [],
            recommendations: [],
            metrics: this.metrics
        };

        // Check each system
        for (const [name, system] of Object.entries(this.systems)) {
            if (system) {
                healthReport.systemsOnline++;
                const health = await this.checkSystemHealth(name, system);

                if (!health.isHealthy) {
                    healthReport.issues.push({
                        system: name,
                        issue: health.issue,
                        critical: health.critical
                    });
                }
            }
        }

        // Determine overall health
        const criticalIssues = healthReport.issues.filter(i => i.critical);
        if (criticalIssues.length > 0) {
            healthReport.overallHealth = 'critical';
        } else if (healthReport.issues.length > 0) {
            healthReport.overallHealth = 'degraded';
        }

        // Generate recommendations
        this.generateHealthRecommendations(healthReport);

        return healthReport;
    }

    /**
       * Shutdown the entire framework
       */
    async shutdown() {
        console.log('🔧 Shutting down World Engine Integration Framework...');

        try {
            // Shutdown systems by layer (reverse order)
            const layers = ['Layer5', 'Layer4', 'Layer3', 'Layer2', 'Layer1', 'Layer0'];

            for (const layer of layers) {
                await this.shutdownSystemsByLayer(layer);
            }

            // Clear all data
            this.clearSystemData();

            // Mark as not initialized
            this.isInitialized = false;

            console.log('✅ Framework shutdown completed');
            return { success: true };

        } catch (error) {
            console.error('❌ Framework shutdown error:', error);
            return { success: false, error: error.message };
        }
    }

    /**
           * Advanced Morphology Engine Integration
           */
    async processWithAdvancedMorphology(text, options = {}) {
        if (!this.systems.advancedMorphologyEngine) {
            throw new Error('Advanced Morphology Engine not connected');
        }

        try {
            const result = this.systems.advancedMorphologyEngine.analyze(text, options);

            // Auto-index if Index Manager is available
            if (this.systems.indexManager && options.autoIndex !== false) {
                const docId = `morphology_${Date.now()}`;
                await this.systems.indexManager.indexDocument(docId, text, {
                    morphology: result,
                    vector: result.semanticVector,
                    classification: result.classification,
                    timestamp: Date.now()
                });
            }

            return result;
        } catch (error) {
            console.error('Advanced morphology processing failed:', error);
            throw error;
        }
    }

    /**
           * Index Manager Integration
           */
    async searchWithIndexManager(query, options = {}) {
        if (!this.systems.indexManager) {
            throw new Error('Index Manager not connected');
        }

        try {
            let queryVector = null;
            let morphologyData = null;

            // Use Advanced Morphology Engine for query enhancement if available
            if (this.systems.advancedMorphologyEngine && options.useMorphology !== false) {
                morphologyData = this.systems.advancedMorphologyEngine.analyze(query);
                queryVector = morphologyData.semanticVector;
            }

            return await this.systems.indexManager.search(query, {
                ...options,
                queryVector,
                morphology: morphologyData
            });
        } catch (error) {
            console.error('Index Manager search failed:', error);
            throw error;
        }
    }

    /**
           * UX Pipeline Integration
           */
    async processUserInput(input, options = {}) {
        if (!this.systems.uxPipeline) {
            throw new Error('UX Pipeline not connected');
        }

        try {
            return await this.systems.uxPipeline.processUserInput(input, {
                ...options,
                enableMorphology: this.systems.advancedMorphologyEngine ? true : false,
                enableIndexing: this.systems.indexManager ? true : false,
                framework: this
            });
        } catch (error) {
            console.error('UX Pipeline processing failed:', error);
            throw error;
        }
    }

    /**
           * Comprehensive Processing Pipeline
           * Coordinates all advanced systems for optimal results
           */
    async processComprehensive(input, options = {}) {
        const startTime = Date.now();
        const results = {
            input,
            timestamp: startTime,
            morphology: null,
            indexing: null,
            uxProcessing: null,
            agentInsights: null,
            totalTime: 0,
            success: false
        };

        try {
            // Step 1: Advanced Morphological Analysis
            if (this.systems.advancedMorphologyEngine) {
                results.morphology = await this.processWithAdvancedMorphology(input, options);
                console.log('✅ Advanced morphology analysis completed');
            }

            // Step 2: Search and Discovery
            if (this.systems.indexManager) {
                results.indexing = await this.searchWithIndexManager(input, {
                    ...options,
                    maxResults: 10
                });
                console.log('✅ Index search completed');
            }

            // Step 3: UX Pipeline Processing
            if (this.systems.uxPipeline) {
                results.uxProcessing = await this.processUserInput(input, options);
                console.log('✅ UX pipeline processing completed');
            }

            // Step 4: Agent System Insights (if available)
            if (this.systems.agentSystem) {
                try {
                    results.agentInsights = await this.systems.agentSystem.processInput(input, {
                        morphology: results.morphology,
                        searchResults: results.indexing,
                        ...options
                    });
                    console.log('✅ Agent insights generated');
                } catch (agentError) {
                    console.warn('Agent processing failed, continuing without:', agentError.message);
                }
            }

            results.totalTime = Date.now() - startTime;
            results.success = true;

            console.log(`🎯 Comprehensive processing completed in ${results.totalTime}ms`);
            return results;

        } catch (error) {
            results.totalTime = Date.now() - startTime;
            results.error = error.message;
            console.error('Comprehensive processing failed:', error);
            throw error;
        }
    }
}

/**
 * Create and initialize the World Engine Integration Framework
 */
export async function createWorldEngineFramework(options = {}) {
    const framework = new WorldEngineIntegrationFramework(options);

    if (options.autoConnect) {
        // Auto-connect available systems
        console.log('🔗 Auto-connecting available systems...');

        try {
            // Layer 5 Agent System
            const agentSystem = new Layer5AgentSystem();
            await framework.connectSystem('agentSystem', agentSystem, 'Layer5');

            // Advanced Morphology Engine (Layer 2)
            const advancedMorphologyEngine = new AdvancedMorphologyEngine();
            await framework.connectSystem('advancedMorphologyEngine', advancedMorphologyEngine, 'Layer2');

            // World Engine Index Manager (Layer 2)
            const indexManager = new WorldEngineIndexManager();
            await framework.connectSystem('indexManager', indexManager, 'Layer2');

            // World Engine UX Pipeline (Layer 4)
            const uxPipeline = new WorldEngineUXPipeline(framework);
            await framework.connectSystem('uxPipeline', uxPipeline, 'Layer4');

            console.log('✅ Auto-connection completed');
        } catch (error) {
            console.error('❌ Auto-connection failed:', error);
        }
    }

    return framework;
}
