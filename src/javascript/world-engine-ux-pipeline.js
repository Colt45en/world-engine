/**
 * World Engine UX Pipeline Integration
 * Combines advanced morphology, indexing, and agent systems for seamless user experience
 * Implements scalable HTML pipeline with proper error handling and performance monitoring
 */

import { AdvancedMorphologyEngine } from './models/advanced-morphology-engine.js';
import { WorldEngineIndexManager } from './models/world-engine-index-manager.js';

export class WorldEngineUXPipeline {
    constructor(framework, options = {}) {
        this.framework = framework;
        this.systemId = `WEUXP_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
        this.isInitialized = false;

        // Core UX systems
        this.advancedMorphology = null;
        this.indexManager = null;

        // UX configuration
        this.config = {
            enableAdvancedMorphology: options.enableAdvancedMorphology !== false,
            enableIndexing: options.enableIndexing !== false,
            enableRealTimeProcessing: options.enableRealTimeProcessing !== false,
            enablePerformanceMonitoring: options.enablePerformanceMonitoring !== false,
            maxProcessingTime: options.maxProcessingTime || 10000,
            vectorDimension: options.vectorDimension || 128
        };

        // Performance tracking
        this.performanceMetrics = {
            totalRequests: 0,
            averageProcessingTime: 0,
            successRate: 0,
            errorRate: 0,
            systemLoadAverage: 0,
            lastUpdate: Date.now()
        };

        // Error handling
        this.errorHandler = new UXErrorHandler();
        this.circuitBreaker = new CircuitBreaker();

        // UX state management
        this.userSessions = new Map();
        this.processingQueue = [];
        this.isProcessing = false;

        // HTML pipeline components
        this.htmlRenderer = new HTMLPipelineRenderer();
        this.responseCache = new Map();
        this.maxCacheSize = options.maxCacheSize || 500;

        this.init();
    }

    async init() {
        console.log('🎨 Initializing World Engine UX Pipeline');

        try {
            // Initialize advanced morphology engine
            if (this.config.enableAdvancedMorphology) {
                this.advancedMorphology = new AdvancedMorphologyEngine({
                    vectorDimension: this.config.vectorDimension,
                    maxCacheSize: 5000
                });
                console.log('✅ Advanced Morphology Engine connected');
            }

            // Initialize index manager
            if (this.config.enableIndexing) {
                this.indexManager = new WorldEngineIndexManager({
                    vectorDimension: this.config.vectorDimension,
                    enableCaching: true,
                    maxCacheSize: 1000
                });
                console.log('✅ Index Manager connected');
            }

            // Connect to framework systems
            await this.connectFrameworkSystems();

            // Start performance monitoring
            if (this.config.enablePerformanceMonitoring) {
                this.startPerformanceMonitoring();
            }

            // Initialize HTML renderer
            this.initializeHTMLRenderer();

            this.isInitialized = true;
            console.log('✅ UX Pipeline ready');

        } catch (error) {
            console.error('❌ UX Pipeline initialization failed:', error);
            throw error;
        }
    }

    async connectFrameworkSystems() {
        if (!this.framework) {
            console.warn('No framework provided to UX Pipeline');
            return;
        }

        // Connect advanced morphology to framework systems
        if (this.advancedMorphology && this.framework.systems) {
            // Replace existing morphology engine with advanced version
            this.framework.systems.morphologyEngine = this.advancedMorphology;

            // Connect to agent system
            if (this.framework.systems.agentSystem) {
                this.framework.systems.agentSystem.morphologyEngine = this.advancedMorphology;
                console.log('🔗 Advanced morphology connected to agent system');
            }
        }

        console.log('🔗 Framework systems connected');
    }

    /**
       * Main UX processing endpoint - handles all user interactions
       */
    async processUserInput(userInput, options = {}) {
        const startTime = performance.now();
        const sessionId = options.sessionId || this.generateSessionId();

        try {
            // Validate input
            if (!this.validateInput(userInput)) {
                throw new Error('Invalid user input');
            }

            // Check circuit breaker
            if (!this.circuitBreaker.canProcess()) {
                throw new Error('System temporarily unavailable');
            }

            // Create processing context
            const context = this.createProcessingContext(userInput, sessionId, options);

            // Check cache first
            const cacheKey = this.generateCacheKey(userInput, options);
            if (this.responseCache.has(cacheKey)) {
                const cachedResponse = this.responseCache.get(cacheKey);
                return this.enhanceResponse(cachedResponse, context);
            }

            // Process through pipeline
            const result = await this.executeProcessingPipeline(context);

            // Cache successful results
            if (result.success) {
                this.cacheResponse(cacheKey, result);
            }

            // Update metrics
            this.updatePerformanceMetrics(startTime, true);

            // Generate UX response
            return this.generateUXResponse(result, context);

        } catch (error) {
            console.error('UX Pipeline processing error:', error);

            // Handle error gracefully
            const errorResponse = this.errorHandler.handleError(error, userInput, options);
            this.updatePerformanceMetrics(startTime, false);

            return errorResponse;
        }
    }

    async executeProcessingPipeline(context) {
        const pipeline = [];
        const results = {
            morphologyAnalysis: null,
            frameworkProcessing: null,
            indexSearch: null,
            agentInteraction: null,
            success: true,
            metadata: {
                startTime: context.startTime,
                sessionId: context.sessionId,
                components: []
            }
        };

        // Step 1: Advanced morphological analysis
        if (this.advancedMorphology && context.options.enableMorphology !== false) {
            pipeline.push('morphology');
            const morphStart = performance.now();

            results.morphologyAnalysis = this.advancedMorphology.analyze(context.input);
            results.metadata.components.push({
                name: 'morphology',
                processingTime: performance.now() - morphStart,
                complexity: results.morphologyAnalysis.complexity
            });
        }

        // Step 2: Framework processing with enhanced morphology
        if (this.framework && context.options.enableFramework !== false) {
            pipeline.push('framework');
            const frameworkStart = performance.now();

            const frameworkOptions = {
                generateVisuals: context.options.generateVisuals,
                generateAudio: context.options.generateAudio,
                useAgents: context.options.useAgents,
                morphologyData: results.morphologyAnalysis
            };

            results.frameworkProcessing = await this.framework.processText(
                context.input,
                frameworkOptions
            );

            results.metadata.components.push({
                name: 'framework',
                processingTime: performance.now() - frameworkStart,
                systemsUsed: results.frameworkProcessing?.metadata?.systemsUsed || []
            });
        }

        // Step 3: Index search and document retrieval
        if (this.indexManager && context.options.enableSearch !== false) {
            pipeline.push('indexing');
            const indexStart = performance.now();

            // Index current input if not already indexed
            const docId = `user_${context.sessionId}_${Date.now()}`;
            await this.indexManager.indexDocument(docId, context.input, {
                vector: results.morphologyAnalysis?.semanticVector,
                morphology: results.morphologyAnalysis,
                classification: results.morphologyAnalysis?.classification,
                metadata: { sessionId: context.sessionId, timestamp: Date.now() }
            });

            // Perform search for related content
            const searchResults = await this.indexManager.search(context.input, {
                queryVector: results.morphologyAnalysis?.semanticVector,
                morphology: results.morphologyAnalysis,
                classification: results.morphologyAnalysis?.classification,
                maxResults: context.options.maxSearchResults || 10
            });

            results.indexSearch = {
                indexed: true,
                documentId: docId,
                searchResults: searchResults,
                totalResults: searchResults.length
            };

            results.metadata.components.push({
                name: 'indexing',
                processingTime: performance.now() - indexStart,
                documentsFound: searchResults.length
            });
        }

        // Step 4: Enhanced agent interaction
        if (this.framework?.systems?.agentSystem && context.options.useAgents !== false) {
            pipeline.push('agents');
            const agentStart = performance.now();

            const agentSystem = this.framework.systems.agentSystem;

            // Create specialized environment based on morphology
            let environment = null;
            if (results.morphologyAnalysis?.classification) {
                const agentType = this.mapClassificationToAgentType(
                    results.morphologyAnalysis.classification.type
                );
                environment = agentSystem.createEnvironment(context.input, agentType);
                agentSystem.spawnAgentBatch(3, environment);
            }

            // Facilitate interactions with enhanced context
            const activeAgents = Array.from(agentSystem.activeAgents.values()).slice(0, 5);
            const agentResponses = [];

            for (const agent of activeAgents) {
                const interaction = agentSystem.facilitateHumanAgentInteraction(
                    agent.id,
                    context.input
                );

                if (interaction) {
                    agentResponses.push({
                        agentId: agent.id,
                        agentType: agent.type,
                        success: interaction,
                        morphologyInfluence: this.calculateMorphologyInfluence(
                            results.morphologyAnalysis,
                            agent
                        )
                    });
                }
            }

            results.agentInteraction = {
                environment: environment ? {
                    id: environment.id,
                    agentType: environment.preferredAgentType,
                    agentCount: environment.agents.length
                } : null,
                interactions: agentResponses,
                totalAgents: activeAgents.length
            };

            results.metadata.components.push({
                name: 'agents',
                processingTime: performance.now() - agentStart,
                agentsInteracted: agentResponses.length
            });
        }

        // Calculate total pipeline time
        results.metadata.totalPipelineTime = performance.now() - context.startTime;
        results.metadata.pipeline = pipeline;

        return results;
    }

    generateUXResponse(results, context) {
        const response = {
            success: results.success,
            sessionId: context.sessionId,
            timestamp: Date.now(),
            processing: {
                pipeline: results.metadata.pipeline,
                totalTime: results.metadata.totalPipelineTime,
                components: results.metadata.components
            }
        };

        // Primary content from framework
        if (results.frameworkProcessing) {
            response.primaryResponse = results.frameworkProcessing.finalOutput?.primaryResponse || 'Processing completed';
            response.enrichments = results.frameworkProcessing.finalOutput?.enrichments || {};
            response.confidence = results.frameworkProcessing.finalOutput?.confidence || 0.5;
        }

        // Enhanced morphological insights
        if (results.morphologyAnalysis) {
            response.morphology = {
                complexity: results.morphologyAnalysis.complexity,
                classification: results.morphologyAnalysis.classification,
                morphemeBreakdown: results.morphologyAnalysis.morphemes?.map(m =>
                    `${m.type}:${m.text}`
                ).join(' + '),
                semanticClass: results.morphologyAnalysis.classification?.semanticClass
            };
        }

        // Search and discovery
        if (results.indexSearch) {
            response.discovery = {
                indexed: results.indexSearch.indexed,
                relatedContent: results.indexSearch.searchResults.slice(0, 3).map(r => ({
                    score: r.score.toFixed(3),
                    preview: r.document?.content?.slice(0, 100) + '...',
                    metadata: r.metadata
                })),
                totalMatches: results.indexSearch.totalResults
            };
        }

        // Agent insights
        if (results.agentInteraction) {
            response.agentInsights = {
                environmentCreated: !!results.agentInteraction.environment,
                agentType: results.agentInteraction.environment?.agentType,
                successfulInteractions: results.agentInteraction.interactions?.filter(i => i.success).length || 0,
                totalAgents: results.agentInteraction.totalAgents
            };
        }

        // UX enhancements
        response.uiElements = this.generateUIElements(results, context);
        response.suggestions = this.generateSuggestions(results, context);
        response.htmlPreview = this.htmlRenderer.generatePreview(response);

        return response;
    }

    generateUIElements(results, context) {
        const elements = [];

        // Morphology visualization
        if (results.morphologyAnalysis) {
            elements.push({
                type: 'morphology-breakdown',
                data: {
                    morphemes: results.morphologyAnalysis.morphemes,
                    complexity: results.morphologyAnalysis.complexity,
                    positions: results.morphologyAnalysis.positions
                }
            });
        }

        // Agent interaction panel
        if (results.agentInteraction) {
            elements.push({
                type: 'agent-panel',
                data: {
                    environment: results.agentInteraction.environment,
                    interactions: results.agentInteraction.interactions
                }
            });
        }

        // Related content discovery
        if (results.indexSearch?.searchResults?.length > 0) {
            elements.push({
                type: 'discovery-panel',
                data: {
                    results: results.indexSearch.searchResults.slice(0, 5)
                }
            });
        }

        return elements;
    }

    generateSuggestions(results, context) {
        const suggestions = [];

        // Morphology-based suggestions
        if (results.morphologyAnalysis) {
            const classification = results.morphologyAnalysis.classification;

            if (classification.type === 'Action') {
                suggestions.push('Try exploring the process or steps involved');
                suggestions.push('Consider the outcome or result of this action');
            } else if (classification.type === 'Entity') {
                suggestions.push('Explore the properties or characteristics');
                suggestions.push('Look into related entities or components');
            }
        }

        // Agent-based suggestions
        if (results.agentInteraction?.environment) {
            suggestions.push(`Created ${results.agentInteraction.environment.agentType} environment for exploration`);
            suggestions.push('Agents are learning from your interaction');
        }

        // Search-based suggestions
        if (results.indexSearch?.totalResults > 0) {
            suggestions.push(`Found ${results.indexSearch.totalResults} related items to explore`);
            suggestions.push('Try refining your query for more specific results');
        }

        return suggestions;
    }

    // Helper methods

    validateInput(input) {
        return input && typeof input === 'string' && input.trim().length > 0;
    }

    generateSessionId() {
        return `session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    }

    createProcessingContext(input, sessionId, options) {
        return {
            input: input,
            sessionId: sessionId,
            options: options,
            startTime: performance.now(),
            userSession: this.getUserSession(sessionId)
        };
    }

    getUserSession(sessionId) {
        if (!this.userSessions.has(sessionId)) {
            this.userSessions.set(sessionId, {
                id: sessionId,
                created: Date.now(),
                interactions: 0,
                preferences: {},
                history: []
            });
        }

        const session = this.userSessions.get(sessionId);
        session.interactions++;
        session.lastActivity = Date.now();

        return session;
    }

    generateCacheKey(input, options) {
        return JSON.stringify({
            input: input.toLowerCase().trim(),
            options: {
                enableMorphology: options.enableMorphology,
                enableSearch: options.enableSearch,
                useAgents: options.useAgents,
                generateVisuals: options.generateVisuals,
                generateAudio: options.generateAudio
            }
        });
    }

    cacheResponse(key, response) {
        if (this.responseCache.size >= this.maxCacheSize) {
            const firstKey = this.responseCache.keys().next().value;
            this.responseCache.delete(firstKey);
        }

        this.responseCache.set(key, {
            ...response,
            cached: true,
            cacheTime: Date.now()
        });
    }

    enhanceResponse(cachedResponse, context) {
        return {
            ...cachedResponse,
            sessionId: context.sessionId,
            fromCache: true,
            cacheAge: Date.now() - cachedResponse.cacheTime
        };
    }

    mapClassificationToAgentType(classificationType) {
        const mapping = {
            'Action': 'ADAPTIVE',
            'Entity': 'ANALYTICAL',
            'Property': 'ANALYTICAL',
            'State': 'SOCIAL',
            'Structure': 'COOPERATIVE',
            'Process': 'CREATIVE'
        };

        return mapping[classificationType] || 'HYBRID';
    }

    calculateMorphologyInfluence(morphology, agent) {
        if (!morphology || !agent) return 0;

        let influence = 0;

        // Complexity influence
        influence += morphology.complexity * 0.1;

        // Classification alignment
        if (morphology.classification) {
            const agentTypeAlignment = {
                'COOPERATIVE': ['Structure', 'Entity'],
                'SOCIAL': ['State', 'Property'],
                'ADAPTIVE': ['Action', 'Process'],
                'ANALYTICAL': ['Entity', 'Property'],
                'CREATIVE': ['Process', 'Action'],
                'HYBRID': ['General']
            };

            const alignedTypes = agentTypeAlignment[agent.type] || [];
            if (alignedTypes.includes(morphology.classification.type)) {
                influence += 0.3;
            }
        }

        return Math.min(influence, 1.0);
    }

    updatePerformanceMetrics(startTime, success) {
        const processingTime = performance.now() - startTime;

        this.performanceMetrics.totalRequests++;
        this.performanceMetrics.averageProcessingTime =
            (this.performanceMetrics.averageProcessingTime * 0.9) + (processingTime * 0.1);

        if (success) {
            this.performanceMetrics.successRate =
                (this.performanceMetrics.successRate * 0.95) + (1.0 * 0.05);
        } else {
            this.performanceMetrics.errorRate =
                (this.performanceMetrics.errorRate * 0.95) + (1.0 * 0.05);
        }

        this.performanceMetrics.lastUpdate = Date.now();
    }

    startPerformanceMonitoring() {
        setInterval(() => {
            const status = this.getSystemStatus();

            // Log performance warnings
            if (status.performance.averageProcessingTime > 5000) {
                console.warn('⚠️ High average processing time:', status.performance.averageProcessingTime.toFixed(2), 'ms');
            }

            if (status.performance.errorRate > 0.1) {
                console.warn('⚠️ High error rate:', (status.performance.errorRate * 100).toFixed(2), '%');
            }

        }, 30000); // Check every 30 seconds
    }

    initializeHTMLRenderer() {
        this.htmlRenderer.initialize({
            enableMorphologyVisualization: true,
            enableAgentPanels: true,
            enableDiscoveryPanels: true,
            theme: 'world-engine'
        });
    }

    // System integration methods

    getSystemStatus() {
        return {
            isInitialized: this.isInitialized,
            systemId: this.systemId,
            activeSessions: this.userSessions.size,
            cacheSize: this.responseCache.size,
            performance: { ...this.performanceMetrics },
            components: {
                advancedMorphology: this.advancedMorphology?.getSystemStatus() || null,
                indexManager: this.indexManager?.getSystemStatus() || null,
                framework: this.framework?.getSystemStatus() || null
            },
            circuitBreaker: this.circuitBreaker.getStatus()
        };
    }

    onSystemMessage(message) {
        switch (message.type) {
            case 'clear_cache':
                this.responseCache.clear();
                break;
            case 'update_config':
                this.updateConfig(message.data);
                break;
            case 'performance_report':
                return this.getSystemStatus();
        }
    }

    updateConfig(newConfig) {
        this.config = { ...this.config, ...newConfig };
        console.log('UX Pipeline configuration updated');
    }
}

// Error handling system
class UXErrorHandler {
    handleError(error, userInput, options) {
        const errorType = this.classifyError(error);

        return {
            success: false,
            error: {
                type: errorType,
                message: this.getUserFriendlyMessage(errorType),
                details: error.message,
                timestamp: Date.now()
            },
            fallback: this.generateFallbackResponse(userInput, errorType),
            suggestions: this.generateErrorSuggestions(errorType)
        };
    }

    classifyError(error) {
        if (error.message.includes('timeout')) return 'timeout';
        if (error.message.includes('validation')) return 'validation';
        if (error.message.includes('unavailable')) return 'service_unavailable';
        return 'general';
    }

    getUserFriendlyMessage(errorType) {
        const messages = {
            timeout: 'Processing is taking longer than expected. Please try again.',
            validation: 'There seems to be an issue with your input. Please check and try again.',
            service_unavailable: 'Some services are temporarily unavailable. Please try again in a moment.',
            general: 'Something went wrong. Please try again.'
        };

        return messages[errorType] || messages.general;
    }

    generateFallbackResponse(userInput, errorType) {
        return {
            message: `I received your input: "${userInput}" but encountered a ${errorType} error.`,
            suggestion: 'You can try rephrasing your request or waiting a moment before trying again.'
        };
    }

    generateErrorSuggestions(errorType) {
        const suggestions = {
            timeout: ['Try a shorter or simpler request', 'Check your internet connection'],
            validation: ['Make sure your input contains text', 'Try rephrasing your request'],
            service_unavailable: ['Wait a moment and try again', 'Try a different type of request'],
            general: ['Refresh the page and try again', 'Contact support if the problem persists']
        };

        return suggestions[errorType] || suggestions.general;
    }
}

// Circuit breaker for system protection
class CircuitBreaker {
    constructor() {
        this.failureCount = 0;
        this.failureThreshold = 5;
        this.timeout = 30000; // 30 seconds
        this.state = 'CLOSED'; // CLOSED, OPEN, HALF_OPEN
        this.nextAttempt = Date.now();
    }

    canProcess() {
        if (this.state === 'CLOSED') {
            return true;
        }

        if (this.state === 'OPEN') {
            if (Date.now() > this.nextAttempt) {
                this.state = 'HALF_OPEN';
                return true;
            }
            return false;
        }

        // HALF_OPEN state
        return true;
    }

    recordSuccess() {
        this.failureCount = 0;
        this.state = 'CLOSED';
    }

    recordFailure() {
        this.failureCount++;

        if (this.failureCount >= this.failureThreshold) {
            this.state = 'OPEN';
            this.nextAttempt = Date.now() + this.timeout;
        }
    }

    getStatus() {
        return {
            state: this.state,
            failureCount: this.failureCount,
            nextAttempt: this.nextAttempt
        };
    }
}

// HTML Pipeline Renderer
class HTMLPipelineRenderer {
    constructor() {
        this.templates = new Map();
        this.isInitialized = false;
    }

    initialize(options) {
        this.loadTemplates();
        this.isInitialized = true;
    }

    loadTemplates() {
        // Load HTML templates for different UI components
        this.templates.set('morphology-breakdown', this.createMorphologyTemplate());
        this.templates.set('agent-panel', this.createAgentPanelTemplate());
        this.templates.set('discovery-panel', this.createDiscoveryPanelTemplate());
    }

    generatePreview(response) {
        if (!this.isInitialized) return '';

        const elements = response.uiElements || [];
        const htmlParts = [];

        for (const element of elements) {
            const template = this.templates.get(element.type);
            if (template) {
                htmlParts.push(template(element.data));
            }
        }

        return htmlParts.join('\n');
    }

    createMorphologyTemplate() {
        return (data) => `
      <div class="morphology-breakdown">
        <h3>Word Structure (Complexity: ${data.complexity})</h3>
        <div class="morpheme-chain">
          ${data.morphemes.map(m => `
            <span class="morpheme morpheme-${m.type}" title="${m.type}: ${m.text}">
              ${m.text}
            </span>
          `).join(' + ')}
        </div>
      </div>
    `;
    }

    createAgentPanelTemplate() {
        return (data) => `
      <div class="agent-panel">
        <h3>Agent Interactions</h3>
        ${data.environment ? `
          <div class="environment-info">
            <strong>Environment:</strong> ${data.environment.agentType}
            (${data.environment.agentCount} agents)
          </div>
        ` : ''}
        <div class="interactions">
          ${data.interactions.map(i => `
            <div class="interaction ${i.success ? 'success' : 'failed'}">
              Agent ${i.agentId} (${i.agentType}): ${i.success ? '✓' : '✗'}
            </div>
          `).join('')}
        </div>
      </div>
    `;
    }

    createDiscoveryPanelTemplate() {
        return (data) => `
      <div class="discovery-panel">
        <h3>Related Content</h3>
        <div class="discovery-results">
          ${data.results.map(r => `
            <div class="discovery-item">
              <div class="score">Score: ${r.score.toFixed(3)}</div>
              <div class="preview">${r.document?.content?.slice(0, 100)}...</div>
            </div>
          `).join('')}
        </div>
      </div>
    `;
    }
}

export function createWorldEngineUXPipeline(framework, options = {}) {
    return new WorldEngineUXPipeline(framework, options);
}
