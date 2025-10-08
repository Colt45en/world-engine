/**
 * World Engine Studio Integration - Complete System
 * Combines Chat + Engine + Recorder with unified orchestration
 * Eliminates standalone components in favor of integrated architecture
 */

import { StudioOrchestrator } from './orchestrator.js';
import { ChatController } from './controllers/chat-controller.js';
import { EngineController } from './controllers/engine-controller.js';
import { RecorderController } from './controllers/recorder-controller.js';
import { WorldEngine } from './models/world-engine.js';

class WorldEngineStudio {
    constructor(options = {}) {
        this.options = {
            maxConcurrentRuns: 3,
            enginePoolSize: 2,
            enableRecording: true,
            encoder: 'rgcn', // 'rgcn' | 'gat' | 'hybrid'
            ...options
        };

        this.components = new Map();
        this.bus = null;
        this.orchestrator = null;
        this.initialized = false;

        this.setupEventBus();
        this.initializeComponents();
    }

    setupEventBus() {
        // Simple event bus implementation
        this.bus = {
            listeners: new Map(),

            sendBus: (msg) => {
                const type = msg.type;
                if (this.listeners.has(type)) {
                    this.listeners.get(type).forEach(callback => {
                        try {
                            callback(msg);
                        } catch (error) {
                            console.error(`Event handler error for ${type}:`, error);
                        }
                    });
                }

                // Also trigger wildcard listeners
                if (this.listeners.has('*')) {
                    this.listeners.get('*').forEach(callback => {
                        try {
                            callback(msg);
                        } catch (error) {
                            console.error('Wildcard event handler error:', error);
                        }
                    });
                }
            },

            onBus: (typeOrCallback, callback) => {
                if (typeof typeOrCallback === 'function') {
                    // Wildcard listener
                    if (!this.listeners.has('*')) {
                        this.listeners.set('*', []);
                    }
                    this.listeners.get('*').push(typeOrCallback);
                } else {
                    // Specific type listener
                    const type = typeOrCallback;
                    if (!this.listeners.has(type)) {
                        this.listeners.set(type, []);
                    }
                    this.listeners.get(type).push(callback);
                }
            }
        };
    }

    async initializeComponents() {
        try {
            console.log('Initializing World Engine Studio...');

            // Create WorldEngine model
            const worldEngine = new WorldEngine({
                encoder: this.options.encoder,
                hiddenSize: 512,
                numLayers: 3
            });

            // Create orchestrator
            this.orchestrator = new StudioOrchestrator({
                maxConcurrentRuns: this.options.maxConcurrentRuns,
                retryAttempts: 3,
                backpressureLimit: 10
            });
            this.orchestrator.bus = this.bus;

            // Create controllers
            const chatController = new ChatController({
                enableTranscripts: true,
                maxHistory: 100
            });
            chatController.bus = this.bus;

            const engineController = new EngineController({
                poolSize: this.options.enginePoolSize,
                resultTimeout: 10000
            });
            engineController.bus = this.bus;

            let recorderController = null;
            if (this.options.enableRecording) {
                recorderController = new RecorderController({
                    video: true,
                    audio: true,
                    maxDuration: 120000
                });
                recorderController.bus = this.bus;
            }

            // Store components
            this.components.set('orchestrator', this.orchestrator);
            this.components.set('chat', chatController);
            this.components.set('engine', engineController);
            this.components.set('worldEngine', worldEngine);

            if (recorderController) {
                this.components.set('recorder', recorderController);
            }

            // Setup cross-component communication
            this.setupIntegration();

            this.initialized = true;
            console.log('World Engine Studio initialized successfully');

            // Notify ready
            this.bus.sendBus({
                type: 'studio.ready',
                components: Array.from(this.components.keys()),
                options: this.options
            });

        } catch (error) {
            console.error('Failed to initialize World Engine Studio:', error);
            throw error;
        }
    }

    setupIntegration() {
        // Chat → Engine integration
        this.bus.onBus('chat.command', (msg) => {
            if (msg.command === 'run' && msg.text) {
                // Route to orchestrator for smart execution
                this.bus.sendBus({
                    type: 'orchestrator.execute',
                    text: msg.text,
                    priority: msg.priority || 'normal',
                    metadata: {
                        source: 'chat',
                        timestamp: Date.now(),
                        sessionId: msg.sessionId
                    }
                });
            }
        });

        // Engine results → Chat transcript
        this.bus.onBus('eng.result', (msg) => {
            this.bus.sendBus({
                type: 'chat.transcript',
                role: 'assistant',
                content: this.formatEngineResult(msg),
                metadata: {
                    runId: msg.runId,
                    engineId: msg.engineId,
                    processingTime: Date.now() - (msg.startTime || 0)
                }
            });
        });

        // Error handling integration
        this.bus.onBus('eng.error', (msg) => {
            this.bus.sendBus({
                type: 'chat.transcript',
                role: 'error',
                content: `Engine error: ${msg.error}`,
                metadata: {
                    runId: msg.runId,
                    error: true
                }
            });
        });

        // Recording integration
        if (this.components.has('recorder')) {
            this.bus.onBus('chat.record', (msg) => {
                this.bus.sendBus({
                    type: 'rec.start',
                    sessionId: msg.sessionId || 'default'
                });
            });

            this.bus.onBus('rec.complete', (msg) => {
                this.bus.sendBus({
                    type: 'chat.transcript',
                    role: 'system',
                    content: `Recording complete: ${msg.size} bytes, ${msg.duration}ms`,
                    metadata: {
                        recording: {
                            url: msg.url,
                            mimeType: msg.mimeType,
                            duration: msg.duration
                        }
                    }
                });
            });
        }

        // WorldEngine model integration
        this.bus.onBus('orchestrator.process', async (msg) => {
            try {
                const worldEngine = this.components.get('worldEngine');
                const result = await worldEngine.processInput(msg.text, {
                    includeRaw: false,
                    encoder: msg.encoder || this.options.encoder
                });

                // Send processed result to engine
                this.bus.sendBus({
                    type: 'eng.run',
                    text: msg.text,
                    runId: msg.runId,
                    worldEngineResult: result,
                    metadata: msg.metadata
                });
            } catch (error) {
                this.bus.sendBus({
                    type: 'orchestrator.error',
                    runId: msg.runId,
                    error: error.message
                });
            }
        });
    }

    formatEngineResult(result) {
        if (!result.outcome) return 'No result';

        if (result.outcome.items) {
            return `Found ${result.outcome.items.length} items`;
        } else if (result.outcome.result) {
            return result.outcome.result;
        } else if (result.outcome.count !== undefined) {
            return `Count: ${result.outcome.count}`;
        }

        return JSON.stringify(result.outcome, null, 2);
    }

    // Public API methods
    async sendMessage(text, options = {}) {
        if (!this.initialized) {
            throw new Error('Studio not initialized');
        }

        const sessionId = options.sessionId || 'default';

        this.bus.sendBus({
            type: 'chat.message',
            text,
            sessionId,
            timestamp: Date.now(),
            ...options
        });
    }

    async switchEncoder(encoderName) {
        const worldEngine = this.components.get('worldEngine');
        if (worldEngine) {
            worldEngine.switchEncoder(encoderName);
            this.options.encoder = encoderName;

            this.bus.sendBus({
                type: 'studio.encoder-changed',
                encoder: encoderName
            });
        }
    }

    async startRecording(sessionId = 'default') {
        if (this.components.has('recorder')) {
            this.bus.sendBus({
                type: 'rec.start',
                sessionId
            });
        } else {
            throw new Error('Recording not enabled');
        }
    }

    async stopRecording(sessionId = 'default') {
        if (this.components.has('recorder')) {
            this.bus.sendBus({
                type: 'rec.stop',
                sessionId
            });
        }
    }

    getStatus() {
        const status = {
            initialized: this.initialized,
            components: {},
            options: this.options,
            bus: {
                listenerTypes: Array.from(this.bus.listeners.keys()),
                totalListeners: Array.from(this.bus.listeners.values())
                    .reduce((sum, arr) => sum + arr.length, 0)
            }
        };

        // Get component statuses
        for (const [name, component] of this.components) {
            if (component.getStatus) {
                status.components[name] = component.getStatus();
            } else {
                status.components[name] = { available: true };
            }
        }

        return status;
    }

    // Event subscription helpers
    onMessage(callback) {
        this.bus.onBus('chat.transcript', callback);
    }

    onEngineResult(callback) {
        this.bus.onBus('eng.result', callback);
    }

    onRecordingComplete(callback) {
        this.bus.onBus('rec.complete', callback);
    }

    onError(callback) {
        this.bus.onBus((msg) => {
            if (msg.type.endsWith('.error')) {
                callback(msg);
            }
        });
    }

    // Cleanup
    destroy() {
        // Stop all components
        if (this.components.has('recorder')) {
            this.components.get('recorder').cleanup();
        }

        // Clear event listeners
        this.bus.listeners.clear();

        // Clear references
        this.components.clear();
        this.initialized = false;

        console.log('World Engine Studio destroyed');
    }
}

// Global instance management
let globalStudio = null;

export function createStudio(options = {}) {
    if (globalStudio) {
        globalStudio.destroy();
    }

    globalStudio = new WorldEngineStudio(options);
    return globalStudio;
}

export function getStudio() {
    return globalStudio;
}

export { WorldEngineStudio };

// Auto-initialize if in browser
if (typeof window !== 'undefined') {
    window.WorldEngineStudio = WorldEngineStudio;
    window.createStudio = createStudio;
    window.getStudio = getStudio;

    // Auto-create default instance
    document.addEventListener('DOMContentLoaded', () => {
        const defaultStudio = createStudio({
            maxConcurrentRuns: 2,
            enginePoolSize: 1,
            enableRecording: true,
            encoder: 'rgcn'
        });

        console.log('World Engine Studio auto-initialized');
        window.studio = defaultStudio;
    });
}
