/**
 * Studio Bridge Layer 1 - Foundation Transport & Communication System
 * Replaces the existing studio.js with a more robust, layered architecture
 * Based on upgrade specifications for BroadcastChannel, namespaced store, and transport
 */

(function () {
    'use strict';

    // ===== 1) Version & Guard =====
    if (typeof window.__STUDIO_BRIDGE__ !== 'undefined') {
        const existing = window.__STUDIO_BRIDGE__;
        if (existing.version >= '0.1.0-base') {
            console.log('[Studio:init] Bridge already installed v' + existing.version);
            return;
        }
    }

    const VERSION = '0.1.0-base';
    const INSTALLED_AT = Date.now();

    console.log('[Studio:init] v' + VERSION + ' transport=BroadcastChannel+fallback');

    // ===== 2) Immutable Config & Constants =====
    const CONST = Object.freeze({
        BUS_NAME: 'studio-bus',
        NAMESPACE: 'studio:',
        MESSAGE_TYPES: Object.freeze([
            'eng.run', 'eng.test', 'eng.result', 'eng.status', 'eng.error',
            'rec.start', 'rec.stop', 'rec.clip', 'rec.transcript', 'rec.mark',
            'chat.cmd', 'chat.announce', 'chat.message', 'chat.transcript',
            'meta.process', 'meta.entangle', 'meta.collapse',
            'orchestrator.execute', 'orchestrator.queue', 'orchestrator.error'
        ]),
        STORAGE_SIZE_WARNING: 1024 * 1024 // 1MB
    });

    // ===== 3) Event Bus (BroadcastChannel + fallback) =====
    const bc = ('BroadcastChannel' in self) ? new BroadcastChannel(CONST.BUS_NAME) : null;
    const listeners = new Set();

    function _fan(msg) {
        if (!msg || typeof msg.type !== 'string') return; // boundary guard
        listeners.forEach(function (fn) {
            try {
                fn(msg);
            } catch (e) {
                console.warn('[Studio:bus-listener]', e);
            }
        });
    }

    if (bc) {
        bc.onmessage = function (e) { _fan(e.data); };
    } else {
        window.addEventListener('studio:msg', function (e) { _fan(e.detail); });
    }

    function onBus(fn) {
        listeners.add(fn);
        return function off() { listeners.delete(fn); };
    }

    function sendBus(msg) {
        // Validate message type
        if (!msg.type || typeof msg.type !== 'string') {
            console.warn('[Studio:sendBus] Invalid message type:', msg);
            return;
        }

        // Add timestamp if not present
        if (!msg.timestamp) {
            msg.timestamp = Date.now();
        }

        if (bc) {
            bc.postMessage(msg);
        } else {
            window.dispatchEvent(new CustomEvent('studio:msg', { detail: msg }));
        }
    }

    // ===== 4) Store (Namespaced, JSON-safe) =====
    const Store = {
        save: function (key, value) {
            try {
                const namespacedKey = CONST.NAMESPACE + key;
                const serialized = JSON.stringify(value);

                // Size warning
                if (serialized.length > CONST.STORAGE_SIZE_WARNING) {
                    console.warn('[Studio:store] Large value for key:', key, 'Size:', serialized.length);
                }

                localStorage.setItem(namespacedKey, serialized);
                return true;
            } catch (error) {
                console.error('[Studio:store] Save failed for key:', key, error);
                return false;
            }
        },

        load: function (key, fallback) {
            if (fallback === void 0) { fallback = null; }
            try {
                const namespacedKey = CONST.NAMESPACE + key;
                const item = localStorage.getItem(namespacedKey);
                return item ? JSON.parse(item) : fallback;
            } catch (error) {
                console.warn('[Studio:store] Load failed for key:', key, error);
                return fallback;
            }
        },

        remove: function (key) {
            try {
                const namespacedKey = CONST.NAMESPACE + key;
                localStorage.removeItem(namespacedKey);
                return true;
            } catch (error) {
                console.error('[Studio:store] Remove failed for key:', key, error);
                return false;
            }
        },

        clear: function () {
            try {
                const keysToRemove = [];
                for (let i = 0; i < localStorage.length; i++) {
                    const key = localStorage.key(i);
                    if (key && key.startsWith(CONST.NAMESPACE)) {
                        keysToRemove.push(key);
                    }
                }
                keysToRemove.forEach(function (key) {
                    localStorage.removeItem(key);
                });
                return true;
            } catch (error) {
                console.error('[Studio:store] Clear failed:', error);
                return false;
            }
        }
    };

    // ===== 5) Utils =====
    const Utils = {
        generateId: function () {
            return ('crypto' in window && 'randomUUID' in crypto) ?
                crypto.randomUUID() :
                'id_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
        },

        parseCommand: function (line) {
            if (!line || typeof line !== 'string') return null;

            line = line.trim();
            if (!line.startsWith('/')) {
                return { command: 'run', args: [line], raw: line };
            }

            const parts = line.slice(1).split(/\s+/);
            const command = parts[0].toLowerCase();
            const args = parts.slice(1);

            // Validate command
            const validCommands = ['run', 'test', 'rec', 'mark', 'status', 'history', 'clear', 'help'];
            if (validCommands.indexOf(command) === -1) {
                return { command: 'run', args: [line], raw: line };
            }

            return { command: command, args: args, raw: line };
        },

        log: function (msg, level) {
            if (level === void 0) { level = 'info'; }
            const timestamp = new Date().toISOString();
            const prefix = '[Studio:' + timestamp + ']';

            // Check debug flag
            const debugEnabled = localStorage.getItem('debug') === 'studio' ||
                new URLSearchParams(location.search).get('debug') === 'studio';

            if (level === 'debug' && !debugEnabled) return;

            switch (level) {
                case 'error':
                    console.error(prefix, msg);
                    break;
                case 'warn':
                    console.warn(prefix, msg);
                    break;
                case 'debug':
                    console.log(prefix + '[DEBUG]', msg);
                    break;
                default:
                    console.log(prefix, msg);
            }
        }
    };

    // ===== 6) Engine Transport =====
    function setupEngineTransport(iframe) {
        if (!iframe || !iframe.contentWindow) {
            Utils.log('Invalid iframe provided to setupEngineTransport', 'error');
            return null;
        }

        return {
            isSameOrigin: function () {
                try {
                    return !!iframe.contentDocument;
                } catch (e) {
                    return false;
                }
            },

            withEngine: function (fn) {
                if (!this.isSameOrigin()) {
                    Utils.log('Cross-origin iframe detected, cannot access engine', 'warn');
                    return;
                }

                try {
                    const doc = iframe.contentDocument || iframe.contentWindow.document;
                    if (doc && doc.readyState === 'complete') {
                        fn(doc);
                    } else {
                        Utils.log('Engine document not ready', 'warn');
                    }
                } catch (error) {
                    Utils.log('Transport error: ' + error.message, 'error');
                }
            }
        };
    }

    // ===== 7) Enhanced Orchestrator Integration =====
    var OrchestratorIntegration = {
        init: function () {
            // Listen for orchestrator commands from Layer 0
            onBus(function (msg) {
                switch (msg.type) {
                    case 'orchestrator.execute':
                        OrchestratorIntegration.handleExecute(msg);
                        break;
                    case 'orchestrator.queue':
                        OrchestratorIntegration.handleQueue(msg);
                        break;
                    case 'eng.result':
                        OrchestratorIntegration.handleResult(msg);
                        break;
                }
            });
        },

        handleExecute: function (msg) {
            Utils.log('Orchestrator execute: ' + msg.text, 'debug');

            // Route to appropriate engine based on priority and load
            sendBus({
                type: 'eng.run',
                text: msg.text,
                runId: msg.runId || Utils.generateId(),
                priority: msg.priority || 'normal',
                metadata: msg.metadata || {}
            });
        },

        handleQueue: function (msg) {
            Utils.log('Orchestrator queue status: ' + msg.queueSize, 'debug');

            // Store queue info for monitoring
            Store.save('queue_status', {
                size: msg.queueSize,
                activeRuns: msg.activeRuns,
                timestamp: Date.now()
            });
        },

        handleResult: function (msg) {
            Utils.log('Engine result received: ' + msg.runId, 'debug');

            // Forward to Meta-Base Thought Engine if needed
            if (msg.outcome && msg.outcome.needsProcessing) {
                sendBus({
                    type: 'meta.process',
                    input: msg.input,
                    rawResult: msg.outcome,
                    runId: msg.runId
                });
            }
        }
    };

    // ===== 8) Layer Boundary Enforcement =====
    const LayerEnforcer = {
        currentLayer: 1, // This is Layer 1

        validateAccess: function (targetLayer, operation) {
            if (targetLayer < this.currentLayer) {
                Utils.log('Layer boundary violation: Layer 1 cannot modify Layer ' + targetLayer, 'error');
                return false;
            }
            return true;
        },

        enforceCanvasLaw: function (moduleName) {
            // All modules must be registered in canvas
            const validModules = Store.load('canvas_modules', []);
            if (validModules.indexOf(moduleName) === -1) {
                Utils.log('Canvas law violation: Module not in canvas: ' + moduleName, 'error');
                sendBus({
                    type: 'orchestrator.error',
                    error: 'Canvas law violation',
                    module: moduleName,
                    layer: this.currentLayer
                });
                return false;
            }
            return true;
        }
    };

    // ===== 9) Enhanced Recording Integration =====
    var RecordingBridge = {
        activeSession: null,

        init: function () {
            onBus(function (msg) {
                if (msg.type.startsWith('rec.')) {
                    RecordingBridge.handleRecordingEvent(msg);
                }
            });
        },

        handleRecordingEvent: function (msg) {
            switch (msg.type) {
                case 'rec.start':
                    this.activeSession = {
                        sessionId: msg.sessionId,
                        startTime: Date.now(),
                        format: msg.format || 'webm'
                    };
                    Utils.log('Recording started: ' + msg.sessionId);
                    break;

                case 'rec.stop':
                    if (this.activeSession) {
                        const duration = Date.now() - this.activeSession.startTime;
                        Utils.log('Recording stopped. Duration: ' + duration + 'ms');
                        this.activeSession = null;
                    }
                    break;

                case 'rec.clip':
                    // Handle recording clips for meta-base processing
                    if (msg.audioData) {
                        sendBus({
                            type: 'meta.process',
                            input: msg.audioData,
                            inputType: 'audio',
                            timestamp: Date.now()
                        });
                    }
                    break;
            }
        },

        getStatus: function () {
            return {
                active: !!this.activeSession,
                session: this.activeSession
            };
        }
    };

    // ===== 10) API Export =====
    const StudioBridge = {
        // Version info
        version: VERSION,
        installedAt: INSTALLED_AT,

        // Core APIs
        onBus: onBus,
        sendBus: sendBus,
        store: Store,
        utils: Utils,
        setupEngineTransport: setupEngineTransport,

        // Layer 1 specific
        orchestrator: OrchestratorIntegration,
        layerEnforcer: LayerEnforcer,
        recording: RecordingBridge,

        // Status and diagnostics
        getStatus: function () {
            return {
                version: VERSION,
                layer: 1,
                transport: bc ? 'BroadcastChannel' : 'CustomEvent',
                listeners: listeners.size,
                storage: {
                    available: 'localStorage' in window,
                    namespace: CONST.NAMESPACE
                },
                recording: RecordingBridge.getStatus(),
                timestamp: Date.now()
            };
        },

        // Health check
        healthCheck: function () {
            const health = {
                bus: true,
                storage: true,
                transport: true,
                layer0: false // Will be set by Layer 0 if available
            };

            try {
                // Test bus
                const testId = Utils.generateId();
                let testReceived = false;
                const off = onBus(function (msg) {
                    if (msg.testId === testId) {
                        testReceived = true;
                    }
                });
                sendBus({ type: 'test', testId: testId });
                setTimeout(function () {
                    health.bus = testReceived;
                    off();
                }, 100);

                // Test storage
                Store.save('health_test', 'ok');
                health.storage = Store.load('health_test') === 'ok';
                Store.remove('health_test');

            } catch (error) {
                Utils.log('Health check failed: ' + error.message, 'error');
                health.bus = false;
                health.storage = false;
            }

            return health;
        }
    };

    // ===== 11) Initialize & Export =====
    // Initialize subsystems
    OrchestratorIntegration.init();
    RecordingBridge.init();

    // Global export
    window.__STUDIO_BRIDGE__ = StudioBridge;

    // Also export as module if in module context
    if (typeof module !== 'undefined' && module.exports) {
        module.exports = StudioBridge;
    }

    Utils.log('Studio Bridge Layer 1 initialized successfully');

    // Health check on load
    setTimeout(function () {
        const health = StudioBridge.healthCheck();
        Utils.log('Initial health check:', 'debug');
        Utils.log(health, 'debug');
    }, 1000);

})();
