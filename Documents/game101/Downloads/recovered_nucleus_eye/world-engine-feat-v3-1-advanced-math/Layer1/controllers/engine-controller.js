/**
 * Enhanced Engine Controller - Multi-iframe pool with robust dispatch
 * Features:
 * - MutationObserver for reliable result detection
 * - Multi-engine pool with round-robin
 * - Proper message re-dispatch when not ready
 * - Active run lifecycle management
 */

import { assertShape, Schemas } from '../schemas/validation.js';

export class EngineController {
    constructor(options = {}) {
        this.options = {
            resultTimeout: 5000,
            retryDelay: 100,
            poolSize: 1,
            ...options
        };

        this.engines = new Map(); // engineId -> {frame, isReady, transport}
        this.activeRuns = new Map(); // runId -> {startTime, engineId, cleanup}
        this.currentEngine = 0;
        this.messageQueue = [];

        this.setupEngines();
        this.bindEvents();
    }

    async setupEngines() {
        // Create initial engine pool
        for (let i = 0; i < this.options.poolSize; i++) {
            await this.addEngine(`engine-${i}`);
        }
    }

    async addEngine(engineId) {
        // Create iframe for this engine
        const frame = document.createElement('iframe');
        frame.id = `${engineId}-frame`;
        frame.title = `World Engine ${engineId}`;
        frame.src = 'worldengine.html';
        frame.sandbox = 'allow-scripts allow-same-origin';
        frame.allow = 'microphone; camera; display-capture';
        frame.style.display = 'none'; // Hidden engines

        document.body.appendChild(frame);

        const engine = {
            id: engineId,
            frame,
            isReady: false,
            transport: null,
            activeRuns: new Set()
        };

        // Setup transport when loaded
        frame.addEventListener('load', () => {
            engine.transport = this.setupEngineTransport(frame);
            setTimeout(() => {
                engine.isReady = true;
                console.log(`Engine ${engineId} ready`);
                this.bus.sendBus({ type: 'eng.ready', engineId });
                this.processQueuedMessages();
            }, 500);
        });

        this.engines.set(engineId, engine);
        this.bus.sendBus({ type: 'orchestrator.add-engine', engineId });
        return engineId;
    }

    bindEvents() {
        this.bus.onBus(async (msg) => {
            switch (msg.type) {
                case 'eng.run':
                    await this.handleRun(msg);
                    break;
                case 'eng.test':
                    await this.handleTest(msg);
                    break;
                case 'eng.status':
                    await this.handleStatus(msg);
                    break;
            }
        });
    }

    async handleRun(msg) {
        const runId = msg.runId || this.generateId();
        const engineId = msg.engineId || this.selectEngine();

        if (!engineId) {
            // No engines ready, queue the message
            this.messageQueue.push({ ...msg, runId });
            console.log('No engines ready, queueing message');
            return;
        }

        const engine = this.engines.get(engineId);
        if (!engine?.isReady) {
            // Engine not ready, re-queue with delay
            setTimeout(() => {
                this.bus.sendBus({ ...msg, runId });
            }, this.options.retryDelay);
            return;
        }

        console.log(`Starting engine run: ${runId} on ${engineId}`);

        try {
            // Track active run
            this.activeRuns.set(runId, {
                startTime: Date.now(),
                engineId,
                cleanup: null
            });

            engine.activeRuns.add(runId);

            // Execute with MutationObserver
            const result = await this.executeOnEngine(engine, msg.text, runId);

            // Success
            this.bus.sendBus({
                type: 'eng.result',
                runId,
                outcome: result,
                input: msg.text,
                engineId
            });

        } catch (error) {
            console.error(`Engine run failed: ${runId}`, error);
            this.bus.sendBus({
                type: 'eng.error',
                runId,
                error: error.message,
                engineId
            });
        } finally {
            // Cleanup
            this.activeRuns.delete(runId);
            engine.activeRuns.delete(runId);
        }
    }

    async executeOnEngine(engine, text, runId) {
        return new Promise((resolve, reject) => {
            let completed = false;

            const timeout = setTimeout(() => {
                if (!completed) {
                    completed = true;
                    cleanup?.();
                    reject(new Error('Engine execution timeout'));
                }
            }, this.options.resultTimeout);

            let cleanup = null;

            engine.transport.withEngine((doc) => {
                const input = doc.getElementById('input');
                const runBtn = doc.getElementById('run');
                const output = doc.getElementById('out');

                if (!input || !runBtn || !output) {
                    completed = true;
                    clearTimeout(timeout);
                    reject(new Error('Engine UI elements not found'));
                    return;
                }

                // Set input and trigger
                input.value = text.trim();
                runBtn.click();

                // Setup MutationObserver for result detection
                const observer = new MutationObserver(() => {
                    if (completed) return;

                    try {
                        const rawText = output.textContent || '{}';
                        let outcome;

                        try {
                            outcome = JSON.parse(rawText);
                        } catch (parseError) {
                            // Not JSON, wrap as text result
                            outcome = {
                                type: 'text',
                                result: rawText,
                                input: text,
                                timestamp: Date.now()
                            };
                        }

                        // Validate result
                        if (outcome && (outcome.items || outcome.result || outcome.count !== undefined)) {
                            completed = true;
                            clearTimeout(timeout);
                            observer.disconnect();
                            resolve(outcome);
                        }
                    } catch (err) {
                        console.warn('Error processing engine result:', err);
                    }
                });

                // Observe output changes
                observer.observe(output, {
                    childList: true,
                    characterData: true,
                    subtree: true
                });

                cleanup = () => {
                    try { observer.disconnect(); } catch { }
                };

                // Store cleanup for timeout
                if (this.activeRuns.has(runId)) {
                    this.activeRuns.get(runId).cleanup = cleanup;
                }
            });
        });
    }

    selectEngine() {
        const readyEngines = Array.from(this.engines.entries())
            .filter(([id, engine]) => engine.isReady)
            .sort(([, a], [, b]) => a.activeRuns.size - b.activeRuns.size); // Load balance

        if (readyEngines.length === 0) return null;

        // Round-robin with load balancing
        const [engineId] = readyEngines[this.currentEngine++ % readyEngines.length];
        return engineId;
    }

    processQueuedMessages() {
        const toProcess = this.messageQueue.splice(0, 5); // Process in batches
        toProcess.forEach(msg => {
            setTimeout(() => this.bus.sendBus(msg), 10);
        });
    }

    setupEngineTransport(frame) {
        // Same-origin transport helper
        return {
            withEngine: (callback) => {
                try {
                    const doc = frame.contentDocument || frame.contentWindow?.document;
                    if (doc && doc.readyState === 'complete') {
                        callback(doc);
                    } else {
                        console.warn('Engine document not ready');
                    }
                } catch (error) {
                    console.error('Transport error:', error);
                }
            },
            isSameOrigin: () => {
                try {
                    return !!frame.contentDocument;
                } catch {
                    return false;
                }
            }
        };
    }

    generateId() {
        return `run_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;
    }

    getStatus() {
        const engineStats = {};
        for (const [id, engine] of this.engines) {
            engineStats[id] = {
                ready: engine.isReady,
                activeRuns: engine.activeRuns.size,
                frame: !!engine.frame
            };
        }

        return {
            engines: engineStats,
            totalActiveRuns: this.activeRuns.size,
            queuedMessages: this.messageQueue.length
        };
    }

    async scaleEngines(targetSize) {
        const currentSize = this.engines.size;
        if (targetSize > currentSize) {
            // Add engines
            for (let i = currentSize; i < targetSize; i++) {
                await this.addEngine(`engine-${i}`);
            }
        } else if (targetSize < currentSize) {
            // Remove engines (implement if needed)
            console.log('Engine removal not implemented');
        }
    }
}
