/**
 * Studio Orchestrator - Central coordinator for Chat↔Engine↔Recorder
 *
 * Features:
 * - Queue management with backpressure
 * - Automatic retries with exponential backoff
 * - Multi-engine pool support (round-robin)
 * - Timeline linking between runs and recordings
 * - Bulk corpus processing
 */

export class StudioOrchestrator {
    constructor(bus, store, options = {}) {
        this.bus = bus;
        this.store = store;
        this.opts = {
            maxConcurrency: 2,
            maxRetries: 3,
            backoffMs: 1000,
            timeoutMs: 5000,
            autoLinkRecording: true,
            ...options
        };

        this.q = [];
        this.active = 0;
        this.lastClipId = null;
        this.engines = []; // Multi-engine pool
        this.currentEngine = 0;

        this.bindEvents();
    }

    bindEvents() {
        this.bus.onBus((msg) => {
            switch (msg.type) {
                case 'chat.cmd.parsed':
                    if (msg.cmd.type === 'run') {
                        this.enqueueRun({
                            text: msg.cmd.args,
                            linkRecording: this.opts.autoLinkRecording
                        });
                    }
                    break;
                case 'rec.clip':
                    this.lastClipId = msg.clipId;
                    break;
                case 'orchestrator.add-engine':
                    this.engines.push(msg.engineId);
                    break;
            }
        });
    }

    // Public API
    enqueueRun(payload) {
        const task = {
            id: this.generateId(),
            kind: 'run',
            payload,
            retries: 0,
            createdAt: Date.now()
        };
        this.q.push(task);
        this._drain();
        return task.id;
    }

    enqueueBatch(textArray, options = {}) {
        const taskIds = [];
        for (const text of textArray) {
            const id = this.enqueueRun({
                text,
                linkRecording: false,
                ...options
            });
            taskIds.push(id);
        }
        return taskIds;
    }

    size() {
        return this.q.length + this.active;
    }

    getStats() {
        return {
            queued: this.q.length,
            active: this.active,
            engines: this.engines.length,
            lastClip: this.lastClipId
        };
    }

    // Internals
    async _drain() {
        while (this.active < this.opts.maxConcurrency && this.q.length) {
            const task = this.q.shift();
            this.active++;
            this._exec(task).finally(() => {
                this.active--;
                this._drain();
            });
        }
    }

    async _exec(task) {
        if (task.kind !== 'run') return;

        const { text, meta, linkRecording } = task.payload;
        const runId = this.generateId();

        try {
            // Mark start on recorder timeline
            if (linkRecording) {
                this.bus.sendBus({ type: 'rec.mark', tag: 'run-start', runId });
            }

            // Get result from engine (with timeout + retry)
            const result = await this._runEngineAwait(text, runId);

            // Store the run record
            const record = {
                runId,
                taskId: task.id,
                ts: Date.now(),
                input: text,
                outcome: result,
                clipId: linkRecording ? this.lastClipId : null,
                meta: meta || {}
            };

            await this.store.save(`runs.${runId}`, record);
            await this.store.save('wordEngine.lastRun', result);

            // Emit success
            this.bus.sendBus({ type: 'orchestrator.run.ok', runId, record });

            if (linkRecording) {
                this.bus.sendBus({ type: 'rec.mark', tag: 'run-end', runId });
            }

        } catch (err) {
            if (task.retries < this.opts.maxRetries) {
                task.retries++;
                const delay = this.opts.backoffMs * Math.pow(2, task.retries - 1);
                setTimeout(() => this.q.push(task), delay);

                this.bus.sendBus({
                    type: 'orchestrator.run.retry',
                    runId,
                    taskId: task.id,
                    err: err.message,
                    attempt: task.retries
                });
            } else {
                this.bus.sendBus({
                    type: 'orchestrator.run.fail',
                    runId,
                    taskId: task.id,
                    err: err.message
                });
            }
        }
    }

    _runEngineAwait(text, runId) {
        return new Promise((resolve, reject) => {
            const timeout = setTimeout(() => {
                cleanup();
                reject(new Error('Engine timeout'));
            }, this.opts.timeoutMs);

            const cleanup = this.bus.onBus((msg) => {
                if (msg.runId === runId) {
                    if (msg.type === 'eng.result') {
                        clearTimeout(timeout);
                        cleanup();
                        resolve(msg.outcome);
                    } else if (msg.type === 'eng.error') {
                        clearTimeout(timeout);
                        cleanup();
                        reject(new Error(msg.error || 'Engine error'));
                    }
                }
            });

            // Send to engine (with optional round-robin)
            const engineId = this.engines.length > 0 ?
                this.engines[this.currentEngine++ % this.engines.length] :
                null;

            this.bus.sendBus({
                type: 'eng.run',
                text,
                runId,
                engineId
            });
        });
    }

    generateId() {
        return `orch_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;
    }
}

// Helper for bulk operations
export async function enqueueCorpus(orchestrator, textArray, options = {}) {
    const taskIds = orchestrator.enqueueBatch(textArray, options);
    return {
        count: textArray.length,
        taskIds,
        stats: orchestrator.getStats()
    };
}
