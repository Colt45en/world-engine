/**
 * WebSocket Clock Synchronization Handler
 * ======================================
 *
 * Handles real-time clock synchronization across all connected clients
 * Integrates with controllers.combined.js and EngineRoom WebSocket connections
 */

class WebSocketClockSync {
    constructor(websocket, clockInstance) {
        this.ws = websocket;
        this.clock = clockInstance;
        this.clockDebugger = null;
        this.isMaster = false;
        this.clientId = this.generateClientId();
        this.lastSyncTime = 0;
        this.syncInterval = null;
        this.listeners = new Map();

        this.setupMessageHandlers();
        this.setupHeartbeat();

        console.log(`🕐 WebSocket Clock Sync initialized - Client: ${this.clientId}`);
    }

    generateClientId() {
        return 'client_' + Math.random().toString(36).substr(2, 9);
    }

    setupMessageHandlers() {
        if (!this.ws) return;

        this.ws.addEventListener('message', (event) => {
            try {
                const message = JSON.parse(event.data);
                this.handleMessage(message);
            } catch (error) {
                console.warn('Failed to parse WebSocket message:', error);
            }
        });

        this.ws.addEventListener('open', () => {
            this.requestMasterStatus();
        });

        this.ws.addEventListener('close', () => {
            this.cleanup();
        });
    }

    setupHeartbeat() {
        // Send clock state every 500ms if we're the master
        this.syncInterval = setInterval(() => {
            if (this.isMaster && this.clock && this.clock.running) {
                this.broadcastClockState();
            }
        }, 500);
    }

    handleMessage(message) {
        switch (message.type) {
            case 'clock.master.request':
                this.handleMasterRequest(message);
                break;

            case 'clock.master.assign':
                this.handleMasterAssignment(message);
                break;

            case 'clock.sync':
                this.handleClockSync(message);
                break;

            case 'clock.bpm.change':
                this.handleBPMChange(message);
                break;

            case 'clock.start':
                this.handleClockStart(message);
                break;

            case 'clock.stop':
                this.handleClockStop(message);
                break;

            case 'clock.debug.request':
                this.handleDebugRequest(message);
                break;

            default:
                // Pass to other handlers
                this.emit('message', message);
                break;
        }
    }

    requestMasterStatus() {
        this.send({
            type: 'clock.master.request',
            clientId: this.clientId,
            timestamp: Date.now()
        });
    }

    handleMasterRequest(message) {
        // Server should handle master assignment logic
        // For now, first client becomes master
        if (!this.isMaster) {
            this.send({
                type: 'clock.master.assign',
                targetClientId: message.clientId,
                assignerId: this.clientId,
                timestamp: Date.now()
            });
        }
    }

    handleMasterAssignment(message) {
        if (message.targetClientId === this.clientId) {
            this.isMaster = true;
            console.log('🎯 Assigned as clock master');

            // Start broadcasting if clock is running
            if (this.clock && this.clock.running) {
                this.broadcastClockState();
            }
        }
    }

    handleClockSync(message) {
        if (message.clientId === this.clientId) return; // Ignore own messages

        const { data } = message;
        if (!data) return;

        // Update our clock to match master
        if (this.clock && !this.isMaster) {
            const serverTime = data.timestamp;
            const localTime = Date.now();
            const latency = (localTime - serverTime) / 2; // Rough latency estimate

            // Apply clock state with latency compensation
            this.clock.bar = data.bar;
            this.clock.beat = data.beat;
            this.clock.phase = data.phase;
            this.clock.bpm = data.bpm;

            // Compensate for network latency in phase
            if (latency > 0 && this.clock.running) {
                const secPerBeat = 60 / data.bpm;
                const latencyBeats = (latency / 1000) / secPerBeat;
                this.clock.phase = (data.phase + latencyBeats) % 1.0;
            }

            this.emit('clock.synced', {
                ...data,
                latency: latency,
                clientId: message.clientId
            });
        }

        // Update debug info if available
        if (this.clockDebugger) {
            this.clockDebugger.recordSyncEvent({
                type: 'websocket_sync',
                masterClientId: message.clientId,
                localTime: Date.now(),
                serverTime: data.timestamp,
                clockState: data
            });
        }
    }

    handleBPMChange(message) {
        if (this.clock) {
            this.clock.setBPM(message.bpm);
            console.log(`🎵 BPM changed to ${message.bpm} by ${message.clientId}`);

            this.emit('bpm.changed', {
                bpm: message.bpm,
                clientId: message.clientId
            });
        }
    }

    handleClockStart(message) {
        if (this.clock && !this.isMaster) {
            this.clock.start(message.data.startTime);
            console.log('▶️ Clock started by master');

            this.emit('clock.started', message.data);
        }
    }

    handleClockStop(message) {
        if (this.clock && !this.isMaster) {
            this.clock.stop();
            console.log('⏸️ Clock stopped by master');

            this.emit('clock.stopped', message.data);
        }
    }

    handleDebugRequest(message) {
        if (this.clockDebugger) {
            const debugData = this.clockDebugger.generateReport();
            this.send({
                type: 'clock.debug.response',
                requestId: message.requestId,
                clientId: this.clientId,
                data: debugData
            });
        }
    }

    // Master methods
    broadcastClockState() {
        if (!this.isMaster || !this.clock) return;

        const state = this.clock.getState();
        this.send({
            type: 'clock.sync',
            clientId: this.clientId,
            data: {
                ...state,
                timestamp: Date.now()
            }
        });
    }

    startClock(bpm) {
        if (!this.isMaster) return;

        if (this.clock) {
            if (bpm) this.clock.setBPM(bpm);
            this.clock.start(Date.now() / 1000);
        }

        this.send({
            type: 'clock.start',
            clientId: this.clientId,
            data: {
                startTime: Date.now() / 1000,
                bpm: bpm || this.clock?.bpm || 120
            }
        });
    }

    stopClock() {
        if (!this.isMaster) return;

        if (this.clock) {
            this.clock.stop();
        }

        this.send({
            type: 'clock.stop',
            clientId: this.clientId,
            data: {
                timestamp: Date.now()
            }
        });
    }

    changeBPM(newBPM) {
        if (!this.isMaster) return;

        if (this.clock) {
            this.clock.setBPM(newBPM);
        }

        this.send({
            type: 'clock.bpm.change',
            clientId: this.clientId,
            bpm: newBPM,
            timestamp: Date.now()
        });
    }

    // Debug integration
    attachClockDebugger(debugger) {
    this.clockDebugger = debugger;
    console.log('🐛 Clock debugger attached to WebSocket sync');
}

requestDebugFromAll() {
    const requestId = 'debug_' + Date.now();
    this.send({
        type: 'clock.debug.request',
        requestId: requestId,
        clientId: this.clientId
    });
    return requestId;
}

// Utility methods
send(message) {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
        this.ws.send(JSON.stringify(message));
    }
}

on(eventType, callback) {
    if (!this.listeners.has(eventType)) {
        this.listeners.set(eventType, []);
    }
    this.listeners.get(eventType).push(callback);
}

emit(eventType, data) {
    const callbacks = this.listeners.get(eventType) || [];
    callbacks.forEach(callback => {
        try {
            callback(data);
        } catch (error) {
            console.error('Error in clock sync callback:', error);
        }
    });
}

cleanup() {
    if (this.syncInterval) {
        clearInterval(this.syncInterval);
        this.syncInterval = null;
    }
    this.listeners.clear();
    console.log('🕐 WebSocket Clock Sync cleaned up');
}

// Status getters
getStatus() {
    return {
        clientId: this.clientId,
        isMaster: this.isMaster,
        connected: this.ws && this.ws.readyState === WebSocket.OPEN,
        clockRunning: this.clock ? this.clock.running : false,
        lastSyncTime: this.lastSyncTime
    };
}
}

// Integration with controllers.combined.js
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { WebSocketClockSync };
}

// Browser global
if (typeof window !== 'undefined') {
    window.WebSocketClockSync = WebSocketClockSync;
}

// Example usage in studio context
/*
// In studio.html or EngineRoom integration:
const ws = new WebSocket('ws://localhost:9000');
const clock = new HolyBeatClock(120);
const clockSync = new WebSocketClockSync(ws, clock);

// Attach clock debugger if available
if (window.ClockDebug) {
    const debugger = new window.ClockDebug.ClockBugDetector();
    clockSync.attachClockDebugger(debugger);
}

// Listen for sync events
clockSync.on('clock.synced', (data) => {
    console.log('Clock synced:', data);
    // Update UI, visualizations, etc.
});

clockSync.on('bpm.changed', (data) => {
    console.log('BPM changed:', data);
    // Update BPM displays
});

// Control clock (only works if this client is master)
// clockSync.startClock(140); // Start at 140 BPM
// clockSync.changeBPM(160);  // Change to 160 BPM
// clockSync.stopClock();     // Stop the clock
*/
