/**
 * Tier-4 WebSocket State Synchronizer
 * ===================================
 *
 * Connects your WebSocket relay (ws_relay.js) with the Tier-4 Meta System
 * for real-time distributed state synchronization and collaborative reasoning.
 */

interface NDJSONEvent {
    type: string;
    ts: string;
    cycle?: number;
    total?: number;
    id?: string;
    role?: string;
    state?: string;
    tag?: string;
    thought?: string;
    nucleus?: string;
    need_tag?: string;
    from?: string;
    to?: string;
    memory_size?: number;
}

interface Tier4StateUpdate {
    type: 'tier4_state_update';
    ts: string;
    sessionId: string;
    state: {
        x: number[];
        kappa: number;
        level: number;
    };
    lastOperator: string;
    metadata: {
        userId?: string;
        source: string;
        confidence: number;
    };
}

interface CollaborativeSession {
    sessionId: string;
    participants: Set<string>;
    state: any;
    lastUpdate: string;
    operatorHistory: string[];
    conflictResolution: 'merge' | 'latest' | 'vote';
}

class Tier4WebSocketSync {
    private ws: WebSocket | null = null;
    private isConnected = false;
    private sessionId: string;
    private userId: string;
    private tier4System: any;
    private activeSessions = new Map<string, CollaborativeSession>();
    private eventBuffer: NDJSONEvent[] = [];
    private nucleusToOperatorMap = new Map<string, string>();
    private reconnectAttempts = 0;
    private maxReconnectAttempts = 5;

    constructor(tier4System: any, userId: string = 'anonymous') {
        this.tier4System = tier4System;
        this.userId = userId;
        this.sessionId = this.generateSessionId();
        this.setupNucleusMapping();
        this.connect();
    }

    // ============================= Connection Management =============================

    private connect() {
        try {
            this.ws = new WebSocket('ws://localhost:9000');

            this.ws.onopen = () => {
                this.isConnected = true;
                this.reconnectAttempts = 0;
                console.log('🌐 Tier-4 WebSocket connected to relay');

                // Send initial state
                this.broadcastTier4State('connection', 'Connected to distributed session');
            };

            this.ws.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data);
                    this.handleIncomingEvent(data);
                } catch (error) {
                    console.warn('Failed to parse WebSocket message:', error);
                }
            };

            this.ws.onclose = () => {
                this.isConnected = false;
                console.log('🌐 WebSocket connection closed');
                this.attemptReconnect();
            };

            this.ws.onerror = (error) => {
                console.error('🌐 WebSocket error:', error);
                this.isConnected = false;
            };

        } catch (error) {
            console.error('Failed to connect to WebSocket:', error);
            this.attemptReconnect();
        }
    }

    private attemptReconnect() {
        if (this.reconnectAttempts < this.maxReconnectAttempts) {
            this.reconnectAttempts++;
            const delay = Math.pow(2, this.reconnectAttempts) * 1000; // Exponential backoff

            console.log(`🔄 Reconnecting in ${delay}ms (attempt ${this.reconnectAttempts})`);

            setTimeout(() => {
                this.connect();
            }, delay);
        } else {
            console.error('❌ Max reconnection attempts reached. Working offline.');
        }
    }

    // ============================= Nucleus-to-Operator Mapping =============================

    private setupNucleusMapping() {
        // Map nucleus roles to Tier-4 operators based on the NDJSON data
        this.nucleusToOperatorMap.set('VIBRATE', 'ST'); // Vibration → Snapshot (capture initial state)
        this.nucleusToOperatorMap.set('OPTIMIZATION', 'UP'); // Optimization → Update
        this.nucleusToOperatorMap.set('STATE', 'CV'); // State → Convert (change representation)
        this.nucleusToOperatorMap.set('SEED', 'RB'); // Seed → Rebuild (plant new structure)

        // Memory operations
        this.nucleusToOperatorMap.set('energy', 'CH'); // Energy tag → Channel
        this.nucleusToOperatorMap.set('refined', 'PR'); // Refined tag → Prevent
        this.nucleusToOperatorMap.set('condition', 'SL'); // Condition tag → Select
        this.nucleusToOperatorMap.set('seed', 'MD'); // Seed tag → Module
    }

    // ============================= Event Processing =============================

    private handleIncomingEvent(event: NDJSONEvent) {
        this.eventBuffer.push(event);

        switch (event.type) {
            case 'cycle_start':
                this.handleCycleStart(event);
                break;

            case 'nucleus_exec':
                this.handleNucleusExecution(event);
                break;

            case 'memory_store':
                this.handleMemoryStore(event);
                break;

            case 'loop_back':
                this.handleLoopBack(event);
                break;

            case 'cycle_end':
                this.handleCycleEnd(event);
                break;

            case 'hallucination_guard':
                this.handleHallucinationGuard(event);
                break;

            case 'tier4_state_update':
                this.handleTier4StateUpdate(event as unknown as Tier4StateUpdate);
                break;

            default:
                console.log('🔍 Unknown event type:', event.type);
        }
    }

    private handleCycleStart(event: NDJSONEvent) {
        console.log(`🔄 Cycle ${event.cycle} started (${event.total} total)`);

        // Trigger cycle preparation in Tier-4
        if (this.tier4System?.applyOperator) {
            this.tier4System.applyOperator('ST', {
                source: 'cycle_start',
                cycle: event.cycle,
                collaborative: true
            });
        }
    }

    private handleNucleusExecution(event: NDJSONEvent) {
        const operator = this.nucleusToOperatorMap.get(event.role || '');

        if (operator && this.tier4System?.applyOperator) {
            console.log(`🧠 Nucleus ${event.id} (${event.role}) → Tier-4 ${operator}`);

            try {
                this.tier4System.applyOperator(operator, {
                    source: 'nucleus_exec',
                    nucleusId: event.id,
                    nucleusRole: event.role,
                    collaborative: true,
                    timestamp: event.ts
                });

                // Broadcast the state change
                this.broadcastTier4State(operator, `Triggered by nucleus ${event.id} (${event.role})`);

            } catch (error) {
                console.error(`Failed to apply operator ${operator}:`, error);
                this.sendHallucinationGuard(event.id || '', operator);
            }
        }
    }

    private handleMemoryStore(event: NDJSONEvent) {
        const operator = this.nucleusToOperatorMap.get(event.tag || '');

        if (operator && this.tier4System?.applyOperator) {
            console.log(`💾 Memory store "${event.tag}" → ${operator}: ${event.thought}`);

            this.tier4System.applyOperator(operator, {
                source: 'memory_store',
                tag: event.tag,
                thought: event.thought,
                collaborative: true
            });
        }
    }

    private handleLoopBack(event: NDJSONEvent) {
        console.log(`🔁 Loop: ${event.from} → ${event.to}`);

        // Map loop-backs to macro sequences
        if (event.from === 'seed' && event.to === 'energy') {
            // Seed→Energy loop maps to IDE_A analysis cycle
            if (this.tier4System?.runMacro) {
                this.tier4System.runMacro('IDE_A', {
                    source: 'loop_back',
                    collaborative: true
                });

                this.broadcastTier4State('IDE_A', `Loop-back macro: ${event.from} → ${event.to}`);
            }
        }
    }

    private handleCycleEnd(event: NDJSONEvent) {
        console.log(`✅ Cycle ${event.cycle} completed, memory size: ${event.memory_size}`);

        // Sync final state after cycle completion
        this.syncCollaborativeState();
    }

    private handleHallucinationGuard(event: NDJSONEvent) {
        console.log(`🛡️ Hallucination guard for ${event.nucleus}, needs: ${event.need_tag}`);

        // Apply prevention operator when hallucination is detected
        if (this.tier4System?.applyOperator) {
            this.tier4System.applyOperator('PR', {
                source: 'hallucination_guard',
                nucleus: event.nucleus,
                needTag: event.need_tag,
                collaborative: true
            });
        }
    }

    private handleTier4StateUpdate(event: Tier4StateUpdate) {
        // Handle incoming Tier-4 state updates from other clients
        if (event.sessionId !== this.sessionId && this.tier4System?.setState) {
            console.log(`🔄 Applying remote state update from ${event.metadata.userId}`);

            try {
                this.tier4System.setState(event.state, {
                    source: 'collaborative_update',
                    fromUser: event.metadata.userId,
                    operator: event.lastOperator
                });
            } catch (error) {
                console.error('Failed to apply remote state update:', error);
            }
        }
    }

    // ============================= State Broadcasting =============================

    private broadcastTier4State(operator: string, reason: string) {
        if (!this.isConnected || !this.tier4System?.getCurrentState) return;

        const state = this.tier4System.getCurrentState();

        const update: Tier4StateUpdate = {
            type: 'tier4_state_update',
            ts: new Date().toISOString(),
            sessionId: this.sessionId,
            state: {
                x: state.x || [0, 0.5, 0.4, 0.6],
                kappa: state.kappa || 0.6,
                level: state.level || 0
            },
            lastOperator: operator,
            metadata: {
                userId: this.userId,
                source: reason,
                confidence: state.kappa || 0.6
            }
        };

        this.sendEvent(update);
    }

    private sendEvent(event: any) {
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify(event));
        } else {
            console.warn('WebSocket not connected, event queued:', event.type);
        }
    }

    private sendHallucinationGuard(nucleusId: string, operator: string) {
        this.sendEvent({
            type: 'tier4_hallucination_guard',
            ts: new Date().toISOString(),
            sessionId: this.sessionId,
            nucleusId: nucleusId,
            operator: operator,
            userId: this.userId,
            message: `Operator ${operator} failed validation`
        });
    }

    // ============================= Collaborative Session Management =============================

    private generateSessionId(): string {
        return `tier4_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    }

    private syncCollaborativeState() {
        // Implement CRDT-like conflict resolution
        const currentState = this.tier4System?.getCurrentState();
        if (!currentState) return;

        // Simple last-writer-wins for now (could be enhanced with vector clocks)
        this.broadcastTier4State('SYNC', 'Collaborative state synchronization');
    }

    // ============================= Public API =============================

    public connectToSession(sessionId: string) {
        this.sessionId = sessionId;
        console.log(`🤝 Joined collaborative session: ${sessionId}`);

        this.sendEvent({
            type: 'tier4_session_join',
            ts: new Date().toISOString(),
            sessionId: sessionId,
            userId: this.userId
        });
    }

    public leaveSession() {
        this.sendEvent({
            type: 'tier4_session_leave',
            ts: new Date().toISOString(),
            sessionId: this.sessionId,
            userId: this.userId
        });
    }

    public getConnectionStatus() {
        return {
            connected: this.isConnected,
            sessionId: this.sessionId,
            userId: this.userId,
            eventBufferSize: this.eventBuffer.length,
            activeSessions: this.activeSessions.size,
            reconnectAttempts: this.reconnectAttempts
        };
    }

    public getEventHistory(limit = 50): NDJSONEvent[] {
        return this.eventBuffer.slice(-limit);
    }

    public clearEventBuffer() {
        this.eventBuffer = [];
    }

    // ============================= Manual Sync Triggers =============================

    public triggerOperatorFromNucleus(nucleusRole: string) {
        const operator = this.nucleusToOperatorMap.get(nucleusRole);
        if (operator && this.tier4System?.applyOperator) {
            this.tier4System.applyOperator(operator, {
                source: 'manual_trigger',
                nucleusRole: nucleusRole,
                collaborative: true
            });

            this.broadcastTier4State(operator, `Manual trigger from nucleus role: ${nucleusRole}`);
        }
    }

    public injectNDJSONEvent(eventStr: string) {
        try {
            const event = JSON.parse(eventStr);
            this.handleIncomingEvent(event);
        } catch (error) {
            console.error('Failed to inject NDJSON event:', error);
        }
    }

    // ============================= Cleanup =============================

    public disconnect() {
        if (this.ws) {
            this.leaveSession();
            this.ws.close();
            this.ws = null;
        }
        this.isConnected = false;
    }
}

// ============================= Integration Helper =============================

export function createTier4WebSocketSync(tier4System: any, userId?: string) {
    return new Tier4WebSocketSync(tier4System, userId);
}

// ============================= Usage Example =============================

/*
// Connect Tier-4 system to WebSocket relay
const wsSync = createTier4WebSocketSync(tier4System, 'user123');

// Monitor status
console.log(wsSync.getConnectionStatus());

// Manually trigger operators from nucleus events
wsSync.triggerOperatorFromNucleus('VIBRATE'); // → ST operator

// Inject events manually for testing
wsSync.injectNDJSONEvent('{"type":"nucleus_exec","id":"N1","role":"OPTIMIZATION","state":"active"}');

// Join collaborative session
wsSync.connectToSession('shared-session-id');

// View event history
console.log(wsSync.getEventHistory(10));

// Cleanup
wsSync.disconnect();
*/

export { Tier4WebSocketSync, NDJSONEvent, Tier4StateUpdate, CollaborativeSession };
