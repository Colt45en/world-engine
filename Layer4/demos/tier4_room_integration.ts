// tier4_room_integration.ts - Bridge between Tier-4 WebSocket system and Engine Room

import { Tier4Room, Tier4State } from '../websocket/roomAdapter';

export interface NDJSONEvent {
    type: string;
    ts: string;
    id?: string;
    role?: string;
    state?: string;
    tag?: string;
    thought?: string;
    cycle?: number;
    total?: number;
    nucleus?: string;
    need_tag?: string;
    tier4_operator?: string;
    tier4_suggested_macro?: string;
}

export interface Tier4Operator {
    name: string;
    matrix: number[][];
    bias: number[];
    kappaDelta: number;
}

export class Tier4RoomBridge {
    private readonly room: Tier4Room;
    private readonly websocketUrl: string;
    private ws: WebSocket | null = null;
    private reconnectAttempts = 0;
    private readonly maxReconnectAttempts = 5;
    private readonly reconnectDelay = 1000;
    private isConnected = false;

    // Current Tier-4 state
    private currentState: Tier4State = {
        x: [0, 0.5, 0.4, 0.6],
        kappa: 0.6,
        level: 0
    };

    // Nucleus → Tier-4 operator mappings
    private readonly nucleusOperatorMap: Record<string, string> = {
        'VIBRATE': 'ST',
        'OPTIMIZATION': 'UP',
        'STATE': 'CV',
        'SEED': 'RB',
        'energy': 'CH',
        'refined': 'PR',
        'condition': 'SL',
        'seed': 'MD'
    };

    // Tier-4 operators (simplified matrix transformations)
    private readonly operators: Record<string, Tier4Operator> = {
        ST: { name: 'Stabilize', matrix: [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], bias: [0, 0, 0, 0], kappaDelta: 0 },
        UP: { name: 'Update', matrix: [[1, 0, 0, 0], [0, 1.05, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1.05]], bias: [0, 0.01, 0, 0.01], kappaDelta: 0.05 },
        PR: { name: 'Progress', matrix: [[1, 0, 0, 0], [0, 0.9, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1.1]], bias: [0, -0.02, 0, 0.02], kappaDelta: 0.1 },
        CV: { name: 'Converge', matrix: [[0.95, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1.1, 0], [0, 0, 0, 1]], bias: [0, 0, 0.01, 0.01], kappaDelta: -0.05 },
        RB: { name: 'Rollback', matrix: [[1, 0, 0, 0], [0, 1.05, 0, 0], [0, 0, 1.05, 0], [0, 0, 0, 0.95]], bias: [0, 0.02, 0.03, -0.01], kappaDelta: -0.1 },
        RS: { name: 'Reset', matrix: [[0.8, 0, 0, 0], [0, 0.8, 0, 0], [0, 0, 0.8, 0], [0, 0, 0, 0.8]], bias: [0.1, 0.1, 0.1, 0.1], kappaDelta: -0.2 },
        CH: { name: 'Change', matrix: [[1.1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], bias: [0.05, 0, 0, 0], kappaDelta: 0.02 },
        SL: { name: 'Select', matrix: [[1, 0, 0, 0], [0, 1, 0.1, 0], [0, 0, 0.9, 0], [0, 0, 0, 1]], bias: [0, 0, 0, 0], kappaDelta: 0.03 },
        MD: { name: 'Multidim', matrix: [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1.15]], bias: [0, 0, 0, 0.05], kappaDelta: 0.08 }
    };

    constructor(roomWindow: Window, websocketUrl = 'ws://localhost:9000') {
        this.websocketUrl = websocketUrl;
        this.room = new Tier4Room(roomWindow, `tier4_room_${Date.now()}`);

        this.setupRoomHandlers();
        this.connectWebSocket();
    }

    private setupRoomHandlers() {
        // Handle operator applications from room panels
        window.addEventListener('message', (event) => {
            // Verify the origin of the received message
            const trustedOrigins = [window.location.origin];
            if (!trustedOrigins.includes(event.origin)) {
                console.warn(`Untrusted message origin: ${event.origin}`);
                return;
            }
            if (event.data.type === 'apply-operator') {
                this.applyOperator(event.data.operator);
            } else if (event.data.type === 'apply-macro') {
                this.applyMacro(event.data.macro);
            }
        });

        // Handle state load requests from room
        window.addEventListener('tier4-load-state', (event: Event) => {
            const customEvent = event as CustomEvent;
            const { state, cid } = customEvent.detail;
            this.loadState(state, cid);
        });
    }

    private connectWebSocket() {
        if (this.ws) {
            this.ws.close();
        }

        try {
            this.ws = new WebSocket(this.websocketUrl);

            this.ws.onopen = () => {
                this.isConnected = true;
                this.reconnectAttempts = 0;
                this.room.toast('Connected to Tier-4 WebSocket relay', 'info');
                console.log('Tier-4 Room Bridge: WebSocket connected');

                // Send initialization message
                this.sendWebSocketMessage({
                    type: 'tier4_room_init',
                    sessionId: this.room['sessionId'],
                    timestamp: Date.now()
                });
            };

            this.ws.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data);
                    this.handleWebSocketMessage(data);
                } catch (error) {
                    console.error('Tier-4 Room Bridge: Failed to parse WebSocket message', error);
                }
            };

            this.ws.onclose = () => {
                this.isConnected = false;
                this.room.toast('WebSocket connection lost, attempting to reconnect...', 'warning');
                this.attemptReconnect();
            };

            this.ws.onerror = (error) => {
                console.error('Tier-4 Room Bridge: WebSocket error', error);
                this.room.toast('WebSocket error occurred', 'error');
            };

        } catch (error) {
            console.error('Tier-4 Room Bridge: Failed to create WebSocket connection', error);
            this.attemptReconnect();
        }
    }

    private attemptReconnect() {
        if (this.reconnectAttempts >= this.maxReconnectAttempts) {
            this.room.toast('Max reconnection attempts reached', 'error');
            return;
        }

        this.reconnectAttempts++;
        const delay = this.reconnectDelay * Math.pow(2, this.reconnectAttempts - 1);

        setTimeout(() => {
            console.log(`Tier-4 Room Bridge: Reconnection attempt ${this.reconnectAttempts}`);
            this.connectWebSocket();
        }, delay);
    }

    private sendWebSocketMessage(message: any) {
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify(message));
        }
    }

    private handleWebSocketMessage(data: NDJSONEvent) {
        console.log('Tier-4 Room Bridge: Received WebSocket message', data);

        // Map nucleus events to Tier-4 operators
        let operatorToApply: string | null = null;

        if (data.type === 'nucleus_exec' && data.role) {
            operatorToApply = this.nucleusOperatorMap[data.role];
        } else if (data.type === 'memory_store' && data.tag) {
            operatorToApply = this.nucleusOperatorMap[data.tag];
        } else if (data.tier4_operator) {
            operatorToApply = data.tier4_operator;
        }

        // Auto-apply operator if mapped
        if (operatorToApply && this.operators[operatorToApply]) {
            setTimeout(() => {
                this.applyOperator(operatorToApply, {
                    source: 'nucleus_auto',
                    triggerEvent: data.type,
                    triggerId: data.id || data.nucleus || 'unknown'
                });
            }, 100);
        }

        // Handle cycle events for macro suggestions
        if (data.type === 'cycle_start' && data.tier4_suggested_macro) {
            this.room.toast(`Suggested macro: ${data.tier4_suggested_macro}`, 'info');
            // Could auto-apply macro here based on configuration
        }

        // Handle hallucination guards
        if (data.type === 'hallucination_guard') {
            this.room.toast(`Hallucination guard triggered for ${data.nucleus}`, 'warning');
        }

        // Forward all events to room for visualization
        this.room.publishEvent({
            id: `ws_${Date.now()}_${Math.random().toString(36).substr(2, 8)}`,
            button: data.type,
            inputCid: 'ws_stream',
            outputCid: 'room_display',
            timestamp: new Date(data.ts).getTime() || Date.now(),
            meta: data
        });
    }

    /**
     * Apply a Tier-4 operator to the current state
     */
    public applyOperator(operatorName: string, meta?: any) {
        const operator = this.operators[operatorName];
        if (!operator) {
            this.room.toast(`Unknown operator: ${operatorName}`, 'error');
            return;
        }

        const previousState = { ...this.currentState };
        const newState = this.transformState(this.currentState, operator);

        // Update current state
        this.currentState = newState;

        // Notify room of the transformation
        this.room.applyOperator(operatorName, previousState, newState);

        // Send to WebSocket if connected
        this.sendWebSocketMessage({
            type: 'tier4_operator_applied',
            operator: operatorName,
            previousState,
            newState,
            meta,
            timestamp: Date.now()
        });

        // Update metrics
        this.updateMetrics();

        console.log(`Tier-4 Room Bridge: Applied ${operatorName}`, { previousState, newState });
    }

    /**
     * Apply a Three Ides macro (sequence of operators)
     */
    public applyMacro(macroName: string) {
        const macros: Record<string, string[]> = {
            'IDE_A': ['ST', 'SL', 'CP'],
            'IDE_B': ['CV', 'PR', 'RC'],
            'IDE_C': ['TL', 'RB', 'MD'],
            'MERGE_ABC': ['CV', 'CV', 'ST', 'SL', 'CV', 'PR', 'RB', 'MD']
        };

        const sequence = macros[macroName];
        if (!sequence) {
            this.room.toast(`Unknown macro: ${macroName}`, 'error');
            return;
        }

        this.room.toast(`Executing macro: ${macroName}`, 'info');

        // Apply operators in sequence with delays
        sequence.forEach((op, index) => {
            setTimeout(() => {
                // Use available operators, fall back to ST for unknown ones
                const operatorName = this.operators[op] ? op : 'ST';
                this.applyOperator(operatorName, {
                    source: 'macro',
                    macro: macroName,
                    step: index + 1,
                    total: sequence.length
                });

                if (index === sequence.length - 1) {
                    this.room.toast(`Macro ${macroName} completed`, 'info');
                }
            }, index * 300);
        });
    }

    /**
     * Load a specific state (called from room snapshot panel)
     */
    private loadState(state: Tier4State, cid: string) {
        this.currentState = { ...state };

        // Send state change to WebSocket
        this.sendWebSocketMessage({
            type: 'tier4_state_loaded',
            state: this.currentState,
            cid,
            timestamp: Date.now()
        });

        console.log(`Tier-4 Room Bridge: Loaded state ${cid}`, state);
    }

    /**
     * Transform state using operator matrix math
     */
    private transformState(state: Tier4State, operator: Tier4Operator): Tier4State {
        const newX = new Array(state.x.length);

        // Matrix-vector multiplication: Mx + b
        for (let i = 0; i < state.x.length; i++) {
            let sum = 0;
            for (let j = 0; j < state.x.length; j++) {
                const matrixValue = operator.matrix[i]?.[j] || (i === j ? 1 : 0);
                sum += matrixValue * state.x[j];
            }
            newX[i] = sum + (operator.bias[i] || 0);
        }

        // Apply kappa change and clamp
        const newKappa = Math.max(0, Math.min(1, state.kappa + operator.kappaDelta));

        // Level changes for certain operators
        let levelChange = 0;
        if (operator.name === 'Update') levelChange = 1;
        else if (operator.name === 'Rollback') levelChange = -1;
        else if (operator.name === 'Reset') levelChange = -state.level;

        return {
            x: newX,
            kappa: newKappa,
            level: Math.max(0, state.level + levelChange),
            operator: operator.name,
            meta: {
                transformation: operator.name,
                timestamp: Date.now()
            }
        };
    }

    /**
     * Update performance metrics in room
     */
    private updateMetrics() {
        const metrics = {
            opsRate: '1.2/s',
            kappaDelta: this.currentState.kappa.toFixed(3),
            transitions: this.room.getStateHistory().size.toString(),
            memory: `${Math.round(this.room.getStateHistory().size * 0.5)}KB`,
            chartValue: this.currentState.kappa
        };

        this.room.updateMetrics(metrics);
    }

    /**
     * Get current Tier-4 state
     */
    public getCurrentState(): Tier4State {
        return { ...this.currentState };
    }

    /**
     * Set current state (for external updates)
     */
    public setState(state: Tier4State) {
        this.currentState = { ...state };
        this.updateMetrics();
    }

    /**
     * Check if WebSocket is connected
     */
    public isWebSocketConnected(): boolean {
        return this.isConnected;
    }

    /**
     * Get the room instance
     */
    public getRoom(): Tier4Room {
        return this.room;
    }

    /**
     * Manually trigger a nucleus event (for testing)
     */
    public triggerNucleusEvent(role: 'VIBRATE' | 'OPTIMIZATION' | 'STATE' | 'SEED') {
        const event: NDJSONEvent = {
            type: 'nucleus_exec',
            ts: new Date().toISOString(),
            id: `N${Math.floor(Math.random() * 4) + 1}`,
            role,
            state: 'active',
            tier4_operator: this.nucleusOperatorMap[role]
        };

        this.handleWebSocketMessage(event);
    }

    /**
     * Cleanup and disconnect
     */
    public disconnect() {
        if (this.ws) {
            this.ws.close();
            this.ws = null;
        }
        this.isConnected = false;
    }
}

// Export a factory function for easy initialization
export function createTier4RoomBridge(engineRoomIframe: HTMLIFrameElement, websocketUrl?: string): Promise<Tier4RoomBridge> {
    return new Promise((resolve, reject) => {
        const iframe = engineRoomIframe;

        if (!iframe.contentWindow) {
            reject(new Error('Engine room iframe not loaded'));
            return;
        }

        // Wait for iframe to load
        const checkReady = () => {
            if (iframe.contentWindow) {
                const bridge = new Tier4RoomBridge(iframe.contentWindow, websocketUrl);

                // Wait for room to be ready
                bridge.getRoom().whenReady(() => {
                    resolve(bridge);
                });
            } else {
                setTimeout(checkReady, 100);
            }
        };

        // If the iframe's content is already loaded, proceed; otherwise, wait for 'load' event
        if (iframe.contentDocument && iframe.contentDocument.readyState === 'complete') {
            setTimeout(checkReady, 100);
        } else {
            iframe.addEventListener('load', () => {
                setTimeout(checkReady, 100);
            });
        }
    });
}

export default Tier4RoomBridge;
