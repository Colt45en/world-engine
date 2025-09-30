// Tier 4 Room Integration Bridge
// Provides the WebSocket bridge between React components and the Engine Room

import { Tier4State, DefaultRoomAdapter } from '../components/roomAdapter';

export interface Tier4RoomBridge {
    /** Apply a Tier-4 operator */
    applyOperator(operator: string, meta?: any): void;

    /** Apply a Three Ides macro sequence */
    applyMacro(macro: string): void;

    /** Get current Tier-4 state */
    getCurrentState(): Tier4State;

    /** Set the current state */
    setState(state: Tier4State): void;

    /** Trigger a nucleus intelligence event */
    triggerNucleusEvent(role: 'VIBRATE' | 'OPTIMIZATION' | 'STATE' | 'SEED'): void;

    /** Get the underlying room interface */
    getRoom(): EngineRoomInterface;

    /** Check WebSocket connection status */
    isWebSocketConnected(): boolean;

    /** Send custom WebSocket message */
    sendWebSocketMessage(message: any): void;

    /** Disconnect the bridge */
    disconnect(): void;
}

export interface EngineRoomInterface {
    /** Show a toast message */
    toast(message: string): void;

    /** Add a custom panel to the room */
    addPanel(config: {
        wall: 'front' | 'left' | 'right' | 'back' | 'floor' | 'ceil';
        x: number;
        y: number;
        w: number;
        h: number;
        title?: string;
        html?: string;
    }): void;

    /** Update room lighting */
    setLighting(config: { intensity: number; color: string }): void;

    /** Play audio in the room */
    playAudio(url: string, volume?: number): void;

    /** Get room statistics */
    getStats(): any;
}

class Tier4RoomBridgeImpl implements Tier4RoomBridge {
    private iframe: HTMLIFrameElement;
    private websocketUrl: string;
    private websocket: WebSocket | null = null;
    private roomAdapter: DefaultRoomAdapter;
    private messageQueue: any[] = [];
    private isReady = false;

    constructor(iframe: HTMLIFrameElement, websocketUrl: string) {
        this.iframe = iframe;
        this.websocketUrl = websocketUrl;
        this.roomAdapter = new DefaultRoomAdapter();

        this.initializeWebSocket();
        this.setupIframeListeners();
    }

    private async initializeWebSocket() {
        try {
            this.websocket = new WebSocket(this.websocketUrl);

            this.websocket.onopen = () => {
                console.log('[Tier4Bridge] WebSocket connected');
                this.flushMessageQueue();
            };

            this.websocket.onmessage = (event) => {
                try {
                    const message = JSON.parse(event.data);
                    this.handleWebSocketMessage(message);
                } catch (error) {
                    console.error('[Tier4Bridge] Failed to parse WebSocket message:', error);
                }
            };

            this.websocket.onclose = () => {
                console.log('[Tier4Bridge] WebSocket disconnected');
                // Attempt to reconnect after 3 seconds
                setTimeout(() => this.initializeWebSocket(), 3000);
            };

            this.websocket.onerror = (error) => {
                console.error('[Tier4Bridge] WebSocket error:', error);
            };

        } catch (error) {
            console.error('[Tier4Bridge] Failed to initialize WebSocket:', error);
        }
    }

    private setupIframeListeners() {
        window.addEventListener('message', (event) => {
            if (event.source !== this.iframe.contentWindow) return;

            try {
                const message = typeof event.data === 'string' ? JSON.parse(event.data) : event.data;
                this.handleIframeMessage(message);
            } catch (error) {
                console.error('[Tier4Bridge] Failed to handle iframe message:', error);
            }
        });

        // Wait for iframe to be ready
        const checkReady = () => {
            if (this.iframe.contentWindow) {
                this.sendToIframe({ type: 'tier4.ping' });
                this.isReady = true;
            } else {
                setTimeout(checkReady, 100);
            }
        };

        checkReady();
    }

    private handleWebSocketMessage(message: any) {
        switch (message.type) {
            case 'state.update':
                if (message.data) {
                    const state = this.roomAdapter.deserialize(JSON.stringify(message.data));
                    this.roomAdapter.setState(state);
                    this.sendToIframe({
                        type: 'tier4.state.update',
                        data: state
                    });
                }
                break;

            case 'operator.applied':
                if (message.data) {
                    const { operator, result } = message.data;
                    this.sendToIframe({
                        type: 'tier4.operator.result',
                        data: { operator, result }
                    });

                    // Emit custom event for React components
                    const event = new CustomEvent('tier4-operator-applied', {
                        detail: {
                            operator,
                            previousState: this.roomAdapter.getState(),
                            newState: result
                        }
                    });
                    window.dispatchEvent(event);
                }
                break;

            case 'nucleus.event':
                if (message.data) {
                    this.handleNucleusEvent(message.data);
                }
                break;

            case 'clock.sync':
                if (message.data) {
                    this.sendToIframe({
                        type: 'tier4.clock.sync',
                        data: message.data
                    });
                }
                break;

            default:
                console.log('[Tier4Bridge] Unknown WebSocket message:', message);
        }
    }

    private handleIframeMessage(message: any) {
        switch (message.type) {
            case 'tier4.pong':
                console.log('[Tier4Bridge] Iframe is ready');
                break;

            case 'tier4.operator.request':
                if (message.data) {
                    this.applyOperator(message.data.operator, message.data.meta);
                }
                break;

            case 'tier4.state.request':
                this.sendToIframe({
                    type: 'tier4.state.response',
                    data: this.roomAdapter.getState()
                });
                break;

            case 'tier4.toast':
                if (message.data && message.data.message) {
                    console.log('[Tier4Bridge] Toast:', message.data.message);
                }
                break;

            default:
                console.log('[Tier4Bridge] Unknown iframe message:', message);
        }
    }

    private handleNucleusEvent(eventData: any) {
        const { role, intensity, metadata } = eventData;

        // Apply nucleus transformation to current state
        const currentState = this.roomAdapter.getState();
        const newState = { ...currentState };

        // Update nucleus state
        newState.nucleus = {
            role: role,
            active: true,
            intensity: intensity || 1
        };

        // Apply role-specific transformations
        switch (role) {
            case 'VIBRATE':
                // Add subtle oscillations to the state
                newState.x = newState.x.map((x, i) =>
                    Math.max(0, Math.min(1, x + Math.sin(Date.now() / 200 + i) * 0.02))
                ) as [number, number, number, number];
                break;

            case 'OPTIMIZATION': {
                // Move toward optimal configuration
                const target = [0.3, 0.7, 0.5, 0.8];
                newState.x = newState.x.map((x, i) =>
                    x + (target[i] - x) * 0.1
                ) as [number, number, number, number];
                break;
            }

            case 'STATE': {
                // Stabilize current state
                const avg = newState.x.reduce((sum, x) => sum + x, 0) / 4;
                newState.x = newState.x.map(x => x * 0.9 + avg * 0.1) as [number, number, number, number];
                break;
            }

            case 'SEED':
                // Introduce controlled randomness
                newState.x = newState.x.map(x =>
                    Math.max(0, Math.min(1, x + (Math.random() - 0.5) * 0.1))
                ) as [number, number, number, number];
                break;
        }

        this.roomAdapter.setState(newState);

        // Send update to iframe
        this.sendToIframe({
            type: 'tier4.nucleus.event',
            data: { role, state: newState, metadata }
        });

        // Emit event for React components
        const event = new CustomEvent('tier4-nucleus-event', {
            detail: { role, state: newState, metadata }
        });
        window.dispatchEvent(event);
    }

    private sendToIframe(message: any) {
        if (this.iframe.contentWindow && this.isReady) {
            this.iframe.contentWindow.postMessage(JSON.stringify(message), '*');
        }
    }

    private sendToWebSocket(message: any) {
        if (this.websocket && this.websocket.readyState === WebSocket.OPEN) {
            this.websocket.send(JSON.stringify(message));
        } else {
            this.messageQueue.push(message);
        }
    }

    private flushMessageQueue() {
        while (this.messageQueue.length > 0) {
            const message = this.messageQueue.shift();
            if (this.websocket && this.websocket.readyState === WebSocket.OPEN) {
                this.websocket.send(JSON.stringify(message));
            } else {
                // Re-queue if still not connected
                this.messageQueue.unshift(message);
                break;
            }
        }
    }

    // Public API implementation
    applyOperator(operator: string, meta?: any): void {
        const previousState = this.roomAdapter.getState();
        const newState = this.roomAdapter.applyOperator(operator, meta);

        // Send to WebSocket
        this.sendToWebSocket({
            type: 'operator.apply',
            data: { operator, previousState, newState, meta }
        });

        // Send to iframe
        this.sendToIframe({
            type: 'tier4.operator.apply',
            data: { operator, state: newState, meta }
        });

        // Emit event for React components
        const event = new CustomEvent('tier4-operator-applied', {
            detail: { operator, previousState, newState }
        });
        window.dispatchEvent(event);
    }

    applyMacro(macro: string): void {
        // Parse macro into individual operators
        const operators = this.parseMacro(macro);

        operators.forEach((operator, index) => {
            setTimeout(() => {
                this.applyOperator(operator, { macro, step: index });
            }, index * 100); // 100ms delay between operations
        });
    }

    getCurrentState(): Tier4State {
        return this.roomAdapter.getState();
    }

    setState(state: Tier4State): void {
        this.roomAdapter.setState(state);

        // Send to WebSocket
        this.sendToWebSocket({
            type: 'state.set',
            data: state
        });

        // Send to iframe
        this.sendToIframe({
            type: 'tier4.state.set',
            data: state
        });
    }

    triggerNucleusEvent(role: 'VIBRATE' | 'OPTIMIZATION' | 'STATE' | 'SEED'): void {
        const eventData = {
            role,
            timestamp: Date.now(),
            intensity: 1,
            metadata: { triggered: true }
        };

        this.sendToWebSocket({
            type: 'nucleus.trigger',
            data: eventData
        });

        // Also handle locally
        this.handleNucleusEvent(eventData);
    }

    getRoom(): EngineRoomInterface {
        return {
            toast: (message: string) => {
                this.sendToIframe({
                    type: 'tier4.toast',
                    data: { message }
                });
            },

            addPanel: (config) => {
                this.sendToIframe({
                    type: 'tier4.panel.add',
                    data: config
                });
            },

            setLighting: (config) => {
                this.sendToIframe({
                    type: 'tier4.lighting.set',
                    data: config
                });
            },

            playAudio: (url: string, volume = 1) => {
                this.sendToIframe({
                    type: 'tier4.audio.play',
                    data: { url, volume }
                });
            },

            getStats: () => {
                return {
                    state: this.roomAdapter.getState(),
                    connected: this.isWebSocketConnected(),
                    ready: this.isReady
                };
            }
        };
    }

    isWebSocketConnected(): boolean {
        return this.websocket !== null && this.websocket.readyState === WebSocket.OPEN;
    }

    sendWebSocketMessage(message: any): void {
        this.sendToWebSocket(message);
    }

    disconnect(): void {
        if (this.websocket) {
            this.websocket.close();
            this.websocket = null;
        }
        this.isReady = false;
        this.messageQueue = [];
    }

    private parseMacro(macro: string): string[] {
        // Simple macro parsing - extend as needed
        const macros: { [key: string]: string[] } = {
            'RESET': ['CV', 'SC', 'CV'],
            'CHAOS': ['NG', 'MV', 'SA', 'MO'],
            'STABILIZE': ['CV', 'SC', 'CV', 'CN'],
            'EXPLORE': ['MV', 'RB', 'UP', 'TL'],
            'NUCLEUS_WAKE': ['PS', 'UP', 'CN', 'MO'],
            'TIER_UP': ['UP', 'SC', 'CN', 'UP']
        };

        if (macros[macro]) {
            return macros[macro];
        }

        // Fallback: try to parse as space-separated operators
        return macro.split(' ').filter(op => op.length > 0);
    }
}

// Factory function to create the bridge
export async function createTier4RoomBridge(
    iframe: HTMLIFrameElement,
    websocketUrl: string = 'ws://localhost:9000'
): Promise<Tier4RoomBridge> {
    return new Promise((resolve, reject) => {
        try {
            const bridge = new Tier4RoomBridgeImpl(iframe, websocketUrl);

            // Wait a moment for initialization
            setTimeout(() => {
                resolve(bridge);
            }, 500);

        } catch (error) {
            reject(error);
        }
    });
}

// Export types for external use
export { Tier4State } from '../components/roomAdapter';
