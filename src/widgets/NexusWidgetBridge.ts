// Nexus Widget Bridge Communication System
// Allows widgets to communicate with the main Nexus system

interface WidgetMessage {
    type: string;
    data: any;
    widgetId: string;
    timestamp: number;
}

interface BridgeCommand {
    command: string;
    args: any[];
    callback?: (result: any) => void;
}

export class NexusWidgetBridge {
    private static instance: NexusWidgetBridge;
    private bridge: any = null;
    private connectedWidgets = new Set<string>();
    private messageHandlers = new Map<string, (message: WidgetMessage) => void>();

    static getInstance(): NexusWidgetBridge {
        if (!NexusWidgetBridge.instance) {
            NexusWidgetBridge.instance = new NexusWidgetBridge();
        }
        return NexusWidgetBridge.instance;
    }

    constructor() {
        // Listen for messages from widgets
        window.addEventListener('message', this.handleWidgetMessage.bind(this));
    }

    setBridge(bridge: any) {
        this.bridge = bridge;
        console.log('🌉 Nexus Widget Bridge activated');
    }

    private handleWidgetMessage(event: MessageEvent) {
        if (event.origin === window.location.origin && event.data.type?.startsWith('NEXUS_')) {
            const message: WidgetMessage = {
                type: event.data.type,
                data: event.data.data,
                widgetId: event.data.widgetId || 'unknown',
                timestamp: Date.now()
            };

            this.processMessage(message);
        }
    }

    private processMessage(message: WidgetMessage) {
        console.log('📨 Widget message received:', message);

        switch (message.type) {
            case 'NEXUS_WIDGET_READY':
                this.handleWidgetReady(message);
                break;
            case 'NEXUS_QUERY_REQUEST':
                this.handleQueryRequest(message);
                break;
            case 'NEXUS_TRAINING_CONTROL':
                this.handleTrainingControl(message);
                break;
            case 'NEXUS_STATUS_REQUEST':
                this.handleStatusRequest(message);
                break;
            case 'NEXUS_BRIDGE_COMMAND':
                this.handleBridgeCommand(message);
                break;
            default:
                console.warn('Unknown widget message type:', message.type);
        }

        // Call custom handlers
        const handler = this.messageHandlers.get(message.type);
        if (handler) {
            handler(message);
        }
    }

    private handleWidgetReady(message: WidgetMessage) {
        this.connectedWidgets.add(message.widgetId);

        // Send bridge capabilities to widget
        this.sendToWidget(message.widgetId, {
            type: 'NEXUS_BRIDGE_CAPABILITIES',
            data: {
                hasQuery: !!this.bridge?.query,
                hasTraining: !!this.bridge?.startTraining,
                isReady: this.bridge?.isReady || false,
                trainingActive: this.bridge?.trainingState?.isTraining || false
            }
        });
    }

    private async handleQueryRequest(message: WidgetMessage) {
        if (!this.bridge?.query) {
            this.sendToWidget(message.widgetId, {
                type: 'NEXUS_QUERY_RESPONSE',
                data: { error: 'Query bridge not available' }
            });
            return;
        }

        try {
            const response = await this.bridge.query(message.data.question);
            this.sendToWidget(message.widgetId, {
                type: 'NEXUS_QUERY_RESPONSE',
                data: { response, success: true }
            });
        } catch (error) {
            this.sendToWidget(message.widgetId, {
                type: 'NEXUS_QUERY_RESPONSE',
                data: { error: error instanceof Error ? error.message : String(error), success: false }
            });
        }
    }

    private handleTrainingControl(message: WidgetMessage) {
        const { action } = message.data;
        let result = { success: false, message: 'Training control not available' };

        if (action === 'start' && this.bridge?.startTraining) {
            result = {
                success: this.bridge.startTraining(),
                message: 'Training started'
            };
        } else if (action === 'stop' && this.bridge?.stopTraining) {
            this.bridge.stopTraining();
            result = {
                success: true,
                message: 'Training stopped'
            };
        }

        this.sendToWidget(message.widgetId, {
            type: 'NEXUS_TRAINING_RESPONSE',
            data: result
        });
    }

    private handleStatusRequest(message: WidgetMessage) {
        const status = {
            bridge: {
                connected: !!this.bridge,
                ready: this.bridge?.isReady || false
            },
            training: this.bridge?.trainingState || {
                isTraining: false,
                dataPoints: 0,
                errors: 0,
                memoryUsage: 0
            },
            widgets: {
                connected: Array.from(this.connectedWidgets),
                count: this.connectedWidgets.size
            },
            system: {
                timestamp: Date.now(),
                uptime: performance.now()
            }
        };

        this.sendToWidget(message.widgetId, {
            type: 'NEXUS_STATUS_RESPONSE',
            data: status
        });
    }

    private handleBridgeCommand(message: WidgetMessage) {
        const { command, args } = message.data as BridgeCommand;

        if (!this.bridge) {
            this.sendToWidget(message.widgetId, {
                type: 'NEXUS_COMMAND_RESPONSE',
                data: { error: 'Bridge not available' }
            });
            return;
        }

        try {
            let result;

            switch (command) {
                case 'getTrainingState':
                    result = this.bridge.trainingState;
                    break;
                case 'isReady':
                    result = this.bridge.isReady;
                    break;
                default:
                    // Try to call method dynamically
                    if (typeof this.bridge[command] === 'function') {
                        result = this.bridge[command](...args);
                    } else {
                        throw new Error(`Unknown command: ${command}`);
                    }
            }

            this.sendToWidget(message.widgetId, {
                type: 'NEXUS_COMMAND_RESPONSE',
                data: { result, success: true }
            });
        } catch (error) {
            this.sendToWidget(message.widgetId, {
                type: 'NEXUS_COMMAND_RESPONSE',
                data: { error: error instanceof Error ? error.message : String(error), success: false }
            });
        }
    }

    private sendToWidget(widgetId: string, message: any) {
        // Find widget iframe and send message
        const iframe = document.getElementById(`widget-${widgetId}`) as HTMLIFrameElement;
        if (iframe && iframe.contentWindow) {
            iframe.contentWindow.postMessage({
                ...message,
                targetWidget: widgetId
            }, '*');
        }
    }

    // Public API for widgets to register message handlers
    onMessage(type: string, handler: (message: WidgetMessage) => void) {
        this.messageHandlers.set(type, handler);
    }

    // Send message to all connected widgets
    broadcast(message: any) {
        this.connectedWidgets.forEach(widgetId => {
            this.sendToWidget(widgetId, message);
        });
    }

    // Get bridge status
    getStatus() {
        return {
            bridgeConnected: !!this.bridge,
            connectedWidgets: Array.from(this.connectedWidgets),
            isReady: this.bridge?.isReady || false
        };
    }
}

// Widget-side communication helper
// This code should be injected into widgets that want to communicate with Nexus
export const createWidgetConnector = (widgetId: string) => {
    const connector = {
        connected: false,
        bridge: null as any,

        // Initialize connection to Nexus
        connect() {
            window.parent.postMessage({
                type: 'NEXUS_WIDGET_READY',
                widgetId,
                data: {
                    capabilities: ['query', 'training', 'status'],
                    version: '1.0.0'
                }
            }, '*');

            // Listen for responses
            window.addEventListener('message', (event) => {
                if (event.data.targetWidget === widgetId) {
                    this.handleMessage(event.data);
                }
            });

            this.connected = true;
            console.log(`🔗 Widget ${widgetId} connected to Nexus Bridge`);
        },

        // Handle messages from Nexus
        handleMessage(message: any) {
            switch (message.type) {
                case 'NEXUS_BRIDGE_CAPABILITIES':
                    this.bridge = message.data;
                    this.onBridgeReady?.(this.bridge);
                    break;
                case 'NEXUS_QUERY_RESPONSE':
                    this.onQueryResponse?.(message.data);
                    break;
                case 'NEXUS_TRAINING_RESPONSE':
                    this.onTrainingResponse?.(message.data);
                    break;
                case 'NEXUS_STATUS_RESPONSE':
                    this.onStatusResponse?.(message.data);
                    break;
                case 'NEXUS_COMMAND_RESPONSE':
                    this.onCommandResponse?.(message.data);
                    break;
            }
        },

        // Send query to Nexus
        query(question: string) {
            window.parent.postMessage({
                type: 'NEXUS_QUERY_REQUEST',
                widgetId,
                data: { question }
            }, '*');
        },

        // Control training
        startTraining() {
            window.parent.postMessage({
                type: 'NEXUS_TRAINING_CONTROL',
                widgetId,
                data: { action: 'start' }
            }, '*');
        },

        stopTraining() {
            window.parent.postMessage({
                type: 'NEXUS_TRAINING_CONTROL',
                widgetId,
                data: { action: 'stop' }
            }, '*');
        },

        // Request status
        getStatus() {
            window.parent.postMessage({
                type: 'NEXUS_STATUS_REQUEST',
                widgetId,
                data: {}
            }, '*');
        },

        // Execute bridge command
        executeCommand(command: string, ...args: any[]) {
            window.parent.postMessage({
                type: 'NEXUS_BRIDGE_COMMAND',
                widgetId,
                data: { command, args }
            }, '*');
        },

        // Event handlers (to be set by widget)
        onBridgeReady: null as ((bridge: any) => void) | null,
        onQueryResponse: null as ((response: any) => void) | null,
        onTrainingResponse: null as ((response: any) => void) | null,
        onStatusResponse: null as ((status: any) => void) | null,
        onCommandResponse: null as ((response: any) => void) | null
    };

    return connector;
};

// Auto-initialize bridge for main app
export const nexusWidgetBridge = NexusWidgetBridge.getInstance();
