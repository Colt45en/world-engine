/**
 * Nexus Widget Connector Script
 * Include this script in any widget to connect it to the Nexus Bridge
 *
 * Usage in widget:
 * <script src="nexus-widget-connector.js"></script>
 * <script>
 *   const nexus = createNexusConnector('my-widget-id');
 *   nexus.connect();
 *
 *   nexus.onBridgeReady = (bridge) => {
 *     console.log('Connected to Nexus!', bridge);
 *   };
 * </script>
 */

(function (global) {
    'use strict';

    // Check if we're in an iframe (widget context)
    const isInIframe = window !== window.parent;

    function createNexusConnector(widgetId) {
        if (!widgetId) {
            throw new Error('Widget ID is required');
        }

        const connector = {
            widgetId: widgetId,
            connected: false,
            bridge: null,
            eventHandlers: {},

            // Initialize connection to Nexus
            connect() {
                if (!isInIframe) {
                    console.warn('Nexus connector should only be used in iframe widgets');
                    return false;
                }

                console.log(`🔌 Connecting widget ${widgetId} to Nexus Bridge...`);

                // Send ready signal to parent
                window.parent.postMessage({
                    type: 'NEXUS_WIDGET_READY',
                    widgetId: widgetId,
                    data: {
                        capabilities: ['query', 'training', 'status'],
                        version: '1.0.0',
                        url: window.location.href,
                        title: document.title
                    }
                }, '*');

                // Listen for responses from Nexus
                window.addEventListener('message', (event) => {
                    if (event.data.targetWidget === widgetId || !event.data.targetWidget) {
                        this.handleMessage(event.data);
                    }
                });

                this.connected = true;
                this.emit('connected');
                return true;
            },

            // Handle messages from Nexus
            handleMessage(message) {
                switch (message.type) {
                    case 'NEXUS_BRIDGE_CAPABILITIES':
                        this.bridge = message.data;
                        this.emit('bridgeReady', this.bridge);
                        break;
                    case 'NEXUS_QUERY_RESPONSE':
                        this.emit('queryResponse', message.data);
                        break;
                    case 'NEXUS_TRAINING_RESPONSE':
                        this.emit('trainingResponse', message.data);
                        break;
                    case 'NEXUS_STATUS_RESPONSE':
                        this.emit('statusResponse', message.data);
                        break;
                    case 'NEXUS_COMMAND_RESPONSE':
                        this.emit('commandResponse', message.data);
                        break;
                    case 'NEXUS_BROADCAST':
                        this.emit('broadcast', message.data);
                        break;
                }
            },

            // Send query to Nexus AI
            query(question, callback) {
                if (!this.connected) {
                    console.error('Widget not connected to Nexus');
                    return;
                }

                if (callback) {
                    this.once('queryResponse', callback);
                }

                window.parent.postMessage({
                    type: 'NEXUS_QUERY_REQUEST',
                    widgetId: widgetId,
                    data: { question }
                }, '*');
            },

            // Control training
            startTraining(callback) {
                if (!this.connected) return;

                if (callback) {
                    this.once('trainingResponse', callback);
                }

                window.parent.postMessage({
                    type: 'NEXUS_TRAINING_CONTROL',
                    widgetId: widgetId,
                    data: { action: 'start' }
                }, '*');
            },

            stopTraining(callback) {
                if (!this.connected) return;

                if (callback) {
                    this.once('trainingResponse', callback);
                }

                window.parent.postMessage({
                    type: 'NEXUS_TRAINING_CONTROL',
                    widgetId: widgetId,
                    data: { action: 'stop' }
                }, '*');
            },

            // Request status from Nexus
            getStatus(callback) {
                if (!this.connected) return;

                if (callback) {
                    this.once('statusResponse', callback);
                }

                window.parent.postMessage({
                    type: 'NEXUS_STATUS_REQUEST',
                    widgetId: widgetId,
                    data: {}
                }, '*');
            },

            // Execute bridge command
            executeCommand(command, args, callback) {
                if (!this.connected) return;

                if (callback) {
                    this.once('commandResponse', callback);
                }

                window.parent.postMessage({
                    type: 'NEXUS_BRIDGE_COMMAND',
                    widgetId: widgetId,
                    data: { command, args: args || [] }
                }, '*');
            },

            // Event system
            on(event, handler) {
                if (!this.eventHandlers[event]) {
                    this.eventHandlers[event] = [];
                }
                this.eventHandlers[event].push(handler);
            },

            once(event, handler) {
                const onceHandler = (...args) => {
                    handler(...args);
                    this.off(event, onceHandler);
                };
                this.on(event, onceHandler);
            },

            off(event, handler) {
                if (!this.eventHandlers[event]) return;

                const index = this.eventHandlers[event].indexOf(handler);
                if (index > -1) {
                    this.eventHandlers[event].splice(index, 1);
                }
            },

            emit(event, ...args) {
                if (!this.eventHandlers[event]) return;

                this.eventHandlers[event].forEach(handler => {
                    try {
                        handler(...args);
                    } catch (error) {
                        console.error('Error in event handler:', error);
                    }
                });
            },

            // Helper methods for common UI updates
            updateStatus(status) {
                const statusEl = document.getElementById('nexus-status');
                if (statusEl) {
                    statusEl.textContent = status;
                }
            },

            showMessage(message, type = 'info') {
                const messageEl = document.getElementById('nexus-message');
                if (messageEl) {
                    messageEl.textContent = message;
                    messageEl.className = `nexus-message nexus-message-${type}`;
                }
            },

            // Create UI elements for Nexus integration
            createNexusUI() {
                const style = document.createElement('style');
                style.textContent = `
          .nexus-integration {
            position: fixed;
            top: 10px;
            right: 10px;
            background: rgba(102, 126, 234, 0.9);
            color: white;
            padding: 10px;
            border-radius: 8px;
            font-family: 'Segoe UI', sans-serif;
            font-size: 12px;
            z-index: 10000;
            backdrop-filter: blur(10px);
          }

          .nexus-status {
            display: flex;
            align-items: center;
            gap: 5px;
          }

          .nexus-indicator {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: #4CAF50;
            animation: pulse 2s infinite;
          }

          .nexus-controls {
            margin-top: 8px;
            display: flex;
            gap: 5px;
          }

          .nexus-btn {
            background: rgba(255,255,255,0.2);
            border: none;
            color: white;
            padding: 4px 8px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 10px;
          }

          .nexus-btn:hover {
            background: rgba(255,255,255,0.3);
          }

          .nexus-message {
            margin-top: 5px;
            padding: 4px;
            border-radius: 4px;
            font-size: 10px;
          }

          .nexus-message-info { background: rgba(33, 150, 243, 0.3); }
          .nexus-message-success { background: rgba(76, 175, 80, 0.3); }
          .nexus-message-error { background: rgba(244, 67, 54, 0.3); }

          @keyframes pulse {
            0% { opacity: 1; }
            50% { opacity: 0.5; }
            100% { opacity: 1; }
          }
        `;
                document.head.appendChild(style);

                const ui = document.createElement('div');
                ui.className = 'nexus-integration';
                ui.innerHTML = `
          <div class="nexus-status">
            <div class="nexus-indicator"></div>
            <span>Nexus Connected</span>
          </div>
          <div class="nexus-controls">
            <button class="nexus-btn" onclick="nexusConnector.getStatus()">Status</button>
            <button class="nexus-btn" onclick="nexusConnector.query('Hello from widget')">Test</button>
          </div>
          <div id="nexus-message" class="nexus-message" style="display: none;"></div>
        `;

                document.body.appendChild(ui);

                // Make connector globally available for UI
                window.nexusConnector = this;
            }
        };

        return connector;
    }

    // Auto-initialize if widget ID is provided in script tag
    const scriptTag = document.currentScript;
    if (scriptTag && scriptTag.dataset.widgetId) {
        const widgetId = scriptTag.dataset.widgetId;
        const autoConnect = scriptTag.dataset.autoConnect !== 'false';
        const showUI = scriptTag.dataset.showUI !== 'false';

        window.addEventListener('DOMContentLoaded', () => {
            const connector = createNexusConnector(widgetId);

            if (autoConnect) {
                connector.connect();
            }

            if (showUI) {
                connector.on('connected', () => {
                    connector.createNexusUI();
                });
            }

            // Make connector globally available
            window.nexusConnector = connector;
            global.nexusConnector = connector;
        });
    }

    // Export for manual initialization
    global.createNexusConnector = createNexusConnector;

})(typeof window !== 'undefined' ? window : this);
