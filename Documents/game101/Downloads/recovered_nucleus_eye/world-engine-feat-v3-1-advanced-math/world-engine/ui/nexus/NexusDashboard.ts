/**
 * VectorLab Nexus Dashboard
 * Control interface for VectorLab brain integration and monitoring
 */

export class NexusDashboard {
    private container: HTMLElement;
    private websocket: WebSocket | null = null;
    private brainState: any = {};
    private connectionStatus: string = 'disconnected';
    private dataStreams: Map<string, any[]> = new Map();
    private updateInterval: number = 0;

    constructor(containerId: string, options: DashboardOptions = {}) {
        this.container = document.getElementById(containerId) || document.body;
        this.initializeUI();
        this.connectToNexus();
        this.startDataStreamMonitoring();
    }

    private initializeUI(): void {
        const dashboard = document.createElement('div');
        dashboard.id = 'nexus-dashboard';
        dashboard.className = 'nexus-dashboard';
        dashboard.innerHTML = `
      <div class="dashboard-header">
        <div class="header-info">
          <h1>🧠 VectorLab Nexus Dashboard</h1>
          <div class="connection-indicator" id="connection-indicator">
            <span class="status-dot disconnected"></span>
            <span class="status-text">Disconnected</span>
          </div>
        </div>
        <div class="header-controls">
          <button id="refresh-dashboard">🔄 Refresh</button>
          <button id="export-brain-state">📊 Export State</button>
          <button id="emergency-brake">🛑 Emergency Brake</button>
        </div>
      </div>

      <div class="dashboard-content">
        <!-- Brain State Monitor -->
        <div class="dashboard-panel brain-state-panel">
          <h3>🧠 Brain State Monitor</h3>
          <div class="brain-visualization" id="brain-visualization">
            <div class="neural-activity-display">
              <canvas id="neural-canvas" width="300" height="200"></canvas>
            </div>
            <div class="brain-metrics">
              <div class="metric">
                <label>Neural Activity Level</label>
                <div class="metric-bar">
                  <div class="metric-fill" id="neural-activity-bar"></div>
                </div>
                <span id="neural-activity-value">0%</span>
              </div>
              <div class="metric">
                <label>Memory Usage</label>
                <div class="metric-bar">
                  <div class="metric-fill" id="memory-usage-bar"></div>
                </div>
                <span id="memory-usage-value">0%</span>
              </div>
              <div class="metric">
                <label>Processing Load</label>
                <div class="metric-bar">
                  <div class="metric-fill" id="processing-load-bar"></div>
                </div>
                <span id="processing-load-value">0%</span>
              </div>
            </div>
          </div>
        </div>

        <!-- WebSocket Connections -->
        <div class="dashboard-panel connections-panel">
          <h3>🔗 Live Connections</h3>
          <div class="connection-list" id="connection-list">
            <!-- Dynamic content -->
          </div>
          <div class="connection-controls">
            <button id="test-connections">🔍 Test All Connections</button>
            <button id="reconnect-all">🔄 Reconnect All</button>
          </div>
        </div>

        <!-- Data Streams -->
        <div class="dashboard-panel streams-panel">
          <h3>📊 Data Streams</h3>
          <div class="stream-tabs" id="stream-tabs">
            <button class="stream-tab active" data-stream="ai_chat">AI Chat</button>
            <button class="stream-tab" data-stream="brain_processing">Brain Processing</button>
            <button class="stream-tab" data-stream="demo_events">Demo Events</button>
            <button class="stream-tab" data-stream="system_monitoring">System Monitor</button>
          </div>
          <div class="stream-content" id="stream-content">
            <div class="stream-display" id="stream-display">
              <!-- Dynamic stream data -->
            </div>
          </div>
        </div>

        <!-- Control Interface -->
        <div class="dashboard-panel control-panel">
          <h3>⚡ Neural Control Interface</h3>
          <div class="control-sections">
            <div class="control-section">
              <h4>Brain Operations</h4>
              <div class="control-buttons">
                <button class="control-btn" id="optimize-brain">🎯 Optimize Neural Pathways</button>
                <button class="control-btn" id="clear-memory">🧹 Clear Working Memory</button>
                <button class="control-btn" id="deep-analysis">🔬 Deep Analysis Mode</button>
                <button class="control-btn" id="learning-mode">📚 Learning Mode</button>
              </div>
            </div>

            <div class="control-section">
              <h4>Data Processing</h4>
              <div class="control-buttons">
                <button class="control-btn" id="process-queue">⚡ Process Queue</button>
                <button class="control-btn" id="batch-processing">📦 Batch Processing</button>
                <button class="control-btn" id="real-time-mode">⏱️ Real-time Mode</button>
                <button class="control-btn" id="pause-processing">⏸️ Pause Processing</button>
              </div>
            </div>

            <div class="control-section">
              <h4>System Integration</h4>
              <div class="control-buttons">
                <button class="control-btn" id="sync-domains">🔄 Sync Domains</button>
                <button class="control-btn" id="update-schemas">📋 Update Schemas</button>
                <button class="control-btn" id="calibrate-math">🔢 Calibrate Math Pro</button>
                <button class="control-btn" id="test-rnes">🔒 Test RNES Vault</button>
              </div>
            </div>
          </div>
        </div>

        <!-- Live Console -->
        <div class="dashboard-panel console-panel">
          <h3>💻 Live Console</h3>
          <div class="console-display" id="console-display">
            <!-- Live console output -->
          </div>
          <div class="console-input-section">
            <input type="text" id="console-input" placeholder="Enter VectorLab command..." />
            <button id="execute-command">Execute</button>
            <button id="clear-console">Clear</button>
          </div>
        </div>
      </div>

      <div class="dashboard-footer">
        <div class="system-status">
          <span>System Status: <span id="system-status">Initializing...</span></span>
          <span>Uptime: <span id="system-uptime">--</span></span>
          <span>Last Update: <span id="last-update-time">--</span></span>
        </div>
      </div>
    `;

        this.container.appendChild(dashboard);
        this.setupEventListeners();
        this.initializeVisualization();
    }

    private setupEventListeners(): void {
        // Header controls
        document.getElementById('refresh-dashboard')?.addEventListener('click', () => this.refreshDashboard());
        document.getElementById('export-brain-state')?.addEventListener('click', () => this.exportBrainState());
        document.getElementById('emergency-brake')?.addEventListener('click', () => this.emergencyBrake());

        // Connection controls
        document.getElementById('test-connections')?.addEventListener('click', () => this.testAllConnections());
        document.getElementById('reconnect-all')?.addEventListener('click', () => this.reconnectAll());

        // Stream tabs
        document.querySelectorAll('.stream-tab').forEach(tab => {
            tab.addEventListener('click', (e) => {
                const target = e.target as HTMLElement;
                const stream = target.dataset.stream;
                if (stream) this.switchStream(stream);
            });
        });

        // Control buttons
        this.setupControlButtons();

        // Console
        const consoleInput = document.getElementById('console-input') as HTMLInputElement;
        document.getElementById('execute-command')?.addEventListener('click', () => this.executeCommand(consoleInput.value));
        document.getElementById('clear-console')?.addEventListener('click', () => this.clearConsole());

        consoleInput?.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') this.executeCommand(consoleInput.value);
        });

        // Global keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            if (e.ctrlKey && e.key === '`') {
                e.preventDefault();
                this.toggle();
            }
        });
    }

    private setupControlButtons(): void {
        const controls = [
            'optimize-brain', 'clear-memory', 'deep-analysis', 'learning-mode',
            'process-queue', 'batch-processing', 'real-time-mode', 'pause-processing',
            'sync-domains', 'update-schemas', 'calibrate-math', 'test-rnes'
        ];

        controls.forEach(controlId => {
            document.getElementById(controlId)?.addEventListener('click', () => {
                this.executeControlCommand(controlId);
            });
        });
    }

    private connectToNexus(): void {
        try {
            this.websocket = new WebSocket('ws://localhost:9000');

            this.websocket.onopen = () => {
                this.connectionStatus = 'connected';
                this.updateConnectionIndicator('connected');
                this.addConsoleMessage('Connected to VectorLab Nexus', 'success');

                // Initialize dashboard data
                this.sendToNexus({
                    type: 'dashboard_init',
                    timestamp: Date.now()
                });
            };

            this.websocket.onmessage = (event) => {
                const data = JSON.parse(event.data);
                this.handleNexusData(data);
            };

            this.websocket.onclose = () => {
                this.connectionStatus = 'disconnected';
                this.updateConnectionIndicator('disconnected');
                this.addConsoleMessage('Disconnected from VectorLab Nexus', 'error');
                this.attemptReconnect();
            };

            this.websocket.onerror = (error) => {
                this.connectionStatus = 'error';
                this.updateConnectionIndicator('error');
                this.addConsoleMessage('WebSocket error occurred', 'error');
            };

        } catch (error) {
            this.connectionStatus = 'error';
            this.updateConnectionIndicator('error');
            this.addConsoleMessage('Failed to connect to VectorLab Nexus', 'error');
        }
    }

    private handleNexusData(data: any): void {
        switch (data.type) {
            case 'brain_state_update':
                this.updateBrainState(data.state);
                break;
            case 'connection_status':
                this.updateConnectionList(data.connections);
                break;
            case 'data_stream':
                this.updateDataStream(data.stream, data.data);
                break;
            case 'console_output':
                this.addConsoleMessage(data.message, data.level || 'info');
                break;
            case 'system_status':
                this.updateSystemStatus(data.status);
                break;
        }

        this.updateLastUpdateTime();
    }

    private updateBrainState(state: any): void {
        this.brainState = { ...this.brainState, ...state };

        // Update neural activity visualization
        this.updateNeuralVisualization(state);

        // Update metrics
        this.updateMetric('neural-activity', state.neural_activity || 0);
        this.updateMetric('memory-usage', state.memory_usage || 0);
        this.updateMetric('processing-load', state.processing_load || 0);
    }

    private updateNeuralVisualization(state: any): void {
        const canvas = document.getElementById('neural-canvas') as HTMLCanvasElement;
        const ctx = canvas?.getContext('2d');
        if (!ctx) return;

        // Clear canvas
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        // Draw neural network visualization
        const centerX = canvas.width / 2;
        const centerY = canvas.height / 2;
        const maxRadius = Math.min(canvas.width, canvas.height) / 3;

        // Draw neural activity as pulsing circles
        const activity = state.neural_activity || 0;
        const pulseRadius = (activity / 100) * maxRadius;

        // Background network
        ctx.strokeStyle = 'rgba(0, 212, 170, 0.3)';
        ctx.lineWidth = 1;
        for (let i = 0; i < 20; i++) {
            const angle = (i / 20) * Math.PI * 2;
            const x = centerX + Math.cos(angle) * maxRadius;
            const y = centerY + Math.sin(angle) * maxRadius;

            ctx.beginPath();
            ctx.moveTo(centerX, centerY);
            ctx.lineTo(x, y);
            ctx.stroke();

            // Neural nodes
            ctx.fillStyle = `rgba(0, 212, 170, ${0.5 + (activity / 200)})`;
            ctx.beginPath();
            ctx.arc(x, y, 3, 0, Math.PI * 2);
            ctx.fill();
        }

        // Central pulse
        ctx.fillStyle = `rgba(0, 212, 170, ${0.3 + (activity / 200)})`;
        ctx.beginPath();
        ctx.arc(centerX, centerY, pulseRadius, 0, Math.PI * 2);
        ctx.fill();

        // Core
        ctx.fillStyle = '#00d4aa';
        ctx.beginPath();
        ctx.arc(centerX, centerY, 8, 0, Math.PI * 2);
        ctx.fill();
    }

    private updateMetric(metricId: string, value: number): void {
        const bar = document.getElementById(`${metricId}-bar`);
        const valueEl = document.getElementById(`${metricId}-value`);

        if (bar) {
            bar.style.width = `${Math.min(value, 100)}%`;

            // Color coding based on value
            if (value < 30) bar.style.backgroundColor = '#00ff00';
            else if (value < 70) bar.style.backgroundColor = '#ff9500';
            else bar.style.backgroundColor = '#ff5555';
        }

        if (valueEl) {
            valueEl.textContent = `${value.toFixed(1)}%`;
        }
    }

    private updateConnectionList(connections: any[]): void {
        const connectionList = document.getElementById('connection-list');
        if (!connectionList) return;

        connectionList.innerHTML = connections.map(conn => `
      <div class="connection-item">
        <div class="connection-info">
          <div class="connection-name">${conn.name}</div>
          <div class="connection-url">${conn.url}</div>
        </div>
        <div class="connection-status">
          <span class="status-dot ${conn.status}"></span>
          <span class="connection-metrics">
            ${conn.latency}ms | ${conn.messages || 0} msgs
          </span>
        </div>
        <div class="connection-actions">
          <button class="action-btn" onclick="this.testConnection('${conn.id}')">Test</button>
          <button class="action-btn" onclick="this.reconnectConnection('${conn.id}')">Reconnect</button>
        </div>
      </div>
    `).join('');
    }

    private updateDataStream(streamName: string, data: any[]): void {
        // Store stream data
        this.dataStreams.set(streamName, data.slice(-100)); // Keep last 100 entries

        // Update display if this stream is currently visible
        const activeTab = document.querySelector('.stream-tab.active');
        if (activeTab && activeTab.getAttribute('data-stream') === streamName) {
            this.displayStreamData(streamName, data);
        }
    }

    private displayStreamData(streamName: string, data: any[]): void {
        const streamDisplay = document.getElementById('stream-display');
        if (!streamDisplay) return;

        streamDisplay.innerHTML = `
      <div class="stream-header">
        <h4>${streamName.replace('_', ' ').toUpperCase()}</h4>
        <div class="stream-stats">
          <span>Entries: ${data.length}</span>
          <span>Latest: ${data.length > 0 ? new Date(data[data.length - 1].timestamp).toLocaleTimeString() : 'None'}</span>
        </div>
      </div>
      <div class="stream-entries">
        ${data.slice(-20).reverse().map(entry => `
          <div class="stream-entry">
            <div class="entry-timestamp">${new Date(entry.timestamp).toLocaleTimeString()}</div>
            <div class="entry-content">${this.formatStreamEntry(streamName, entry)}</div>
          </div>
        `).join('')}
      </div>
    `;
    }

    private formatStreamEntry(streamName: string, entry: any): string {
        switch (streamName) {
            case 'ai_chat':
                return `<strong>${entry.type}:</strong> ${entry.message || entry.response || 'No content'}`;
            case 'brain_processing':
                return `<strong>Processing:</strong> ${entry.operation || 'Unknown'} (${entry.duration || 0}ms)`;
            case 'demo_events':
                return `<strong>Demo:</strong> ${entry.event_type || 'Event'} in ${entry.demo || 'Unknown'}`;
            case 'system_monitoring':
                return `<strong>System:</strong> ${entry.component || 'Component'} - ${entry.status || 'Status'}`;
            default:
                return JSON.stringify(entry);
        }
    }

    private switchStream(streamName: string): void {
        // Update active tab
        document.querySelectorAll('.stream-tab').forEach(tab => tab.classList.remove('active'));
        document.querySelector(`[data-stream="${streamName}"]`)?.classList.add('active');

        // Display stream data
        const data = this.dataStreams.get(streamName) || [];
        this.displayStreamData(streamName, data);
    }

    private executeControlCommand(commandId: string): void {
        const commandMap = {
            'optimize-brain': 'optimize_neural_pathways',
            'clear-memory': 'clear_working_memory',
            'deep-analysis': 'enable_deep_analysis',
            'learning-mode': 'enable_learning_mode',
            'process-queue': 'process_data_queue',
            'batch-processing': 'enable_batch_processing',
            'real-time-mode': 'enable_realtime_mode',
            'pause-processing': 'pause_data_processing',
            'sync-domains': 'sync_world_engine_domains',
            'update-schemas': 'update_data_schemas',
            'calibrate-math': 'calibrate_math_pro',
            'test-rnes': 'test_rnes_vault'
        };

        const command = commandMap[commandId as keyof typeof commandMap];
        if (!command) return;

        this.addConsoleMessage(`Executing: ${command}`, 'info');

        if (this.websocket?.readyState === WebSocket.OPEN) {
            this.sendToNexus({
                type: 'control_command',
                command: command,
                timestamp: Date.now()
            });
        }
    }

    private executeCommand(command: string): void {
        if (!command.trim()) return;

        this.addConsoleMessage(`> ${command}`, 'user');

        if (this.websocket?.readyState === WebSocket.OPEN) {
            this.sendToNexus({
                type: 'console_command',
                command: command,
                timestamp: Date.now()
            });
        } else {
            this.addConsoleMessage('Error: Not connected to VectorLab Nexus', 'error');
        }

        (document.getElementById('console-input') as HTMLInputElement).value = '';
    }

    private addConsoleMessage(message: string, level: 'info' | 'success' | 'error' | 'warning' | 'user' = 'info'): void {
        const consoleDisplay = document.getElementById('console-display');
        if (!consoleDisplay) return;

        const messageEl = document.createElement('div');
        messageEl.className = `console-message ${level}`;
        messageEl.innerHTML = `
      <span class="message-time">[${new Date().toLocaleTimeString()}]</span>
      <span class="message-content">${message}</span>
    `;

        consoleDisplay.appendChild(messageEl);
        consoleDisplay.scrollTop = consoleDisplay.scrollHeight;

        // Keep only last 100 messages
        while (consoleDisplay.children.length > 100) {
            consoleDisplay.removeChild(consoleDisplay.firstChild as Node);
        }
    }

    private clearConsole(): void {
        const consoleDisplay = document.getElementById('console-display');
        if (consoleDisplay) {
            consoleDisplay.innerHTML = '';
        }
    }

    private updateConnectionIndicator(status: string): void {
        const indicator = document.getElementById('connection-indicator');
        if (!indicator) return;

        const statusDot = indicator.querySelector('.status-dot');
        const statusText = indicator.querySelector('.status-text');

        if (statusDot) {
            statusDot.className = `status-dot ${status}`;
        }

        if (statusText) {
            const statusMessages = {
                connected: 'Connected',
                disconnected: 'Disconnected',
                connecting: 'Connecting...',
                error: 'Connection Error'
            };
            statusText.textContent = statusMessages[status as keyof typeof statusMessages] || 'Unknown';
        }
    }

    private updateSystemStatus(status: any): void {
        const systemStatusEl = document.getElementById('system-status');
        const uptimeEl = document.getElementById('system-uptime');

        if (systemStatusEl) {
            systemStatusEl.textContent = status.overall || 'Unknown';
        }

        if (uptimeEl) {
            uptimeEl.textContent = this.formatUptime(status.uptime || 0);
        }
    }

    private updateLastUpdateTime(): void {
        const timeEl = document.getElementById('last-update-time');
        if (timeEl) {
            timeEl.textContent = new Date().toLocaleTimeString();
        }
    }

    private formatUptime(seconds: number): string {
        const hours = Math.floor(seconds / 3600);
        const minutes = Math.floor((seconds % 3600) / 60);
        const secs = seconds % 60;
        return `${hours}h ${minutes}m ${secs}s`;
    }

    private startDataStreamMonitoring(): void {
        this.updateInterval = window.setInterval(() => {
            if (this.websocket?.readyState === WebSocket.OPEN) {
                this.sendToNexus({
                    type: 'request_stream_data',
                    timestamp: Date.now()
                });
            }
        }, 1000); // Update every second
    }

    private sendToNexus(data: any): void {
        if (this.websocket?.readyState === WebSocket.OPEN) {
            this.websocket.send(JSON.stringify(data));
        }
    }

    private attemptReconnect(): void {
        setTimeout(() => {
            if (!this.websocket || this.websocket.readyState === WebSocket.CLOSED) {
                this.addConsoleMessage('Attempting to reconnect...', 'info');
                this.connectToNexus();
            }
        }, 5000);
    }

    private refreshDashboard(): void {
        this.addConsoleMessage('Refreshing dashboard...', 'info');
        if (this.websocket?.readyState === WebSocket.OPEN) {
            this.sendToNexus({
                type: 'dashboard_refresh',
                timestamp: Date.now()
            });
        }
    }

    private exportBrainState(): void {
        const stateData = {
            timestamp: new Date().toISOString(),
            brain_state: this.brainState,
            connection_status: this.connectionStatus,
            data_streams: Object.fromEntries(this.dataStreams)
        };

        const blob = new Blob([JSON.stringify(stateData, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `vectorlab-brain-state-${Date.now()}.json`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);

        this.addConsoleMessage('Brain state exported successfully', 'success');
    }

    private emergencyBrake(): void {
        if (confirm('Are you sure you want to activate the emergency brake? This will halt all VectorLab brain processing.')) {
            this.sendToNexus({
                type: 'emergency_brake',
                timestamp: Date.now()
            });
            this.addConsoleMessage('EMERGENCY BRAKE ACTIVATED', 'error');
        }
    }

    private testAllConnections(): void {
        this.addConsoleMessage('Testing all connections...', 'info');
        this.sendToNexus({
            type: 'test_all_connections',
            timestamp: Date.now()
        });
    }

    private reconnectAll(): void {
        this.addConsoleMessage('Reconnecting all connections...', 'info');
        this.sendToNexus({
            type: 'reconnect_all',
            timestamp: Date.now()
        });
    }

    private initializeVisualization(): void {
        // Initialize with default visualization
        this.updateNeuralVisualization({ neural_activity: 0 });
    }

    public toggle(): void {
        const dashboard = document.getElementById('nexus-dashboard');
        if (!dashboard) return;

        if (dashboard.style.display === 'none') {
            dashboard.style.display = 'block';
            this.refreshDashboard();
        } else {
            dashboard.style.display = 'none';
        }
    }

    public destroy(): void {
        if (this.updateInterval) {
            clearInterval(this.updateInterval);
        }
        if (this.websocket) {
            this.websocket.close();
        }
    }
}

// Type definitions
interface DashboardOptions {
    theme?: string;
    updateInterval?: number;
}

// CSS styles for Nexus Dashboard
export const nexusDashboardStyles = `
.nexus-dashboard {
  position: fixed;
  top: 0;
  left: 0;
  width: 100vw;
  height: 100vh;
  background: linear-gradient(135deg, #001122 0%, #000811 100%);
  color: #ccffff;
  font-family: 'Segoe UI', 'Consolas', monospace;
  z-index: 9999;
  overflow: auto;
  display: flex;
  flex-direction: column;
}

.dashboard-header {
  background: rgba(0, 51, 68, 0.8);
  backdrop-filter: blur(10px);
  padding: 20px;
  border-bottom: 2px solid #00ffcc;
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.header-info h1 {
  margin: 0 0 8px 0;
  color: #00ffcc;
  font-size: 24px;
  font-weight: 700;
}

.connection-indicator {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 14px;
}

.status-dot {
  width: 12px;
  height: 12px;
  border-radius: 50%;
  display: inline-block;
}

.status-dot.connected { background: #00ff00; box-shadow: 0 0 8px #00ff00; }
.status-dot.disconnected { background: #ff5555; }
.status-dot.connecting { background: #ff9500; animation: pulse 1s infinite; }
.status-dot.error { background: #ff0000; }

@keyframes pulse {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.5; }
}

.header-controls {
  display: flex;
  gap: 12px;
}

.header-controls button {
  background: #00ffcc;
  color: #001122;
  border: none;
  border-radius: 8px;
  padding: 12px 20px;
  cursor: pointer;
  font-weight: 600;
  font-size: 14px;
  transition: all 0.2s ease;
}

.header-controls button:hover {
  background: #00d4aa;
  transform: translateY(-2px);
  box-shadow: 0 4px 12px rgba(0, 255, 204, 0.3);
}

.dashboard-content {
  flex: 1;
  padding: 20px;
  display: grid;
  grid-template-columns: 1fr 1fr;
  grid-template-rows: auto auto auto;
  gap: 20px;
  overflow: auto;
}

.dashboard-panel {
  background: rgba(0, 51, 68, 0.6);
  backdrop-filter: blur(10px);
  border: 1px solid #00ffcc;
  border-radius: 12px;
  padding: 20px;
  box-shadow: 0 8px 32px rgba(0, 255, 204, 0.1);
}

.dashboard-panel h3 {
  margin: 0 0 16px 0;
  color: #00ffcc;
  font-size: 18px;
  border-bottom: 1px solid #00ffcc;
  padding-bottom: 8px;
}

.brain-state-panel {
  grid-column: 1 / -1;
}

.brain-visualization {
  display: flex;
  gap: 20px;
  align-items: center;
}

.neural-activity-display {
  flex-shrink: 0;
}

.brain-metrics {
  flex: 1;
  display: grid;
  grid-template-columns: 1fr;
  gap: 16px;
}

.metric {
  display: grid;
  grid-template-columns: 120px 1fr auto;
  gap: 12px;
  align-items: center;
}

.metric label {
  font-size: 12px;
  color: #ccffff;
  font-weight: 500;
}

.metric-bar {
  height: 8px;
  background: #001122;
  border-radius: 4px;
  border: 1px solid #00ffcc;
  overflow: hidden;
}

.metric-fill {
  height: 100%;
  background: #00ffcc;
  transition: width 0.5s ease;
  border-radius: 3px;
}

.connection-list {
  display: flex;
  flex-direction: column;
  gap: 12px;
  margin-bottom: 16px;
}

.connection-item {
  display: flex;
  justify-content: space-between;
  align-items: center;
  background: rgba(0, 17, 34, 0.5);
  padding: 12px 16px;
  border-radius: 8px;
  border: 1px solid #003344;
}

.connection-info {
  flex: 1;
}

.connection-name {
  font-weight: 600;
  color: #00ffcc;
  margin-bottom: 4px;
}

.connection-url {
  font-size: 12px;
  color: #99ccdd;
  font-family: monospace;
}

.connection-status {
  display: flex;
  flex-direction: column;
  align-items: flex-end;
  margin: 0 16px;
}

.connection-metrics {
  font-size: 11px;
  color: #99ccdd;
}

.connection-actions {
  display: flex;
  gap: 8px;
}

.action-btn {
  background: #00ffcc;
  color: #001122;
  border: none;
  border-radius: 6px;
  padding: 6px 12px;
  cursor: pointer;
  font-size: 11px;
  font-weight: 600;
}

.stream-tabs {
  display: flex;
  border-bottom: 1px solid #00ffcc;
  margin-bottom: 16px;
}

.stream-tab {
  background: none;
  border: none;
  color: #99ccdd;
  padding: 12px 20px;
  cursor: pointer;
  border-bottom: 2px solid transparent;
  font-size: 14px;
  transition: all 0.2s ease;
}

.stream-tab.active {
  color: #00ffcc;
  border-bottom-color: #00ffcc;
}

.stream-tab:hover {
  color: #ccffff;
}

.stream-display {
  max-height: 300px;
  overflow-y: auto;
}

.stream-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 12px;
}

.stream-header h4 {
  margin: 0;
  color: #00ffcc;
}

.stream-stats {
  font-size: 12px;
  color: #99ccdd;
}

.stream-stats span {
  margin-left: 16px;
}

.stream-entries {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.stream-entry {
  background: rgba(0, 17, 34, 0.3);
  padding: 8px 12px;
  border-radius: 6px;
  border-left: 3px solid #00ffcc;
  font-size: 12px;
}

.entry-timestamp {
  color: #99ccdd;
  font-size: 10px;
  margin-bottom: 4px;
}

.entry-content {
  color: #ccffff;
}

.control-panel {
  grid-column: 1 / -1;
}

.control-sections {
  display: grid;
  grid-template-columns: 1fr 1fr 1fr;
  gap: 20px;
}

.control-section h4 {
  margin: 0 0 12px 0;
  color: #00ffcc;
  font-size: 14px;
  border-bottom: 1px solid #003344;
  padding-bottom: 6px;
}

.control-buttons {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.control-btn {
  background: rgba(0, 255, 204, 0.1);
  color: #00ffcc;
  border: 1px solid #00ffcc;
  border-radius: 6px;
  padding: 10px 16px;
  cursor: pointer;
  font-size: 12px;
  text-align: left;
  transition: all 0.2s ease;
}

.control-btn:hover {
  background: rgba(0, 255, 204, 0.2);
  transform: translateX(4px);
}

.console-panel {
  grid-column: 1 / -1;
  max-height: 300px;
  display: flex;
  flex-direction: column;
}

.console-display {
  flex: 1;
  background: #000811;
  border: 1px solid #00ffcc;
  border-radius: 8px;
  padding: 12px;
  overflow-y: auto;
  font-family: 'Consolas', monospace;
  font-size: 12px;
  margin-bottom: 12px;
  max-height: 200px;
}

.console-message {
  margin-bottom: 4px;
  display: flex;
  gap: 8px;
}

.console-message.info { color: #ccffff; }
.console-message.success { color: #00ff00; }
.console-message.error { color: #ff5555; }
.console-message.warning { color: #ff9500; }
.console-message.user { color: #00ffcc; }

.message-time {
  color: #99ccdd;
  flex-shrink: 0;
}

.console-input-section {
  display: flex;
  gap: 8px;
}

.console-input-section input {
  flex: 1;
  background: #001122;
  border: 1px solid #00ffcc;
  border-radius: 6px;
  padding: 8px 12px;
  color: #ccffff;
  font-family: monospace;
  font-size: 12px;
}

.console-input-section input:focus {
  outline: none;
  border-color: #00d4aa;
  box-shadow: 0 0 8px rgba(0, 212, 170, 0.3);
}

.console-input-section button {
  background: #00ffcc;
  color: #001122;
  border: none;
  border-radius: 6px;
  padding: 8px 16px;
  cursor: pointer;
  font-size: 12px;
  font-weight: 600;
}

.dashboard-footer {
  background: rgba(0, 17, 34, 0.8);
  padding: 16px 20px;
  border-top: 1px solid #00ffcc;
  font-size: 12px;
}

.system-status {
  display: flex;
  justify-content: space-between;
  color: #99ccdd;
}

.system-status span {
  display: flex;
  align-items: center;
  gap: 8px;
}

#neural-canvas {
  border: 1px solid #00ffcc;
  border-radius: 8px;
  background: #000811;
}
`;
