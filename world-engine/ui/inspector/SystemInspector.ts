/**
 * World Engine System Inspector
 * Real-time monitoring and debugging interface for World Engine components
 */

export class SystemInspector {
    private container: HTMLElement;
    private isVisible: boolean = false;
    private websocket: WebSocket | null = null;
    private updateInterval: number = 0;
    private systemMetrics: any = {};

    constructor(containerId: string, options: InspectorOptions = {}) {
        this.container = document.getElementById(containerId) || document.body;
        this.initializeUI();
        this.connectToNexus();
        this.startMonitoring();
    }

    private initializeUI(): void {
        const inspector = document.createElement('div');
        inspector.id = 'system-inspector';
        inspector.className = 'system-inspector';
        inspector.innerHTML = `
      <div class="inspector-header">
        <h2>🔧 World Engine System Inspector</h2>
        <div class="inspector-controls">
          <button id="refresh-btn">🔄</button>
          <button id="export-btn">📁</button>
          <button id="minimize-inspector">−</button>
        </div>
      </div>

      <div class="inspector-content">
        <div class="inspector-tabs">
          <button class="tab-btn active" data-tab="domains">Domains</button>
          <button class="tab-btn" data-tab="services">Services</button>
          <button class="tab-btn" data-tab="performance">Performance</button>
          <button class="tab-btn" data-tab="websockets">WebSockets</button>
          <button class="tab-btn" data-tab="console">Console</button>
        </div>

        <div class="tab-content">
          <!-- Domains Tab -->
          <div id="domains-tab" class="tab-panel active">
            <div class="domain-status">
              <h3>Domain Status Overview</h3>
              <div class="domain-grid" id="domain-grid">
                <!-- Dynamic content -->
              </div>
            </div>
          </div>

          <!-- Services Tab -->
          <div id="services-tab" class="tab-panel">
            <div class="service-status">
              <h3>Service Registry</h3>
              <div class="service-list" id="service-list">
                <!-- Dynamic content -->
              </div>
            </div>
          </div>

          <!-- Performance Tab -->
          <div id="performance-tab" class="tab-panel">
            <div class="performance-metrics">
              <h3>System Performance</h3>
              <div class="metrics-grid" id="metrics-grid">
                <div class="metric-card">
                  <div class="metric-label">Memory Usage</div>
                  <div class="metric-value" id="memory-usage">-- MB</div>
                  <div class="metric-bar">
                    <div class="metric-fill" id="memory-bar"></div>
                  </div>
                </div>
                <div class="metric-card">
                  <div class="metric-label">CPU Usage</div>
                  <div class="metric-value" id="cpu-usage">--%</div>
                  <div class="metric-bar">
                    <div class="metric-fill" id="cpu-bar"></div>
                  </div>
                </div>
                <div class="metric-card">
                  <div class="metric-label">WebSocket Latency</div>
                  <div class="metric-value" id="ws-latency">-- ms</div>
                </div>
                <div class="metric-card">
                  <div class="metric-label">Active Connections</div>
                  <div class="metric-value" id="connections">--</div>
                </div>
              </div>
              <div class="performance-chart" id="performance-chart">
                <canvas id="perf-canvas" width="400" height="200"></canvas>
              </div>
            </div>
          </div>

          <!-- WebSockets Tab -->
          <div id="websockets-tab" class="tab-panel">
            <div class="websocket-status">
              <h3>WebSocket Connections</h3>
              <div class="websocket-list" id="websocket-list">
                <!-- Dynamic content -->
              </div>
            </div>
          </div>

          <!-- Console Tab -->
          <div id="console-tab" class="tab-panel">
            <div class="debug-console">
              <h3>Debug Console</h3>
              <div class="console-output" id="console-output">
                <!-- Dynamic content -->
              </div>
              <div class="console-input">
                <input type="text" id="console-command" placeholder="Enter command..." />
                <button id="execute-btn">Execute</button>
              </div>
            </div>
          </div>
        </div>
      </div>

      <div class="inspector-status">
        <span class="status-indicator" id="inspector-status">🟢 Connected</span>
        <span class="update-time" id="last-update">Last update: --</span>
      </div>
    `;

        inspector.style.display = 'none';
        this.container.appendChild(inspector);

        this.setupEventListeners();
    }

    private setupEventListeners(): void {
        // Tab switching
        const tabBtns = document.querySelectorAll('.tab-btn');
        tabBtns.forEach(btn => {
            btn.addEventListener('click', (e) => {
                const target = e.target as HTMLElement;
                const tabName = target.dataset.tab;
                if (tabName) this.switchTab(tabName);
            });
        });

        // Control buttons
        document.getElementById('refresh-btn')?.addEventListener('click', () => this.refreshData());
        document.getElementById('export-btn')?.addEventListener('click', () => this.exportData());
        document.getElementById('minimize-inspector')?.addEventListener('click', () => this.minimize());

        // Console command
        const consoleInput = document.getElementById('console-command') as HTMLInputElement;
        document.getElementById('execute-btn')?.addEventListener('click', () => this.executeCommand(consoleInput.value));
        consoleInput?.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') this.executeCommand(consoleInput.value);
        });

        // Global toggle (F12 key)
        document.addEventListener('keydown', (e) => {
            if (e.key === 'F12' && !e.ctrlKey) {
                e.preventDefault();
                this.toggle();
            }
        });
    }

    private connectToNexus(): void {
        try {
            this.websocket = new WebSocket('ws://localhost:9000');

            this.websocket.onopen = () => {
                this.updateStatus('Connected', 'success');
                this.sendToNexus({ type: 'inspector_init', timestamp: Date.now() });
            };

            this.websocket.onmessage = (event) => {
                const data = JSON.parse(event.data);
                this.handleNexusData(data);
            };

            this.websocket.onclose = () => {
                this.updateStatus('Disconnected', 'error');
                this.attemptReconnect();
            };

        } catch (error) {
            this.updateStatus('Connection Failed', 'error');
            console.error('Inspector connection error:', error);
        }
    }

    private startMonitoring(): void {
        this.updateInterval = window.setInterval(() => {
            this.requestSystemUpdate();
        }, 2000); // Update every 2 seconds
    }

    private requestSystemUpdate(): void {
        if (this.websocket?.readyState === WebSocket.OPEN) {
            this.sendToNexus({
                type: 'system_status_request',
                timestamp: Date.now()
            });
        }
    }

    private handleNexusData(data: any): void {
        switch (data.type) {
            case 'system_status':
                this.updateSystemStatus(data.status);
                break;
            case 'domain_status':
                this.updateDomainStatus(data.domains);
                break;
            case 'service_status':
                this.updateServiceStatus(data.services);
                break;
            case 'performance_metrics':
                this.updatePerformanceMetrics(data.metrics);
                break;
            case 'websocket_status':
                this.updateWebSocketStatus(data.connections);
                break;
            case 'console_response':
                this.addConsoleOutput(data.response);
                break;
        }

        this.updateLastUpdateTime();
    }

    private updateDomainStatus(domains: any[]): void {
        const grid = document.getElementById('domain-grid');
        if (!grid) return;

        grid.innerHTML = domains.map(domain => `
      <div class="domain-card">
        <div class="domain-header">
          <span class="domain-name">${domain.name}</span>
          <span class="domain-status ${domain.status}">${this.getStatusIcon(domain.status)}</span>
        </div>
        <div class="domain-details">
          <div class="detail-item">Files: ${domain.fileCount || 0}</div>
          <div class="detail-item">Size: ${this.formatBytes(domain.size || 0)}</div>
          <div class="detail-item">Last Modified: ${domain.lastModified ? new Date(domain.lastModified).toLocaleString() : 'Unknown'}</div>
        </div>
        <div class="domain-actions">
          <button class="action-btn" onclick="this.inspectDomain('${domain.name}')">Inspect</button>
          <button class="action-btn" onclick="this.reloadDomain('${domain.name}')">Reload</button>
        </div>
      </div>
    `).join('');
    }

    private updateServiceStatus(services: any[]): void {
        const serviceList = document.getElementById('service-list');
        if (!serviceList) return;

        serviceList.innerHTML = services.map(service => `
      <div class="service-item">
        <div class="service-info">
          <div class="service-name">${service.name}</div>
          <div class="service-description">${service.description || 'No description'}</div>
        </div>
        <div class="service-status">
          <span class="status-badge ${service.status}">${service.status}</span>
          <span class="service-uptime">${this.formatUptime(service.uptime)}</span>
        </div>
        <div class="service-actions">
          <button class="action-btn small" onclick="this.restartService('${service.name}')">Restart</button>
        </div>
      </div>
    `).join('');
    }

    private updatePerformanceMetrics(metrics: any): void {
        const updateMetric = (id: string, value: string, percentage?: number) => {
            const element = document.getElementById(id);
            if (element) element.textContent = value;

            if (percentage !== undefined) {
                const bar = document.getElementById(id.replace('-usage', '-bar'));
                if (bar) bar.style.width = `${percentage}%`;
            }
        };

        updateMetric('memory-usage', `${metrics.memoryUsage?.toFixed(1) || 0} MB`, metrics.memoryPercentage);
        updateMetric('cpu-usage', `${metrics.cpuUsage?.toFixed(1) || 0}%`, metrics.cpuUsage);
        updateMetric('ws-latency', `${metrics.websocketLatency || 0} ms`);
        updateMetric('connections', `${metrics.activeConnections || 0}`);

        this.updatePerformanceChart(metrics);
    }

    private updatePerformanceChart(metrics: any): void {
        const canvas = document.getElementById('perf-canvas') as HTMLCanvasElement;
        const ctx = canvas?.getContext('2d');
        if (!ctx) return;

        // Simple line chart for performance over time
        // This would be expanded with a proper charting library in production
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.strokeStyle = '#00d4aa';
        ctx.lineWidth = 2;
        ctx.beginPath();

        // Mock performance data visualization
        const points = Array.from({ length: 50 }, (_, i) => ({
            x: (i / 49) * canvas.width,
            y: canvas.height - (Math.random() * canvas.height * 0.8 + canvas.height * 0.1)
        }));

        points.forEach((point, index) => {
            if (index === 0) ctx.moveTo(point.x, point.y);
            else ctx.lineTo(point.x, point.y);
        });

        ctx.stroke();
    }

    private updateWebSocketStatus(connections: any[]): void {
        const wsList = document.getElementById('websocket-list');
        if (!wsList) return;

        wsList.innerHTML = connections.map(conn => `
      <div class="websocket-item">
        <div class="ws-info">
          <div class="ws-url">${conn.url}</div>
          <div class="ws-details">
            <span>State: ${conn.readyState}</span>
            <span>Messages: ${conn.messageCount || 0}</span>
            <span>Latency: ${conn.latency || 0}ms</span>
          </div>
        </div>
        <div class="ws-status">
          <span class="status-dot ${conn.status}"></span>
        </div>
      </div>
    `).join('');
    }

    private addConsoleOutput(output: string): void {
        const consoleOutput = document.getElementById('console-output');
        if (!consoleOutput) return;

        const entry = document.createElement('div');
        entry.className = 'console-entry';
        entry.innerHTML = `
      <span class="console-timestamp">[${new Date().toLocaleTimeString()}]</span>
      <span class="console-text">${output}</span>
    `;

        consoleOutput.appendChild(entry);
        consoleOutput.scrollTop = consoleOutput.scrollHeight;
    }

    private executeCommand(command: string): void {
        if (!command.trim()) return;

        this.addConsoleOutput(`> ${command}`);

        if (this.websocket?.readyState === WebSocket.OPEN) {
            this.sendToNexus({
                type: 'console_command',
                command: command,
                timestamp: Date.now()
            });
        } else {
            this.addConsoleOutput('Error: Not connected to VectorLab Nexus');
        }

        (document.getElementById('console-command') as HTMLInputElement).value = '';
    }

    private switchTab(tabName: string): void {
        // Update tab buttons
        document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
        document.querySelector(`[data-tab="${tabName}"]`)?.classList.add('active');

        // Update tab panels
        document.querySelectorAll('.tab-panel').forEach(panel => panel.classList.remove('active'));
        document.getElementById(`${tabName}-tab`)?.classList.add('active');
    }

    private sendToNexus(data: any): void {
        if (this.websocket?.readyState === WebSocket.OPEN) {
            this.websocket.send(JSON.stringify(data));
        }
    }

    private updateStatus(status: string, type: 'success' | 'error' | 'warning'): void {
        const statusEl = document.getElementById('inspector-status');
        if (statusEl) {
            const icons = { success: '🟢', error: '🔴', warning: '🟡' };
            statusEl.textContent = `${icons[type]} ${status}`;
        }
    }

    private updateLastUpdateTime(): void {
        const timeEl = document.getElementById('last-update');
        if (timeEl) {
            timeEl.textContent = `Last update: ${new Date().toLocaleTimeString()}`;
        }
    }

    private getStatusIcon(status: string): string {
        const icons = {
            active: '🟢',
            inactive: '🔴',
            loading: '🟡',
            error: '❌'
        };
        return icons[status as keyof typeof icons] || '❓';
    }

    private formatBytes(bytes: number): string {
        if (bytes === 0) return '0 B';
        const k = 1024;
        const sizes = ['B', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return `${parseFloat((bytes / Math.pow(k, i)).toFixed(1))} ${sizes[i]}`;
    }

    private formatUptime(seconds: number): string {
        const hours = Math.floor(seconds / 3600);
        const minutes = Math.floor((seconds % 3600) / 60);
        return `${hours}h ${minutes}m`;
    }

    public toggle(): void {
        const inspector = document.getElementById('system-inspector');
        if (!inspector) return;

        if (this.isVisible) {
            inspector.style.display = 'none';
            this.isVisible = false;
        } else {
            inspector.style.display = 'block';
            this.isVisible = true;
            this.refreshData();
        }
    }

    public minimize(): void {
        const content = document.querySelector('.inspector-content') as HTMLElement;
        const minimizeBtn = document.getElementById('minimize-inspector');

        if (content && minimizeBtn) {
            if (content.style.display === 'none') {
                content.style.display = 'block';
                minimizeBtn.textContent = '−';
            } else {
                content.style.display = 'none';
                minimizeBtn.textContent = '+';
            }
        }
    }

    private refreshData(): void {
        this.requestSystemUpdate();
    }

    private exportData(): void {
        const data = {
            timestamp: new Date().toISOString(),
            systemMetrics: this.systemMetrics,
            domains: [], // Would collect from UI
            services: [], // Would collect from UI
            performance: {} // Would collect current metrics
        };

        const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `world-engine-inspector-${Date.now()}.json`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    }

    private attemptReconnect(): void {
        setTimeout(() => {
            if (!this.websocket || this.websocket.readyState === WebSocket.CLOSED) {
                this.connectToNexus();
            }
        }, 5000);
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
export interface InspectorOptions {
    theme?: string;
    updateInterval?: number;
}

// CSS styles for the System Inspector
export const inspectorStyles = `
.system-inspector {
  position: fixed;
  top: 20px;
  left: 20px;
  width: calc(100vw - 40px);
  height: calc(100vh - 40px);
  max-width: 1200px;
  max-height: 800px;
  background: var(--bg-primary, #2a2a2a);
  border: 1px solid var(--border-color, #4a4a4a);
  border-radius: 12px;
  display: flex;
  flex-direction: column;
  font-family: 'Segoe UI', monospace;
  z-index: 9000;
  box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
}

.inspector-header {
  padding: 16px 20px;
  background: var(--bg-secondary, #4a4a4a);
  border-bottom: 1px solid var(--border-color, #4a4a4a);
  display: flex;
  justify-content: space-between;
  align-items: center;
  border-radius: 12px 12px 0 0;
}

.inspector-header h2 {
  margin: 0;
  color: var(--text-light, #fff);
  font-size: 18px;
}

.inspector-controls {
  display: flex;
  gap: 8px;
}

.inspector-controls button {
  background: var(--accent-color, #00d4aa);
  border: none;
  border-radius: 6px;
  padding: 8px 12px;
  cursor: pointer;
  font-size: 14px;
}

.inspector-tabs {
  display: flex;
  background: var(--bg-tertiary, #1a1a1a);
  border-bottom: 1px solid var(--border-color, #4a4a4a);
}

.tab-btn {
  padding: 12px 20px;
  background: none;
  border: none;
  color: var(--text-secondary, #aaa);
  cursor: pointer;
  transition: all 0.2s ease;
}

.tab-btn.active {
  color: var(--accent-color, #00d4aa);
  background: var(--bg-primary, #2a2a2a);
}

.tab-content {
  flex: 1;
  overflow: auto;
}

.tab-panel {
  display: none;
  padding: 20px;
  height: 100%;
}

.tab-panel.active {
  display: block;
}

.domain-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
  gap: 16px;
  margin-top: 16px;
}

.domain-card {
  background: var(--bg-secondary, #4a4a4a);
  border-radius: 8px;
  padding: 16px;
  border: 1px solid var(--border-color, #4a4a4a);
}

.domain-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 12px;
}

.domain-name {
  font-weight: 600;
  color: var(--text-light, #fff);
}

.domain-status {
  font-size: 18px;
}

.domain-details {
  margin-bottom: 12px;
}

.detail-item {
  color: var(--text-secondary, #aaa);
  font-size: 12px;
  margin-bottom: 4px;
}

.domain-actions {
  display: flex;
  gap: 8px;
}

.action-btn {
  padding: 6px 12px;
  background: var(--accent-color, #00d4aa);
  border: none;
  border-radius: 4px;
  cursor: pointer;
  font-size: 12px;
  color: var(--text-dark, #000);
}

.action-btn.small {
  padding: 4px 8px;
  font-size: 11px;
}

.metrics-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 16px;
  margin: 16px 0;
}

.metric-card {
  background: var(--bg-secondary, #4a4a4a);
  padding: 16px;
  border-radius: 8px;
  border: 1px solid var(--border-color, #4a4a4a);
}

.metric-label {
  font-size: 12px;
  color: var(--text-secondary, #aaa);
  margin-bottom: 8px;
}

.metric-value {
  font-size: 24px;
  font-weight: 600;
  color: var(--accent-color, #00d4aa);
  margin-bottom: 8px;
}

.metric-bar {
  height: 4px;
  background: var(--bg-tertiary, #1a1a1a);
  border-radius: 2px;
  overflow: hidden;
}

.metric-fill {
  height: 100%;
  background: var(--accent-color, #00d4aa);
  transition: width 0.3s ease;
}

.performance-chart {
  margin-top: 20px;
  text-align: center;
}

.console-output {
  height: 300px;
  background: var(--bg-tertiary, #1a1a1a);
  border: 1px solid var(--border-color, #4a4a4a);
  border-radius: 8px;
  padding: 12px;
  overflow-y: auto;
  font-family: 'Consolas', 'Monaco', monospace;
  font-size: 12px;
  margin-bottom: 12px;
}

.console-entry {
  margin-bottom: 4px;
  color: var(--text-secondary, #aaa);
}

.console-timestamp {
  color: var(--text-tertiary, #666);
}

.console-input {
  display: flex;
  gap: 8px;
}

.console-input input {
  flex: 1;
  padding: 8px 12px;
  background: var(--bg-secondary, #4a4a4a);
  border: 1px solid var(--border-color, #4a4a4a);
  border-radius: 6px;
  color: var(--text-light, #fff);
  font-family: monospace;
}

.inspector-status {
  padding: 12px 20px;
  background: var(--bg-tertiary, #1a1a1a);
  border-top: 1px solid var(--border-color, #4a4a4a);
  display: flex;
  justify-content: space-between;
  align-items: center;
  font-size: 12px;
  border-radius: 0 0 12px 12px;
}

.status-indicator {
  color: var(--text-light, #fff);
}

.update-time {
  color: var(--text-secondary, #aaa);
}

.service-item, .websocket-item {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 12px;
  background: var(--bg-secondary, #4a4a4a);
  border-radius: 8px;
  margin-bottom: 8px;
  border: 1px solid var(--border-color, #4a4a4a);
}

.status-badge {
  padding: 4px 8px;
  border-radius: 12px;
  font-size: 11px;
  font-weight: 600;
  text-transform: uppercase;
}

.status-badge.active { background: var(--success-color, #00ff00); color: var(--text-dark, #000); }
.status-badge.inactive { background: var(--error-color, #ff5555); color: var(--text-light, #fff); }
.status-badge.loading { background: var(--warning-color, #ff9500); color: var(--text-dark, #000); }

.status-dot {
  width: 12px;
  height: 12px;
  border-radius: 50%;
  display: inline-block;
}

.status-dot.connected { background: var(--success-color, #00ff00); }
.status-dot.disconnected { background: var(--error-color, #ff5555); }
.status-dot.connecting { background: var(--warning-color, #ff9500); }
`;
