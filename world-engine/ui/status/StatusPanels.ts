/**
 * World Engine Status Panels
 * Real-time status displays and notification system
 */

export class StatusPanels {
    private container: HTMLElement;
    private panels: Map<string, StatusPanel> = new Map();
    private websocket: WebSocket | null = null;
    private notificationQueue: Notification[] = [];
    private isVisible: boolean = true;

    constructor(containerId: string, options: StatusOptions = {}) {
        this.container = document.getElementById(containerId) || document.body;
        this.initializeUI();
        this.connectToNexus();
        this.startHealthChecks();
    }

    private initializeUI(): void {
        const statusContainer = document.createElement('div');
        statusContainer.id = 'status-panels-container';
        statusContainer.className = 'status-panels-container';
        statusContainer.innerHTML = `
      <div class="status-header">
        <h3>System Status</h3>
        <div class="status-controls">
          <button id="clear-notifications">🗑️</button>
          <button id="toggle-status">👁️</button>
        </div>
      </div>

      <div class="status-grid" id="status-grid">
        <!-- Dynamic status panels -->
      </div>

      <div class="notification-center" id="notification-center">
        <div class="notification-header">
          <h4>Notifications</h4>
          <span class="notification-count" id="notification-count">0</span>
        </div>
        <div class="notification-list" id="notification-list">
          <!-- Dynamic notifications -->
        </div>
      </div>

      <div class="quick-actions" id="quick-actions">
        <button class="quick-btn" id="emergency-stop">🛑 Emergency Stop</button>
        <button class="quick-btn" id="restart-services">🔄 Restart Services</button>
        <button class="quick-btn" id="clear-cache">🧹 Clear Cache</button>
      </div>
    `;

        this.container.appendChild(statusContainer);
        this.setupEventListeners();
        this.createDefaultPanels();
    }

    private setupEventListeners(): void {
        document.getElementById('clear-notifications')?.addEventListener('click', () => this.clearNotifications());
        document.getElementById('toggle-status')?.addEventListener('click', () => this.toggle());

        // Quick actions
        document.getElementById('emergency-stop')?.addEventListener('click', () => this.emergencyStop());
        document.getElementById('restart-services')?.addEventListener('click', () => this.restartServices());
        document.getElementById('clear-cache')?.addEventListener('click', () => this.clearCache());
    }

    private connectToNexus(): void {
        try {
            this.websocket = new WebSocket('ws://localhost:9000');

            this.websocket.onopen = () => {
                this.sendToNexus({ type: 'status_init', timestamp: Date.now() });
                this.addNotification('Connected to VectorLab Nexus', 'success');
            };

            this.websocket.onmessage = (event) => {
                const data = JSON.parse(event.data);
                this.handleNexusData(data);
            };

            this.websocket.onclose = () => {
                this.addNotification('Disconnected from VectorLab Nexus', 'error');
                this.attemptReconnect();
            };

        } catch (error) {
            this.addNotification('Failed to connect to VectorLab Nexus', 'error');
        }
    }

    private createDefaultPanels(): void {
        // System Health Panel
        this.createPanel('system-health', {
            title: 'System Health',
            icon: '💚',
            status: 'healthy',
            details: ['CPU: Normal', 'Memory: 45% used', 'Disk: 78% available'],
            actions: ['View Details', 'Run Diagnostics']
        });

        // Connection Status Panel
        this.createPanel('connections', {
            title: 'Connections',
            icon: '🔗',
            status: 'connected',
            details: ['VectorLab Nexus: Active', 'WebSocket: Connected', 'Services: 3/3 running'],
            actions: ['Test Connections', 'Refresh']
        });

        // World Engine Status Panel
        this.createPanel('world-engine', {
            title: 'World Engine',
            icon: '🌍',
            status: 'active',
            details: ['Domains: 6 loaded', 'Services: All operational', 'Last sync: 2s ago'],
            actions: ['Reload Domains', 'Force Sync']
        });

        // VectorLab Brain Panel
        this.createPanel('vectorlab-brain', {
            title: 'VectorLab Brain',
            icon: '🧠',
            status: 'processing',
            details: ['Neural Activity: High', 'Memory Units: 1.2M active', 'Processing: 15 queries/sec'],
            actions: ['View Activity', 'Optimize']
        });

        // Security Status Panel
        this.createPanel('security', {
            title: 'Security',
            icon: '🔒',
            status: 'secure',
            details: ['ECDSA P-256: Active', 'Vault: Sealed', 'Threats: None detected'],
            actions: ['Security Audit', 'Update Keys']
        });

        // Performance Panel
        this.createPanel('performance', {
            title: 'Performance',
            icon: '⚡',
            status: 'optimal',
            details: ['Response Time: 45ms', 'Throughput: 250 ops/sec', 'Latency: Low'],
            actions: ['Benchmark', 'Optimize']
        });
    }

    private createPanel(id: string, config: PanelConfig): void {
        const panel = new StatusPanel(id, config, this);
        this.panels.set(id, panel);
        this.renderPanel(panel);
    }

    private renderPanel(panel: StatusPanel): void {
        const grid = document.getElementById('status-grid');
        if (!grid) return;

        const panelElement = document.createElement('div');
        panelElement.className = `status-panel ${panel.config.status}`;
        panelElement.id = `panel-${panel.id}`;
        panelElement.innerHTML = `
      <div class="panel-header">
        <span class="panel-icon">${panel.config.icon}</span>
        <div class="panel-title-group">
          <h4 class="panel-title">${panel.config.title}</h4>
          <span class="panel-status-badge ${panel.config.status}">${panel.config.status}</span>
        </div>
      </div>

      <div class="panel-details">
        ${panel.config.details.map(detail => `<div class="detail-line">${detail}</div>`).join('')}
      </div>

      <div class="panel-actions">
        ${panel.config.actions.map(action => `
          <button class="panel-action-btn" data-panel="${panel.id}" data-action="${action.toLowerCase().replace(' ', '-')}">${action}</button>
        `).join('')}
      </div>

      <div class="panel-timestamp">
        Last updated: ${new Date().toLocaleTimeString()}
      </div>
    `;

        grid.appendChild(panelElement);

        // Add event listeners for action buttons
        panelElement.querySelectorAll('.panel-action-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const target = e.target as HTMLElement;
                const panelId = target.dataset.panel;
                const action = target.dataset.action;
                if (panelId && action) {
                    this.handlePanelAction(panelId, action);
                }
            });
        });
    }

    private handlePanelAction(panelId: string, action: string): void {
        this.addNotification(`Executing action: ${action} on ${panelId}`, 'info');

        // Send action to VectorLab Nexus
        if (this.websocket?.readyState === WebSocket.OPEN) {
            this.sendToNexus({
                type: 'panel_action',
                panelId,
                action,
                timestamp: Date.now()
            });
        }
    }

    private handleNexusData(data: any): void {
        switch (data.type) {
            case 'status_update':
                this.updatePanelStatus(data.panelId, data.status);
                break;
            case 'system_notification':
                this.addNotification(data.message, data.level);
                break;
            case 'health_check_result':
                this.updateHealthStatus(data.results);
                break;
            case 'performance_update':
                this.updatePerformanceMetrics(data.metrics);
                break;
        }
    }

    private updatePanelStatus(panelId: string, newStatus: any): void {
        const panel = this.panels.get(panelId);
        if (!panel) return;

        panel.config.status = newStatus.status;
        panel.config.details = newStatus.details || panel.config.details;

        const panelElement = document.getElementById(`panel-${panelId}`);
        if (panelElement) {
            // Update status badge
            const statusBadge = panelElement.querySelector('.panel-status-badge');
            if (statusBadge) {
                statusBadge.className = `panel-status-badge ${newStatus.status}`;
                statusBadge.textContent = newStatus.status;
            }

            // Update panel class
            panelElement.className = `status-panel ${newStatus.status}`;

            // Update details
            const detailsContainer = panelElement.querySelector('.panel-details');
            if (detailsContainer) {
                detailsContainer.innerHTML = panel.config.details.map(detail => `<div class="detail-line">${detail}</div>`).join('');
            }

            // Update timestamp
            const timestamp = panelElement.querySelector('.panel-timestamp');
            if (timestamp) {
                timestamp.textContent = `Last updated: ${new Date().toLocaleTimeString()}`;
            }
        }
    }

    private addNotification(message: string, level: 'info' | 'success' | 'warning' | 'error'): void {
        const notification: Notification = {
            id: Date.now().toString(),
            message,
            level,
            timestamp: new Date(),
            read: false
        };

        this.notificationQueue.unshift(notification);

        // Keep only last 50 notifications
        if (this.notificationQueue.length > 50) {
            this.notificationQueue = this.notificationQueue.slice(0, 50);
        }

        this.renderNotifications();
        this.showToast(notification);
    }

    private renderNotifications(): void {
        const notificationList = document.getElementById('notification-list');
        const notificationCount = document.getElementById('notification-count');

        if (!notificationList || !notificationCount) return;

        notificationCount.textContent = this.notificationQueue.filter(n => !n.read).length.toString();

        notificationList.innerHTML = this.notificationQueue.map(notification => `
      <div class="notification-item ${notification.level} ${notification.read ? 'read' : 'unread'}"
           data-id="${notification.id}">
        <div class="notification-content">
          <div class="notification-message">${notification.message}</div>
          <div class="notification-time">${notification.timestamp.toLocaleTimeString()}</div>
        </div>
        <button class="notification-dismiss" data-id="${notification.id}">×</button>
      </div>
    `).join('');

        // Add event listeners
        notificationList.querySelectorAll('.notification-dismiss').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const target = e.target as HTMLElement;
                const id = target.dataset.id;
                if (id) this.dismissNotification(id);
            });
        });

        notificationList.querySelectorAll('.notification-item').forEach(item => {
            item.addEventListener('click', (e) => {
                const target = e.currentTarget as HTMLElement;
                const id = target.dataset.id;
                if (id) this.markNotificationRead(id);
            });
        });
    }

    private showToast(notification: Notification): void {
        const toast = document.createElement('div');
        toast.className = `toast-notification ${notification.level}`;
        toast.innerHTML = `
      <div class="toast-content">
        <div class="toast-message">${notification.message}</div>
        <button class="toast-close">×</button>
      </div>
    `;

        // Position toast
        toast.style.position = 'fixed';
        toast.style.top = '20px';
        toast.style.right = '20px';
        toast.style.zIndex = '99999';
        toast.style.minWidth = '300px';
        toast.style.maxWidth = '500px';

        document.body.appendChild(toast);

        // Close button
        toast.querySelector('.toast-close')?.addEventListener('click', () => {
            document.body.removeChild(toast);
        });

        // Auto-dismiss
        setTimeout(() => {
            if (document.body.contains(toast)) {
                document.body.removeChild(toast);
            }
        }, 5000);

        // Slide in animation
        toast.style.transform = 'translateX(100%)';
        toast.style.opacity = '0';
        requestAnimationFrame(() => {
            toast.style.transition = 'transform 0.3s ease, opacity 0.3s ease';
            toast.style.transform = 'translateX(0)';
            toast.style.opacity = '1';
        });
    }

    private dismissNotification(id: string): void {
        this.notificationQueue = this.notificationQueue.filter(n => n.id !== id);
        this.renderNotifications();
    }

    private markNotificationRead(id: string): void {
        const notification = this.notificationQueue.find(n => n.id === id);
        if (notification) {
            notification.read = true;
            this.renderNotifications();
        }
    }

    private clearNotifications(): void {
        this.notificationQueue = [];
        this.renderNotifications();
    }

    private startHealthChecks(): void {
        setInterval(() => {
            this.performHealthCheck();
        }, 10000); // Every 10 seconds
    }

    private performHealthCheck(): void {
        if (this.websocket?.readyState === WebSocket.OPEN) {
            this.sendToNexus({
                type: 'health_check_request',
                timestamp: Date.now()
            });
        }
    }

    private updateHealthStatus(results: any): void {
        // Update system health panel based on health check results
        const healthPanel = this.panels.get('system-health');
        if (healthPanel) {
            const overallStatus = results.overall_status || 'unknown';
            const details = [
                `CPU: ${results.cpu_status || 'Unknown'}`,
                `Memory: ${results.memory_usage || 'Unknown'}`,
                `Disk: ${results.disk_usage || 'Unknown'}`
            ];

            this.updatePanelStatus('system-health', {
                status: overallStatus,
                details: details
            });
        }
    }

    private updatePerformanceMetrics(metrics: any): void {
        const performancePanel = this.panels.get('performance');
        if (performancePanel) {
            const details = [
                `Response Time: ${metrics.response_time || 'Unknown'}`,
                `Throughput: ${metrics.throughput || 'Unknown'}`,
                `Latency: ${metrics.latency || 'Unknown'}`
            ];

            this.updatePanelStatus('performance', {
                status: metrics.overall_performance || 'unknown',
                details: details
            });
        }
    }

    private emergencyStop(): void {
        if (confirm('Are you sure you want to perform an emergency stop? This will halt all World Engine services.')) {
            this.sendToNexus({
                type: 'emergency_stop',
                timestamp: Date.now()
            });
            this.addNotification('Emergency stop initiated', 'warning');
        }
    }

    private restartServices(): void {
        if (confirm('Are you sure you want to restart all services? This may cause temporary downtime.')) {
            this.sendToNexus({
                type: 'restart_services',
                timestamp: Date.now()
            });
            this.addNotification('Services restart initiated', 'info');
        }
    }

    private clearCache(): void {
        this.sendToNexus({
            type: 'clear_cache',
            timestamp: Date.now()
        });
        this.addNotification('Cache clearing initiated', 'info');
    }

    private sendToNexus(data: any): void {
        if (this.websocket?.readyState === WebSocket.OPEN) {
            this.websocket.send(JSON.stringify(data));
        }
    }

    private attemptReconnect(): void {
        setTimeout(() => {
            if (!this.websocket || this.websocket.readyState === WebSocket.CLOSED) {
                this.connectToNexus();
            }
        }, 5000);
    }

    public toggle(): void {
        const container = document.getElementById('status-panels-container');
        if (!container) return;

        if (this.isVisible) {
            container.style.display = 'none';
            this.isVisible = false;
        } else {
            container.style.display = 'block';
            this.isVisible = true;
        }
    }
}

// Supporting classes and interfaces
class StatusPanel {
    constructor(
        public id: string,
        public config: PanelConfig,
        private parent: StatusPanels
    ) { }
}

interface PanelConfig {
    title: string;
    icon: string;
    status: string;
    details: string[];
    actions: string[];
}

interface Notification {
    id: string;
    message: string;
    level: 'info' | 'success' | 'warning' | 'error';
    timestamp: Date;
    read: boolean;
}

interface StatusOptions {
    theme?: string;
    position?: string;
}

// CSS styles for Status Panels
export const statusPanelStyles = `
.status-panels-container {
  position: fixed;
  top: 20px;
  right: 20px;
  width: 400px;
  max-height: calc(100vh - 40px);
  background: var(--bg-primary, #2a2a2a);
  border: 1px solid var(--border-color, #4a4a4a);
  border-radius: 12px;
  display: flex;
  flex-direction: column;
  font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
  z-index: 8000;
  box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
}

.status-header {
  padding: 16px 20px;
  background: var(--bg-secondary, #4a4a4a);
  border-bottom: 1px solid var(--border-color, #4a4a4a);
  display: flex;
  justify-content: space-between;
  align-items: center;
  border-radius: 12px 12px 0 0;
}

.status-header h3 {
  margin: 0;
  color: var(--text-light, #fff);
  font-size: 16px;
}

.status-controls {
  display: flex;
  gap: 8px;
}

.status-controls button {
  background: var(--accent-color, #00d4aa);
  border: none;
  border-radius: 6px;
  padding: 6px 10px;
  cursor: pointer;
  font-size: 14px;
}

.status-grid {
  padding: 16px;
  display: grid;
  gap: 12px;
  max-height: 400px;
  overflow-y: auto;
}

.status-panel {
  background: var(--bg-secondary, #4a4a4a);
  border: 1px solid var(--border-color, #4a4a4a);
  border-radius: 8px;
  padding: 16px;
  transition: all 0.2s ease;
}

.status-panel.healthy, .status-panel.connected, .status-panel.active, .status-panel.secure, .status-panel.optimal {
  border-left: 4px solid var(--success-color, #00ff00);
}

.status-panel.warning, .status-panel.processing {
  border-left: 4px solid var(--warning-color, #ff9500);
}

.status-panel.error, .status-panel.disconnected, .status-panel.critical {
  border-left: 4px solid var(--error-color, #ff5555);
}

.panel-header {
  display: flex;
  align-items: center;
  margin-bottom: 12px;
}

.panel-icon {
  font-size: 24px;
  margin-right: 12px;
}

.panel-title-group {
  flex: 1;
}

.panel-title {
  margin: 0 0 4px 0;
  color: var(--text-light, #fff);
  font-size: 14px;
  font-weight: 600;
}

.panel-status-badge {
  font-size: 10px;
  padding: 2px 8px;
  border-radius: 10px;
  text-transform: uppercase;
  font-weight: 600;
}

.panel-status-badge.healthy, .panel-status-badge.connected, .panel-status-badge.active, .panel-status-badge.secure, .panel-status-badge.optimal {
  background: var(--success-color, #00ff00);
  color: var(--text-dark, #000);
}

.panel-status-badge.warning, .panel-status-badge.processing {
  background: var(--warning-color, #ff9500);
  color: var(--text-dark, #000);
}

.panel-status-badge.error, .panel-status-badge.disconnected, .panel-status-badge.critical {
  background: var(--error-color, #ff5555);
  color: var(--text-light, #fff);
}

.panel-details {
  margin-bottom: 12px;
}

.detail-line {
  font-size: 12px;
  color: var(--text-secondary, #aaa);
  margin-bottom: 4px;
}

.panel-actions {
  display: flex;
  gap: 8px;
  flex-wrap: wrap;
  margin-bottom: 8px;
}

.panel-action-btn {
  background: var(--accent-color, #00d4aa);
  border: none;
  border-radius: 4px;
  padding: 6px 10px;
  cursor: pointer;
  font-size: 11px;
  color: var(--text-dark, #000);
  font-weight: 500;
}

.panel-action-btn:hover {
  opacity: 0.8;
}

.panel-timestamp {
  font-size: 10px;
  color: var(--text-tertiary, #666);
  text-align: right;
}

.notification-center {
  border-top: 1px solid var(--border-color, #4a4a4a);
  max-height: 200px;
  overflow-y: auto;
}

.notification-header {
  padding: 12px 16px;
  background: var(--bg-tertiary, #1a1a1a);
  display: flex;
  justify-content: space-between;
  align-items: center;
  border-bottom: 1px solid var(--border-color, #4a4a4a);
}

.notification-header h4 {
  margin: 0;
  color: var(--text-light, #fff);
  font-size: 14px;
}

.notification-count {
  background: var(--accent-color, #00d4aa);
  color: var(--text-dark, #000);
  padding: 2px 8px;
  border-radius: 10px;
  font-size: 11px;
  font-weight: 600;
}

.notification-list {
  max-height: 150px;
  overflow-y: auto;
}

.notification-item {
  padding: 8px 16px;
  border-bottom: 1px solid var(--border-color, #4a4a4a);
  display: flex;
  justify-content: space-between;
  align-items: center;
  cursor: pointer;
}

.notification-item:hover {
  background: var(--bg-secondary, #4a4a4a);
}

.notification-item.unread {
  background: rgba(0, 212, 170, 0.1);
}

.notification-content {
  flex: 1;
}

.notification-message {
  font-size: 12px;
  color: var(--text-light, #fff);
  margin-bottom: 2px;
}

.notification-time {
  font-size: 10px;
  color: var(--text-secondary, #aaa);
}

.notification-dismiss {
  background: none;
  border: none;
  color: var(--text-secondary, #aaa);
  cursor: pointer;
  font-size: 16px;
  padding: 4px;
}

.notification-dismiss:hover {
  color: var(--text-light, #fff);
}

.notification-item.info { border-left: 3px solid var(--info-color, #0099ff); }
.notification-item.success { border-left: 3px solid var(--success-color, #00ff00); }
.notification-item.warning { border-left: 3px solid var(--warning-color, #ff9500); }
.notification-item.error { border-left: 3px solid var(--error-color, #ff5555); }

.quick-actions {
  padding: 12px 16px;
  border-top: 1px solid var(--border-color, #4a4a4a);
  display: flex;
  gap: 8px;
  flex-wrap: wrap;
  border-radius: 0 0 12px 12px;
}

.quick-btn {
  background: var(--bg-secondary, #4a4a4a);
  border: 1px solid var(--border-color, #4a4a4a);
  border-radius: 6px;
  padding: 8px 12px;
  cursor: pointer;
  font-size: 11px;
  color: var(--text-light, #fff);
  font-weight: 500;
  flex: 1;
  min-width: 100px;
}

.quick-btn:hover {
  background: var(--accent-color, #00d4aa);
  color: var(--text-dark, #000);
}

.toast-notification {
  background: var(--bg-primary, #2a2a2a);
  border: 1px solid var(--border-color, #4a4a4a);
  border-radius: 8px;
  padding: 16px;
  margin-bottom: 8px;
  box-shadow: 0 4px 16px rgba(0, 0, 0, 0.3);
}

.toast-notification.info { border-left: 4px solid var(--info-color, #0099ff); }
.toast-notification.success { border-left: 4px solid var(--success-color, #00ff00); }
.toast-notification.warning { border-left: 4px solid var(--warning-color, #ff9500); }
.toast-notification.error { border-left: 4px solid var(--error-color, #ff5555); }

.toast-content {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.toast-message {
  color: var(--text-light, #fff);
  font-size: 14px;
  flex: 1;
  margin-right: 12px;
}

.toast-close {
  background: none;
  border: none;
  color: var(--text-secondary, #aaa);
  cursor: pointer;
  font-size: 18px;
}

.toast-close:hover {
  color: var(--text-light, #fff);
}
`;
