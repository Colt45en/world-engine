/**
 * World Engine UI Manager
 * Central coordinator for all UI components with unified styling and state management
 */

import { WorldEngineChatbot, chatbotStyles } from './chatbot/WorldEngineChatbot.js';
import { SystemInspector, inspectorStyles } from './inspector/SystemInspector.js';
import { StatusPanels, statusPanelStyles } from './status/StatusPanels.js';
import { NexusDashboard, nexusDashboardStyles } from './nexus/NexusDashboard.js';

export class WorldEngineUIManager {
    private chatbot: WorldEngineChatbot | null = null;
    private inspector: SystemInspector | null = null;
    private statusPanels: StatusPanels | null = null;
    private nexusDashboard: NexusDashboard | null = null;

    private currentTheme: string = 'nexus';
    private currentLayout: string = 'development_mode';
    private isInitialized: boolean = false;

    constructor(options: UIManagerOptions = {}) {
        this.currentTheme = options.theme || 'nexus';
        this.currentLayout = options.layout || 'development_mode';

        this.initializeUI();
    }

    private async initializeUI(): Promise<void> {
        try {
            // Inject CSS styles
            this.injectStyles();

            // Create containers for each UI component
            this.createUIContainers();

            // Initialize components based on layout
            await this.initializeComponents();

            // Set up global event handlers
            this.setupGlobalHandlers();

            // Apply theme
            this.applyTheme(this.currentTheme);

            this.isInitialized = true;
            console.log('World Engine UI Manager initialized successfully');

        } catch (error) {
            console.error('Failed to initialize World Engine UI Manager:', error);
        }
    }

    private injectStyles(): void {
        const styleSheet = document.createElement('style');
        styleSheet.textContent = `
      ${this.getBaseStyles()}
      ${chatbotStyles}
      ${inspectorStyles}
      ${statusPanelStyles}
      ${nexusDashboardStyles}
    `;
        document.head.appendChild(styleSheet);
    }

    private getBaseStyles(): string {
        return `
      :root {
        /* Nexus Theme Variables */
        --bg-primary: #001122;
        --bg-secondary: #003344;
        --bg-tertiary: #000811;
        --accent-color: #00ffcc;
        --text-light: #ccffff;
        --text-dark: #001122;
        --text-secondary: #99ccdd;
        --text-tertiary: #666;
        --border-color: #00ffcc;
        --success-color: #00ff00;
        --warning-color: #ff9500;
        --error-color: #ff5555;
        --info-color: #0099ff;

        /* Default Theme Variables (fallback) */
        --default-bg-primary: #2a2a2a;
        --default-bg-secondary: #4a4a4a;
        --default-bg-tertiary: #1a1a1a;
        --default-accent-color: #00d4aa;
        --default-text-light: #ffffff;
        --default-text-dark: #000000;
        --default-border-color: #4a4a4a;
      }

      /* World Engine UI Base Styles */
      .world-engine-ui {
        font-family: 'Segoe UI', 'SF Pro Display', -apple-system, BlinkMacSystemFont, sans-serif;
        font-size: 14px;
        line-height: 1.5;
        color: var(--text-light);
        background: var(--bg-primary);
      }

      /* Utility Classes */
      .we-hidden { display: none !important; }
      .we-visible { display: block !important; }
      .we-fade-in {
        animation: weFadeIn 0.3s ease-in-out;
      }
      .we-fade-out {
        animation: weFadeOut 0.3s ease-in-out;
      }
      .we-slide-up {
        animation: weSlideUp 0.3s ease-in-out;
      }

      @keyframes weGlow {
        0%, 100% { box-shadow: 0 0 8px var(--accent-color); }
        50% { box-shadow: 0 0 16px var(--accent-color); }
      }

      @keyframes weHeartbeat {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.05); }
      }

      @keyframes wePulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.7; }
      }

      @keyframes weRotate {
        from { transform: rotate(0deg); }
        to { transform: rotate(360deg); }
      }

      @keyframes weSlideUp {
        from { transform: translateY(20px); opacity: 0; }
        to { transform: translateY(0); opacity: 1; }
      }

      @keyframes weFadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
      }

      @keyframes weFadeOut {
        from { opacity: 1; }
        to { opacity: 0; }
      }

      /* Global UI Container */
      #world-engine-ui-root {
        position: fixed;
        top: 0;
        left: 0;
        width: 100vw;
        height: 100vh;
        pointer-events: none;
        z-index: 1000;
      }

      #world-engine-ui-root > * {
        pointer-events: auto;
      }

      /* Theme Switcher */
      .theme-switcher {
        position: fixed;
        top: 20px;
        left: 50%;
        transform: translateX(-50%);
        background: var(--bg-secondary);
        border: 1px solid var(--border-color);
        border-radius: 20px;
        padding: 4px;
        display: flex;
        gap: 4px;
        z-index: 10001;
        font-size: 12px;
      }

      .theme-option {
        background: none;
        border: none;
        color: var(--text-secondary);
        padding: 8px 16px;
        border-radius: 16px;
        cursor: pointer;
        transition: all 0.2s ease;
        font-size: 12px;
        font-weight: 500;
      }

      .theme-option.active {
        background: var(--accent-color);
        color: var(--text-dark);
      }

      .theme-option:hover {
        color: var(--text-light);
      }

      /* Layout Indicators */
      .layout-indicator {
        position: fixed;
        bottom: 20px;
        left: 20px;
        background: rgba(0, 0, 0, 0.8);
        color: var(--accent-color);
        padding: 8px 12px;
        border-radius: 8px;
        font-size: 11px;
        font-weight: 600;
        border: 1px solid var(--accent-color);
        z-index: 10001;
      }

      /* Notification Toast Styles */
      .we-toast-container {
        position: fixed;
        top: 80px;
        right: 20px;
        z-index: 10002;
        display: flex;
        flex-direction: column;
        gap: 8px;
        pointer-events: none;
      }

      .we-toast {
        background: var(--bg-secondary);
        border: 1px solid var(--border-color);
        border-radius: 8px;
        padding: 12px 16px;
        color: var(--text-light);
        font-size: 13px;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.3);
        animation: weSlideUp 0.3s ease;
        pointer-events: auto;
        max-width: 300px;
      }

      .we-toast.success { border-left: 4px solid var(--success-color); }
      .we-toast.warning { border-left: 4px solid var(--warning-color); }
      .we-toast.error { border-left: 4px solid var(--error-color); }
      .we-toast.info { border-left: 4px solid var(--info-color); }

      /* Loading Spinner */
      .we-loading {
        display: inline-block;
        width: 20px;
        height: 20px;
        border: 2px solid var(--bg-secondary);
        border-radius: 50%;
        border-top-color: var(--accent-color);
        animation: weRotate 1s linear infinite;
      }

      /* Custom Scrollbars */
      .we-scrollbar::-webkit-scrollbar {
        width: 8px;
        height: 8px;
      }

      .we-scrollbar::-webkit-scrollbar-track {
        background: var(--bg-tertiary);
        border-radius: 4px;
      }

      .we-scrollbar::-webkit-scrollbar-thumb {
        background: var(--accent-color);
        border-radius: 4px;
        opacity: 0.7;
      }

      .we-scrollbar::-webkit-scrollbar-thumb:hover {
        opacity: 1;
      }
    `;
    }

    private createUIContainers(): void {
        // Create main UI root container
        const uiRoot = document.createElement('div');
        uiRoot.id = 'world-engine-ui-root';
        uiRoot.className = 'world-engine-ui';
        document.body.appendChild(uiRoot);

        // Create individual component containers
        const containers = [
            { id: 'we-chatbot-container', class: 'we-chatbot-container' },
            { id: 'we-inspector-container', class: 'we-inspector-container' },
            { id: 'we-status-container', class: 'we-status-container' },
            { id: 'we-nexus-container', class: 'we-nexus-container' },
            { id: 'we-toast-container', class: 'we-toast-container' }
        ];

        containers.forEach(({ id, class: className }) => {
            const container = document.createElement('div');
            container.id = id;
            container.className = className;
            uiRoot.appendChild(container);
        });

        // Add theme switcher
        this.createThemeSwitcher(uiRoot);

        // Add layout indicator
        this.createLayoutIndicator(uiRoot);
    }

    private createThemeSwitcher(parent: HTMLElement): void {
        const themeSwitcher = document.createElement('div');
        themeSwitcher.className = 'theme-switcher';
        themeSwitcher.innerHTML = `
      <button class="theme-option" data-theme="nexus">Nexus</button>
      <button class="theme-option active" data-theme="default">Default</button>
    `;

        parent.appendChild(themeSwitcher);

        // Add event listeners
        themeSwitcher.querySelectorAll('.theme-option').forEach(option => {
            option.addEventListener('click', (e) => {
                const target = e.target as HTMLElement;
                const theme = target.dataset.theme;
                if (theme) this.switchTheme(theme);
            });
        });
    }

    private createLayoutIndicator(parent: HTMLElement): void {
        const indicator = document.createElement('div');
        indicator.className = 'layout-indicator';
        indicator.id = 'layout-indicator';
        indicator.textContent = `Layout: ${this.currentLayout.replace('_', ' ').toUpperCase()}`;
        parent.appendChild(indicator);
    }

    private async initializeComponents(): Promise<void> {
        const layoutConfig = this.getLayoutConfig(this.currentLayout);

        // Initialize components based on layout configuration
        if (layoutConfig.chatbot !== 'hidden') {
            this.chatbot = new WorldEngineChatbot('we-chatbot-container', {
                position: layoutConfig.chatbot as any,
                contextData: { layout: this.currentLayout, theme: this.currentTheme }
            });
        }

        if (layoutConfig.inspector !== 'hidden') {
            this.inspector = new SystemInspector('we-inspector-container', {});
        }

        if (layoutConfig.status !== 'hidden') {
            this.statusPanels = new StatusPanels('we-status-container', {});
        }

        if (layoutConfig.nexus !== 'hidden') {
            this.nexusDashboard = new NexusDashboard('we-nexus-container', {});
        }
    }

    private getLayoutConfig(layout: string): any {
        const layouts = {
            demo_mode: {
                chatbot: 'bottom-right',
                inspector: 'hidden',
                status: 'minimal',
                nexus: 'hidden'
            },
            training_mode: {
                chatbot: 'bottom-left',
                inspector: 'hidden',
                status: 'progress',
                nexus: 'hidden'
            },
            development_mode: {
                chatbot: 'right-panel',
                inspector: 'visible',
                status: 'full',
                nexus: 'visible'
            }
        };

        return layouts[layout as keyof typeof layouts] || layouts.development_mode;
    }

    private setupGlobalHandlers(): void {
        // Global keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            // Ctrl+Shift+D: Toggle Development Mode
            if (e.ctrlKey && e.shiftKey && e.key === 'D') {
                e.preventDefault();
                this.toggleDevelopmentMode();
            }

            // Ctrl+Shift+T: Switch Theme
            if (e.ctrlKey && e.shiftKey && e.key === 'T') {
                e.preventDefault();
                this.toggleTheme();
            }

            // Ctrl+Shift+H: Toggle Help
            if (e.ctrlKey && e.shiftKey && e.key === 'H') {
                e.preventDefault();
                this.showHelp();
            }
        });

        // Handle window resize
        window.addEventListener('resize', () => {
            this.handleResize();
        });

        // Handle visibility change
        document.addEventListener('visibilitychange', () => {
            if (document.hidden) {
                this.handlePageHide();
            } else {
                this.handlePageShow();
            }
        });
    }

    private applyTheme(theme: string): void {
        const themeConfigs = {
            nexus: {
                '--bg-primary': '#001122',
                '--bg-secondary': '#003344',
                '--bg-tertiary': '#000811',
                '--accent-color': '#00ffcc',
                '--text-light': '#ccffff',
                '--text-dark': '#001122',
                '--text-secondary': '#99ccdd',
                '--border-color': '#00ffcc'
            },
            default: {
                '--bg-primary': '#2a2a2a',
                '--bg-secondary': '#4a4a4a',
                '--bg-tertiary': '#1a1a1a',
                '--accent-color': '#00d4aa',
                '--text-light': '#ffffff',
                '--text-dark': '#000000',
                '--text-secondary': '#aaaaaa',
                '--border-color': '#4a4a4a'
            }
        };

        const config = themeConfigs[theme as keyof typeof themeConfigs] || themeConfigs.default;

        Object.entries(config).forEach(([property, value]) => {
            document.documentElement.style.setProperty(property, value);
        });

        // Update theme switcher
        document.querySelectorAll('.theme-option').forEach(option => {
            option.classList.toggle('active', option.getAttribute('data-theme') === theme);
        });

        this.currentTheme = theme;
        this.showToast(`Theme switched to ${theme}`, 'info');
    }

    private switchTheme(theme: string): void {
        this.applyTheme(theme);

        // Update components with new theme context
        if (this.chatbot) {
            this.chatbot.updateContext({ theme });
        }
    }

    private toggleTheme(): void {
        const nextTheme = this.currentTheme === 'nexus' ? 'default' : 'nexus';
        this.switchTheme(nextTheme);
    }

    private toggleDevelopmentMode(): void {
        const newLayout = this.currentLayout === 'development_mode' ? 'demo_mode' : 'development_mode';
        this.switchLayout(newLayout);
    }

    private switchLayout(layout: string): void {
        this.currentLayout = layout;

        // Update layout indicator
        const indicator = document.getElementById('layout-indicator');
        if (indicator) {
            indicator.textContent = `Layout: ${layout.replace('_', ' ').toUpperCase()}`;
        }

        // Reconfigure components for new layout
        this.reconfigureComponents(layout);

        this.showToast(`Layout switched to ${layout.replace('_', ' ')}`, 'info');
    }

    private reconfigureComponents(layout: string): void {
        const layoutConfig = this.getLayoutConfig(layout);

        // Reconfigure chatbot position
        if (this.chatbot && layoutConfig.chatbot !== 'hidden') {
            // This would require extending the chatbot class to support position changes
            this.chatbot.updateContext({ layout, position: layoutConfig.chatbot });
        }

        // Show/hide components based on layout
        const components = {
            inspector: this.inspector,
            status: this.statusPanels,
            nexus: this.nexusDashboard
        };

        Object.entries(components).forEach(([key, component]) => {
            const config = layoutConfig[key];
            if (component) {
                if (config === 'hidden') {
                    // Hide component
                    const container = document.getElementById(`we-${key}-container`);
                    if (container) container.style.display = 'none';
                } else {
                    // Show component
                    const container = document.getElementById(`we-${key}-container`);
                    if (container) container.style.display = 'block';
                }
            }
        });
    }

    private showToast(message: string, type: 'info' | 'success' | 'warning' | 'error' = 'info'): void {
        const container = document.getElementById('we-toast-container');
        if (!container) return;

        const toast = document.createElement('div');
        toast.className = `we-toast ${type}`;
        toast.textContent = message;

        container.appendChild(toast);

        // Auto-remove after 3 seconds
        setTimeout(() => {
            if (container.contains(toast)) {
                toast.style.animation = 'weFadeOut 0.3s ease';
                setTimeout(() => {
                    if (container.contains(toast)) {
                        container.removeChild(toast);
                    }
                }, 300);
            }
        }, 3000);
    }

    private showHelp(): void {
        const helpContent = `
      <h3>🎮 World Engine UI Controls</h3>
      <div class="help-section">
        <h4>Keyboard Shortcuts:</h4>
        <ul>
          <li><kbd>C</kbd> - Toggle Chatbot</li>
          <li><kbd>F12</kbd> - Toggle System Inspector</li>
          <li><kbd>Ctrl+\`</kbd> - Toggle Nexus Dashboard</li>
          <li><kbd>Ctrl+Shift+D</kbd> - Toggle Development Mode</li>
          <li><kbd>Ctrl+Shift+T</kbd> - Switch Theme</li>
          <li><kbd>Ctrl+Shift+H</kbd> - Show this help</li>
        </ul>
      </div>
      <div class="help-section">
        <h4>Components:</h4>
        <ul>
          <li><strong>Chatbot</strong> - AI assistant for World Engine operations</li>
          <li><strong>Inspector</strong> - System monitoring and debugging</li>
          <li><strong>Status Panels</strong> - Real-time system status</li>
          <li><strong>Nexus Dashboard</strong> - VectorLab brain control interface</li>
        </ul>
      </div>
      <div class="help-section">
        <h4>Themes:</h4>
        <ul>
          <li><strong>Nexus</strong> - VectorLab-inspired cyan theme</li>
          <li><strong>Default</strong> - Standard dark theme</li>
        </ul>
      </div>
    `;

        this.showModal('World Engine UI Help', helpContent);
    }

    private showModal(title: string, content: string): void {
        const modal = document.createElement('div');
        modal.className = 'we-modal';
        modal.innerHTML = `
      <div class="we-modal-backdrop"></div>
      <div class="we-modal-content">
        <div class="we-modal-header">
          <h2>${title}</h2>
          <button class="we-modal-close">×</button>
        </div>
        <div class="we-modal-body">
          ${content}
        </div>
      </div>
    `;

        // Add modal styles
        const modalStyles = `
      .we-modal {
        position: fixed;
        top: 0;
        left: 0;
        width: 100vw;
        height: 100vh;
        z-index: 10003;
        display: flex;
        align-items: center;
        justify-content: center;
      }

      .we-modal-backdrop {
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: rgba(0, 0, 0, 0.8);
        backdrop-filter: blur(4px);
      }

      .we-modal-content {
        position: relative;
        background: var(--bg-primary);
        border: 2px solid var(--accent-color);
        border-radius: 12px;
        max-width: 600px;
        max-height: 80vh;
        overflow-y: auto;
        box-shadow: 0 16px 48px rgba(0, 0, 0, 0.5);
      }

      .we-modal-header {
        padding: 20px 24px;
        border-bottom: 1px solid var(--border-color);
        display: flex;
        justify-content: space-between;
        align-items: center;
      }

      .we-modal-header h2 {
        margin: 0;
        color: var(--accent-color);
      }

      .we-modal-close {
        background: none;
        border: none;
        color: var(--text-secondary);
        font-size: 24px;
        cursor: pointer;
        padding: 4px;
        border-radius: 4px;
      }

      .we-modal-close:hover {
        background: var(--bg-secondary);
        color: var(--text-light);
      }

      .we-modal-body {
        padding: 20px 24px;
        color: var(--text-light);
      }

      .help-section {
        margin-bottom: 20px;
      }

      .help-section h4 {
        color: var(--accent-color);
        margin: 0 0 8px 0;
      }

      .help-section ul {
        margin: 0;
        padding-left: 20px;
      }

      .help-section li {
        margin-bottom: 4px;
        color: var(--text-secondary);
      }

      kbd {
        background: var(--bg-secondary);
        border: 1px solid var(--border-color);
        border-radius: 4px;
        padding: 2px 6px;
        font-family: monospace;
        font-size: 11px;
        color: var(--accent-color);
      }
    `;

        // Inject modal styles
        const styleSheet = document.createElement('style');
        styleSheet.textContent = modalStyles;
        document.head.appendChild(styleSheet);

        document.body.appendChild(modal);

        // Event handlers
        const closeModal = () => {
            document.body.removeChild(modal);
            document.head.removeChild(styleSheet);
        };

        modal.querySelector('.we-modal-close')?.addEventListener('click', closeModal);
        modal.querySelector('.we-modal-backdrop')?.addEventListener('click', closeModal);

        // ESC key to close
        const handleEsc = (e: KeyboardEvent) => {
            if (e.key === 'Escape') {
                closeModal();
                document.removeEventListener('keydown', handleEsc);
            }
        };
        document.addEventListener('keydown', handleEsc);
    }

    private handleResize(): void {
        // Handle window resize events
        if (this.nexusDashboard) {
            // Trigger dashboard resize handling if needed
        }
    }

    private handlePageHide(): void {
        // Handle page visibility change (hidden)
        console.log('World Engine UI: Page hidden');
    }

    private handlePageShow(): void {
        // Handle page visibility change (visible)
        console.log('World Engine UI: Page visible');
    }

    // Public API methods

    public getChatbot(): WorldEngineChatbot | null {
        return this.chatbot;
    }

    public getInspector(): SystemInspector | null {
        return this.inspector;
    }

    public getStatusPanels(): StatusPanels | null {
        return this.statusPanels;
    }

    public getNexusDashboard(): NexusDashboard | null {
        return this.nexusDashboard;
    }

    public getCurrentTheme(): string {
        return this.currentTheme;
    }

    public getCurrentLayout(): string {
        return this.currentLayout;
    }

    public isReady(): boolean {
        return this.isInitialized;
    }

    public showNotification(message: string, type: 'info' | 'success' | 'warning' | 'error' = 'info'): void {
        this.showToast(message, type);
    }

    public destroy(): void {
        // Clean up all components
        this.chatbot = null;
        this.inspector?.destroy();
        this.inspector = null;
        this.statusPanels = null;
        this.nexusDashboard?.destroy();
        this.nexusDashboard = null;

        // Remove UI root
        const uiRoot = document.getElementById('world-engine-ui-root');
        if (uiRoot) {
            document.body.removeChild(uiRoot);
        }

        this.isInitialized = false;
    }
}

// Type definitions
export interface UIManagerOptions {
    theme?: 'nexus' | 'default';
    layout?: 'demo_mode' | 'training_mode' | 'development_mode';
}

// Global instance
declare global {
    interface Window {
        WorldEngineUI: WorldEngineUIManager;
    }
}

// Initialize global instance when script loads
if (typeof window !== 'undefined') {
    window.WorldEngineUI = new WorldEngineUIManager();
}

export default WorldEngineUIManager;
