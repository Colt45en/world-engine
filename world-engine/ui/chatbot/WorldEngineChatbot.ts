/**
 * World Engine Chatbot Component
 * AI-powered assistant with persistent connection and context awareness
 */

export class WorldEngineChatbot {
    private container: HTMLElement;
    private chatWindow: HTMLElement;
    private inputField: HTMLInputElement;
    private sendButton: HTMLElement;
    private toggleButton: HTMLElement;
    private isVisible: boolean = false;
    private position: 'bottom-right' | 'bottom-left' | 'right-panel' = 'bottom-right';
    private websocket: WebSocket | null = null;
    private contextData: any = {};

    constructor(containerId: string, options: ChatbotOptions = {}) {
        this.container = document.getElementById(containerId) || document.body;
        this.position = options.position || 'bottom-right';
        this.contextData = options.contextData || {};

        this.initializeUI();
        this.connectToNexus();
        this.setupEventListeners();
    }

    private initializeUI(): void {
        // Create main chatbot container
        this.chatWindow = document.createElement('div');
        this.chatWindow.className = `chatbot-window ${this.position}`;
        this.chatWindow.innerHTML = `
      <div class="chatbot-header">
        <h3>VectorLab AI Assistant</h3>
        <button class="minimize-btn">−</button>
        <button class="close-btn">×</button>
      </div>
      <div class="chatbot-messages" id="chatbot-messages">
        <div class="ai-message">
          <div class="message-content">
            Hello! I'm your VectorLab AI Assistant. I'm here to help with World Engine operations,
            mathematical computations, and system navigation. How can I assist you today?
          </div>
          <div class="message-time">${new Date().toLocaleTimeString()}</div>
        </div>
      </div>
      <div class="chatbot-input">
        <input type="text" id="chatbot-input" placeholder="Ask about World Engine, math, or system status..." />
        <button id="chatbot-send">Send</button>
      </div>
      <div class="chatbot-status">
        <span class="connection-status">🔗 Connected to VectorLab Nexus</span>
      </div>
    `;

        // Create toggle button
        this.toggleButton = document.createElement('button');
        this.toggleButton.className = 'chatbot-toggle';
        this.toggleButton.innerHTML = '💬';
        this.toggleButton.title = 'Toggle AI Assistant (C key)';

        // Position elements
        this.setChatbotPosition();

        // Add to DOM
        this.container.appendChild(this.chatWindow);
        this.container.appendChild(this.toggleButton);

        // Get references to interactive elements
        this.inputField = this.chatWindow.querySelector('#chatbot-input') as HTMLInputElement;
        this.sendButton = this.chatWindow.querySelector('#chatbot-send') as HTMLElement;

        // Initially hidden
        this.chatWindow.style.display = 'none';
    }

    private setChatbotPosition(): void {
        const positions = {
            'bottom-right': {
                chatWindow: { bottom: '20px', right: '20px', left: 'auto', top: 'auto' },
                toggleButton: { bottom: '20px', right: '20px', left: 'auto', top: 'auto' }
            },
            'bottom-left': {
                chatWindow: { bottom: '20px', left: '20px', right: 'auto', top: 'auto' },
                toggleButton: { bottom: '20px', left: '20px', right: 'auto', top: 'auto' }
            },
            'right-panel': {
                chatWindow: { top: '20px', right: '20px', bottom: '20px', left: 'auto', width: '350px', height: 'auto' },
                toggleButton: { top: '20px', right: '20px', left: 'auto', bottom: 'auto' }
            }
        };

        const pos = positions[this.position];
        Object.assign(this.chatWindow.style, pos.chatWindow);
        Object.assign(this.toggleButton.style, pos.toggleButton);
    }

    private connectToNexus(): void {
        try {
            this.websocket = new WebSocket('ws://localhost:9000');

            this.websocket.onopen = () => {
                console.log('Connected to VectorLab Nexus');
                this.updateConnectionStatus('Connected', 'success');

                // Send initial context
                this.sendToNexus({
                    type: 'ai_chat_init',
                    context: this.contextData,
                    timestamp: Date.now()
                });
            };

            this.websocket.onmessage = (event) => {
                const message = JSON.parse(event.data);
                this.handleNexusMessage(message);
            };

            this.websocket.onclose = () => {
                console.log('Disconnected from VectorLab Nexus');
                this.updateConnectionStatus('Disconnected', 'error');
                this.attemptReconnect();
            };

            this.websocket.onerror = (error) => {
                console.error('WebSocket error:', error);
                this.updateConnectionStatus('Error', 'error');
            };

        } catch (error) {
            console.error('Failed to connect to VectorLab Nexus:', error);
            this.updateConnectionStatus('Failed', 'error');
        }
    }

    private setupEventListeners(): void {
        // Toggle button
        this.toggleButton.addEventListener('click', () => this.toggle());

        // Keyboard shortcut (C key)
        document.addEventListener('keydown', (e) => {
            if (e.key.toLowerCase() === 'c' && !e.ctrlKey && !e.altKey && !e.metaKey) {
                const activeElement = document.activeElement;
                if (activeElement && (activeElement.tagName === 'INPUT' || activeElement.tagName === 'TEXTAREA')) {
                    return; // Don't toggle if user is typing in an input field
                }
                this.toggle();
            }
        });

        // Send button
        this.sendButton.addEventListener('click', () => this.sendMessage());

        // Enter key in input
        this.inputField.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                this.sendMessage();
            }
        });

        // Header buttons
        this.chatWindow.querySelector('.minimize-btn')?.addEventListener('click', () => this.minimize());
        this.chatWindow.querySelector('.close-btn')?.addEventListener('click', () => this.hide());
    }

    public toggle(): void {
        if (this.isVisible) {
            this.hide();
        } else {
            this.show();
        }
    }

    public show(): void {
        this.chatWindow.style.display = 'block';
        this.toggleButton.style.display = 'none';
        this.isVisible = true;
        this.inputField.focus();

        // Animate in
        this.chatWindow.style.opacity = '0';
        this.chatWindow.style.transform = 'scale(0.9)';
        requestAnimationFrame(() => {
            this.chatWindow.style.transition = 'opacity 0.2s ease, transform 0.2s ease';
            this.chatWindow.style.opacity = '1';
            this.chatWindow.style.transform = 'scale(1)';
        });
    }

    public hide(): void {
        this.chatWindow.style.display = 'none';
        this.toggleButton.style.display = 'block';
        this.isVisible = false;
    }

    public minimize(): void {
        const messages = this.chatWindow.querySelector('.chatbot-messages') as HTMLElement;
        const input = this.chatWindow.querySelector('.chatbot-input') as HTMLElement;

        if (messages.style.display === 'none') {
            // Restore
            messages.style.display = 'block';
            input.style.display = 'block';
            this.chatWindow.querySelector('.minimize-btn')!.textContent = '−';
        } else {
            // Minimize
            messages.style.display = 'none';
            input.style.display = 'none';
            this.chatWindow.querySelector('.minimize-btn')!.textContent = '+';
        }
    }

    private sendMessage(): void {
        const message = this.inputField.value.trim();
        if (!message) return;

        // Add user message to chat
        this.addMessage(message, 'user');

        // Clear input
        this.inputField.value = '';

        // Send to VectorLab Nexus
        if (this.websocket && this.websocket.readyState === WebSocket.OPEN) {
            this.sendToNexus({
                type: 'ai_chat_request',
                message: message,
                context: this.contextData,
                timestamp: Date.now()
            });
        } else {
            this.addMessage('Unable to send message - not connected to VectorLab Nexus', 'error');
        }
    }

    private addMessage(content: string, type: 'user' | 'ai' | 'system' | 'error'): void {
        const messagesContainer = this.chatWindow.querySelector('#chatbot-messages');
        const messageDiv = document.createElement('div');
        messageDiv.className = `${type}-message`;

        messageDiv.innerHTML = `
      <div class="message-content">${this.formatMessage(content)}</div>
      <div class="message-time">${new Date().toLocaleTimeString()}</div>
    `;

        messagesContainer?.appendChild(messageDiv);
        messagesContainer?.scrollTo(0, messagesContainer.scrollHeight);
    }

    private formatMessage(content: string): string {
        // Basic markdown-like formatting
        return content
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
            .replace(/\*(.*?)\*/g, '<em>$1</em>')
            .replace(/`(.*?)`/g, '<code>$1</code>')
            .replace(/\n/g, '<br>');
    }

    private sendToNexus(data: any): void {
        if (this.websocket && this.websocket.readyState === WebSocket.OPEN) {
            this.websocket.send(JSON.stringify(data));
        }
    }

    private handleNexusMessage(message: any): void {
        switch (message.type) {
            case 'ai_chat_response':
                this.addMessage(message.response, 'ai');
                break;
            case 'system_notification':
                this.addMessage(message.content, 'system');
                break;
            case 'error':
                this.addMessage(`Error: ${message.message}`, 'error');
                break;
        }
    }

    private updateConnectionStatus(status: string, type: 'success' | 'error' | 'warning'): void {
        const statusElement = this.chatWindow.querySelector('.connection-status');
        if (statusElement) {
            const icons = { success: '🔗', error: '❌', warning: '⚠️' };
            statusElement.textContent = `${icons[type]} ${status}`;
            statusElement.className = `connection-status ${type}`;
        }
    }

    private attemptReconnect(): void {
        setTimeout(() => {
            if (!this.websocket || this.websocket.readyState === WebSocket.CLOSED) {
                console.log('Attempting to reconnect to VectorLab Nexus...');
                this.connectToNexus();
            }
        }, 5000);
    }

    public updateContext(newContext: any): void {
        this.contextData = { ...this.contextData, ...newContext };
        if (this.websocket && this.websocket.readyState === WebSocket.OPEN) {
            this.sendToNexus({
                type: 'context_update',
                context: this.contextData,
                timestamp: Date.now()
            });
        }
    }
}

// CSS styles (to be included in stylesheet)
export const chatbotStyles = `
.chatbot-window {
  position: fixed;
  width: 300px;
  height: 400px;
  background: var(--bg-primary, #2a2a2a);
  border: 1px solid var(--border-color, #4a4a4a);
  border-radius: 12px;
  box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
  display: flex;
  flex-direction: column;
  font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
  z-index: 10000;
  backdrop-filter: blur(10px);
}

.chatbot-window.right-panel {
  width: 350px;
  height: calc(100vh - 40px);
}

.chatbot-header {
  padding: 12px 16px;
  background: var(--accent-color, #00d4aa);
  color: var(--text-dark, #000);
  border-radius: 12px 12px 0 0;
  display: flex;
  justify-content: space-between;
  align-items: center;
  font-weight: 600;
}

.chatbot-header h3 {
  margin: 0;
  font-size: 14px;
}

.chatbot-header button {
  background: none;
  border: none;
  color: var(--text-dark, #000);
  font-size: 16px;
  cursor: pointer;
  padding: 4px 8px;
  border-radius: 4px;
  margin-left: 4px;
}

.chatbot-header button:hover {
  background: rgba(0, 0, 0, 0.1);
}

.chatbot-messages {
  flex: 1;
  padding: 16px;
  overflow-y: auto;
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.user-message, .ai-message, .system-message, .error-message {
  max-width: 85%;
  padding: 12px;
  border-radius: 12px;
  position: relative;
}

.user-message {
  align-self: flex-end;
  background: var(--accent-color, #00d4aa);
  color: var(--text-dark, #000);
}

.ai-message {
  align-self: flex-start;
  background: var(--bg-secondary, #4a4a4a);
  color: var(--text-light, #fff);
}

.system-message {
  align-self: center;
  background: var(--warning-color, #ff9500);
  color: var(--text-dark, #000);
  font-size: 12px;
}

.error-message {
  align-self: center;
  background: var(--error-color, #ff5555);
  color: var(--text-light, #fff);
}

.message-content {
  margin-bottom: 4px;
  line-height: 1.4;
}

.message-time {
  font-size: 10px;
  opacity: 0.7;
}

.chatbot-input {
  display: flex;
  padding: 12px 16px;
  border-top: 1px solid var(--border-color, #4a4a4a);
  gap: 8px;
}

.chatbot-input input {
  flex: 1;
  padding: 8px 12px;
  border: 1px solid var(--border-color, #4a4a4a);
  border-radius: 8px;
  background: var(--bg-secondary, #4a4a4a);
  color: var(--text-light, #fff);
  font-size: 14px;
}

.chatbot-input input:focus {
  outline: none;
  border-color: var(--accent-color, #00d4aa);
}

.chatbot-input button {
  padding: 8px 16px;
  background: var(--accent-color, #00d4aa);
  color: var(--text-dark, #000);
  border: none;
  border-radius: 8px;
  cursor: pointer;
  font-weight: 500;
}

.chatbot-input button:hover {
  opacity: 0.9;
}

.chatbot-status {
  padding: 8px 16px;
  border-top: 1px solid var(--border-color, #4a4a4a);
  font-size: 12px;
  background: var(--bg-tertiary, #1a1a1a);
  border-radius: 0 0 12px 12px;
}

.connection-status.success { color: var(--success-color, #00ff00); }
.connection-status.error { color: var(--error-color, #ff5555); }
.connection-status.warning { color: var(--warning-color, #ff9500); }

.chatbot-toggle {
  position: fixed;
  width: 56px;
  height: 56px;
  background: var(--accent-color, #00d4aa);
  border: none;
  border-radius: 50%;
  font-size: 24px;
  cursor: pointer;
  box-shadow: 0 4px 16px rgba(0, 212, 170, 0.3);
  z-index: 9999;
  transition: transform 0.2s ease, box-shadow 0.2s ease;
}

.chatbot-toggle:hover {
  transform: scale(1.1);
  box-shadow: 0 6px 20px rgba(0, 212, 170, 0.4);
}
`;

// Type definitions
export interface ChatbotOptions {
    position?: 'bottom-right' | 'bottom-left' | 'right-panel';
    contextData?: any;
    theme?: string;
}
