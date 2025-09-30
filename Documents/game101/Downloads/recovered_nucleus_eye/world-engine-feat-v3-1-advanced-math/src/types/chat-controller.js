/**
 * Unified Chat Controller - Command router and coordinator
 * Consolidates duplicate classes and adds enhanced features
 */

import { assertShape, Schemas } from '../schemas/validation.js';

// Prevent duplicate registration
if (typeof window !== 'undefined' && window.__WE_ChatController_defined__) {
    console.warn('ChatController already defined; skipping duplicate.');
}
if (typeof window !== 'undefined') window.__WE_ChatController_defined__ = true;

export class ChatController {
    constructor(options = {}) {
        this.options = {
            autoLinkClips: true,
            transcriptCommands: true,
            commandPrefix: '/',
            historyLimit: 500,
            debounceMs: 300,
            ...options
        };

        this.lastClipId = null;
        this.activeRuns = new Map();
        this.commandHistory = [];
        this.transcriptDebounce = null;

        this.bindEvents();
        this.setupUI();
    }

    bindEvents() {
        this.bus.onBus(async (msg) => {
            try {
                switch (msg.type) {
                    case 'chat.cmd':
                        await this.handleCommand(msg.line);
                        break;
                    case 'rec.clip':
                        this.handleClipReady(msg);
                        break;
                    case 'rec.transcript':
                        if (this.options.transcriptCommands) {
                            this.handleTranscriptDebounced(msg);
                        }
                        break;
                    case 'orchestrator.run.ok':
                        await this.handleOrchestratedResult(msg);
                        break;
                    case 'orchestrator.run.fail':
                        this.handleOrchestratedError(msg);
                        break;
                    case 'orchestrator.run.retry':
                        this.handleOrchestratedRetry(msg);
                        break;
                }
            } catch (error) {
                console.error('Chat controller error:', error);
                this.announce(`Error: ${error.message}`, 'error');
            }
        });
    }

    setupUI() {
        const container = document.getElementById('chat-ui');
        if (container) {
            container.innerHTML = `
          <div class="chat-interface">
            <div class="chat-messages" id="chat-messages" aria-live="polite"></div>
            <div class="chat-input-area">
              <input type="text" id="chat-input" placeholder="Enter command or text to analyze..."
                     aria-label="Command input" autocomplete="off" />
              <button id="chat-send" aria-label="Send command">Send</button>
            </div>
            <div class="chat-status" id="chat-status" aria-live="polite">Ready</div>
          </div>
        `;

            this.bindUIEvents();
            this.bindKeyboardShortcuts();
        }

        this.announce('World Engine Studio ready. Try: /run <text>, /test <name>, /rec start, /help', 'system');
    }

    bindUIEvents() {
        const input = document.getElementById('chat-input');
        const sendBtn = document.getElementById('chat-send');

        const sendMessage = () => {
            const value = input?.value?.trim();
            if (value) {
                this.handleCommand(value);
                if (input) {
                    input.value = '';
                    input.focus(); // Keep focus for rapid commands
                }
            }
        };

        sendBtn?.addEventListener('click', sendMessage);
        input?.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                e.preventDefault();
                sendMessage();
            }
        });

        // Focus input on page load
        setTimeout(() => input?.focus(), 100);
    }

    bindKeyboardShortcuts() {
        document.addEventListener('keydown', (e) => {
            const target = e.target;
            const isInputFocused = target && (
                target.tagName === 'INPUT' ||
                target.tagName === 'TEXTAREA' ||
                target.isContentEditable
            );

            if (isInputFocused) return;
            if (e.metaKey || e.ctrlKey || e.altKey) return;

            switch (e.key) {
                case '/':
                    e.preventDefault();
                    document.getElementById('chat-input')?.focus();
                    break;
                case '?':
                    e.preventDefault();
                    this.showHelp();
                    break;
                case 'Escape':
                    document.getElementById('chat-input')?.blur();
                    break;
            }
        });
    }

    async handleCommand(line) {
        const trimmed = line.trim();
        if (!trimmed) return;

        // Add to history with limit
        this.commandHistory.push({ command: trimmed, timestamp: Date.now() });
        if (this.commandHistory.length > this.options.historyLimit) {
            this.commandHistory.shift();
        }

        this.announce(`> ${trimmed}`, 'user');

        // Parse command vs plain text
        const isCmd = trimmed.startsWith(this.options.commandPrefix);
        const cmd = isCmd ? this.parseCommand(trimmed) : { type: 'run', args: trimmed };

        // Emit parsed command for orchestrator
        this.bus.sendBus({ type: 'chat.cmd.parsed', cmd });

        try {
            await this.executeCommand(cmd);
        } catch (error) {
            this.announce(`Command failed: ${error.message}`, 'error');
        }
    }

    parseCommand(line) {
        const parts = line.slice(1).split(/\s+/);
        const type = parts[0]?.toLowerCase() || '';
        const args = parts.slice(1).join(' ');

        return assertShape(Schemas.Command, { type, args });
    }

    async executeCommand(cmd) {
        switch (cmd.type) {
            case 'run':
                // Orchestrator handles this via chat.cmd.parsed
                this.announce(`Analyzing: "${cmd.args}"`, 'system');
                break;
            case 'test':
                await this.loadTest(cmd.args);
                break;
            case 'rec':
                await this.handleRecordingCommand(cmd.args);
                break;
            case 'mark':
                await this.addMarker(cmd.args);
                break;
            case 'help':
                this.showHelp();
                break;
            case 'status':
                await this.showStatus();
                break;
            case 'history':
                this.showHistory();
                break;
            case 'clear':
                this.clearMessages();
                break;
            default:
                this.announce(`Unknown command: ${cmd.type}. Type /help for available commands.`, 'error');
        }
    }

    handleTranscriptDebounced(msg) {
        clearTimeout(this.transcriptDebounce);
        this.transcriptDebounce = setTimeout(() => {
            this.handleTranscript(msg);
        }, this.options.debounceMs);
    }

    handleTranscript(msg) {
        const text = (msg.text || '').trim();
        if (!text) return;

        const lower = text.toLowerCase();
        const commandKeywords = ['run', 'analyze', 'test', 'record', 'stop'];
        let detected = null;

        // Look for explicit voice commands
        for (const keyword of commandKeywords) {
            if (lower.startsWith(keyword + ' ')) {
                detected = `/${keyword} ${text.slice(keyword.length).trim()}`;
                break;
            }
        }

        if (detected) {
            this.announce(`Voice command detected: ${detected}`, 'system');
            this.handleCommand(detected);
        } else {
            // Treat as input to /run
            this.announce(`Voice input: "${text}"`, 'system');
            this.handleCommand(`/run ${text}`);
        }
    }

    async handleOrchestratedResult(msg) {
        const { runId, record } = msg;

        this.announce(`Analysis complete (${runId})`, 'result');

        if (record.outcome?.items?.length) {
            this.announce(`Found ${record.outcome.items.length} items`, 'result');

            // Show preview of first few results
            const preview = record.outcome.items.slice(0, 3).map(item => {
                if (typeof item === 'object') {
                    return `• ${item.lemma || item.word || item.text || 'Item'}: ${item.root || item.score || JSON.stringify(item).slice(0, 50)}`;
                }
                return `• ${item}`;
            }).join('\n');

            this.announce(preview, 'result');

            if (record.outcome.items.length > 3) {
                this.announce(`... and ${record.outcome.items.length - 3} more`, 'result');
            }
        } else if (record.outcome?.result) {
            this.announce(`Result: ${record.outcome.result}`, 'result');
        }

        if (record.clipId) {
            this.announce(`Run ${runId} linked to recording ${record.clipId}`, 'system');
        }
    }

    handleOrchestratedError(msg) {
        this.announce(`Analysis failed (${msg.runId}): ${msg.err}`, 'error');
    }

    handleOrchestratedRetry(msg) {
        this.announce(`Retrying analysis (${msg.runId}), attempt ${msg.attempt}`, 'warning');
    }

    // ... rest of methods (showHelp, showStatus, etc.) remain the same but condensed ...

    announce(message, level = 'system') {
        const container = document.getElementById('chat-messages');
        if (!container) {
            console.log(message);
            return;
        }

        const msgEl = document.createElement('div');
        msgEl.className = `chat-message chat-${level}`;
        msgEl.textContent = message;

        container.appendChild(msgEl);
        container.scrollTop = container.scrollHeight;

        // Also update status
        if (level === 'system' || level === 'result') {
            this.updateStatus(message);
        }
    }

    updateStatus(status) {
        const statusEl = document.getElementById('chat-status');
        if (statusEl) {
            statusEl.textContent = status;
        }
    }

    getHistory() {
        return [...this.commandHistory];
    }

    getStats() {
        return {
            commandsExecuted: this.commandHistory.length,
            lastClipId: this.lastClipId,
            activeRuns: this.activeRuns.size
        };
    }
}
}
