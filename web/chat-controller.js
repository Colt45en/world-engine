/**
 * Chat Controller - Command router and coordinator for World Engine Studio
 *
 * Handles chat commands, coordinates Engine and Recorder, manages workflow.
 * Acts as the central controller in the Recorder ↔ Chat ↔ Engine system.
 */

class ChatController {
  constructor(options = {}) {
    this.options = {
      autoLinkClips: true,
      transcriptCommands: true,
      commandPrefix: '/',
      ...options
    };

    this.lastClipId = null;
    this.activeRuns = new Map();
    this.commandHistory = [];

    this.bindEvents();
    this.setupUI();
  }

  bindEvents() {
    onBus(async (msg) => {
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
              this.handleTranscript(msg);
            }
            break;
          case 'eng.result':
            await this.handleEngineResult(msg);
            break;
          case 'eng.error':
            this.handleEngineError(msg);
            break;
          case 'rec.error':
            this.handleRecorderError(msg);
            break;
        }
      } catch (error) {
        Utils.log(`Chat controller error: ${error.message}`, 'error');
        this.announce(`Error: ${error.message}`, 'error');
      }
    });
  }

  setupUI() {
    const container = document.getElementById('chat-ui');
    if (container) {
      container.innerHTML = `
        <div class="chat-interface">
          <div class="chat-messages" id="chat-messages"></div>
          <div class="chat-input-area">
            <input type="text" id="chat-input" placeholder="Enter command or text to analyze..." />
            <button id="chat-send">Send</button>
          </div>
          <div class="chat-status" id="chat-status">Ready</div>
        </div>
        <style>
          .chat-interface { display: flex; flex-direction: column; height: 300px; border: 1px solid #333; border-radius: 8px; }
          .chat-messages { flex: 1; padding: 8px; overflow-y: auto; background: #1a1a1a; border-radius: 8px 8px 0 0; }
          .chat-input-area { display: flex; padding: 8px; border-top: 1px solid #333; }
          #chat-input { flex: 1; padding: 8px; border: 1px solid #444; background: #222; color: #fff; border-radius: 4px; }
          #chat-send { padding: 8px 16px; margin-left: 8px; border: 1px solid #444; background: #333; color: #fff; border-radius: 4px; cursor: pointer; }
          .chat-status { padding: 4px 8px; font-size: 12px; color: #888; }
          .chat-message { margin: 4px 0; padding: 6px; border-radius: 4px; }
          .chat-user { background: #2a4a6b; }
          .chat-system { background: #2a2a2a; color: #ccc; }
          .chat-error { background: #6b2a2a; }
          .chat-result { background: #2a6b2a; }
        </style>
      `;

      // Bind UI events
      const input = document.getElementById('chat-input');
      const sendBtn = document.getElementById('chat-send');

      const sendMessage = () => {
        const value = input.value.trim();
        if (value) {
          this.handleCommand(value);
          input.value = '';
        }
      };

      sendBtn?.addEventListener('click', sendMessage);
      input?.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') sendMessage();
      });
    }

    // Initial status
    this.announce('World Engine Studio ready. Try: /run <text>, /test <name>, /rec start, /help', 'system');
  }

  async handleCommand(line) {
    const trimmed = line.trim();
    if (!trimmed) return;

    // Add to history
    this.commandHistory.push({ command: trimmed, timestamp: Date.now() });
    this.announce(`> ${trimmed}`, 'user');

    const cmd = Utils.parseCommand(trimmed);

    try {
      switch (cmd.type) {
        case 'run':
          await this.runEngine(cmd.args);
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
    } catch (error) {
      this.announce(`Command failed: ${error.message}`, 'error');
    }
  }

  async runEngine(text) {
    if (!text.trim()) {
      this.announce('Please provide text to analyze', 'error');
      return;
    }

    this.announce(`Running analysis on: "${text}"`, 'system');
    this.updateStatus('Analyzing...');

    // Start recording marker if recording is active
    if (this.lastClipId) {
      sendBus({ type: 'rec.mark', tag: 'run-start', runId: null });
    }

    sendBus({ type: 'eng.run', text });
  }

  async loadTest(testName) {
    if (!testName.trim()) {
      this.announce('Please specify a test name', 'error');
      return;
    }

    this.announce(`Loading test: ${testName}`, 'system');
    sendBus({ type: 'eng.test', name: testName });
  }

  async handleRecordingCommand(action) {
    switch (action) {
      case 'start':
        this.announce('Starting microphone recording...', 'system');
        sendBus({ type: 'rec.start', mode: 'mic', meta: { source: 'chat' } });
        break;
      case 'screen':
        this.announce('Starting screen recording...', 'system');
        sendBus({ type: 'rec.start', mode: 'screen', meta: { source: 'chat' } });
        break;
      case 'stop':
        this.announce('Stopping recording...', 'system');
        sendBus({ type: 'rec.stop' });
        break;
      default:
        this.announce('Recording commands: start, screen, stop', 'error');
    }
  }

  async addMarker(tag) {
    if (!tag.trim()) {
      this.announce('Please specify a marker tag', 'error');
      return;
    }

    sendBus({ type: 'rec.mark', tag: tag.trim() });
    this.announce(`Marker added: ${tag}`, 'system');
  }

  handleTranscript(msg) {
    const text = msg.text.trim();
    if (!text) return;

    // Simple command detection from transcripts
    const commandKeywords = ['run', 'analyze', 'test', 'record', 'stop'];
    const lowerText = text.toLowerCase();

    let detectedCommand = null;

    for (const keyword of commandKeywords) {
      if (lowerText.startsWith(keyword)) {
        const args = text.slice(keyword.length).trim();
        detectedCommand = `/${keyword} ${args}`;
        break;
      }
    }

    if (detectedCommand) {
      this.announce(`Voice command detected: ${detectedCommand}`, 'system');
      this.handleCommand(detectedCommand);
    } else {
      // Treat as run command
      this.announce(`Voice input: "${text}"`, 'system');
      this.handleCommand(text);
    }
  }

  handleClipReady(msg) {
    this.lastClipId = msg.clipId;
    this.announce(`Recording saved: ${msg.clipId} (${msg.size ? Math.round(msg.size/1024) + 'KB' : 'unknown size'})`, 'system');
  }

  async handleEngineResult(msg) {
    const { runId, outcome, input } = msg;

    // Link with current clip if available
    if (this.lastClipId && this.options.autoLinkClips) {
      const runData = {
        runId,
        ts: Date.now(),
        input,
        outcome,
        clipId: this.lastClipId
      };

      await Store.save(`runs.${runId}`, runData);
      this.announce(`Run ${runId} linked to recording ${this.lastClipId}`, 'system');
      this.lastClipId = null; // Consume the clip ID
    }

    // Display results
    this.announce(`Analysis complete (${runId})`, 'result');

    if (outcome.items && Array.isArray(outcome.items)) {
      this.announce(`Found ${outcome.items.length} items`, 'result');

      // Show first few results
      const preview = outcome.items.slice(0, 3).map(item => {
        if (typeof item === 'object') {
          return `• ${item.lemma || item.word || item.text || 'Item'}: ${item.root || item.score || JSON.stringify(item).slice(0, 50)}`;
        }
        return `• ${item}`;
      }).join('\n');

      this.announce(preview, 'result');

      if (outcome.items.length > 3) {
        this.announce(`... and ${outcome.items.length - 3} more`, 'result');
      }
    } else if (outcome.result) {
      this.announce(`Result: ${outcome.result}`, 'result');
    }

    this.updateStatus('Ready');
  }

  handleEngineError(msg) {
    this.announce(`Engine error: ${msg.error}`, 'error');
    this.updateStatus('Error - Ready');
  }

  handleRecorderError(msg) {
    this.announce(`Recording error: ${msg.error}`, 'error');
  }

  showHelp() {
    const help = `
Available commands:
• /run <text> - Analyze text with World Engine
• /test <name> - Load a test case
• /rec start - Start microphone recording
• /rec screen - Start screen recording
• /rec stop - Stop recording
• /mark <tag> - Add timeline marker
• /status - Show system status
• /history - Show command history
• /clear - Clear messages
• /help - Show this help

You can also type text directly to analyze it.
Voice commands work if recording is active.
    `.trim();

    this.announce(help, 'system');
  }

  async showStatus() {
    // This would query actual component status
    const status = `
System Status:
• Engine: Ready
• Recorder: ${this.lastClipId ? 'Has recent clip' : 'Idle'}
• Commands: ${this.commandHistory.length} in history
• Store: Available
    `.trim();

    this.announce(status, 'system');
  }

  showHistory() {
    if (this.commandHistory.length === 0) {
      this.announce('No command history', 'system');
      return;
    }

    const recent = this.commandHistory.slice(-5).map((entry, i) =>
      `${i + 1}. ${entry.command}`
    ).join('\n');

    this.announce(`Recent commands:\n${recent}`, 'system');
  }

  clearMessages() {
    const container = document.getElementById('chat-messages');
    if (container) {
      container.innerHTML = '';
    }
    this.announce('Messages cleared', 'system');
  }

  announce(message, level = 'system') {
    const container = document.getElementById('chat-messages');
    if (!container) {
      Utils.log(message, level);
      return;
    }

    const msgEl = document.createElement('div');
    msgEl.className = `chat-message chat-${level}`;
    msgEl.textContent = message;

    container.appendChild(msgEl);
    container.scrollTop = container.scrollHeight;

    // Also log to console
    Utils.log(`[Chat:${level}] ${message}`);
  }

  updateStatus(status) {
    const statusEl = document.getElementById('chat-status');
    if (statusEl) {
      statusEl.textContent = status;
    }
  }

  // Public API methods
  getHistory() {
    return [...this.commandHistory];
  }

  async execute(command) {
    return this.handleCommand(command);
  }

  getStats() {
    return {
      commandsExecuted: this.commandHistory.length,
      lastClipId: this.lastClipId,
      activeRuns: this.activeRuns.size
    };
  }
}

// Export for use
if (typeof module !== 'undefined' && module.exports) {
  module.exports = ChatController;
} else {
  window.ChatController = ChatController;
}
