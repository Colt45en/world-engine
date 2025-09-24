/**
 * AI Bot Controller - Intelligent assistant integrated with World Engine Studio
 *
 * Features:
 * - Natural language chat interface
 * - Self-learning from interactions
 * - Integration with World Engine semantic analysis
 * - Persistent knowledge system
 * - Feedback learning for continuous improvement
 */

class AIBotController {
  constructor(options = {}) {
    this.options = {
      apiEndpoint: '/api/ai-bot',
      autoLearn: true,
      showConfidence: true,
      maxChatHistory: 100,
      feedbackEnabled: true,
      ...options
    };

    this.chatHistory = [];
    this.currentInteractionId = null;
    this.isTyping = false;
    this.knowledge_stats = null;

    this.setupUI();
    this.bindEvents();
    this.loadChatHistory();
    this.fetchStats();
  }

  setupUI() {
    const container = document.getElementById('ai-bot-ui') || this.createContainer();

    container.innerHTML = `
      <div class="ai-bot-interface">
        <div class="ai-bot-header">
          <h3 class="ai-bot-title">🤖 AI Assistant</h3>
          <div class="ai-bot-stats" id="ai-bot-stats">
            <span class="stat-item">Knowledge: <span id="knowledge-count">Loading...</span></span>
            <span class="stat-item">Success: <span id="success-rate">--</span>%</span>
          </div>
        </div>

        <div class="ai-bot-chat" id="ai-bot-chat">
          <div class="chat-welcome">
            <div class="welcome-message">
              👋 Hello! I'm your AI assistant for the World Engine Studio.
              I can help you with commands, explain how things work, and learn from our conversations.
              <br><br>
              Try asking me: "How do I analyze text?" or "What commands are available?"
            </div>
          </div>
        </div>

        <div class="ai-bot-input-area">
          <input type="text" id="ai-bot-input" placeholder="Ask me anything about World Engine..." />
          <button id="ai-bot-send" class="ai-bot-send-btn">Send</button>
          <button id="ai-bot-clear" class="ai-bot-clear-btn" title="Clear chat">🗑️</button>
        </div>

        <div class="ai-bot-typing" id="ai-bot-typing" style="display: none;">
          <span class="typing-indicator">AI is thinking</span>
          <div class="typing-dots">
            <span></span><span></span><span></span>
          </div>
        </div>
      </div>

      <style>
        .ai-bot-interface {
          display: flex;
          flex-direction: column;
          height: 100%;
          background: var(--panel);
          border-radius: 8px;
          overflow: hidden;
        }

        .ai-bot-header {
          padding: 12px 16px;
          border-bottom: 1px solid var(--border);
          background: var(--bg);
        }

        .ai-bot-title {
          margin: 0 0 8px 0;
          color: var(--accent);
          font-size: 16px;
        }

        .ai-bot-stats {
          display: flex;
          gap: 16px;
          font-size: 12px;
          color: var(--mut);
        }

        .ai-bot-chat {
          flex: 1;
          padding: 12px;
          overflow-y: auto;
          max-height: 400px;
        }

        .chat-welcome {
          text-align: center;
          margin: 20px 0;
        }

        .welcome-message {
          background: rgba(84, 240, 184, 0.1);
          border: 1px solid rgba(84, 240, 184, 0.3);
          border-radius: 8px;
          padding: 16px;
          font-size: 14px;
          line-height: 1.5;
          color: var(--fg);
        }

        .chat-message {
          margin: 8px 0;
          padding: 8px 12px;
          border-radius: 8px;
          max-width: 90%;
          word-wrap: break-word;
          animation: messageSlideIn 0.3s ease-out;
        }

        .chat-message.user {
          background: rgba(124, 220, 255, 0.2);
          border: 1px solid rgba(124, 220, 255, 0.4);
          margin-left: auto;
          text-align: right;
        }

        .chat-message.ai {
          background: rgba(84, 240, 184, 0.1);
          border: 1px solid rgba(84, 240, 184, 0.3);
        }

        .chat-message.system {
          background: rgba(255, 255, 255, 0.05);
          border: 1px solid var(--border);
          font-style: italic;
          text-align: center;
          color: var(--mut);
        }

        .message-meta {
          font-size: 11px;
          color: var(--mut);
          margin-top: 4px;
          display: flex;
          justify-content: space-between;
          align-items: center;
        }

        .confidence-badge {
          background: rgba(84, 240, 184, 0.2);
          padding: 2px 6px;
          border-radius: 12px;
          font-size: 10px;
        }

        .feedback-buttons {
          display: flex;
          gap: 4px;
          margin-top: 6px;
        }

        .feedback-btn {
          background: transparent;
          border: 1px solid var(--border);
          color: var(--mut);
          padding: 2px 8px;
          border-radius: 4px;
          cursor: pointer;
          font-size: 10px;
        }

        .feedback-btn:hover {
          background: var(--border);
        }

        .feedback-btn.positive { border-color: var(--success); color: var(--success); }
        .feedback-btn.negative { border-color: var(--error); color: var(--error); }

        .ai-bot-input-area {
          display: flex;
          padding: 12px;
          border-top: 1px solid var(--border);
          gap: 8px;
        }

        #ai-bot-input {
          flex: 1;
          background: var(--bg);
          border: 1px solid var(--border);
          color: var(--fg);
          padding: 8px 12px;
          border-radius: 6px;
          font-size: 14px;
        }

        #ai-bot-input:focus {
          outline: none;
          border-color: var(--accent);
        }

        .ai-bot-send-btn, .ai-bot-clear-btn {
          background: var(--accent);
          color: var(--bg);
          border: none;
          padding: 8px 16px;
          border-radius: 6px;
          cursor: pointer;
          font-weight: 600;
        }

        .ai-bot-clear-btn {
          background: var(--border);
          color: var(--mut);
          padding: 8px 12px;
        }

        .ai-bot-send-btn:hover {
          background: rgba(124, 220, 255, 0.8);
        }

        .ai-bot-clear-btn:hover {
          background: var(--error);
          color: white;
        }

        .ai-bot-typing {
          padding: 8px 16px;
          border-top: 1px solid var(--border);
          display: flex;
          align-items: center;
          gap: 8px;
          font-size: 12px;
          color: var(--mut);
        }

        .typing-dots {
          display: flex;
          gap: 2px;
        }

        .typing-dots span {
          width: 4px;
          height: 4px;
          background: var(--accent);
          border-radius: 50%;
          animation: typingDots 1.4s ease-in-out infinite both;
        }

        .typing-dots span:nth-child(1) { animation-delay: -0.32s; }
        .typing-dots span:nth-child(2) { animation-delay: -0.16s; }

        @keyframes typingDots {
          0%, 80%, 100% { transform: scale(0.8); opacity: 0.5; }
          40% { transform: scale(1); opacity: 1; }
        }

        @keyframes messageSlideIn {
          from { opacity: 0; transform: translateY(10px); }
          to { opacity: 1; transform: translateY(0); }
        }
      </style>
    `;

    // Make interface globally available for debugging
    window.AIBotController = this;
  }

  createContainer() {
    // Create container if it doesn't exist (for standalone use)
    const container = document.createElement('div');
    container.id = 'ai-bot-ui';
    container.style.cssText = 'height: 500px; width: 100%; border: 1px solid #333; border-radius: 8px;';
    document.body.appendChild(container);
    return container;
  }

  bindEvents() {
    const input = document.getElementById('ai-bot-input');
    const sendBtn = document.getElementById('ai-bot-send');
    const clearBtn = document.getElementById('ai-bot-clear');

    const sendMessage = () => {
      const message = input.value.trim();
      if (message && !this.isTyping) {
        this.sendMessage(message);
        input.value = '';
      }
    };

    sendBtn?.addEventListener('click', sendMessage);
    clearBtn?.addEventListener('click', () => this.clearChat());

    input?.addEventListener('keypress', (e) => {
      if (e.key === 'Enter') sendMessage();
    });

    // Listen for studio bridge messages
    onBus((msg) => {
      switch (msg.type) {
        case 'eng.result':
          this.handleWorldEngineResult(msg);
          break;
        case 'chat.cmd':
          this.handleChatCommand(msg);
          break;
      }
    });
  }

  async sendMessage(message) {
    this.addMessage(message, 'user');
    this.showTyping(true);

    try {
      const response = await fetch(`${this.options.apiEndpoint}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message: message,
          context: this.buildContext()
        })
      });

      if (!response.ok) {
        throw new Error(`API error: ${response.status}`);
      }

      const data = await response.json();
      this.currentInteractionId = data.interaction_id;

      this.addMessage(data.response, 'ai', {
        confidence: data.confidence,
        sources: data.knowledge_sources,
        interactionId: data.interaction_id
      });

      // Update stats after interaction
      setTimeout(() => this.fetchStats(), 1000);

    } catch (error) {
      this.addMessage(`Sorry, I encountered an error: ${error.message}`, 'ai', { isError: true });
      Utils.log(`AI Bot error: ${error.message}`, 'error');
    } finally {
      this.showTyping(false);
    }
  }

  addMessage(content, type, meta = {}) {
    const chatContainer = document.getElementById('ai-bot-chat');
    const messageDiv = document.createElement('div');
    messageDiv.className = `chat-message ${type}`;

    let metaHtml = '';
    if (type === 'ai' && this.options.showConfidence && meta.confidence) {
      metaHtml += `<div class="message-meta">`;
      metaHtml += `<span class="confidence-badge">Confidence: ${Math.round(meta.confidence * 100)}%</span>`;

      if (this.options.feedbackEnabled && meta.interactionId) {
        metaHtml += `<div class="feedback-buttons">
          <button class="feedback-btn positive" onclick="window.AIBotController.provideFeedback('${meta.interactionId}', true)">👍</button>
          <button class="feedback-btn negative" onclick="window.AIBotController.provideFeedback('${meta.interactionId}', false)">👎</button>
        </div>`;
      }
      metaHtml += `</div>`;
    }

    messageDiv.innerHTML = `
      <div class="message-content">${this.formatMessage(content)}</div>
      ${metaHtml}
    `;

    // Remove welcome message if present
    const welcome = chatContainer.querySelector('.chat-welcome');
    if (welcome) welcome.remove();

    chatContainer.appendChild(messageDiv);
    chatContainer.scrollTop = chatContainer.scrollHeight;

    // Store in history
    this.chatHistory.push({ content, type, timestamp: Date.now(), meta });
    this.saveChatHistory();
  }

  formatMessage(content) {
    // Basic markdown-like formatting
    return content
      .replace(/`([^`]+)`/g, '<code style="background: rgba(255,255,255,0.1); padding: 2px 4px; border-radius: 3px;">$1</code>')
      .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
      .replace(/\*([^*]+)\*/g, '<em>$1</em>');
  }

  async provideFeedback(interactionId, isPositive) {
    try {
      await fetch(`${this.options.apiEndpoint}/feedback`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          interaction_id: interactionId,
          feedback: isPositive ? 'positive' : 'negative',
          success: isPositive
        })
      });

      this.addMessage(`Thank you for the feedback! I'm ${isPositive ? 'glad I could help' : 'sorry I missed the mark. I\'ll learn from this'}.`, 'system');

      // Refresh stats after feedback
      setTimeout(() => this.fetchStats(), 500);

    } catch (error) {
      Utils.log(`Feedback error: ${error.message}`, 'error');
    }
  }

  async fetchStats() {
    try {
      const response = await fetch(`${this.options.apiEndpoint}/stats`);
      const stats = await response.json();

      this.knowledge_stats = stats;
      this.updateStatsDisplay(stats);

    } catch (error) {
      Utils.log(`Stats fetch error: ${error.message}`, 'warn');
    }
  }

  updateStatsDisplay(stats) {
    const knowledgeCount = document.getElementById('knowledge-count');
    const successRate = document.getElementById('success-rate');

    if (knowledgeCount) knowledgeCount.textContent = stats.knowledge_entries || 0;
    if (successRate) successRate.textContent = Math.round((stats.success_rate || 0) * 100);
  }

  buildContext() {
    return {
      recent_messages: this.chatHistory.slice(-5),
      world_engine_connected: true,
      session_info: {
        timestamp: Date.now(),
        message_count: this.chatHistory.length
      }
    };
  }

  showTyping(show) {
    this.isTyping = show;
    const typingDiv = document.getElementById('ai-bot-typing');
    if (typingDiv) {
      typingDiv.style.display = show ? 'flex' : 'none';
    }
  }

  clearChat() {
    if (confirm('Clear chat history? This cannot be undone.')) {
      this.chatHistory = [];
      const chatContainer = document.getElementById('ai-bot-chat');
      if (chatContainer) {
        chatContainer.innerHTML = `
          <div class="chat-welcome">
            <div class="welcome-message">
              Chat cleared. How can I help you with World Engine Studio?
            </div>
          </div>
        `;
      }
      this.saveChatHistory();
    }
  }

  handleWorldEngineResult(msg) {
    if (msg.outcome && this.options.autoLearn) {
      // Learn from World Engine results
      this.addMessage(`I noticed the World Engine analyzed: "${msg.text}" with results. This helps me understand semantic patterns better!`, 'system');
    }
  }

  handleChatCommand(msg) {
    // Handle commands sent through chat controller
    if (msg.line && msg.line.startsWith('/ai ')) {
      const query = msg.line.substring(4);
      this.sendMessage(query);
    }
  }

  saveChatHistory() {
    try {
      localStorage.setItem('ai_bot_chat_history', JSON.stringify(this.chatHistory.slice(-this.options.maxChatHistory)));
    } catch (error) {
      Utils.log('Failed to save chat history', 'warn');
    }
  }

  loadChatHistory() {
    try {
      const saved = localStorage.getItem('ai_bot_chat_history');
      if (saved) {
        this.chatHistory = JSON.parse(saved);
        // Optionally restore recent messages to UI
        // this.restoreChatHistory();
      }
    } catch (error) {
      Utils.log('Failed to load chat history', 'warn');
    }
  }

  getStats() {
    return {
      message_count: this.chatHistory.length,
      current_interaction: this.currentInteractionId,
      knowledge_stats: this.knowledge_stats,
      is_typing: this.isTyping
    };
  }
}

// Integration with Studio Bridge
if (typeof window !== 'undefined' && window.StudioBridge) {
  window.StudioBridge.AIBotController = AIBotController;
}

// Export for module systems
if (typeof module !== 'undefined' && module.exports) {
  module.exports = AIBotController;
}
