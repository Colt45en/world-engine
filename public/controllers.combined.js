/* controllers.combined.js
 * Unified Controllers: RecorderController, AIBotController, ChatController, EngineController
 * + Orchestrator + Safe shims + StudioBridge integration
 * Version: 1.0.0
 */
(function (global, factory) {
  if (typeof module === 'object' && typeof module.exports === 'object') {
    module.exports = factory(global);
  } else {
    global.WorldEngineControllers = factory(global);
  }
})(typeof window !== 'undefined' ? window : globalThis, function (root) {
  'use strict';

  // ---------------------------
  // Safe shims (no-ops if absent)
  // ---------------------------
  const SB = root.StudioBridge || {};
  const Bus = (function () {
    const listeners = [];
    const send = (msg) => {
      try { (SB.sendBus ? SB.sendBus : (m => listeners.forEach(fn => fn(m))))(msg); } catch {}
      // also emit to local listeners if StudioBridge is not present
      if (!SB.sendBus) listeners.forEach(fn => { try { fn(msg); } catch {} });
    };
    const on = (fn) => {
      if (SB.onBus) return SB.onBus(fn);
      listeners.push(fn);
      return () => {
        const i = listeners.indexOf(fn);
        if (i >= 0) listeners.splice(i, 1);
      };
    };
    return { send, on };
  })();

  const Utils = root.Utils || SB.Utils || {
    log: (msg, level = 'info') => console[(level === 'error' ? 'error' :
      level === 'warn' ? 'warn' : 'log')](msg),
    parseCommand: (line, prefix = '/') => {
      const l = line.trim();
      if (!l.startsWith(prefix)) return { type: 'run', args: l };
      const parts = l.slice(prefix.length).split(/\s+/);
      return { type: (parts[0] || '').toLowerCase(), args: parts.slice(1).join(' ') };
    },
    generateId: () => Math.random().toString(36).slice(2)
  };

  const Store = root.Store || {
    async save(k, v) {
      try { localStorage.setItem(k, JSON.stringify(v)); } catch {}
      return true;
    },
    async load(k, d=null) {
      try { const v = localStorage.getItem(k); return v ? JSON.parse(v) : d; } catch { return d; }
    }
  };

  const setupEngineTransport = root.setupEngineTransport || function (iframe) {
    const isSameOrigin = () => {
      try { void iframe.contentWindow.document; return true; } catch { return false; }
    };
    return {
      isSameOrigin,
      withEngine(fn) {
        if (!iframe || !isSameOrigin()) throw new Error('Engine iframe not same-origin or missing');
        fn(iframe.contentWindow.document);
      }
    };
  };

  // Alias for concise usage
  const sendBus = Bus.send;
  const onBus = Bus.on;

  // ------------------------------------------------------
  // RecorderController (unchanged API; compact minor edits)
  // ------------------------------------------------------
  class RecorderController {
    constructor(options = {}) {
      this.options = { transcription: false, autoMarkRuns: true, chunkSize: 200, ...options };
      this.mediaRecorder = null;
      this.mediaStream = null;
      this.chunks = [];
      this.currentClipId = null;
      this.isRecording = false;
      this.transcriptionService = null;
      this._timer = null;
      this.bindEvents();
      this.setupUI();
    }
    bindEvents() {
      onBus(async (msg) => {
        try {
          switch (msg.type) {
          case 'rec.start': await this.startRecording(msg.mode, msg.meta); break;
          case 'rec.stop': await this.stopRecording(); break;
          case 'rec.mark': await this.addMarker(msg.tag, msg.runId); break;
          case 'eng.result':
            if (this.options.autoMarkRuns) { await this.addMarker('engine-result', msg.runId); }
            break;
          }
        } catch (error) {
          Utils.log(`Recorder error: ${error.message}`, 'error');
          sendBus({ type: 'rec.error', error: error.message, originalMessage: msg });
        }
      });
    }
    setupUI() {
      const container = document.getElementById('recorder-ui');
      if (!container) return;
      container.innerHTML = `
        <div class="recorder-controls">
          <button id="rec-mic" class="rec-btn">🎤 Start Mic</button>
          <button id="rec-screen" class="rec-btn">🖥️ Start Screen</button>
          <button id="rec-stop" class="rec-btn" disabled>⏹️ Stop</button>
          <div class="rec-status">
            <span id="rec-indicator" class="rec-dot"></span>
            <span id="rec-time">00:00</span>
          </div>
        </div>
        <div class="rec-clips" id="rec-clips"></div>
        <style>
          .recorder-controls { display:flex; gap:8px; align-items:center; padding:8px; }
          .rec-btn { padding:6px 12px; border:1px solid #333; background:#222; color:#fff; border-radius:4px; cursor:pointer; }
          .rec-btn:disabled { opacity:.5; cursor:not-allowed; }
          .rec-status { margin-left:auto; display:flex; align-items:center; gap:8px; }
          .rec-dot { width:8px; height:8px; border-radius:50%; background:#666; }
          .rec-dot.recording { background:#f44; animation:pulse 1s infinite; }
          @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.3} }
          .rec-clips { padding:8px; max-height:200px; overflow-y:auto; }
          .rec-clip { display:flex; align-items:center; gap:8px; padding:4px; border:1px solid #333; margin:2px 0; border-radius:4px; }
        </style>`;
      document.getElementById('rec-mic')?.addEventListener('click', () => sendBus({ type: 'rec.start', mode: 'mic' }));
      document.getElementById('rec-screen')?.addEventListener('click', () => sendBus({ type: 'rec.start', mode: 'screen' }));
      document.getElementById('rec-stop')?.addEventListener('click', () => sendBus({ type: 'rec.stop' }));
    }
    _tickStart() {
      const el = document.getElementById('rec-time'); const t0 = Date.now();
      this._timer = setInterval(() => {
        const s = Math.floor((Date.now() - t0) / 1000);
        const mm = String(Math.floor(s / 60)).padStart(2, '0');
        const ss = String(s % 60).padStart(2, '0');
        if (el) el.textContent = `${mm}:${ss}`;
      }, 1000);
    }
    _tickStop() { if (this._timer) clearInterval(this._timer); this._timer = null; }
    async startRecording(mode = 'mic', meta = {}) {
      if (this.isRecording) { Utils.log('Already recording', 'warn'); return; }
      try {
        this.currentClipId = Utils.generateId();
        this.chunks = [];
        switch (mode) {
        case 'screen':
          this.mediaStream = await navigator.mediaDevices.getDisplayMedia({ video: true, audio: true }); break;
        case 'both':
          this.mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true, video: true }); break;
        case 'mic':
        default:
          this.mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true, video: false });
        }
        this.mediaRecorder = new MediaRecorder(this.mediaStream, { mimeType: 'audio/webm' });
        this.mediaRecorder.ondataavailable = (e) => {
          if (e.data.size > 0) {
            this.chunks.push(e.data);
            if (this.options.transcription && this.transcriptionService) this.handleChunkTranscription(e.data);
          }
        };
        this.mediaRecorder.onstop = async () => { await this.finalizeRecording(meta); };
        this.mediaRecorder.onerror = (e) => { Utils.log(`MediaRecorder error: ${e.error}`, 'error'); this.cleanup(); };
        this.mediaRecorder.start(this.options.chunkSize);
        this.isRecording = true;
        this.updateUI('recording'); this._tickStart();
        Utils.log(`Recording started: ${mode} (${this.currentClipId})`);
        sendBus({ type: 'rec.ready', clipId: this.currentClipId, mode, meta });
      } catch (error) {
        Utils.log(`Failed to start recording: ${error.message}`, 'error');
        this.cleanup(); throw error;
      }
    }
    async stopRecording() {
      if (!this.isRecording || !this.mediaRecorder) { Utils.log('Not currently recording', 'warn'); return; }
      this.mediaRecorder.stop(); this.cleanup();
    }
    async finalizeRecording(meta) {
      if (this.chunks.length === 0) { Utils.log('No recording data to process', 'warn'); return; }
      try {
        const blob = new Blob(this.chunks, { type: 'audio/webm' });
        const url = URL.createObjectURL(blob);
        const clipData = { clipId: this.currentClipId, url, blob, duration: 0, size: blob.size, timestamp: Date.now(), meta: meta || {} };
        await Store.save(`clips.${this.currentClipId}`, clipData);
        this.addClipToUI(clipData);
        sendBus({ type: 'rec.clip', clipId: this.currentClipId, url, meta: clipData.meta, size: blob.size });
        Utils.log(`Recording saved: ${this.currentClipId} (${blob.size} bytes)`);
      } catch (error) {
        Utils.log(`Failed to finalize recording: ${error.message}`, 'error'); throw error;
      } finally {
        this.isRecording = false; this.currentClipId = null; this.chunks = []; this.updateUI('stopped'); this._tickStop();
        const el = document.getElementById('rec-time'); if (el) el.textContent = '00:00';
      }
    }
    async addMarker(tag, runId = null) {
      const marker = { id: Utils.generateId(), tag, runId, timestamp: Date.now(), clipId: this.currentClipId };
      await Store.save(`marks.${marker.id}`, marker);
      Utils.log(`Marker added: ${tag} ${runId ? `(run:${runId})` : ''}`);
      sendBus({ type: 'rec.marker', marker });
      return marker;
    }
    cleanup() {
      if (this.mediaStream) { this.mediaStream.getTracks().forEach(t => t.stop()); this.mediaStream = null; }
      this.mediaRecorder = null; this.isRecording = false; this.updateUI('stopped'); this._tickStop();
    }
    updateUI(state) {
      const indicator = document.getElementById('rec-indicator');
      const startBtns = document.querySelectorAll('#rec-mic, #rec-screen');
      const stopBtn = document.getElementById('rec-stop');
      if (indicator) indicator.className = state === 'recording' ? 'rec-dot recording' : 'rec-dot';
      startBtns.forEach(btn => { btn.disabled = state === 'recording'; });
      if (stopBtn) stopBtn.disabled = state !== 'recording';
    }
    addClipToUI(clipData) {
      const container = document.getElementById('rec-clips'); if (!container) return;
      const clipEl = document.createElement('div'); clipEl.className = 'rec-clip';
      clipEl.innerHTML = `
        <audio controls src="${clipData.url}"></audio>
        <span>${new Date(clipData.timestamp).toLocaleTimeString()}</span>
        <span>${(clipData.size / 1024).toFixed(1)}KB</span>
        <button onclick="navigator.clipboard.writeText('${clipData.clipId}')">📋</button>`;
      container.insertBefore(clipEl, container.firstChild);
    }
    handleChunkTranscription(chunk) {
      if (!this.transcriptionService) return;
      this.transcriptionService.transcribe(chunk)
        .then(text => { if (text.trim()) sendBus({ type: 'rec.transcript', clipId: this.currentClipId, text: text.trim(), ts: Date.now() }); })
        .catch(err => Utils.log(`Transcription error: ${err.message}`, 'warn'));
    }
    getStatus() { return { isRecording: this.isRecording, currentClipId: this.currentClipId, hasStream: !!this.mediaStream }; }
    async getClips() { return []; }
  }

  // ------------------------------------------------------
  // AIBotController (unchanged API; tiny safety tweaks)
  // ------------------------------------------------------
  class AIBotController {
    constructor(options = {}) {
      this.options = { apiEndpoint: '/api/ai-bot', autoLearn: true, showConfidence: true, maxChatHistory: 100, feedbackEnabled: true, ...options };
      this.chatHistory = [];
      this.currentInteractionId = null;
      this.isTyping = false;
      this.knowledge_stats = null;
      this.setupUI(); this.bindEvents(); this.loadChatHistory();
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
                Try asking me: "How do I analyze text?" or "What commands are available?"
              </div>
            </div>
          </div>
          <div class="ai-bot-input-area">
            <input type="text" id="ai-bot-input" placeholder="Ask me anything about World Engine..." />
            <button id="ai-bot-send" class="ai-bot-send-btn">Send</button>
            <button id="ai-bot-clear" class="ai-bot-clear-btn" title="Clear chat">🗑️</button>
          </div>
          <div class="ai-bot-typing" id="ai-bot-typing" style="display:none;">
            <span class="typing-indicator">AI is thinking</span>
            <div class="typing-dots"><span></span><span></span><span></span></div>
          </div>
        </div>`;
      root.AIBotController = this; // handy for feedback buttons
    }
    createContainer() {
      const container = document.createElement('div');
      container.id = 'ai-bot-ui';
      container.style.cssText = 'height: 500px; width: 100%; border: 1px solid #333; border-radius: 8px;';
      document.body.appendChild(container); return container;
    }
    bindEvents() {
      const input = document.getElementById('ai-bot-input');
      const sendBtn = document.getElementById('ai-bot-send');
      const clearBtn = document.getElementById('ai-bot-clear');
      const sendMessage = () => {
        const message = (input?.value || '').trim();
        if (message && !this.isTyping) { this.sendMessage(message); if (input) input.value = ''; }
      };
      sendBtn?.addEventListener('click', sendMessage);
      clearBtn?.addEventListener('click', () => this.clearChat());
      input?.addEventListener('keypress', (e) => { if (e.key === 'Enter') sendMessage(); });
      onBus((msg) => {
        switch (msg.type) {
        case 'eng.result': this.handleWorldEngineResult(msg); break;
        case 'chat.cmd': this.handleChatCommand(msg); break;
        }
      });
    }
    async sendMessage(message) {
      this.addMessage(message, 'user'); this.showTyping(true);
      try {
        const response = await fetch(`${this.options.apiEndpoint}/chat`, {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ message, context: this.buildContext() })
        });
        if (!response.ok) throw new Error(`API error: ${response.status}`);
        const data = await response.json();
        this.currentInteractionId = data.interaction_id;
        this.addMessage(data.response, 'ai', { confidence: data.confidence, sources: data.knowledge_sources, interactionId: data.interaction_id });
        setTimeout(() => this.fetchStats(), 1000);
      } catch (error) {
        this.addMessage(`Sorry, I encountered an error: ${error.message || error}`, 'ai', { isError: true });
        Utils.log(`AI Bot error: ${error.message || error}`, 'error');
      } finally { this.showTyping(false); }
    }
    addMessage(content, type, meta = {}) {
      const chatContainer = document.getElementById('ai-bot-chat');
      const messageDiv = document.createElement('div'); messageDiv.className = `chat-message ${type}`;
      let metaHtml = '';
      if (type === 'ai' && this.options.showConfidence && meta.confidence) {
        metaHtml += `<div class="message-meta">
          <span class="confidence-badge">Confidence: ${Math.round(meta.confidence * 100)}%</span>${
  this.options.feedbackEnabled && meta.interactionId ? `
            <div class="feedback-buttons">
              <button class="feedback-btn positive" onclick="window.AIBotController.provideFeedback('${meta.interactionId}', true)">👍</button>
              <button class="feedback-btn negative" onclick="window.AIBotController.provideFeedback('${meta.interactionId}', false)">👎</button>
            </div>` : ''
}</div>`;
      }
      const fmt = (s) => s
        .replace(/`([^`]+)`/g, '<code style="background: rgba(255,255,255,0.1); padding: 2px 4px; border-radius: 3px;">$1</code>')
        .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
        .replace(/\*([^*]+)\*/g, '<em>$1</em>');
      messageDiv.innerHTML = `<div class="message-content">${fmt(content)}</div>${metaHtml}`;
      const welcome = chatContainer?.querySelector?.('.chat-welcome'); if (welcome) welcome.remove();
      chatContainer?.appendChild(messageDiv);
      if (chatContainer) chatContainer.scrollTop = chatContainer.scrollHeight;
      this.chatHistory.push({ content, type, timestamp: Date.now(), meta }); this.saveChatHistory();
    }
    async provideFeedback(interactionId, isPositive) {
      try {
        await fetch(`${this.options.apiEndpoint}/feedback`, {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ interaction_id: interactionId, feedback: isPositive ? 'positive' : 'negative', success: isPositive })
        });
        this.addMessage('Thank you for the feedback!', 'system');
        setTimeout(() => this.fetchStats(), 500);
      } catch (error) { Utils.log(`Feedback error: ${error.message || error}`, 'error'); }
    }
    async fetchStats() {
      try {
        const r = await fetch(`${this.options.apiEndpoint}/stats`);
        if (!r.ok) throw new Error(`HTTP ${r.status}: ${r.statusText}`);
        const ct = r.headers.get('content-type'); if (!ct?.includes('application/json')) throw new Error('Response is not JSON');
        const stats = await r.json(); this.knowledge_stats = stats; this.updateStatsDisplay(stats);
      } catch (e) {
        const defaultStats = { knowledge_entries: 0, success_rate: 0.75, total_interactions: 0 };
        this.knowledge_stats = defaultStats; this.updateStatsDisplay(defaultStats);
      }
    }
    updateStatsDisplay(stats) {
      const knowledgeCount = document.getElementById('knowledge-count');
      const successRate = document.getElementById('success-rate');
      if (knowledgeCount) knowledgeCount.textContent = String(stats.knowledge_entries || 0);
      if (successRate) successRate.textContent = String(Math.round((stats.success_rate || 0) * 100));
    }
    buildContext() {
      return { recent_messages: this.chatHistory.slice(-5), world_engine_connected: true,
        session_info: { timestamp: Date.now(), message_count: this.chatHistory.length } };
    }
    showTyping(show) { this.isTyping = show; const el = document.getElementById('ai-bot-typing'); if (el) el.style.display = show ? 'flex' : 'none'; }
    clearChat() {
      this.chatHistory = [];
      const chatContainer = document.getElementById('ai-bot-chat');
      if (chatContainer) chatContainer.innerHTML = '<div class="chat-welcome"><div class="welcome-message">Chat cleared. How can I help?</div></div>';
      this.saveChatHistory();
    }
    handleWorldEngineResult(msg) {
      if (msg.outcome && this.options.autoLearn) {
        this.addMessage(`I noticed the World Engine ran on: "${msg.input || msg.text || ''}". Logged for learning.`, 'system');
      }
    }
    handleChatCommand(msg) {
      if (msg.line?.startsWith('/ai ')) {
        const query = msg.line.substring(4); this.sendMessage(query);
      }
    }
    saveChatHistory() {
      try { localStorage.setItem('ai_bot_chat_history', JSON.stringify(this.chatHistory.slice(-this.options.maxChatHistory))); } catch {}
    }
    loadChatHistory() {
      try { const saved = localStorage.getItem('ai_bot_chat_history'); if (saved) this.chatHistory = JSON.parse(saved); } catch {}
    }
    getStats() { return { message_count: this.chatHistory.length, current_interaction: this.currentInteractionId, knowledge_stats: this.knowledge_stats, is_typing: this.isTyping }; }
  }

  // ------------------------------------------------------
  // ChatController (as provided; trimmed logs/notes)
  // ------------------------------------------------------
  class ChatController {
    constructor(options = {}) {
      this.options = { autoLinkClips: true, transcriptCommands: true, commandPrefix: '/', ...options };
      this.lastClipId = null;
      this.activeRuns = new Map();
      this.commandHistory = [];
      this.bindEvents(); this.setupUI();
    }
    bindEvents() {
      onBus(async (msg) => {
        try {
          switch (msg.type) {
          case 'chat.cmd': await this.handleCommand(msg.line); break;
          case 'rec.clip': this.handleClipReady(msg); break;
          case 'rec.transcript': if (this.options.transcriptCommands) this.handleTranscript(msg); break;
          case 'eng.result': await this.handleEngineResult(msg); break;
          case 'eng.error': this.handleEngineError(msg); break;
          case 'rec.error': this.handleRecorderError(msg); break;
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
            .chat-interface { display:flex; flex-direction:column; height:300px; border:1px solid #333; border-radius:8px; }
            .chat-messages { flex:1; padding:8px; overflow-y:auto; background:#1a1a1a; border-radius:8px 8px 0 0; }
            .chat-input-area { display:flex; padding:8px; border-top:1px solid #333; }
            #chat-input { flex:1; padding:8px; border:1px solid #444; background:#222; color:#fff; border-radius:4px; }
            #chat-send { padding:8px 16px; margin-left:8px; border:1px solid #444; background:#333; color:#fff; border-radius:4px; cursor:pointer; }
            .chat-status { padding:4px 8px; font-size:12px; color:#888; }
            .chat-message { margin:4px 0; padding:6px; border-radius:4px; white-space:pre-wrap; }
            .chat-user { background:#2a4a6b; }
            .chat-system { background:#2a2a2a; color:#ccc; }
            .chat-error { background:#6b2a2a; }
            .chat-result { background:#2a6b2a; }
          </style>`;
        const input = document.getElementById('chat-input'); const sendBtn = document.getElementById('chat-send');
        const sendMessage = () => { const v = input.value.trim(); if (v) { this.handleCommand(v); input.value = ''; } };
        sendBtn?.addEventListener('click', sendMessage);
        input?.addEventListener('keypress', (e) => { if (e.key === 'Enter') sendMessage(); });
      }
      this.announce('World Engine Studio ready. Try: /run <text>, /test <name>, /rec start, /help', 'system');
    }
    async handleCommand(line) {
      const trimmed = line.trim(); if (!trimmed) return;
      this.commandHistory.push({ command: trimmed, timestamp: Date.now() });
      this.announce(`> ${trimmed}`, 'user');
      const cmd = Utils.parseCommand(trimmed);
      switch (cmd.type) {
      case 'run': await this.runEngine(cmd.args); break;
      case 'test': await this.loadTest(cmd.args); break;
      case 'rec': await this.handleRecordingCommand(cmd.args); break;
      case 'mark': await this.addMarker(cmd.args); break;
      case 'help': this.showHelp(); break;
      case 'status': await this.showStatus(); break;
      case 'history': this.showHistory(); break;
      case 'clear': this.clearMessages(); break;
      default: this.announce(`Unknown command: ${cmd.type}. Type /help for available commands.`, 'error');
      }
    }
    async runEngine(text) {
      if (!text.trim()) { this.announce('Please provide text to analyze', 'error'); return; }
      this.announce(`Running analysis on: "${text}"`, 'system'); this.updateStatus('Analyzing...');
      if (this.lastClipId) sendBus({ type: 'rec.mark', tag: 'run-start', runId: null });
      sendBus({ type: 'eng.run', text });
    }
    async loadTest(testName) {
      if (!testName.trim()) { this.announce('Please specify a test name', 'error'); return; }
      this.announce(`Loading test: ${testName}`, 'system');
      sendBus({ type: 'eng.test', name: testName });
    }
    async handleRecordingCommand(action) {
      switch ((action || '').trim()) {
      case 'start': this.announce('Starting microphone recording...', 'system'); sendBus({ type: 'rec.start', mode: 'mic', meta: { source: 'chat' } }); break;
      case 'screen': this.announce('Starting screen recording...', 'system'); sendBus({ type: 'rec.start', mode: 'screen', meta: { source: 'chat' } }); break;
      case 'stop': this.announce('Stopping recording...', 'system'); sendBus({ type: 'rec.stop' }); break;
      default: this.announce('Recording commands: start, screen, stop', 'error');
      }
    }
    async addMarker(tag) { if (!tag.trim()) { this.announce('Please specify a marker tag', 'error'); return; } sendBus({ type: 'rec.mark', tag: tag.trim() }); this.announce(`Marker added: ${tag}`, 'system'); }
    handleTranscript(msg) {
      const text = (msg.text || '').trim(); if (!text) return;
      const lower = text.toLowerCase();
      const keys = ['run', 'analyze', 'test', 'record', 'stop'];
      let detected = null;
      for (const k of keys) if (lower.startsWith(k)) { detected = `/${k} ${text.slice(k.length).trim()}`; break; }
      if (detected) { this.announce(`Voice command detected: ${detected}`, 'system'); this.handleCommand(detected); }
      else { this.announce(`Voice input: "${text}"`, 'system'); this.handleCommand(text); }
    }
    handleClipReady(msg) { this.lastClipId = msg.clipId; this.announce(`Recording saved: ${msg.clipId} (${msg.size ? Math.round(msg.size/1024) + 'KB' : 'unknown size'})`, 'system'); }
    async handleEngineResult(msg) {
      const { runId, outcome, input } = msg;
      if (this.lastClipId && this.options.autoLinkClips) {
        await Store.save(`runs.${runId}`, { runId, ts: Date.now(), input, outcome, clipId: this.lastClipId });
        this.announce(`Run ${runId} linked to recording ${this.lastClipId}`, 'system'); this.lastClipId = null;
      }
      this.announce(`Analysis complete (${runId})`, 'result');
      if (outcome.items && Array.isArray(outcome.items)) {
        this.announce(`Found ${outcome.items.length} items`, 'result');
        const preview = outcome.items.slice(0, 3).map(item => typeof item === 'object'
          ? `• ${item.lemma || item.word || item.text || 'Item'}: ${item.root || item.score || JSON.stringify(item).slice(0, 50)}`
          : `• ${item}`).join('\n');
        this.announce(preview, 'result');
        if (outcome.items.length > 3) this.announce(`... and ${outcome.items.length - 3} more`, 'result');
      } else if (outcome.result) {
        this.announce(`Result: ${outcome.result}`, 'result');
      }
      this.updateStatus('Ready');
    }
    handleEngineError(msg) { this.announce(`Engine error: ${msg.error}`, 'error'); this.updateStatus('Error - Ready'); }
    handleRecorderError(msg) { this.announce(`Recording error: ${msg.error}`, 'error'); }
    showHelp() {
      const help = `
Available commands:
• /run <text> - Analyze text with World Engine
• /test <name> - Load a test case
• /rec start|screen|stop - Recording controls
• /mark <tag> - Add timeline marker
• /status - Show system status
• /history - Show command history
• /clear - Clear messages
• /help - Show this help

You can also type text directly to analyze it. Voice commands work if recording is active.`.trim();
      this.announce(help, 'system');
    }
    async showStatus() {
      const status = `
System Status:
• Engine: Ready
• Recorder: ${this.lastClipId ? 'Has recent clip' : 'Idle'}
• Commands: ${this.commandHistory.length} in history
• Store: Available`.trim();
      this.announce(status, 'system');
    }
    showHistory() {
      if (this.commandHistory.length === 0) { this.announce('No command history', 'system'); return; }
      const recent = this.commandHistory.slice(-5).map((e,i)=>`${i+1}. ${e.command}`).join('\n');
      this.announce(`Recent commands:\n${recent}`, 'system');
    }
    clearMessages() { const c = document.getElementById('chat-messages'); if (c) c.innerHTML = ''; this.announce('Messages cleared', 'system'); }
    announce(message, level = 'system') {
      const c = document.getElementById('chat-messages');
      if (!c) { Utils.log(message, level); return; }
      const el = document.createElement('div'); el.className = `chat-message chat-${level}`; el.textContent = message;
      c.appendChild(el); c.scrollTop = c.scrollHeight; Utils.log(`[Chat:${level}] ${message}`);
    }
    updateStatus(s) { const el = document.getElementById('chat-status'); if (el) el.textContent = s; }
    getHistory() { return [...this.commandHistory]; }
    async execute(command) { return this.handleCommand(command); }
    getStats() { return { commandsExecuted: this.commandHistory.length, lastClipId: this.lastClipId, activeRuns: this.activeRuns.size }; }
  }

  // ------------------------------------------------------
  // EngineController (unchanged API; robustness tweaks)
  // ------------------------------------------------------
  class EngineController {
    constructor(engineFrame) {
      this.engineFrame = engineFrame;
      this.transport = null;
      this.isReady = false;
      this.setupTransport(); this.bindEvents();
    }
    setupTransport() {
      this.transport = setupEngineTransport(this.engineFrame);
      this.engineFrame.addEventListener('load', () => {
        setTimeout(() => {
          this.isReady = true;
          Utils.log('Engine iframe loaded and ready');
          sendBus({ type: 'eng.ready' });
        }, 500);
      });
    }
    bindEvents() {
      onBus(async (msg) => {
        if (!this.isReady) {
          Utils.log('Engine not ready, queuing message', 'warn');
          setTimeout(() => onBus(msg), 100);
          return;
        }
        try {
          switch (msg.type) {
          case 'eng.run': await this.handleRun(msg); break;
          case 'eng.test': await this.handleTest(msg); break;
          case 'eng.status': await this.handleStatus(msg); break;
          }
        } catch (error) {
          Utils.log(`Engine error: ${error.message}`, 'error');
          sendBus({ type: 'eng.error', error: error.message, originalMessage: msg });
        }
      });
    }
    async handleRun(msg) {
      const runId = Utils.generateId();
      Utils.log(`Starting engine run: ${runId}`);
      sendBus({ type: 'rec.mark', tag: 'run-start', runId });
      this.transport.withEngine((doc) => {
        const input = doc.getElementById('input');
        const runBtn = doc.getElementById('run');
        if (!input || !runBtn) throw new Error('Engine input/run elements not found');
        input.value = (msg.text || '').trim(); runBtn.click();
        setTimeout(() => {
          try {
            const output = doc.getElementById('out'); if (!output) throw new Error('Engine output element not found');
            const raw = output.textContent || '{}';
            let outcome = {};
            try { outcome = JSON.parse(raw); }
            catch { outcome = { type: 'text', result: raw, input: msg.text, timestamp: Date.now() }; }
            Store.save('wordEngine.lastRun', outcome);
            Store.save(`runs.${runId}`, { runId, ts: Date.now(), input: msg.text, outcome, clipId: null });
            sendBus({ type: 'rec.mark', tag: 'run-end', runId });
            sendBus({ type: 'eng.result', runId, outcome, input: msg.text });
            Utils.log(`Engine run completed: ${runId}`);
          } catch (error) {
            Utils.log(`Engine result processing error: ${error.message}`, 'error');
            sendBus({ type: 'eng.error', runId, error: error.message });
          }
        }, 150);
      });
    }
    async handleTest(msg) {
      Utils.log(`Loading test: ${msg.name}`);
      this.transport.withEngine((doc) => {
        const testsContainer = doc.getElementById('tests');
        if (!testsContainer) throw new Error('Tests container not found');
        const buttons = testsContainer.querySelectorAll('button');
        const testName = (msg.name || '').toLowerCase(); let found = false;
        for (const btn of buttons) {
          const txt = (btn.textContent || '').trim().toLowerCase();
          if (txt === testName || txt.includes(testName)) {
            btn.click(); found = true; Utils.log(`Test loaded: ${msg.name}`);
            setTimeout(() => {
              const input = doc.getElementById('input');
              if (input && input.value.trim()) sendBus({ type: 'eng.run', text: input.value.trim() });
            }, 100);
            break;
          }
        }
        if (!found) throw new Error(`Test not found: ${msg.name}`);
      });
    }
    async handleStatus() {
      this.transport.withEngine((doc) => {
        const input = doc.getElementById('input'); const output = doc.getElementById('out');
        const status = { ready: this.isReady, hasInput: !!input, hasOutput: !!output,
          inputValue: input?.value || '', lastOutput: output?.textContent || '',
          transport: this.transport.isSameOrigin() ? 'same-origin' : 'cross-origin' };
        sendBus({ type: 'eng.status.result', status }); Utils.log(`Engine status: ${JSON.stringify(status)}`);
      });
    }
    async runText(text) {
      return new Promise((resolve) => {
        const runId = Utils.generateId();
        const off = onBus((msg) => {
          if (msg.type === 'eng.result' && msg.runId === runId) { off?.(); resolve(msg.outcome); }
        });
        sendBus({ type: 'eng.run', text });
      });
    }
    async loadTest(name) { sendBus({ type: 'eng.test', name }); }
    getStatus() { return { ready: this.isReady, transport: this.transport?.isSameOrigin() ? 'same-origin' : 'cross-origin' }; }
  }

  // ------------------------------------------------------
  // Orchestrator: one call to boot everything and wire it
  // ------------------------------------------------------
  class WorldEngineOrchestrator {
    constructor({ engineFrameSelector = 'iframe[src*="engine"], iframe[src*="lexicon"]', ai = {}, chat = {}, rec = {} } = {}) {
      // Instances
      this.recorder = new RecorderController(rec);
      this.ai = new AIBotController(ai);
      const engineFrame = document.querySelector(engineFrameSelector);
      this.engine = engineFrame ? new EngineController(engineFrame) : null;
      this.chat = new ChatController(chat);

      // Publish to StudioBridge if present
      if (root.StudioBridge) {
        root.StudioBridge.RecorderController = this.recorder;
        root.StudioBridge.AIBotController = this.ai;
        root.StudioBridge.ChatController = this.chat;
        root.StudioBridge.EngineController = this.engine;
      }

      Utils.log('WorldEngineOrchestrator ready');
      sendBus({ type: 'controllers.ready', ok: true, hasEngine: !!this.engine });
    }

    getStats() {
      return {
        recorder: this.recorder?.getStatus?.(),
        chat: this.chat?.getStats?.(),
        ai: this.ai?.getStats?.(),
        engine: this.engine?.getStatus?.()
      };
    }
  }

  // ---------------------------
  // Public API (UMD)
  // ---------------------------
  const API = {
    RecorderController,
    AIBotController,
    ChatController,
    EngineController,
    WorldEngineOrchestrator,
    shims: { Utils, Store, sendBus, onBus, setupEngineTransport }
  };

  // Optional: auto-boot if a marker is present
  if (root.WE_AUTO_BOOT_CONTROLLERS) {
    root.__WE_CONTROLLERS__ = new WorldEngineOrchestrator();
  }

  return API;
});
