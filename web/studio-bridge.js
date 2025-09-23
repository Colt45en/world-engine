/**
 * Studio Bridge - Event Bus System for World Engine Connections
 *
 * Connects Recorder ↔ AI Chat ↔ World Engine with stable message contracts.
 * Supports both same-origin DOM and cross-origin postMessage transport.
 */

// -------- Event Bus (BroadcastChannel with fallback to window events) --------
const busName = 'studio-bus';
const bc = ('BroadcastChannel' in self) ? new BroadcastChannel(busName) : null;

const onBus = (fn) => bc ? (bc.onmessage = (e) => fn(e.data)) :
  window.addEventListener('studio:msg', (e) => fn(e.detail));

const sendBus = (msg) => bc ? bc.postMessage(msg) :
  window.dispatchEvent(new CustomEvent('studio:msg', { detail: msg }));

// -------- Recorder API (Recorder page exposes this) --------
window.RecorderAPI = {
  async startMic(meta = {}) { sendBus({ type: 'rec.start', mode: 'mic', meta }); },
  async startScreen(meta = {}) { sendBus({ type: 'rec.start', mode: 'screen', meta }); },
  async startBoth(meta = {}) { sendBus({ type: 'rec.start', mode: 'both', meta }); },
  async stop() { sendBus({ type: 'rec.stop' }); },
  async mark(tag = 'mark', runId = null) { sendBus({ type: 'rec.mark', tag, runId }); },
  // Recorder will emit: {type:'rec.ready'|'rec.clip'|'rec.transcript', ...}
};

// -------- Engine API (Chat page consumes this) --------
window.EngineAPI = {
  async run(text) { sendBus({ type: 'eng.run', text }); },
  async test(name) { sendBus({ type: 'eng.test', name }); },
  async getStatus() { sendBus({ type: 'eng.status' }); },
  // Engine will reply: {type:'eng.result', outcome, runId}
};

// -------- Chat API (external triggers like voice commands) --------
window.ChatAPI = {
  async command(line) { sendBus({ type: 'chat.cmd', line }); },
  async announce(message, level = 'info') { sendBus({ type: 'chat.announce', message, level }); }
};

// -------- External Store Helper --------
const Store = {
  async save(key, value) {
    if (window.externalStore?.upsert) {
      return await window.externalStore.upsert(key, value);
    } else if (window.localStorage) {
      window.localStorage.setItem(key, JSON.stringify(value));
      return value;
    }
    throw new Error('No storage available');
  },

  async load(key) {
    if (window.externalStore?.get) {
      return await window.externalStore.get(key);
    } else if (window.localStorage) {
      const item = window.localStorage.getItem(key);
      return item ? JSON.parse(item) : null;
    }
    return null;
  }
};

// -------- Transport Detection and Setup --------
function setupEngineTransport(engineFrame) {
  const isSameOrigin = () => {
    try {
      return engineFrame.contentWindow.document !== null;
    } catch (e) {
      return false; // Cross-origin
    }
  };

  const withEngine = (fn) => {
    if (isSameOrigin()) {
      const doc = engineFrame.contentWindow.document;
      fn(doc);
    } else {
      console.warn('Cross-origin detected, use postMessage transport');
    }
  };

  return { isSameOrigin, withEngine };
}

// -------- Message Type Definitions (TypeScript-style comments) --------
/**
 * @typedef {Object} EngineRunMessage
 * @property {'eng.run'} type
 * @property {string} text
 */

/**
 * @typedef {Object} EngineTestMessage
 * @property {'eng.test'} type
 * @property {string} name
 */

/**
 * @typedef {Object} EngineResultMessage
 * @property {'eng.result'} type
 * @property {string} runId
 * @property {Object} outcome
 */

/**
 * @typedef {Object} RecorderStartMessage
 * @property {'rec.start'} type
 * @property {'mic'|'screen'|'both'} mode
 * @property {Object} [meta]
 */

/**
 * @typedef {Object} RecorderClipMessage
 * @property {'rec.clip'} type
 * @property {string} clipId
 * @property {string} [url]
 * @property {Object} [meta]
 */

/**
 * @typedef {Object} RecorderTranscriptMessage
 * @property {'rec.transcript'} type
 * @property {string} [clipId]
 * @property {string} text
 * @property {number} ts
 */

// -------- Utilities --------
const Utils = {
  generateId: () => String(Date.now() + Math.random().toString(36).substr(2, 9)),

  parseCommand: (line) => {
    const t = line.trim();
    if (t.startsWith('/run ')) return { type: 'run', args: t.slice(5) };
    if (t.startsWith('/test ')) return { type: 'test', args: t.slice(6) };
    if (t === '/rec start') return { type: 'rec', args: 'start' };
    if (t === '/rec stop') return { type: 'rec', args: 'stop' };
    if (t.startsWith('/mark ')) return { type: 'mark', args: t.slice(6) };
    return { type: 'run', args: t }; // fallthrough: treat as run
  },

  log: (message, level = 'info') => {
    const timestamp = new Date().toISOString();
    console[level](`[Studio:${level}] ${timestamp} - ${message}`);
  }
};

// -------- Export for modules --------
if (typeof module !== 'undefined' && module.exports) {
  module.exports = { onBus, sendBus, RecorderAPI: window.RecorderAPI, EngineAPI: window.EngineAPI, ChatAPI: window.ChatAPI, Store, Utils, setupEngineTransport };
} else {
  window.StudioBridge = { onBus, sendBus, RecorderAPI: window.RecorderAPI, EngineAPI: window.EngineAPI, ChatAPI: window.ChatAPI, Store, Utils, setupEngineTransport };
}

// -------- Debug Helper --------
if (window.location.search.includes('debug=studio')) {
  onBus((msg) => {
    Utils.log(`Bus message: ${JSON.stringify(msg)}`, 'debug');
  });
}
