/*!
 * WorldEngine Tier-4 Bundle
 * Combines: EngineRoom + RoomLayout + Tier4Room (browser)
 *          + StudioBridge wiring
 *          + createTier4RoomBridge(iframe, wsUrl)
 *          + Tier4EnhancedRelay (Node-only)
 *          + Nucleus System Integration
 * UMD export: window.WorldEngineTier4 or module.exports
 */
(function (global, factory) {
  if (typeof module === 'object' && typeof module.exports === 'object') {
    module.exports = factory(global);
  } else {
    global.WorldEngineTier4 = factory(global);
  }
})(typeof window !== 'undefined' ? window : globalThis, function (root) {
  'use strict';

  // --------------------------------------
  // Nucleus System Configuration
  // Core intelligence mappings and communication protocols
  // --------------------------------------
  const NUCLEUS_CONFIG = {
    operators: {
      'VIBRATE': 'ST',      // Stabilization
      'OPTIMIZATION': 'UP',  // Update/Progress
      'STATE': 'CV',        // Convergence
      'SEED': 'RB'          // Rollback
    },

    aiBot: {
      messageTypes: ['query', 'learning', 'feedback'],
      routing: {
        query: 'VIBRATE',
        learning: 'OPTIMIZATION',
        feedback: 'STATE'
      }
    },

    librarians: {
      types: ['Math Librarian', 'English Librarian', 'Pattern Librarian'],
      dataTypes: ['pattern', 'classification', 'analysis'],
      routing: {
        pattern: 'VIBRATE',
        classification: 'STATE',
        analysis: 'OPTIMIZATION'
      }
    }
  };

  // --------------------------------------
  // StudioBridge (link or shim)
  // Enhanced with nucleus event support
  // --------------------------------------
  const StudioBridge = (function ensureBridge() {
    if (root.StudioBridge) return root.StudioBridge;

    const busName = 'studio-bus';
    const bc = (typeof BroadcastChannel !== 'undefined') ? new BroadcastChannel(busName) : null;

    const listeners = [];
    function callListeners(msg) {
      listeners.forEach(function (entry) {
        try {
          if (!entry.type || entry.type === msg.type) entry.fn(msg);
        } catch (e) {
          console.error('[StudioBridge shim] listener error', e);
        }
      });
    }

    if (bc) {
      bc.onmessage = function (e) { callListeners(e.data); };
    } else if (root && root.addEventListener) {
      root.addEventListener('studio:msg', function (e) { callListeners(e.detail); });
    }

    function onBus(typeOrFn, maybeFn) {
      if (typeof typeOrFn === 'function') {
        listeners.push({ type: null, fn: typeOrFn });
      } else if (typeof maybeFn === 'function') {
        listeners.push({ type: typeOrFn, fn: maybeFn });
      }
    }

    function sendBus(msg) {
      if (bc) bc.postMessage(msg);
      else if (root && root.dispatchEvent)
        root.dispatchEvent(new CustomEvent('studio:msg', { detail: msg }));
    }

    const shim = {
      onBus: onBus,
      sendBus: sendBus,
      RecorderAPI: root.RecorderAPI || {},
      EngineAPI: root.EngineAPI || {},
      ChatAPI: root.ChatAPI || {},
      Store: {
        save: async function (key, value) {
          if (root.localStorage) root.localStorage.setItem(key, JSON.stringify(value));
          return value;
        },
        load: async function (key) {
          if (!root.localStorage) return null;
          const v = root.localStorage.getItem(key);
          return v ? JSON.parse(v) : null;
        }
      },
      Utils: {
        generateId: function () { return String(Date.now() + Math.random().toString(36).slice(2)); },
        parseCommand: function (line) { return { type: 'run', args: String(line || '') }; },
        log: function (m, level) { console[level || 'log']('[StudioBridge shim]', m); }
      },
      // Enhanced nucleus event support
      NucleusAPI: {
        processEvent: function(role, data) {
          const operator = NUCLEUS_CONFIG.operators[role];
          if (operator) {
            sendBus({
              type: 'tier4.nucleusEvent',
              role: role,
              operator: operator,
              data: data,
              timestamp: Date.now()
            });
            return { role, operator, processed: true };
          }
          return { role, operator: null, processed: false };
        },
        routeAIBotMessage: function(message, messageType) {
          const nucleusRole = NUCLEUS_CONFIG.aiBot.routing[messageType];
          if (nucleusRole) {
            return this.processEvent(nucleusRole, {
              source: 'ai_bot',
              message: message,
              messageType: messageType
            });
          }
          return null;
        },
        routeLibrarianData: function(librarian, dataType, data) {
          const nucleusRole = NUCLEUS_CONFIG.librarians.routing[dataType];
          if (nucleusRole) {
            return this.processEvent(nucleusRole, {
              source: 'librarian',
              librarian: librarian,
              dataType: dataType,
              data: data
            });
          }
          return null;
        }
      }
    };

    return shim;
  })();

  // --------------------------------------
  // Tier4 Room Implementation
  // Enhanced with nucleus intelligence
  // --------------------------------------
  function Tier4Room(container) {
    this.container = container;
    this.panels = [];
    this.state = { x: [0, 0.5, 0.4, 0.6], kappa: 0.6, level: 0 };
    this.snapshots = new Map();
    this.eventLog = [];
    this.nucleusActive = true;

    this.setupRoom();
    this.setupNucleusUI();
  }

  Tier4Room.prototype.setupRoom = function() {
    this.container.style.cssText = `
      position: relative; width: 100%; height: 100%;
      background: linear-gradient(135deg, #0a0e1a 0%, #1a1f2e 50%, #2d1b3d 100%);
      color: #e6f0ff; overflow: hidden; font-family: system-ui;
    `;

    // Create nucleus heartbeat indicator
    const heartbeat = document.createElement('div');
    heartbeat.id = 'nucleus-heartbeat';
    heartbeat.style.cssText = `
      position: absolute; top: 10px; left: 10px; z-index: 1000;
      background: rgba(84, 240, 184, 0.2); border: 1px solid #54f0b8;
      border-radius: 20px; padding: 8px 16px; font-size: 12px;
      display: flex; align-items: center; gap: 8px;
    `;
    heartbeat.innerHTML = '🧠 <span id="nucleus-status">Nucleus Active</span>';
    this.container.appendChild(heartbeat);

    // Create communication feed
    const commFeed = document.createElement('div');
    commFeed.id = 'communication-feed';
    commFeed.style.cssText = `
      position: absolute; top: 10px; right: 10px; z-index: 1000;
      background: rgba(0, 0, 0, 0.7); border: 1px solid #2a3548;
      border-radius: 6px; padding: 10px; width: 300px; height: 150px;
      overflow-y: auto; font-size: 10px; font-family: monospace;
    `;
    commFeed.innerHTML = '<div style="color: #54f0b8; font-weight: bold; margin-bottom: 5px;">📡 Communication Feed</div>';
    this.container.appendChild(commFeed);

    this.logCommunication('system', 'Nucleus system initialized');
  };

  Tier4Room.prototype.setupNucleusUI = function() {
    // Create nucleus control panel
    const controlPanel = document.createElement('div');
    controlPanel.id = 'nucleus-controls';
    controlPanel.style.cssText = `
      position: absolute; bottom: 10px; left: 10px; z-index: 1000;
      background: rgba(26, 35, 50, 0.9); border: 1px solid #2a3548;
      border-radius: 8px; padding: 15px; min-width: 200px;
    `;

    const title = document.createElement('div');
    title.textContent = '🧠 Nucleus Controls';
    title.style.cssText = 'color: #54f0b8; font-weight: bold; margin-bottom: 10px; font-size: 14px;';
    controlPanel.appendChild(title);

    // Add nucleus trigger buttons
    const self = this;
    ['VIBRATE', 'OPTIMIZATION', 'STATE', 'SEED'].forEach(function(role) {
      const button = document.createElement('button');
      const operator = NUCLEUS_CONFIG.operators[role];
      const icon = role === 'VIBRATE' ? '🌊' : role === 'OPTIMIZATION' ? '⚡' : role === 'STATE' ? '🎯' : '🌱';

      button.innerHTML = `${icon} ${role}<br><small>→ ${operator}</small>`;
      button.style.cssText = `
        background: #243344; border: 1px solid #ff9f43; color: #e6f0ff;
        padding: 8px; margin: 2px; border-radius: 4px; cursor: pointer;
        font-size: 10px; width: 80px; height: 50px;
      `;

      button.onclick = function() {
        self.triggerNucleusEvent(role);
      };

      controlPanel.appendChild(button);
    });

    this.container.appendChild(controlPanel);
  };

  Tier4Room.prototype.logCommunication = function(type, message) {
    const feed = this.container.querySelector('#communication-feed');
    if (feed) {
      const timestamp = new Date().toLocaleTimeString();
      const color = type === 'ai_bot' ? '#7cdcff' : type === 'librarian' ? '#ff9f43' : '#54f0b8';

      const entry = document.createElement('div');
      entry.style.cssText = `color: ${color}; margin-bottom: 2px;`;
      entry.innerHTML = `<span style="opacity: 0.6;">[${timestamp}]</span> ${message}`;

      feed.appendChild(entry);
      feed.scrollTop = feed.scrollHeight;

      // Keep only last 20 entries
      while (feed.children.length > 21) { // +1 for title
        feed.removeChild(feed.children[1]); // Skip title
      }
    }
  };

  Tier4Room.prototype.triggerNucleusEvent = function(role) {
    const operator = NUCLEUS_CONFIG.operators[role];
    if (!operator) return;

    this.logCommunication('nucleus', `${role} → ${operator}`);

    // Apply the corresponding operator
    this.applyOperator(operator, {
      source: 'nucleus',
      role: role,
      timestamp: Date.now()
    });

    // Emit event through StudioBridge
    StudioBridge.sendBus({
      type: 'tier4.nucleusEvent',
      role: role,
      operator: operator,
      timestamp: Date.now()
    });
  };

  Tier4Room.prototype.processAIBotMessage = function(message, messageType) {
    const result = StudioBridge.NucleusAPI.routeAIBotMessage(message, messageType);
    if (result && result.processed) {
      this.logCommunication('ai_bot', `${messageType}: ${message.substring(0, 30)}...`);
      this.applyOperator(result.operator, {
        source: 'ai_bot',
        message: message,
        messageType: messageType
      });
    }
    return result;
  };

  Tier4Room.prototype.processLibrarianData = function(librarian, dataType, data) {
    const result = StudioBridge.NucleusAPI.routeLibrarianData(librarian, dataType, data);
    if (result && result.processed) {
      this.logCommunication('librarian', `${librarian}: ${dataType} data`);
      this.applyOperator(result.operator, {
        source: 'librarian',
        librarian: librarian,
        dataType: dataType,
        data: data
      });
    }
    return result;
  };

  Tier4Room.prototype.addPanel = function(config) {
    const panel = document.createElement('div');
    panel.className = 'tier4-panel';
    panel.style.cssText = `
      position: absolute; background: rgba(26, 35, 50, 0.95);
      border: 1px solid #2a3548; border-radius: 6px;
      left: ${config.x || 50}px; top: ${config.y || 50}px;
      width: ${config.w || 200}px; height: ${config.h || 150}px;
      resize: both; overflow: auto; z-index: 100;
    `;

    const header = document.createElement('div');
    header.textContent = config.title || 'Panel';
    header.style.cssText = `
      background: #54f0b8; color: #0b0e14; padding: 4px 8px;
      font-weight: bold; cursor: move; font-size: 12px;
    `;
    panel.appendChild(header);

    if (config.html) {
      const content = document.createElement('div');
      content.innerHTML = config.html;
      content.style.cssText = 'padding: 8px; height: calc(100% - 28px); overflow: auto;';
      panel.appendChild(content);
    }

    this.makeDraggable(panel, header);
    this.container.appendChild(panel);
    this.panels.push(panel);

    return panel;
  };

  Tier4Room.prototype.makeDraggable = function(element, handle) {
    let isDragging = false;
    const offset = { x: 0, y: 0 };

    handle.onmousedown = function(e) {
      isDragging = true;
      const rect = element.getBoundingClientRect();
      offset.x = e.clientX - rect.left;
      offset.y = e.clientY - rect.top;
      e.preventDefault();
    };

    document.onmousemove = function(e) {
      if (!isDragging) return;
      const containerRect = element.parentElement.getBoundingClientRect();
      element.style.left = (e.clientX - containerRect.left - offset.x) + 'px';
      element.style.top = (e.clientY - containerRect.top - offset.y) + 'px';
    };

    document.onmouseup = function() {
      isDragging = false;
    };
  };

  Tier4Room.prototype.applyOperator = function(operator, context) {
    const previousState = Object.assign({}, this.state);

    // Enhanced state transformation with nucleus intelligence
    switch (operator) {
    case 'ST': // Stabilization (VIBRATE)
      this.state.kappa = Math.max(0, Math.min(1, this.state.kappa + (Math.random() - 0.5) * 0.1));
      this.state.x = this.state.x.map(function(v) {
        return Math.max(0, Math.min(1, v + (Math.random() - 0.5) * 0.05));
      });
      break;
    case 'UP': // Update/Progress (OPTIMIZATION)
      this.state.level += 1;
      this.state.kappa = Math.min(1, this.state.kappa + 0.05);
      break;
    case 'CV': // Convergence (STATE)
      this.state.x = this.state.x.map(function(v) {
        return v + (0.5 - v) * 0.1;
      });
      break;
    case 'RB': // Rollback (SEED)
      this.state = { x: [0, 0.5, 0.4, 0.6], kappa: 0.6, level: 0 };
      break;
    default:
      // Generic transformation
      this.state.kappa = Math.max(0, Math.min(1, this.state.kappa + (Math.random() - 0.5) * 0.02));
    }

    this.eventLog.push({
      operator: operator,
      context: context,
      previousState: previousState,
      newState: Object.assign({}, this.state),
      timestamp: Date.now()
    });

    // Update nucleus status display
    this.updateNucleusStatus();

    // Emit event
    StudioBridge.sendBus({
      type: 'tier4.operatorApplied',
      operator: operator,
      previousState: previousState,
      newState: this.state,
      context: context,
      timestamp: Date.now()
    });

    return this.state;
  };

  Tier4Room.prototype.updateNucleusStatus = function() {
    const status = this.container.querySelector('#nucleus-status');
    if (status) {
      const confidence = (this.state.kappa * 100).toFixed(1);
      status.textContent = `Nucleus Active - Level ${this.state.level} - ${confidence}% Confidence`;
    }
  };

  Tier4Room.prototype.applyMacro = function(macro) {
    this.logCommunication('system', `Executing macro: ${macro}`);

    let operators = [];
    switch (macro) {
    case 'IDE_A': operators = ['ST', 'PR']; break;
    case 'IDE_B': operators = ['UP', 'SL']; break;
    case 'IDE_C': operators = ['CV', 'MD']; break;
    case 'MERGE_ABC': operators = ['ST', 'UP', 'CV']; break;
    default: operators = ['UP'];
    }

    const self = this;
    operators.forEach(function(op, i) {
      setTimeout(function() {
        self.applyOperator(op, { source: 'macro', macro: macro, step: i + 1 });
      }, i * 200);
    });
  };

  Tier4Room.prototype.saveSnapshot = function(id) {
    if (!id) id = 'snapshot_' + Date.now();
    this.snapshots.set(id, {
      state: Object.assign({}, this.state),
      timestamp: Date.now(),
      panels: this.panels.length
    });
    return id;
  };

  Tier4Room.prototype.loadSnapshot = function(id) {
    const snapshot = this.snapshots.get(id);
    if (snapshot) {
      this.state = Object.assign({}, snapshot.state);
      this.updateNucleusStatus();
      return this.state;
    }
    return null;
  };

  Tier4Room.prototype.toast = function(message) {
    const toast = document.createElement('div');
    toast.textContent = message;
    toast.style.cssText = `
      position: fixed; top: 50%; left: 50%; transform: translate(-50%, -50%);
      background: rgba(84, 240, 184, 0.9); color: #0b0e14; padding: 12px 20px;
      border-radius: 20px; font-weight: bold; z-index: 10000; font-size: 14px;
      box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
    `;

    document.body.appendChild(toast);
    setTimeout(function() {
      if (toast.parentNode) toast.parentNode.removeChild(toast);
    }, 3000);
  };

  // --------------------------------------
  // Tier4RoomBridge - Enhanced with nucleus intelligence
  // --------------------------------------
  function createTier4RoomBridge(iframe, wsUrl, options) {
    options = options || {};

    const bridge = {
      iframe: iframe,
      wsUrl: wsUrl,
      room: null,
      ws: null,
      eventHandlers: {},
      stateHistory: new Map(),
      nucleusConfig: NUCLEUS_CONFIG,
      isReady: false,
      isConnected: false
    };

    // Initialize room when iframe loads
    function initializeRoom() {
      try {
        const doc = iframe.contentDocument || iframe.contentWindow.document;
        const container = doc.body || doc.documentElement;

        bridge.room = new Tier4Room(container);
        bridge.isReady = true;

        bridge.emit('roomReady', bridge);

        // Setup nucleus event handlers
        bridge.setupNucleusHandlers();

        console.log('[Tier4Bridge] Room initialized with nucleus intelligence');
      } catch (e) {
        console.error('[Tier4Bridge] Failed to initialize room:', e);
        setTimeout(initializeRoom, 1000);
      }
    }

    if (iframe.readyState === 'complete' || iframe.contentDocument) {
      initializeRoom();
    } else {
      iframe.onload = initializeRoom;
    }

    // Setup WebSocket connection
    if (wsUrl) {
      bridge.connectWebSocket();
    }

    // Setup StudioBridge integration
    bridge.setupStudioBridge();

    // Bridge methods
    bridge.on = function(event, handler) {
      if (!this.eventHandlers[event]) this.eventHandlers[event] = [];
      this.eventHandlers[event].push(handler);
    };

    bridge.emit = function(event) {
      const args = Array.prototype.slice.call(arguments, 1);
      if (this.eventHandlers[event]) {
        this.eventHandlers[event].forEach(function(handler) {
          try { handler.apply(null, args); } catch (e) { console.error('[Bridge] Event handler error:', e); }
        });
      }
    };

    bridge.applyOperator = function(operator, context) {
      if (!this.room) return null;
      const result = this.room.applyOperator(operator, context);
      this.emit('operatorApplied', operator, this.room.eventLog[this.room.eventLog.length - 1].previousState, result);
      return result;
    };

    bridge.applyMacro = function(macro) {
      if (!this.room) return;
      this.room.applyMacro(macro);
      this.emit('macroApplied', macro);
    };

    bridge.triggerNucleusEvent = function(role) {
      if (!this.room) return null;
      return this.room.triggerNucleusEvent(role);
    };

    bridge.processAIBotMessage = function(message, messageType) {
      if (!this.room) return null;
      return this.room.processAIBotMessage(message, messageType);
    };

    bridge.processLibrarianData = function(librarian, dataType, data) {
      if (!this.room) return null;
      return this.room.processLibrarianData(librarian, dataType, data);
    };

    bridge.addPanel = function(config) {
      if (!this.room) return null;
      return this.room.addPanel(config);
    };

    bridge.toast = function(message) {
      if (!this.room) return;
      this.room.toast(message);
    };

    bridge.saveSnapshot = function(id) {
      if (!this.room) return null;
      const snapshotId = this.room.saveSnapshot(id);
      this.stateHistory.set(snapshotId, this.room.snapshots.get(snapshotId));
      return snapshotId;
    };

    bridge.loadSnapshot = function(id) {
      if (!this.room) return null;
      const state = this.room.loadSnapshot(id);
      if (state) this.emit('stateLoaded', state, id);
      return state;
    };

    bridge.getState = function() {
      return this.room ? this.room.state : null;
    };

    bridge.getStateHistory = function() {
      return this.stateHistory;
    };

    bridge.setupNucleusHandlers = function() {
      const self = this;

      // Listen for nucleus events from StudioBridge
      StudioBridge.onBus('nucleus.aiBot', function(msg) {
        self.processAIBotMessage(msg.message, msg.messageType);
      });

      StudioBridge.onBus('nucleus.librarian', function(msg) {
        self.processLibrarianData(msg.librarian, msg.dataType, msg.data);
      });

      StudioBridge.onBus('nucleus.trigger', function(msg) {
        self.triggerNucleusEvent(msg.role);
      });
    };

    bridge.connectWebSocket = function() {
      const self = this;

      try {
        this.ws = new WebSocket(this.wsUrl);

        this.ws.onopen = function() {
          self.isConnected = true;
          self.emit('connectionStatus', true);

          // Send nucleus system identification
          self.ws.send(JSON.stringify({
            type: 'tier4_join_session',
            sessionId: 'nucleus_session',
            userId: 'nucleus_user',
            clientType: 'nucleus_bridge',
            nucleusVersion: '1.0.0'
          }));
        };

        this.ws.onmessage = function(event) {
          try {
            const data = JSON.parse(event.data);
            self.handleWebSocketMessage(data);
          } catch (e) {
            console.error('[Bridge] Invalid WebSocket message:', e);
          }
        };

        this.ws.onclose = function() {
          self.isConnected = false;
          self.emit('connectionStatus', false);

          // Attempt reconnection
          setTimeout(function() {
            if (!self.isConnected) self.connectWebSocket();
          }, 5000);
        };

        this.ws.onerror = function(error) {
          console.error('[Bridge] WebSocket error:', error);
        };

      } catch (e) {
        console.error('[Bridge] Failed to connect WebSocket:', e);
      }
    };

    bridge.handleWebSocketMessage = function(data) {
      switch (data.type) {
      case 'tier4_operator_applied':
        if (data.operator && this.room) {
          this.applyOperator(data.operator, { source: 'websocket', ...data });
        }
        break;

      case 'nucleus_exec':
        if (data.role && this.room) {
          this.triggerNucleusEvent(data.role);
        }
        break;

      case 'ai_bot_message':
        if (this.room) {
          this.processAIBotMessage(data.message, data.messageType);
        }
        break;

      case 'librarian_data':
        if (this.room) {
          this.processLibrarianData(data.librarian, data.dataType, data.data);
        }
        break;
      }
    };

    bridge.setupStudioBridge = function() {
      const self = this;

      // Listen for Tier-4 events
      StudioBridge.onBus('tier4.applyOperator', function(msg) {
        if (msg.operator) self.applyOperator(msg.operator, { source: 'studio_bridge' });
      });

      StudioBridge.onBus('tier4.applyMacro', function(msg) {
        if (msg.macro) self.applyMacro(msg.macro);
      });

      StudioBridge.onBus('tier4.setState', function(msg) {
        if (msg.state && self.room) {
          self.room.state = Object.assign({}, msg.state);
          self.room.updateNucleusStatus();
        }
      });
    };

    return bridge;
  }

  // --------------------------------------
  // Tier4EnhancedRelay (Node.js only)
  // Enhanced with nucleus intelligence
  // --------------------------------------
  let Tier4EnhancedRelay;

  if (typeof require !== 'undefined') {
    const WebSocket = require('ws');
    const readline = require('readline');
    const fs = require('fs');
    const path = require('path');

    Tier4EnhancedRelay = function(port) {
      port = port || 9000;

      this.port = port;
      this.clients = new Map();
      this.sessions = new Map();
      this.eventBuffer = [];
      this.maxBufferSize = 1000;
      this.nucleusConfig = NUCLEUS_CONFIG;

      // Enhanced nucleus mappings
      this.nucleusToOperator = new Map([
        ['VIBRATE', 'ST'],
        ['OPTIMIZATION', 'UP'],
        ['STATE', 'CV'],
        ['SEED', 'RB']
      ]);

      this.setupWebSocketServer();
      this.setupStdinReader();
      this.setupPeriodicSave();
    };

    Tier4EnhancedRelay.prototype.setupWebSocketServer = function() {
      const self = this;

      this.wss = new WebSocket.Server({
        port: this.port,
        clientTracking: true
      });

      this.wss.on('connection', function(ws, request) {
        const clientId = self.generateClientId();
        const clientInfo = {
          id: clientId,
          connectedAt: new Date().toISOString(),
          lastSeen: new Date().toISOString(),
          tier4Session: null,
          userId: null,
          ip: request.socket.remoteAddress,
          nucleusCapable: false
        };

        self.clients.set(ws, clientInfo);

        // Send enhanced welcome with nucleus capabilities
        self.sendToClient(ws, {
          type: 'tier4_welcome',
          clientId: clientId,
          bufferSize: self.eventBuffer.length,
          availableSessions: Array.from(self.sessions.keys()),
          nucleusConfig: self.nucleusConfig,
          ts: new Date().toISOString()
        });

        ws.on('message', function(message) {
          try {
            const data = JSON.parse(message.toString());
            self.handleClientMessage(ws, data);
          } catch (error) {
            console.error('Invalid JSON from client:', error.message);
          }
        });

        ws.on('close', function() {
          self.handleClientDisconnect(ws);
        });

        self.logEvent({
          type: 'nucleus_client_connected',
          clientId: clientId,
          ts: new Date().toISOString(),
          totalClients: self.clients.size
        });
      });

      this.logEvent({
        type: 'nucleus_relay_status',
        ts: new Date().toISOString(),
        port: this.port,
        status: 'listening',
        features: ['nucleus_intelligence', 'ai_bot_routing', 'librarian_data', 'collaborative_sessions']
      });
    };

    Tier4EnhancedRelay.prototype.handleClientMessage = function(ws, message) {
      const clientInfo = this.clients.get(ws);
      clientInfo.lastSeen = new Date().toISOString();

      switch (message.type) {
      case 'nucleus_ai_bot':
        this.handleAIBotMessage(ws, message);
        break;

      case 'nucleus_librarian':
        this.handleLibrarianData(ws, message);
        break;

      case 'nucleus_trigger':
        this.handleNucleusTrigger(ws, message);
        break;

      case 'tier4_join_session':
        if (message.clientType === 'nucleus_bridge') {
          clientInfo.nucleusCapable = true;
        }
        this.joinClientToSession(ws, message.sessionId, message.userId);
        break;

      default:
        this.broadcastToOtherClients(ws, message);
      }
    };

    Tier4EnhancedRelay.prototype.handleAIBotMessage = function(ws, message) {
      const nucleusRole = this.nucleusConfig.aiBot.routing[message.messageType];
      if (nucleusRole) {
        const operator = this.nucleusConfig.operators[nucleusRole];

        const enhancedMessage = {
          ...message,
          type: 'ai_bot_processed',
          nucleusRole: nucleusRole,
          operator: operator,
          processed: true,
          ts: new Date().toISOString()
        };

        this.broadcastToClients(enhancedMessage);
        this.logEvent({
          type: 'nucleus_ai_bot_processed',
          messageType: message.messageType,
          nucleusRole: nucleusRole,
          operator: operator,
          ts: new Date().toISOString()
        });
      }
    };

    Tier4EnhancedRelay.prototype.handleLibrarianData = function(ws, message) {
      const nucleusRole = this.nucleusConfig.librarians.routing[message.dataType];
      if (nucleusRole) {
        const operator = this.nucleusConfig.operators[nucleusRole];

        const enhancedMessage = {
          ...message,
          type: 'librarian_processed',
          nucleusRole: nucleusRole,
          operator: operator,
          processed: true,
          ts: new Date().toISOString()
        };

        this.broadcastToClients(enhancedMessage);
        this.logEvent({
          type: 'nucleus_librarian_processed',
          librarian: message.librarian,
          dataType: message.dataType,
          nucleusRole: nucleusRole,
          operator: operator,
          ts: new Date().toISOString()
        });
      }
    };

    Tier4EnhancedRelay.prototype.handleNucleusTrigger = function(ws, message) {
      const operator = this.nucleusConfig.operators[message.role];
      if (operator) {
        const enhancedMessage = {
          ...message,
          type: 'nucleus_triggered',
          operator: operator,
          ts: new Date().toISOString()
        };

        this.broadcastToClients(enhancedMessage);
      }
    };

    // ... (rest of the relay methods remain the same)

    Tier4EnhancedRelay.prototype.generateClientId = function() {
      return `nucleus_client_${Date.now()}_${Math.random().toString(36).substr(2, 6)}`;
    };

    Tier4EnhancedRelay.prototype.logEvent = function(event) {
      console.log(JSON.stringify(event));
      this.eventBuffer.push(event);
      if (this.eventBuffer.length > this.maxBufferSize) {
        this.eventBuffer.shift();
      }
    };

    // Additional relay methods would go here...
    Tier4EnhancedRelay.prototype.sendToClient = function(ws, message) {
      if (ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify(message));
        return true;
      }
      return false;
    };

    Tier4EnhancedRelay.prototype.broadcastToClients = function(message) {
      const messageStr = JSON.stringify(message);
      let sentCount = 0;

      for (const ws of this.clients.keys()) {
        if (ws.readyState === WebSocket.OPEN) {
          ws.send(messageStr);
          sentCount++;
        }
      }

      return sentCount;
    };

    Tier4EnhancedRelay.prototype.broadcastToOtherClients = function(senderWs, message) {
      const messageStr = JSON.stringify(message);
      let sentCount = 0;

      for (const ws of this.clients.keys()) {
        if (ws !== senderWs && ws.readyState === WebSocket.OPEN) {
          ws.send(messageStr);
          sentCount++;
        }
      }

      return sentCount;
    };

    Tier4EnhancedRelay.prototype.joinClientToSession = function(ws, sessionId, userId) {
      // Session management implementation
      const clientInfo = this.clients.get(ws);
      clientInfo.tier4Session = sessionId;
      clientInfo.userId = userId;

      if (!this.sessions.has(sessionId)) {
        this.sessions.set(sessionId, {
          id: sessionId,
          createdAt: new Date().toISOString(),
          participants: new Set(),
          nucleusActive: true
        });
      }

      const session = this.sessions.get(sessionId);
      session.participants.add(userId);

      this.sendToClient(ws, {
        type: 'tier4_session_joined',
        sessionId: sessionId,
        participants: Array.from(session.participants),
        nucleusActive: session.nucleusActive,
        ts: new Date().toISOString()
      });
    };

    Tier4EnhancedRelay.prototype.handleClientDisconnect = function(ws) {
      const clientInfo = this.clients.get(ws);
      if (clientInfo) {
        this.logEvent({
          type: 'nucleus_client_disconnected',
          clientId: clientInfo.id,
          nucleusCapable: clientInfo.nucleusCapable,
          ts: new Date().toISOString()
        });
      }
      this.clients.delete(ws);
    };

    Tier4EnhancedRelay.prototype.setupStdinReader = function() {
      const self = this;
      this.rl = readline.createInterface({ input: process.stdin });

      this.rl.on('line', function(line) {
        try {
          const event = JSON.parse(line);
          self.processIncomingEvent(event);
        } catch (error) {
          self.processRawInput(line);
        }
      });
    };

    Tier4EnhancedRelay.prototype.processIncomingEvent = function(event) {
      this.eventBuffer.push(event);
      if (this.eventBuffer.length > this.maxBufferSize) {
        this.eventBuffer.shift();
      }

      const enhancedEvent = this.enhanceWithNucleusMapping(event);
      this.broadcastToClients(enhancedEvent);
    };

    Tier4EnhancedRelay.prototype.enhanceWithNucleusMapping = function(event) {
      const enhanced = { ...event };

      if (event.type === 'nucleus_exec' && event.role) {
        const operator = this.nucleusToOperator.get(event.role);
        if (operator) {
          enhanced.tier4_operator = operator;
          enhanced.tier4_mapping = `${event.role} → ${operator}`;
        }
      }

      return enhanced;
    };

    Tier4EnhancedRelay.prototype.processRawInput = function(line) {
      const rawEvent = {
        type: 'raw_input',
        content: line,
        ts: new Date().toISOString()
      };
      this.broadcastToClients(rawEvent);
    };

    Tier4EnhancedRelay.prototype.setupPeriodicSave = function() {
      const self = this;
      setInterval(function() {
        self.saveState();
      }, 30000);
    };

    Tier4EnhancedRelay.prototype.saveState = function() {
      const state = {
        timestamp: new Date().toISOString(),
        clients: this.clients.size,
        sessions: this.sessions.size,
        eventBufferSize: this.eventBuffer.length,
        nucleusActive: true
      };

      // Simple state saving (would need fs module properly configured)
      if (typeof console !== 'undefined') {
        console.log(JSON.stringify({ type: 'nucleus_state_saved', state }));
      }
    };
  }

  // --------------------------------------
  // UMD Export
  // --------------------------------------
  const exports = {
    StudioBridge: StudioBridge,
    Tier4Room: Tier4Room,
    createTier4RoomBridge: createTier4RoomBridge,
    NUCLEUS_CONFIG: NUCLEUS_CONFIG
  };

  // Node.js only exports
  if (typeof require !== 'undefined' && Tier4EnhancedRelay) {
    exports.Tier4EnhancedRelay = Tier4EnhancedRelay;
  }

  // Browser globals
  if (typeof window !== 'undefined') {
    window.WorldEngineUI = window.WorldEngineUI || {};
    Object.assign(window.WorldEngineUI, exports);
  }

  return exports;
});
