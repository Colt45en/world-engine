/*!
 * Tier-4 Unified Bundle
 *
 * One-file bundle:
 *  • Browser: EngineRoom + RoomLayout + Tier4Room + StudioBridge shim + createTier4RoomBridge
 *  • Node: Tier4EnhancedRelay + setup runner (replaces setup scripts / batch files)
 *
 * Usage:
 *   Browser:
 *     <script src="/js/tier4_unified_bundle.js"></script>
 *     // React EngineRoom expects window.WorldEngineTier4.createTier4RoomBridge(iframeEl, wsUrl, options?)
 *
 *   Node:
 *     node tier4_unified_bundle.js --setup --port 9000 --demo tier4_collaborative_demo.html
 *     node tier4_unified_bundle.js --relay --port 9000
 */
(function umd(root, factory) {
  if (typeof module === 'object' && typeof module.exports === 'object') {
    module.exports = factory('node');
  } else {
    root.WorldEngineTier4 = factory('browser');
  }
})(typeof self !== 'undefined' ? self : this, function bootstrap(env) {
  const isNode = env === 'node';
  const isBrowser = !isNode;

  // ------------------------------------------------------------
  // Shared utilities
  // ------------------------------------------------------------
  const nowISO = () => new Date().toISOString();
  const genId = (prefix = 'id') => `${prefix}_${Date.now()}_${Math.random().toString(36).slice(2, 10)}`;

  // ------------------------------------------------------------
  // Node-only implementation: Tier4EnhancedRelay & CLI setup
  // ------------------------------------------------------------
  let Tier4EnhancedRelay = null;
  let runSetup = null;

  if (isNode) {
    const fs = require('fs');
    const path = require('path');
    const cp = require('child_process');

    function ensureModule(name) {
      try {
        require.resolve(name);
        return true;
      } catch (err) {
        console.log(`📦 Installing missing module: ${name}`);
        try {
          cp.spawnSync('npm', ['install', name], { stdio: 'inherit', shell: true });
          require.resolve(name);
          return true;
        } catch (installErr) {
          console.error(`❌ Failed to install "${name}". Please run: npm install ${name}`);
          return false;
        }
      }
    }

    const hasWs = ensureModule('ws');
    const WebSocket = hasWs ? require('ws') : null;
    const readline = require('readline');

    class Relay {
      constructor(port = 9000) {
        if (!WebSocket) {
          throw new Error('Module "ws" is required to start the relay.');
        }

        this.port = Number(port) || 9000;
        this.eventBuffer = [];
        this.maxBufferSize = 1000;
        this.clients = new Map();
        this.sessions = new Map();

        this.nucleusToOperator = new Map([
          ['VIBRATE', 'ST'],
          ['OPTIMIZATION', 'UP'],
          ['STATE', 'CV'],
          ['SEED', 'RB']
        ]);

        this.tagToOperator = new Map([
          ['energy', 'CH'],
          ['refined', 'PR'],
          ['condition', 'SL'],
          ['seed', 'MD']
        ]);

        this.setupServer();
        this.setupStdin();
        this.setupPeriodicSave();
      }

      setupServer() {
        this.wss = new WebSocket.Server({ port: this.port, clientTracking: true });

        this.wss.on('connection', (ws, request) => {
          const info = {
            id: genId('client'),
            connectedAt: nowISO(),
            lastSeen: nowISO(),
            tier4Session: null,
            userId: null,
            ip: request?.socket?.remoteAddress || 'unknown'
          };

          this.clients.set(ws, info);

          this.send(ws, {
            type: 'tier4_welcome',
            clientId: info.id,
            bufferSize: this.eventBuffer.length,
            availableSessions: Array.from(this.sessions.keys()),
            ts: nowISO()
          });

          this.eventBuffer.slice(-10).forEach((ev) => this.send(ws, ev));

          ws.on('message', (msg) => {
            try {
              const data = JSON.parse(msg.toString());
              this.handleClientMessage(ws, data);
            } catch (e) {
              console.error('Invalid JSON from client:', e.message);
            }
          });

          ws.on('error', (err) => console.error('Client error:', err.message));
          ws.on('close', () => this.handleClientDisconnect(ws, info));

          this.logEvent({
            type: 'client_connected',
            clientId: info.id,
            totalClients: this.clients.size,
            ts: nowISO()
          });
        });

        this.logEvent({
          type: 'relay_status',
          status: 'listening',
          port: this.port,
          features: ['tier4_integration', 'collaboration', 'event_buffering'],
          ts: nowISO()
        });
      }

      setupStdin() {
        const rl = readline.createInterface({ input: process.stdin });
        rl.on('line', (line) => {
          try {
            const event = JSON.parse(line);
            this.processIncomingEvent(event);
          } catch (err) {
            this.processIncomingEvent({ type: 'raw_input', content: line, ts: nowISO() });
          }
        });
      }

      setupPeriodicSave() {
        setInterval(() => this.saveState(), 30_000);
      }

      saveState() {
        try {
          const state = {
            timestamp: nowISO(),
            clients: this.clients.size,
            sessions: this.sessions.size,
            eventBufferSize: this.eventBuffer.length,
            recentEvents: this.eventBuffer.slice(-10)
          };
          const dir = path.join(process.cwd(), 'backup', nowISO().split('T')[0].replace(/-/g, ''));
          fs.mkdirSync(dir, { recursive: true });
          fs.writeFileSync(path.join(dir, `relay-state-${Date.now()}.json`), JSON.stringify(state, null, 2));
        } catch (err) {
          console.warn('⚠️  Unable to save relay state:', err.message);
        }
      }

      processIncomingEvent(event) {
        this.buffer(event);
        const enhanced = this.enhanceEvent(event);
        this.broadcast(enhanced);
        this.processCollaboration(enhanced);
      }

      enhanceEvent(event) {
        const out = { ...event };

        if (event.type === 'nucleus_exec' && event.role) {
          const operator = this.nucleusToOperator.get(event.role);
          if (operator) {
            out.tier4_operator = operator;
            out.tier4_mapping = `${event.role} → ${operator}`;
          }
        }

        if (event.type === 'memory_store' && event.tag) {
          const operator = this.tagToOperator.get(event.tag);
          if (operator) {
            out.tier4_operator = operator;
            out.tier4_mapping = `${event.tag} → ${operator}`;
          }
        }

        if (event.type === 'cycle_start') {
          out.tier4_suggested_macro = this.suggestMacroForCycle(event.cycle, event.total);
        }

        if (event.type === 'loop_back') {
          out.tier4_macro = this.mapLoopBackToMacro(event.from, event.to);
        }

        return out;
      }

      suggestMacroForCycle(cycle, total) {
        if (total <= 1) return 'IDE_A';
        if (cycle === 1) return 'IDE_A';
        if (cycle === total) return 'IDE_C';
        return 'IDE_B';
      }

      mapLoopBackToMacro(from, to) {
        const mapping = {
          'seed->energy': 'IDE_A',
          'energy->refined': 'IDE_B',
          'refined->condition': 'IDE_C',
          'condition->seed': 'MERGE_ABC'
        };
        return mapping[`${from}->${to}`] || 'OPTIMIZE';
      }

      processCollaboration(event) {
        if (event.type === 'tier4_session_join') {
          this.ensureSession(event.sessionId);
        }
        if (event.type === 'tier4_session_leave') {
          const session = this.sessions.get(event.sessionId);
          if (session) {
            session.participants.delete(event.userId);
            if (session.participants.size === 0) {
              this.sessions.delete(event.sessionId);
            }
          }
        }
      }

      handleClientMessage(ws, message) {
        const info = this.clients.get(ws);
        if (!info) return;
        info.lastSeen = nowISO();

        switch (message.type) {
        case 'tier4_join_session':
          this.joinClientToSession(ws, message.sessionId, message.userId);
          break;
        case 'tier4_state_update': {
          const payload = { ...message, relayedAt: nowISO() };
          if (info.tier4Session) {
            this.broadcastToSession(info.tier4Session, payload, ws);
          } else {
            this.broadcast(payload);
          }
          break;
        }
        case 'tier4_operator_request': {
          const enhanced = { ...message, type: 'tier4_operator_applied', ts: nowISO(), processed_by_relay: true };
          if (info.tier4Session) {
            this.broadcastToSession(info.tier4Session, enhanced, ws);
          } else {
            this.broadcast(enhanced);
          }
          break;
        }
        case 'ping':
          this.send(ws, { type: 'pong', ts: nowISO() });
          break;
        default:
          this.broadcastExcept(ws, message);
        }
      }

      joinClientToSession(ws, sessionId, userId) {
        const info = this.clients.get(ws);
        info.tier4Session = sessionId;
        info.userId = userId;

        const session = this.ensureSession(sessionId);
        session.participants.add(userId);

        this.send(ws, {
          type: 'tier4_session_joined',
          sessionId,
          participants: Array.from(session.participants),
          ts: nowISO()
        });

        this.broadcastToSession(sessionId, {
          type: 'tier4_participant_joined',
          sessionId,
          userId,
          participants: Array.from(session.participants),
          ts: nowISO()
        }, ws);

        this.logEvent({
          type: 'session_join',
          sessionId,
          userId,
          totalParticipants: session.participants.size,
          ts: nowISO()
        });
      }

      ensureSession(sessionId) {
        if (!this.sessions.has(sessionId)) {
          this.sessions.set(sessionId, {
            id: sessionId,
            createdAt: nowISO(),
            participants: new Set(),
            lastState: null,
            operatorHistory: []
          });
        }
        return this.sessions.get(sessionId);
      }

      handleClientDisconnect(ws, info) {
        if (info?.tier4Session) {
          const session = this.sessions.get(info.tier4Session);
          if (session) {
            session.participants.delete(info.userId);
            this.broadcastToSession(info.tier4Session, {
              type: 'tier4_participant_left',
              sessionId: info.tier4Session,
              userId: info.userId,
              participants: Array.from(session.participants),
              ts: nowISO()
            }, ws);
            if (session.participants.size === 0) {
              this.sessions.delete(info.tier4Session);
            }
          }
        }

        this.clients.delete(ws);
        this.logEvent({
          type: 'client_disconnected',
          clientId: info?.id,
          totalClients: this.clients.size,
          ts: nowISO()
        });
      }

      broadcast(message) {
        const payload = JSON.stringify(message);
        let count = 0;
        for (const ws of this.clients.keys()) {
          if (ws.readyState === 1) {
            ws.send(payload);
            count += 1;
          }
        }
        return count;
      }

      broadcastExcept(sender, message) {
        const payload = JSON.stringify(message);
        let count = 0;
        for (const ws of this.clients.keys()) {
          if (ws !== sender && ws.readyState === 1) {
            ws.send(payload);
            count += 1;
          }
        }
        return count;
      }

      broadcastToSession(sessionId, message, exclude) {
        const payload = JSON.stringify(message);
        let count = 0;
        for (const [ws, info] of this.clients.entries()) {
          if (info.tier4Session === sessionId && ws !== exclude && ws.readyState === 1) {
            ws.send(payload);
            count += 1;
          }
        }
        return count;
      }

      send(ws, message) {
        if (ws.readyState === 1) {
          ws.send(JSON.stringify(message));
          return true;
        }
        return false;
      }

      buffer(event) {
        this.eventBuffer.push(event);
        if (this.eventBuffer.length > this.maxBufferSize) {
          this.eventBuffer.shift();
        }
      }

      logEvent(event) {
        console.log(JSON.stringify(event));
        this.buffer(event);
      }

      getStats() {
        return {
          clients: this.clients.size,
          sessions: this.sessions.size,
          eventBufferSize: this.eventBuffer.length,
          uptime: process.uptime(),
          memory: process.memoryUsage(),
          activeSessions: Array.from(this.sessions.entries()).map(([id, session]) => ({
            id,
            participants: session.participants.size,
            createdAt: session.createdAt
          }))
        };
      }
    }

    Tier4EnhancedRelay = Relay;

    function openDemo(demoPath) {
      if (!fs.existsSync(demoPath)) {
        console.log('⚠️  Demo file not found. Relay is running; connect to ws://localhost:9000');
        return;
      }

      const platform = process.platform;
      const command = platform === 'win32'
        ? `start "" "${demoPath}"`
        : platform === 'darwin'
          ? `open "${demoPath}"`
          : `xdg-open "${demoPath}"`;

      cp.exec(command, (err) => {
        if (err) {
          console.log('⚠️  Could not open demo automatically.');
          console.log(`Please open: ${demoPath}`);
        } else {
          console.log('🎨 Demo opened in browser');
        }
      });
    }

    function printRelayGuide() {
      console.log('\n📖 Integration Guide');
      console.log('1) Nucleus operators: VIBRATE→ST, OPTIMIZATION→UP, STATE→CV, SEED→RB');
      console.log('2) Memory tags: energy→CH, refined→PR, condition→SL, seed→MD');
      console.log('3) Cycle macros: cycle_start → IDE_A/IDE_B/IDE_C based on position');
      console.log('4) Collaboration: sessions, participants, real-time operator broadcast');
      console.log('5) Pipe NDJSON into stdin to rebroadcast enriched events.\n');
    }

    runSetup = function runSetup({ port = 9000, demoFile = 'tier4_collaborative_demo.html', autoOpen = true } = {}) {
      console.log('🚀 Tier-4 WebSocket Integration Setup');
      console.log('=====================================');

      const relay = new Tier4EnhancedRelay(port);

      setTimeout(() => {
        console.log(`\n✅ Relay running on ws://localhost:${port}`);
        console.log(`🌐 Demo file: ${path.resolve(demoFile)}`);
        if (autoOpen) openDemo(path.resolve(demoFile));
        printRelayGuide();
        console.log('\nPress Ctrl+C to stop.');
      }, 1200);

      process.on('SIGINT', () => {
        console.log('\n🔄 Shutting down...');
        console.log(JSON.stringify({ type: 'relay_shutdown', ts: nowISO(), stats: relay.getStats() }));
        process.exit(0);
      });
    };

    if (require.main === module) {
      const args = process.argv.slice(2);
      const hasSetup = args.includes('--setup');
      const hasRelay = args.includes('--relay') || (!hasSetup && args.length === 0);

      const portIndex = args.indexOf('--port');
      const portArg = portIndex !== -1 ? Number(args[portIndex + 1]) : 9000;

      const demoIndex = args.indexOf('--demo');
      const demoArg = demoIndex !== -1 ? args[demoIndex + 1] : 'tier4_collaborative_demo.html';

      if (hasSetup) {
        runSetup({ port: portArg, demoFile: demoArg, autoOpen: true });
      } else if (hasRelay) {
        console.log('🌐 Starting Tier-4 WebSocket Relay...');
        new Tier4EnhancedRelay(portArg);
        printRelayGuide();
        console.log(`\nws://localhost:${portArg} ready. Ctrl+C to exit.`);
      }
    }
  }

  // ------------------------------------------------------------
  // Browser implementation: StudioBridge shim + room classes + bridge
  // ------------------------------------------------------------
  let EngineRoom = null;
  let RoomLayout = null;
  let Tier4Room = null;
  let createTier4RoomBridge = null;

  if (isBrowser) {
    (function ensureStudioBridge() {
      if (window.StudioBridge) return;

      const busName = 'studio-bus';
      const bc = typeof BroadcastChannel !== 'undefined' ? new BroadcastChannel(busName) : null;
      const listeners = [];

      function invokeListeners(message) {
        listeners.forEach((entry) => {
          try {
            if (!entry.type || entry.type === message.type) {
              entry.fn(message);
            }
          } catch (err) {
            console.error('[StudioBridge shim] listener error', err);
          }
        });
      }

      if (bc) {
        bc.onmessage = (event) => invokeListeners(event.data);
      } else {
        window.addEventListener('studio:msg', (event) => invokeListeners(event.detail));
      }

      function onBus(typeOrFn, maybeFn) {
        if (typeof typeOrFn === 'function') {
          listeners.push({ type: null, fn: typeOrFn });
        } else if (typeof maybeFn === 'function') {
          listeners.push({ type: typeOrFn, fn: maybeFn });
        }
      }

      function sendBus(message) {
        if (bc) {
          bc.postMessage(message);
        } else {
          window.dispatchEvent(new CustomEvent('studio:msg', { detail: message }));
        }
      }

      window.StudioBridge = {
        onBus,
        sendBus,
        RecorderAPI: window.RecorderAPI || {},
        EngineAPI: window.EngineAPI || {},
        ChatAPI: window.ChatAPI || {},
        Store: {
          async save(key, value) {
            if (window.localStorage) {
              window.localStorage.setItem(key, JSON.stringify(value));
            }
            return value;
          },
          async load(key) {
            if (!window.localStorage) return null;
            const raw = window.localStorage.getItem(key);
            return raw ? JSON.parse(raw) : null;
          }
        },
        Utils: {
          generateId: () => `${Date.now()}_${Math.random().toString(36).slice(2)}`,
          parseCommand: (line) => ({ type: 'run', args: String(line || '') }),
          log: (message, level = 'log') => {
            console[level]('[StudioBridge shim]', message);
          }
        },
        setupEngineTransport(frame) {
          function isSameOrigin() {
            try {
              return frame.contentWindow.document != null;
            } catch (err) {
              return false;
            }
          }
          return {
            isSameOrigin,
            withEngine(fn) {
              if (isSameOrigin()) {
                fn(frame.contentWindow.document);
              }
            }
          };
        }
      };
    })();

    (function initRoomComponents() {
      class BaseEngineRoom {
        constructor(targetWindow, origin = '*') {
          this.target = targetWindow;
          this.origin = origin;
          this.eventHandlers = new Map();
          this.isReady = false;
          this.readyCallbacks = [];

          window.addEventListener('message', (event) => this.handleMessage(event));
        }

        handleMessage(event) {
          const data = event.data;
          if (!data || data.room !== 'gridroom') return;

          if (data.type === 'ROOM_READY') {
            this.isReady = true;
            this.readyCallbacks.forEach((cb) => cb());
            this.readyCallbacks = [];
          }

          const handler = this.eventHandlers.get(data.type);
          if (handler) handler(data.payload || {});
        }

        send(message) {
          this.target.postMessage({ room: 'gridroom', ...message }, this.origin);
        }

        whenReady(cb) {
          if (this.isReady) cb();
          else this.readyCallbacks.push(cb);
        }

        on(eventType, handler) {
          this.eventHandlers.set(eventType, handler);
        }

        init(sessionId, title) {
          this.send({ type: 'INIT', payload: { sessionId, title } });
        }

        publishSnapshot(cid, state, parentCid) {
          this.send({
            type: 'SNAPSHOT',
            payload: { cid, state, parentCid: parentCid ?? null, timestamp: Date.now() }
          });
        }

        publishEvent(event) {
          this.send({ type: 'EVENT', payload: { ...event, timestamp: event.timestamp || Date.now() } });
        }

        publishOperatorEvent(operator, inputCid, outputCid, meta) {
          this.publishEvent({
            id: genId('op'),
            button: `Tier4_${operator}`,
            inputCid,
            outputCid,
            timestamp: Date.now(),
            meta
          });
        }

        toast(message) {
          this.send({ type: 'TOAST', payload: { msg: message } });
        }

        addPanel(config) {
          this.send({ type: 'ADD_PANEL', payload: config });
        }

        setPanelHTML(id, html) {
          this.send({ type: 'SET_PANEL_HTML', payload: { panelId: id, html } });
        }

        removePanel(id) {
          this.send({ type: 'REMOVE_PANEL', payload: { panelId: id } });
        }

        focusPanel(id) {
          this.send({ type: 'FOCUS_PANEL', payload: { panelId: id } });
        }

        getState() {
          this.send({ type: 'REQUEST', payload: { op: 'GET_STATE' } });
        }

        listPanels() {
          this.send({ type: 'REQUEST', payload: { op: 'LIST_PANELS' } });
        }

        loadSession(events, snapshots, currentCid) {
          this.send({ type: 'LOAD_SESSION', payload: { events, snapshots, currentCid } });
        }
      }

      class Layout {
        constructor(roomWidth = 1400, roomHeight = 800) {
          this.width = roomWidth;
          this.height = roomHeight;
        }

        getPosition(semantic) {
          const w = this.width;
          const h = this.height;
          const map = {
            events: { wall: 'left', x: 60, y: 260, w: 420, h: 260 },
            snapshots: { wall: 'right', x: w - 480, y: 260, w: 420, h: 260 },
            nucleus: { wall: 'front', x: w / 2 - 200, y: 50, w: 400, h: 300 },
            operators: { wall: 'front', x: w / 2 - 300, y: 380, w: 600, h: 200 },
            metrics: { wall: 'floor', x: w / 2 - 240, y: 180, w: 480, h: 220 },
            docs: { wall: 'back', x: 100, y: 100, w: 500, h: 400 },
            health: { wall: 'ceil', x: w - 350, y: 50, w: 300, h: 150 }
          };
          return map[semantic];
        }
      }

      class Tier4EngineRoom extends BaseEngineRoom {
        constructor(targetWindow, sessionId, origin = '*') {
          super(targetWindow, origin);
          this.sessionId = sessionId || `tier4_${Date.now()}`;
          this.layout = new Layout();
          this.stateHistory = new Map();
          this.currentCid = null;

          this.on('REQUEST_LOAD_CID', (payload) => this.loadByCid(payload.cid));

          this.whenReady(() => {
            this.init(this.sessionId, 'Tier-4 Engine Room');
            setTimeout(() => this.createPanels(), 300);
          });
        }

        createPanels() {
          const panels = [
            { semantic: 'operators', title: 'Tier-4 Operators', html: this.operatorHTML() },
            { semantic: 'metrics', title: 'Performance Metrics', html: this.metricsHTML() },
            { semantic: 'docs', title: 'Tier-4 Documentation', html: this.docsHTML() },
            { semantic: 'health', title: 'System Health', html: this.healthHTML() }
          ];

          panels.forEach((panel) => {
            const pos = this.layout.getPosition(panel.semantic);
            if (pos) {
              this.addPanel({ ...pos, title: panel.title, html: panel.html });
            }
          });
        }

        applyOperator(operator, currentState, newState) {
          const inputCid = this.makeCid(currentState);
          const outputCid = this.makeCid(newState);

          this.storeState(inputCid, currentState);
          this.storeState(outputCid, newState, inputCid);
          this.publishOperatorEvent(operator, inputCid, outputCid, {
            deltaKappa: newState.kappa - currentState.kappa,
            levelChange: newState.level - currentState.level,
            vectorDelta: newState.x.map((v, i) => v - (currentState.x[i] || 0))
          });

          this.currentCid = outputCid;
          this.publishSnapshot(outputCid, newState, inputCid);
        }

        storeState(cid, state, parentCid) {
          this.stateHistory.set(cid, {
            cid,
            state: { ...state },
            parentCid,
            timestamp: Date.now()
          });
        }

        makeCid(state) {
          const normalized = {
            x: state.x.map((v) => Math.round(v * 1000) / 1000),
            kappa: Math.round(state.kappa * 1000) / 1000,
            level: state.level
          };
          return `tier4_${btoa(JSON.stringify(normalized)).replace(/[+/=]/g, '').slice(0, 16)}`;
        }

        loadByCid(cid) {
          const snapshot = this.stateHistory.get(cid);
          if (snapshot) {
            this.currentCid = cid;
            window.dispatchEvent(new CustomEvent('tier4-load-state', { detail: { state: snapshot.state, cid } }));
            this.toast(`Loaded state ${cid.slice(0, 12)}...`);
          }
        }

        getCurrentCid() {
          return this.currentCid;
        }

        getStateHistory() {
          return new Map(this.stateHistory);
        }

        operatorHTML() {
          return `<!doctype html><meta charset="utf-8"><style>
            body{margin:0;background:#0b1418;color:#cfe;font:14px system-ui;padding:16px}
            .grid{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin:16px 0}
            .btn{background:#1a2332;border:1px solid #2a3548;border-radius:8px;padding:12px;text-align:center;cursor:pointer;transition:.2s}
            .btn:hover{background:#243344;border-color:#54f0b8}
            .macro{background:#2d1a33;border-color:#8a4a9f}.macro:hover{background:#3a2440;border-color:#b366d9}
          </style>
          <h3>Tier-4 Operators</h3>
          <div class="grid">
            ${['ST','UP','PR','CV','RB','RS'].map((op) => `<div class="btn" onclick="parent.postMessage({type:'apply-operator',operator:'${op}'},'*')">${op}</div>`).join('')}
          </div>
          <h3>Three Ides Macros</h3>
          <div class="grid">
            ${['IDE_A','IDE_B','MERGE_ABC'].map((macro) => `<div class="btn macro" onclick="parent.postMessage({type:'apply-macro',macro:'${macro}'},'*')">${macro}</div>`).join('')}
          </div>`;
        }

        metricsHTML() {
          return `<!doctype html><meta charset="utf-8"><style>
            body{margin:0;background:#0b1418;color:#cfe;font:12px system-ui;padding:12px}
            .metric{display:flex;justify-content:space-between;padding:8px;background:#1a2332;border-radius:4px;margin:4px 0}
            .chart{height:60px;background:#06101c;border:1px solid #2a3548;border-radius:4px;margin:8px 0}
          </style>
          <h3>Performance Metrics</h3>
          <div class="metric">Operations/sec: <span id="ops-rate">--</span></div>
          <div class="metric">Avg κ Change: <span id="kappa-delta">--</span></div>
          <div class="metric">State Transitions: <span id="transitions">--</span></div>
          <div class="metric">Memory Usage: <span id="memory">--</span></div>
          <canvas class="chart" id="perf-chart"></canvas>
          <script>
            const chart = document.getElementById('perf-chart');
            const ctx = chart.getContext('2d');
            chart.width = chart.offsetWidth; chart.height = 60;
            let dataPoints = [];
            function updateChart(v){
              dataPoints.push(v);
              if(dataPoints.length>50) dataPoints.shift();
              ctx.clearRect(0,0,chart.width,chart.height);
              ctx.strokeStyle='#54f0b8';
              ctx.lineWidth=2;
              ctx.beginPath();
              dataPoints.forEach((p,i)=>{
                const x=(i/(dataPoints.length-1))*chart.width;
                const y=chart.height-(p*chart.height);
                if(i===0) ctx.moveTo(x,y);
                else ctx.lineTo(x,y);
              });
              ctx.stroke();
            }
            window.addEventListener('message',(e)=>{
              if(e.data.type==='update-metrics'){
                const m=e.data.metrics;
                document.getElementById('ops-rate').textContent=m.opsRate||'--';
                document.getElementById('kappa-delta').textContent=m.kappaDelta||'--';
                document.getElementById('transitions').textContent=m.transitions||'--';
                document.getElementById('memory').textContent=m.memory||'--';
                if(typeof m.chartValue==='number') updateChart(m.chartValue);
              }
            });
          </script>`;
        }

        docsHTML() {
          return `<!doctype html><meta charset="utf-8"><style>
            body{margin:0;background:#0b1418;color:#cfe;font:13px system-ui;padding:16px;line-height:1.5}
            h3{color:#54f0b8;margin-top:0}
            .op{background:#1a2332;padding:8px;border-radius:4px;margin:8px 0}
            .name{color:#9fd6ff;font-weight:600}
            code{background:#0a1222;padding:2px 6px;border-radius:3px;color:#ff9f43}
          </style>
          <h3>Tier-4 Meta System</h3>
          <p>State vector <code>[p,i,g,c]</code> with κ as global confidence. Operators: ST, UP, PR, CV, RB, RS; Macros: IDE_A, IDE_B, MERGE_ABC.</p>
          <div class="op"><div class="name">ST</div> Stabilize (dampen oscillations).</div>
          <div class="op"><div class="name">UP</div> Update (raise information & certainty).</div>
          <div class="op"><div class="name">PR</div> Progress (goal alignment).</div>
          <div class="op"><div class="name">CV</div> Converge (move toward optimum).</div>
          <div class="op"><div class="name">RB</div> Rollback (restore prior state).</div>`;
        }

        healthHTML() {
          return `<!doctype html><meta charset="utf-8"><style>
            body{margin:0;background:#0b1418;color:#cfe;font:11px system-ui;padding:8px}
            .item{display:flex;justify-content:space-between;align-items:center;padding:4px 0}
            .status{width:8px;height:8px;border-radius:50%;background:#18c08f}
            .warn{background:#e5c558}
            .err{background:#ff4d4f}
          </style>
          <h4>System Health</h4>
          <div class="item"><span>WebSocket</span><div class="status" id="ws"></div></div>
          <div class="item"><span>FPS</span><span id="fps">--</span></div>
          <div class="item"><span>Latency</span><span id="lat">--ms</span></div>
          <script>
            setInterval(()=>{
              document.getElementById('fps').textContent = Math.floor(Math.random()*10+55);
              document.getElementById('lat').textContent = Math.floor(Math.random()*20+5)+'ms';
            },2000);
          </script>`;
        }
      }

      EngineRoom = BaseEngineRoom;
      RoomLayout = Layout;
      Tier4Room = Tier4EngineRoom;
    })();

    (function initBridgeFactory() {
      function defaultTransform(operator, state) {
        const clamp = (v) => Math.max(0, Math.min(1, v));
        const next = { ...state, x: state.x.slice() };
        switch (operator) {
        case 'ST':
          next.kappa = clamp(next.kappa * 0.99);
          break;
        case 'UP':
          next.x = next.x.map((value, idx) => (idx === 1 || idx === 3 ? clamp(value + 0.02) : value));
          next.kappa = clamp(next.kappa + 0.01);
          next.level = (next.level || 0) + 1;
          break;
        case 'PR':
          next.x = next.x.map((value, idx) => (idx === 2 ? clamp(value + 0.03) : value));
          break;
        case 'CV':
          next.x = next.x.map((value) => clamp(value * 0.98));
          next.kappa = clamp(next.kappa + 0.005);
          break;
        case 'RB':
          return { x: [0, 0.5, 0.4, 0.6], kappa: 0.6, level: Math.max(0, (state.level || 0) - 1) };
        case 'RS':
          return { x: [0, 0.5, 0.4, 0.6], kappa: 0.6, level: 0 };
        default:
          return state;
        }
        return next;
      }

      createTier4RoomBridge = function createTier4RoomBridge(iframe, wsUrl = 'ws://localhost:9000', options = {}) {
        if (!iframe || !iframe.contentWindow) {
          throw new Error('Iframe element with accessible contentWindow is required.');
        }

        const { transform = defaultTransform } = options;
        const room = new Tier4Room(iframe.contentWindow, `sess_${Date.now()}`);
        let state = { x: [0, 0.5, 0.4, 0.6], kappa: 0.6, level: 0 };
        let ws = null;
        let wsConnected = false;

        window.addEventListener('message', (event) => {
          if (event.source !== iframe.contentWindow) return;
          if (event.data?.type === 'apply-operator') applyOperator(event.data.operator, { source: 'panel' });
          if (event.data?.type === 'apply-macro') applyMacro(event.data.macro);
        });

        if (window.StudioBridge) {
          window.StudioBridge.onBus((msg) => {
            if (!msg || typeof msg !== 'object') return;
            if (msg.type === 'tier4.applyOperator') applyOperator(msg.operator, msg.meta || {});
            if (msg.type === 'tier4.applyMacro') applyMacro(msg.macro);
            if (msg.type === 'tier4.setState' && msg.state) setState(msg.state);
          });
        }

        function connectWebSocket() {
          try {
            ws = new WebSocket(wsUrl);
            ws.onopen = () => {
              wsConnected = true;
              notifyConnection(true);
              ws.send(JSON.stringify({
                type: 'tier4_join_session',
                sessionId: 'nucleus_session',
                userId: 'nucleus_user',
                clientType: 'nucleus_bridge',
                nucleusVersion: '1.0.0'
              }));
            };
            ws.onclose = () => {
              wsConnected = false;
              notifyConnection(false);
              setTimeout(connectWebSocket, 1500);
            };
            ws.onerror = () => {};
            ws.onmessage = (event) => {
              try {
                const data = JSON.parse(event.data);
                if (data.type === 'tier4_operator_applied' && data.operator) {
                  applyOperator(data.operator, { source: 'websocket', relay: true });
                }
                if (data.type === 'nucleus_exec' && data.role) {
                  triggerNucleus(data.role);
                }
              } catch (err) {
                console.warn('Invalid WebSocket payload:', err.message);
              }
            };
          } catch (err) {
            console.warn('WebSocket connection failed:', err.message);
          }
        }

        if (typeof WebSocket !== 'undefined') {
          connectWebSocket();
        }

        function notifyConnection(connected) {
          window.dispatchEvent(new CustomEvent('tier4-connection', { detail: { connected } }));
        }

        function emitOperator(operator, previousState, nextState) {
          const detail = { operator, previousState, newState: nextState };
          window.dispatchEvent(new CustomEvent('tier4-operator-applied', { detail }));
          if (window.StudioBridge) {
            window.StudioBridge.sendBus({ type: 'tier4.operatorApplied', ...detail });
          }
        }

        function applyOperator(operator, meta = {}) {
          const previousState = { ...state };
          const nextState = transform(operator, previousState, meta);
          room.applyOperator(operator, previousState, nextState);
          state = nextState;
          emitOperator(operator, previousState, nextState);

          if (wsConnected && ws && ws.readyState === 1 && !meta.relay) {
            ws.send(JSON.stringify({
              type: 'tier4_operator_request',
              operator,
              meta,
              state: nextState,
              ts: nowISO()
            }));
          }
        }

        function applyMacro(macro) {
          const sequence = macro === 'IDE_A'
            ? ['ST', 'UP', 'PR']
            : macro === 'IDE_B'
              ? ['CV', 'PR', 'UP']
              : macro === 'MERGE_ABC'
                ? ['ST', 'UP', 'CV']
                : ['UP'];
          sequence.forEach((operator, index) => {
            setTimeout(() => applyOperator(operator, { macro }), index * 120);
          });
        }

        function triggerNucleus(role) {
          if (!role) return;
          const operator = {
            VIBRATE: 'ST',
            OPTIMIZATION: 'UP',
            STATE: 'CV',
            SEED: 'RB'
          }[role];
          if (operator) applyOperator(operator, { source: 'nucleus', role });
        }

        function setState(nextState) {
          const previousState = state;
          state = { ...nextState };
          room.applyOperator('SET', previousState, state);
          emitOperator('SET', previousState, state);
        }

        return {
          applyOperator,
          applyMacro,
          triggerNucleus,
          setState,
          getCurrentState: () => ({ ...state }),
          getRoom: () => room,
          isWebSocketConnected: () => wsConnected,
          disconnect: () => {
            try {
              ws?.close();
            } catch (err) {
              console.warn('Failed to close WebSocket:', err.message);
            }
          }
        };
      };

      if (!window.WorldEngineTier4) {
        window.WorldEngineTier4 = {};
      }

      window.WorldEngineTier4.createTier4RoomBridge = createTier4RoomBridge;
      window.WorldEngineTier4.EngineRoom = EngineRoom;
      window.WorldEngineTier4.RoomLayout = RoomLayout;
      window.WorldEngineTier4.Tier4Room = Tier4Room;
    })();
  }

  // ------------------------------------------------------------
  // Public API
  // ------------------------------------------------------------
  return {
    createTier4RoomBridge: isBrowser ? createTier4RoomBridge : undefined,
    EngineRoom: isBrowser ? EngineRoom : undefined,
    RoomLayout: isBrowser ? RoomLayout : undefined,
    Tier4Room: isBrowser ? Tier4Room : undefined,
    Tier4EnhancedRelay: isNode ? Tier4EnhancedRelay : undefined,
    runSetup: isNode ? runSetup : undefined
  };
});
