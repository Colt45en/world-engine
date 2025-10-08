// Enhanced WebSocket Relay with Tier-4 Integration
// Usage: .\kernel_step3.exe | node tier4_ws_relay.js
// Now supports: NDJSON → Tier-4 operators + collaborative sessions

const WebSocket = require("ws");
const readline = require("readline");
const fs = require("fs");
const path = require("path");

class Tier4EnhancedRelay {
  constructor(port = 9000) {
    this.port = port;
    this.clients = new Map(); // WebSocket -> clientInfo
    this.sessions = new Map(); // sessionId -> session data
    this.eventBuffer = [];
    this.maxBufferSize = 1000;

    // Tier-4 specific mappings
    this.nucleusToOperator = new Map([
      ['VIBRATE', 'ST'],      // Vibration → Snapshot
      ['OPTIMIZATION', 'UP'],  // Optimization → Update
      ['STATE', 'CV'],         // State → Convert
      ['SEED', 'RB']          // Seed → Rebuild
    ]);

    this.tagToOperator = new Map([
      ['energy', 'CH'],       // Energy → Channel
      ['refined', 'PR'],      // Refined → Prevent
      ['condition', 'SL'],    // Condition → Select
      ['seed', 'MD']         // Seed → Module
    ]);

    this.setupWebSocketServer();
    this.setupStdinReader();
    this.setupPeriodicSave();
  }

  setupWebSocketServer() {
    this.wss = new WebSocket.Server({
      port: this.port,
      clientTracking: true
    });

    this.wss.on("connection", (ws, request) => {
      const clientId = this.generateClientId();
      const clientInfo = {
        id: clientId,
        connectedAt: new Date().toISOString(),
        lastSeen: new Date().toISOString(),
        tier4Session: null,
        userId: null,
        ip: request.socket.remoteAddress
      };

      this.clients.set(ws, clientInfo);

      // Send welcome message with current buffer
      this.sendToClient(ws, {
        type: "tier4_welcome",
        clientId: clientId,
        bufferSize: this.eventBuffer.length,
        availableSessions: Array.from(this.sessions.keys()),
        ts: new Date().toISOString()
      });

      // Send recent events to new client
      if (this.eventBuffer.length > 0) {
        const recentEvents = this.eventBuffer.slice(-10);
        recentEvents.forEach(event => {
          if (ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify(event));
          }
        });
      }

      ws.on("message", (message) => {
        try {
          const data = JSON.parse(message.toString());
          this.handleClientMessage(ws, data);
        } catch (error) {
          console.error("Invalid JSON from client:", error.message);
        }
      });

      ws.on("close", () => {
        this.handleClientDisconnect(ws);
      });

      ws.on("error", (error) => {
        console.error("Client error:", error.message);
      });

      this.logEvent({
        type: "client_connected",
        clientId: clientId,
        ts: new Date().toISOString(),
        totalClients: this.clients.size
      });
    });

    this.logEvent({
      type: "relay_status",
      ts: new Date().toISOString(),
      port: this.port,
      status: "listening",
      features: ["tier4_integration", "collaborative_sessions", "event_buffering"]
    });
  }

  setupStdinReader() {
    this.rl = readline.createInterface({ input: process.stdin });

    this.rl.on("line", (line) => {
      try {
        const event = JSON.parse(line);
        this.processIncomingEvent(event);
      } catch (error) {
        // Not JSON, treat as raw text
        this.processRawInput(line);
      }
    });
  }

  processIncomingEvent(event) {
    // Add to buffer
    this.eventBuffer.push(event);
    if (this.eventBuffer.length > this.maxBufferSize) {
      this.eventBuffer.shift(); // Remove oldest
    }

    // Enhance with Tier-4 mappings
    const enhancedEvent = this.enhanceWithTier4Mapping(event);

    // Broadcast to all clients
    this.broadcastToClients(enhancedEvent);

    // Process collaborative aspects
    this.processCollaborativeEvent(enhancedEvent);
  }

  enhanceWithTier4Mapping(event) {
    const enhanced = { ...event };

    // Add Tier-4 operator mappings
    if (event.type === "nucleus_exec" && event.role) {
      const operator = this.nucleusToOperator.get(event.role);
      if (operator) {
        enhanced.tier4_operator = operator;
        enhanced.tier4_mapping = `${event.role} → ${operator}`;
      }
    }

    if (event.type === "memory_store" && event.tag) {
      const operator = this.tagToOperator.get(event.tag);
      if (operator) {
        enhanced.tier4_operator = operator;
        enhanced.tier4_mapping = `${event.tag} → ${operator}`;
      }
    }

    // Add cycle-based macro suggestions
    if (event.type === "cycle_start") {
      enhanced.tier4_suggested_macro = this.suggestMacroForCycle(event.cycle, event.total);
    }

    if (event.type === "loop_back") {
      enhanced.tier4_macro = this.mapLoopBackToMacro(event.from, event.to);
    }

    return enhanced;
  }

  suggestMacroForCycle(cycle, total) {
    if (total <= 1) return "IDE_A"; // Single cycle → Analysis
    if (cycle === 1) return "IDE_A"; // First cycle → Analysis
    if (cycle === total) return "IDE_C"; // Last cycle → Build
    return "IDE_B"; // Middle cycles → Constraints
  }

  mapLoopBackToMacro(from, to) {
    const mapping = {
      "seed->energy": "IDE_A",    // Analysis path
      "energy->refined": "IDE_B", // Constraint path
      "refined->condition": "IDE_C", // Build path
      "condition->seed": "MERGE_ABC"  // Full integration
    };

    const key = `${from}->${to}`;
    return mapping[key] || "OPTIMIZE";
  }

  processCollaborativeEvent(event) {
    // Handle Tier-4 collaborative events
    if (event.type === "tier4_state_update") {
      this.handleTier4StateUpdate(event);
    }

    if (event.type === "tier4_session_join") {
      this.handleSessionJoin(event);
    }

    if (event.type === "tier4_session_leave") {
      this.handleSessionLeave(event);
    }
  }

  handleClientMessage(ws, message) {
    const clientInfo = this.clients.get(ws);
    clientInfo.lastSeen = new Date().toISOString();

    switch (message.type) {
      case "tier4_join_session":
        this.joinClientToSession(ws, message.sessionId, message.userId);
        break;

      case "tier4_state_update":
        this.broadcastStateUpdate(ws, message);
        break;

      case "tier4_operator_request":
        this.handleOperatorRequest(ws, message);
        break;

      case "ping":
        this.sendToClient(ws, { type: "pong", ts: new Date().toISOString() });
        break;

      default:
        // Forward client messages to other clients
        this.broadcastToOtherClients(ws, message);
    }
  }

  joinClientToSession(ws, sessionId, userId) {
    const clientInfo = this.clients.get(ws);
    clientInfo.tier4Session = sessionId;
    clientInfo.userId = userId;

    // Create or join session
    if (!this.sessions.has(sessionId)) {
      this.sessions.set(sessionId, {
        id: sessionId,
        createdAt: new Date().toISOString(),
        participants: new Set(),
        lastState: null,
        operatorHistory: []
      });
    }

    const session = this.sessions.get(sessionId);
    session.participants.add(userId);

    this.sendToClient(ws, {
      type: "tier4_session_joined",
      sessionId: sessionId,
      participants: Array.from(session.participants),
      ts: new Date().toISOString()
    });

    // Notify other session participants
    this.broadcastToSession(sessionId, {
      type: "tier4_participant_joined",
      sessionId: sessionId,
      userId: userId,
      participants: Array.from(session.participants),
      ts: new Date().toISOString()
    }, ws);

    this.logEvent({
      type: "session_join",
      sessionId: sessionId,
      userId: userId,
      totalParticipants: session.participants.size,
      ts: new Date().toISOString()
    });
  }

  handleOperatorRequest(ws, message) {
    // Process Tier-4 operator request from client
    const enhancedRequest = {
      ...message,
      type: "tier4_operator_applied",
      ts: new Date().toISOString(),
      processed_by_relay: true
    };

    // Broadcast to all clients in the session
    const clientInfo = this.clients.get(ws);
    if (clientInfo.tier4Session) {
      this.broadcastToSession(clientInfo.tier4Session, enhancedRequest);
    } else {
      this.broadcastToClients(enhancedRequest);
    }
  }

  broadcastToSession(sessionId, message, excludeWs = null) {
    let count = 0;
    for (const [ws, clientInfo] of this.clients.entries()) {
      if (clientInfo.tier4Session === sessionId && ws !== excludeWs) {
        this.sendToClient(ws, message);
        count++;
      }
    }
    return count;
  }

  broadcastToClients(message) {
    const messageStr = JSON.stringify(message);
    let sentCount = 0;

    for (const ws of this.clients.keys()) {
      if (ws.readyState === WebSocket.OPEN) {
        ws.send(messageStr);
        sentCount++;
      }
    }

    return sentCount;
  }

  broadcastToOtherClients(senderWs, message) {
    const messageStr = JSON.stringify(message);
    let sentCount = 0;

    for (const ws of this.clients.keys()) {
      if (ws !== senderWs && ws.readyState === WebSocket.OPEN) {
        ws.send(messageStr);
        sentCount++;
      }
    }

    return sentCount;
  }

  sendToClient(ws, message) {
    if (ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify(message));
      return true;
    }
    return false;
  }

  handleClientDisconnect(ws) {
    const clientInfo = this.clients.get(ws);
    if (clientInfo) {
      // Remove from session
      if (clientInfo.tier4Session) {
        const session = this.sessions.get(clientInfo.tier4Session);
        if (session) {
          session.participants.delete(clientInfo.userId);

          // Notify other participants
          this.broadcastToSession(clientInfo.tier4Session, {
            type: "tier4_participant_left",
            sessionId: clientInfo.tier4Session,
            userId: clientInfo.userId,
            participants: Array.from(session.participants),
            ts: new Date().toISOString()
          });

          // Clean up empty sessions
          if (session.participants.size === 0) {
            this.sessions.delete(clientInfo.tier4Session);
            this.logEvent({
              type: "session_cleanup",
              sessionId: clientInfo.tier4Session,
              ts: new Date().toISOString()
            });
          }
        }
      }

      this.logEvent({
        type: "client_disconnected",
        clientId: clientInfo.id,
        duration: Date.now() - new Date(clientInfo.connectedAt).getTime(),
        ts: new Date().toISOString(),
        totalClients: this.clients.size - 1
      });
    }

    this.clients.delete(ws);
  }

  processRawInput(line) {
    // Handle non-JSON input as raw event
    const rawEvent = {
      type: "raw_input",
      content: line,
      ts: new Date().toISOString()
    };

    this.broadcastToClients(rawEvent);
  }

  setupPeriodicSave() {
    setInterval(() => {
      this.saveState();
    }, 30000); // Save every 30 seconds
  }

  saveState() {
    const state = {
      timestamp: new Date().toISOString(),
      clients: this.clients.size,
      sessions: this.sessions.size,
      eventBufferSize: this.eventBuffer.length,
      recentEvents: this.eventBuffer.slice(-10)
    };

    // Create backup directory if it doesn't exist
    const backupDir = path.join(__dirname, 'backup', new Date().toISOString().split('T')[0].replace(/-/g, ''));
    if (!fs.existsSync(backupDir)) {
      fs.mkdirSync(backupDir, { recursive: true });
    }

    const stateFile = path.join(backupDir, `relay-state-${Date.now()}.json`);
    fs.writeFileSync(stateFile, JSON.stringify(state, null, 2));
  }

  generateClientId() {
    return `client_${Date.now()}_${Math.random().toString(36).substr(2, 6)}`;
  }

  logEvent(event) {
    console.log(JSON.stringify(event));

    // Also add to event buffer for new clients
    this.eventBuffer.push(event);
    if (this.eventBuffer.length > this.maxBufferSize) {
      this.eventBuffer.shift();
    }
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

// Start the enhanced relay
const relay = new Tier4EnhancedRelay(process.env.PORT || 9000);

// Handle graceful shutdown
process.on('SIGINT', () => {
  console.log(JSON.stringify({
    type: "relay_shutdown",
    ts: new Date().toISOString(),
    stats: relay.getStats()
  }));

  relay.saveState();
  process.exit(0);
});

// Export for testing
module.exports = Tier4EnhancedRelay;
