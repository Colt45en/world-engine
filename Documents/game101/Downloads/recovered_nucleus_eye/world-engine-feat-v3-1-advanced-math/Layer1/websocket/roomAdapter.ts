// roomAdapter.ts — connects Tier-4 engine to Engine Room (the "head nucleus")

export type RoomWall = "front" | "left" | "right" | "back" | "floor" | "ceil";

export type RoomMessage =
    | { type: "INIT"; payload: { sessionId: string; title?: string } }
    | { type: "SNAPSHOT"; payload: { cid: string; state: Tier4State; parentCid?: string | null; timestamp: number } }
    | { type: "EVENT"; payload: { id: string; button: string; inputCid: string; outputCid: string; timestamp: number; meta?: any } }
    | { type: "SET_PANEL_HTML"; payload: { panelId: number; html: string } }
    | { type: "ADD_PANEL"; payload: { wall: RoomWall; x: number; y: number; w: number; h: number; title?: string; html?: string } }
    | { type: "REMOVE_PANEL"; payload: { panelId: number } }
    | { type: "FOCUS_PANEL"; payload: { panelId: number } }
    | { type: "TOAST"; payload: { msg: string; type?: string } }
    | { type: "LOAD_SESSION"; payload: { events: any[]; snapshots: Record<string, any>; currentCid?: string | null } }
    | { type: "REQUEST"; payload: { op: "GET_STATE" | "LIST_PANELS" } }
    | { type: "PIPELINE_DOWN"; payload: { knowledge: any; segment: string; timestamp: number } }
    | { type: "PIPELINE_UP"; payload: { gameState: any; recursive: boolean; timestamp: number } }
    | { type: "NUCLEUS_CONNECT"; payload: { nucleusId: string; companionId: string } }
    | { type: "META_ANCHOR"; payload: { anchorStrength: number; pipelineStatus: string } };

export type RoomResponse =
    | { type: "ROOM_READY"; payload: { sessionId: string } }
    | { type: "ROOM_STATE"; payload: { sessionId: string; panelCount: number; events: number; snapshots: number; currentCid: string | null } }
    | { type: "ROOM_PANELS"; payload: Array<{ id: number; wall: RoomWall; x: number; y: number; w: number; h: number; title: string }> }
    | { type: "ROOM_ACK"; payload: { op: string; panelId?: number } }
    | { type: "REQUEST_LOAD_CID"; payload: { cid: string } }
    | { type: "NUCLEUS_STATUS"; payload: { connected: boolean; conscious: boolean; breathing: boolean } }
    | { type: "PIPELINE_STATUS"; payload: { downActive: boolean; upActive: boolean; zigzagSync: boolean } }
    | { type: "META_FLOOR_UPDATE"; payload: { anchorStrength: number; knowledgeFlow: string } };

export interface Tier4State {
    x: number[];        // State vector [p, i, g, c, ...]
    kappa: number;      // Confidence parameter
    level: number;      // Current level/depth
    operator?: string;  // Last applied operator
    meta?: any;         // Additional metadata
}

export interface RoomEvent {
    id: string;
    button: string;     // Operator name or event type
    inputCid: string;   // Input state CID
    outputCid: string;  // Output state CID
    timestamp: number;
    meta?: any;
}

export interface RoomSnapshot {
    cid: string;
    state: Tier4State;
    parentCid?: string | null;
    timestamp: number;
}

export class EngineRoom {
    private readonly target: Window;
    private readonly origin: string;
    private readonly eventHandlers: Map<string, Function> = new Map();
    private isReady = false;
    private readyCallbacks: Function[] = [];

    constructor(targetWindow: Window, origin = "*") {
        this.target = targetWindow;
        this.origin = origin;

        // Listen for responses from the room
        window.addEventListener("message", (event) => {
            this.handleRoomMessage(event);
        });
    }

    private handleRoomMessage(event: MessageEvent) {
        const data = event.data;
        if (!data || data.room !== "gridroom") return;

        // Handle room ready signal
        if (data.type === "ROOM_READY") {
            this.isReady = true;
            this.readyCallbacks.forEach(callback => callback());
            this.readyCallbacks = [];
        }

        // Dispatch to registered handlers
        const handler = this.eventHandlers.get(data.type);
        if (handler) {
            handler(data.payload || {});
        }
    }

    /**
     * Send a message to the Engine Room
     */
    send(msg: RoomMessage) {
        this.target.postMessage({ room: "gridroom", ...msg }, this.origin);
    }

    /**
     * Wait for the room to be ready before executing callback
     */
    whenReady(callback: () => void) {
        if (this.isReady) {
            callback();
        } else {
            this.readyCallbacks.push(callback);
        }
    }

    /**
     * Register handler for room responses
     */
    on(eventType: string, handler: Function) {
        this.eventHandlers.set(eventType, handler);
    }

    /**
     * Initialize the room with session info
     */
    init(sessionId: string, title?: string) {
        this.send({
            type: "INIT",
            payload: { sessionId, title }
        });
    }

    /**
     * Publish a state snapshot to the room
     */
    publishSnapshot(cid: string, state: Tier4State, parentCid?: string | null) {
        this.send({
            type: "SNAPSHOT",
            payload: {
                cid,
                state,
                parentCid: parentCid ?? null,
                timestamp: Date.now()
            }
        });
    }

    /**
     * Publish an event (operator application, etc.)
     */
    publishEvent(event: RoomEvent) {
        this.send({
            type: "EVENT",
            payload: {
                ...event,
                timestamp: event.timestamp || Date.now()
            }
        });
    }

    /**
     * Create an event from an operator application
     */
    publishOperatorEvent(operator: string, inputCid: string, outputCid: string, meta?: any) {
        this.publishEvent({
            id: `op_${Date.now()}_${Math.random().toString(36).substr(2, 8)}`,
            button: `Tier4_${operator}`,
            inputCid,
            outputCid,
            timestamp: Date.now(),
            meta
        });
    }

    /**
     * Show a toast notification in the room
     */
    toast(message: string, type: "info" | "success" | "warning" | "error" = "info") {
        this.send({
            type: "TOAST",
            payload: { msg: message, type }
        });
    }

    /**
     * Add a panel to the room
     */
    addPanel(config: {
        wall: RoomWall;
        x: number;
        y: number;
        w: number;
        h: number;
        title?: string;
        html?: string
    }) {
        this.send({
            type: "ADD_PANEL",
            payload: config
        });
    }

    /**
     * Update a panel's HTML content
     */
    setPanelHTML(panelId: number, html: string) {
        this.send({
            type: "SET_PANEL_HTML",
            payload: { panelId, html }
        });
    }

    /**
     * Remove a panel from the room
     */
    removePanel(panelId: number) {
        this.send({
            type: "REMOVE_PANEL",
            payload: { panelId }
        });
    }

    /**
     * Focus/highlight a panel
     */
    focusPanel(panelId: number) {
        this.send({
            type: "FOCUS_PANEL",
            payload: { panelId }
        });
    }

    /**
     * Request current room state
     */
    getState() {
        this.send({
            type: "REQUEST",
            payload: { op: "GET_STATE" }
        });
    }

    /**
     * Request list of all panels
     */
    listPanels() {
        this.send({
            type: "REQUEST",
            payload: { op: "LIST_PANELS" }
        });
    }

    /**
     * Load a complete session (events + snapshots)
     */
    loadSession(events: any[], snapshots: Record<string, any>, currentCid?: string | null) {
        this.send({
            type: "LOAD_SESSION",
            payload: { events, snapshots, currentCid }
        });
    }
}

/**
 * Semantic wall layout helper - provides standard positioning for different content types
 */
export class RoomLayout {
    private readonly width: number;
    private readonly height: number;

    constructor(roomWidth = 1400, roomHeight = 800) {
        this.width = roomWidth;
        this.height = roomHeight;
    }

    /**
     * Get standard panel positions by semantic function
     */
    getPosition(semantic: "events" | "snapshots" | "nucleus" | "operators" | "metrics" | "docs" | "health") {
        const positions = {
            // Left wall - Events and activity
            events: { wall: "left" as RoomWall, x: 60, y: 260, w: 420, h: 260 },

            // Right wall - State snapshots
            snapshots: { wall: "right" as RoomWall, x: this.width - 480, y: 260, w: 420, h: 260 },

            // Front wall - Main nucleus display
            nucleus: { wall: "front" as RoomWall, x: this.width / 2 - 200, y: 50, w: 400, h: 300 },

            // Front wall - Operator controls
            operators: { wall: "front" as RoomWall, x: this.width / 2 - 300, y: 380, w: 600, h: 200 },

            // Floor - Metrics and performance
            metrics: { wall: "floor" as RoomWall, x: this.width / 2 - 240, y: 180, w: 480, h: 220 },

            // Back wall - Documentation and notes
            docs: { wall: "back" as RoomWall, x: 100, y: 100, w: 500, h: 400 },

            // Ceiling - System health monitoring
            health: { wall: "ceil" as RoomWall, x: this.width - 350, y: 50, w: 300, h: 150 }
        };

        return positions[semantic];
    }
}

/**
 * Tier-4 specific room manager with built-in state tracking
 */
export class Tier4Room extends EngineRoom {
    private readonly stateHistory: Map<string, RoomSnapshot> = new Map();
    private currentCid: string | null = null;
    private readonly sessionId: string;
    private readonly layout: RoomLayout;
    private panelIds: Record<string, number> = {};

    constructor(targetWindow: Window, sessionId?: string, origin = "*") {
        super(targetWindow, origin);
        this.sessionId = sessionId || `tier4_${Date.now()}`;
        this.layout = new RoomLayout();

        // Handle load requests from room
        this.on("REQUEST_LOAD_CID", (payload: { cid: string }) => {
            this.loadSnapshotByCid(payload.cid);
        });

        this.setupAutoPanels();
    }

    private async setupAutoPanels() {
        this.whenReady(() => {
            // Initialize with session
            this.init(this.sessionId, "Tier-4 Engine Room");

            // Auto-create semantic panels
            setTimeout(() => this.createSemanticPanels(), 500);
        });
    }

    private createSemanticPanels() {
        // Create standard panels for Tier-4 workflow
        const panels = [
            { semantic: "operators", title: "Tier-4 Operators", html: this.buildOperatorsHTML() },
            { semantic: "metrics", title: "Performance Metrics", html: this.buildMetricsHTML() },
            { semantic: "docs", title: "Tier-4 Documentation", html: this.buildDocsHTML() },
            { semantic: "health", title: "System Health", html: this.buildHealthHTML() }
        ];

        panels.forEach(panel => {
            const pos = this.layout.getPosition(panel.semantic as any);
            if (pos) {
                this.addPanel({ ...pos, title: panel.title, html: panel.html });
            }
        });
    }

    /**
     * Apply a Tier-4 operator and track the state transition
     */
    applyOperator(operator: string, currentState: Tier4State, newState: Tier4State) {
        const inputCid = this.generateCid(currentState);
        const outputCid = this.generateCid(newState);

        // Store both states
        this.storeSnapshot(inputCid, currentState);
        this.storeSnapshot(outputCid, newState, inputCid);

        // Publish the operator event
        this.publishOperatorEvent(operator, inputCid, outputCid, {
            deltaKappa: newState.kappa - currentState.kappa,
            levelChange: newState.level - currentState.level,
            vectorDelta: newState.x.map((v, i) => v - (currentState.x[i] || 0))
        });

        // Update current state
        this.currentCid = outputCid;
        this.publishSnapshot(outputCid, newState, inputCid);
    }

    /**
     * Store a snapshot in local history
     */
    private storeSnapshot(cid: string, state: Tier4State, parentCid?: string) {
        this.stateHistory.set(cid, {
            cid,
            state: { ...state },
            parentCid,
            timestamp: Date.now()
        });
    }

    /**
     * Generate content-addressed ID for state
     */
    private generateCid(state: Tier4State): string {
        const normalized = {
            x: state.x.map(v => Math.round(v * 1000) / 1000),
            kappa: Math.round(state.kappa * 1000) / 1000,
            level: state.level
        };
        const hash = btoa(JSON.stringify(normalized)).replace(/[+/=]/g, '').slice(0, 16);
        return `tier4_${hash}`;
    }

    /**
     * Load a snapshot by CID (called when user clicks Load in snapshots panel)
     */
    private loadSnapshotByCid(cid: string) {
        const snapshot = this.stateHistory.get(cid);
        if (snapshot) {
            this.currentCid = cid;
            // Emit event for IDE to handle the state restoration
            window.dispatchEvent(new CustomEvent('tier4-load-state', {
                detail: { state: snapshot.state, cid }
            }));
            this.toast(`Loaded state ${cid.slice(0, 12)}...`, "info");
        }
    }

    /**
     * Get current state CID
     */
    getCurrentCid(): string | null {
        return this.currentCid;
    }

    /**
     * Get state history
     */
    getStateHistory(): Map<string, RoomSnapshot> {
        return new Map(this.stateHistory);
    }

    // HTML builders for standard panels
    private buildOperatorsHTML(): string {
        return `<!doctype html><meta charset="utf-8"><style>
      body{margin:0;background:#0b1418;color:#cfe;font:14px system-ui;padding:16px}
      .operator-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin:16px 0}
      .operator{background:#1a2332;border:1px solid #2a3548;border-radius:8px;padding:12px;text-align:center;cursor:pointer;transition:all 0.2s}
      .operator:hover{background:#243344;border-color:#54f0b8}
      .macro{background:#2d1a33;border-color:#8a4a9f}
      .macro:hover{background:#3a2440;border-color:#b366d9}
    </style>
    <h3>Tier-4 Operators</h3>
    <div class="operator-grid">
      <div class="operator" onclick="parent.postMessage({type:'apply-operator',operator:'ST'},'*')">ST<br><small>Stabilize</small></div>
      <div class="operator" onclick="parent.postMessage({type:'apply-operator',operator:'UP'},'*')">UP<br><small>Update</small></div>
      <div class="operator" onclick="parent.postMessage({type:'apply-operator',operator:'PR'},'*')">PR<br><small>Progress</small></div>
      <div class="operator" onclick="parent.postMessage({type:'apply-operator',operator:'CV'},'*')">CV<br><small>Converge</small></div>
      <div class="operator" onclick="parent.postMessage({type:'apply-operator',operator:'RB'},'*')">RB<br><small>Rollback</small></div>
      <div class="operator" onclick="parent.postMessage({type:'apply-operator',operator:'RS'},'*')">RS<br><small>Reset</small></div>
    </div>
    <h3>Three Ides Macros</h3>
    <div class="operator-grid">
      <div class="operator macro" onclick="parent.postMessage({type:'apply-macro',macro:'IDE_A'},'*')">IDE_A<br><small>ST→SL→CP</small></div>
      <div class="operator macro" onclick="parent.postMessage({type:'apply-macro',macro:'IDE_B'},'*')">IDE_B<br><small>CV→PR→RC</small></div>
      <div class="operator macro" onclick="parent.postMessage({type:'apply-macro',macro:'MERGE_ABC'},'*')">MERGE<br><small>Full Cycle</small></div>
    </div>`;
    }

    private buildMetricsHTML(): string {
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
      chart.width = chart.offsetWidth;
      chart.height = 60;

      let dataPoints = [];
      function updateChart(value) {
        dataPoints.push(value);
        if (dataPoints.length > 50) dataPoints.shift();

        ctx.clearRect(0, 0, chart.width, chart.height);
        ctx.strokeStyle = '#54f0b8';
        ctx.lineWidth = 2;

        ctx.beginPath();
        dataPoints.forEach((point, i) => {
          const x = (i / (dataPoints.length - 1)) * chart.width;
          const y = chart.height - (point * chart.height);
          if (i === 0) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
        });
        ctx.stroke();
      }

      // Listen for metrics updates
      window.addEventListener('message', (e) => {
        if (e.data.type === 'update-metrics') {
          const m = e.data.metrics;
          document.getElementById('ops-rate').textContent = m.opsRate || '--';
          document.getElementById('kappa-delta').textContent = m.kappaDelta || '--';
          document.getElementById('transitions').textContent = m.transitions || '--';
          document.getElementById('memory').textContent = m.memory || '--';
          if (m.chartValue !== undefined) updateChart(m.chartValue);
        }
      });
    </script>`;
    }

    private buildDocsHTML(): string {
        return `<!doctype html><meta charset="utf-8"><style>
      body{margin:0;background:#0b1418;color:#cfe;font:13px system-ui;padding:16px;line-height:1.5}
      h3{color:#54f0b8;margin-top:0}
      .operator{background:#1a2332;padding:8px;border-radius:4px;margin:8px 0}
      .operator-name{color:#9fd6ff;font-weight:600}
      code{background:#0a1222;padding:2px 6px;border-radius:3px;color:#ff9f43}
    </style>
    <h3>Tier-4 Meta System</h3>
    <p>The Tier-4 system operates on state vectors <code>[p, i, g, c]</code> representing:</p>
    <ul>
      <li><strong>p</strong>: Persistence/stability</li>
      <li><strong>i</strong>: Information/input</li>
      <li><strong>g</strong>: Goal/direction</li>
      <li><strong>c</strong>: Confidence/certainty</li>
    </ul>

    <h3>Core Operators</h3>
    <div class="operator">
      <div class="operator-name">ST (Stabilize)</div>
      Identity transformation, preserves current state.
    </div>
    <div class="operator">
      <div class="operator-name">UP (Update)</div>
      Increases information and confidence components.
    </div>
    <div class="operator">
      <div class="operator-name">PR (Progress)</div>
      Refines and improves goal alignment.
    </div>
    <div class="operator">
      <div class="operator-name">CV (Converge)</div>
      Moves toward optimal state configuration.
    </div>
    <div class="operator">
      <div class="operator-name">RB (Rollback)</div>
      Returns to previous stable state.
    </div>

    <h3>Parameter κ (Kappa)</h3>
    <p>Global confidence parameter (0.0 - 1.0) that influences all transformations.</p>`;
    }

    private buildHealthHTML(): string {
        return `<!doctype html><meta charset="utf-8"><style>
      body{margin:0;background:#0b1418;color:#cfe;font:11px system-ui;padding:8px}
      .health-item{display:flex;justify-content:space-between;align-items:center;padding:4px 0}
      .status{width:8px;height:8px;border-radius:50%;background:#18c08f}
      .warning{background:#e5c558}
      .error{background:#ff4d4f}
    </style>
    <h4>System Health</h4>
    <div class="health-item">
      <span>WebSocket</span>
      <div class="status" id="ws-status"></div>
    </div>
    <div class="health-item">
      <span>Memory</span>
      <div class="status" id="memory-status"></div>
    </div>
    <div class="health-item">
      <span>FPS</span>
      <span id="fps-display">--</span>
    </div>
    <div class="health-item">
      <span>Latency</span>
      <span id="latency-display">--ms</span>
    </div>
    <script>
      // Mock health monitoring
      setInterval(() => {
        document.getElementById('fps-display').textContent = Math.floor(Math.random() * 10 + 55);
        document.getElementById('latency-display').textContent = Math.floor(Math.random() * 20 + 5) + 'ms';
      }, 2000);
    </script>`;
    }

    /**
     * Update metrics in the metrics panel
     */
    updateMetrics(metrics: {
        opsRate?: string;
        kappaDelta?: string;
        transitions?: string;
        memory?: string;
        chartValue?: number;
    }) {
        // Find metrics panel and send update
        const iframe = document.querySelector(`iframe[src*="metrics"]`) as HTMLIFrameElement | null;
        const metricsPanel = iframe?.contentWindow;
        if (metricsPanel) {
            metricsPanel.postMessage({ type: 'update-metrics', metrics }, '*');
        }
    }
}

/**
 * Pipeline-aware Tier-4 Room for bidirectional zig-zag communication
 * Integrates with the nucleus-demo bidirectional pipeline system
 */
export class PipelineRoom extends Tier4Room {
    private nucleusConnected: boolean = false;
    private companionAnchor: any = null;
    private pipelineStatus = {
        downActive: false,
        upActive: false,
        zigzagSync: false
    };

    constructor(targetWindow: Window, sessionId?: string, origin = "*") {
        super(targetWindow, sessionId, origin);

        // Setup pipeline-specific handlers
        this.setupPipelineHandlers();
        this.monitorNucleusConnection();
    }

    private setupPipelineHandlers() {
        // Handle nucleus connection status
        this.on("NUCLEUS_STATUS", (payload: { connected: boolean; conscious: boolean; breathing: boolean }) => {
            this.nucleusConnected = payload.connected;
            this.updateConnectionStatus();

            if (payload.connected && payload.conscious) {
                this.toast("🧠 Nucleus consciousness online - pipeline ready", "success");
            }
        });

        // Handle pipeline status updates
        this.on("PIPELINE_STATUS", (payload: { downActive: boolean; upActive: boolean; zigzagSync: boolean }) => {
            this.pipelineStatus = payload;
            this.updatePipelineDisplay();

            if (payload.downActive && payload.upActive && payload.zigzagSync) {
                this.toast("🔄 Bidirectional zig-zag pipeline synchronized", "success");
            }
        });

        // Handle meta floor updates
        this.on("META_FLOOR_UPDATE", (payload: { anchorStrength: number; knowledgeFlow: string }) => {
            this.updateMetaFloorDisplay(payload);
        });
    }

    private monitorNucleusConnection() {
        // Check if nucleus-demo is available
        setInterval(() => {
            if (typeof window !== 'undefined' && (window as any).nucleusAI && (window as any).companionAI) {
                if (!this.nucleusConnected) {
                    this.establishNucleusConnection();
                }
            }
        }, 2000);
    }

    private establishNucleusConnection() {
        const nucleusAI = (window as any).nucleusAI;
        const companionAI = (window as any).companionAI;

        if (nucleusAI && companionAI) {
            this.nucleusConnected = true;
            this.companionAnchor = companionAI;

            // Send connection event to room
            this.send({
                type: "NUCLEUS_CONNECT",
                payload: {
                    nucleusId: nucleusAI.identity?.id || 'nucleus_main',
                    companionId: companionAI.identity?.name || 'companion_anchor'
                }
            });

            // Start monitoring pipeline flows
            this.startPipelineMonitoring();

            this.toast("🌌 Companion meta floor anchor established", "success");
        }
    }

    private startPipelineMonitoring() {
        if (!this.companionAnchor) return;

        setInterval(() => {
            const companion = this.companionAnchor;
            if (companion?.connectionSystem?.zigZagPipelines) {
                const pipelines = companion.connectionSystem.zigZagPipelines;

                // Send pipeline status
                this.send({
                    type: "PIPELINE_DOWN",
                    payload: {
                        knowledge: pipelines.downPipeline?.currentSegment || 0,
                        segment: pipelines.downPipeline?.segments?.[pipelines.downPipeline.currentSegment]?.level || 'unknown',
                        timestamp: Date.now()
                    }
                });

                this.send({
                    type: "PIPELINE_UP",
                    payload: {
                        gameState: pipelines.upPipeline?.currentSegment || 0,
                        recursive: pipelines.upPipeline?.recursiveNature || false,
                        timestamp: Date.now()
                    }
                });

                // Send meta anchor status
                if (companion.connectionSystem?.metaFloorAnchor) {
                    this.send({
                        type: "META_ANCHOR",
                        payload: {
                            anchorStrength: companion.consciousness?.anchorStrength || 0,
                            pipelineStatus: 'active'
                        }
                    });
                }
            }
        }, 3000);
    }

    private updateConnectionStatus() {
        // Update status indicators in the UI
        if (typeof document !== 'undefined') {
            const nucleusStatus = document.getElementById('nucleus-status');
            const bridgeStatus = document.getElementById('bridge-status');

            if (nucleusStatus) {
                nucleusStatus.className = this.nucleusConnected ? 'status-indicator status-connected' : 'status-indicator status-disconnected';
                nucleusStatus.textContent = this.nucleusConnected ? '🧠 Nucleus Online' : '🧠 Nucleus Unavailable';
            }

            if (bridgeStatus) {
                bridgeStatus.className = this.nucleusConnected ? 'status-indicator status-connected' : 'status-indicator status-disconnected';
                bridgeStatus.textContent = this.nucleusConnected ? '🌉 Bridge Active' : '🌉 Bridge Failed';
            }
        }
    }

    private updatePipelineDisplay() {
        // Update pipeline status in UI elements
        if (typeof document !== 'undefined') {
            const downStatus = document.getElementById('downStatus');
            const upStatus = document.getElementById('upStatus');

            if (downStatus) {
                downStatus.textContent = this.pipelineStatus.downActive ? 'Active' : 'Inactive';
            }

            if (upStatus) {
                upStatus.textContent = this.pipelineStatus.upActive ? 'Active' : 'Inactive';
            }
        }
    }

    private updateMetaFloorDisplay(payload: { anchorStrength: number; knowledgeFlow: string }) {
        if (typeof document !== 'undefined') {
            const anchorStrength = document.getElementById('anchorStrength');
            const lastActivity = document.getElementById('lastPipelineActivity');

            if (anchorStrength) {
                anchorStrength.textContent = `${Math.round(payload.anchorStrength * 100)}%`;
            }

            if (lastActivity) {
                lastActivity.textContent = new Date().toLocaleTimeString();
            }
        }
    }

    /**
     * Send knowledge through DOWN pipeline
     */
    sendDownPipelineKnowledge(knowledge: any, segment: string) {
        this.send({
            type: "PIPELINE_DOWN",
            payload: {
                knowledge,
                segment,
                timestamp: Date.now()
            }
        });
    }

    /**
     * Send game state through UP pipeline (recursive)
     */
    sendUpPipelineGameState(gameState: any, recursive: boolean = true) {
        this.send({
            type: "PIPELINE_UP",
            payload: {
                gameState,
                recursive,
                timestamp: Date.now()
            }
        });
    }

    /**
     * Get current pipeline status
     */
    getPipelineStatus() {
        return {
            nucleusConnected: this.nucleusConnected,
            downActive: this.pipelineStatus.downActive,
            upActive: this.pipelineStatus.upActive,
            zigzagSync: this.pipelineStatus.zigzagSync,
            anchorEstablished: this.companionAnchor !== null
        };
    }
}

export default EngineRoom;
