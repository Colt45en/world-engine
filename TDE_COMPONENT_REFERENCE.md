# TDE Component Architecture Manual

*Deep dive into Tier-4 Distributed Engine components, APIs, and internal systems*

---

## 🏗️ Core Components Reference

### **1. WebSocket Relay (tier4_ws_relay.js)**

**Purpose**: Central communication hub for distributed Tier-4 operations

**Class: Tier4EnhancedRelay**
```javascript
constructor(port = 9000) {
  // Initializes WebSocket server, client tracking, session management
  this.clients = new Map();     // WebSocket -> clientInfo
  this.sessions = new Map();    // sessionId -> session data
  this.eventBuffer = [];       // Circular buffer for events
  this.nucleusToOperator;       // Mapping tables
  this.tagToOperator;
}
```

**Key Methods:**
- `processIncomingEvent(event)` — NDJSON event processing pipeline
- `enhanceWithTier4Mapping(event)` — Automatic operator mapping
- `joinClientToSession(ws, sessionId, userId)` — Collaborative session management
- `broadcastToSession(sessionId, message)` — Targeted message distribution
- `suggestMacroForCycle(cycle, total)` — Intelligent macro recommendations

**Event Types Handled:**
```javascript
// Nucleus execution events
{type: "nucleus_exec", role: "VIBRATE|OPTIMIZATION|STATE|SEED"}

// Memory storage events
{type: "memory_store", tag: "energy|refined|condition|seed"}

// Collaborative events
{type: "tier4_state_update", sessionId, state}
{type: "tier4_session_join", sessionId, userId}
{type: "loop_back", from, to}
{type: "cycle_start", cycle, total}
```

**Configuration Options:**
```javascript
const config = {
  port: 9000,                    // WebSocket port
  maxBufferSize: 1000,          // Event buffer size
  saveInterval: 30000,          // Auto-save interval (ms)
  sessionTimeout: 300000,       // Session cleanup timeout
  maxClientsPerSession: 50      // Collaboration limits
};
```

### **2. Room Adapter (roomAdapter.ts)**

**Purpose**: TypeScript interface layer between Tier-4 system and Engine Room UI

**Class: EngineRoom**
```typescript
class EngineRoom {
  constructor(
    private container: HTMLElement,
    private config: RoomConfig = {}
  ) {}

  // Core methods
  public init(sessionId?: string): Promise<void>
  public addPanel(panel: PanelConfig): void
  public sendCommand(command: RoomCommand): void
  public getSnapshot(): StateSnapshot
  public loadSnapshot(snapshot: StateSnapshot): void
}
```

**Interface: RoomMessage**
```typescript
interface RoomMessage {
  type: 'INIT' | 'SNAPSHOT' | 'EVENT' | 'SET_PANEL_HTML' | 'APPLY_OPERATOR';
  sessionId?: string;
  data: any;
  timestamp: string;
  cid?: string;  // Content-addressed ID
}
```

**Panel Management System:**
```typescript
interface PanelConfig {
  id: string;
  title: string;
  position: WallPosition; // 'front' | 'left' | 'right' | 'back' | 'floor' | 'ceiling'
  content: string;        // HTML content
  draggable: boolean;
  resizable: boolean;
  initialSize?: {width: number, height: number};
}
```

**Semantic Wall Layout Helper:**
```typescript
class SemanticWallLayout {
  static getDefaultPanels(): PanelConfig[] {
    return [
      {id: 'operators', position: 'front', title: 'Command Center'},
      {id: 'events', position: 'left', title: 'Activity Feed'},
      {id: 'snapshots', position: 'right', title: 'State Management'},
      {id: 'docs', position: 'back', title: 'Documentation'},
      {id: 'metrics', position: 'floor', title: 'Performance'},
      {id: 'health', position: 'ceiling', title: 'System Status'}
    ];
  }

  static positionPanel(panel: PanelConfig, container: HTMLElement): void;
  static createWallBoundaries(container: HTMLElement): void;
}
```

### **3. WebSocket Integration Bridge (tier4_room_integration.ts)**

**Purpose**: Bridges WebSocket NDJSON streams with Tier-4 operator system

**Class: Tier4RoomBridge**
```typescript
class Tier4RoomBridge {
  constructor(
    private room: EngineRoom,
    private websocketUrl: string = 'ws://localhost:9000'
  ) {}

  // Connection management
  public connect(): Promise<void>
  public disconnect(): void
  public reconnect(maxAttempts: number = 5): Promise<void>

  // Event processing
  private handleNDJSONEvent(event: NDJSONEvent): void
  private applyOperatorFromNucleus(nucleusRole: string): void
  private executeThreeIdesMacro(macro: string): void
}
```

**NDJSON Event Processor:**
```typescript
interface NDJSONEvent {
  type: string;
  role?: 'VIBRATE' | 'OPTIMIZATION' | 'STATE' | 'SEED';
  tag?: string;
  cycle?: number;
  total?: number;
  timestamp: string;
  data?: any;
}

// Auto-mapping configuration
const nucleusToOperatorMap = {
  'VIBRATE': 'ST',        // Stabilization
  'OPTIMIZATION': 'UP',   // Update/Progress
  'STATE': 'CV',          // Convergence
  'SEED': 'RB'           // Rollback
};

const memoryTagToOperatorMap = {
  'energy': 'CH',         // Change
  'refined': 'PR',        // Progress
  'condition': 'SL',      // Selection
  'seed': 'MD'           // Multidimensional
};
```

**Macro Execution System:**
```typescript
class MacroExecutor {
  static executeThreeIdes(macro: 'IDE_A' | 'IDE_B' | 'IDE_C' | 'MERGE_ABC'): void {
    switch(macro) {
      case 'IDE_A': // Analysis
        this.applySequence(['ST', 'UP', 'PR']);
        break;
      case 'IDE_B': // Constraints
        this.applySequence(['CV', 'SL', 'CH']);
        break;
      case 'IDE_C': // Build
        this.applySequence(['MD', 'RB', 'RS']);
        break;
      case 'MERGE_ABC': // Integration
        this.applySequence(['ST', 'CV', 'MD', 'OPTIMIZE']);
        break;
    }
  }
}
```

### **4. React Engine Room Component (EngineRoom.tsx)**

**Purpose**: React wrapper for Engine Room with hooks and imperative API

**Component Interface:**
```typescript
interface EngineRoomProps {
  webSocketUrl?: string;
  sessionId?: string;
  initialPanels?: PanelConfig[];
  onOperatorApplied?: (operator: string, context: any) => void;
  onStateLoaded?: (snapshot: StateSnapshot) => void;
  onSessionJoined?: (sessionId: string, participants: string[]) => void;
  autoConnect?: boolean;
  debugMode?: boolean;
}

interface EngineRoomRef {
  // Imperative API for parent components
  applyOperator: (operator: string) => void;
  loadSnapshot: (cid: string) => Promise<void>;
  saveSnapshot: () => Promise<string>;
  joinSession: (sessionId: string) => Promise<void>;
  getMetrics: () => SystemMetrics;
}
```

**Hook: useEngineRoom**
```typescript
function useEngineRoom(config: EngineRoomConfig) {
  const [connected, setConnected] = useState(false);
  const [session, setSession] = useState<string | null>(null);
  const [participants, setParticipants] = useState<string[]>([]);
  const [metrics, setMetrics] = useState<SystemMetrics | null>(null);

  // Returns control interface
  return {
    connected,
    session,
    participants,
    metrics,
    applyOperator: (op: string) => void,
    saveSnapshot: () => Promise<string>,
    loadSnapshot: (cid: string) => Promise<void>
  };
}
```

**Integration Example:**
```tsx
function MyDevelopmentEnvironment() {
  const roomRef = useRef<EngineRoomRef>(null);
  const [operatorHistory, setOperatorHistory] = useState<string[]>([]);

  const handleOperatorApplied = useCallback((operator: string, context: any) => {
    setOperatorHistory(prev => [...prev, `${operator}: ${JSON.stringify(context)}`]);

    // Custom logic based on operator
    switch(operator) {
      case 'ST': // Stabilization - save current state
        roomRef.current?.saveSnapshot();
        break;
      case 'RB': // Rollback - load previous state
        loadPreviousSnapshot();
        break;
    }
  }, []);

  return (
    <div className="development-workspace">
      <EngineRoom
        ref={roomRef}
        sessionId="my-dev-session"
        onOperatorApplied={handleOperatorApplied}
        autoConnect={true}
        debugMode={process.env.NODE_ENV === 'development'}
      />

      <OperatorHistoryPanel history={operatorHistory} />
    </div>
  );
}
```

### **5. Engine Room HTML Interface (worldengine.html)**

**Purpose**: Standalone HTML interface with embedded message bus system

**Message Bus Architecture:**
```javascript
class EngineRoomMessageBus {
  constructor() {
    this.kvStore = new Map();
    this.panels = new Map();
    this.subscribers = new Map();
    this.commandQueue = [];
  }

  // Core API
  processCommand(command) {
    switch(command.type) {
      case 'INIT': return this.handleInit(command);
      case 'SNAPSHOT': return this.handleSnapshot(command);
      case 'EVENT': return this.handleEvent(command);
      case 'SET_PANEL_HTML': return this.setPanel(command);
      case 'APPLY_OPERATOR': return this.applyOperator(command);
    }
  }
}
```

**Panel Management:**
```javascript
// Virtual panel system with drag-and-drop
class VirtualPanelManager {
  createPanel(config) {
    const panel = {
      id: config.id,
      element: document.createElement('div'),
      position: config.position,
      draggable: config.draggable,
      content: config.content
    };

    this.setupDragAndDrop(panel);
    this.positionByWall(panel);
    return panel;
  }

  positionByWall(panel) {
    const positions = {
      front: {top: '40%', left: '45%'},
      left: {top: '20%', left: '5%'},
      right: {top: '20%', right: '5%'},
      back: {top: '60%', left: '45%'},
      floor: {bottom: '5%', left: '20%'},
      ceiling: {top: '5%', left: '45%'}
    };

    Object.assign(panel.element.style, positions[panel.position]);
  }
}
```

**PostMessage Integration:**
```javascript
// Communication with parent window/iframe
window.addEventListener('message', (event) => {
  if (event.origin !== window.location.origin) return;

  const {type, data} = event.data;
  const result = messageBus.processCommand({type, ...data});

  // Send response back to parent
  event.source.postMessage({
    type: 'ENGINE_ROOM_RESPONSE',
    originalType: type,
    result: result
  }, event.origin);
});

// Auto-provisioned panels
const defaultPanels = [
  {id: 'events', title: 'Events', position: 'left'},
  {id: 'snapshots', title: 'Snapshots', position: 'right'},
  {id: 'nucleus', title: 'Nucleus', position: 'front'}
];
```

---

## 🔧 Configuration & Customization

### **Environment Variables**
```bash
# WebSocket configuration
TDE_WEBSOCKET_PORT=9000
TDE_WEBSOCKET_HOST=localhost

# Session management
TDE_SESSION_TIMEOUT=300000
TDE_MAX_CLIENTS_PER_SESSION=50

# Performance tuning
TDE_EVENT_BUFFER_SIZE=1000
TDE_SAVE_INTERVAL=30000

# Debug settings
TDE_DEBUG_MODE=true
TDE_LOG_LEVEL=info
```

### **Custom Operator Mappings**
```typescript
// Extend default mappings
const customOperatorMap = new Map([
  ...defaultNucleusToOperatorMap,
  ['CUSTOM_EVENT', 'TRANSFORM'],
  ['VALIDATION', 'CHECK'],
  ['COMPILATION', 'BUILD']
]);

// Register with bridge
bridge.setOperatorMapping(customOperatorMap);
```

### **Panel Templates**
```typescript
// Create reusable panel templates
const debuggingPanelTemplate: PanelConfig = {
  id: 'debugger',
  title: 'Debug Console',
  position: 'left',
  content: `
    <div class="debug-panel">
      <div id="breakpoints"></div>
      <div id="call-stack"></div>
      <div id="variables"></div>
    </div>
  `,
  scripts: ['debugger-panel.js'],
  styles: ['debugger-panel.css']
};

// Performance monitoring template
const performancePanelTemplate: PanelConfig = {
  id: 'performance',
  title: 'Performance Metrics',
  position: 'floor',
  content: `
    <div class="metrics-panel">
      <canvas id="performance-chart"></canvas>
      <div id="memory-usage"></div>
    </div>
  `
};
```

### **Session Configuration**
```typescript
interface SessionConfig {
  id: string;
  persistent: boolean;      // Save session across disconnects
  maxParticipants: number;  // Collaboration limits
  operatorRestrictions: string[]; // Which operators are allowed
  autoSave: boolean;        // Automatic state snapshots
  conflictResolution: 'last-write' | 'merge' | 'manual';
}

// Create specialized sessions
const pairProgrammingSession: SessionConfig = {
  id: 'pair-programming',
  persistent: true,
  maxParticipants: 2,
  operatorRestrictions: ['ST', 'UP', 'PR', 'CV'], // No destructive ops
  autoSave: true,
  conflictResolution: 'merge'
};
```

---

## 📊 Monitoring & Analytics

### **System Metrics Interface**
```typescript
interface SystemMetrics {
  // Connection metrics
  activeConnections: number;
  sessionsActive: number;
  averageSessionDuration: number;

  // Performance metrics
  eventProcessingRate: number;    // events/second
  averageLatency: number;         // ms
  memoryUsage: number;           // MB

  // Collaboration metrics
  operatorsApplied: {[op: string]: number};
  macrosExecuted: {[macro: string]: number};
  conflictsResolved: number;

  // Quality metrics
  errorRate: number;
  recoveryTime: number;          // ms
  uptime: number;               // percentage
}
```

### **Event Analytics**
```typescript
class EventAnalytics {
  static analyzeOperatorUsage(events: NDJSONEvent[]): OperatorUsageReport {
    // Identify most common operator patterns
    // Detect optimization opportunities
    // Suggest workflow improvements
  }

  static detectCollaborationPatterns(sessions: SessionData[]): CollaborationInsights {
    // Analyze team interaction patterns
    // Identify productive collaboration sequences
    // Suggest optimal session structures
  }

  static measurePerformanceImpact(beforeMetrics: SystemMetrics, afterMetrics: SystemMetrics): ImpactReport {
    // Calculate TDE's impact on development velocity
    // Measure debugging time reduction
    // Quantify collaboration efficiency gains
  }
}
```

### **Health Monitoring**
```typescript
class HealthMonitor {
  private checks = new Map<string, HealthCheck>();

  addCheck(name: string, check: HealthCheck): void {
    this.checks.set(name, check);
  }

  async runAllChecks(): Promise<HealthReport> {
    const results = new Map();

    for (const [name, check] of this.checks) {
      try {
        const result = await check.execute();
        results.set(name, {status: 'healthy', ...result});
      } catch (error) {
        results.set(name, {status: 'unhealthy', error: error.message});
      }
    }

    return {
      timestamp: new Date().toISOString(),
      overall: this.calculateOverallHealth(results),
      details: results
    };
  }
}

// Built-in health checks
const defaultHealthChecks = [
  new WebSocketConnectivityCheck(),
  new SessionIntegrityCheck(),
  new OperatorMappingCheck(),
  new MemoryUsageCheck(),
  new EventProcessingCheck()
];
```

---

## 🔐 Security & Authentication

### **Session Security**
```typescript
interface SecurityConfig {
  enableAuthentication: boolean;
  sessionEncryption: boolean;
  operatorPermissions: Map<string, string[]>; // user -> allowed operators
  maxSessionDuration: number;
  allowAnonymous: boolean;
}

class SessionSecurity {
  static validateUser(userId: string, sessionId: string): boolean {
    // Implement user validation logic
  }

  static checkOperatorPermission(userId: string, operator: string): boolean {
    // Verify user can execute specific operator
  }

  static encryptSessionData(data: any, sessionKey: string): string {
    // Encrypt sensitive session data
  }
}
```

### **Data Privacy**
```typescript
class PrivacyManager {
  static sanitizeEventForLogging(event: NDJSONEvent): NDJSONEvent {
    // Remove sensitive data before logging
    const sanitized = {...event};
    delete sanitized.secretData;
    delete sanitized.credentials;
    return sanitized;
  }

  static enforceRetentionPolicy(events: NDJSONEvent[], retentionDays: number): NDJSONEvent[] {
    const cutoffDate = new Date();
    cutoffDate.setDate(cutoffDate.getDate() - retentionDays);

    return events.filter(event =>
      new Date(event.timestamp) >= cutoffDate
    );
  }
}
```

---

## 🧪 Testing & Validation

### **Component Testing**
```typescript
// Test utilities for TDE components
class TDETestHarness {
  static createMockWebSocketServer(port: number = 9001): MockWebSocketServer {
    return new MockWebSocketServer(port);
  }

  static generateNDJSONEvents(count: number, pattern?: EventPattern): NDJSONEvent[] {
    // Generate realistic test events
  }

  static validateOperatorMapping(mapping: Map<string, string>): ValidationResult {
    // Ensure operator mappings are valid
  }
}

// Integration tests
describe('Tier4RoomBridge Integration', () => {
  let testHarness: TDETestHarness;
  let mockServer: MockWebSocketServer;

  beforeEach(() => {
    testHarness = new TDETestHarness();
    mockServer = testHarness.createMockWebSocketServer();
  });

  test('should map nucleus events to operators correctly', async () => {
    const events = testHarness.generateNDJSONEvents(100, 'nucleus_execution');
    // Test event processing pipeline
  });
});
```

### **Performance Testing**
```typescript
class PerformanceTestSuite {
  static async testConcurrentUsers(userCount: number): Promise<PerformanceReport> {
    // Simulate multiple concurrent users
    // Measure latency and throughput
    // Identify bottlenecks
  }

  static async testEventThroughput(eventsPerSecond: number): Promise<ThroughputReport> {
    // Test event processing capacity
    // Measure queue backlog
    // Validate no event loss
  }

  static async testMemoryLeaks(durationMinutes: number): Promise<MemoryReport> {
    // Long-running memory usage test
    // Detect memory leaks in event buffer
    // Validate garbage collection
  }
}
```

---

This component reference provides deep technical details for developers who need to understand, extend, or integrate with TDE's architecture. Each component is documented with full APIs, configuration options, and practical examples.
