# TDE Troubleshooting & Diagnostics Guide

*Comprehensive problem-solving reference for Tier-4 Distributed Engine issues*

---

## 🚨 Common Issues & Solutions

### **Connection & Network Problems**

#### **Issue: WebSocket Connection Failed**
```
Error: ECONNREFUSED 127.0.0.1:9000
WebSocket connection could not be established
```

**Symptoms**:
- Engine Room interface shows "Disconnected" status
- No real-time events appearing in panels
- Browser console shows WebSocket errors

**Diagnosis Steps**:
```powershell
# 1. Check if relay is running
netstat -an | findstr ":9000"

# 2. Check Node.js process
tasklist | findstr "node.exe"

# 3. Test direct connection
curl -I http://localhost:9000
```

**Solutions**:

**A. Restart WebSocket Relay**
```powershell
# Kill existing processes
taskkill /f /im node.exe

# Restart relay
cd c:\Users\colte\Documents\GitHub\websocket
npm run relay
```

**B. Check Port Availability**
```powershell
# Find process using port 9000
netstat -ano | findstr ":9000"
taskkill /f /pid <PID>

# Start relay on different port
set TDE_WEBSOCKET_PORT=9001
npm run relay
```

**C. Firewall Configuration**
```powershell
# Allow Node.js through Windows Firewall
netsh advfirewall firewall add rule name="TDE WebSocket" dir=in action=allow protocol=TCP localport=9000

# Check Windows Defender settings
Get-NetFirewallRule | Where-Object {$_.DisplayName -like "*Node*"}
```

**D. Network Interface Binding**
```javascript
// In tier4_ws_relay.js, modify binding
const wss = new WebSocket.Server({
  port: this.port,
  host: '0.0.0.0',  // Bind to all interfaces
  clientTracking: true
});
```

---

#### **Issue: Session Synchronization Problems**
```
Warning: Session state inconsistent across clients
Events arriving out of order
```

**Symptoms**:
- Operators applied on one client don't appear on others
- Panels show different content across sessions
- Event timestamps are inconsistent

**Diagnosis Steps**:
```javascript
// Enable debug mode in relay
const relay = new Tier4EnhancedRelay(9000);
relay.enableDebugMode(true);

// Check client synchronization
relay.getStats().activeSessions.forEach(session => {
  console.log(`Session ${session.id}:`);
  console.log(`  Participants: ${session.participants}`);
  console.log(`  Last sync: ${session.lastSync}`);
  console.log(`  Event count: ${session.eventCount}`);
});
```

**Solutions**:

**A. Session Reset**
```typescript
// Force session resync
const bridge = new Tier4RoomBridge(room, 'ws://localhost:9000');
bridge.forceSessionSync('your-session-id');

// Or recreate session
bridge.leaveSession();
bridge.createSession(`new-session-${Date.now()}`);
```

**B. Event Buffer Adjustment**
```javascript
// Increase buffer size for better synchronization
const relay = new Tier4EnhancedRelay(9000);
relay.maxBufferSize = 5000;  // Increase from default 1000

// Reduce save interval for more frequent syncs
relay.setSaveInterval(10000);  // 10 seconds instead of 30
```

**C. Clock Synchronization**
```javascript
// Add NTP synchronization check
class TimeSyncValidator {
  static async validateClientTime(clientTimestamp) {
    const serverTime = Date.now();
    const drift = Math.abs(serverTime - clientTimestamp);

    if (drift > 5000) {  // 5 second tolerance
      throw new Error(`Client clock drift detected: ${drift}ms`);
    }
  }
}
```

---

### **Performance & Memory Issues**

#### **Issue: High Memory Usage**
```
Warning: Heap usage approaching limit
EventBuffer consuming excessive memory
```

**Symptoms**:
- Slow response times in Engine Room
- Browser tab crashes
- Node.js process memory warnings

**Diagnosis Steps**:
```javascript
// Memory profiling in relay
setInterval(() => {
  const usage = process.memoryUsage();
  console.log(JSON.stringify({
    type: 'memory_diagnostic',
    heapUsed: (usage.heapUsed / 1024 / 1024).toFixed(2) + ' MB',
    heapTotal: (usage.heapTotal / 1024 / 1024).toFixed(2) + ' MB',
    external: (usage.external / 1024 / 1024).toFixed(2) + ' MB',
    eventBufferSize: this.eventBuffer.length,
    clientCount: this.clients.size
  }));
}, 30000);
```

**Solutions**:

**A. Event Buffer Management**
```javascript
class OptimizedEventBuffer {
  constructor(maxSize = 1000, compressionThreshold = 500) {
    this.events = [];
    this.maxSize = maxSize;
    this.compressionThreshold = compressionThreshold;
  }

  add(event) {
    // Compress old events when threshold reached
    if (this.events.length > this.compressionThreshold) {
      this.compressOldEvents();
    }

    this.events.push(event);

    if (this.events.length > this.maxSize) {
      this.events.shift();  // Remove oldest
    }
  }

  compressOldEvents() {
    // Keep only essential data for old events
    const compressionPoint = Math.floor(this.events.length / 2);
    for (let i = 0; i < compressionPoint; i++) {
      this.events[i] = this.compressEvent(this.events[i]);
    }
  }

  compressEvent(event) {
    return {
      type: event.type,
      role: event.role,
      timestamp: event.timestamp,
      // Remove detailed data for old events
      data: { compressed: true, originalSize: JSON.stringify(event.data).length }
    };
  }
}
```

**B. Client Connection Limits**
```javascript
class ConnectionLimiter {
  constructor(maxConnections = 100) {
    this.maxConnections = maxConnections;
    this.connectionQueue = [];
  }

  handleNewConnection(ws) {
    if (this.clients.size >= this.maxConnections) {
      // Queue connection or reject
      this.connectionQueue.push(ws);
      ws.send(JSON.stringify({
        type: 'connection_queued',
        position: this.connectionQueue.length,
        estimated_wait: this.connectionQueue.length * 2000
      }));
      return false;
    }

    this.acceptConnection(ws);
    return true;
  }
}
```

**C. Garbage Collection Optimization**
```javascript
// Periodic cleanup
setInterval(() => {
  // Clean up disconnected clients
  for (const [ws, clientInfo] of this.clients.entries()) {
    if (ws.readyState === WebSocket.CLOSED) {
      this.clients.delete(ws);
    }
  }

  // Clean up empty sessions
  for (const [sessionId, session] of this.sessions.entries()) {
    if (session.participants.size === 0) {
      this.sessions.delete(sessionId);
    }
  }

  // Force garbage collection in development
  if (process.env.NODE_ENV === 'development' && global.gc) {
    global.gc();
  }
}, 60000);  // Every minute
```

---

#### **Issue: Slow Event Processing**
```
Warning: Event processing lag detected
NDJSON events backing up in queue
```

**Symptoms**:
- Delays between events and UI updates
- Operators not applying in real-time
- Event buffer growing rapidly

**Diagnosis Steps**:
```javascript
class EventProcessingProfiler {
  constructor() {
    this.processingTimes = [];
    this.queueSizes = [];
  }

  profileEvent(event, processingTime) {
    this.processingTimes.push(processingTime);
    this.queueSizes.push(this.eventBuffer.length);

    if (this.processingTimes.length > 100) {
      this.processingTimes.shift();
      this.queueSizes.shift();
    }

    // Log performance metrics
    if (this.processingTimes.length === 100) {
      const avgProcessingTime = this.processingTimes.reduce((a, b) => a + b) / 100;
      const avgQueueSize = this.queueSizes.reduce((a, b) => a + b) / 100;

      console.log(JSON.stringify({
        type: 'performance_metrics',
        avgProcessingTime: avgProcessingTime.toFixed(2) + 'ms',
        avgQueueSize: avgQueueSize.toFixed(0),
        processingRate: (1000 / avgProcessingTime).toFixed(0) + ' events/sec'
      }));
    }
  }
}
```

**Solutions**:

**A. Batch Event Processing**
```javascript
class BatchEventProcessor {
  constructor(batchSize = 10, batchInterval = 100) {
    this.eventQueue = [];
    this.batchSize = batchSize;
    this.batchInterval = batchInterval;

    setInterval(() => this.processBatch(), batchInterval);
  }

  addEvent(event) {
    this.eventQueue.push(event);
  }

  processBatch() {
    if (this.eventQueue.length === 0) return;

    const batch = this.eventQueue.splice(0, this.batchSize);
    const startTime = Date.now();

    // Process events in batch
    const processedEvents = batch.map(event => this.enhanceEvent(event));

    // Broadcast batch to all clients
    this.broadcastBatch(processedEvents);

    const processingTime = Date.now() - startTime;
    this.profiler.profileEvent(batch, processingTime);
  }

  broadcastBatch(events) {
    const message = JSON.stringify({
      type: 'event_batch',
      events: events,
      timestamp: Date.now()
    });

    for (const ws of this.clients.keys()) {
      if (ws.readyState === WebSocket.OPEN) {
        ws.send(message);
      }
    }
  }
}
```

**B. Selective Event Processing**
```javascript
class SelectiveEventProcessor {
  constructor() {
    this.priorityMap = new Map([
      ['nucleus_exec', 10],      // Highest priority
      ['memory_store', 8],
      ['cycle_start', 9],
      ['cycle_end', 7],
      ['loop_back', 8],
      ['raw_input', 1]           // Lowest priority
    ]);
  }

  processEvent(event) {
    const priority = this.priorityMap.get(event.type) || 5;

    // Skip low-priority events if queue is backing up
    if (this.eventQueue.length > 100 && priority < 5) {
      return; // Skip this event
    }

    // Process high-priority events immediately
    if (priority >= 9) {
      this.processImmediately(event);
    } else {
      this.addToQueue(event);
    }
  }
}
```

---

### **Data & State Issues**

#### **Issue: State Snapshots Not Saving**
```
Error: Failed to save state snapshot
CID generation failed
```

**Symptoms**:
- Right panel (State Management) shows no snapshots
- "Save Snapshot" button doesn't work
- Console shows CID generation errors

**Diagnosis Steps**:
```javascript
// Test snapshot functionality
class SnapshotDiagnostic {
  static async testSnapshotSaving() {
    const testState = {
      operators: ['ST', 'UP', 'PR'],
      timestamp: Date.now(),
      sessionId: 'diagnostic-test'
    };

    try {
      const cid = await this.generateCID(testState);
      console.log(`✅ CID generation successful: ${cid}`);

      const saved = await this.saveSnapshot(cid, testState);
      console.log(`✅ Snapshot save successful: ${saved}`);

      const loaded = await this.loadSnapshot(cid);
      console.log(`✅ Snapshot load successful: ${JSON.stringify(loaded) === JSON.stringify(testState)}`);

    } catch (error) {
      console.error(`❌ Snapshot diagnostic failed: ${error.message}`);
    }
  }
}
```

**Solutions**:

**A. Fix CID Generation**
```javascript
class RobustCIDGenerator {
  static async generateCID(data) {
    try {
      // Ensure consistent serialization
      const normalizedData = this.normalizeData(data);
      const jsonString = JSON.stringify(normalizedData, Object.keys(normalizedData).sort());

      // Use Web Crypto API for hash generation
      const encoder = new TextEncoder();
      const dataBuffer = encoder.encode(jsonString);
      const hashBuffer = await crypto.subtle.digest('SHA-256', dataBuffer);

      // Convert to hex string
      const hashArray = Array.from(new Uint8Array(hashBuffer));
      const hashHex = hashArray.map(b => b.toString(16).padStart(2, '0')).join('');

      return `cid_${hashHex.substring(0, 16)}`; // Truncate for readability

    } catch (error) {
      // Fallback to simple hash
      console.warn('CID generation failed, using fallback');
      return `cid_fallback_${Date.now()}_${Math.random().toString(36).substring(2)}`;
    }
  }

  static normalizeData(data) {
    // Remove timestamp variations that would affect CID
    const normalized = JSON.parse(JSON.stringify(data));
    if (normalized.timestamp) {
      delete normalized.timestamp;
    }
    return normalized;
  }
}
```

**B. Implement Snapshot Persistence**
```javascript
class SnapshotStorage {
  constructor() {
    this.storageKey = 'tde_snapshots';
    this.maxSnapshots = 100;
  }

  async saveSnapshot(cid, data) {
    try {
      const snapshots = this.getStoredSnapshots();

      snapshots[cid] = {
        data: data,
        created: new Date().toISOString(),
        size: JSON.stringify(data).length
      };

      // Limit snapshot count
      const snapshotKeys = Object.keys(snapshots);
      if (snapshotKeys.length > this.maxSnapshots) {
        // Remove oldest snapshots
        const sortedKeys = snapshotKeys.sort((a, b) =>
          new Date(snapshots[a].created) - new Date(snapshots[b].created)
        );

        const toRemove = sortedKeys.slice(0, snapshotKeys.length - this.maxSnapshots);
        toRemove.forEach(key => delete snapshots[key]);
      }

      localStorage.setItem(this.storageKey, JSON.stringify(snapshots));
      return cid;

    } catch (error) {
      throw new Error(`Snapshot save failed: ${error.message}`);
    }
  }

  async loadSnapshot(cid) {
    try {
      const snapshots = this.getStoredSnapshots();
      const snapshot = snapshots[cid];

      if (!snapshot) {
        throw new Error(`Snapshot ${cid} not found`);
      }

      return snapshot.data;

    } catch (error) {
      throw new Error(`Snapshot load failed: ${error.message}`);
    }
  }

  getStoredSnapshots() {
    try {
      const stored = localStorage.getItem(this.storageKey);
      return stored ? JSON.parse(stored) : {};
    } catch (error) {
      console.warn('Failed to parse stored snapshots, resetting');
      localStorage.removeItem(this.storageKey);
      return {};
    }
  }

  listSnapshots() {
    const snapshots = this.getStoredSnapshots();
    return Object.keys(snapshots).map(cid => ({
      cid: cid,
      created: snapshots[cid].created,
      size: snapshots[cid].size
    })).sort((a, b) => new Date(b.created) - new Date(a.created));
  }
}
```

---

#### **Issue: Operator Mappings Not Working**
```
Warning: Unknown nucleus event type
Operator mapping failed for event
```

**Symptoms**:
- NDJSON events processed but no operators applied
- Front panel shows no operator activity
- Console shows mapping warnings

**Diagnosis Steps**:
```javascript
class OperatorMappingDiagnostic {
  static validateMappings() {
    const diagnostics = {
      nucleusMappings: {},
      memoryMappings: {},
      unknownEvents: []
    };

    // Test nucleus mappings
    const nucleusEvents = ['VIBRATE', 'OPTIMIZATION', 'STATE', 'SEED'];
    nucleusEvents.forEach(role => {
      const operator = this.nucleusToOperator.get(role);
      diagnostics.nucleusMappings[role] = operator || 'MISSING';
    });

    // Test memory mappings
    const memoryTags = ['energy', 'refined', 'condition', 'seed'];
    memoryTags.forEach(tag => {
      const operator = this.tagToOperator.get(tag);
      diagnostics.memoryMappings[tag] = operator || 'MISSING';
    });

    console.log('🔍 Operator Mapping Diagnostics:', JSON.stringify(diagnostics, null, 2));
    return diagnostics;
  }

  static testEventMapping(event) {
    console.log(`🧪 Testing event mapping for: ${JSON.stringify(event)}`);

    const enhanced = this.enhanceWithTier4Mapping(event);

    if (enhanced.tier4_operator) {
      console.log(`✅ Successfully mapped to operator: ${enhanced.tier4_operator}`);
    } else {
      console.log(`❌ No operator mapping found for event type: ${event.type}`);
    }

    return enhanced;
  }
}
```

**Solutions**:

**A. Fix Mapping Configuration**
```javascript
class RobustOperatorMapping {
  constructor() {
    // Comprehensive nucleus mappings
    this.nucleusToOperator = new Map([
      ['VIBRATE', 'ST'],         // Stabilization
      ['OPTIMIZATION', 'UP'],    // Update/Progress
      ['STATE', 'CV'],           // Convergence
      ['SEED', 'RB'],            // Rollback

      // Alternative event names
      ['INITIALIZE', 'ST'],
      ['OPTIMIZE', 'UP'],
      ['PROCESS', 'UP'],
      ['CONVERGE', 'CV'],
      ['FINALIZE', 'CV'],
      ['RESET', 'RB'],
      ['ROLLBACK', 'RB']
    ]);

    // Comprehensive memory tag mappings
    this.tagToOperator = new Map([
      ['energy', 'CH'],          // Change
      ['refined', 'PR'],         // Progress
      ['condition', 'SL'],       // Selection
      ['seed', 'MD'],            // Multidimensional

      // Alternative tag names
      ['power', 'CH'],
      ['improvement', 'PR'],
      ['optimization', 'PR'],
      ['constraint', 'SL'],
      ['filter', 'SL'],
      ['dimension', 'MD'],
      ['space', 'MD']
    ]);

    // Fallback mapping
    this.fallbackOperator = 'CV'; // Default to Convergence
  }

  enhanceWithTier4Mapping(event) {
    const enhanced = { ...event };
    let operator = null;

    // Try nucleus mapping
    if (event.type === 'nucleus_exec' && event.role) {
      operator = this.nucleusToOperator.get(event.role.toUpperCase());
    }

    // Try memory mapping
    if (event.type === 'memory_store' && event.tag) {
      operator = this.tagToOperator.get(event.tag.toLowerCase());
    }

    // Try pattern matching for unknown events
    if (!operator) {
      operator = this.inferOperatorFromContent(event);
    }

    // Use fallback if still no mapping
    if (!operator) {
      operator = this.fallbackOperator;
      enhanced.mapping_note = 'Using fallback operator';
    }

    enhanced.tier4_operator = operator;
    enhanced.tier4_mapping = event.role ? `${event.role} → ${operator}` :
                           event.tag ? `${event.tag} → ${operator}` :
                           `inferred → ${operator}`;

    return enhanced;
  }

  inferOperatorFromContent(event) {
    const content = JSON.stringify(event).toLowerCase();

    // Pattern matching for operator inference
    if (content.includes('start') || content.includes('init')) return 'ST';
    if (content.includes('update') || content.includes('progress')) return 'UP';
    if (content.includes('change') || content.includes('modify')) return 'CH';
    if (content.includes('improve') || content.includes('refine')) return 'PR';
    if (content.includes('select') || content.includes('filter')) return 'SL';
    if (content.includes('converge') || content.includes('complete')) return 'CV';
    if (content.includes('rollback') || content.includes('reset')) return 'RB';
    if (content.includes('multi') || content.includes('dimension')) return 'MD';

    return null; // No inference possible
  }
}
```

---

### **Browser & UI Issues**

#### **Issue: Engine Room Interface Not Loading**
```
Error: Unable to load Engine Room panels
HTML panels not rendering correctly
```

**Symptoms**:
- Blank Engine Room interface
- Panels not positioned correctly
- Missing drag-and-drop functionality

**Diagnosis Steps**:
```javascript
// Browser console diagnostics
function diagnoseEngineRoom() {
  console.log('🔍 Engine Room Diagnostics');

  // Check if message bus is loaded
  if (typeof EngineRoomMessageBus === 'undefined') {
    console.error('❌ EngineRoomMessageBus not loaded');
  } else {
    console.log('✅ Message bus available');
  }

  // Check panel container
  const container = document.querySelector('.engine-room-container');
  if (!container) {
    console.error('❌ Engine Room container not found');
  } else {
    console.log('✅ Container found:', container);
  }

  // Check WebSocket connection
  if (window.engineRoomWS) {
    console.log('✅ WebSocket status:', window.engineRoomWS.readyState);
  } else {
    console.error('❌ WebSocket not initialized');
  }

  // Check panels
  const panels = document.querySelectorAll('.room-panel');
  console.log(`📊 Found ${panels.length} panels`);
  panels.forEach((panel, index) => {
    console.log(`  Panel ${index}: ${panel.id} at position ${panel.style.position}`);
  });
}

// Run in browser console
diagnoseEngineRoom();
```

**Solutions**:

**A. Fix Panel Positioning**
```css
/* Add to worldengine.html styles */
.engine-room-container {
  position: relative;
  width: 100vw;
  height: 100vh;
  overflow: hidden;
  background: #0a0f16;
}

.room-panel {
  position: absolute;
  background: rgba(15, 26, 43, 0.95);
  border: 1px solid rgba(100, 160, 200, 0.2);
  border-radius: 8px;
  padding: 12px;
  min-width: 200px;
  min-height: 150px;
  resize: both;
  overflow: auto;
  z-index: 100;
}

.room-panel.dragging {
  z-index: 1000;
  cursor: move;
}

/* Semantic wall positions */
.wall-front { top: 40%; left: 45%; }
.wall-left { top: 20%; left: 5%; }
.wall-right { top: 20%; right: 5%; }
.wall-back { top: 60%; left: 45%; }
.wall-floor { bottom: 5%; left: 20%; }
.wall-ceiling { top: 5%; left: 45%; }
```

**B. Fix Drag and Drop**
```javascript
class RobustDragDrop {
  constructor() {
    this.dragState = {
      isDragging: false,
      element: null,
      offset: { x: 0, y: 0 }
    };

    this.setupEventListeners();
  }

  setupEventListeners() {
    document.addEventListener('mousedown', (e) => this.handleMouseDown(e));
    document.addEventListener('mousemove', (e) => this.handleMouseMove(e));
    document.addEventListener('mouseup', (e) => this.handleMouseUp(e));

    // Touch support
    document.addEventListener('touchstart', (e) => this.handleTouchStart(e));
    document.addEventListener('touchmove', (e) => this.handleTouchMove(e));
    document.addEventListener('touchend', (e) => this.handleTouchEnd(e));
  }

  handleMouseDown(e) {
    const panel = e.target.closest('.room-panel');
    if (!panel) return;

    // Only drag from title bar
    const titleBar = e.target.closest('.panel-title');
    if (!titleBar) return;

    this.startDrag(panel, e.clientX, e.clientY);
  }

  startDrag(element, x, y) {
    this.dragState.isDragging = true;
    this.dragState.element = element;

    const rect = element.getBoundingClientRect();
    this.dragState.offset = {
      x: x - rect.left,
      y: y - rect.top
    };

    element.classList.add('dragging');
  }

  handleMouseMove(e) {
    if (!this.dragState.isDragging) return;

    e.preventDefault();
    const element = this.dragState.element;

    const newX = e.clientX - this.dragState.offset.x;
    const newY = e.clientY - this.dragState.offset.y;

    // Constrain to viewport
    const maxX = window.innerWidth - element.offsetWidth;
    const maxY = window.innerHeight - element.offsetHeight;

    element.style.left = Math.max(0, Math.min(maxX, newX)) + 'px';
    element.style.top = Math.max(0, Math.min(maxY, newY)) + 'px';
  }

  handleMouseUp(e) {
    if (!this.dragState.isDragging) return;

    this.dragState.element.classList.remove('dragging');
    this.dragState.isDragging = false;
    this.dragState.element = null;
  }
}

// Initialize drag and drop
document.addEventListener('DOMContentLoaded', () => {
  new RobustDragDrop();
});
```

---

### **Development & Debug Issues**

#### **Issue: NDJSON Events Not Being Generated**
```
No events appearing in TDE despite code execution
Application running but no nucleus events detected
```

**Symptoms**:
- Engine Room shows no activity
- Event buffer remains empty
- No operator mappings triggered

**Solutions**:

**A. Add Event Generation to Existing Code**
```javascript
// Instrumentation wrapper for any function
function tdeInstrument(fn, eventConfig) {
  return function(...args) {
    const startTime = Date.now();

    // VIBRATE→ST: Function entry
    console.log(JSON.stringify({
      type: 'nucleus_exec',
      role: 'VIBRATE',
      data: {
        function: fn.name,
        args: args.length,
        timestamp: startTime
      }
    }));

    try {
      const result = fn.apply(this, args);
      const executionTime = Date.now() - startTime;

      // Handle async functions
      if (result instanceof Promise) {
        return result.then(asyncResult => {
          // STATE→CV: Async completion
          console.log(JSON.stringify({
            type: 'nucleus_exec',
            role: 'STATE',
            data: {
              function: fn.name,
              executionTime,
              async: true,
              success: true
            }
          }));
          return asyncResult;
        }).catch(error => {
          // SEED→RB: Async error
          console.log(JSON.stringify({
            type: 'nucleus_exec',
            role: 'SEED',
            data: {
              function: fn.name,
              executionTime: Date.now() - startTime,
              async: true,
              error: error.message
            }
          }));
          throw error;
        });
      } else {
        // OPTIMIZATION→UP: Sync completion
        console.log(JSON.stringify({
          type: 'nucleus_exec',
          role: 'OPTIMIZATION',
          data: {
            function: fn.name,
            executionTime,
            success: true
          }
        }));
        return result;
      }

    } catch (error) {
      // SEED→RB: Sync error
      console.log(JSON.stringify({
        type: 'nucleus_exec',
        role: 'SEED',
        data: {
          function: fn.name,
          executionTime: Date.now() - startTime,
          error: error.message
        }
      }));
      throw error;
    }
  };
}

// Usage example
const originalFunction = function processData(data) {
  // Your existing code
  return data.map(item => item * 2);
};

// Wrap with TDE instrumentation
const instrumentedFunction = tdeInstrument(originalFunction, {
  generateEvents: true
});
```

**B. Auto-Instrumentation for Classes**
```javascript
class TDEAutoInstrument {
  static instrumentClass(targetClass, options = {}) {
    const prototype = targetClass.prototype;
    const methodNames = Object.getOwnPropertyNames(prototype);

    methodNames.forEach(methodName => {
      if (methodName === 'constructor') return;
      if (typeof prototype[methodName] !== 'function') return;
      if (options.exclude && options.exclude.includes(methodName)) return;

      const originalMethod = prototype[methodName];
      prototype[methodName] = tdeInstrument(originalMethod, {
        className: targetClass.name,
        methodName: methodName,
        ...options
      });
    });

    return targetClass;
  }
}

// Usage
@TDEAutoInstrument.instrumentClass
class DataProcessor {
  async fetchData() {
    // Automatically instrumented
  }

  processItems(items) {
    // Automatically instrumented
  }
}
```

---

This comprehensive troubleshooting guide covers the most common issues users encounter when working with TDE, along with systematic diagnostic approaches and proven solutions.
