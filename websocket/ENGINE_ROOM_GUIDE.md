# Tier-4 Engine Room Integration - Complete Setup Guide

🎯 **Transform your Gridroom into the "head nucleus" for distributed Tier-4 reasoning**

This integration turns your existing Gridroom HTML interface into a powerful Engine Room that serves as the central command center for your Tier-4 Meta System, complete with real-time WebSocket streaming, NDJSON nucleus event processing, and collaborative state management.

## 🚀 Quick Start

### 1. Files Overview

**Core Integration Files:**
- `worldengine.html` - ✅ **PATCHED** with Engine Room Message Bus
- `roomAdapter.ts` - TypeScript adapter connecting Tier-4 → Gridroom  
- `tier4_room_integration.ts` - Bridge connecting WebSocket system to room
- `EngineRoom.tsx` - React component wrapper with hooks
- `tier4_engine_room_demo.tsx` - Complete demo showing all features

**Supporting Files:**
- `tier4_ws_relay.js` - Enhanced WebSocket relay (from previous integration)
- `tier4_websocket_sync.ts` - WebSocket ↔ Tier-4 bridge (from previous integration)

### 2. Installation

```bash
# Install dependencies
npm install ws react react-dom @types/react @types/react-dom typescript

# Start the enhanced WebSocket relay
node tier4_ws_relay.js

# In your React app, import and use:
import EngineRoom from './EngineRoom';
```

### 3. Basic Usage

```tsx
import React, { useRef } from 'react';
import EngineRoom, { EngineRoomRef } from './EngineRoom';

function MyApp() {
  const roomRef = useRef<EngineRoomRef>(null);

  const handleOperator = (op: string) => {
    roomRef.current?.applyOperator(op);
  };

  return (
    <div style={{ height: '100vh' }}>
      <EngineRoom
        ref={roomRef}
        iframeUrl="/worldengine.html"
        websocketUrl="ws://localhost:9000"
        title="My Tier-4 Engine Room"
        onOperatorApplied={(op, prev, next) => {
          console.log(`Applied ${op}:`, { prev, next });
        }}
        onRoomReady={(bridge) => {
          console.log('Engine Room ready!', bridge);
        }}
      />
    </div>
  );
}
```

## 🎮 Features Included

### ✅ **Patched Gridroom (worldengine.html)**
- **Message Bus**: Full postMessage API for IDE ↔ Room communication
- **Panel System**: Dynamic panel creation, dragging, HTML content injection
- **Auto Panels**: Events, Snapshots, and Nucleus Monitor panels
- **Local Storage**: Persistent sessions and state management
- **Visual Integration**: Connects to existing lexicon visualization

### ✅ **TypeScript Room Adapter** 
- **Type-Safe API**: Complete TypeScript interfaces for all operations
- **Semantic Layout**: Pre-configured wall positions for different content types
- **State Management**: Content-addressed state snapshots with CID generation
- **Event Pipeline**: publishSnapshot(), publishEvent(), operator tracking

### ✅ **WebSocket Bridge Integration**
- **NDJSON Processing**: Auto-converts nucleus events → Tier-4 operators
- **Real-time Sync**: Bidirectional state synchronization across clients
- **Auto-mapping**: VIBRATE→ST, OPTIMIZATION→UP, STATE→CV, SEED→RB
- **Reconnection**: Automatic reconnection with exponential backoff

### ✅ **React Component Wrapper**
- **Hook Support**: useEngineRoom hook for functional components
- **Ref Interface**: Full imperative API through useRef
- **Status Indicators**: Visual connection and readiness status  
- **Error Handling**: Graceful degradation with error UI

### ✅ **Semantic Wall Layout**
- **Front Wall**: Operator Console + Nucleus Monitor
- **Left Wall**: Events and Activity Log
- **Right Wall**: State Snapshots and History
- **Back Wall**: Documentation and Notes
- **Floor**: Performance Metrics and Charts
- **Ceiling**: System Health Monitoring

## 🧠 Nucleus → Tier-4 Mappings

### **Nucleus Events → Operators**
```javascript
VIBRATE → ST (Stabilization)
OPTIMIZATION → UP (Update/Progress)
STATE → CV (Convergence) 
SEED → RB (Rollback)
```

### **Memory Tags → Operators**
```javascript
energy → CH (Change)
refined → PR (Progress)
condition → SL (Selection)
seed → MD (Multidimensional)
```

### **Cycle Events → Macros**
```javascript
cycle_start → IDE_A, IDE_B, or MERGE_ABC
(Auto-selected based on cycle position and context)
```

## 📊 State Vector Dynamics

The system operates on Tier-4 state vectors `[p, i, g, c]`:
- **p**: Persistence/stability
- **i**: Information/input
- **g**: Goal/direction  
- **c**: Confidence/certainty
- **κ (kappa)**: Global confidence parameter (0.0-1.0)
- **level**: System complexity level (integer)

## 🔄 Integration Flow

```
1. NDJSON Stream → WebSocket Relay → Engine Room Message Bus
2. Nucleus Events → Auto-map to Tier-4 Operators → Apply Transformations
3. State Updates → Broadcast to All Clients → Update Visualizations
4. User Actions → Manual Operator/Macro Execution → State Evolution
5. Snapshots → Content-Addressed Storage → Time Travel Navigation
```

## 🎯 Wall-based Semantic Organization

### **Front Wall - Command Center**
- **Nucleus Monitor**: Real-time state vector display with animated visualization
- **Operator Console**: Manual Tier-4 operator controls (ST, UP, PR, CV, RB, RS)
- **Macro Controls**: Three Ides macro execution (IDE_A, IDE_B, MERGE_ABC)

### **Left Wall - Activity Stream** 
- **Events Panel**: Live feed of all operations, nucleus events, state changes
- **Operation Log**: Historical record of transformations with timestamps
- **Nucleus Activity**: Real-time display of VIBRATE, OPTIMIZATION, STATE, SEED events

### **Right Wall - State Management**
- **Snapshots Panel**: Content-addressed state history with Load buttons
- **State Browser**: Navigate through state evolution with CID references  
- **Collaboration**: Multi-user state synchronization and conflict resolution

### **Back Wall - Documentation**
- **Tier-4 Reference**: Operator documentation and mathematical definitions
- **Integration Guide**: How-to information and API references
- **Session Notes**: Persistent notes and annotations

### **Floor - Performance**
- **Metrics Dashboard**: Operations/sec, κ changes, memory usage, latency
- **Performance Charts**: Real-time graphs of system performance
- **State Statistics**: Dimensional analysis and convergence metrics

### **Ceiling - System Health**
- **Connection Status**: WebSocket, room readiness, participant count
- **Resource Monitor**: Memory usage, FPS, browser performance
- **Error Tracking**: System alerts and diagnostics

## 🛠️ Advanced Usage

### **Custom Panel Creation**
```tsx
// Add custom visualization panel
room.addPanel({
  wall: 'floor',
  x: 100, y: 200, w: 400, h: 300,
  title: 'Custom Analytics',
  html: `<!doctype html>
    <meta charset="utf-8">
    <style>/* your styles */</style>
    <div>Your custom content with live data updates</div>
    <script>
      window.addEventListener('message', (e) => {
        if (e.data.type === 'update-data') {
          // Handle real-time data updates
        }
      });
    </script>`
});
```

### **State Time Travel**
```tsx
// Load specific state by CID
window.addEventListener('tier4-load-state', (event) => {
  const { state, cid } = event.detail;
  console.log(`Time travel to ${cid}:`, state);
  // Your state restoration logic here
});
```

### **Custom Nucleus Events**
```tsx
// Inject custom nucleus events
bridge.triggerNucleusEvent('VIBRATE'); // Triggers ST operator
bridge.triggerNucleusEvent('OPTIMIZATION'); // Triggers UP operator

// Or send raw NDJSON events
const customEvent = {
  type: 'nucleus_exec',
  ts: new Date().toISOString(),
  id: 'N_custom',
  role: 'CUSTOM_ROLE',
  tier4_operator: 'CV' // Will trigger CV operator
};
```

## 🔧 Configuration

### **WebSocket Settings**
```typescript
// Custom WebSocket URL and reconnection settings
const bridge = new Tier4RoomBridge(roomWindow, 'ws://your-server:9000');
```

### **Operator Customization**
```typescript
// Add custom operators to tier4_room_integration.ts
private operators: Record<string, Tier4Operator> = {
  // ... existing operators
  CUSTOM: { 
    name: 'Custom',
    matrix: [[1.2,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],
    bias: [0.1,0,0,0],
    kappaDelta: 0.15
  }
};
```

### **Panel Layout Customization**
```typescript
// Customize semantic positions in roomAdapter.ts
getPosition(semantic: "events" | "snapshots" | "nucleus" | /* ... */) {
  const positions = {
    events: { wall: "left", x: 60, y: 260, w: 420, h: 260 },
    // Modify positions as needed
  };
}
```

## 📝 API Reference

### **EngineRoom Component Props**
```typescript
interface EngineRoomProps {
  iframeUrl?: string;           // Engine room HTML file
  websocketUrl?: string;        // WebSocket relay URL  
  sessionId?: string;           // Session identifier
  title?: string;               // Room title
  initialState?: Tier4State;    // Starting state vector
  onOperatorApplied?: (op, prev, next) => void;
  onStateLoaded?: (state, cid) => void;
  onRoomReady?: (bridge) => void;
  onConnectionStatus?: (connected) => void;
}
```

### **EngineRoom Ref Methods**
```typescript
interface EngineRoomRef {
  applyOperator(operator: string, meta?: any): void;
  applyMacro(macro: string): void;
  getCurrentState(): Tier4State;
  setState(state: Tier4State): void;
  triggerNucleusEvent(role): void;
  toast(message: string): void;
  addPanel(config): void;
  getBridge(): Tier4RoomBridge | null;
  isConnected(): boolean;
}
```

## 🚦 Testing Checklist

- [ ] **Room Initialization**: Engine Room loads and shows "Ready" status
- [ ] **WebSocket Connection**: Shows "Connected" and processes NDJSON events  
- [ ] **Operator Application**: Manual operators update state vector and broadcast
- [ ] **Nucleus Auto-mapping**: VIBRATE → ST, OPTIMIZATION → UP, etc.
- [ ] **Macro Execution**: IDE_A, IDE_B, MERGE_ABC run operator sequences
- [ ] **State Snapshots**: CID generation, storage, and Load button functionality
- [ ] **Panel System**: Events, Snapshots, Nucleus Monitor auto-created and updating
- [ ] **Collaborative Sync**: Multiple clients share state changes in real-time
- [ ] **Time Travel**: Loading historical states from Snapshots panel works
- [ ] **Custom Panels**: addPanel() creates draggable panels with HTML content

## 🎉 Result

Your Gridroom is now the **"head nucleus"** - a distributed command center that:

✅ **Visualizes** Tier-4 state evolution in real-time  
✅ **Processes** NDJSON nucleus events automatically  
✅ **Synchronizes** state across multiple collaborative clients  
✅ **Provides** semantic organization with wall-based panels  
✅ **Enables** time travel through content-addressed state history  
✅ **Integrates** with existing WebSocket infrastructure seamlessly  

**The room becomes your single source of truth for all Tier-4 operations, turning your glowing ball visualization into a fully-featured distributed reasoning engine!** 🌟