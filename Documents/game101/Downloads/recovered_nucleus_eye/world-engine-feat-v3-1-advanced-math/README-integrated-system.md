# World Engine Studio - Integrated System

A unified architecture that combines Chat, Engine, and Recorder components into a scalable, orchestrated system. This eliminates standalone pieces in favor of integrated components that work better together.

## 🏗️ Architecture Overview

### Core Philosophy
"There's no point in having a standalone piece if it could scale better with another piece to still do its job"

The system uses a **central orchestrator pattern** to coordinate all components through an event bus, enabling:
- **Queue management** with backpressure control
- **Retry logic** with exponential backoff
- **Multi-engine pools** for parallel processing
- **Unified error handling** across components

### Component Structure
```
/controllers/           # Core control logic
  ├── chat-controller.js      # Enhanced chat with transcript handling
  ├── engine-controller.js    # Multi-iframe engine pool
  └── recorder-controller.js  # Multi-format recording sessions

/models/               # Data processing logic
  └── world-engine.js        # Unified RGCN/GAT encoder registry

/schemas/              # Type validation
  └── validation.js          # Shared schemas and validators

/utils/                # Common utilities
  └── helpers.js            # Cross-component helper functions

orchestrator.js        # Central coordinator
studio.js             # Main integration layer
demo.html             # Interactive demonstration
```

## 🚀 Key Features

### 1. Central Orchestration
- **StudioOrchestrator** manages all inter-component communication
- Event bus architecture for loose coupling
- Queue management with priority handling
- Automatic retry with exponential backoff

### 2. Enhanced Controllers

#### ChatController
- Unified message handling (eliminates duplicate classes)
- Enhanced command parsing with validation
- Transcript management with metadata
- Orchestrator integration for smart routing

#### EngineController
- Multi-iframe engine pool for parallel processing
- MutationObserver for reliable result detection
- Proper message re-dispatch when engines not ready
- Active run lifecycle management

#### RecorderController
- Multi-format stream handling (audio/video)
- MIME type auto-detection
- Multiple concurrent recording sessions
- Proper stream cleanup and blob management

### 3. Unified Model System
- **WorldEngine** with swappable encoder architectures
- Support for RGCN, GAT, and Hybrid encoders
- Runtime encoder switching via configuration
- Token processing with graph neural networks

### 4. Type Safety
- Lightweight validation system (no external dependencies)
- Shared schemas across components
- Runtime type checking with helpful error messages

## 📋 Usage Examples

### Basic Integration
```javascript
import { createStudio } from './studio.js';

// Create integrated studio
const studio = createStudio({
    maxConcurrentRuns: 3,
    enginePoolSize: 2,
    enableRecording: true,
    encoder: 'rgcn'
});

// Send messages
await studio.sendMessage("Analyze this text");

// Switch encoders dynamically
await studio.switchEncoder('gat');

// Start recording
await studio.startRecording();
```

### Event Handling
```javascript
// Listen for results
studio.onEngineResult((result) => {
    console.log('Engine result:', result);
});

// Listen for recordings
studio.onRecordingComplete((recording) => {
    console.log('Recording ready:', recording.url);
});

// Error handling
studio.onError((error) => {
    console.error('System error:', error);
});
```

### Component Access
```javascript
// Get system status
const status = studio.getStatus();
console.log('Components:', status.components);

// Access individual components
const orchestrator = studio.components.get('orchestrator');
const worldEngine = studio.components.get('worldEngine');
```

## 🔧 Configuration Options

### Studio Options
```javascript
{
    maxConcurrentRuns: 3,      // Max parallel engine runs
    enginePoolSize: 2,         // Number of engine iframes
    enableRecording: true,     // Enable recording component
    encoder: 'rgcn'           // Default encoder: 'rgcn'|'gat'|'hybrid'
}
```

### Component-Specific Options
```javascript
// Chat Controller
{
    enableTranscripts: true,
    maxHistory: 100,
    commandPrefix: '/'
}

// Engine Controller
{
    resultTimeout: 10000,
    retryDelay: 100,
    poolSize: 2
}

// Recorder Controller
{
    video: true,
    audio: true,
    maxDuration: 120000,
    bitrate: 1000000
}

// World Engine
{
    encoder: 'rgcn',
    hiddenSize: 512,
    numLayers: 3,
    dropout: 0.1,
    attentionHeads: 8
}
```

## 🎯 Integration Benefits

### Before: Standalone Components
- Separate ChatController, EngineController, RecorderController
- Manual coordination between components
- Duplicate code and inconsistent patterns
- Complex error handling across boundaries

### After: Integrated Architecture
- Single studio instance coordinates everything
- Event bus eliminates tight coupling
- Shared utilities and validation
- Unified error handling and retry logic
- Scalable pool management

### Performance Improvements
- **Multi-engine pools** enable parallel processing
- **Queue management** prevents system overload
- **Retry logic** handles transient failures
- **MutationObserver** provides reliable result detection

## 📊 System Flow

```
User Input → Chat Controller → Orchestrator → Engine Pool → World Engine
     ↑                                                           ↓
Recording ← Recorder Controller ← Event Bus ← Results Processing
```

1. **Input Processing**: Chat controller validates and parses user input
2. **Smart Routing**: Orchestrator queues and prioritizes requests
3. **Engine Selection**: Round-robin selection from available engine pool
4. **Model Processing**: WorldEngine processes with selected encoder
5. **Result Flow**: Results flow back through event bus to chat
6. **Recording Integration**: Optional recording captures entire session

## 🚧 Development Notes

### File Organization Completed
✅ Created `/controllers`, `/models`, `/schemas`, `/utils` folders
✅ Consolidated duplicate ChatController classes
✅ Enhanced all controllers with orchestrator integration
✅ Built unified WorldEngine with encoder swapping
✅ Implemented shared validation system

### Integration Features
✅ Central orchestrator with queue management
✅ Event bus architecture for loose coupling
✅ Multi-engine pool with load balancing
✅ Unified error handling and retry logic
✅ Runtime encoder switching (RGCN ↔ GAT ↔ Hybrid)

### Demonstration
✅ Interactive HTML demo showing full integration
✅ Real-time status monitoring
✅ Live encoder switching
✅ Recording session management

## 🎮 Demo Usage

1. **Open `demo.html`** in a modern browser
2. **Send messages** using the input field
3. **Switch encoders** using the RGCN/GAT/Hybrid buttons
4. **Start/stop recording** using the recording controls
5. **Monitor system status** in the status panel

The demo showcases the integrated architecture in action, demonstrating how all components work together seamlessly.

## 🔮 Next Steps

- **File Integration**: Process the "12 or so regular txt files" mentioned
- **Enhanced Models**: Add more encoder architectures
- **Performance Monitoring**: Add detailed metrics collection
- **Scaling**: Support for distributed engine pools
- **Persistence**: Add state management and session persistence

---

*This integrated architecture eliminates standalone components in favor of a scalable, orchestrated system where components work better together than apart.*
