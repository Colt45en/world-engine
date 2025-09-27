# 🧠 WorldEngine Tier-4 Bundle - Nucleus System Integration

A comprehensive UMD bundle that combines WorldEngine's Tier-4 Room system with advanced nucleus intelligence, AI bot communication, and librarian data processing capabilities.

## 🌟 Features

- **🧠 Nucleus Intelligence System**: Core AI decision-making with operator mapping
- **🤖 AI Bot Communication**: Intelligent message routing through nucleus protocols
- **📚 Librarian Data Processing**: Multi-librarian data classification and analysis
- **🌉 StudioBridge Integration**: Seamless event bus communication
- **🏠 Tier-4 Room System**: Interactive 3D workspace with real-time state management
- **📡 WebSocket Relay**: Enhanced communication with nucleus capabilities
- **🎮 Browser & Node.js Support**: Universal UMD format for all environments

## 🚀 Quick Start

### Browser Usage

```html
<!DOCTYPE html>
<html>
<head>
    <title>Nucleus System Demo</title>
</head>
<body>
    <!-- Load the bundle -->
    <script src="worldengine-tier4-bundle.js"></script>

    <!-- Create a room container -->
    <iframe id="tier4-room"></iframe>

    <script>
        // Initialize the nucleus-enabled bridge
        const bridge = WorldEngineTier4.createTier4RoomBridge(
            document.getElementById('tier4-room'),
            'ws://localhost:9000',
            { enableNucleus: true }
        );

        // Listen for nucleus events
        bridge.on('roomReady', () => {
            console.log('🧠 Nucleus system active');
        });

        // Trigger nucleus intelligence
        bridge.triggerNucleusEvent('VIBRATE');
        bridge.processAIBotMessage('Analyze patterns', 'query');
        bridge.processLibrarianData('Math Librarian', 'pattern', data);
    </script>
</body>
</html>
```

### React Component Integration

```jsx
import React, { useRef, useEffect } from 'react';

function NucleusEngineRoom({ websocketUrl = 'ws://localhost:9000' }) {
    const iframeRef = useRef(null);
    const [bridge, setBridge] = useState(null);

    useEffect(() => {
        if (window.WorldEngineTier4 && iframeRef.current) {
            const newBridge = window.WorldEngineTier4.createTier4RoomBridge(
                iframeRef.current,
                websocketUrl,
                { enableNucleus: true }
            );

            newBridge.on('operatorApplied', (operator, prevState, newState) => {
                console.log(`🧠 ${operator} applied:`, newState);
            });

            setBridge(newBridge);
        }
    }, [websocketUrl]);

    return (
        <div>
            <iframe ref={iframeRef} style={{ width: '100%', height: '600px' }} />
            <button onClick={() => bridge?.triggerNucleusEvent('VIBRATE')}>
                🌊 Trigger VIBRATE
            </button>
        </div>
    );
}
```

### Node.js Server Usage

```javascript
const { Tier4EnhancedRelay } = require('./worldengine-tier4-bundle.js');

// Start nucleus-enhanced relay server
const relay = new Tier4EnhancedRelay(9000);

console.log('🧠 Nucleus-enhanced WebSocket relay listening on port 9000');
```

## 🧠 Nucleus Intelligence System

The nucleus system provides intelligent routing and decision-making:

### Core Operator Mappings

```javascript
const NUCLEUS_OPERATORS = {
    'VIBRATE': 'ST',        // Vibration → Stabilization
    'OPTIMIZATION': 'UP',   // Optimization → Update/Progress
    'STATE': 'CV',          // State → Convergence
    'SEED': 'RB'           // Seed → Rollback
};
```

### AI Bot Communication

```javascript
// AI Bot message types and routing
const AI_BOT_ROUTING = {
    query: 'VIBRATE',       // Questions → Stabilization
    learning: 'OPTIMIZATION', // Learning → Progress
    feedback: 'STATE'       // Feedback → Convergence
};

// Usage
bridge.processAIBotMessage('What is the current state?', 'query');
// → Routes through VIBRATE → ST operator
```

### Librarian Data Processing

```javascript
// Librarian data types and routing
const LIBRARIAN_ROUTING = {
    pattern: 'VIBRATE',         // Pattern data → Stabilization
    classification: 'STATE',    // Classification → Convergence
    analysis: 'OPTIMIZATION'    // Analysis → Progress
};

// Usage
bridge.processLibrarianData('Math Librarian', 'pattern', patternData);
// → Routes through VIBRATE → ST operator
```

## 🎮 Interactive Controls

The system provides multiple control interfaces:

### Nucleus Event Triggers
- **🌊 VIBRATE**: Triggers stabilization processes
- **⚡ OPTIMIZATION**: Initiates progress/update cycles
- **🎯 STATE**: Manages convergence operations
- **🌱 SEED**: Handles rollback/reset functionality

### Direct Operator Controls
- **ST**: Stabilization operator
- **UP**: Update/progress operator
- **CV**: Convergence operator
- **RB**: Rollback operator

### Communication Interfaces
- **🤖 AI Bot**: Query, learning, and feedback message processing
- **📚 Librarians**: Pattern analysis, classification, and data processing

## 📡 WebSocket Communication

The enhanced relay provides nucleus-aware communication:

```javascript
// Client message format
{
    type: "nucleus_ai_bot",
    message: "Analyze system patterns",
    messageType: "query",
    timestamp: Date.now()
}

// Server response format
{
    type: "ai_bot_processed",
    nucleusRole: "VIBRATE",
    operator: "ST",
    processed: true,
    timestamp: Date.now()
}
```

## 🔧 Configuration Options

### Bridge Configuration

```javascript
const bridge = WorldEngineTier4.createTier4RoomBridge(iframe, wsUrl, {
    enableNucleus: true,        // Enable nucleus intelligence
    nucleusConfig: customConfig, // Custom nucleus configuration
    autoConnect: true,          // Auto-connect to WebSocket
    reconnectDelay: 5000       // Reconnection delay (ms)
});
```

### Nucleus Configuration

```javascript
const customNucleusConfig = {
    operators: {
        'VIBRATE': 'ST',
        'CUSTOM_OP': 'CX'  // Custom operator mapping
    },
    aiBot: {
        routing: {
            query: 'VIBRATE',
            custom: 'CUSTOM_OP'  // Custom AI routing
        }
    },
    librarians: {
        routing: {
            pattern: 'VIBRATE',
            special: 'CUSTOM_OP'  // Custom librarian routing
        }
    }
};
```

## 📊 Event System

### Event Types

```javascript
// Nucleus events
bridge.on('roomReady', () => { /* Room initialized */ });
bridge.on('operatorApplied', (op, prev, next) => { /* State changed */ });
bridge.on('nucleusEvent', (role, operator) => { /* Nucleus triggered */ });
bridge.on('connectionStatus', (connected) => { /* Connection changed */ });

// StudioBridge events
StudioBridge.onBus('tier4.nucleusEvent', (msg) => {
    // { role: 'VIBRATE', operator: 'ST', timestamp: ... }
});

StudioBridge.onBus('tier4.operatorApplied', (msg) => {
    // { operator: 'ST', previousState: {...}, newState: {...} }
});
```

### Custom Event Emission

```javascript
// Send custom nucleus events
StudioBridge.sendBus({
    type: 'tier4.customNucleusEvent',
    role: 'VIBRATE',
    data: customData,
    timestamp: Date.now()
});
```

## 🧪 Testing & Validation

The system includes comprehensive test coverage:

```bash
# Run nucleus integration tests
cd tests/nucleus
./run_tests.ps1     # Windows
./run_tests.sh      # Linux/Mac
```

### Test Categories

- ✅ **Nucleus Operator Mappings**: Core intelligence routing (4/4 tests)
- ✅ **AI Bot Communication**: Message processing and routing (3/3 tests)
- ✅ **Librarian Data Flow**: Multi-librarian data processing (3/3 tests)
- ✅ **Communication Simulation**: End-to-end workflow validation (3/3 tests)
- ✅ **Error Handling**: Edge cases and validation (3/3 tests)
- ✅ **Full Integration**: Complete system workflow (5-step simulation)
- ✅ **Bundle Integration**: WorldEngine bundle validation
- ✅ **StudioBridge Compatibility**: Event bus compatibility testing

## 📁 Project Structure

```
src/nucleus/
├── worldengine-tier4-bundle.js     # Main UMD bundle
├── nucleus_control_center.tsx      # React control center
├── nucleus_integration.ts          # Integration utilities
├── nucleus_ws_relay.js             # WebSocket relay
└── NucleusEngineRoom.tsx           # Enhanced React component

tests/nucleus/
├── integration_test.js             # Comprehensive test suite
├── nucleus_system_tests.js         # System-specific tests
├── run_tests.ps1                   # Windows test runner
└── run_tests.sh                    # Unix test runner

demo/
└── nucleus-demo.html               # Interactive browser demo
```

## 🌐 Browser Demo

Open `demo/nucleus-demo.html` in your browser for a complete interactive demonstration of the nucleus system capabilities.

### Demo Features

- 📊 **Real-time Status Display**: Connection and nucleus status indicators
- 🎮 **Interactive Controls**: Nucleus triggers, operators, AI bot, and librarian controls
- 📋 **Live Event Log**: Real-time communication and routing visualization
- 🏠 **Tier-4 Room**: Interactive 3D workspace with nucleus intelligence
- 🎬 **Auto Demo**: Automated demonstration sequence

## 🔧 Development & Customization

### Adding Custom Operators

```javascript
// Extend nucleus configuration
const customConfig = {
    ...WorldEngineTier4.NUCLEUS_CONFIG,
    operators: {
        ...WorldEngineTier4.NUCLEUS_CONFIG.operators,
        'CUSTOM_ROLE': 'CX'  // Custom role → operator mapping
    }
};

// Use in bridge initialization
const bridge = WorldEngineTier4.createTier4RoomBridge(iframe, wsUrl, {
    nucleusConfig: customConfig
});
```

### Custom Event Handlers

```javascript
// Custom nucleus event processing
bridge.on('customNucleusEvent', (role, data) => {
    console.log(`Custom nucleus event: ${role}`, data);

    // Apply custom logic
    if (role === 'CUSTOM_ROLE') {
        bridge.applyOperator('CX', { source: 'custom', data });
    }
});
```

### Extending the Relay Server

```javascript
const relay = new WorldEngineTier4.Tier4EnhancedRelay(9000);

// Add custom message handling
relay.handleCustomMessage = function(ws, message) {
    if (message.type === 'custom_nucleus') {
        // Process custom nucleus messages
        this.broadcastToClients({
            type: 'custom_nucleus_processed',
            ...message,
            processed: true
        });
    }
};
```

## 📞 API Reference

### WorldEngineTier4 Namespace

- `NUCLEUS_CONFIG`: Core nucleus configuration object
- `StudioBridge`: Enhanced event bus with nucleus API
- `Tier4Room`: Interactive room with nucleus intelligence
- `createTier4RoomBridge(iframe, wsUrl, options)`: Main bridge factory
- `Tier4EnhancedRelay`: Node.js WebSocket relay with nucleus support

### Bridge Methods

- `triggerNucleusEvent(role)`: Trigger nucleus intelligence
- `processAIBotMessage(message, type)`: Process AI bot communications
- `processLibrarianData(librarian, dataType, data)`: Handle librarian data
- `applyOperator(operator, context)`: Apply state transformations
- `applyMacro(macro)`: Execute operator sequences
- `saveSnapshot(id)`: Save current state
- `loadSnapshot(id)`: Restore saved state

### StudioBridge NucleusAPI

- `processEvent(role, data)`: Process nucleus events
- `routeAIBotMessage(message, type)`: Route AI bot messages
- `routeLibrarianData(librarian, dataType, data)`: Route librarian data

## 🚀 Production Deployment

### WebSocket Relay Server

```javascript
// Production server setup
const relay = new WorldEngineTier4.Tier4EnhancedRelay(process.env.PORT || 9000);

// Graceful shutdown
process.on('SIGTERM', () => {
    relay.saveState();
    relay.shutdown();
});
```

### Client Configuration

```javascript
// Production client configuration
const bridge = WorldEngineTier4.createTier4RoomBridge(
    iframe,
    process.env.WEBSOCKET_URL || 'wss://your-domain.com/nucleus',
    {
        enableNucleus: true,
        reconnectDelay: 3000,
        maxReconnectAttempts: 10
    }
);
```

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Run the test suite (`npm run test:nucleus`)
4. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
5. Push to the branch (`git push origin feature/AmazingFeature`)
6. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Nucleus Intelligence System**: Advanced AI decision-making architecture
- **WorldEngine Team**: Core Tier-4 room system development
- **StudioBridge**: Event bus communication framework
- **Community Contributors**: Testing, feedback, and feature development

---

**🧠 Experience the power of nucleus intelligence with WorldEngine Tier-4!**

For questions, support, or feature requests, please open an issue on the GitHub repository.
