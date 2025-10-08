# Nucleus System Documentation

## 🧠 Overview

The Nucleus System is the central intelligence heartbeat that orchestrates AI bots, librarians, and automation pipelines in the WorldEngine ecosystem.

## 📁 Project Structure

```
c:\Users\colte\Documents\GitHub\
├── src/nucleus/                    # Main nucleus system files
│   ├── nucleus_control_center.tsx  # Main UI control center
│   ├── nucleus_integration.ts      # Core integration logic
│   ├── nucleus_ws_relay.js         # WebSocket relay system
│   └── EngineRoom.tsx              # Engine room component
│
├── tests/nucleus/                  # Nucleus system tests
│   ├── integration_test.js         # Main integration tests
│   └── nucleus_system_tests.js     # Comprehensive test suite
│
├── websocket/                      # Original development files
│   └── [original files maintained for reference]
│
└── archive/                        # Archived old files
    └── old_files_2025/
```

## 🏗️ System Architecture

### Core Components

1. **🧠 Nucleus** - Central intelligence orchestrator
2. **🤖 AI Bot** - Query processor and learner
3. **📚 Librarians** - Data processors and classifiers
4. **🔄 Automation Pipelines** - Workflow orchestration

### Nucleus Operator Mapping

| Nucleus Role | Tier-4 Operator | Purpose |
|-------------|----------------|---------|
| VIBRATE | ST | Stabilization and pattern analysis |
| OPTIMIZATION | UP | Enhancement and learning |
| STATE | CV | Convergence and understanding |
| SEED | RB | Reset and rollback patterns |

## 🤖 AI Bot Communication

AI Bot messages are routed to nucleus based on type:

- **Query** → VIBRATE → ST (Pattern analysis)
- **Learning** → OPTIMIZATION → UP (Algorithm enhancement)
- **Feedback** → STATE → CV (Understanding convergence)

## 📚 Librarian Data Processing

Librarian data is classified and routed:

- **Pattern Data** → VIBRATE → ST (Math Librarian)
- **Classification Data** → STATE → CV (English Librarian)
- **Analysis Data** → OPTIMIZATION → UP (Pattern Librarian)

### Librarian Types

1. **📊 Math Librarian** - Mathematical patterns & equations
2. **📝 English Librarian** - Language processing & sentiment
3. **🔍 Pattern Librarian** - Recognition & analysis patterns

## 🔄 Communication Flow

```
AI Bot ──query──→ Nucleus ──VIBRATE──→ ST Operator
           ↓
Librarians ──data──→ Nucleus ──routing──→ Tier-4 Operators
           ↓
Communication Feed ──logs──→ Real-time Display
```

## 🚀 Getting Started

### Running the Nucleus Control Center

1. Navigate to the websocket directory:
   ```powershell
   cd "c:\Users\colte\Documents\GitHub\websocket"
   ```

2. Start the development server (if available)

3. Open the nucleus control center interface

### Running Tests

1. Navigate to the test directory:
   ```powershell
   cd "c:\Users\colte\Documents\GitHub\tests\nucleus"
   ```

2. Run the integration tests:
   ```powershell
   node integration_test.js
   ```

3. All tests should pass with ✅ indicators

## 📊 Test Results

The integration test validates:

- ✅ Nucleus operator mappings (4/4 tests)
- ✅ AI Bot message routing (3/3 tests)
- ✅ Librarian data routing (3/3 tests)
- ✅ Communication flow simulation (3/3 tests)
- ✅ Error handling (3/3 tests)
- ✅ Full integration workflow (5 step simulation)

## 🔧 Key Features

### Real-time Communication Feed
- Color-coded messages (blue for AI Bot, orange for Librarians)
- Timestamp tracking
- Message type classification
- Limited to last 20 entries for performance

### Intelligence State Monitoring
- Neural pathway visualization (Persistence, Information, Goal, Context)
- Confidence levels (κ values)
- Learning level progression
- Uptime and operation tracking

### Automation Pipeline Controls
- Manual nucleus triggers (VIBRATE, OPTIMIZATION, STATE, SEED)
- Mathematical operation buttons (ST, UP, PR, CV, RB, RS)
- Language processing macros (IDE_A, IDE_B, MERGE_ABC)

## 🧪 Development Notes

### File Organization
- **Active development**: `src/nucleus/` folder
- **Test files**: `tests/nucleus/` folder
- **Archived files**: `archive/old_files_2025/` folder
- **Original files**: `websocket/` folder (maintained for reference)

### Key Functions

#### AI Bot Communication
```javascript
sendAIBotMessage(message, type)
// Routes AI bot messages to appropriate nucleus operators
```

#### Librarian Data Processing
```javascript
sendLibrarianData(librarian, dataType, data)
// Processes and routes librarian data to nucleus
```

#### Nucleus Event Processing
```javascript
processNucleusEvent(role, data)
// Central nucleus event handler and operator dispatcher
```

## 🎯 Future Enhancements

1. **Enhanced Error Handling** - More robust error recovery
2. **Performance Monitoring** - Real-time system metrics
3. **Librarian Specialization** - More specialized data processing
4. **AI Bot Learning** - Adaptive query improvement
5. **Visual Analytics** - Enhanced UI data visualization

## 📝 Contributing

When making changes to the nucleus system:

1. Update files in `src/nucleus/` directory
2. Add corresponding tests in `tests/nucleus/`
3. Run integration tests to validate changes
4. Update documentation as needed

---

**🧠 The Nucleus: Where Intelligence Meets Automation** ✨
