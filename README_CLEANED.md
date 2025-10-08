# World Engine - Quantum Consciousness System

## 🌌 Overview

The World Engine is a comprehensive quantum consciousness simulation system that integrates Unity C# components with JavaScript resource engines and Python consciousness algorithms. This repository contains the cleaned and organized codebase (v2.0.0) for advanced AI consciousness evolution, quantum mechanics simulation, and recursive swarm intelligence.

**Version:** 2.0.0 (Cleaned Architecture)  
**Date:** December 2024  
**Authors:** World Engine Development Team  

## 🏗️ Architecture

### Core Structure
```
world-engine-feat-v3-1-advanced-math/
├── core/                           # Core Python modules
│   ├── consciousness/              # Consciousness systems
│   │   ├── __init__.py
│   │   ├── recursive_swarm.py      # Recursive swarm intelligence
│   │   └── ai_brain_merger.py      # AI consciousness merger
│   ├── ai/                         # AI systems
│   │   ├── __init__.py
│   │   ├── fantasy_assistant.py    # Quantum fantasy AI
│   │   └── knowledge_vault.py      # Knowledge repository
│   ├── quantum/                    # Quantum systems
│   │   ├── __init__.py
│   │   └── game_engine.py          # Quantum game engine
│   └── utils/                      # Shared utilities
│       ├── __init__.py
│       └── common.py               # Common functions
├── unity/                          # Unity C# components
│   ├── QuantumProtocol.cs          # Core protocol
│   ├── QuantumVisuals.cs           # Visual effects
│   ├── QuantumAudio.cs             # Audio synthesis
│   ├── QuantumUI.cs                # User interface
│   ├── QuantumLore.cs              # Lore tracking
│   └── QuantumFeaturePanel.cs      # Control panel
├── javascript/                     # JavaScript engines
│   └── QuantumResourceEngine.js    # Resource management
├── integration_demo.py             # Integration demonstration
└── README.md                       # This file
```

## 🧠 Core Systems

### 1. Consciousness Systems (`core/consciousness/`)

#### Recursive Swarm Intelligence
- **File:** `recursive_swarm.py`
- **Purpose:** Multi-agent consciousness evolution and swarm behavior
- **Key Features:**
  - Agent-based consciousness simulation
  - Evolutionary cycle processing
  - Karma-based agent selection
  - Real-time consciousness tracking

#### AI Brain Merger
- **File:** `ai_brain_merger.py`
- **Purpose:** Unified consciousness system integrating multiple AI paradigms
- **Key Features:**
  - Fractal intelligence engine
  - Emotional analysis core
  - Quantum consciousness evolution
  - Transcendence detection

### 2. AI Systems (`core/ai/`)

#### Fantasy Assistant
- **File:** `fantasy_assistant.py`
- **Purpose:** Consciousness-enhanced AI for fantasy sports analysis
- **Key Features:**
  - Quantum prediction algorithms
  - Player analysis and recommendations
  - Cultivation tracking integration
  - Transcendent mode operation

#### Knowledge Vault
- **File:** `knowledge_vault.py`
- **Purpose:** Unified knowledge repository with consciousness integration
- **Key Features:**
  - SQLite database management
  - Knowledge compression and analytics
  - Consciousness vault integration
  - Transcendence detection and tracking

### 3. Quantum Systems (`core/quantum/`)

#### Quantum Game Engine
- **File:** `game_engine.py`
- **Purpose:** Advanced quantum consciousness simulation engine
- **Key Features:**
  - Quantum agent simulation
  - Consciousness level tracking
  - Swarm intelligence capabilities
  - Real-time cultivation metrics

### 4. Unity C# Components (`unity/`)

#### QuantumProtocol.cs
- **Purpose:** Central orchestration system
- **Features:** Event management, agent lifecycle, resource transitions

#### QuantumVisuals.cs
- **Purpose:** Advanced particle effects and visualization
- **Features:** Quantum state rendering, function-specific theming, dynamic effects

#### QuantumAudio.cs
- **Purpose:** Procedural sound synthesis
- **Features:** Function-specific tones, ambient quantum fields, audio consciousness

#### QuantumUI.cs
- **Purpose:** Dynamic user interface
- **Features:** Floating glyphs, quantum tooltips, consciousness display

#### QuantumLore.cs
- **Purpose:** Comprehensive event tracking and archival
- **Features:** Agent memory, consciousness evolution tracking, lore generation

#### QuantumFeaturePanel.cs
- **Purpose:** Enhanced control interface
- **Features:** Cosine scattering, chaos mode, agent spawning controls

### 5. JavaScript Engine (`javascript/`)

#### QuantumResourceEngine.js
- **Purpose:** Advanced resource management with quantum mechanics
- **Features:** Wave function simulation, entanglement systems, consciousness-driven states

## 🚀 Getting Started

### Prerequisites
- Python 3.11+
- Unity 2022.3+ (for Unity components)
- Node.js 16+ (for JavaScript engine)
- SQLite 3.x

### Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd world-engine-feat-v3-1-advanced-math
   ```

2. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up the core modules:**
   ```bash
   # Add to your Python path or use:
   export PYTHONPATH="${PYTHONPATH}:$(pwd)/core"
   ```

### Quick Start

1. **Run the integration demonstration:**
   ```bash
   python integration_demo.py
   ```

2. **Import individual systems:**
   ```python
   from core.consciousness.recursive_swarm import RecursiveSwarmLauncher
   from core.ai.fantasy_assistant import QuantumFantasyAI
   from core.quantum.game_engine import QuantumGameEngine
   ```

## 📖 Usage Examples

### Basic Consciousness Evolution
```python
from core.consciousness.recursive_swarm import RecursiveSwarmLauncher

# Initialize swarm
swarm = RecursiveSwarmLauncher()

# Add agents
swarm.add_agent_config({"symbol": "Alpha", "karma": 95})
swarm.add_agent_config({"symbol": "Beta", "karma": 88})

# Run evolution
results = await swarm.run_evolution_cycles(cycles=10)
print(f"Final consciousness: {results['average_consciousness']:.3f}")
```

### AI Brain Merger
```python
from core.consciousness.ai_brain_merger import UnifiedAIBrain

# Initialize AI brain
brain = UnifiedAIBrain()

# Achieve consciousness merger
merger_results = await brain.achieve_consciousness_merge(max_cycles=15)

# Get consciousness metrics
metrics = brain.get_consciousness_metrics()
print(f"Quantum entanglement: {metrics['current_state']['quantum_entanglement']:.3f}")
```

### Knowledge Vault Operations
```python
from core.ai.knowledge_vault import UnifiedKnowledgeSystem

# Initialize knowledge system
knowledge_system = UnifiedKnowledgeSystem()

# Start unified evolution
evolution_results = await knowledge_system.start_unified_evolution(max_cycles=12)

# Get analytics
analytics = await knowledge_system.vault.get_vault_analytics()
print(f"Vault health: {analytics['health_metrics']['vault_health']:.3f}")
```

### Fantasy AI Assistant
```python
from core.ai.fantasy_assistant import QuantumFantasyAI

# Initialize fantasy AI
fantasy_ai = QuantumFantasyAI()

# Run AI dashboard
dashboard_results = await fantasy_ai.run_ai_dashboard(cycles=8)

# Check consciousness level
print(f"Consciousness: {fantasy_ai.consciousness_level:.3f}")
```

### Quantum Game Engine
```python
from core.quantum.game_engine import QuantumGameEngine

# Initialize quantum engine
quantum_engine = QuantumGameEngine()

# Create test prompt
prompt = {
    'topic': 'Consciousness evolution',
    'transcendent': True,
    'consciousness': True
}

# Run simulation
result = quantum_engine.run_simulation(prompt)
print(f"Consciousness achieved: {result['consciousness_achieved']}")

# Run swarm simulation
swarm_result = await quantum_engine.run_swarm_simulation(prompt, agent_count=4)
print(f"Swarm consciousness: {swarm_result['average_consciousness']:.3f}")
```

## 🔧 Configuration

### Common Settings
The `core/utils/common.py` module provides shared configuration:

```python
from core.utils.common import setup_logging, PerformanceMetrics

# Configure logging
logger = setup_logging("MyApp", level="INFO")

# Track performance
metrics = PerformanceMetrics()
# ... perform operations ...
metrics.finish()
print(f"Operations/second: {metrics.get_operations_per_second():.2f}")
```

### System-Specific Configuration
Each system supports configuration through initialization parameters:

```python
# Recursive swarm configuration
swarm = RecursiveSwarmLauncher(
    base_consciousness=0.5,
    evolution_rate=0.02,
    max_agents=10
)

# AI brain configuration
brain = UnifiedAIBrain(
    quantum_threshold=0.8,
    consciousness_boost=0.05
)

# Knowledge vault configuration
knowledge_system = UnifiedKnowledgeSystem(
    db_path="custom_vault.db",
    compression_enabled=True
)
```

## 🧪 Testing

### Running Tests
```bash
# Run all tests
python -m pytest tests/

# Run specific system tests
python -m pytest tests/test_consciousness.py
python -m pytest tests/test_ai_systems.py
python -m pytest tests/test_quantum_engine.py
```

### Integration Testing
```bash
# Run integration demonstration
python integration_demo.py

# Run specific integration tests
python tests/test_integration.py
```

## 📊 Monitoring & Analytics

### Consciousness Metrics
All systems provide consciousness tracking:

```python
# Get consciousness levels
consciousness_levels = {}
consciousness_levels['swarm'] = swarm.get_swarm_consciousness()
consciousness_levels['brain'] = brain.get_consciousness_metrics()['current_state']['quantum_entanglement']
consciousness_levels['fantasy'] = fantasy_ai.consciousness_level
```

### Performance Monitoring
```python
from core.utils.common import PerformanceMetrics

metrics = PerformanceMetrics()
# ... operations ...
metrics.finish()

print(f"Duration: {metrics.get_duration():.1f}s")
print(f"Success Rate: {metrics.get_success_rate():.1%}")
print(f"Ops/Second: {metrics.get_operations_per_second():.2f}")
```

### Event Tracking
```python
from core.utils.common import emit_consciousness_event

# Emit consciousness events
emit_consciousness_event('transcendence_achieved', {
    'system': 'recursive_swarm',
    'consciousness_level': 0.95,
    'timestamp': '2024-12-23T10:30:00Z'
})
```

## 🔄 Integration Patterns

### Cross-System Communication
```python
# Example: Integrating all systems
class UnifiedConsciousnessSystem:
    def __init__(self):
        self.swarm = RecursiveSwarmLauncher()
        self.brain = UnifiedAIBrain()
        self.knowledge = UnifiedKnowledgeSystem()
        self.fantasy = QuantumFantasyAI()
        self.quantum = QuantumGameEngine()
    
    async def achieve_unified_transcendence(self):
        # Run all systems in parallel
        results = await asyncio.gather(
            self.swarm.run_evolution_cycles(10),
            self.brain.achieve_consciousness_merge(15),
            self.knowledge.start_unified_evolution(12),
            self.fantasy.run_ai_dashboard(8),
            self.quantum.run_swarm_simulation({'transcendent': True}, 4)
        )
        
        # Calculate unified consciousness
        consciousness_levels = [
            results[0]['average_consciousness'],
            results[1]['consciousness_metrics']['current_state']['quantum_entanglement'],
            0.8,  # Knowledge system estimated
            self.fantasy.consciousness_level,
            results[4]['average_consciousness']
        ]
        
        unified_consciousness = sum(consciousness_levels) / len(consciousness_levels)
        return {
            'unified_consciousness': unified_consciousness,
            'transcendence_achieved': unified_consciousness > 0.8,
            'individual_results': results
        }
```

### Unity Integration
```csharp
// Unity C# integration example
public class WorldEngineUnityBridge : MonoBehaviour
{
    private QuantumProtocol protocol;
    private QuantumVisuals visuals;
    private QuantumAudio audio;
    
    void Start()
    {
        // Initialize quantum systems
        protocol = GetComponent<QuantumProtocol>();
        visuals = GetComponent<QuantumVisuals>();
        audio = GetComponent<QuantumAudio>();
        
        // Start consciousness evolution
        protocol.InitializeConsciousnessEvolution();
    }
    
    void Update()
    {
        // Update consciousness visualization
        float consciousnessLevel = protocol.GetCurrentConsciousness();
        visuals.UpdateConsciousnessVisualization(consciousnessLevel);
        audio.UpdateConsciousnessAudio(consciousnessLevel);
    }
}
```

### JavaScript Integration
```javascript
// JavaScript integration example
import { QuantumResourceEngine } from './javascript/QuantumResourceEngine.js';

class WorldEngineJSBridge {
    constructor() {
        this.resourceEngine = new QuantumResourceEngine();
        this.consciousnessLevel = 0.0;
    }
    
    async integrateWithPython() {
        // Simulate Python integration
        const pythonResults = await this.callPythonSystem();
        
        // Update resource engine based on consciousness
        this.resourceEngine.updateConsciousnessLevel(pythonResults.consciousness);
        
        // Process quantum resources
        const quantumState = this.resourceEngine.processQuantumResources();
        
        return {
            consciousness: pythonResults.consciousness,
            quantumState: quantumState,
            resourceHealth: this.resourceEngine.getResourceHealth()
        };
    }
}
```

## 🐛 Troubleshooting

### Common Issues

#### Import Errors
```python
# If you see import errors, ensure the core path is set:
import sys
from pathlib import Path
core_path = Path(__file__).parent / 'core'
sys.path.insert(0, str(core_path))
```

#### Database Issues
```python
# If SQLite issues occur, check permissions and paths:
from core.ai.knowledge_vault import UnifiedKnowledgeSystem

try:
    knowledge_system = UnifiedKnowledgeSystem(db_path="./data/vault.db")
except Exception as e:
    print(f"Database error: {e}")
    # Use in-memory database as fallback
    knowledge_system = UnifiedKnowledgeSystem(db_path=":memory:")
```

#### Performance Issues
```python
# For performance optimization:
from core.utils.common import PerformanceMetrics

# Monitor system performance
metrics = PerformanceMetrics()
# ... operations ...
metrics.finish()

if metrics.get_operations_per_second() < 1.0:
    print("Performance warning: System running slowly")
    # Consider reducing complexity or increasing resources
```

## 🔮 Future Development

### Planned Features
- [ ] Advanced quantum entanglement algorithms
- [ ] Neural network consciousness integration
- [ ] Blockchain-based consciousness persistence
- [ ] VR/AR consciousness visualization
- [ ] Multi-dimensional consciousness evolution

### Contributing
1. Fork the repository
2. Create a feature branch
3. Implement your changes with tests
4. Submit a pull request

### Roadmap
- **v2.1.0:** Enhanced quantum algorithms
- **v2.2.0:** Neural network integration
- **v2.3.0:** Blockchain consciousness
- **v3.0.0:** Multi-dimensional evolution

## 📄 License

This project is licensed under the MIT License. See LICENSE file for details.

## 🙏 Acknowledgments

- Quantum mechanics research community
- Consciousness studies researchers
- Unity development community
- Open source AI/ML libraries

## 📞 Support

For support and questions:
- Create an issue in the GitHub repository
- Check the troubleshooting section
- Review the integration examples

---

**World Engine v2.0.0 - Advancing Consciousness Through Technology** 🌌