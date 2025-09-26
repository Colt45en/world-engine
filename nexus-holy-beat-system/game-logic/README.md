# NEXUS Game Logic - C++ Development Environment

A high-performance C++ game logic system integrated with the NEXUS Holy Beat System. Now enhanced with **Quantum Protocol** patterns inspired by Unreal Engine for advanced audio-reactive and art-reactive game development.

## ✨ **New: Quantum Protocol Integration**

### 🌟 **NEXUS Protocol System**
- **ProcessingMode Enum**: 8 different modes (Mirror, Cosine, Chaos, Absorb, Amplify, Pulse, Flow, Fragment)
- **Auto-Mode Selection**: Intelligent mode switching based on real-time audio analysis
- **Event-Driven Architecture**: C++ delegate-style callbacks for real-time updates
- **Statistics Tracking**: Comprehensive monitoring of mode changes and data updates

### 🎨 **Advanced Visual Palette System**
- **Dynamic Color Palettes**: Each processing mode has unique color schemes
- **Audio-Reactive Colors**: Bass/treble/midrange affect color intensity and hue shifts
- **Art-Reactive Modifications**: Complexity, brightness, contrast, and style influence visuals
- **Palette Variants**: Neon, Pastel, Monochrome, and Complementary palette generation
- **Smooth Transitions**: Interpolated color animations between palette changes

### 🌈 **Professional Trail Rendering**
- **Animated Trail System**: Smooth color transitions with customizable fade times
- **Audio-Reactive Trails**: Width, glow, and effects respond to audio analysis
- **Beat Synchronization**: Pulse effects triggered by beat detection
- **Multiple Effect Modes**: Chaos (random colors), Fragment (glitch effects), Flow (smooth)
- **Memory Management**: Configurable trail length, lifetime, and cleanup

### 🧠 **Recursive Keeper Engine** (New!)
- **Cognitive Processing**: Philosophical topic analysis with state evolution (solid, liquid, gas)
- **Memory Tracking**: Dual-state memory with forward/reverse cognitive patterns
- **Eternal Imprints**: Persistent cognitive concepts with influence tracking
- **Symbolic Mapping**: Abstract concept to symbol associations
- **NEXUS Integration**: Cognitive analysis drives processing mode changes
- **Real-time Evolution**: Topics evolve recursively based on audio/art input## 🎮 Core Features

- **Audio Synchronization**: Real-time sync with BPM, harmonics, and audio analysis
- **Art Engine Integration**: Petal patterns, mandala formations, and fractal animations
- **Physics Simulation**: Basic physics with collision detection
- **Component System**: Flexible entity-component architecture
- **Node.js Bindings**: Seamless integration with JavaScript/Web systems
- **Cross-Platform**: Windows, macOS, Linux support

## 🏗️ Project Structure

```
game-logic/
├── include/           # Header files
│   ├── NexusGameEngine.hpp
│   ├── GameEntity.hpp
│   └── GameComponents.hpp
├── src/              # Source implementations
├── examples/         # Example programs
├── bindings/         # Node.js addon code
├── build/           # Build output
└── CMakeLists.txt   # Build configuration
```

## 🚀 Quick Start

### Prerequisites

- **C++ Compiler**: GCC 7+ / Clang 6+ / MSVC 2019+
- **CMake**: 3.16 or later
- **Node.js**: 16+ (for JavaScript bindings)

### Build Instructions

#### Option 1: Automated Build (Recommended)

**Linux/macOS/WSL:**
```bash
cd nexus-holy-beat-system/game-logic
chmod +x build.sh
./build.sh
```

**Windows:**
```cmd
cd nexus-holy-beat-system\game-logic
build.bat
```

#### Option 2: Manual CMake Build

```bash
# Navigate to game logic directory
cd nexus-holy-beat-system/game-logic

# Create build directory
mkdir build && cd build

# Configure with CMake
cmake .. -DBUILD_EXAMPLES=ON

# Build the project
cmake --build . --config Release

# Run examples
./examples/basic_game_demo
./nexus_resource_demo
```

#### Option 3: Direct Compilation (Fastest for Testing)

```bash
# Compile resource engine demo directly
g++ -std=c++17 -O2 -I./include examples/nexus_resource_demo.cpp -o nexus_demo -pthread
./nexus_demo
```

## 🌟 **Quantum Protocol Demos**

### New Demo: NEXUS Quantum Protocol Showcase

```bash
# Compile and run the quantum demo
g++ -std=c++17 -O2 -I./include examples/nexus_quantum_demo.cpp -o nexus_quantum_demo -pthread
./nexus_quantum_demo

# Or use the automated build
./build.sh  # Builds all demos including quantum
./nexus_quantum_demo.exe  # Windows
```

**Features demonstrated:**
- 🌀 **Processing Mode Switching**: See all 8 modes (Mirror, Cosine, Chaos, etc.)
- 🎵 **Audio Reactivity**: Auto-mode selection based on simulated audio
- 🎨 **Dynamic Palettes**: Real-time color changes responding to audio/art data
- 🌈 **Trail Rendering**: Animated trails with beat synchronization and effects
- 📊 **Statistics Tracking**: Monitor mode changes, audio updates, and beat detection

### New Demo: NEXUS Cognitive Evolution Showcase

```bash
# Compile and run the cognitive demo
g++ -std=c++17 -O2 -I./include examples/nexus_cognitive_demo.cpp -o nexus_cognitive_demo -pthread
./nexus_cognitive_demo

# Or use the automated build
./build.sh  # Builds all demos including cognitive
./nexus_cognitive_demo.exe  # Windows
```

**Features demonstrated:**
- 🧠 **Recursive Keeper Engine**: Philosophical topic analysis and evolution
- 🔄 **Cognitive Memory**: Forward/reverse memory tracking with pattern analysis
- 🌀 **Topic Evolution**: Watch concepts evolve through solid/liquid/gas states
- 🎵 **Cognitive Audio Integration**: Topics influence and respond to audio characteristics
- 🎨 **Art-Reactive Processing**: Visual complexity drives cognitive analysis
- 🌈 **Cognitive Trails**: Visual representation of thought evolution
- 💭 **Interactive Exploration**: Enter topics and watch cognitive analysis unfold
- 📊 **Memory Pattern Analysis**: Statistical analysis of cognitive evolution patterns### Windows (Visual Studio)

```cmd
cd nexus-holy-beat-system\game-logic
mkdir build && cd build
cmake .. -G "Visual Studio 16 2019"
cmake --build . --config Release
```

## 📚 Core Components

### NexusGameEngine

The main engine class that coordinates all systems:

```cpp
#include "NexusGameEngine.hpp"

NexusGame::NexusGameEngine engine;
engine.Initialize();

// Sync with NEXUS system parameters
engine.SetBPM(120.0);
engine.SetHarmonics(6);
engine.SetPetalCount(8);

// Main game loop
while (engine.IsRunning()) {
    engine.Update();
    engine.Render();
}
```

### Game Entities & Components

```cpp
// Create game entity
auto entity = engine.CreateEntity("AudioParticle");

// Add components
auto transform = entity->AddComponent<Transform>();
auto audioSync = entity->AddComponent<AudioSync>();
auto artSync = entity->AddComponent<ArtSync>();
auto physics = entity->AddComponent<Physics>();

// Configure components
audioSync->SetSyncMode(AudioSync::SyncMode::BPM_PULSE);
artSync->SetPatternMode(ArtSync::PatternMode::PETAL_FORMATION);
physics->SetMass(1.0);
```

## 🔗 NEXUS System Integration

### Audio Engine Sync

```cpp
// Responds to real-time audio data
audioSync->UpdateAudioData(bpm, harmonics, amplitude, frequency);

// Get beat-synchronized values
double beatPhase = audioSync->GetBeatPhase();
bool onBeat = audioSync->IsOnBeat();
```

### Art Engine Sync

```cpp
// Synchronize with art parameters
artSync->UpdateArtData(petalCount, terrainRoughness);

// Get artistic transformations
auto color = artSync->GetCurrentColor();
auto petalPos = artSync->GetPetalPosition(0);
```

### Physics Integration

```cpp
// Apply forces based on audio/art data
if (audioSync->IsOnBeat()) {
    physics->AddImpulse({0, beatIntensity, 0});
}

// Terrain collision with world engine
physics->SetUseGravity(true);
```

## 🌐 JavaScript Integration

### Node.js Addon

The C++ engine can be used from JavaScript:

```javascript
const nexusGame = require('./nexus_game.node');

// Create engine instance
const engine = new nexusGame.NexusGameEngine();

// Initialize with NEXUS parameters
engine.initialize({
    bpm: 120,
    harmonics: 6,
    petalCount: 8,
    terrainRoughness: 0.4
});

// Create and configure entities
const entity = engine.createEntity('TestEntity');
entity.addTransform();
entity.addAudioSync();
entity.addArtSync();

// Game loop
setInterval(() => {
    engine.update();
}, 16); // 60 FPS
```

### Web Integration

Connect with NEXUS dashboards:

```javascript
// Sync with NEXUS API data
fetch('/api/status')
    .then(response => response.json())
    .then(data => {
        engine.syncWithNexusSystems(JSON.stringify(data));
    });
```

## 🎯 Example Use Cases

### Audio-Reactive Particles

```cpp
class AudioParticleSystem : public ComponentSystem {
public:
    void Update(double deltaTime) override {
        for (auto* component : GetComponents()) {
            auto* audioSync = static_cast<AudioSync*>(component);
            auto* transform = audioSync->GetEntity()->GetComponent<Transform>();

            if (audioSync->IsOnBeat()) {
                // Pulse on beat
                double scale = 1.0 + audioSync->GetIntensity() * 0.5;
                transform->SetScale(scale);
            }
        }
    }
};
```

### Mandala Formation

```cpp
auto center = engine.CreateEntity("MandalaCenter");
center->AddComponent<Transform>();
center->AddComponent<ArtSync>()->SetPatternMode(ArtSync::PatternMode::MANDALA_SYNC);

// Create petal entities
for (int i = 0; i < petalCount; ++i) {
    auto petal = engine.CreateEntity("Petal" + std::to_string(i));
    auto artSync = petal->AddComponent<ArtSync>();
    auto pos = artSync->GetPetalPosition(i);
    petal->GetComponent<Transform>()->SetPosition(pos);
}
```

## 🔧 Configuration Options

### CMake Build Options

- `BUILD_NODE_ADDON=ON/OFF` - Build JavaScript bindings
- `BUILD_EXAMPLES=ON/OFF` - Build example programs
- `CMAKE_BUILD_TYPE=Debug/Release` - Build configuration

### Runtime Parameters

- **BPM**: 60-200 (beats per minute)
- **Harmonics**: 1-16 (harmonic complexity)
- **Petal Count**: 3-32 (geometric patterns)
- **Terrain Roughness**: 0.0-1.0 (world complexity)

## 🚦 System Requirements

### Minimum

- **CPU**: Dual-core 2.0GHz
- **RAM**: 2GB
- **Compiler**: C++17 support

### Recommended

- **CPU**: Quad-core 3.0GHz+
- **RAM**: 8GB+
- **GPU**: OpenGL 3.3+ support

## 📈 Performance Notes

- **Update Rate**: Targets 60 FPS (16.67ms per frame)
- **Audio Latency**: <10ms for real-time sync
- **Entity Count**: Optimized for 1000+ entities
- **Memory Usage**: ~50MB baseline + entity data

## 🤝 Integration with NEXUS Dashboards

The C++ game logic seamlessly integrates with:

- **World Engine Tier 4**: 3D consciousness-driven environments
- **Glyph Forge**: Vector art and animation systems
- **Lexical Logic Engine**: Mathematical processing
- **Audio Synthesis**: Real-time sound generation
- **Training Systems**: ML/AI data processing

---

*Part of the NEXUS Holy Beat System - Where Sound, Art, and World Converge* 🎵✨
