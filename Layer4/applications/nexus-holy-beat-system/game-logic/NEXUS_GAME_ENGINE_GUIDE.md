# NEXUS Holy Beat System - Game Logic Engine

## Overview

This C++ game logic engine integrates seamlessly with the NEXUS Holy Beat System, providing professional-grade game development capabilities with real-time audio and art synchronization.

## Features

### Core Game Engine
- **Entity-Component Architecture**: Flexible ECS system with Transform, AudioSync, ArtSync, and Physics components
- **Resource Management**: Advanced async loading with LRU caching, memory management, and LOD system
- **Audio-Visual Sync**: Real-time synchronization with NEXUS Holy Beat System audio analysis and art generation
- **Cross-Platform**: Windows (MSVC) and Linux/macOS (GCC/Clang) support

### Professional Game Systems
- **Asset Management**: LRU cache with configurable memory limits and automatic resource cleanup
- **World Simulation**: Player/enemy/projectile physics with collision detection and spatial partitioning
- **Math Library**: Comprehensive Vec3 implementation with optimized vector operations
- **Camera System**: Smooth camera controls with configurable parameters

### NEXUS Integration
- **Beat Synchronization**: BPM-synchronized gameplay elements and enemy behaviors
- **Audio Reactivity**: Enemies and environment react to real-time audio analysis
- **Art Reactivity**: Dynamic environments that respond to generated art parameters
- **Real-time API**: JSON data streaming from NEXUS server (localhost:8080)

## Quick Start

### Prerequisites
- C++17 compatible compiler (MSVC 2019+, GCC 7+, Clang 5+)
- CMake 3.15+ (optional, for advanced builds)
- Node.js 14+ (optional, for JavaScript bindings)

### Building

#### Windows (Recommended)
```batch
# Clone and navigate
git clone <repository>
cd nexus-holy-beat-system/game-logic

# Build everything
build.bat
```

#### Linux/macOS
```bash
# Make build script executable
chmod +x build.sh

# Build everything
./build.sh
```

#### Manual Compilation
```bash
# Basic demo
g++ -std=c++17 -O2 -Wall -Wextra -I./include ./examples/basic_game_demo.cpp -o basic_demo -pthread

# NEXUS resource demo
g++ -std=c++17 -O2 -Wall -Wextra -I./include ./examples/nexus_resource_demo.cpp -o nexus_resource_demo -pthread

# NEXUS world demo (comprehensive)
g++ -std=c++17 -O2 -Wall -Wextra -I./include ./examples/nexus_world_demo.cpp -o nexus_world_demo -pthread
```

## Demo Programs

### 1. Basic Game Demo (`basic_game_demo`)
Simple demonstration of the entity-component system and basic game mechanics.

**Run:**
```bash
./basic_demo        # Linux/macOS
basic_demo.exe      # Windows
```

### 2. NEXUS Resource Demo (`nexus_resource_demo`)
Demonstrates advanced resource management with async loading, LRU caching, and NEXUS integration.

**Run:**
```bash
./nexus_resource_demo        # Linux/macOS
nexus_resource_demo.exe      # Windows
```

### 3. NEXUS World Demo (`nexus_world_demo`) - **Flagship Demo**
Comprehensive game world simulation with full NEXUS Holy Beat System integration.

**Features:**
- Player movement with WASD + mouse controls
- Enemy AI with audio-reactive behaviors
- Art-reactive environmental effects
- Real-time collision detection
- Dynamic resource loading
- BPM-synchronized gameplay elements

**Controls:**
- **W/A/S/D**: Player movement
- **Mouse**: Look around
- **Space**: Jump/Action
- **ESC**: Exit

**Run:**
```bash
./nexus_world_demo        # Linux/macOS
nexus_world_demo.exe      # Windows
```

## API Integration

### NEXUS Server Connection
The game logic engine automatically connects to the NEXUS Holy Beat System server:
- **Default URL**: `http://localhost:8080`
- **API Endpoints**: `/api/audio`, `/api/art`, `/api/settings`
- **Data Format**: JSON with real-time audio analysis and art generation parameters

### Audio-Reactive Parameters
```cpp
struct AudioData {
    float volume;          // Current volume level (0.0-1.0)
    float bass;           // Bass frequency intensity
    float midrange;       // Midrange frequency intensity
    float treble;         // Treble frequency intensity
    float bpm;            // Detected beats per minute
    bool beat_detected;   // Real-time beat detection
};
```

### Art-Reactive Parameters
```cpp
struct ArtData {
    float complexity;     // Art complexity factor (0.0-1.0)
    float brightness;     // Overall brightness level
    float contrast;       // Contrast intensity
    float saturation;     // Color saturation level
    std::string style;    // Current art style ("abstract", "geometric", etc.)
    Vec3 dominant_color;  // Primary color (RGB 0.0-1.0)
};
```

## Architecture

### Directory Structure
```
nexus-holy-beat-system/game-logic/
├── include/                    # Header files
│   ├── NexusGameEngine.hpp    # Main engine class
│   ├── GameEntity.hpp         # Entity-component system
│   ├── GameComponents.hpp     # Audio/Art reactive components
│   └── NexusResourceEngine.hpp # Resource management
├── src/                       # Source implementations (header-only for now)
├── examples/                  # Demo programs
│   ├── basic_game_demo.cpp    # Basic ECS demo
│   ├── nexus_resource_demo.cpp # Resource management demo
│   └── nexus_world_demo.cpp   # Comprehensive world simulation
├── bindings/                  # Node.js addon for JavaScript integration
├── build/                     # Build outputs
├── CMakeLists.txt            # CMake configuration
├── package.json              # Node.js addon configuration
├── build.sh                  # Linux/macOS build script
└── build.bat                 # Windows build script
```

### Core Components

#### GameEntity System
```cpp
class GameEntity {
public:
    template<typename T>
    void AddComponent(T component);

    template<typename T>
    T* GetComponent();

    void Update(float deltaTime);
    void SyncWithNexus(const AudioData& audio, const ArtData& art);
};
```

#### Resource Management
```cpp
class NexusResourceEngine {
public:
    void LoadResourceAsync(const std::string& path);
    void SetMemoryLimit(size_t maxBytes);
    void EnableLODSystem(bool enabled);
    void Update(float deltaTime);
};
```

## Performance Characteristics

### Memory Management
- **LRU Cache**: Configurable memory limits with automatic cleanup
- **Async Loading**: Non-blocking resource loading with job queues
- **LOD System**: Level-of-detail for memory optimization

### Rendering Optimization
- **Frustum Culling**: Visibility-based rendering optimization
- **Spatial Partitioning**: Efficient collision detection and object management
- **Dynamic LOD**: Distance-based level-of-detail switching

### Real-time Performance
- **60 FPS Target**: Optimized for smooth real-time gameplay
- **Low Latency**: Minimal delay between NEXUS audio/art analysis and game response
- **Adaptive Quality**: Dynamic quality adjustment based on performance

## Advanced Usage

### Custom Components
```cpp
// Create custom audio-reactive component
class AudioReactiveEnemy : public GameComponent {
public:
    void Update(float deltaTime) override;
    void OnAudioSync(const AudioData& audio) override {
        // React to bass frequencies
        if (audio.bass > 0.7f) {
            speed *= 1.5f; // Speed boost on heavy bass
        }
    }
};
```

### Resource Loading
```cpp
// Load assets with memory management
resourceEngine.SetMemoryLimit(512 * 1024 * 1024); // 512MB limit
resourceEngine.LoadResourceAsync("models/enemy.obj");
resourceEngine.LoadResourceAsync("textures/environment.png");
resourceEngine.EnableLODSystem(true);
```

### NEXUS Integration
```cpp
// Sync game state with NEXUS Holy Beat System
nexusEngine.ConnectToServer("http://localhost:8080");
nexusEngine.EnableAudioSync(true);
nexusEngine.EnableArtSync(true);
nexusEngine.SetBPMSyncEnabled(true);
```

## Node.js Integration

### Building JavaScript Bindings
```bash
# Install dependencies
npm install

# Build native addon
npm run build

# Test bindings
npm test
```

### JavaScript Usage
```javascript
const nexusGame = require('./build/Release/nexus_game_addon');

// Create game instance
const game = new nexusGame.GameEngine();
game.connectToNexus('http://localhost:8080');

// Game loop
setInterval(() => {
    game.update(16.67); // 60 FPS
}, 16);
```

## Troubleshooting

### Common Issues

#### Build Errors
- Ensure C++17 compiler is available
- Check CMake version (3.15+ required)
- Verify include paths are correct

#### NEXUS Connection Issues
- Confirm NEXUS Holy Beat System server is running on localhost:8080
- Check firewall settings
- Verify network connectivity

#### Performance Issues
- Reduce memory limits if experiencing crashes
- Disable LOD system on low-end hardware
- Lower audio/art sync frequency

### Debug Mode
```bash
# Build with debug symbols
g++ -std=c++17 -g -Wall -Wextra -DDEBUG -I./include ./examples/nexus_world_demo.cpp -o nexus_world_demo_debug -pthread
```

## Contributing

1. Follow C++17 standards
2. Maintain NEXUS Holy Beat System integration
3. Add comprehensive documentation
4. Include performance benchmarks
5. Test on multiple platforms

## License

This game logic engine is part of the NEXUS Holy Beat System project. See LICENSE file for details.

---

**Ready to create audio-reactive games with professional-grade systems and NEXUS Holy Beat integration!** 🎮🎵✨
