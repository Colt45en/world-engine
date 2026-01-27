# World Engine

**Full-Scale Multi-Language Game Engine**

World Engine is a comprehensive game engine built with CMake that supports multiple programming languages including C++, Python, TypeScript, and provides web interfaces with HTML/CSS/JavaScript.

## Features

- **Multi-Language Support**: Write game logic in C++, Python, or TypeScript
- **Modern C++17 Core**: High-performance entity-component-system architecture
- **Python Bindings**: Seamless Python integration for rapid prototyping
- **TypeScript/JavaScript**: Type-safe web and Node.js support
- **CMake Build System**: Cross-platform build support for all major platforms
- **Web Interface**: HTML5-based demo and documentation

## Supported Languages

- **C++17**: Core engine implementation
- **Python 3.7+**: High-level scripting and bindings
- **TypeScript 5.0+**: Web and Node.js support
- **HTML/CSS/JavaScript**: Web interfaces and demos

## Quick Start

### Prerequisites

- CMake 3.15+
- C++17 compatible compiler
- Python 3.7+ (optional, for Python bindings)
- Node.js 14+ and npm (optional, for TypeScript)

### Build

```bash
git clone https://github.com/Colt45en/world-Engine.git
cd world-Engine
mkdir build
cd build
cmake ..
cmake --build .
```

### Run Examples

```bash
# C++ Example
./bin/examples/cpp_example

# Python Example
python3 ../examples/python_example.py

# View HTML Demo
open ../src/html/index.html
```

## Project Structure

```
world-Engine/
├── CMakeLists.txt          # Main CMake configuration
├── BUILD.md                # Build documentation
├── include/                # C++ headers
│   └── world_engine/
├── src/
│   ├── cpp/               # C++ implementation
│   ├── python/            # Python bindings
│   ├── typescript/        # TypeScript implementation
│   └── html/              # Web interface
├── examples/              # Example programs
├── tests/                 # Test suite
└── docs/                  # Documentation
```

## Architecture

World Engine uses an Entity-Component-System (ECS) architecture:

- **Entity**: Game objects in the world
- **Component**: Data attached to entities
- **System**: Logic that operates on entities with specific components
- **Scene**: Container for entities and systems

## Language Examples

### C++

```cpp
#include <world_engine/world_engine.h>

int main() {
    WorldEngine::Engine engine;
    engine.initialize();
    
    auto scene = std::make_shared<WorldEngine::Scene>("Main");
    auto entity = scene->createEntity("Player");
    
    engine.setActiveScene(scene);
    return 0;
}
```

### Python

```python
import world_engine as we

engine = we.Engine()
engine.initialize()

scene = we.Scene("Main")
entity = scene.create_entity("Player")

engine.active_scene = scene
```

### TypeScript

```typescript
import { Engine, Scene } from '@worldengine/core';

const engine = new Engine();
engine.initialize();

const scene = new Scene('Main');
const entity = scene.createEntity('Player');

engine.setActiveScene(scene);
```

## Documentation

- [Build Instructions](BUILD.md) - Detailed build guide
- [API Reference](docs/API.md) - Complete API documentation
- [Examples](examples/) - Sample code for each language
- [Web Demo](src/html/demo.html) - Interactive demo

## Building for Different Platforms

### Linux
```bash
cmake -B build -S .
cmake --build build
```

### macOS
```bash
cmake -B build -S .
cmake --build build
```

### Windows
```bash
cmake -B build -S . -G "Visual Studio 17 2022"
cmake --build build --config Release
```

## Build Options

Configure the build with CMake options:

```bash
cmake -B build -S . \
  -DBUILD_PYTHON_BINDINGS=ON \
  -DBUILD_TYPESCRIPT=ON \
  -DBUILD_EXAMPLES=ON \
  -DBUILD_TESTS=ON
```

## Testing

```bash
cd build
ctest
```

## Installation

```bash
cmake --install build --prefix /usr/local
```

## License

See [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit pull requests.

## Version

Current version: 1.0.0

## Authors

World Engine Team

## Links

- Repository: https://github.com/Colt45en/world-Engine
- Issues: https://github.com/Colt45en/world-Engine/issues
