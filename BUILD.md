# Building World Engine

This document describes how to build World Engine from source.

## Prerequisites

### Required
- **CMake** 3.15 or higher
- **C++ Compiler** with C++17 support (GCC 7+, Clang 5+, MSVC 2017+)
- **Git** for cloning the repository

### Optional (for full multi-language support)
- **Python 3.7+** with development headers for Python bindings
- **Node.js 14+** and **npm** for TypeScript/JavaScript support
- **TypeScript 5.0+** for TypeScript compilation

## Quick Start

### Clone the Repository
```bash
git clone https://github.com/Colt45en/world-Engine.git
cd world-Engine
```

### Build with CMake

#### Linux/macOS
```bash
mkdir build
cd build
cmake ..
cmake --build .
```

#### Windows (Visual Studio)
```bash
mkdir build
cd build
cmake .. -G "Visual Studio 17 2022"
cmake --build . --config Release
```

## Build Options

World Engine supports several CMake options to customize the build:

| Option | Default | Description |
|--------|---------|-------------|
| `BUILD_SHARED_LIBS` | ON | Build shared libraries instead of static |
| `BUILD_PYTHON_BINDINGS` | ON | Build Python bindings (requires Python) |
| `BUILD_TYPESCRIPT` | ON | Build TypeScript components (requires Node.js) |
| `BUILD_EXAMPLES` | ON | Build example programs |
| `BUILD_TESTS` | ON | Build tests |
| `INSTALL_WEB_ASSETS` | ON | Install HTML/web assets |

### Example: Minimal C++ Only Build
```bash
cmake .. -DBUILD_PYTHON_BINDINGS=OFF -DBUILD_TYPESCRIPT=OFF -DBUILD_EXAMPLES=OFF
```

### Example: Full Multi-Language Build
```bash
cmake .. -DBUILD_PYTHON_BINDINGS=ON -DBUILD_TYPESCRIPT=ON -DBUILD_EXAMPLES=ON
```

## Build Steps Explained

### 1. Configure
```bash
cmake -B build -S .
```
This generates the build files in the `build` directory.

### 2. Build
```bash
cmake --build build
```
This compiles all enabled components.

### 3. Test (Optional)
```bash
cd build
ctest
```
Run the test suite to verify the build.

### 4. Install (Optional)
```bash
cmake --install build --prefix /usr/local
```
Install World Engine system-wide.

## Platform-Specific Notes

### Linux
- Install build dependencies:
  ```bash
  sudo apt-get install cmake g++ python3-dev nodejs npm
  ```

### macOS
- Install via Homebrew:
  ```bash
  brew install cmake python node
  ```

### Windows
- Install Visual Studio 2019 or later with C++ support
- Install CMake from https://cmake.org/
- Install Python from https://python.org/
- Install Node.js from https://nodejs.org/

## Language-Specific Build Details

### C++ Core
The C++ core is always built and produces:
- `libworld_engine.so` (Linux)
- `libworld_engine.dylib` (macOS)
- `world_engine.dll` (Windows)

Headers are installed to `include/world_engine/`.

### Python Bindings
When `BUILD_PYTHON_BINDINGS=ON`:
1. CMake locates Python installation
2. Builds C++ Python extension module
3. Installs `world_engine` Python package

**Usage:**
```python
import world_engine as we
engine = we.Engine()
```

### TypeScript/JavaScript
When `BUILD_TYPESCRIPT=ON`:
1. CMake runs `npm install` in `src/typescript/`
2. Compiles TypeScript to JavaScript
3. Generates type definitions (.d.ts files)
4. Installs to `share/world_engine/js/`

**Usage:**
```typescript
import { Engine } from '@worldengine/core';
const engine = new Engine();
```

### HTML/Web Assets
When `INSTALL_WEB_ASSETS=ON`:
- Installs HTML demo pages
- Installs CSS stylesheets
- Installs JavaScript files
- Located in `share/world_engine/html/`

## Troubleshooting

### CMake can't find Python
```bash
cmake .. -DPython3_EXECUTABLE=/path/to/python3
```

### CMake can't find Node.js
```bash
cmake .. -DNODE_EXECUTABLE=/path/to/node
```

### Build fails with C++17 errors
Make sure your compiler supports C++17:
- GCC 7 or later
- Clang 5 or later
- MSVC 2017 or later

### TypeScript compilation fails
Install TypeScript globally:
```bash
npm install -g typescript
```

## Advanced Configuration

### Custom Install Prefix
```bash
cmake .. -DCMAKE_INSTALL_PREFIX=/custom/path
```

### Debug Build
```bash
cmake .. -DCMAKE_BUILD_TYPE=Debug
```

### Release Build with Optimizations
```bash
cmake .. -DCMAKE_BUILD_TYPE=Release
```

### Verbose Build Output
```bash
cmake --build build --verbose
```

## Integration with Other Projects

### Using CMake
```cmake
find_package(WorldEngine REQUIRED)
target_link_libraries(your_target WorldEngine::world_engine)
```

### Using pkg-config
```bash
pkg-config --cflags --libs world_engine
```

## Next Steps

After building:
1. Run the examples: `./build/bin/examples/cpp_example`
2. Check the tests: `cd build && ctest`
3. View the HTML demo: Open `build/share/world_engine/html/index.html`
4. Read the API documentation

## Support

For issues and questions:
- GitHub Issues: https://github.com/Colt45en/world-Engine/issues
- Documentation: See `docs/` directory
