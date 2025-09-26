#!/bin/bash

# NEXUS Game Logic Build Script
# Supports Linux/macOS/WSL compilation and testing

echo "🎵✨ Building NEXUS Game Logic System... ✨🎵"

# Check for required tools
check_dependency() {
    if ! command -v $1 &> /dev/null; then
        echo "❌ $1 is not installed. Please install it first."
        exit 1
    fi
}

echo "🔍 Checking dependencies..."
check_dependency "g++"
check_dependency "cmake"

# Clean previous build
echo "🧹 Cleaning previous build..."
rm -rf build
rm -f demo nexus_resource_demo

# Create build directory
echo "📁 Creating build directory..."
mkdir -p build
cd build

# Configure with CMake
echo "⚙️  Configuring with CMake..."
cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_EXAMPLES=ON

# Build
echo "🔨 Building NEXUS Game Logic..."
cmake --build . --config Release --parallel

if [ $? -ne 0 ]; then
    echo "❌ Build failed!"
    exit 1
fi

cd ..

# Build standalone demos (your original approach)
echo "🎮 Building standalone demos..."

# Original demo
echo "📦 Building original resource engine demo..."
g++ -std=c++17 -O2 -Wall -Wextra -pedantic \
    -I./include \
    ./examples/basic_game_demo.cpp \
    -o basic_demo -pthread

if [ $? -eq 0 ]; then
    echo "✅ Original demo built successfully"
else
    echo "⚠️  Original demo build failed"
fi

# NEXUS enhanced demo
echo "📦 Building NEXUS resource engine demo..."
g++ -std=c++17 -O2 -Wall -Wextra -pedantic \
    -I./include \
    ./examples/nexus_resource_demo.cpp \
    -o nexus_resource_demo -pthread

if [ $? -eq 0 ]; then
    echo "✅ NEXUS resource demo built successfully"
else
    echo "⚠️  NEXUS resource demo build failed"
fi

# NEXUS world simulation demo
echo "📦 Building NEXUS world simulation demo..."
g++ -std=c++17 -O2 -Wall -Wextra -pedantic \
    -I./include \
    ./examples/nexus_world_demo.cpp \
    -o nexus_world_demo -pthread

if [ $? -eq 0 ]; then
    echo "✅ NEXUS world demo built successfully"
else
    echo "⚠️  NEXUS world demo build failed"
fi

# NEXUS Quantum Protocol demo
echo "📦 Building NEXUS Quantum Protocol demo..."
g++ -std=c++17 -O2 -Wall -Wextra -pedantic \
    -I./include \
    ./examples/nexus_quantum_demo.cpp \
    -o nexus_quantum_demo -pthread

if [ $? -eq 0 ]; then
    echo "✅ NEXUS Quantum demo built successfully"
else
    echo "⚠️  NEXUS Quantum demo build failed"
fi

# NEXUS Cognitive Recursive Keeper demo
echo "📦 Building NEXUS Cognitive demo..."
g++ -std=c++17 -O2 -Wall -Wextra -pedantic \
    -I./include \
    ./examples/nexus_cognitive_demo.cpp \
    -o nexus_cognitive_demo -pthread

if [ $? -eq 0 ]; then
    echo "✅ NEXUS Cognitive demo built successfully"
else
    echo "⚠️  NEXUS Cognitive demo build failed"
fi

echo ""
echo "🎯 Build Summary:"
echo "=================="
if [ -f "build/examples/basic_game_demo" ]; then
    echo "✅ CMake Basic Demo: ./build/examples/basic_game_demo"
fi
if [ -f "basic_demo" ]; then
    echo "✅ Standalone Basic Demo: ./basic_demo"
fi
if [ -f "nexus_resource_demo" ]; then
    echo "✅ NEXUS Resource Demo: ./nexus_resource_demo"
fi
if [ -d "build" ] && [ "$(ls -A build)" ]; then
    echo "✅ CMake Build: ./build/"
fi

echo ""
echo "🚀 Quick Test Commands:"
echo "======================="
echo "# Test original basic game demo:"
echo "./basic_demo"
echo ""
echo "# Test NEXUS resource engine (recommended):"
echo "./nexus_resource_demo"
echo ""
echo "# Test CMake-built version:"
echo "./build/examples/basic_game_demo"
echo ""
echo "# Build Node.js bindings:"
echo "npm install && npm run build"
echo ""
echo "# Test Node.js integration:"
echo "node examples/test_bindings.js"

echo ""
echo "💡 Integration Notes:"
echo "===================="
echo "• Resource engine mirrors your JavaScript API exactly"
echo "• Async loading with configurable throttling (4 jobs/frame)"
echo "• Distance + frustum culling for performance"
echo "• Multi-level LOD: ultra/high/medium/low/billboard"
echo "• Audio-reactive resources sync to BPM beats"
echo "• Art-reactive resources follow petal patterns"
echo "• Physics-enabled resources respond to forces"
echo "• 2.5D world with Z-axis elevation support"
echo "• Full NEXUS Holy Beat System integration"

echo ""
echo "🎵✨ NEXUS Game Logic build complete! ✨🎵"
