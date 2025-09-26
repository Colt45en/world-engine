#!/bin/bash

# NEXUS Nova Combat Quick Build & Test Script

set -e

NEXUS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${NEXUS_ROOT}/build-nova"
GAME_LOGIC_DIR="${NEXUS_ROOT}/game-logic"

echo "⚔️✨ NEXUS Nova Combat Integration - Quick Build ✨⚔️"
echo "===================================================="
echo "Root: $NEXUS_ROOT"
echo "Build: $BUILD_DIR"
echo ""

# Clean previous build if requested
if [ "$1" = "--clean" ]; then
    echo "🧹 Cleaning previous build..."
    rm -rf "$BUILD_DIR"
    echo "✅ Build directory cleaned"
fi

# Create build directory
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# Use the updated CMakeLists.txt
if [ -f "${GAME_LOGIC_DIR}/CMakeLists-updated.txt" ]; then
    echo "🔄 Using updated CMakeLists.txt..."
    cp "${GAME_LOGIC_DIR}/CMakeLists-updated.txt" "${GAME_LOGIC_DIR}/CMakeLists.txt"
fi

cd "$GAME_LOGIC_DIR"

# Configure CMake
echo "🔧 Configuring CMake for Nova Combat..."
cmake -S . -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release

# Build the project
echo "🔨 Building NEXUS Nova Combat Integration..."
cmake --build "$BUILD_DIR" --config Release --parallel

echo ""
echo "✅ Build complete!"
echo ""

# Check what was built
echo "📋 Available executables:"
cd "$BUILD_DIR"

for exe in nexus_resource_demo holy_beat_demo nexus_nova_combat_demo nexus_test_suite; do
    if [ -f "./${exe}" ] || [ -f "./${exe}.exe" ]; then
        echo "  ✅ ${exe}"
    else
        echo "  ❌ ${exe} (not found)"
    fi
done

echo ""

# Offer to run the Nova combat demo
if [ -f "./nexus_nova_combat_demo" ] || [ -f "./nexus_nova_combat_demo.exe" ]; then
    echo "🎮 Nova Combat Demo is ready!"
    echo ""
    echo "Run it with:"
    if [ -f "./nexus_nova_combat_demo" ]; then
        echo "  ./nexus_nova_combat_demo"
    else
        echo "  ./nexus_nova_combat_demo.exe"
    fi
    echo ""

    # Ask if user wants to run it now
    read -p "🚀 Run the Nova Combat Demo now? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "⚔️✨ Starting NEXUS Nova Combat Integration Demo... ✨⚔️"
        echo ""
        if [ -f "./nexus_nova_combat_demo" ]; then
            ./nexus_nova_combat_demo
        else
            ./nexus_nova_combat_demo.exe
        fi
    fi
else
    echo "⚠️ Nova Combat Demo was not built successfully"
    echo "   Check the build output above for errors"
fi

echo ""
echo "💡 Next steps:"
echo "   • Test different combat phases and quantum abilities"
echo "   • Experiment with sacred geometry movement patterns"
echo "   • Observe cognitive engine insights during combat"
echo "   • Try integrating with web visualization interfaces"
echo ""
echo "⚔️✨ NEXUS Nova Combat Integration ready! ✨⚔️"
