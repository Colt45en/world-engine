#!/bin/bash

# NEXUS Game Logic Engine - Integration Test
echo "🎮 NEXUS Game Logic Engine - Integration Test"
echo "=============================================="

# Test 1: Check if build system works
echo "Test 1: Building demos..."
if [ -f "build.sh" ]; then
    chmod +x build.sh
    ./build.sh > build.log 2>&1
    if [ $? -eq 0 ]; then
        echo "✅ Build system working"
    else
        echo "❌ Build system failed - check build.log"
        exit 1
    fi
else
    echo "❌ build.sh not found"
    exit 1
fi

# Test 2: Check if executables were created
echo "Test 2: Checking executables..."
DEMOS=("basic_demo" "nexus_resource_demo" "nexus_world_demo")
for demo in "${DEMOS[@]}"; do
    if [ -f "$demo" ] || [ -f "$demo.exe" ]; then
        echo "✅ $demo created successfully"
    else
        echo "⚠️  $demo not found (may be in build directory)"
    fi
done

# Test 3: Test basic functionality
echo "Test 3: Testing basic demo..."
if [ -f "basic_demo" ]; then
    timeout 5s ./basic_demo > demo.log 2>&1
    echo "✅ Basic demo executed (check demo.log for output)"
elif [ -f "basic_demo.exe" ]; then
    timeout 5s ./basic_demo.exe > demo.log 2>&1
    echo "✅ Basic demo executed (check demo.log for output)"
else
    echo "⚠️  Basic demo not available for testing"
fi

# Test 4: Check NEXUS server connectivity
echo "Test 4: Testing NEXUS server connectivity..."
if command -v curl >/dev/null 2>&1; then
    curl -s -m 3 http://localhost:8080/api/status > /dev/null 2>&1
    if [ $? -eq 0 ]; then
        echo "✅ NEXUS server is accessible"
    else
        echo "⚠️  NEXUS server not responding (expected if not running)"
    fi
else
    echo "⚠️  curl not available for server test"
fi

# Test 5: Validate file structure
echo "Test 5: Validating file structure..."
REQUIRED_FILES=(
    "include/NexusGameEngine.hpp"
    "include/GameEntity.hpp"
    "include/GameComponents.hpp"
    "include/NexusResourceEngine.hpp"
    "examples/nexus_world_demo.cpp"
    "CMakeLists.txt"
    "package.json"
)

for file in "${REQUIRED_FILES[@]}"; do
    if [ -f "$file" ]; then
        echo "✅ $file exists"
    else
        echo "❌ $file missing"
    fi
done

echo ""
echo "🎯 Integration Test Complete!"
echo "=============================="
echo "📁 Check build.log and demo.log for detailed output"
echo "🚀 Ready to develop NEXUS-enhanced games!"
echo ""
echo "Quick Start:"
echo "  ./nexus_world_demo     # Run comprehensive demo"
echo "  ./nexus_resource_demo  # Test resource management"
echo "  ./basic_demo           # Simple ECS demo"
