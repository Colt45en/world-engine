#!/bin/bash

echo "🧪 NEXUS Quantum Integration Test"
echo "================================="

# Test compilation of the new quantum demo
echo "Test 1: Compiling NEXUS Quantum demo..."

# Try to compile with basic error checking
g++ -std=c++17 -Wall -Wextra -I./include -c ./examples/nexus_quantum_demo.cpp -o nexus_quantum_test.o 2>&1

if [ $? -eq 0 ]; then
    echo "✅ Quantum demo compiles successfully"
    rm -f nexus_quantum_test.o

    # Try full compilation
    echo "Test 2: Full compilation test..."
    g++ -std=c++17 -O2 -Wall -Wextra -I./include ./examples/nexus_quantum_demo.cpp -o nexus_quantum_test -pthread 2>&1

    if [ $? -eq 0 ]; then
        echo "✅ Full quantum demo builds successfully"

        # Try to run the demo briefly
        echo "Test 3: Runtime test (5 seconds)..."
        timeout 5s ./nexus_quantum_test > quantum_output.log 2>&1

        if [ $? -eq 0 ] || [ $? -eq 124 ]; then  # 124 = timeout (expected)
            echo "✅ Quantum demo runs without crashes"
            echo "📄 Sample output:"
            head -n 10 quantum_output.log
        else
            echo "❌ Runtime test failed"
            cat quantum_output.log
        fi

        # Cleanup
        rm -f nexus_quantum_test quantum_output.log
    else
        echo "❌ Full compilation failed"
    fi
else
    echo "❌ Compilation test failed"
fi

# Test header dependencies
echo "Test 4: Header dependency check..."

# Check that each header compiles independently
HEADERS=("NexusProtocol.hpp" "NexusVisuals.hpp" "NexusTrailRenderer.hpp")

for header in "${HEADERS[@]}"; do
    echo "#include \"$header\"" > header_test.cpp
    echo "int main() { return 0; }" >> header_test.cpp

    g++ -std=c++17 -I./include -c header_test.cpp 2>&1
    if [ $? -eq 0 ]; then
        echo "✅ $header compiles independently"
    else
        echo "❌ $header has compilation issues"
    fi

    rm -f header_test.cpp header_test.o
done

echo "Test 5: Integration with existing NEXUS system..."

# Test that new headers work with existing system
cat > integration_test.cpp << 'EOF'
#include "NexusProtocol.hpp"
#include "NexusVisuals.hpp"
#include "NexusTrailRenderer.hpp"
#include "GameEntity.hpp"
#include "NexusGameEngine.hpp"

int main() {
    // Test basic instantiation
    NEXUS::NexusProtocol& protocol = NEXUS::NexusProtocol::Instance();

    // Test enum usage
    protocol.SetProcessingMode(NEXUS::ProcessingMode::COSINE);

    // Test palette generation
    auto palette = NEXUS::NexusVisuals::GetPalette(NEXUS::ProcessingMode::MIRROR);

    // Test trail manager
    auto& trail_mgr = NEXUS::NexusTrailManager::Instance();
    auto trail = trail_mgr.CreateTrail("test");

    // Test entity system integration
    GameEntity entity;
    entity.Update(0.016f);

    return 0;
}
EOF

g++ -std=c++17 -I./include integration_test.cpp -o integration_test 2>&1
if [ $? -eq 0 ]; then
    echo "✅ Integration with existing NEXUS system successful"

    # Quick runtime test
    ./integration_test
    if [ $? -eq 0 ]; then
        echo "✅ Integration runtime test passed"
    else
        echo "⚠️  Integration runtime issues"
    fi
else
    echo "❌ Integration compilation failed"
fi

rm -f integration_test.cpp integration_test

echo ""
echo "🎯 NEXUS Quantum Integration Test Complete!"
echo "==========================================="
echo "🚀 Ready for Quantum-enhanced NEXUS development!"
