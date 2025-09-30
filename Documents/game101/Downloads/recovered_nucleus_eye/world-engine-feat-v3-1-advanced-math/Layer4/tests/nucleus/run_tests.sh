#!/bin/bash
# Nucleus System Test Runner

echo "🧠 Nucleus System Test Suite Runner"
echo "================================="

# Check if Node.js is available
if ! command -v node &> /dev/null; then
    echo "❌ Node.js is not installed or not in PATH"
    exit 1
fi

# Navigate to test directory
cd "$(dirname "$0")" || exit 1

echo "📂 Current directory: $(pwd)"
echo ""

# Run integration test
echo "🔄 Running integration tests..."
if node integration_test.js; then
    echo "✅ Integration tests completed successfully"
else
    echo "❌ Integration tests failed"
    exit 1
fi

echo ""
echo "🎉 All nucleus tests completed successfully!"
echo "================================="
