#!/bin/bash

echo "🧠 NEXUS Intelligence Operations Center - Startup Script"
echo "================================================================"

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo "❌ Node.js not found! Please install Node.js first."
    exit 1
fi

# Check if Python is installed
if ! command -v python3 &> /dev/null && ! command -v python &> /dev/null; then
    echo "❌ Python not found! Please install Python first."
    exit 1
fi

# Use python3 if available, otherwise python
if command -v python3 &> /dev/null; then
    PYTHON_CMD=python3
    PIP_CMD=pip3
else
    PYTHON_CMD=python
    PIP_CMD=pip
fi

echo "✅ Node.js found: $(node --version)"
echo "✅ Python found: $($PYTHON_CMD --version)"

# Install Python dependencies
echo ""
echo "📦 Installing Python dependencies..."
$PIP_CMD install -r requirements.txt

# Start WebSocket relay server in background
echo ""
echo "🚀 Starting WebSocket Relay (Port 9000)..."
cd websocket
node tier4_ws_relay.js &
WS_PID=$!
cd ..

# Wait for WebSocket server to start
sleep 3

# Set up cleanup function
cleanup() {
    echo ""
    echo "🛑 Shutting down servers..."
    kill $WS_PID 2>/dev/null
    exit 0
}
trap cleanup INT

# Start AI Bot Server
echo ""
echo "🧠 Starting AI Bot Server (Port 8000)..."
echo ""
echo "🌐 Intelligence Operations Center will be available at:"
echo "   http://localhost:8000"
echo ""
echo "📊 Individual panels:"
echo "   http://localhost:8000/intelligence-hub"
echo "   http://localhost:8000/librarian-math"
echo "   http://localhost:8000/nucleus-control-center"
echo ""
echo "Press Ctrl+C to stop the servers"
echo "================================================================"

$PYTHON_CMD ai_bot_server.py
