@echo off
echo 🧠 NEXUS Intelligence Operations Center - Startup Script
echo ================================================================

REM Check if Node.js is installed
node --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Node.js not found! Please install Node.js first.
    pause
    exit /b 1
)

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Python not found! Please install Python first.
    pause
    exit /b 1
)

echo ✅ Node.js found
echo ✅ Python found

REM Install Python dependencies if needed
echo.
echo 📦 Installing Python dependencies...
pip install -r requirements.txt

REM Start WebSocket relay server in background
echo.
echo 🚀 Starting WebSocket Relay (Port 9000)...
cd websocket
start "WebSocket Relay" cmd /k "node tier4_ws_relay.js"
cd ..

REM Wait a moment for WebSocket server to start
timeout /t 3 /nobreak >nul

REM Start AI Bot Server
echo.
echo 🧠 Starting AI Bot Server (Port 8000)...
echo.
echo 🌐 Intelligence Operations Center will be available at:
echo    http://localhost:8000
echo.
echo 📊 Individual panels:
echo    http://localhost:8000/intelligence-hub
echo    http://localhost:8000/librarian-math
echo    http://localhost:8000/nucleus-control-center
echo.
echo Press Ctrl+C to stop the server
echo ================================================================

python ai_bot_server.py
