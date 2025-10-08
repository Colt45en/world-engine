@echo off
echo 🚀 Starting Nexus Core Chat Bridge...
echo.

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Error: Python is not installed or not in PATH
    echo Please install Python 3.8+ and try again
    pause
    exit /b 1
)

REM Check if pip is available
pip --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Error: pip is not installed or not in PATH
    pause
    exit /b 1
)

echo ✅ Python environment detected
echo.

REM Install dependencies
echo 📦 Installing dependencies...
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo ❌ Error: Failed to install dependencies
    pause
    exit /b 1
)

echo ✅ Dependencies installed
echo.

REM Start the bridge
echo 🌉 Starting Nexus Bridge on localhost:8888...
echo.
echo 🔧 Available endpoints:
echo   - http://localhost:8888/health (health check)
echo   - http://localhost:8888/query (RAG queries)
echo   - http://localhost:8888/stats (system stats)
echo   - http://localhost:8888/cleanup (memory cleanup)
echo.
echo 💡 Press Ctrl+C to stop the bridge
echo.

python nexus_bridge.py

echo.
echo 👋 Nexus Bridge stopped
pause
