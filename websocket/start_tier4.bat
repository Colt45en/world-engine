@echo off
echo ========================================
echo    Tier-4 WebSocket Integration
echo ========================================
echo.

REM Check if Node.js is installed
node --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: Node.js is not installed or not in PATH
    echo Please install Node.js from https://nodejs.org/
    pause
    exit /b 1
)

echo Node.js found, starting integration...
echo.

REM Run the setup script
node setup_tier4_integration.js

pause
