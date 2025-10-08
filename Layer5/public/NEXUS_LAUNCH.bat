@echo off
title NEXUS Forge Complete System Launcher v5.0.0
color 0B
echo.
echo  ╔══════════════════════════════════════════════════════════════════════════════╗
echo  ║                     🧠 NEXUS FORGE COMPLETE SYSTEM v5.0.0                    ║
echo  ║                        Advanced AI Training Ecosystem                        ║
echo  ╚══════════════════════════════════════════════════════════════════════════════╝
echo.
echo [INFO] Starting NEXUS Forge Complete System...
echo.

REM Set current directory
cd /d "%~dp0"

REM Check if required files exist
echo [SYSTEM CHECK] Verifying system files...
if not exist "nexus-master-control-center.html" (
    echo [ERROR] Master Control Center not found!
    echo [INFO] Please ensure all NEXUS Forge files are in the current directory.
    pause
    exit /b 1
)

if not exist "nexus-3d-mathematical-playground.html" (
    echo [ERROR] 3D Mathematical Playground not found!
    pause
    exit /b 1
)

if not exist "nucleus-training-interface.html" (
    echo [ERROR] Nucleus Training Interface not found!
    pause
    exit /b 1
)

echo [✓] All core system files verified
echo.

REM Check for optional system files
if not exist "nexus-system-optimizer.js" (
    echo [WARNING] System Optimizer not found - Performance optimization disabled
) else (
    echo [✓] System Optimizer available
)

if not exist "nexus-system-integration.js" (
    echo [WARNING] System Integration not found - Cross-system communication limited
) else (
    echo [✓] System Integration available
)

echo.
echo [INITIALIZATION] Preparing system launch...
echo.

REM Create a simple HTTP server for local development
echo [NETWORK] Setting up local development server...

REM Check for Python
python --version >nul 2>&1
if %errorlevel% == 0 (
    echo [✓] Python detected - Using Python HTTP server
    start "NEXUS Server" cmd /k "echo [SERVER] NEXUS Forge Development Server && echo [INFO] Server running on http://localhost:8000 && echo [INFO] Press Ctrl+C to stop server && python -m http.server 8000"
    timeout /t 3 /nobreak >nul
    set SERVER_URL=http://localhost:8000
    goto :launch_browser
)

REM Check for Node.js
node --version >nul 2>&1
if %errorlevel% == 0 (
    echo [✓] Node.js detected - Using Node.js HTTP server
    start "NEXUS Server" cmd /k "echo [SERVER] NEXUS Forge Development Server && echo [INFO] Server running on http://localhost:8000 && echo [INFO] Press Ctrl+C to stop server && npx http-server -p 8000"
    timeout /t 3 /nobreak >nul
    set SERVER_URL=http://localhost:8000
    goto :launch_browser
)

REM Fallback to file:// protocol
echo [WARNING] No HTTP server available - Using file:// protocol
echo [INFO] Some features may be limited without a local server
set SERVER_URL=file:///%CD%

:launch_browser
echo.
echo [LAUNCH] Starting NEXUS Forge ecosystem...
echo.

REM Launch Master Control Center
echo [SYSTEM] Launching Master Control Center...
start "" "%SERVER_URL%/nexus-master-control-center.html"
timeout /t 2 /nobreak >nul

echo [✓] Master Control Center launched
echo.

REM Optional: Launch additional systems directly
echo [OPTION] Would you like to launch all systems immediately? (Y/N)
set /p LAUNCH_ALL=Enter choice:

if /i "%LAUNCH_ALL%"=="Y" (
    echo.
    echo [SYSTEM] Launching all NEXUS systems...
    echo.

    echo [SYSTEM] Launching Nucleus AI Training Interface...
    start "" "%SERVER_URL%/nucleus-training-interface.html"
    timeout /t 1 /nobreak >nul

    echo [SYSTEM] Launching 3D Mathematical Playground...
    start "" "%SERVER_URL%/nexus-3d-mathematical-playground.html"
    timeout /t 1 /nobreak >nul

    echo [✓] All systems launched successfully
) else (
    echo [INFO] Systems can be launched individually from the Master Control Center
)

echo.
echo  ╔══════════════════════════════════════════════════════════════════════════════╗
echo  ║                            🚀 SYSTEM READY 🚀                               ║
echo  ╠══════════════════════════════════════════════════════════════════════════════╣
echo  ║  Master Control Center: Access all NEXUS systems from one interface         ║
echo  ║  3D Mathematical Playground: Interactive 3D math with real-time calculations ║
echo  ║  Nucleus AI Training: Direct AI communication and training capabilities      ║
echo  ║  Educational Bridge: Guided learning with Tier 5-8 curriculum               ║
echo  ║                                                                              ║
echo  ║  💡 KEYBOARD SHORTCUTS (in Control Center):                                 ║
echo  ║  • Ctrl+1: Launch Nucleus AI System                                         ║
echo  ║  • Ctrl+2: Launch 3D Playground                                             ║
echo  ║  • Ctrl+3: Launch Educational Bridge                                        ║
echo  ║  • Ctrl+I: Initialize All Systems                                           ║
echo  ║  • Ctrl+D: Run System Diagnostics                                           ║
echo  ║                                                                              ║
echo  ║  🔧 SYSTEM FEATURES:                                                         ║
echo  ║  • Real-time AI training with mathematical patterns                         ║
echo  ║  • Hardware-accelerated 3D WebGL visualization                              ║
echo  ║  • Cross-system communication and data sharing                              ║
echo  ║  • Performance optimization and monitoring                                  ║
echo  ║  • Educational curriculum integration                                       ║
echo  ║  • Export capabilities for training data and results                       ║
echo  ╚══════════════════════════════════════════════════════════════════════════════╝
echo.

if defined SERVER_URL (
    if not "%SERVER_URL%"=="file:///%CD%" (
        echo [SERVER] Development server is running at %SERVER_URL%
        echo [INFO] Leave the server window open while using NEXUS Forge
        echo [INFO] Close the server window when finished to stop the server
    )
)

echo.
echo [INFO] NEXUS Forge Complete System is now running!
echo [INFO] Use the Master Control Center to launch and manage all systems.
echo [INFO] Check the console output for system status and debug information.
echo.

REM System monitoring loop
echo [MONITOR] System monitoring active...
echo [INFO] Press any key to show system status, or Ctrl+C to exit
echo.

:monitor_loop
timeout /t 5 /nobreak >nul
if errorlevel 1 (
    REM Key was pressed
    echo.
    echo  ╔══════════════════════════════════════════════════════════════════════════════╗
    echo  ║                              SYSTEM STATUS                                   ║
    echo  ╚══════════════════════════════════════════════════════════════════════════════╝
    echo [STATUS] NEXUS Forge Complete System v5.0.0
    echo [TIME] %date% %time%

    REM Check if server is still running
    if defined SERVER_URL (
        if not "%SERVER_URL%"=="file:///%CD%" (
            tasklist /fi "windowtitle eq NEXUS Server*" 2>nul | find "cmd.exe" >nul
            if %errorlevel% == 0 (
                echo [✓] Development server: RUNNING
            ) else (
                echo [✗] Development server: STOPPED
                echo [WARNING] Server may have been closed - some features may not work
            )
        ) else (
            echo [INFO] File protocol mode: ACTIVE
        )
    )

    echo [INFO] Master Control Center should be open in your browser
    echo [INFO] Use the Control Center to monitor individual system status
    echo.
    echo [OPTIONS] Available commands:
    echo   R - Restart all systems
    echo   S - Show system URLs
    echo   H - Show help information
    echo   Q - Quit system monitor
    echo   Any other key - Continue monitoring
    echo.
    set /p COMMAND=Enter command:

    if /i "%COMMAND%"=="R" (
        echo [RESTART] Restarting NEXUS systems...
        start "" "%SERVER_URL%/nexus-master-control-center.html"
        echo [✓] Master Control Center restarted
    ) else if /i "%COMMAND%"=="S" (
        echo [URLS] System URLs:
        echo   Master Control: %SERVER_URL%/nexus-master-control-center.html
        echo   3D Playground: %SERVER_URL%/nexus-3d-mathematical-playground.html
        echo   Nucleus AI: %SERVER_URL%/nucleus-training-interface.html
    ) else if /i "%COMMAND%"=="H" (
        echo [HELP] NEXUS Forge Complete System Help:
        echo   • Use Master Control Center for system management
        echo   • Each system window can be used independently
        echo   • Systems communicate automatically when running
        echo   • Mathematical results are shared between systems
        echo   • AI training data is synchronized across components
        echo   • Performance is monitored and optimized automatically
    ) else if /i "%COMMAND%"=="Q" (
        echo [EXIT] Stopping system monitor...
        echo [INFO] Systems will continue running in browser windows
        echo [INFO] Close browser windows manually to stop systems
        goto :end
    )

    echo.
    echo [MONITOR] Continuing system monitoring...
)
goto :monitor_loop

:end
echo.
echo [SHUTDOWN] NEXUS Forge system monitor stopped
echo [INFO] Browser windows may still be running with active systems
echo [INFO] Close browser windows to fully stop the NEXUS ecosystem
echo.
pause
exit /b 0
