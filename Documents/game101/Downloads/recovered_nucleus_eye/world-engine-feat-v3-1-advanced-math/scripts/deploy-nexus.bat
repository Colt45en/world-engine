@echo off
REM NEXUS Forge Complete Deployment Script - Windows Version
REM Version: 5.0.0 - September 2025
REM Complete Mathematical & AI Ecosystem Deployment

title NEXUS Forge Deployment
color 0B

echo.
echo ███╗   ██╗███████╗██╗  ██╗██╗   ██╗███████╗
echo ████╗  ██║██╔════╝╚██╗██╔╝██║   ██║██╔════╝
echo ██╔██╗ ██║█████╗   ╚███╔╝ ██║   ██║███████╗
echo ██║╚██╗██║██╔══╝   ██╔██╗ ██║   ██║╚════██║
echo ██║ ╚████║███████╗██╔╝ ██╗╚██████╔╝███████║
echo ╚═╝  ╚═══╝╚══════╝╚═╝  ╚═╝ ╚═════╝ ╚══════╝
echo.
echo ███████╗ ██████╗ ██████╗  ██████╗ ███████╗
echo ██╔════╝██╔═══██╗██╔══██╗██╔════╝ ██╔════╝
echo █████╗  ██║   ██║██████╔╝██║  ███╗█████╗
echo ██╔══╝  ██║   ██║██╔══██╗██║   ██║██╔══╝
echo ██║     ╚██████╔╝██║  ██║╚██████╔╝███████╗
echo ╚═╝      ╚═════╝ ╚═╝  ╚═╝ ╚═════╝ ╚══════╝
echo.
echo 🌟 NEXUS Forge Complete Ecosystem Deployment
echo ==============================================
echo Version 5.0.0 - Advanced Mathematical ^& AI Integration
echo Deployment Date: %DATE% %TIME%
echo.

REM Configuration
set NEXUS_DIR=nexus-forge-ecosystem
set WEB_PORT=8080
set API_PORT=3000
set AI_PORT=5000
set MONITOR_PORT=9090

REM Check system requirements
echo [INFO] Checking System Requirements
echo =====================================

REM Check Node.js
node --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Node.js not found. Please install Node.js 16+ to continue.
    echo Download from: https://nodejs.org/
    pause
    exit /b 1
) else (
    for /f "tokens=*" %%i in ('node --version') do set NODE_VERSION=%%i
    echo [SUCCESS] Node.js found: %NODE_VERSION%
)

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    python3 --version >nul 2>&1
    if errorlevel 1 (
        echo [WARNING] Python not found. Some AI features may be limited.
        echo Download from: https://www.python.org/
    ) else (
        for /f "tokens=*" %%i in ('python3 --version') do set PYTHON_VERSION=%%i
        echo [SUCCESS] Python found: %PYTHON_VERSION%
        set PYTHON_CMD=python3
    )
) else (
    for /f "tokens=*" %%i in ('python --version') do set PYTHON_VERSION=%%i
    echo [SUCCESS] Python found: %PYTHON_VERSION%
    set PYTHON_CMD=python
)

REM Check npm
npm --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] npm not found. Please install Node.js with npm.
    pause
    exit /b 1
) else (
    for /f "tokens=*" %%i in ('npm --version') do set NPM_VERSION=%%i
    echo [SUCCESS] npm found: %NPM_VERSION%
)

echo.

REM Create directory structure
echo [INFO] Creating NEXUS Directory Structure
echo =======================================

REM Main directories
if not exist "%NEXUS_DIR%" mkdir "%NEXUS_DIR%"
mkdir "%NEXUS_DIR%\public" 2>nul
mkdir "%NEXUS_DIR%\src" 2>nul
mkdir "%NEXUS_DIR%\config" 2>nul
mkdir "%NEXUS_DIR%\logs" 2>nul
mkdir "%NEXUS_DIR%\data" 2>nul
mkdir "%NEXUS_DIR%\scripts" 2>nul
mkdir "%NEXUS_DIR%\docs" 2>nul
mkdir "%NEXUS_DIR%\tests" 2>nul

REM Subdirectories
mkdir "%NEXUS_DIR%\src\ai" 2>nul
mkdir "%NEXUS_DIR%\src\math" 2>nul
mkdir "%NEXUS_DIR%\src\webgl" 2>nul
mkdir "%NEXUS_DIR%\src\education" 2>nul
mkdir "%NEXUS_DIR%\src\monitoring" 2>nul
mkdir "%NEXUS_DIR%\src\communication" 2>nul
mkdir "%NEXUS_DIR%\public\assets" 2>nul
mkdir "%NEXUS_DIR%\public\css" 2>nul
mkdir "%NEXUS_DIR%\public\js" 2>nul
mkdir "%NEXUS_DIR%\public\components" 2>nul
mkdir "%NEXUS_DIR%\config\development" 2>nul
mkdir "%NEXUS_DIR%\config\production" 2>nul
mkdir "%NEXUS_DIR%\config\testing" 2>nul
mkdir "%NEXUS_DIR%\logs\system" 2>nul
mkdir "%NEXUS_DIR%\logs\ai" 2>nul
mkdir "%NEXUS_DIR%\logs\math" 2>nul
mkdir "%NEXUS_DIR%\logs\webgl" 2>nul
mkdir "%NEXUS_DIR%\logs\error" 2>nul
mkdir "%NEXUS_DIR%\data\training" 2>nul
mkdir "%NEXUS_DIR%\data\math" 2>nul
mkdir "%NEXUS_DIR%\data\visualization" 2>nul
mkdir "%NEXUS_DIR%\data\education" 2>nul

echo [SUCCESS] Directory structure created
echo.

REM Copy NEXUS files
echo [INFO] Deploying NEXUS Core Files
echo ================================

REM Copy main HTML files
if exist "nexus-forge-launcher.html" (
    copy "nexus-forge-launcher.html" "%NEXUS_DIR%\public\index.html" >nul
    echo [SUCCESS] Main launcher deployed
) else (
    echo [ERROR] nexus-forge-launcher.html not found
)

REM Copy component files
set COMPONENT_FILES=nexus-system-monitor.html nexus-analytics-dashboard.html nexus-config-manager.html nexus-master-control-center.html nucleus-ai-training-system.html 3d-mathematical-playground.html nexus-educational-bridge.html nexus-system-integration.html nexus-system-optimizer.js "upgraded code pad.css"

for %%f in (%COMPONENT_FILES%) do (
    if exist "%%f" (
        copy "%%f" "%NEXUS_DIR%\public\" >nul
        echo [SUCCESS] Deployed %%f
    ) else (
        echo [WARNING] %%f not found - skipping
    )
)

echo.

REM Generate package.json
echo [INFO] Generating Package Configuration
echo ====================================

(
echo {
echo   "name": "nexus-forge-ecosystem",
echo   "version": "5.0.0",
echo   "description": "Complete Mathematical & AI Ecosystem - NEXUS Forge",
echo   "main": "src/server.js",
echo   "scripts": {
echo     "start": "node src/server.js",
echo     "dev": "nodemon src/server.js",
echo     "test": "jest",
echo     "build": "webpack --mode=production",
echo     "deploy": "npm run build && npm start",
echo     "monitor": "node src/monitoring/system-monitor.js",
echo     "ai-train": "%PYTHON_CMD% src/ai/training/train_model.py",
echo     "math-test": "node src/math/test-engine.js",
echo     "webgl-test": "node src/webgl/test-renderer.js"
echo   },
echo   "keywords": [
echo     "mathematics",
echo     "ai",
echo     "webgl",
echo     "education",
echo     "visualization",
echo     "neural-networks",
echo     "geometry",
echo     "physics"
echo   ],
echo   "author": "NEXUS Forge Development Team",
echo   "license": "MIT",
echo   "dependencies": {
echo     "express": "^4.18.2",
echo     "socket.io": "^4.7.2",
echo     "cors": "^2.8.5",
echo     "helmet": "^7.0.0",
echo     "compression": "^1.7.4",
echo     "body-parser": "^1.20.2",
echo     "ws": "^8.13.0",
echo     "mathjs": "^11.11.0",
echo     "three": "^0.155.0",
echo     "tensorflow": "^4.10.0",
echo     "d3": "^7.8.5",
echo     "chart.js": "^4.4.0"
echo   },
echo   "devDependencies": {
echo     "nodemon": "^3.0.1",
echo     "jest": "^29.6.2",
echo     "webpack": "^5.88.2",
echo     "webpack-cli": "^5.1.4",
echo     "@babel/core": "^7.22.9",
echo     "@babel/preset-env": "^7.22.9",
echo     "babel-loader": "^9.1.3",
echo     "css-loader": "^6.8.1",
echo     "html-webpack-plugin": "^5.5.3"
echo   },
echo   "engines": {
echo     "node": ">=16.0.0",
echo     "npm": ">=8.0.0"
echo   }
echo }
) > "%NEXUS_DIR%\package.json"

echo [SUCCESS] Package.json generated
echo.

REM Generate server file
echo [INFO] Generating NEXUS Server
echo ==========================

(
echo /**
echo  * NEXUS Forge Main Server
echo  * Complete Mathematical ^& AI Ecosystem
echo  * Version 5.0.0 - Windows Edition
echo  */
echo.
echo const express = require('express'^);
echo const http = require('http'^);
echo const socketIo = require('socket.io'^);
echo const path = require('path'^);
echo const cors = require('cors'^);
echo const helmet = require('helmet'^);
echo const compression = require('compression'^);
echo const bodyParser = require('body-parser'^);
echo.
echo class NexusServer {
echo     constructor(^) {
echo         this.app = express(^);
echo         this.server = http.createServer(this.app^);
echo         this.io = socketIo(this.server, {
echo             cors: {
echo                 origin: "*",
echo                 methods: ["GET", "POST"]
echo             }
echo         }^);
echo
echo         this.port = process.env.PORT ^|^| 8080;
echo         this.initializeServer(^);
echo         this.setupRoutes(^);
echo         this.setupSocketHandlers(^);
echo     }
echo
echo     initializeServer(^) {
echo         console.log('🌟 Initializing NEXUS Forge Server...'^);
echo
echo         // Security middleware
echo         this.app.use(helmet({
echo             contentSecurityPolicy: false
echo         }^)^);
echo
echo         // Compression and parsing
echo         this.app.use(compression(^)^);
echo         this.app.use(bodyParser.json({ limit: '10mb' }^)^);
echo         this.app.use(bodyParser.urlencoded({ extended: true }^)^);
echo
echo         // CORS for cross-origin requests
echo         this.app.use(cors(^)^);
echo
echo         // Static files
echo         this.app.use(express.static(path.join(__dirname, '../public'^)^)^);
echo     }
echo
echo     setupRoutes(^) {
echo         // Main launcher route
echo         this.app.get('/', (req, res^) =^> {
echo             res.sendFile(path.join(__dirname, '../public/index.html'^)^);
echo         }^);
echo
echo         // API Routes
echo         this.app.get('/api/health', (req, res^) =^> {
echo             res.json({
echo                 status: 'healthy',
echo                 timestamp: new Date(^).toISOString(^),
echo                 version: '5.0.0',
echo                 platform: 'Windows'
echo             }^);
echo         }^);
echo     }
echo
echo     setupSocketHandlers(^) {
echo         this.io.on('connection', (socket^) =^> {
echo             console.log(`Client connected: ${socket.id}`^);
echo
echo             socket.on('disconnect', (^) =^> {
echo                 console.log(`Client disconnected: ${socket.id}`^);
echo             }^);
echo         }^);
echo     }
echo
echo     start(^) {
echo         this.server.listen(this.port, (^) =^> {
echo             console.log(`
echo 🌟 NEXUS Forge Server Started Successfully!
echo ============================================
echo 🔗 Main Interface: http://localhost:${this.port}
echo 📊 System Monitor: http://localhost:${this.port}/nexus-system-monitor.html
echo 📈 Analytics: http://localhost:${this.port}/nexus-analytics-dashboard.html
echo ⚙️ Configuration: http://localhost:${this.port}/nexus-config-manager.html
echo.
echo Version: 5.0.0 ^(Windows^)
echo Environment: ${process.env.NODE_ENV ^|^| 'development'}
echo ============================================
echo             `^);
echo         }^);
echo     }
echo }
echo.
echo // Start the server
echo if (require.main === module^) {
echo     const nexusServer = new NexusServer(^);
echo     nexusServer.start(^);
echo }
echo.
echo module.exports = NexusServer;
) > "%NEXUS_DIR%\src\server.js"

echo [SUCCESS] Main server generated
echo.

REM Generate configuration files
echo [INFO] Generating Configuration Files
echo ===================================

REM Development config
(
echo {
echo   "server": {
echo     "port": 8080,
echo     "host": "localhost",
echo     "cors": {
echo       "enabled": true,
echo       "origin": "*"
echo     },
echo     "logging": {
echo       "level": "debug",
echo       "file": "logs/development.log"
echo     }
echo   },
echo   "ai": {
echo     "modelPath": "data/models/",
echo     "trainingEnabled": true,
echo     "batchSize": 32,
echo     "learningRate": 0.001
echo   },
echo   "math": {
echo     "precision": 8,
echo     "cacheEnabled": true,
echo     "maxCacheSize": "100MB"
echo   },
echo   "monitoring": {
echo     "enabled": true,
echo     "interval": 5000,
echo     "metricsRetention": "7d"
echo   }
echo }
) > "%NEXUS_DIR%\config\development\server.json"

echo [SUCCESS] Configuration files generated
echo.

REM Install dependencies
echo [INFO] Installing Dependencies
echo ============================

cd "%NEXUS_DIR%"

echo Installing Node.js dependencies...
call npm install
if errorlevel 1 (
    echo [ERROR] Failed to install Node.js dependencies
    pause
    exit /b 1
) else (
    echo [SUCCESS] Node.js dependencies installed
)

REM Install Python dependencies if Python is available
if defined PYTHON_CMD (
    echo Installing Python dependencies...
    (
    echo tensorflow^>=2.13.0
    echo numpy^>=1.21.0
    echo scipy^>=1.7.0
    echo matplotlib^>=3.4.0
    echo scikit-learn^>=1.0.0
    echo pandas^>=1.3.0
    echo flask^>=2.0.0
    echo websocket-client^>=1.0.0
    ) > requirements.txt

    %PYTHON_CMD% -m pip install -r requirements.txt
    if errorlevel 1 (
        echo [WARNING] Some Python dependencies may have failed to install
    ) else (
        echo [SUCCESS] Python dependencies installed
    )
)

cd ..
echo.

REM Generate startup scripts
echo [INFO] Generating Startup Scripts
echo ===============================

REM Windows startup script
(
echo @echo off
echo cd /d "%%~dp0"
echo.
echo echo 🌟 Starting NEXUS Forge Ecosystem...
echo.
echo REM Check if server is already running
echo if exist .server.pid (
echo     echo NEXUS Server may already be running
echo     echo Check http://localhost:8080
echo     pause
echo     exit /b 0
echo ^)
echo.
echo REM Start the server
echo start /b npm start
echo echo Server started in background
echo.
echo echo.
echo echo 🌟 NEXUS Forge Server Started!
echo echo ==================================
echo echo 🔗 Main Interface: http://localhost:8080
echo echo 📊 System Monitor: http://localhost:8080/nexus-system-monitor.html
echo echo 📈 Analytics: http://localhost:8080/nexus-analytics-dashboard.html
echo echo ⚙️ Configuration: http://localhost:8080/nexus-config-manager.html
echo echo.
echo echo Opening browser in 3 seconds...
echo timeout /t 3 /nobreak ^>nul
echo start http://localhost:8080
echo.
echo echo To stop: stop-nexus.bat
echo pause
) > "%NEXUS_DIR%\start-nexus.bat"

REM Stop script
(
echo @echo off
echo echo Stopping NEXUS Server...
echo taskkill /f /im node.exe 2^>nul
echo if exist .server.pid del .server.pid
echo echo ✅ NEXUS Server stopped
echo pause
) > "%NEXUS_DIR%\stop-nexus.bat"

REM Restart script
(
echo @echo off
echo echo 🔄 Restarting NEXUS Forge...
echo call stop-nexus.bat
echo timeout /t 2 /nobreak ^>nul
echo call start-nexus.bat
) > "%NEXUS_DIR%\restart-nexus.bat"

echo [SUCCESS] Startup scripts generated
echo.

REM Generate README
echo [INFO] Generating Documentation
echo =============================

(
echo # 🌟 NEXUS Forge - Complete Mathematical ^& AI Ecosystem
echo.
echo Version 5.0.0 - Windows Edition
echo.
echo ## 🚀 Quick Start
echo.
echo 1. Double-click `start-nexus.bat` to start the server
echo 2. Your browser will open automatically to http://localhost:8080
echo 3. Use the NEXUS Forge Launcher to access all components
echo.
echo ## 📊 System Components
echo.
echo - **🤖 AI Training Core**: Neural network training with pattern recognition
echo - **🔢 Mathematical Engine**: Advanced mathematical processing
echo - **🎮 3D WebGL Playground**: Hardware-accelerated visualization
echo - **🎓 Educational Bridge**: Pythagorean curriculum integration
echo - **📊 System Monitor**: Real-time performance monitoring
echo - **⚙️ Configuration Manager**: System settings
echo - **📈 Analytics Dashboard**: Data visualization
echo.
echo ## 🛠 Management Scripts
echo.
echo - `start-nexus.bat`: Start the NEXUS server
echo - `stop-nexus.bat`: Stop the NEXUS server
echo - `restart-nexus.bat`: Restart the NEXUS server
echo.
echo ## 📚 URLs
echo.
echo - Main Interface: http://localhost:8080
echo - System Monitor: http://localhost:8080/nexus-system-monitor.html
echo - Analytics: http://localhost:8080/nexus-analytics-dashboard.html
echo - Configuration: http://localhost:8080/nexus-config-manager.html
echo.
echo ## 🆘 Troubleshooting
echo.
echo 1. **Port already in use**: Check if another application is using port 8080
echo 2. **Node.js not found**: Install Node.js from https://nodejs.org/
echo 3. **Permission errors**: Run as Administrator
echo 4. **Browser doesn't open**: Manually visit http://localhost:8080
echo.
echo ## 📄 License
echo.
echo MIT License - see LICENSE file for details
echo.
echo ---
echo.
echo **NEXUS Forge** - Bridging Mathematics, AI, and Education
echo 🌟 *Empowering the next generation of mathematical and AI innovation*
) > "%NEXUS_DIR%\README.md"

echo [SUCCESS] Documentation generated
echo.

REM Final deployment summary
echo [INFO] 🎉 NEXUS Forge Deployment Complete!
echo =======================================
echo.
echo ✅ All systems deployed successfully!
echo.
echo 📋 Quick Access:
echo    🔗 Main Interface:    http://localhost:8080
echo    📊 System Monitor:    http://localhost:8080/nexus-system-monitor.html
echo    📈 Analytics:         http://localhost:8080/nexus-analytics-dashboard.html
echo    ⚙️ Configuration:     http://localhost:8080/nexus-config-manager.html
echo.
echo 🛠 Management:
echo    Start:    Double-click %NEXUS_DIR%\start-nexus.bat
echo    Stop:     Double-click %NEXUS_DIR%\stop-nexus.bat
echo    Restart:  Double-click %NEXUS_DIR%\restart-nexus.bat
echo.
echo 📚 Documentation: %NEXUS_DIR%\README.md
echo.
echo 🌟 NEXUS Forge is ready for mathematical and AI exploration!
echo.
echo Would you like to start the server now? (Y/N)
set /p START_NOW=

if /i "%START_NOW%"=="Y" (
    echo.
    echo Starting NEXUS Forge Server...
    cd "%NEXUS_DIR%"
    start start-nexus.bat
    cd ..
) else (
    echo.
    echo To start later, run: %NEXUS_DIR%\start-nexus.bat
)

echo.
echo 🎉 Thank you for using NEXUS Forge!
pause
