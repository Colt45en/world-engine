# NEXUS Holy Beat System - Windows PowerShell Deployment Script
# Complete setup and deployment for Windows systems

param(
    [switch]$Clean,
    [switch]$NoTests,
    [switch]$NoServer,
    [switch]$Help
)

# Colors for PowerShell output
function Write-ColorText {
    param([string]$Text, [string]$Color = "White")

    switch ($Color) {
        "Red"    { Write-Host $Text -ForegroundColor Red }
        "Green"  { Write-Host $Text -ForegroundColor Green }
        "Blue"   { Write-Host $Text -ForegroundColor Blue }
        "Yellow" { Write-Host $Text -ForegroundColor Yellow }
        "Cyan"   { Write-Host $Text -ForegroundColor Cyan }
        "Magenta" { Write-Host $Text -ForegroundColor Magenta }
        default  { Write-Host $Text }
    }
}

# Configuration
$NexusRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$BuildDir = Join-Path $NexusRoot "build"
$GameLogicDir = Join-Path $NexusRoot "game-logic"
$WebPort = 8080
$NodeJSPort = 3000

Write-ColorText "🎵✨ NEXUS Holy Beat System - Windows Deployment ✨🎵" "Cyan"
Write-ColorText "=========================================================" "Cyan"
Write-ColorText "Root Directory: $NexusRoot" "Blue"
Write-ColorText "Build Directory: $BuildDir" "Blue"
Write-Host ""

# Help display
if ($Help) {
    Write-Host "NEXUS Holy Beat System Windows Deployment Script"
    Write-Host ""
    Write-Host "Usage: .\deploy.ps1 [options]"
    Write-Host ""
    Write-Host "Options:"
    Write-Host "  -Clean      Clean build directory before building"
    Write-Host "  -NoTests    Skip running tests"
    Write-Host "  -NoServer   Don't start Node.js server"
    Write-Host "  -Help       Show this help message"
    Write-Host ""
    exit 0
}

function Write-Section {
    param([string]$Title)
    Write-Host ""
    Write-ColorText "=== $Title ===" "Magenta"
}

function Test-CommandExists {
    param([string]$Command)
    try {
        Get-Command $Command -ErrorAction Stop | Out-Null
        return $true
    } catch {
        return $false
    }
}

function Setup-NodeJS {
    if (Test-Path (Join-Path $NexusRoot "package.json")) {
        Write-ColorText "📦 Installing Node.js dependencies..." "Yellow"
        Push-Location $NexusRoot

        if (Test-CommandExists "npm") {
            npm install
            if ($LASTEXITCODE -eq 0) {
                Write-ColorText "✅ Node.js dependencies installed" "Green"
            } else {
                Write-ColorText "❌ Failed to install Node.js dependencies" "Red"
                Pop-Location
                return $false
            }
        } elseif (Test-CommandExists "yarn") {
            yarn install
            if ($LASTEXITCODE -eq 0) {
                Write-ColorText "✅ Node.js dependencies installed" "Green"
            } else {
                Write-ColorText "❌ Failed to install Node.js dependencies" "Red"
                Pop-Location
                return $false
            }
        } else {
            Write-ColorText "❌ Neither npm nor yarn found. Please install Node.js first." "Red"
            Pop-Location
            return $false
        }
        Pop-Location
    } else {
        Write-ColorText "⚠️ No package.json found, skipping Node.js setup" "Yellow"
    }
    return $true
}

function Build-CPP {
    Write-Section "Building C++ Game Engine"

    # Check if CMakeLists.txt exists
    $CMakeFile = Join-Path $GameLogicDir "CMakeLists.txt"
    $NewCMakeFile = Join-Path $GameLogicDir "CMakeLists-new.txt"

    if (!(Test-Path $CMakeFile)) {
        if (Test-Path $NewCMakeFile) {
            Write-ColorText "🔄 Using new CMakeLists.txt..." "Yellow"
            Copy-Item $NewCMakeFile $CMakeFile
        } else {
            Write-ColorText "❌ No CMakeLists.txt found in game-logic directory" "Red"
            return $false
        }
    }

    Push-Location $GameLogicDir

    # Create build directory
    if (!(Test-Path $BuildDir)) {
        New-Item -ItemType Directory -Path $BuildDir | Out-Null
    }

    Push-Location $BuildDir

    Write-ColorText "🔧 Configuring CMake..." "Yellow"

    # Configure for Visual Studio (most common on Windows)
    if (Test-CommandExists "cmake") {
        cmake .. -G "Visual Studio 17 2022" -A x64 -DCMAKE_BUILD_TYPE=Release
        if ($LASTEXITCODE -ne 0) {
            # Try with Visual Studio 2019 if 2022 isn't available
            cmake .. -G "Visual Studio 16 2019" -A x64 -DCMAKE_BUILD_TYPE=Release
            if ($LASTEXITCODE -ne 0) {
                # Fall back to MinGW or default generator
                cmake .. -DCMAKE_BUILD_TYPE=Release
            }
        }
    } else {
        Write-ColorText "❌ CMake not found. Please install CMake 3.16+" "Red"
        Pop-Location
        Pop-Location
        return $false
    }

    Write-ColorText "🔨 Building NEXUS Core Engine..." "Yellow"
    cmake --build . --config Release --parallel

    if ($LASTEXITCODE -eq 0) {
        Write-ColorText "✅ C++ build completed successfully" "Green"

        # List built executables
        Write-ColorText "📋 Built executables:" "Blue"
        $executables = @("nexus_resource_demo", "holy_beat_demo", "nexus_test_suite")
        foreach ($exe in $executables) {
            $exePath = Join-Path $BuildDir "$exe.exe"
            $exePathDebug = Join-Path $BuildDir "Debug\$exe.exe"
            $exePathRelease = Join-Path $BuildDir "Release\$exe.exe"

            if ((Test-Path $exePath) -or (Test-Path $exePathDebug) -or (Test-Path $exePathRelease)) {
                Write-ColorText "  ✅ $exe" "Green"
            } else {
                Write-ColorText "  ⚠️ $exe (not found)" "Yellow"
            }
        }

        Pop-Location
        Pop-Location
        return $true
    } else {
        Write-ColorText "❌ C++ build failed" "Red"
        Pop-Location
        Pop-Location
        return $false
    }
}

function Setup-Web {
    Write-Section "Setting Up Web Components"

    Push-Location $NexusRoot

    # Check for web files
    $webFiles = @(
        "nexus-live-bridge.html",
        "nexus-3d-visualization.html",
        "system-overview.html",
        "beat-room.html"
    )

    Write-ColorText "🌐 Checking web components..." "Yellow"
    foreach ($file in $webFiles) {
        if (Test-Path $file) {
            Write-ColorText "  ✅ $file" "Green"
        } else {
            Write-ColorText "  ⚠️ $file (not found)" "Yellow"
        }
    }

    # Copy web files to build directory for easy access
    $webBuildDir = Join-Path $BuildDir "web"
    if (!(Test-Path $webBuildDir)) {
        New-Item -ItemType Directory -Path $webBuildDir | Out-Null
    }

    foreach ($file in $webFiles) {
        if (Test-Path $file) {
            Copy-Item $file $webBuildDir
        }
    }

    Write-ColorText "✅ Web components prepared" "Green"
    Pop-Location
}

function Start-NodeJSServer {
    $serverFile = Join-Path $NexusRoot "server.js"
    if (Test-Path $serverFile) {
        Write-Section "Starting Node.js Server"

        Push-Location $NexusRoot
        Write-ColorText "🚀 Starting Node.js server on port $NodeJSPort..." "Yellow"

        # Start server in background
        $logFile = Join-Path $NexusRoot "nodejs_server.log"
        $pidFile = Join-Path $NexusRoot "nodejs_server.pid"

        $process = Start-Process -FilePath "node" -ArgumentList "server.js" -RedirectStandardOutput $logFile -RedirectStandardError $logFile -PassThru -NoNewWindow
        $process.Id | Out-File -FilePath $pidFile

        # Wait a moment and check if it's running
        Start-Sleep 2
        if (!$process.HasExited) {
            Write-ColorText "✅ Node.js server started (PID: $($process.Id))" "Green"
            Write-ColorText "🌐 Server URL: http://localhost:$NodeJSPort" "Blue"
        } else {
            Write-ColorText "❌ Failed to start Node.js server" "Red"
        }

        Pop-Location
    } else {
        Write-ColorText "⚠️ No server.js found, skipping Node.js server" "Yellow"
    }
}

function Run-Tests {
    Write-Section "Running Tests"

    Push-Location $BuildDir

    $testExe = $null
    $testPaths = @(
        "nexus_test_suite.exe",
        "Debug\nexus_test_suite.exe",
        "Release\nexus_test_suite.exe"
    )

    foreach ($path in $testPaths) {
        if (Test-Path $path) {
            $testExe = $path
            break
        }
    }

    if ($testExe) {
        Write-ColorText "🧪 Running NEXUS test suite..." "Yellow"
        & $testExe --basic

        if ($LASTEXITCODE -eq 0) {
            Write-ColorText "✅ All tests passed" "Green"
        } else {
            Write-ColorText "❌ Some tests failed" "Red"
        }
    } else {
        Write-ColorText "⚠️ Test suite not found, skipping tests" "Yellow"
    }

    Pop-Location
}

function Create-LaunchScript {
    Write-Section "Creating Launch Script"

    $launchScript = Join-Path $BuildDir "launch_holy_beat_system.bat"

    @"
@echo off
echo 🎵✨ Starting NEXUS Holy Beat System ✨🎵
echo.
echo 🌐 Web Interfaces:
echo    • web\nexus-live-bridge.html - Real-time dashboard
echo    • web\nexus-3d-visualization.html - 3D sacred geometry
echo.
echo 🎮 Starting C++ engine in 3 seconds...
echo    Press Ctrl+C to stop the demo
echo.

timeout /t 3 /nobreak > nul

if exist "holy_beat_demo.exe" (
    holy_beat_demo.exe
) else if exist "Release\holy_beat_demo.exe" (
    Release\holy_beat_demo.exe
) else if exist "Debug\holy_beat_demo.exe" (
    Debug\holy_beat_demo.exe
) else if exist "nexus_resource_demo.exe" (
    echo ⚠️ Holy Beat demo not found, running resource demo instead
    nexus_resource_demo.exe
) else if exist "Release\nexus_resource_demo.exe" (
    echo ⚠️ Holy Beat demo not found, running resource demo instead
    Release\nexus_resource_demo.exe
) else if exist "Debug\nexus_resource_demo.exe" (
    echo ⚠️ Holy Beat demo not found, running resource demo instead
    Debug\nexus_resource_demo.exe
) else (
    echo ❌ No demo executables found
    pause
)
"@ | Out-File -FilePath $launchScript -Encoding ASCII

    # Also create PowerShell version
    $launchScriptPS = Join-Path $BuildDir "launch_holy_beat_system.ps1"

    @"
Write-Host "🎵✨ Starting NEXUS Holy Beat System ✨🎵"
Write-Host ""
Write-Host "🌐 Web Interfaces:"
Write-Host "   • web\nexus-live-bridge.html - Real-time dashboard"
Write-Host "   • web\nexus-3d-visualization.html - 3D sacred geometry"
Write-Host ""
Write-Host "🎮 Starting C++ engine in 3 seconds..."
Write-Host "   Press Ctrl+C to stop the demo"
Write-Host ""

Start-Sleep 3

`$exePaths = @(
    "holy_beat_demo.exe",
    "Release\holy_beat_demo.exe",
    "Debug\holy_beat_demo.exe"
)

`$foundExe = `$null
foreach (`$path in `$exePaths) {
    if (Test-Path `$path) {
        `$foundExe = `$path
        break
    }
}

if (`$foundExe) {
    & `$foundExe
} else {
    `$fallbackPaths = @(
        "nexus_resource_demo.exe",
        "Release\nexus_resource_demo.exe",
        "Debug\nexus_resource_demo.exe"
    )

    `$foundFallback = `$null
    foreach (`$path in `$fallbackPaths) {
        if (Test-Path `$path) {
            `$foundFallback = `$path
            break
        }
    }

    if (`$foundFallback) {
        Write-Host "⚠️ Holy Beat demo not found, running resource demo instead"
        & `$foundFallback
    } else {
        Write-Host "❌ No demo executables found"
        Read-Host "Press Enter to continue"
    }
}
"@ | Out-File -FilePath $launchScriptPS -Encoding UTF8

    Write-ColorText "✅ Launch scripts created" "Green"
}

function Show-FinalInstructions {
    Write-Section "🎉 Deployment Complete!"

    Write-ColorText "✅ NEXUS Holy Beat System is ready to run!" "Green"
    Write-Host ""
    Write-ColorText "🚀 Quick Start Instructions:" "Cyan"
    Write-ColorText "  1. Navigate to: $BuildDir" "Blue"
    Write-ColorText "  2. Run: .\launch_holy_beat_system.bat (or .ps1)" "Blue"
    Write-ColorText "  3. Open web interfaces in your browser:" "Blue"
    Write-ColorText "     • $BuildDir\web\nexus-live-bridge.html" "Yellow"
    Write-ColorText "     • $BuildDir\web\nexus-3d-visualization.html" "Yellow"
    Write-Host ""

    Write-ColorText "🎮 Available Demos:" "Cyan"
    Push-Location $BuildDir
    $executables = @("nexus_resource_demo", "holy_beat_demo", "nexus_test_suite")
    foreach ($exe in $executables) {
        $found = $false
        $paths = @("$exe.exe", "Release\$exe.exe", "Debug\$exe.exe")
        foreach ($path in $paths) {
            if (Test-Path $path) {
                Write-ColorText "  ✅ $path" "Green"
                $found = $true
                break
            }
        }
        if (!$found) {
            Write-ColorText "  ⚠️ $exe (not found)" "Yellow"
        }
    }
    Pop-Location
    Write-Host ""

    $pidFile = Join-Path $NexusRoot "nodejs_server.pid"
    if (Test-Path $pidFile) {
        Write-ColorText "🌐 Node.js Server:" "Cyan"
        Write-ColorText "  Running on: http://localhost:$NodeJSPort" "Green"
        $pid = Get-Content $pidFile
        Write-ColorText "  Stop with: Stop-Process -Id $pid" "Blue"
        Write-Host ""
    }

    Write-ColorText "💡 Pro Tips:" "Magenta"
    Write-ColorText "  • The C++ engine will stream data to WebSocket port $WebPort" "Blue"
    Write-ColorText "  • Use Ctrl+C to gracefully stop any running demo" "Blue"
    Write-ColorText "  • Check the generated logs for performance metrics" "Blue"
    Write-ColorText "  • Modify examples\ directory to create custom demos" "Blue"
    Write-Host ""

    Write-ColorText "🎵✨ Enjoy the NEXUS Holy Beat System! ✨🎵" "Cyan"
}

function Main {
    Write-Section "System Checks"

    # Check required tools
    Write-ColorText "🔍 Checking system requirements..." "Yellow"

    if (Test-CommandExists "cmake") {
        Write-ColorText "  ✅ CMake found" "Green"
    } else {
        Write-ColorText "  ❌ CMake not found - please install CMake 3.16+" "Red"
        exit 1
    }

    # Check for C++ compiler (Visual Studio, MinGW, etc.)
    if ((Test-CommandExists "cl") -or (Test-CommandExists "gcc") -or (Test-CommandExists "clang")) {
        Write-ColorText "  ✅ C++ compiler found" "Green"
    } else {
        Write-ColorText "  ❌ No C++ compiler found - please install Visual Studio or MinGW" "Red"
        exit 1
    }

    $NodeAvailable = Test-CommandExists "node"
    if ($NodeAvailable) {
        Write-ColorText "  ✅ Node.js found" "Green"
    } else {
        Write-ColorText "  ⚠️ Node.js not found - web server features will be limited" "Yellow"
    }

    # Run deployment steps
    if ($Clean) {
        Write-Section "Cleaning Build Directory"
        if (Test-Path $BuildDir) {
            Remove-Item $BuildDir -Recurse -Force
        }
        Write-ColorText "✅ Build directory cleaned" "Green"
    }

    if ($NodeAvailable -and !$NoServer) {
        $result = Setup-NodeJS
        if (!$result) {
            Write-ColorText "⚠️ Node.js setup failed, continuing..." "Yellow"
        }
    }

    if (!(Build-CPP)) {
        exit 1
    }

    Setup-Web

    if ($NodeAvailable -and !$NoServer) {
        Start-NodeJSServer
    }

    if (!$NoTests) {
        Run-Tests
    }

    Create-LaunchScript
    Show-FinalInstructions
}

# Run main function
Main
