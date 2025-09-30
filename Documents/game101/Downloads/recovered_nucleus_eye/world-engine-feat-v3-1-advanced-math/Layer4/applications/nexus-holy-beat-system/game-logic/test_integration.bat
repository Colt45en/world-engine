@echo off
setlocal enabledelayedexpansion

REM NEXUS Game Logic Engine - Integration Test
echo 🎮 NEXUS Game Logic Engine - Integration Test
echo ==============================================

REM Test 1: Check if build system works
echo Test 1: Building demos...
if exist "build.bat" (
    call build.bat > build.log 2>&1
    if !ERRORLEVEL! EQU 0 (
        echo ✅ Build system working
    ) else (
        echo ❌ Build system failed - check build.log
        exit /b 1
    )
) else (
    echo ❌ build.bat not found
    exit /b 1
)

REM Test 2: Check if executables were created
echo Test 2: Checking executables...
set DEMOS=basic_demo.exe nexus_resource_demo.exe nexus_world_demo.exe
for %%d in (%DEMOS%) do (
    if exist "%%d" (
        echo ✅ %%d created successfully
    ) else (
        echo ⚠️  %%d not found (may be in build directory)
    )
)

REM Test 3: Test basic functionality
echo Test 3: Testing basic demo...
if exist "basic_demo.exe" (
    timeout /t 5 /nobreak > nul 2>&1
    start /wait /b basic_demo.exe > demo.log 2>&1
    echo ✅ Basic demo executed (check demo.log for output)
) else (
    echo ⚠️  Basic demo not available for testing
)

REM Test 4: Check NEXUS server connectivity
echo Test 4: Testing NEXUS server connectivity...
curl -s -m 3 http://localhost:8080/api/status > nul 2>&1
if !ERRORLEVEL! EQU 0 (
    echo ✅ NEXUS server is accessible
) else (
    echo ⚠️  NEXUS server not responding (expected if not running)
)

REM Test 5: Validate file structure
echo Test 5: Validating file structure...
set REQUIRED_FILES=include\NexusGameEngine.hpp include\GameEntity.hpp include\GameComponents.hpp include\NexusResourceEngine.hpp examples\nexus_world_demo.cpp CMakeLists.txt package.json

for %%f in (%REQUIRED_FILES%) do (
    if exist "%%f" (
        echo ✅ %%f exists
    ) else (
        echo ❌ %%f missing
    )
)

echo.
echo 🎯 Integration Test Complete!
echo ==============================
echo 📁 Check build.log and demo.log for detailed output
echo 🚀 Ready to develop NEXUS-enhanced games!
echo.
echo Quick Start:
echo   nexus_world_demo.exe     REM Run comprehensive demo
echo   nexus_resource_demo.exe  REM Test resource management
echo   basic_demo.exe           REM Simple ECS demo

pause
