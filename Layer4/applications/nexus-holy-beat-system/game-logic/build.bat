@echo off
REM NEXUS Game Logic Build Script for Windows
REM Supports MSVC compilation and testing

echo 🎵✨ Building NEXUS Game Logic System (Windows)... ✨🎵
echo.

REM Check for Visual Studio tools
where cl >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo ❌ Visual Studio compiler not found. Please run from Visual Studio Developer Command Prompt.
    echo 💡 Or install Visual Studio with C++ development tools.
    pause
    exit /b 1
)

where cmake >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo ❌ CMake not found. Please install CMake and add to PATH.
    pause
    exit /b 1
)

echo ✅ Found Visual Studio compiler and CMake

REM Clean previous build
echo 🧹 Cleaning previous build...
if exist build rmdir /s /q build
if exist nexus_resource_demo.exe del nexus_resource_demo.exe
if exist basic_demo.exe del basic_demo.exe

REM Create build directory
echo 📁 Creating build directory...
mkdir build
cd build

REM Configure with CMake
echo ⚙️ Configuring with CMake...
cmake .. -G "Visual Studio 16 2019" -A x64 -DCMAKE_BUILD_TYPE=Release -DBUILD_EXAMPLES=ON

if %ERRORLEVEL% NEQ 0 (
    echo ❌ CMake configuration failed!
    pause
    exit /b 1
)

REM Build
echo 🔨 Building NEXUS Game Logic...
cmake --build . --config Release

if %ERRORLEVEL% NEQ 0 (
    echo ❌ Build failed!
    pause
    exit /b 1
)

cd ..

REM Build standalone demos
echo 🎮 Building standalone demos...

echo 📦 Building basic game demo...
cl /EHsc /std:c++17 /O2 /I./include examples\basic_game_demo.cpp /Fe:basic_demo.exe
if %ERRORLEVEL% EQU 0 (
    echo ✅ Basic demo built successfully
) else (
    echo ⚠️ Basic demo build failed
)

echo 📦 Building NEXUS resource demo...
cl /EHsc /std:c++17 /O2 /I./include examples\nexus_resource_demo.cpp /Fe:nexus_resource_demo.exe
if %ERRORLEVEL% EQU 0 (
    echo ✅ NEXUS resource demo built successfully
) else (
    echo ⚠️ NEXUS resource demo build failed
)

echo 📦 Building NEXUS world demo...
cl /EHsc /std:c++17 /O2 /I./include examples\nexus_world_demo.cpp /Fe:nexus_world_demo.exe
if %ERRORLEVEL% EQU 0 (
    echo ✅ NEXUS world demo built successfully
) else (
    echo ⚠️ NEXUS world demo build failed
)

echo 📦 Building NEXUS Quantum demo...
cl /EHsc /std:c++17 /O2 /I./include examples\nexus_quantum_demo.cpp /Fe:nexus_quantum_demo.exe
if %ERRORLEVEL% EQU 0 (
    echo ✅ NEXUS Quantum demo built successfully
) else (
    echo ⚠️ NEXUS Quantum demo build failed
)

echo 📦 Building NEXUS Cognitive demo...
cl /EHsc /std:c++17 /O2 /I./include examples\nexus_cognitive_demo.cpp /Fe:nexus_cognitive_demo.exe
if %ERRORLEVEL% EQU 0 (
    echo ✅ NEXUS Cognitive demo built successfully
) else (
    echo ⚠️ NEXUS Cognitive demo build failed
)

REM Clean up compiler temp files
if exist *.obj del *.obj
if exist *.pdb del *.pdb

echo.
echo 🎯 Build Summary:
echo ==================
if exist "build\examples\Release\basic_game_demo.exe" (
    echo ✅ CMake Basic Demo: build\examples\Release\basic_game_demo.exe
)
if exist "basic_demo.exe" (
    echo ✅ Standalone Basic Demo: basic_demo.exe
)
if exist "nexus_resource_demo.exe" (
    echo ✅ NEXUS Resource Demo: nexus_resource_demo.exe
)

echo.
echo 🚀 Quick Test Commands:
echo =======================
echo # Test basic game demo:
echo basic_demo.exe
echo.
echo # Test NEXUS resource engine ^(recommended^):
echo nexus_resource_demo.exe
echo.
echo # Test CMake-built version:
echo build\examples\Release\basic_game_demo.exe
echo.
echo # Build Node.js bindings:
echo npm install ^&^& npm run build
echo.
echo # Test Node.js integration:
echo node examples\test_bindings.js

echo.
echo 💡 Integration Features:
echo ========================
echo • Resource engine mirrors your JavaScript API exactly
echo • Async loading with configurable throttling ^(4 jobs/frame^)
echo • Distance + frustum culling for performance
echo • Multi-level LOD: ultra/high/medium/low/billboard
echo • Audio-reactive resources sync to BPM beats
echo • Art-reactive resources follow petal patterns
echo • Physics-enabled resources respond to forces
echo • 2.5D world with Z-axis elevation support
echo • Full NEXUS Holy Beat System integration

echo.
echo 🎵✨ NEXUS Game Logic build complete! ✨🎵
pause
