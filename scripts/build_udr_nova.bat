@echo off
REM Build script for UDR Nova Omega Call Handler System
echo.
echo 🚀 Building UDR Nova Omega Call Handler System...
echo.

REM Create build directory
if not exist "build" mkdir build
cd build

REM Configure with CMake
echo ⚙️  Configuring build...
cmake .. -DCMAKE_BUILD_TYPE=Release

REM Build the project
echo 🔨 Building project...
cmake --build . --config Release

REM Check if build was successful
if %ERRORLEVEL% NEQ 0 (
    echo ❌ Build failed!
    cd ..
    exit /b 1
)

echo ✅ Build completed successfully!
echo.
echo 📋 Available executables:
if exist "Release\AutomationNucleus.exe" (
    echo   - AutomationNucleus.exe  ^(Main application^)
)
if exist "Debug\AutomationNucleus.exe" (
    echo   - AutomationNucleus.exe  ^(Main application^)
)
if exist "AutomationNucleus.exe" (
    echo   - AutomationNucleus.exe  ^(Main application^)
)
if exist "Release\UDRNovaTest.exe" (
    echo   - UDRNovaTest.exe        ^(Test suite^)
)
if exist "Debug\UDRNovaTest.exe" (
    echo   - UDRNovaTest.exe        ^(Test suite^)
)
if exist "UDRNovaTest.exe" (
    echo   - UDRNovaTest.exe        ^(Test suite^)
)

echo.
echo 🧪 Running UDR Nova Omega tests...
echo.

REM Run tests if they exist
if exist "Release\UDRNovaTest.exe" (
    Release\UDRNovaTest.exe
) else if exist "Debug\UDRNovaTest.exe" (
    Debug\UDRNovaTest.exe
) else if exist "UDRNovaTest.exe" (
    UDRNovaTest.exe
) else (
    echo ⚠️  UDRNovaTest executable not found
)

echo.
echo 🎯 To run the main application:
if exist "Release\AutomationNucleus.exe" (
    echo   Release\AutomationNucleus.exe
) else if exist "Debug\AutomationNucleus.exe" (
    echo   Debug\AutomationNucleus.exe
) else if exist "AutomationNucleus.exe" (
    echo   AutomationNucleus.exe
) else (
    echo ⚠️  AutomationNucleus executable not found
)

cd ..
echo.
echo 🏁 Build script completed!
pause
