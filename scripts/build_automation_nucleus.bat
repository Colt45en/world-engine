@echo off
echo === NEXUS Automation Nucleus C++ Build Script ===
echo.

REM Create build directory
if not exist "build" (
    echo Creating build directory...
    mkdir build
)

cd build

echo Configuring CMake...
cmake .. -DCMAKE_BUILD_TYPE=Release

if %errorlevel% neq 0 (
    echo CMake configuration failed!
    pause
    exit /b 1
)

echo Building Automation Nucleus...
cmake --build . --config Release

if %errorlevel% neq 0 (
    echo Build failed!
    pause
    exit /b 1
)

echo.
echo === Build Complete! ===
echo.
echo To run the Automation Nucleus:
echo   cd build
echo   .\Release\AutomationNucleus.exe
echo.
echo Or run it now? (y/n)
set /p run_now=

if /i "%run_now%"=="y" (
    echo.
    echo Starting Automation Nucleus Engine Room...
    echo.
    if exist "Release\AutomationNucleus.exe" (
        Release\AutomationNucleus.exe
    ) else if exist "AutomationNucleus.exe" (
        AutomationNucleus.exe
    ) else (
        echo Executable not found! Check build output.
    )
)

pause
