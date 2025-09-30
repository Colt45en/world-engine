@echo off
echo === NEXUS Automation Nucleus Direct Compilation ===
echo.

REM Check for Visual Studio compiler
where cl >nul 2>nul
if %errorlevel% equ 0 (
    echo Using Visual Studio compiler cl.exe...
    goto :compile_msvc
)

REM Check for MinGW g++
where g++ >nul 2>nul
if %errorlevel% equ 0 (
    echo Using MinGW g++ compiler...
    goto :compile_gcc
)

REM Check for clang++
where clang++ >nul 2>nul
if %errorlevel% equ 0 (
    echo Using Clang++ compiler...
    goto :compile_clang
)

echo No suitable C++ compiler found!
echo Please install one of:
echo   - Visual Studio Build Tools with cl.exe
echo   - MinGW with g++.exe
echo   - LLVM Clang with clang++.exe
pause
exit /b 1

:compile_msvc
echo Compiling with MSVC...
cl /EHsc /std:c++17 /I"../include" AutomationNucleus.cpp AutomationNucleusMain.cpp /Fe:AutomationNucleus.exe /link
if %errorlevel% neq 0 goto :compile_failed
goto :compile_success

:compile_gcc
echo Compiling with g++...
g++ -std=c++17 -I../include -pthread -O2 AutomationNucleus.cpp AutomationNucleusMain.cpp -o AutomationNucleus.exe
if %errorlevel% neq 0 goto :compile_failed
goto :compile_success

:compile_clang
echo Compiling with clang++...
clang++ -std=c++17 -I../include -pthread -O2 AutomationNucleus.cpp AutomationNucleusMain.cpp -o AutomationNucleus.exe
if %errorlevel% neq 0 goto :compile_failed
goto :compile_success

:compile_failed
echo.
echo === Compilation Failed! ===
echo Check the error messages above.
pause
exit /b 1

:compile_success
echo.
echo === Compilation Successful! ===
echo.
echo AutomationNucleus.exe has been created.
echo.
echo To run the Automation Nucleus:
echo   .\AutomationNucleus.exe
echo.
echo Run it now? y/n
set /p run_now=

if /i "%run_now%"=="y" (
    echo.
    echo Starting Automation Nucleus Engine Room...
    echo Press Ctrl+C to stop gracefully.
    echo.
    .\AutomationNucleus.exe
)

pause
