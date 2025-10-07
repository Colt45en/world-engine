@echo off
REM NEXUS World Engine Startup Script
echo Starting NEXUS World Engine Server...

cd /d "c:\Users\colte\World-engine\Documents\game101\Downloads\recovered_nucleus_eye\world-engine-feat-v3-1-advanced-math"

echo Current directory: %CD%
echo Installing dependencies...
call npm install

echo Starting NEXUS server...
node nexus-server.js

pause