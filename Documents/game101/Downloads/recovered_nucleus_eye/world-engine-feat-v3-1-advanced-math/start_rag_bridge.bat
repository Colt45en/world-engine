@echo off
echo ==============================================
echo       World Engine RAG Bridge Server
echo ==============================================
echo.

REM Check if Python is available
C:/Python312/python.exe --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python not found at C:/Python312/python.exe
    echo Please ensure Python 3.12+ is installed and accessible
    pause
    exit /b 1
)

echo Starting RAG Bridge Server...
echo.
echo Server will be available at: http://localhost:8888
echo.
echo API Endpoints:
echo   GET  /health - Health check
echo   GET  /query?q=^<question^> - Simple query
echo   POST /query - Advanced query with context
echo   POST /component-docs - Component documentation
echo.
echo Press Ctrl+C to stop the server
echo ==============================================
echo.

REM Start the server
C:/Python312/python.exe rag_bridge_server.py --host localhost --port 8888

echo.
echo Server stopped.
pause
