@echo off
REM Windows batch script for Docker management

set RED=[91m
set GREEN=[92m
set YELLOW=[93m
set BLUE=[94m
set NC=[0m

echo %BLUE%[INFO]%NC% Nexus Docker Management (Windows)

REM Check if Docker is running
docker info >nul 2>&1
if %errorlevel% neq 0 (
    echo %RED%[ERROR]%NC% Docker is not running. Please start Docker Desktop first.
    pause
    exit /b 1
)

if "%1"=="dev" (
    echo %BLUE%[INFO]%NC% Starting development environment...
    if not exist ".env" (
        echo %YELLOW%[WARNING]%NC% Creating .env from .env.example
        copy .env.example .env
        echo %YELLOW%[WARNING]%NC% Please edit .env file before proceeding
        pause
    )

    REM Prepare bridge files
    if not exist "docker\bridge" mkdir docker\bridge
    copy "src\bridge\requirements.txt" "docker\bridge\"
    copy "src\bridge\nexus_bridge.py" "docker\bridge\"

    docker-compose up --build
    goto :end
)

if "%1"=="prod" (
    echo %BLUE%[INFO]%NC% Starting production environment...
    if not exist ".env" (
        echo %RED%[ERROR]%NC% Production requires .env file. Please create it from .env.example
        pause
        exit /b 1
    )

    REM Prepare bridge files
    if not exist "docker\bridge" mkdir docker\bridge
    copy "src\bridge\requirements.txt" "docker\bridge\"
    copy "src\bridge\nexus_bridge.py" "docker\bridge\"

    docker-compose -f docker-compose.prod.yml up --build -d
    echo %GREEN%[SUCCESS]%NC% Production environment started!
    echo %BLUE%[INFO]%NC% Access your application at: http://localhost
    goto :end
)

if "%1"=="stop" (
    echo %BLUE%[INFO]%NC% Stopping all services...
    docker-compose down
    docker-compose -f docker-compose.prod.yml down 2>nul
    echo %GREEN%[SUCCESS]%NC% All services stopped.
    goto :end
)

if "%1"=="logs" (
    if "%2"=="" (
        docker-compose logs -f
    ) else (
        docker-compose logs -f %2
    )
    goto :end
)

if "%1"=="health" (
    echo %BLUE%[INFO]%NC% Checking service health...
    docker ps --format "table {{.Names}}\t{{.Status}}"
    goto :end
)

if "%1"=="clean" (
    echo %YELLOW%[WARNING]%NC% This will remove all containers, images, and volumes. Are you sure? (y/N)
    set /p response=
    if /i "%response%"=="y" (
        echo %BLUE%[INFO]%NC% Cleaning up Docker resources...
        docker-compose down -v --rmi all
        docker-compose -f docker-compose.prod.yml down -v --rmi all 2>nul
        docker system prune -f
        echo %GREEN%[SUCCESS]%NC% Cleanup complete.
    ) else (
        echo %BLUE%[INFO]%NC% Cleanup cancelled.
    )
    goto :end
)

if "%1"=="backup" (
    echo %BLUE%[INFO]%NC% Creating backup...
    for /f "tokens=2 delims==" %%a in ('wmic OS Get localdatetime /value') do set "dt=%%a"
    set "timestamp=%dt:~0,8%_%dt:~8,6%"
    set "backup_dir=backups\backup_%timestamp%"

    if not exist "backups" mkdir backups
    mkdir "%backup_dir%"

    docker-compose exec postgres pg_dump -U nexus nexus > "%backup_dir%\postgres_backup.sql"
    echo %GREEN%[SUCCESS]%NC% Backup created in %backup_dir%
    goto :end
)

REM Default help
echo.
echo Nexus Docker Management Script (Windows)
echo.
echo Usage: %0 [command]
echo.
echo Commands:
echo   dev        Start development environment
echo   prod       Start production environment
echo   stop       Stop all services
echo   logs       Show logs (optionally specify service name)
echo   health     Check service health
echo   backup     Create backup of data
echo   clean      Clean up Docker resources
echo   help       Show this help
echo.
echo Examples:
echo   %0 dev                    # Start development environment
echo   %0 logs nexus-bridge      # Show bridge service logs
echo   %0 stop                   # Stop all services

:end
pause
