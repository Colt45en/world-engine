#!/bin/bash

# NEXUS Holy Beat System - Complete Build & Deployment Script
# This script sets up the entire NEXUS Holy Beat System environment

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
NEXUS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${NEXUS_ROOT}/build"
GAME_LOGIC_DIR="${NEXUS_ROOT}/game-logic"
WEB_PORT=8080
NODEJS_PORT=3000

echo -e "${CYAN}🎵✨ NEXUS Holy Beat System - Complete Deployment ✨🎵${NC}"
echo -e "${CYAN}============================================================${NC}"
echo -e "${BLUE}Root Directory: ${NEXUS_ROOT}${NC}"
echo -e "${BLUE}Build Directory: ${BUILD_DIR}${NC}"
echo ""

# Function to print section headers
print_section() {
    echo -e "\n${PURPLE}=== $1 ===${NC}"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to install Node.js dependencies if needed
setup_nodejs() {
    if [ -f "${NEXUS_ROOT}/package.json" ]; then
        echo -e "${YELLOW}📦 Installing Node.js dependencies...${NC}"
        cd "${NEXUS_ROOT}"

        if command_exists npm; then
            npm install
        elif command_exists yarn; then
            yarn install
        else
            echo -e "${RED}❌ Neither npm nor yarn found. Please install Node.js first.${NC}"
            return 1
        fi
        echo -e "${GREEN}✅ Node.js dependencies installed${NC}"
    else
        echo -e "${YELLOW}⚠️ No package.json found, skipping Node.js setup${NC}"
    fi
}

# Function to build C++ components
build_cpp() {
    print_section "Building C++ Game Engine"

    # Check if CMakeLists.txt exists
    if [ ! -f "${GAME_LOGIC_DIR}/CMakeLists.txt" ]; then
        if [ -f "${GAME_LOGIC_DIR}/CMakeLists-new.txt" ]; then
            echo -e "${YELLOW}🔄 Using new CMakeLists.txt...${NC}"
            cp "${GAME_LOGIC_DIR}/CMakeLists-new.txt" "${GAME_LOGIC_DIR}/CMakeLists.txt"
        else
            echo -e "${RED}❌ No CMakeLists.txt found in game-logic directory${NC}"
            return 1
        fi
    fi

    cd "${GAME_LOGIC_DIR}"

    # Create build directory
    mkdir -p "${BUILD_DIR}"
    cd "${BUILD_DIR}"

    echo -e "${YELLOW}🔧 Configuring CMake...${NC}"

    # Platform-specific CMake configuration
    if [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "cygwin" ]] || [[ "$OSTYPE" == "win32" ]]; then
        # Windows
        cmake .. -G "Visual Studio 16 2019" -A x64 -DCMAKE_BUILD_TYPE=Release
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        cmake .. -G "Unix Makefiles" -DCMAKE_BUILD_TYPE=Release
    else
        # Linux
        cmake .. -G "Unix Makefiles" -DCMAKE_BUILD_TYPE=Release
    fi

    echo -e "${YELLOW}🔨 Building NEXUS Core Engine...${NC}"
    cmake --build . --config Release --parallel

    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✅ C++ build completed successfully${NC}"

        # List built executables
        echo -e "${BLUE}📋 Built executables:${NC}"
        for exe in nexus_resource_demo holy_beat_demo nexus_test_suite; do
            if [ -f "./${exe}" ] || [ -f "./${exe}.exe" ]; then
                echo -e "${GREEN}  ✅ ${exe}${NC}"
            else
                echo -e "${YELLOW}  ⚠️ ${exe} (not found)${NC}"
            fi
        done
    else
        echo -e "${RED}❌ C++ build failed${NC}"
        return 1
    fi
}

# Function to setup web components
setup_web() {
    print_section "Setting Up Web Components"

    cd "${NEXUS_ROOT}"

    # Check for web files
    web_files=(
        "nexus-live-bridge.html"
        "nexus-3d-visualization.html"
        "system-overview.html"
        "beat-room.html"
    )

    echo -e "${YELLOW}🌐 Checking web components...${NC}"
    for file in "${web_files[@]}"; do
        if [ -f "${file}" ]; then
            echo -e "${GREEN}  ✅ ${file}${NC}"
        else
            echo -e "${YELLOW}  ⚠️ ${file} (not found)${NC}"
        fi
    done

    # Copy web files to build directory for easy access
    mkdir -p "${BUILD_DIR}/web"
    for file in "${web_files[@]}"; do
        if [ -f "${file}" ]; then
            cp "${file}" "${BUILD_DIR}/web/"
        fi
    done

    echo -e "${GREEN}✅ Web components prepared${NC}"
}

# Function to start the Node.js server
start_nodejs_server() {
    if [ -f "${NEXUS_ROOT}/server.js" ]; then
        print_section "Starting Node.js Server"

        cd "${NEXUS_ROOT}"
        echo -e "${YELLOW}🚀 Starting Node.js server on port ${NODEJS_PORT}...${NC}"

        # Start server in background
        node server.js > nodejs_server.log 2>&1 &
        NODEJS_PID=$!
        echo $NODEJS_PID > nodejs_server.pid

        # Wait a moment and check if it's running
        sleep 2
        if kill -0 $NODEJS_PID 2>/dev/null; then
            echo -e "${GREEN}✅ Node.js server started (PID: ${NODEJS_PID})${NC}"
            echo -e "${BLUE}🌐 Server URL: http://localhost:${NODEJS_PORT}${NC}"
        else
            echo -e "${RED}❌ Failed to start Node.js server${NC}"
        fi
    else
        echo -e "${YELLOW}⚠️ No server.js found, skipping Node.js server${NC}"
    fi
}

# Function to run tests
run_tests() {
    print_section "Running Tests"

    cd "${BUILD_DIR}"

    if [ -f "./nexus_test_suite" ] || [ -f "./nexus_test_suite.exe" ]; then
        echo -e "${YELLOW}🧪 Running NEXUS test suite...${NC}"
        ./nexus_test_suite --basic || ./nexus_test_suite.exe --basic

        if [ $? -eq 0 ]; then
            echo -e "${GREEN}✅ All tests passed${NC}"
        else
            echo -e "${RED}❌ Some tests failed${NC}"
        fi
    else
        echo -e "${YELLOW}⚠️ Test suite not found, skipping tests${NC}"
    fi
}

# Function to create launch script
create_launch_script() {
    print_section "Creating Launch Script"

    cat > "${BUILD_DIR}/launch_holy_beat_system.sh" << 'EOF'
#!/bin/bash

# NEXUS Holy Beat System - Launch Script

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "🎵✨ Starting NEXUS Holy Beat System ✨🎵"
echo ""
echo "🌐 Web Interfaces:"
echo "   • nexus-live-bridge.html - Real-time dashboard"
echo "   • nexus-3d-visualization.html - 3D sacred geometry"
echo ""
echo "🎮 Starting C++ engine in 3 seconds..."
echo "   Press Ctrl+C to stop the demo"
echo ""

sleep 3

# Start the holy beat demo
if [ -f "./holy_beat_demo" ]; then
    ./holy_beat_demo
elif [ -f "./holy_beat_demo.exe" ]; then
    ./holy_beat_demo.exe
elif [ -f "./nexus_resource_demo" ]; then
    echo "⚠️ Holy Beat demo not found, running resource demo instead"
    ./nexus_resource_demo
elif [ -f "./nexus_resource_demo.exe" ]; then
    echo "⚠️ Holy Beat demo not found, running resource demo instead"
    ./nexus_resource_demo.exe
else
    echo "❌ No demo executables found"
    exit 1
fi
EOF

    chmod +x "${BUILD_DIR}/launch_holy_beat_system.sh"

    # Create Windows batch file too
    cat > "${BUILD_DIR}/launch_holy_beat_system.bat" << 'EOF'
@echo off
echo 🎵✨ Starting NEXUS Holy Beat System ✨🎵
echo.
echo 🌐 Web Interfaces:
echo    • nexus-live-bridge.html - Real-time dashboard
echo    • nexus-3d-visualization.html - 3D sacred geometry
echo.
echo 🎮 Starting C++ engine in 3 seconds...
echo    Press Ctrl+C to stop the demo
echo.

timeout /t 3 /nobreak > nul

if exist "holy_beat_demo.exe" (
    holy_beat_demo.exe
) else if exist "nexus_resource_demo.exe" (
    echo ⚠️ Holy Beat demo not found, running resource demo instead
    nexus_resource_demo.exe
) else (
    echo ❌ No demo executables found
    pause
)
EOF

    echo -e "${GREEN}✅ Launch scripts created${NC}"
}

# Function to display final instructions
show_final_instructions() {
    print_section "🎉 Deployment Complete!"

    echo -e "${GREEN}✅ NEXUS Holy Beat System is ready to run!${NC}"
    echo ""
    echo -e "${CYAN}🚀 Quick Start Instructions:${NC}"
    echo -e "${BLUE}  1. Navigate to: ${BUILD_DIR}${NC}"
    echo -e "${BLUE}  2. Run: ./launch_holy_beat_system.sh (or .bat on Windows)${NC}"
    echo -e "${BLUE}  3. Open web interfaces in your browser:${NC}"
    echo -e "${YELLOW}     • ${BUILD_DIR}/web/nexus-live-bridge.html${NC}"
    echo -e "${YELLOW}     • ${BUILD_DIR}/web/nexus-3d-visualization.html${NC}"
    echo ""

    echo -e "${CYAN}🎮 Available Demos:${NC}"
    cd "${BUILD_DIR}"
    for exe in nexus_resource_demo holy_beat_demo nexus_test_suite; do
        if [ -f "./${exe}" ] || [ -f "./${exe}.exe" ]; then
            echo -e "${GREEN}  ✅ ./${exe}${NC}"
        fi
    done
    echo ""

    if [ -f "nodejs_server.pid" ]; then
        echo -e "${CYAN}🌐 Node.js Server:${NC}"
        echo -e "${GREEN}  Running on: http://localhost:${NODEJS_PORT}${NC}"
        echo -e "${BLUE}  Stop with: kill \$(cat nodejs_server.pid)${NC}"
        echo ""
    fi

    echo -e "${PURPLE}💡 Pro Tips:${NC}"
    echo -e "${BLUE}  • The C++ engine will stream data to WebSocket port ${WEB_PORT}${NC}"
    echo -e "${BLUE}  • Use Ctrl+C to gracefully stop any running demo${NC}"
    echo -e "${BLUE}  • Check the generated logs for performance metrics${NC}"
    echo -e "${BLUE}  • Modify examples/ directory to create custom demos${NC}"
    echo ""

    echo -e "${CYAN}🎵✨ Enjoy the NEXUS Holy Beat System! ✨🎵${NC}"
}

# Main deployment flow
main() {
    print_section "System Checks"

    # Check required tools
    echo -e "${YELLOW}🔍 Checking system requirements...${NC}"

    if command_exists cmake; then
        echo -e "${GREEN}  ✅ CMake found${NC}"
    else
        echo -e "${RED}  ❌ CMake not found - please install CMake 3.16+${NC}"
        exit 1
    fi

    if command_exists gcc || command_exists clang || command_exists cl; then
        echo -e "${GREEN}  ✅ C++ compiler found${NC}"
    else
        echo -e "${RED}  ❌ No C++ compiler found - please install GCC, Clang, or MSVC${NC}"
        exit 1
    fi

    if command_exists node; then
        echo -e "${GREEN}  ✅ Node.js found${NC}"
        NODE_AVAILABLE=true
    else
        echo -e "${YELLOW}  ⚠️ Node.js not found - web server features will be limited${NC}"
        NODE_AVAILABLE=false
    fi

    # Run deployment steps
    if [ "$NODE_AVAILABLE" = true ]; then
        setup_nodejs || echo -e "${YELLOW}⚠️ Node.js setup failed, continuing...${NC}"
    fi

    build_cpp || exit 1
    setup_web

    if [ "$NODE_AVAILABLE" = true ]; then
        start_nodejs_server
    fi

    run_tests
    create_launch_script
    show_final_instructions
}

# Handle command line arguments
if [ "$1" = "--help" ] || [ "$1" = "-h" ]; then
    echo "NEXUS Holy Beat System Deployment Script"
    echo ""
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  --help, -h    Show this help message"
    echo "  --clean       Clean build directory before building"
    echo "  --no-tests    Skip running tests"
    echo "  --no-server   Don't start Node.js server"
    echo ""
    exit 0
fi

if [ "$1" = "--clean" ]; then
    print_section "Cleaning Build Directory"
    rm -rf "${BUILD_DIR}"
    echo -e "${GREEN}✅ Build directory cleaned${NC}"
fi

# Run main deployment
main "$@"
