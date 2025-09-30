# Comprehensive Engine Suite Build System
# =====================================
#
# This CMake configuration builds the complete engine ecosystem:
# • Quantum Protocol Engine(C++ core)
# • Nexus Intelligence Engine(C++ core)
# • Multi - Engine Registry(TypeScript / Node.js bridge)
# • Python daemons and integration layers
# • VS Code extension infrastructure

cmake_minimum_required(VERSION 3.16)
project(ComprehensiveEngineSuite VERSION 1.0.0 LANGUAGES CXX)

# C++ Standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Build type
if (NOT CMAKE_BUILD_TYPE)
set(CMAKE_BUILD_TYPE Release)
endif()

# Compiler flags
set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

# Platform - specific configurations
if (WIN32)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DWIN32_LEAN_AND_MEAN")
endif()

if (APPLE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mmacosx-version-min=10.15")
endif()

# Find required packages
find_package(Threads REQUIRED)

# Optional packages for enhanced functionality
find_package(PkgConfig)
if (PkgConfig_FOUND)
        pkg_check_modules(FFTW3 fftw3)
pkg_check_modules(GSL gsl)
endif()

# Include directories
include_directories(${ CMAKE_CURRENT_SOURCE_DIR })
include_directories(${ CMAKE_CURRENT_SOURCE_DIR } / engines)

# Define engine sources
set(QUANTUM_PROTOCOL_SOURCES
    engines / QuantumProtocolEngine.cpp
    engines / QuantumProtocolEngine.hpp
)

set(NEXUS_INTELLIGENCE_SOURCES
    engines / NexusIntelligenceEngine.cpp
    engines / NexusIntelligenceEngine.hpp
)

# Create main engine library
add_library(ComprehensiveEngines SHARED
    ${ QUANTUM_PROTOCOL_SOURCES }
    ${ NEXUS_INTELLIGENCE_SOURCES }
)

# Link libraries
target_link_libraries(ComprehensiveEngines
    Threads:: Threads
)

# Optional enhanced features
if (FFTW3_FOUND)
    target_link_libraries(ComprehensiveEngines ${ FFTW3_LIBRARIES })
target_include_directories(ComprehensiveEngines PRIVATE ${ FFTW3_INCLUDE_DIRS })
target_compile_definitions(ComprehensiveEngines PRIVATE HAVE_FFTW3)
endif()

if (GSL_FOUND)
    target_link_libraries(ComprehensiveEngines ${ GSL_LIBRARIES })
target_include_directories(ComprehensiveEngines PRIVATE ${ GSL_INCLUDE_DIRS })
target_compile_definitions(ComprehensiveEngines PRIVATE HAVE_GSL)
endif()

# Create test executables
add_executable(quantum_protocol_test
    tests / test_quantum_protocol.cpp
    ${ QUANTUM_PROTOCOL_SOURCES }
)
target_link_libraries(quantum_protocol_test Threads:: Threads)

add_executable(nexus_intelligence_test
    tests / test_nexus_intelligence.cpp
    ${ NEXUS_INTELLIGENCE_SOURCES }
)
target_link_libraries(nexus_intelligence_test Threads:: Threads)

# Comprehensive integration test
add_executable(comprehensive_test
    tests / comprehensive_engine_test.cpp
    ${ QUANTUM_PROTOCOL_SOURCES }
    ${ NEXUS_INTELLIGENCE_SOURCES }
)
target_link_libraries(comprehensive_test ComprehensiveEngines)

# Demo executables
add_executable(quantum_unity_demo
    quantum_unity_integration_demo.py # Will be converted to C++ wrapper
)

# Install rules
install(TARGETS ComprehensiveEngines
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)

install(FILES
    engines / QuantumProtocolEngine.hpp
    engines / NexusIntelligenceEngine.hpp
    DESTINATION include / engines
)

# Custom targets for Python integration
add_custom_target(python_daemons
    COMMAND ${ CMAKE_COMMAND } - E echo "Setting up Python daemon infrastructure..."
    COMMAND ${ CMAKE_COMMAND } - E copy ${ CMAKE_CURRENT_SOURCE_DIR } / quantum_protocol_daemon.py ${ CMAKE_BINARY_DIR } /
COMMAND ${ CMAKE_COMMAND } - E copy ${ CMAKE_CURRENT_SOURCE_DIR } / nexus_intelligence_daemon.py ${ CMAKE_BINARY_DIR } /
COMMAND ${ CMAKE_COMMAND } - E copy ${ CMAKE_CURRENT_SOURCE_DIR } / comprehensive_engine_demo.py ${ CMAKE_BINARY_DIR } /
COMMENT "Copying Python daemon files"
)

# VS Code extension build target
add_custom_target(vscode_extension
    COMMAND ${ CMAKE_COMMAND } - E echo "Building VS Code extension..."
    WORKING_DIRECTORY ${ CMAKE_CURRENT_SOURCE_DIR } / extensions
    COMMAND npm install
    COMMAND npm run compile
    COMMENT "Building TypeScript VS Code extension"
)

# Node.js native module(for TypeScript integration)
    if (EXISTS ${ CMAKE_CURRENT_SOURCE_DIR } /extensions/binding.gyp)
add_custom_target(node_addon
        COMMAND node - gyp configure
        COMMAND node - gyp build
        WORKING_DIRECTORY ${ CMAKE_CURRENT_SOURCE_DIR } / extensions
        COMMENT "Building Node.js native addon"
)
add_dependencies(vscode_extension node_addon)
endif()

# Documentation generation
find_program(DOXYGEN_EXECUTABLE doxygen)
if (DOXYGEN_EXECUTABLE)
    add_custom_target(documentation
        COMMAND ${ DOXYGEN_EXECUTABLE } ${ CMAKE_CURRENT_SOURCE_DIR } / Doxyfile
        WORKING_DIRECTORY ${ CMAKE_CURRENT_SOURCE_DIR }
        COMMENT "Generating API documentation with Doxygen"
    )
endif()

# Package configuration
set(CPACK_PACKAGE_NAME "ComprehensiveEngineSuite")
set(CPACK_PACKAGE_VERSION ${ PROJECT_VERSION })
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "Comprehensive Engine Suite for VS Code")
set(CPACK_PACKAGE_VENDOR "Advanced Intelligence Systems")
set(CPACK_RESOURCE_FILE_LICENSE ${ CMAKE_CURRENT_SOURCE_DIR } / LICENSE)
set(CPACK_RESOURCE_FILE_README ${ CMAKE_CURRENT_SOURCE_DIR } / README.md)

if (WIN32)
    set(CPACK_GENERATOR "NSIS")
elseif(APPLE)
set(CPACK_GENERATOR "DragNDrop")
else ()
set(CPACK_GENERATOR "DEB;RPM;TGZ")
endif()

include(CPack)

# Testing configuration
enable_testing()

add_test(NAME quantum_protocol_unit_tests
    COMMAND quantum_protocol_test
)

add_test(NAME nexus_intelligence_unit_tests
    COMMAND nexus_intelligence_test
)

add_test(NAME comprehensive_integration_test
    COMMAND comprehensive_test
)

# Python daemon tests
add_test(NAME python_daemon_integration
    COMMAND python - m pytest ${ CMAKE_CURRENT_SOURCE_DIR } / tests /
WORKING_DIRECTORY ${ CMAKE_BINARY_DIR }
)

# Performance benchmarks
add_custom_target(benchmarks
    COMMAND ${ CMAKE_COMMAND } - E echo "Running performance benchmarks..."
    COMMAND./ comprehensive_test--benchmark
    COMMAND python comprehensive_engine_demo.py--benchmark
    DEPENDS comprehensive_test python_daemons
    COMMENT "Running comprehensive performance benchmarks"
)

# Development targets
add_custom_target(dev_setup
    COMMAND ${ CMAKE_COMMAND } - E echo "Setting up development environment..."
    DEPENDS python_daemons vscode_extension
    COMMENT "Complete development environment setup"
)

add_custom_target(clean_all
    COMMAND ${ CMAKE_COMMAND } - E echo "Cleaning all build artifacts..."
    COMMAND ${ CMAKE_COMMAND } --build. --target clean
    COMMAND ${ CMAKE_COMMAND } - E remove_directory ${ CMAKE_CURRENT_SOURCE_DIR } / extensions / out
    COMMAND ${ CMAKE_COMMAND } - E remove_directory ${ CMAKE_CURRENT_SOURCE_DIR } / extensions / node_modules
    COMMENT "Deep clean of all build artifacts"
)

# Print configuration summary
message(STATUS "========================================")
message(STATUS "Comprehensive Engine Suite Configuration")
message(STATUS "========================================")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ compiler: ${CMAKE_CXX_COMPILER}")
message(STATUS "C++ standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "Install prefix: ${CMAKE_INSTALL_PREFIX}")

if (FFTW3_FOUND)
    message(STATUS "FFTW3 support: ENABLED")
else ()
message(STATUS "FFTW3 support: DISABLED (optional)")
endif()

if (GSL_FOUND)
    message(STATUS "GSL support: ENABLED")
else ()
message(STATUS "GSL support: DISABLED (optional)")
endif()

if (DOXYGEN_EXECUTABLE)
    message(STATUS "Documentation: ENABLED")
else ()
message(STATUS "Documentation: DISABLED (doxygen not found)")
endif()

message(STATUS "========================================")
message(STATUS "Available targets:")
message(STATUS "  make              - Build C++ engines")
message(STATUS "  make python_daemons - Setup Python integration")
message(STATUS "  make vscode_extension - Build VS Code extension")
message(STATUS "  make dev_setup    - Complete development setup")
message(STATUS "  make test         - Run all tests")
message(STATUS "  make benchmarks   - Performance benchmarks")
message(STATUS "  make documentation - Generate API docs")
message(STATUS "  make install      - Install system-wide")
message(STATUS "  make package      - Create distribution package")
message(STATUS "  make clean_all    - Deep clean")
message(STATUS "========================================")

# IDE support
set_property(GLOBAL PROPERTY USE_FOLDERS ON)
set_property(TARGET ComprehensiveEngines PROPERTY FOLDER "Engines")
set_property(TARGET quantum_protocol_test PROPERTY FOLDER "Tests")
set_property(TARGET nexus_intelligence_test PROPERTY FOLDER "Tests")
set_property(TARGET comprehensive_test PROPERTY FOLDER "Tests")

# Export configuration for external projects
export (TARGETS ComprehensiveEngines FILE ComprehensiveEnginesConfig.cmake)
install(EXPORT ComprehensiveEnginesConfig DESTINATION lib / cmake / ComprehensiveEngines)
