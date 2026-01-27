cmake_minimum_required(VERSION 3.15)
project(WorldEngine VERSION 1.0.0 LANGUAGES CXX C)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Set C standard
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Export compile commands for IDEs
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Options
option(BUILD_SHARED_LIBS "Build shared libraries" ON)
option(BUILD_PYTHON_BINDINGS "Build Python bindings" ON)
option(BUILD_TYPESCRIPT "Build TypeScript components" ON)
option(BUILD_EXAMPLES "Build example programs" ON)
option(BUILD_TESTS "Build tests" ON)
option(INSTALL_WEB_ASSETS "Install web/HTML assets" ON)

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/include)

# Output directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# Find required packages
find_package(Python3 COMPONENTS Interpreter Development)
find_program(NODE_EXECUTABLE NAMES node nodejs)
find_program(NPM_EXECUTABLE NAMES npm)
find_program(TSC_EXECUTABLE NAMES tsc)

# Display configuration
message(STATUS "=== World Engine Build Configuration ===")
message(STATUS "CMake version: ${CMAKE_VERSION}")
message(STATUS "C++ Compiler: ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Build shared libs: ${BUILD_SHARED_LIBS}")
message(STATUS "Build Python bindings: ${BUILD_PYTHON_BINDINGS}")
message(STATUS "Build TypeScript: ${BUILD_TYPESCRIPT}")
message(STATUS "Build examples: ${BUILD_EXAMPLES}")
message(STATUS "Build tests: ${BUILD_TESTS}")

if(Python3_FOUND)
    message(STATUS "Python version: ${Python3_VERSION}")
    message(STATUS "Python executable: ${Python3_EXECUTABLE}")
endif()

if(NODE_EXECUTABLE)
    message(STATUS "Node.js found: ${NODE_EXECUTABLE}")
endif()

if(NPM_EXECUTABLE)
    message(STATUS "npm found: ${NPM_EXECUTABLE}")
endif()

message(STATUS "=======================================")

# Add subdirectories
add_subdirectory(src/cpp)

if(BUILD_PYTHON_BINDINGS AND Python3_FOUND)
    add_subdirectory(src/python)
endif()

if(BUILD_TYPESCRIPT AND NODE_EXECUTABLE AND NPM_EXECUTABLE)
    add_subdirectory(src/typescript)
endif()

if(INSTALL_WEB_ASSETS)
    add_subdirectory(src/html)
endif()

if(BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# Install headers
install(DIRECTORY include/world_engine
        DESTINATION include
        FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp")

# Create and install CMake config files
include(CMakePackageConfigHelpers)

configure_package_config_file(
    ${CMAKE_SOURCE_DIR}/cmake/WorldEngineConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/WorldEngineConfig.cmake
    INSTALL_DESTINATION lib/cmake/WorldEngine
)

write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/WorldEngineConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/WorldEngineConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/WorldEngineConfigVersion.cmake
    DESTINATION lib/cmake/WorldEngine
)

# Print summary
message(STATUS "")
message(STATUS "World Engine configuration complete!")
message(STATUS "Run 'cmake --build .' to build the project")
message(STATUS "")
