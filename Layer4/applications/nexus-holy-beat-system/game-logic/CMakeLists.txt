cmake_minimum_required(VERSION 3.16)
project(NexusGameLogic VERSION 1.0.0 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set build type if not specified
if (NOT CMAKE_BUILD_TYPE)
set(CMAKE_BUILD_TYPE Release)
endif()

# Compiler flags
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -Wall -Wextra")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

# Include directories
include_directories(${ CMAKE_CURRENT_SOURCE_DIR } / include)

# Find required packages
find_package(PkgConfig REQUIRED)

# Optional: Find OpenGL for 3D rendering(if needed)
    find_package(OpenGL QUIET)
if (OpenGL_FOUND)
    add_definitions(-DNEXUS_OPENGL_SUPPORT)
endif()

# Optional: Find FMOD or similar audio library
# find_package(FMOD QUIET)

# Source files
file(GLOB_RECURSE SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.c"
)

file(GLOB_RECURSE HEADERS
    "${CMAKE_CURRENT_SOURCE_DIR}/include/*.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/*.h"
)

# Create the main game logic library
add_library(NexusGameLogic SHARED ${ SOURCES } ${ HEADERS })

# Link libraries
target_link_libraries(NexusGameLogic)

if (OpenGL_FOUND)
    target_link_libraries(NexusGameLogic ${ OPENGL_LIBRARIES })
endif()

# Windows - specific settings
if (WIN32)
    target_compile_definitions(NexusGameLogic PRIVATE
        NOMINMAX
        WIN32_LEAN_AND_MEAN
        NEXUS_WINDOWS
    )

    # Export symbols for DLL
    set_target_properties(NexusGameLogic PROPERTIES
        WINDOWS_EXPORT_ALL_SYMBOLS ON
    )
endif()

# Node.js addon(optional - for JavaScript bindings)
    option(BUILD_NODE_ADDON "Build Node.js addon for JavaScript integration" ON)

if (BUILD_NODE_ADDON)
    # Find Node.js
find_program(NODE_EXECUTABLE NAMES node nodejs)

if (NODE_EXECUTABLE)
        # Get Node.js version and paths
execute_process(
    COMMAND ${ NODE_EXECUTABLE } - p "process.version"
            OUTPUT_VARIABLE NODE_VERSION
            OUTPUT_STRIP_TRAILING_WHITESPACE
)

execute_process(
    COMMAND ${ NODE_EXECUTABLE } - p "require('path').dirname(process.execPath)"
            OUTPUT_VARIABLE NODE_ROOT_DIR
            OUTPUT_STRIP_TRAILING_WHITESPACE
)

        # Node.js addon
add_library(NexusGameAddon SHARED
            ${ CMAKE_CURRENT_SOURCE_DIR } / bindings / nexus_addon.cpp
            ${ SOURCES }
)

target_include_directories(NexusGameAddon PRIVATE
            ${ NODE_ROOT_DIR } / include / node
            ${ CMAKE_CURRENT_SOURCE_DIR } / include
)

        # Node addon properties
set_target_properties(NexusGameAddon PROPERTIES
            PREFIX ""
            SUFFIX ".node"
            OUTPUT_NAME "nexus_game"
)

target_compile_definitions(NexusGameAddon PRIVATE
            BUILDING_NODE_EXTENSION
            NAPI_VERSION = 6
)

if (WIN32)
    target_link_libraries(NexusGameAddon ${ NODE_ROOT_DIR } / node.lib)
else ()
target_link_libraries(NexusGameAddon)
endif()
    else ()
message(WARNING "Node.js not found. Node addon will not be built.")
endif()
endif()

# Examples
option(BUILD_EXAMPLES "Build example programs" ON)

if (BUILD_EXAMPLES)
    file(GLOB EXAMPLE_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/examples/*.cpp")

foreach(example_source ${ EXAMPLE_SOURCES })
get_filename_component(example_name ${ example_source } NAME_WE)
add_executable(${ example_name } ${ example_source })
target_link_libraries(${ example_name } NexusGameLogic)

        # Set output directory
set_target_properties(${ example_name } PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY ${ CMAKE_CURRENT_SOURCE_DIR } / build / examples
)
endforeach()
endif()

# Installation
install(TARGETS NexusGameLogic
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

install(DIRECTORY include /
    DESTINATION include / nexus - game
    FILES_MATCHING PATTERN "*.hpp" PATTERN "*.h"
)

if (BUILD_NODE_ADDON AND NODE_EXECUTABLE)
install(TARGETS NexusGameAddon
        LIBRARY DESTINATION ${ CMAKE_CURRENT_SOURCE_DIR } /../
RUNTIME DESTINATION ${ CMAKE_CURRENT_SOURCE_DIR } /../
    )
endif()

# Print configuration summary
message(STATUS "")
message(STATUS "NEXUS Game Logic Configuration:")
message(STATUS "  Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  OpenGL Support: ${OpenGL_FOUND}")
message(STATUS "  Node.js Addon: ${BUILD_NODE_ADDON}")
message(STATUS "  Examples: ${BUILD_EXAMPLES}")
if (NODE_EXECUTABLE)
    message(STATUS "  Node.js Version: ${NODE_VERSION}")
endif()
message(STATUS "")

# Build instructions
message(STATUS "Build Instructions:")
message(STATUS "  mkdir -p build && cd build")
message(STATUS "  cmake ..")
message(STATUS "  cmake --build . --config ${CMAKE_BUILD_TYPE}")
message(STATUS "")
