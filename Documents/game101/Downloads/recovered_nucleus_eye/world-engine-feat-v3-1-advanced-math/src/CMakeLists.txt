cmake_minimum_required(VERSION 3.20)
project(AutomationNucleus VERSION 1.0.0 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Compiler - specific options
if (MSVC)
    add_compile_options(/W4)
else ()
add_compile_options("-Wall" "-Wextra" "-Wpedantic")
endif()

# Build type
if (NOT CMAKE_BUILD_TYPE)
set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type")
endif()

# Include directories are now set per - target below

# Find packages
find_package(Threads REQUIRED)

# Create the automation nucleus library
add_library(AutomationNucleusLib
    ${ CMAKE_SOURCE_DIR } / AutomationNucleus.cpp
    ${ CMAKE_SOURCE_DIR } / CallHandler.cpp
    ${ CMAKE_SOURCE_DIR } / NovaOmega.cpp
)
target_include_directories(AutomationNucleusLib PUBLIC
    ${ PROJECT_SOURCE_DIR } / include
    ${ CMAKE_SOURCE_DIR } /../ include
    ${ CMAKE_SOURCE_DIR }
)

target_link_libraries(AutomationNucleusLib
    Threads:: Threads
)

# Create the main executable
add_executable(AutomationNucleus
    AutomationNucleusMain.cpp
)

target_link_libraries(AutomationNucleus
    AutomationNucleusLib
    Threads:: Threads
)

# Installation
install(TARGETS AutomationNucleus AutomationNucleusLib
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib)
# Testing
enable_testing()

# Install headers for UDR Nova integration
install(FILES
        CallHandler.hpp
        NovaOmega.hpp
        DESTINATION include)

# Add UDR Nova test executable
if (EXISTS "${CMAKE_SOURCE_DIR}/tests")
add_executable(UDRNovaTest
        tests / udr_nova_test.cpp
)

target_link_libraries(UDRNovaTest
        AutomationNucleusLib
        Threads:: Threads
)
endif()

# Build instructions message
message(STATUS "AutomationNucleus build configured")
message(STATUS "  Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "")
message(STATUS "Build commands:")
message(STATUS "  mkdir build && cd build")
message(STATUS "  cmake ..")
message(STATUS "  cmake --build .")
message(STATUS "  ./AutomationNucleus")
message(STATUS "")
message(STATUS "Available targets:")
message(STATUS "  AutomationNucleus     - Main application with UDR Nova Omega call handling")
message(STATUS "  AutomationNucleusLib  - Core library with CallHandler and NovaOmega")
if (EXISTS "${CMAKE_SOURCE_DIR}/tests")
message(STATUS "  UDRNovaTest          - UDR Nova testing utility")
endif()
