cmake_minimum_required(VERSION 3.18)
project(qtp LANGUAGES CXX)

# Require C++17 (pybind11 works well here)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(qtp
    cpp/qtp_module.cpp
    cpp/quantum_thought_pipeline.hpp
)

# If you need extra compile options:
# target_compile_options(qtp PRIVATE -Wall -Wextra -Wpedantic)
