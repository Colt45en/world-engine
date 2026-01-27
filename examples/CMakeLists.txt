# Examples directory

# C++ Example
add_executable(cpp_example cpp_example.cpp)
target_link_libraries(cpp_example PRIVATE world_engine)
target_include_directories(cpp_example PRIVATE ${CMAKE_SOURCE_DIR}/include)

install(TARGETS cpp_example
        RUNTIME DESTINATION bin/examples)
