# Tests

# Simple test to verify library builds correctly
add_executable(test_basic test_basic.cpp)
target_link_libraries(test_basic PRIVATE world_engine)
target_include_directories(test_basic PRIVATE ${CMAKE_SOURCE_DIR}/include)

add_test(NAME BasicTest COMMAND test_basic)
