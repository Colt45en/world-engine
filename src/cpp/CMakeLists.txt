# C++ Core Engine Library

set(WORLD_ENGINE_SOURCES
    world_engine.cpp
    entity.cpp
    component.cpp
    system.cpp
    scene.cpp
)

set(WORLD_ENGINE_HEADERS
    ${CMAKE_SOURCE_DIR}/include/world_engine/world_engine.h
    ${CMAKE_SOURCE_DIR}/include/world_engine/entity.h
    ${CMAKE_SOURCE_DIR}/include/world_engine/component.h
    ${CMAKE_SOURCE_DIR}/include/world_engine/system.h
    ${CMAKE_SOURCE_DIR}/include/world_engine/scene.h
    ${CMAKE_SOURCE_DIR}/include/world_engine/types.h
)

# Create the library
add_library(world_engine ${WORLD_ENGINE_SOURCES} ${WORLD_ENGINE_HEADERS})

# Set library properties
set_target_properties(world_engine PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
    PUBLIC_HEADER "${WORLD_ENGINE_HEADERS}"
)

# Include directories for the library
target_include_directories(world_engine
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)

# Compiler options
if(MSVC)
    target_compile_options(world_engine PRIVATE /W4)
else()
    target_compile_options(world_engine PRIVATE -Wall -Wextra -Wpedantic)
endif()

# Install targets
install(TARGETS world_engine
    EXPORT WorldEngineTargets
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
    PUBLIC_HEADER DESTINATION include/world_engine
)

# Export targets
install(EXPORT WorldEngineTargets
    FILE WorldEngineTargets.cmake
    NAMESPACE WorldEngine::
    DESTINATION lib/cmake/WorldEngine
)
