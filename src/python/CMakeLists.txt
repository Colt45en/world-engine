# Python bindings for World Engine

if(NOT Python3_FOUND)
    message(WARNING "Python3 not found, skipping Python bindings")
    return()
endif()

# Create Python module
add_custom_target(python_bindings ALL
    COMMAND ${CMAKE_COMMAND} -E echo "Building Python bindings..."
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/setup.py build_ext --build-lib ${CMAKE_BINARY_DIR}/python
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    COMMENT "Building Python bindings"
    DEPENDS world_engine
)

# Install Python module
install(CODE "
    execute_process(
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/setup.py install --prefix=\${CMAKE_INSTALL_PREFIX}
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    )
")
