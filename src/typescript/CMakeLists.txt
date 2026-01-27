# TypeScript/JavaScript build

if(NOT NODE_EXECUTABLE OR NOT NPM_EXECUTABLE)
    message(WARNING "Node.js or npm not found, skipping TypeScript build")
    return()
endif()

# Install dependencies
add_custom_target(npm_install
    COMMAND ${NPM_EXECUTABLE} install
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMENT "Installing npm dependencies"
)

# Build TypeScript
add_custom_target(typescript_build ALL
    COMMAND ${NPM_EXECUTABLE} run build
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMENT "Building TypeScript"
    DEPENDS npm_install
)

# Install TypeScript output
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/dist/
        DESTINATION share/world_engine/js
        PATTERN "*.js"
        PATTERN "*.d.ts"
        PATTERN "*.map")
