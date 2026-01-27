# HTML/Web assets installation

# Install HTML files
install(FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/index.html
    ${CMAKE_CURRENT_SOURCE_DIR}/demo.html
    DESTINATION share/world_engine/html
)

# Install CSS files
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/css/
        DESTINATION share/world_engine/html/css
        FILES_MATCHING PATTERN "*.css")

# Install JavaScript files
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/js/
        DESTINATION share/world_engine/html/js
        FILES_MATCHING PATTERN "*.js")

message(STATUS "HTML/Web assets will be installed to share/world_engine/html")
