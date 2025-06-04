# CMake function to generate style_*.h files
# Equivalent to the Make.sh style functionality

function(generate_style_headers)
    # Get all header files in the source directory
    file(GLOB ALL_HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/*.h")
    
    # Define style generation rules as a list
    set(STYLE_RULES_LIST
        "APP_CLASS|app_|app|input"
        "COMMAND_CLASS||command|input"
        "DIAG_CLASS|diag_|diag|input"
        "DUMP_CLASS|dump_|dump|output"
        "PAIR_CLASS|pair_|pair|potential"
        "REGION_CLASS|region_|region|domain"
        "SOLVE_CLASS|solve_|solve|input"
    )
    
    foreach(RULE ${STYLE_RULES_LIST})
        string(REPLACE "|" ";" RULE_PARTS "${RULE}")
        
        list(LENGTH RULE_PARTS NUM_PARTS)
        if(NUM_PARTS EQUAL 4)
            list(GET RULE_PARTS 0 CLASS_PATTERN)
            list(GET RULE_PARTS 1 FILE_PREFIX)
            list(GET RULE_PARTS 2 STYLE_NAME)
            list(GET RULE_PARTS 3 DEPENDENCY_FILE)
            
            # Generate the style header
            generate_single_style_header("${CLASS_PATTERN}" "${FILE_PREFIX}" "${STYLE_NAME}" "${DEPENDENCY_FILE}")
        else()
            message(WARNING "Invalid style rule: ${RULE}")
        endif()
    endforeach()
endfunction()

function(generate_single_style_header CLASS_PATTERN FILE_PREFIX STYLE_NAME DEPENDENCY_FILE)
    set(STYLE_HEADER "${CMAKE_CURRENT_SOURCE_DIR}/style_${STYLE_NAME}.h")
    set(STYLE_TEMP "${CMAKE_CURRENT_BINARY_DIR}/style_${STYLE_NAME}.tmp")
    
    # Find matching header files
    if(FILE_PREFIX STREQUAL "")
        # For COMMAND_CLASS, look for any header that defines the pattern
        set(PATTERN_SEARCH "${CLASS_PATTERN}")
    else()
        # For others, look for headers starting with the prefix
        set(PATTERN_SEARCH "${FILE_PREFIX}")
    endif()
    
    file(GLOB MATCHING_HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/${PATTERN_SEARCH}*.h")
    
    # Filter headers that actually contain the class pattern and are feature-enabled
    set(VALID_HEADERS "")
    foreach(HEADER ${MATCHING_HEADERS})
        file(READ "${HEADER}" HEADER_CONTENT)
        if(HEADER_CONTENT MATCHES "${CLASS_PATTERN}")
            # Check if this header requires features that are disabled
            get_filename_component(HEADER_NAME "${HEADER}" NAME)
            set(INCLUDE_HEADER TRUE)
            
            # Use conditional source checking
            include(cmake/ConditionalSources.cmake)
            should_include_header("${HEADER_NAME}" INCLUDE_HEADER)
            if(NOT INCLUDE_HEADER)
                message(STATUS "Skipping ${HEADER_NAME} (missing dependencies)")
            endif()
            
            if(INCLUDE_HEADER)
                list(APPEND VALID_HEADERS "${HEADER}")
            else()
                message(STATUS "Skipping ${HEADER_NAME} (missing dependencies)")
            endif()
        endif()
    endforeach()
    
    # Generate the temporary file
    file(WRITE "${STYLE_TEMP}" "")
    foreach(HEADER ${VALID_HEADERS})
        get_filename_component(HEADER_NAME "${HEADER}" NAME)
        file(APPEND "${STYLE_TEMP}" "#include \"${HEADER_NAME}\"\n")
    endforeach()
    
    # Check if we need to update the actual style header
    set(UPDATE_HEADER FALSE)
    
    if(NOT EXISTS "${STYLE_HEADER}")
        set(UPDATE_HEADER TRUE)
    elseif(EXISTS "${STYLE_TEMP}")
        file(READ "${STYLE_HEADER}" CURRENT_CONTENT)
        file(READ "${STYLE_TEMP}" NEW_CONTENT)
        if(NOT "${CURRENT_CONTENT}" STREQUAL "${NEW_CONTENT}")
            set(UPDATE_HEADER TRUE)
        endif()
    endif()
    
    if(UPDATE_HEADER AND EXISTS "${STYLE_TEMP}")
        file(RENAME "${STYLE_TEMP}" "${STYLE_HEADER}")
        message(STATUS "Generated ${STYLE_HEADER}")
        
        # Remove dependency files (equivalent to rm -f Obj_*/$4.d)
        file(GLOB DEPENDENCY_FILES "${CMAKE_CURRENT_BINARY_DIR}/*/${DEPENDENCY_FILE}.d")
        foreach(DEP_FILE ${DEPENDENCY_FILES})
            file(REMOVE "${DEP_FILE}")
        endforeach()
        
        # Remove spparks.d files as well
        file(GLOB SPPARKS_DEP_FILES "${CMAKE_CURRENT_BINARY_DIR}/*/spparks.d")
        foreach(DEP_FILE ${SPPARKS_DEP_FILES})
            file(REMOVE "${DEP_FILE}")
        endforeach()
    else()
        # Remove temp file if no update needed
        if(EXISTS "${STYLE_TEMP}")
            file(REMOVE "${STYLE_TEMP}")
        endif()
        
        # Create empty file if no headers found
        if(NOT VALID_HEADERS AND NOT EXISTS "${STYLE_HEADER}")
            file(WRITE "${STYLE_HEADER}" "")
            message(STATUS "Created empty ${STYLE_HEADER}")
        endif()
    endif()
endfunction()

# Custom target to regenerate style headers when source files change
add_custom_target(generate_styles
    COMMAND ${CMAKE_COMMAND} -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/regenerate_styles.cmake"
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    COMMENT "Regenerating style headers..."
)

# Create a script to regenerate styles
file(WRITE "${CMAKE_CURRENT_SOURCE_DIR}/cmake/regenerate_styles.cmake"
"# Script to regenerate style headers
include(\"${CMAKE_CURRENT_SOURCE_DIR}/cmake/generate_styles.cmake\")
generate_style_headers()
")

# Add dependencies for style headers
file(GLOB ALL_APP_HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/app_*.h")
file(GLOB ALL_DIAG_HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/diag_*.h")
file(GLOB ALL_DUMP_HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/dump_*.h")
file(GLOB ALL_PAIR_HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/pair_*.h")
file(GLOB ALL_REGION_HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/region_*.h")
file(GLOB ALL_SOLVE_HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/solve_*.h")

# Style headers depend on their respective source headers
add_custom_command(
    OUTPUT "${CMAKE_CURRENT_SOURCE_DIR}/style_app.h"
    DEPENDS ${ALL_APP_HEADERS}
    COMMAND ${CMAKE_COMMAND} -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/regenerate_styles.cmake"
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    COMMENT "Regenerating style_app.h"
)

add_custom_command(
    OUTPUT "${CMAKE_CURRENT_SOURCE_DIR}/style_diag.h"
    DEPENDS ${ALL_DIAG_HEADERS}
    COMMAND ${CMAKE_COMMAND} -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/regenerate_styles.cmake"
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    COMMENT "Regenerating style_diag.h"
)

add_custom_command(
    OUTPUT "${CMAKE_CURRENT_SOURCE_DIR}/style_dump.h"
    DEPENDS ${ALL_DUMP_HEADERS}
    COMMAND ${CMAKE_COMMAND} -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/regenerate_styles.cmake"
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    COMMENT "Regenerating style_dump.h"
)

add_custom_command(
    OUTPUT "${CMAKE_CURRENT_SOURCE_DIR}/style_pair.h"
    DEPENDS ${ALL_PAIR_HEADERS}
    COMMAND ${CMAKE_COMMAND} -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/regenerate_styles.cmake"
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    COMMENT "Regenerating style_pair.h"
)

add_custom_command(
    OUTPUT "${CMAKE_CURRENT_SOURCE_DIR}/style_region.h"
    DEPENDS ${ALL_REGION_HEADERS}
    COMMAND ${CMAKE_COMMAND} -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/regenerate_styles.cmake"
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    COMMENT "Regenerating style_region.h"
)

add_custom_command(
    OUTPUT "${CMAKE_CURRENT_SOURCE_DIR}/style_solve.h"
    DEPENDS ${ALL_SOLVE_HEADERS}
    COMMAND ${CMAKE_COMMAND} -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/regenerate_styles.cmake"
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    COMMENT "Regenerating style_solve.h"
)

# Add style headers as generated sources
set(STYLE_HEADERS
    "${CMAKE_CURRENT_SOURCE_DIR}/style_app.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/style_diag.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/style_dump.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/style_pair.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/style_region.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/style_solve.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/style_command.h"
)

# Ensure style headers are generated before building targets
add_custom_target(style_headers DEPENDS ${STYLE_HEADERS})