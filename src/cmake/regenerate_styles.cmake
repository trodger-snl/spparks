# Script to regenerate style headers (run via cmake -P)
#
# In script mode CMAKE_CURRENT_BINARY_DIR collapses to the working directory, so the
# caller passes the real dirs as SPK_SOURCE_DIR / SPK_BINARY_DIR. Fall back to the
# CMAKE_CURRENT_* values for in-tree (source==build) invocations.
if(NOT DEFINED SPK_SOURCE_DIR)
    set(SPK_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
endif()
if(NOT DEFINED SPK_BINARY_DIR)
    set(SPK_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}")
endif()

function(generate_style_headers)
    # Get all header files in the source directory
    file(GLOB ALL_HEADERS "${SPK_SOURCE_DIR}/*.h")
    
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

# Include the style generation functions from the main file
include("${SPK_SOURCE_DIR}/cmake/ConditionalSources.cmake")

# Copy the generate_single_style_header function (simplified)
function(generate_single_style_header CLASS_PATTERN FILE_PREFIX STYLE_NAME DEPENDENCY_FILE)
    set(STYLE_HEADER "${SPK_SOURCE_DIR}/style_${STYLE_NAME}.h")
    set(STYLE_TEMP "${SPK_BINARY_DIR}/style_${STYLE_NAME}.tmp")

    # Load feature status first
    set(FEATURE_STATUS_FILE "${SPK_BINARY_DIR}/feature_status.cmake")
    if(EXISTS "${FEATURE_STATUS_FILE}")
        include("${FEATURE_STATUS_FILE}")
    else()
        set(SPPARKS_HAS_HDF5 FALSE)
        set(SPPARKS_HAS_HIGHFIVE FALSE)
        set(SPPARKS_HAS_MPI FALSE)
        set(SPPARKS_HAS_JPEG FALSE)
        set(SPPARKS_HAS_PNG FALSE)
        set(SPPARKS_HAS_HDF5_FOUND FALSE)
    endif()
    
    # Find matching header files
    if(FILE_PREFIX STREQUAL "")
        set(PATTERN_SEARCH "${CLASS_PATTERN}")
    else()
        set(PATTERN_SEARCH "${FILE_PREFIX}")
    endif()
    
    file(GLOB MATCHING_HEADERS "${SPK_SOURCE_DIR}/${PATTERN_SEARCH}*.h")
    
    # Filter headers
    set(VALID_HEADERS "")
    foreach(HEADER ${MATCHING_HEADERS})
        file(READ "${HEADER}" HEADER_CONTENT)
        if(HEADER_CONTENT MATCHES "${CLASS_PATTERN}")
            get_filename_component(HEADER_NAME "${HEADER}" NAME)
            set(INCLUDE_HEADER TRUE)
            
            should_include_header("${HEADER_NAME}" INCLUDE_HEADER)
            
            if(INCLUDE_HEADER)
                list(APPEND VALID_HEADERS "${HEADER}")
            endif()
        endif()
    endforeach()
    
    # Generate the temporary file
    file(WRITE "${STYLE_TEMP}" "")
    foreach(HEADER ${VALID_HEADERS})
        get_filename_component(HEADER_NAME "${HEADER}" NAME)
        file(APPEND "${STYLE_TEMP}" "#include \"${HEADER_NAME}\"\n")
    endforeach()
    
    # Update if needed
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
        # COPY+RENAME within the destination dir to support cross-filesystem builds
        # (STYLE_TEMP may live on a different filesystem than the source tree).
        file(COPY "${STYLE_TEMP}" DESTINATION "${SPK_SOURCE_DIR}")
        get_filename_component(_style_tmp_name "${STYLE_TEMP}" NAME)
        file(RENAME "${SPK_SOURCE_DIR}/${_style_tmp_name}" "${STYLE_HEADER}")
        file(REMOVE "${STYLE_TEMP}")
    elseif(EXISTS "${STYLE_TEMP}")
        file(REMOVE "${STYLE_TEMP}")
    endif()
endfunction()

# Execute the generation
generate_style_headers()
