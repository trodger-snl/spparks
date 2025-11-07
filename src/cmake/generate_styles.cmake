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
        # For COMMAND_CLASS, glob all headers and filter by content later
        file(GLOB MATCHING_HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/*.h")
    else()
        # For others, look for headers starting with the prefix
        file(GLOB MATCHING_HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/${FILE_PREFIX}*.h")
    endif()
    
    # Load feature status first to ensure it's available
    set(FEATURE_STATUS_FILE "${CMAKE_CURRENT_BINARY_DIR}/feature_status.cmake")
    if(EXISTS "${FEATURE_STATUS_FILE}")
        include("${FEATURE_STATUS_FILE}")
        message(STATUS "Style generation for ${STYLE_NAME}: Using feature status from ${FEATURE_STATUS_FILE}")
    else()
        message(WARNING "Style generation for ${STYLE_NAME}: Feature status file not found at ${FEATURE_STATUS_FILE}")
        # Set default values to prevent errors
        set(SPPARKS_HAS_HDF5 FALSE)
        set(SPPARKS_HAS_JPEG FALSE)
        set(SPPARKS_HAS_PNG FALSE)
        set(SPPARKS_HAS_HDF5_FOUND FALSE)
    endif()
    
    # Filter headers that actually contain the class pattern and are feature-enabled
    set(VALID_HEADERS "")
    set(SKIPPED_HEADERS "")
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
                message(STATUS "Style generation for ${STYLE_NAME}: Including ${HEADER_NAME}")
            else()
                list(APPEND SKIPPED_HEADERS "${HEADER_NAME}")
                message(STATUS "Style generation for ${STYLE_NAME}: Skipping ${HEADER_NAME} (missing dependencies)")
            endif()
        endif()
    endforeach()
    
    # Validate feature-dependent headers were included when features are enabled
    if(STYLE_NAME STREQUAL "app")
        # Check for HDF5-dependent apps
        if(SPPARKS_HAS_HDF5 AND SPPARKS_HAS_HDF5_FOUND)
            set(HDF5_APPS "app_additive_ext_temp_texture.h")
            foreach(HDF5_APP ${HDF5_APPS})
                set(FOUND_HDF5_APP FALSE)
                foreach(HEADER ${VALID_HEADERS})
                    get_filename_component(HEADER_NAME "${HEADER}" NAME)
                    if(HEADER_NAME STREQUAL HDF5_APP)
                        set(FOUND_HDF5_APP TRUE)
                        break()
                    endif()
                endforeach()
                if(NOT FOUND_HDF5_APP)
                    message(WARNING "Style generation: HDF5 is enabled but ${HDF5_APP} was not included in style_app.h")
                    message(WARNING "This may cause 'App_style specific command before app_style set' errors")
                    message(WARNING "Consider regenerating with: rm ${STYLE_HEADER} && sh Make.sh style")
                endif()
            endforeach()
        endif()
        
        # Log summary of what was included/skipped
        list(LENGTH VALID_HEADERS NUM_VALID)
        list(LENGTH SKIPPED_HEADERS NUM_SKIPPED)
        message(STATUS "Style generation for ${STYLE_NAME}: Included ${NUM_VALID} headers, skipped ${NUM_SKIPPED} headers")
        if(NUM_SKIPPED GREATER 0)
            message(STATUS "Style generation for ${STYLE_NAME}: Skipped headers: ${SKIPPED_HEADERS}")
        endif()
    endif()
    
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
        # Use COPY + REMOVE instead of RENAME to support cross-filesystem moves
        file(COPY "${STYLE_TEMP}" DESTINATION "${CMAKE_CURRENT_SOURCE_DIR}")
        get_filename_component(TEMP_FILENAME "${STYLE_TEMP}" NAME)
        file(RENAME "${CMAKE_CURRENT_SOURCE_DIR}/${TEMP_FILENAME}" "${STYLE_HEADER}")
        file(REMOVE "${STYLE_TEMP}")
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

# NOTE: Dynamic regenerate_styles.cmake generation removed to prevent race conditions
# The regenerate_styles.cmake file is now a static file that handles style generation
if(FALSE) # Disabled
file(WRITE "${CMAKE_CURRENT_SOURCE_DIR}/cmake/regenerate_styles.cmake"
"# Script to regenerate style headers
function(generate_style_headers)
    # Get all header files in the source directory
    file(GLOB ALL_HEADERS \"\${CMAKE_CURRENT_SOURCE_DIR}/*.h\")
    
    # Define style generation rules as a list
    set(STYLE_RULES_LIST
        \"APP_CLASS|app_|app|input\"
        \"COMMAND_CLASS||command|input\"
        \"DIAG_CLASS|diag_|diag|input\"
        \"DUMP_CLASS|dump_|dump|output\"
        \"PAIR_CLASS|pair_|pair|potential\"
        \"REGION_CLASS|region_|region|domain\"
        \"SOLVE_CLASS|solve_|solve|input\"
    )
    
    foreach(RULE \${STYLE_RULES_LIST})
        string(REPLACE \"|\" \";\" RULE_PARTS \"\${RULE}\")
        
        list(LENGTH RULE_PARTS NUM_PARTS)
        if(NUM_PARTS EQUAL 4)
            list(GET RULE_PARTS 0 CLASS_PATTERN)
            list(GET RULE_PARTS 1 FILE_PREFIX)
            list(GET RULE_PARTS 2 STYLE_NAME)
            list(GET RULE_PARTS 3 DEPENDENCY_FILE)
            
            # Generate the style header
            generate_single_style_header(\"\${CLASS_PATTERN}\" \"\${FILE_PREFIX}\" \"\${STYLE_NAME}\" \"\${DEPENDENCY_FILE}\")
        else()
            message(WARNING \"Invalid style rule: \${RULE}\")
        endif()
    endforeach()
endfunction()

# Include the style generation functions from the main file
include(\"\${CMAKE_CURRENT_SOURCE_DIR}/cmake/ConditionalSources.cmake\")

# Copy the generate_single_style_header function (simplified)
function(generate_single_style_header CLASS_PATTERN FILE_PREFIX STYLE_NAME DEPENDENCY_FILE)
    set(STYLE_HEADER \"\${CMAKE_CURRENT_SOURCE_DIR}/style_\${STYLE_NAME}.h\")
    set(STYLE_TEMP \"\${CMAKE_CURRENT_BINARY_DIR}/style_\${STYLE_NAME}.tmp\")
    
    # Load feature status first
    set(FEATURE_STATUS_FILE \"\${CMAKE_CURRENT_BINARY_DIR}/feature_status.cmake\")
    if(EXISTS \"\${FEATURE_STATUS_FILE}\")
        include(\"\${FEATURE_STATUS_FILE}\")
    else()
        set(SPPARKS_HAS_HDF5 FALSE)
        set(SPPARKS_HAS_JPEG FALSE)
        set(SPPARKS_HAS_PNG FALSE)
        set(SPPARKS_HAS_HDF5_FOUND FALSE)
    endif()
    
    # Find matching header files
    if(FILE_PREFIX STREQUAL \"\")
        set(PATTERN_SEARCH \"\${CLASS_PATTERN}\")
    else()
        set(PATTERN_SEARCH \"\${FILE_PREFIX}\")
    endif()
    
    file(GLOB MATCHING_HEADERS \"\${CMAKE_CURRENT_SOURCE_DIR}/\${PATTERN_SEARCH}*.h\")
    
    # Filter headers
    set(VALID_HEADERS \"\")
    foreach(HEADER \${MATCHING_HEADERS})
        file(READ \"\${HEADER}\" HEADER_CONTENT)
        if(HEADER_CONTENT MATCHES \"\${CLASS_PATTERN}\")
            get_filename_component(HEADER_NAME \"\${HEADER}\" NAME)
            set(INCLUDE_HEADER TRUE)
            
            should_include_header(\"\${HEADER_NAME}\" INCLUDE_HEADER)
            
            if(INCLUDE_HEADER)
                list(APPEND VALID_HEADERS \"\${HEADER}\")
            endif()
        endif()
    endforeach()
    
    # Generate the temporary file
    file(WRITE \"\${STYLE_TEMP}\" \"\")
    foreach(HEADER \${VALID_HEADERS})
        get_filename_component(HEADER_NAME \"\${HEADER}\" NAME)
        file(APPEND \"\${STYLE_TEMP}\" \"#include \\\"\${HEADER_NAME}\\\"\\n\")
    endforeach()
    
    # Update if needed
    set(UPDATE_HEADER FALSE)
    if(NOT EXISTS \"\${STYLE_HEADER}\")
        set(UPDATE_HEADER TRUE)
    elseif(EXISTS \"\${STYLE_TEMP}\")
        file(READ \"\${STYLE_HEADER}\" CURRENT_CONTENT)
        file(READ \"\${STYLE_TEMP}\" NEW_CONTENT)
        if(NOT \"\${CURRENT_CONTENT}\" STREQUAL \"\${NEW_CONTENT}\")
            set(UPDATE_HEADER TRUE)
        endif()
    endif()
    
    if(UPDATE_HEADER AND EXISTS \"\${STYLE_TEMP}\")
        file(RENAME \"\${STYLE_TEMP}\" \"\${STYLE_HEADER}\")
    elseif(EXISTS \"\${STYLE_TEMP}\")
        file(REMOVE \"\${STYLE_TEMP}\")
    endif()
endfunction()

# Execute the generation
generate_style_headers()
")
endif() # End disabled section

# Add dependencies for style headers
file(GLOB ALL_APP_HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/app_*.h")
file(GLOB ALL_COMMAND_HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/create_*.h" "${CMAKE_CURRENT_SOURCE_DIR}/read_*.h" "${CMAKE_CURRENT_SOURCE_DIR}/set.h" "${CMAKE_CURRENT_SOURCE_DIR}/shell.h")
file(GLOB ALL_DIAG_HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/diag_*.h")
file(GLOB ALL_DUMP_HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/dump_*.h")
file(GLOB ALL_PAIR_HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/pair_*.h")
file(GLOB ALL_REGION_HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/region_*.h")
file(GLOB ALL_SOLVE_HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/solve_*.h")

# Style headers depend on their respective source headers AND feature status
add_custom_command(
    OUTPUT "${CMAKE_CURRENT_SOURCE_DIR}/style_app.h"
    DEPENDS ${ALL_APP_HEADERS} "${CMAKE_CURRENT_BINARY_DIR}/feature_status.cmake"
    COMMAND ${CMAKE_COMMAND} 
        -DSTYLE_TYPE=app
        -DCLASS_PATTERN=APP_CLASS
        -DFILE_PREFIX=app_
        -DDEPENDENCY_FILE=input
        -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/regenerate_single_style.cmake"
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    COMMENT "Regenerating style_app.h based on features and headers"
)

add_custom_command(
    OUTPUT "${CMAKE_CURRENT_SOURCE_DIR}/style_command.h"
    DEPENDS ${ALL_COMMAND_HEADERS} "${CMAKE_CURRENT_BINARY_DIR}/feature_status.cmake"
    COMMAND ${CMAKE_COMMAND}
        -DSTYLE_TYPE=command
        -DCLASS_PATTERN=COMMAND_CLASS
        -DFILE_PREFIX=
        -DDEPENDENCY_FILE=input
        -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/regenerate_single_style.cmake"
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    COMMENT "Regenerating style_command.h based on features and headers"
)

add_custom_command(
    OUTPUT "${CMAKE_CURRENT_SOURCE_DIR}/style_diag.h"
    DEPENDS ${ALL_DIAG_HEADERS} "${CMAKE_CURRENT_BINARY_DIR}/feature_status.cmake"
    COMMAND ${CMAKE_COMMAND}
        -DSTYLE_TYPE=diag
        -DCLASS_PATTERN=DIAG_CLASS
        -DFILE_PREFIX=diag_
        -DDEPENDENCY_FILE=input
        -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/regenerate_single_style.cmake"
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    COMMENT "Regenerating style_diag.h based on features and headers"
)

add_custom_command(
    OUTPUT "${CMAKE_CURRENT_SOURCE_DIR}/style_dump.h"
    DEPENDS ${ALL_DUMP_HEADERS} "${CMAKE_CURRENT_BINARY_DIR}/feature_status.cmake"
    COMMAND ${CMAKE_COMMAND}
        -DSTYLE_TYPE=dump
        -DCLASS_PATTERN=DUMP_CLASS
        -DFILE_PREFIX=dump_
        -DDEPENDENCY_FILE=output
        -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/regenerate_single_style.cmake"
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    COMMENT "Regenerating style_dump.h based on features and headers"
)

add_custom_command(
    OUTPUT "${CMAKE_CURRENT_SOURCE_DIR}/style_pair.h"
    DEPENDS ${ALL_PAIR_HEADERS} "${CMAKE_CURRENT_BINARY_DIR}/feature_status.cmake"
    COMMAND ${CMAKE_COMMAND}
        -DSTYLE_TYPE=pair
        -DCLASS_PATTERN=PAIR_CLASS
        -DFILE_PREFIX=pair_
        -DDEPENDENCY_FILE=potential
        -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/regenerate_single_style.cmake"
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    COMMENT "Regenerating style_pair.h based on features and headers"
)

add_custom_command(
    OUTPUT "${CMAKE_CURRENT_SOURCE_DIR}/style_region.h"
    DEPENDS ${ALL_REGION_HEADERS} "${CMAKE_CURRENT_BINARY_DIR}/feature_status.cmake"
    COMMAND ${CMAKE_COMMAND}
        -DSTYLE_TYPE=region
        -DCLASS_PATTERN=REGION_CLASS
        -DFILE_PREFIX=region_
        -DDEPENDENCY_FILE=domain
        -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/regenerate_single_style.cmake"
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    COMMENT "Regenerating style_region.h based on features and headers"
)

add_custom_command(
    OUTPUT "${CMAKE_CURRENT_SOURCE_DIR}/style_solve.h"
    DEPENDS ${ALL_SOLVE_HEADERS} "${CMAKE_CURRENT_BINARY_DIR}/feature_status.cmake"
    COMMAND ${CMAKE_COMMAND}
        -DSTYLE_TYPE=solve
        -DCLASS_PATTERN=SOLVE_CLASS
        -DFILE_PREFIX=solve_
        -DDEPENDENCY_FILE=input
        -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/regenerate_single_style.cmake"
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    COMMENT "Regenerating style_solve.h based on features and headers"
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

# Function to validate style headers after generation
function(validate_style_headers)
    # Load feature status
    set(FEATURE_STATUS_FILE "${CMAKE_CURRENT_BINARY_DIR}/feature_status.cmake")
    if(EXISTS "${FEATURE_STATUS_FILE}")
        include("${FEATURE_STATUS_FILE}")
    else()
        message(WARNING "Feature status file not found for validation")
        return()
    endif()
    
    # Validate HDF5-dependent apps
    if(SPPARKS_HAS_HDF5 AND SPPARKS_HAS_HDF5_FOUND)
        set(STYLE_APP_FILE "${CMAKE_CURRENT_SOURCE_DIR}/style_app.h")
        if(EXISTS "${STYLE_APP_FILE}")
            file(READ "${STYLE_APP_FILE}" STYLE_APP_CONTENT)
            if(NOT STYLE_APP_CONTENT MATCHES "app_additive_ext_temp_texture.h")
                message(FATAL_ERROR 
                    "CMake Error: HDF5 is enabled but app_additive_ext_temp_texture.h is not included in style_app.h\n"
                    "This will cause 'App_style specific command before app_style set' errors.\n"
                    "To fix this issue:\n"
                    "  1. Clean build: rm -rf build && ./build_cmake.sh -m mac_arm --hdf5\n"
                    "  2. Or regenerate: rm ${STYLE_APP_FILE} && sh Make.sh style && cmake --build build\n"
                    "  3. Or use hybrid: ./build_cmake.sh -m mac_arm --hdf5 && sh Make.sh style && cmake --build build"
                )
            else()
                message(STATUS "Validation: HDF5 app registration confirmed in style_app.h")
            endif()
        endif()
    endif()
    
    # Validate image-dependent apps  
    if(SPPARKS_HAS_JPEG OR SPPARKS_HAS_PNG)
        set(STYLE_DUMP_FILE "${CMAKE_CURRENT_SOURCE_DIR}/style_dump.h")
        if(EXISTS "${STYLE_DUMP_FILE}")
            file(READ "${STYLE_DUMP_FILE}" STYLE_DUMP_CONTENT)
            if(NOT STYLE_DUMP_CONTENT MATCHES "dump_image.h")
                message(WARNING 
                    "Image support is enabled but dump_image.h is not included in style_dump.h\n"
                    "Image dump functionality may not be available."
                )
            else()
                message(STATUS "Validation: Image dump registration confirmed in style_dump.h")
            endif()
        endif()
    endif()
    
    message(STATUS "Style header validation completed")
endfunction()

# Custom target to regenerate style headers when source files change
add_custom_target(generate_styles
    COMMAND ${CMAKE_COMMAND} -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/regenerate_styles.cmake"
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    COMMENT "Regenerating style headers..."
)

# Ensure style headers are generated before building targets
add_custom_target(style_headers DEPENDS ${STYLE_HEADERS})

# Add validation target that runs after style headers are generated
# Validation target disabled to prevent script mode issues
if(FALSE) # Disabled
add_custom_target(validate_styles
    COMMAND ${CMAKE_COMMAND} -E echo "Validating style headers..."
    COMMAND ${CMAKE_COMMAND} -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/validate_styles.cmake"
    DEPENDS style_headers
    COMMENT "Validating feature-dependent style headers"
)
endif() # End disabled section

# Validation script generation disabled to prevent script mode issues
if(FALSE) # Disabled
file(WRITE "${CMAKE_CURRENT_SOURCE_DIR}/cmake/validate_styles.cmake"
"# Style header validation script
include(\"${CMAKE_CURRENT_SOURCE_DIR}/cmake/generate_styles.cmake\")
validate_style_headers()
")
endif() # End disabled section

