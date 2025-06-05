# Script to regenerate a single style header
# This script is called with the following variables defined:
# STYLE_TYPE - The style type (app, diag, dump, etc.)
# CLASS_PATTERN - The pattern to search for in headers
# FILE_PREFIX - The file prefix to search for
# DEPENDENCY_FILE - The dependency file to remove

# Include necessary functions
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/ConditionalSources.cmake")

# Set up paths
set(STYLE_HEADER "${CMAKE_CURRENT_SOURCE_DIR}/style_${STYLE_TYPE}.h")
set(STYLE_TEMP "${CMAKE_CURRENT_BINARY_DIR}/style_${STYLE_TYPE}.tmp")

# Load feature status - try multiple possible locations
set(FEATURE_STATUS_FILE "${CMAKE_CURRENT_BINARY_DIR}/feature_status.cmake")
set(FEATURE_STATUS_ALT "${CMAKE_CURRENT_SOURCE_DIR}/build/feature_status.cmake")

if(EXISTS "${FEATURE_STATUS_FILE}")
    include("${FEATURE_STATUS_FILE}")
elseif(EXISTS "${FEATURE_STATUS_ALT}")
    include("${FEATURE_STATUS_ALT}")
else()
    # Default to enabled if we can't find feature status
    set(SPPARKS_HAS_HDF5 TRUE)
    set(SPPARKS_HAS_JPEG TRUE)
    set(SPPARKS_HAS_PNG FALSE)
    set(SPPARKS_HAS_HDF5_FOUND TRUE)
    message(STATUS "Using default feature status - HDF5 enabled")
endif()

# Find matching header files
if(FILE_PREFIX STREQUAL "")
    file(GLOB MATCHING_HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/*.h")
else()
    file(GLOB MATCHING_HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/${FILE_PREFIX}*.h")
endif()

# Filter headers based on CLASS_PATTERN and feature requirements
set(VALID_HEADERS "")
foreach(HEADER ${MATCHING_HEADERS})
    file(READ "${HEADER}" HEADER_CONTENT)
    if(HEADER_CONTENT MATCHES "${CLASS_PATTERN}")
        get_filename_component(HEADER_NAME "${HEADER}" NAME)
        set(INCLUDE_HEADER TRUE)
        
        # Check if header should be included based on features
        should_include_header("${HEADER_NAME}" INCLUDE_HEADER)
        
        if(INCLUDE_HEADER)
            list(APPEND VALID_HEADERS "${HEADER_NAME}")
        else()
            message(STATUS "Style generation for ${STYLE_TYPE}: Skipping ${HEADER_NAME} (missing dependencies)")
        endif()
    endif()
endforeach()

# Sort headers to ensure consistent order
list(SORT VALID_HEADERS)

# Generate the temporary file
file(WRITE "${STYLE_TEMP}" "")
foreach(HEADER_NAME ${VALID_HEADERS})
    file(APPEND "${STYLE_TEMP}" "#include \"${HEADER_NAME}\"\n")
endforeach()

# Update if needed
set(UPDATE_HEADER FALSE)
if(NOT EXISTS "${STYLE_HEADER}")
    set(UPDATE_HEADER TRUE)
else()
    file(READ "${STYLE_HEADER}" CURRENT_CONTENT)
    file(READ "${STYLE_TEMP}" NEW_CONTENT)
    if(NOT "${CURRENT_CONTENT}" STREQUAL "${NEW_CONTENT}")
        set(UPDATE_HEADER TRUE)
    endif()
endif()

if(UPDATE_HEADER)
    file(RENAME "${STYLE_TEMP}" "${STYLE_HEADER}")
    message(STATUS "Generated ${STYLE_HEADER}")
else()
    file(REMOVE "${STYLE_TEMP}")
endif()