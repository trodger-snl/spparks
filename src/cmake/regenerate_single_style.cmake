# Script to regenerate a single style header
# This script is called (via cmake -P) with the following variables defined:
# STYLE_TYPE - The style type (app, diag, dump, etc.)
# CLASS_PATTERN - The pattern to search for in headers
# FILE_PREFIX - The file prefix to search for
# DEPENDENCY_FILE - The dependency file to remove
# SPK_SOURCE_DIR - The SPPARKS source dir (where style_*.h and headers live)
# SPK_BINARY_DIR - The CMake build dir (where feature_status.cmake is written)
#
# IMPORTANT: In script mode (cmake -P), CMAKE_CURRENT_BINARY_DIR is NOT the build dir;
# it collapses to the working directory (the source dir). We therefore rely on the
# explicit SPK_SOURCE_DIR / SPK_BINARY_DIR passed by the caller, falling back to the
# CMAKE_CURRENT_* values only for backward compatibility with in-tree (source==build)
# builds.
if(NOT DEFINED SPK_SOURCE_DIR)
    set(SPK_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
endif()
if(NOT DEFINED SPK_BINARY_DIR)
    set(SPK_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}")
endif()

# Include necessary functions (should_include_header consults SPK_BINARY_DIR)
include("${SPK_SOURCE_DIR}/cmake/ConditionalSources.cmake")

# Set up paths
set(STYLE_HEADER "${SPK_SOURCE_DIR}/style_${STYLE_TYPE}.h")
set(STYLE_TEMP "${SPK_BINARY_DIR}/style_${STYLE_TYPE}.tmp")

# Load feature status from the real build dir
set(FEATURE_STATUS_FILE "${SPK_BINARY_DIR}/feature_status.cmake")

if(EXISTS "${FEATURE_STATUS_FILE}")
    include("${FEATURE_STATUS_FILE}")
else()
    # Default to enabled if we can't find feature status
    set(SPPARKS_HAS_HDF5 TRUE)
    set(SPPARKS_HAS_HIGHFIVE TRUE)
    set(SPPARKS_HAS_MPI TRUE)
    set(SPPARKS_HAS_JPEG TRUE)
    set(SPPARKS_HAS_PNG FALSE)
    set(SPPARKS_HAS_HDF5_FOUND TRUE)
    message(STATUS "Using default feature status - HDF5 enabled (feature_status.cmake not found at ${FEATURE_STATUS_FILE})")
endif()

# Find matching header files
if(FILE_PREFIX STREQUAL "")
    file(GLOB MATCHING_HEADERS "${SPK_SOURCE_DIR}/*.h")
else()
    file(GLOB MATCHING_HEADERS "${SPK_SOURCE_DIR}/${FILE_PREFIX}*.h")
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
    # COPY into the destination directory then RENAME within it, so the move works even
    # when STYLE_TEMP (build dir) and STYLE_HEADER (source dir) are on different
    # filesystems -- a plain file(RENAME) across filesystems fails ("cross-device link").
    file(COPY "${STYLE_TEMP}" DESTINATION "${SPK_SOURCE_DIR}")
    get_filename_component(_style_tmp_name "${STYLE_TEMP}" NAME)
    file(RENAME "${SPK_SOURCE_DIR}/${_style_tmp_name}" "${STYLE_HEADER}")
    file(REMOVE "${STYLE_TEMP}")
    message(STATUS "Generated ${STYLE_HEADER}")
else()
    file(REMOVE "${STYLE_TEMP}")
endif()