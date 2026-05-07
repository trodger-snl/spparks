# Conditional source management for SPPARKS
# This handles sources that depend on optional features

function(configure_conditional_sources)
    # Start with base sources (already set in main CMakeLists.txt)
    
    # Add HDF5-dependent sources
    if(SPPARKS_ENABLE_HDF5 AND HDF5_FOUND)
        # Base HDF5 temperature sources (require MPI because they use H5Pset_dxpl_mpio)
        if(SPPARKS_ENABLE_MPI)
            list(APPEND SPPARKS_SOURCES
                app_additive_texture.cpp
                temperature_source.cpp
                temperature_source_rosenthal.cpp
                temperature_source_moser.cpp
                temperature_source_hdf5_unstructured.cpp
                temperature_source_hdf5_csr.cpp
            )
            message(STATUS "Added HDF5+MPI temperature source files")
        else()
            # For serial HDF5, only add non-MPI temperature sources
            list(APPEND SPPARKS_SOURCES
                app_additive_texture.cpp
                temperature_source.cpp
                temperature_source_rosenthal.cpp
                temperature_source_moser.cpp
            )
            message(STATUS "Added HDF5 temperature source files (serial mode - no parallel I/O)")
        endif()
    endif()
    
    # Add image-dependent sources
    if((SPPARKS_ENABLE_JPEG AND JPEG_FOUND) OR (SPPARKS_ENABLE_PNG AND PNG_FOUND))
        list(APPEND SPPARKS_SOURCES dump_image.cpp image.cpp)
        message(STATUS "Added image-dependent sources")
    endif()
    
    # Update the parent scope
    set(SPPARKS_SOURCES "${SPPARKS_SOURCES}" PARENT_SCOPE)
    
    # Create a feature status file for style generation
    set(FEATURE_STATUS_FILE "${CMAKE_CURRENT_BINARY_DIR}/feature_status.cmake")

    # Determine if STITCH library was found
    if(STITCH_LIBRARY)
        set(STITCH_LIBRARY_FOUND TRUE)
    else()
        set(STITCH_LIBRARY_FOUND FALSE)
    endif()

    file(WRITE "${FEATURE_STATUS_FILE}"
        "# Auto-generated feature status for style generation\n"
        "set(SPPARKS_HAS_HDF5 ${SPPARKS_ENABLE_HDF5})\n"
        "set(SPPARKS_HAS_JPEG ${JPEG_FOUND})\n"
        "set(SPPARKS_HAS_PNG ${PNG_FOUND})\n"
        "set(SPPARKS_HAS_HDF5_FOUND ${HDF5_FOUND})\n"
        "set(SPPARKS_HAS_STITCH ${SPPARKS_PACKAGE_STITCH})\n"
        "set(SPPARKS_HAS_STITCH_FOUND ${STITCH_LIBRARY_FOUND})\n"
    )
endfunction()

# Function to check if a header should be included based on features
function(should_include_header HEADER_NAME RESULT_VAR)
    set(INCLUDE_IT TRUE)
    
    # Load feature status
    set(FEATURE_STATUS_FILE "${CMAKE_CURRENT_BINARY_DIR}/feature_status.cmake")
    if(EXISTS "${FEATURE_STATUS_FILE}")
        include("${FEATURE_STATUS_FILE}")
    endif()
    
    # Check image-related headers
    if(HEADER_NAME MATCHES "dump_image\\.h|image\\.h")
        if(NOT (SPPARKS_HAS_JPEG OR SPPARKS_HAS_PNG))
            set(INCLUDE_IT FALSE)
        endif()
    endif()
    
    # Check HDF5-related headers
    if(HEADER_NAME MATCHES "app_additive_texture\\.h")
        if(NOT (SPPARKS_HAS_HDF5 AND SPPARKS_HAS_HDF5_FOUND))
            set(INCLUDE_IT FALSE)
        endif()
    endif()

    # Check STITCH-related headers
    if(HEADER_NAME MATCHES "dump_stitch\\.h")
        if(NOT (SPPARKS_HAS_STITCH AND SPPARKS_HAS_STITCH_FOUND))
            set(INCLUDE_IT FALSE)
        endif()
    endif()

    set(${RESULT_VAR} ${INCLUDE_IT} PARENT_SCOPE)
endfunction()