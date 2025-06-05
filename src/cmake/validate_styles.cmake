# Simple style header validation script
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

message(STATUS "Style header validation completed")