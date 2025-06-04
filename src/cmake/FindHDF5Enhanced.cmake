# Enhanced HDF5 finder with better parallel detection
# This provides more robust HDF5 detection for SPPARKS

# Use the standard FindHDF5 module first
find_package(HDF5 QUIET COMPONENTS C)

if(HDF5_FOUND)
    # Enhanced parallel detection
    if(NOT DEFINED HDF5_IS_PARALLEL)
        # Try to detect if HDF5 is parallel by checking for MPI symbols
        if(HDF5_LIBRARIES)
            foreach(LIB ${HDF5_LIBRARIES})
                if(EXISTS "${LIB}")
                    execute_process(
                        COMMAND nm -D "${LIB}" 2>/dev/null || objdump -T "${LIB}" 2>/dev/null || strings "${LIB}"
                        OUTPUT_VARIABLE LIB_SYMBOLS
                        ERROR_QUIET
                    )
                    if(LIB_SYMBOLS MATCHES "MPI_|PMPI_")
                        set(HDF5_IS_PARALLEL TRUE)
                        break()
                    endif()
                endif()
            endforeach()
        endif()
        
        # Alternative: check for parallel header definitions
        if(NOT DEFINED HDF5_IS_PARALLEL AND HDF5_INCLUDE_DIRS)
            foreach(INC_DIR ${HDF5_INCLUDE_DIRS})
                set(H5_CONFIG_H "${INC_DIR}/H5pubconf.h")
                if(EXISTS "${H5_CONFIG_H}")
                    file(READ "${H5_CONFIG_H}" H5_CONFIG_CONTENT)
                    if(H5_CONFIG_CONTENT MATCHES "#define H5_HAVE_PARALLEL")
                        set(HDF5_IS_PARALLEL TRUE)
                        break()
                    endif()
                endif()
            endforeach()
        endif()
        
        # Fallback: assume serial if we can't determine
        if(NOT DEFINED HDF5_IS_PARALLEL)
            set(HDF5_IS_PARALLEL FALSE)
        endif()
    endif()
    
    # Provide helpful information
    if(HDF5_IS_PARALLEL)
        message(STATUS "HDF5: Found parallel version")
    else()
        message(STATUS "HDF5: Found serial version")
    endif()
    
    # Create modern CMake target if not already available
    if(NOT TARGET HDF5::HDF5)
        add_library(HDF5::HDF5 UNKNOWN IMPORTED)
        set_target_properties(HDF5::HDF5 PROPERTIES
            IMPORTED_LOCATION "${HDF5_C_LIBRARIES}"
            INTERFACE_INCLUDE_DIRECTORIES "${HDF5_INCLUDE_DIRS}"
            INTERFACE_COMPILE_DEFINITIONS "${HDF5_DEFINITIONS}"
        )
    endif()
    
else()
    # Custom search paths for common HDF5 installations
    set(HDF5_SEARCH_PATHS
        /usr/local/hdf5
        /opt/hdf5
        /opt/homebrew
        $ENV{HDF5_ROOT}
        $ENV{HDF5_DIR}
    )
    
    foreach(SEARCH_PATH ${HDF5_SEARCH_PATHS})
        if(EXISTS "${SEARCH_PATH}/include/hdf5.h")
            message(STATUS "Found HDF5 installation at ${SEARCH_PATH}")
            set(HDF5_INCLUDE_DIRS "${SEARCH_PATH}/include")
            
            # Look for libraries
            find_library(HDF5_C_LIBRARIES
                NAMES hdf5
                PATHS "${SEARCH_PATH}/lib" "${SEARCH_PATH}/lib64"
                NO_DEFAULT_PATH
            )
            
            if(HDF5_C_LIBRARIES)
                set(HDF5_LIBRARIES ${HDF5_C_LIBRARIES})
                set(HDF5_FOUND TRUE)
                break()
            endif()
        endif()
    endforeach()
endif()

# Validation and warnings
if(HDF5_FOUND)
    # Check version if possible
    if(HDF5_INCLUDE_DIRS)
        foreach(INC_DIR ${HDF5_INCLUDE_DIRS})
            set(H5_PUBLIC_H "${INC_DIR}/H5public.h")
            if(EXISTS "${H5_PUBLIC_H}")
                file(READ "${H5_PUBLIC_H}" H5_PUBLIC_CONTENT)
                if(H5_PUBLIC_CONTENT MATCHES "#define H5_VERS_MAJOR[ \t]+([0-9]+)")
                    set(HDF5_VERSION_MAJOR ${CMAKE_MATCH_1})
                endif()
                if(H5_PUBLIC_CONTENT MATCHES "#define H5_VERS_MINOR[ \t]+([0-9]+)")
                    set(HDF5_VERSION_MINOR ${CMAKE_MATCH_1})
                endif()
                if(H5_PUBLIC_CONTENT MATCHES "#define H5_VERS_RELEASE[ \t]+([0-9]+)")
                    set(HDF5_VERSION_PATCH ${CMAKE_MATCH_1})
                endif()
                
                if(DEFINED HDF5_VERSION_MAJOR AND DEFINED HDF5_VERSION_MINOR)
                    set(HDF5_VERSION "${HDF5_VERSION_MAJOR}.${HDF5_VERSION_MINOR}")
                    if(DEFINED HDF5_VERSION_PATCH)
                        set(HDF5_VERSION "${HDF5_VERSION}.${HDF5_VERSION_PATCH}")
                    endif()
                endif()
                break()
            endif()
        endforeach()
    endif()
    
    # Validate that we can compile a simple HDF5 program
    include(CheckCSourceCompiles)
    set(CMAKE_REQUIRED_INCLUDES ${HDF5_INCLUDE_DIRS})
    set(CMAKE_REQUIRED_LIBRARIES ${HDF5_LIBRARIES})
    check_c_source_compiles("
        #include <hdf5.h>
        int main() {
            hid_t file_id = H5Fcreate(\"test.h5\", H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
            H5Fclose(file_id);
            return 0;
        }"
        HDF5_COMPILES)
    
    if(NOT HDF5_COMPILES)
        message(WARNING "HDF5 found but test compilation failed. Check library compatibility.")
    endif()
endif()

# Standard CMake FindPackage handling
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(HDF5Enhanced
    REQUIRED_VARS HDF5_LIBRARIES HDF5_INCLUDE_DIRS
    VERSION_VAR HDF5_VERSION
)

if(HDF5Enhanced_FOUND)
    set(HDF5_FOUND TRUE)
endif()