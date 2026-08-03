# Enhanced HDF5 finder with better parallel detection
# This provides more robust HDF5 detection for SPPARKS

# When MPI is on we require a parallel HDF5, so tell FindHDF5 that before it
# searches. Without this it probes the serial `h5cc` wrapper first, and a wrapper
# it finds on PATH wins over HDF5_ROOT -- so on a Cray machine with a serial
# cray-hdf5 module loaded, -DHDF5_ROOT=<parallel prefix> appears to be ignored
# and configure fails on "the HDF5 found ... is serial" while pointing at the
# serial tree. HDF5_PREFER_PARALLEL makes it look for h5pcc first.
if(SPPARKS_ENABLE_MPI AND NOT DEFINED HDF5_PREFER_PARALLEL)
    set(HDF5_PREFER_PARALLEL ON)
endif()

# HDF5_ROOT is an installation PREFIX, not an include directory. Passing the
# include dir (an easy mistake, since that is the path the configure summary
# prints) makes CMake search <root>/include/include and <root>/include/lib, so
# headers may still resolve while libraries do not -- surfacing as the very
# unhelpful "Could NOT find HDF5Enhanced (missing: HDF5_LIBRARIES) (found
# version ...)". No real prefix is named "include", so correct it and say so.
if(HDF5_ROOT)
    get_filename_component(_h5_root_leaf "${HDF5_ROOT}" NAME)
    if(_h5_root_leaf STREQUAL "include")
        get_filename_component(HDF5_ROOT "${HDF5_ROOT}" DIRECTORY)
        message(STATUS
            "HDF5: HDF5_ROOT pointed at an include directory; using the prefix "
            "'${HDF5_ROOT}' instead (HDF5_ROOT must be the installation prefix).")
    endif()
    unset(_h5_root_leaf)
endif()

# Use the standard FindHDF5 module first
find_package(HDF5 QUIET COMPONENTS C)

# Recover a library path when FindHDF5 resolved headers but not libraries.
#
# Two ways that happens in practice, both on Cray:
#   * FindHDF5 learned the include dir and version from the h5pcc wrapper but
#     could not turn the wrapper's link line into a file path, leaving
#     HDF5_LIBRARIES empty while HDF5_C_LIBRARIES is set (or neither is).
#   * cray-hdf5-parallel installs libhdf5_parallel.so, NOT libhdf5.so, so a
#     plain search for "hdf5" finds nothing under an otherwise correct
#     HDF5_ROOT.
#
# Symptom before this: "Could NOT find HDF5Enhanced (missing: HDF5_LIBRARIES)
# (found version 1.14.3)" -- a found-but-unusable HDF5.
if(HDF5_FOUND AND NOT HDF5_LIBRARIES)
    if(HDF5_C_LIBRARIES)
        set(HDF5_LIBRARIES ${HDF5_C_LIBRARIES})
        message(STATUS "HDF5: using HDF5_C_LIBRARIES for HDF5_LIBRARIES")
    else()
        # Derive candidate prefixes from whatever we do know.
        set(_h5_hints "${HDF5_ROOT}" "$ENV{HDF5_ROOT}" "$ENV{HDF5_DIR}")
        foreach(_inc IN LISTS HDF5_INCLUDE_DIRS)
            get_filename_component(_h5_prefix "${_inc}" DIRECTORY)
            list(APPEND _h5_hints "${_h5_prefix}")
        endforeach()

        find_library(HDF5_RECOVERED_LIBRARY
            NAMES hdf5_parallel hdf5      # Cray parallel name first
            HINTS ${_h5_hints}
            PATH_SUFFIXES lib lib64
        )
        if(HDF5_RECOVERED_LIBRARY)
            set(HDF5_LIBRARIES ${HDF5_RECOVERED_LIBRARY})
            set(HDF5_C_LIBRARIES ${HDF5_RECOVERED_LIBRARY})
            message(STATUS "HDF5: recovered library ${HDF5_RECOVERED_LIBRARY}")
        endif()
        unset(_h5_hints)
    endif()
endif()

if(HDF5_FOUND)
    # Enhanced parallel detection
    if(NOT DEFINED HDF5_IS_PARALLEL)
        # Try to detect if HDF5 is parallel by checking for MPI symbols
        if(HDF5_LIBRARIES)
            foreach(LIB ${HDF5_LIBRARIES})
                if(EXISTS "${LIB}")
                    # Platform-specific symbol extraction
                    # Try nm first (with platform-appropriate flag)
                    if(APPLE)
                        set(NM_FLAGS -g)  # macOS uses -g for global symbols
                    else()
                        set(NM_FLAGS -D)  # Linux uses -D for dynamic symbols
                    endif()

                    execute_process(
                        COMMAND nm ${NM_FLAGS} "${LIB}"
                        OUTPUT_VARIABLE LIB_SYMBOLS
                        ERROR_QUIET
                        RESULT_VARIABLE NM_RESULT
                    )

                    # If nm fails, try objdump
                    if(NOT NM_RESULT EQUAL 0)
                        execute_process(
                            COMMAND objdump -T "${LIB}"
                            OUTPUT_VARIABLE LIB_SYMBOLS
                            ERROR_QUIET
                            RESULT_VARIABLE OBJDUMP_RESULT
                        )

                        # If objdump fails, fallback to strings
                        if(NOT OBJDUMP_RESULT EQUAL 0)
                            execute_process(
                                COMMAND strings "${LIB}"
                                OUTPUT_VARIABLE LIB_SYMBOLS
                                ERROR_QUIET
                            )
                        endif()
                    endif()

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