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

# Re-glob a version-pinned HDF5_ROOT that the site has moved past.
#
# The mi300a preset pins HDF5_ROOT, because LC's MI300A systems ship no hdf5
# module at all -- the Cray parallel install is reachable only by absolute path.
# That pin necessarily names one Cray PE version, and it goes stale the first
# time the site upgrades. Treat the version/compiler/ABI components as a hint:
# wildcard them, take the highest match, and say which one was used. A stale pin
# should cost a message, not a failed configure on a path nobody typed.
#
# A non-Cray HDF5_ROOT leaves the pattern unchanged, matches nothing, and simply
# falls through to FindHDF5's ordinary search below.
if(HDF5_ROOT AND NOT IS_DIRECTORY "${HDF5_ROOT}")
    string(REGEX REPLACE "^(.*/hdf5-parallel)/.*$" "\\1/*/*/*" _h5_glob "${HDF5_ROOT}")
    file(GLOB _h5_candidates LIST_DIRECTORIES true "${_h5_glob}")

    set(_h5_usable "")
    foreach(_cand IN LISTS _h5_candidates)
        if(IS_DIRECTORY "${_cand}/lib" OR IS_DIRECTORY "${_cand}/lib64")
            list(APPEND _h5_usable "${_cand}")
        endif()
    endforeach()

    if(_h5_usable)
        list(SORT _h5_usable COMPARE NATURAL)
        list(GET _h5_usable -1 _h5_newest)
        message(STATUS
            "HDF5: pinned HDF5_ROOT '${HDF5_ROOT}' does not exist; using "
            "'${_h5_newest}' found under the same tree. Update the pin in "
            "CMakePresets.json, or pass -DHDF5_ROOT=<prefix> explicitly.")
        set(HDF5_ROOT "${_h5_newest}")
    else()
        message(STATUS
            "HDF5: HDF5_ROOT '${HDF5_ROOT}' does not exist and nothing matched "
            "'${_h5_glob}'; ignoring it and searching the default paths.")
        set(HDF5_ROOT "")
    endif()

    unset(_h5_glob)
    unset(_h5_candidates)
    unset(_h5_usable)
    unset(_h5_newest)
    unset(_cand)
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
    
    # Validate that we can compile a simple HDF5 program.
    #
    # A PARALLEL hdf5.h includes <mpi.h> (H5public.h), so this test needs the MPI
    # include path too. find_package(MPI) runs much later in CMakeLists.txt (the
    # HDF5 search is at ~269, MPI at ~579), so MPI_C_INCLUDE_DIRS does not exist
    # yet -- look it up here. Without this the test fails on any toolchain whose
    # plain C compiler does not carry mpi.h by default, which is most of them; it
    # only passed on the dev machine because a spack view happened to put mpi.h
    # on the default include path.
    set(_h5_test_includes ${HDF5_INCLUDE_DIRS})
    set(_h5_can_test TRUE)
    if(HDF5_IS_PARALLEL)
        # Ask for both languages but require neither: on some Cray setups the C
        # component fails to validate while CXX is fine, and either language's
        # include dir is enough to find mpi.h for this probe.
        find_package(MPI QUIET COMPONENTS C CXX)
        if(MPI_C_INCLUDE_DIRS)
            list(APPEND _h5_test_includes ${MPI_C_INCLUDE_DIRS})
        elseif(MPI_CXX_INCLUDE_DIRS)
            list(APPEND _h5_test_includes ${MPI_CXX_INCLUDE_DIRS})
        elseif(MPI_C_INCLUDE_PATH)
            list(APPEND _h5_test_includes ${MPI_C_INCLUDE_PATH})
        elseif(MPI_CXX_INCLUDE_PATH)
            list(APPEND _h5_test_includes ${MPI_CXX_INCLUDE_PATH})
        else()
            # Parallel HDF5 but no MPI include path to give the probe. The probe
            # would fail for want of mpi.h and say nothing about HDF5, so skip it
            # -- the real build compiles through the MPI wrapper and is fine.
            set(_h5_can_test FALSE)
            message(STATUS
                "HDF5: parallel build and MPI includes not yet located; skipping "
                "the compile probe (the build itself uses the MPI compiler wrapper)")
        endif()
    endif()

    if(_h5_can_test)
        include(CheckCSourceCompiles)
        set(CMAKE_REQUIRED_INCLUDES ${_h5_test_includes})
        set(CMAKE_REQUIRED_LIBRARIES ${HDF5_LIBRARIES})
        check_c_source_compiles("
            #include <hdf5.h>
            int main() {
                hid_t file_id = H5Fcreate(\"test.h5\", H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
                H5Fclose(file_id);
                return 0;
            }"
            HDF5_COMPILES)
        unset(CMAKE_REQUIRED_INCLUDES)
        unset(CMAKE_REQUIRED_LIBRARIES)

        if(NOT HDF5_COMPILES)
            # Advisory only. This probe uses the plain C compiler, while SPPARKS
            # compiles these sources with the MPI C++ wrapper, so a failure here
            # is frequently a probe artifact rather than a broken HDF5.
            message(WARNING
                "HDF5 compile probe failed.\n"
                "  include dirs: ${_h5_test_includes}\n"
                "  libraries:    ${HDF5_LIBRARIES}\n"
                "  parallel:     ${HDF5_IS_PARALLEL}\n"
                "This probe uses the plain C compiler (${CMAKE_C_COMPILER}), not the "
                "MPI wrapper the build actually uses, so it can fail on a perfectly "
                "good parallel HDF5 that simply needs mpi.h from the wrapper. "
                "Configure continues; if the build then fails on a missing mpi.h or "
                "hdf5.h, this was real -- see CMakeFiles/CMakeError.log for the "
                "compiler output.")
        endif()
    endif()
    unset(_h5_test_includes)
    unset(_h5_can_test)
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