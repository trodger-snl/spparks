# FindHDF5 plus the HDF5_ROOT / library-name workarounds Cray systems need.

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

# FindHDF5 sets HDF5_IS_PARALLEL and HDF5_VERSION itself.
if(HDF5_FOUND AND NOT DEFINED HDF5_IS_PARALLEL)
    set(HDF5_IS_PARALLEL FALSE)
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