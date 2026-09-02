# Conditional source management for SPPARKS
# This handles sources that depend on optional features

function(configure_conditional_sources)
    # Start with base sources (already set in main CMakeLists.txt)
    
    # Add additive/texture sources.
    #
    # NOTE: These are gated on HighFive (NOT bare HDF5). app_additive_texture and its
    # temperature sources are written entirely against the HighFive C++ API
    # (#include "highfive/highfive.hpp") rather than the raw HDF5 C API, and the HDF5
    # readers call HighFive::MPIOFileAccess(...) unconditionally -- so they require
    # HighFive built with parallel HDF5, i.e. HighFive AND MPI. HighFive itself
    # auto-enables HDF5 (see CMakeLists.txt), so HighFive implies HDF5_FOUND.
    #
    # Gating on HDF5_FOUND alone is wrong: an env can have HDF5 but not HighFive (e.g.
    # `--hdf5` without `--highfive`), which passes that check and then fails to compile
    # temperature_source_hdf5_unstructured.cpp on the missing highfive header.
    #
    # app_additive_texture.cpp references HDF5UnstructuredTemperatureSource via
    # forward-declared HighFive types, so it must be linked together with the two
    # *_hdf5_*.cpp TUs; we therefore add the whole set together (no serial subset).
    if(SPPARKS_ENABLE_HIGHFIVE AND SPPARKS_ENABLE_MPI)
        list(APPEND SPPARKS_SOURCES
            app_additive_texture.cpp
            thermal_manager.cpp
            temperature_source.cpp
            temperature_source_rosenthal.cpp
            temperature_source_moser.cpp
            temperature_source_hdf5_unstructured.cpp
            temperature_source_hdf5_csr.cpp
        )
        message(STATUS "Added HighFive+MPI temperature source files (additive/texture)")
    endif()
    
    # Add image-dependent sources
    if((SPPARKS_ENABLE_JPEG AND JPEG_FOUND) OR (SPPARKS_ENABLE_PNG AND PNG_FOUND))
        list(APPEND SPPARKS_SOURCES dump_image.cpp image.cpp)
        message(STATUS "Added image-dependent sources")
    endif()

    # Package sources compile in place from src/KOKKOS/ and src/STITCH/ (those
    # directories are on the include path when the package is on).
    if(SPPARKS_PACKAGE_KOKKOS)
        list(APPEND SPPARKS_SOURCES
            KOKKOS/app_lattice_kokkos.cpp
            KOKKOS/app_potts_kokkos.cpp
            KOKKOS/app_potts_neigh_kokkos.cpp
            KOKKOS/app_potts_neighonly_kokkos.cpp
            KOKKOS/app_potts_quaternion_kokkos.cpp
        )
        # The Kokkos additive/texture app links against the HighFive-based HDF5
        # temperature sources (same gating rationale as the CPU app above), so it
        # is only registered when both HighFive and MPI are enabled.
        if(SPPARKS_ENABLE_HIGHFIVE AND SPPARKS_ENABLE_MPI)
            list(APPEND SPPARKS_SOURCES KOKKOS/app_additive_texture_kokkos.cpp)
        endif()
        message(STATUS "Added KOKKOS package sources")
    endif()
    if(SPPARKS_PACKAGE_STITCH)
        list(APPEND SPPARKS_SOURCES STITCH/dump_stitch.cpp)
    endif()

    # Update the parent scope
    set(SPPARKS_SOURCES "${SPPARKS_SOURCES}" PARENT_SCOPE)
    
    # Feature flags for cmake -P regeneration of the style headers (see
    # generate_styles.cmake); at configure time should_include_header() reads
    # the same variables straight from this scope.
    set(_status "# Auto-generated feature status for style generation\n")
    foreach(_v SPPARKS_ENABLE_HIGHFIVE SPPARKS_ENABLE_MPI JPEG_FOUND PNG_FOUND
                SPPARKS_PACKAGE_STITCH SPPARKS_PACKAGE_KOKKOS)
        string(APPEND _status "set(${_v} \"${${_v}}\")\n")
    endforeach()
    file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/feature_status.cmake" "${_status}")

    # Collect optional styles that were NOT built because their feature is off, so the
    # configuration summary can tell the user exactly how to enable each one. This must
    # stay in sync with the gating above and with should_include_header() below.
    set(_skipped_styles "")
    if(NOT (SPPARKS_ENABLE_HIGHFIVE AND SPPARKS_ENABLE_MPI))
        list(APPEND _skipped_styles "additive/texture (app)    needs HighFive+MPI → add --highfive -m mpi")
    endif()
    if(NOT ((SPPARKS_ENABLE_JPEG AND JPEG_FOUND) OR (SPPARKS_ENABLE_PNG AND PNG_FOUND)))
        list(APPEND _skipped_styles "image (dump)              needs JPEG/PNG   → add --jpeg")
    endif()
    if(NOT SPPARKS_PACKAGE_STITCH)
        list(APPEND _skipped_styles "stitch (dump)             needs STITCH pkg → add --package stitch")
    endif()
    set(SPPARKS_SKIPPED_STYLES "${_skipped_styles}" PARENT_SCOPE)
endfunction()

# should_include_header(<header> <result_var> <reason_var>)
#
# Feature gating for style registration. The reason is printed by the caller so
# a build log copied off a remote machine says WHICH feature was missing; the
# symptom otherwise shows up much later as "style not recognized" at runtime.
# Must stay in sync with the source gating in configure_conditional_sources().
function(should_include_header HEADER_NAME RESULT_VAR REASON_VAR)
    set(ok TRUE)
    set(why "")
    if(HEADER_NAME MATCHES "dump_image\\.h|image\\.h" AND NOT (JPEG_FOUND OR PNG_FOUND))
        set(ok FALSE)
        set(why "needs JPEG or PNG -- -DSPPARKS_ENABLE_JPEG=ON plus libjpeg headers")
    elseif(HEADER_NAME MATCHES "app_additive_texture(_kokkos)?\\.h"
           AND NOT (SPPARKS_ENABLE_HIGHFIVE AND SPPARKS_ENABLE_MPI))
        set(ok FALSE)
        set(why "needs HighFive AND MPI; found HighFive=${SPPARKS_ENABLE_HIGHFIVE} "
                "MPI=${SPPARKS_ENABLE_MPI} -- -DSPPARKS_ENABLE_HIGHFIVE=ON -DSPPARKS_ENABLE_MPI=ON")
    elseif(HEADER_NAME MATCHES "dump_stitch\\.h" AND NOT SPPARKS_PACKAGE_STITCH)
        set(ok FALSE)
        set(why "needs the STITCH package -- -DSPPARKS_PACKAGE_STITCH=ON")
    endif()
    set(${RESULT_VAR} ${ok} PARENT_SCOPE)
    string(REPLACE ";" "" why "${why}")
    set(${REASON_VAR} "${why}" PARENT_SCOPE)
endfunction()
