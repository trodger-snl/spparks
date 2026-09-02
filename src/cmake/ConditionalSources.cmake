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

    # Add KOKKOS package sources
    if(SPPARKS_PACKAGE_KOKKOS)
        list(APPEND SPPARKS_SOURCES
            app_lattice_kokkos.cpp
            app_potts_kokkos.cpp
            app_potts_neigh_kokkos.cpp
            app_potts_neighonly_kokkos.cpp
            app_potts_quaternion_kokkos.cpp
        )
        # The Kokkos additive/texture app links against the HighFive-based HDF5
        # temperature sources (same gating rationale as the CPU app above), so it
        # is only registered when both HighFive and MPI are enabled.
        if(SPPARKS_ENABLE_HIGHFIVE AND SPPARKS_ENABLE_MPI)
            list(APPEND SPPARKS_SOURCES app_additive_texture_kokkos.cpp)
        endif()
        message(STATUS "Added KOKKOS package sources")
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
        "set(SPPARKS_HAS_HIGHFIVE ${SPPARKS_ENABLE_HIGHFIVE})\n"
        "set(SPPARKS_HAS_MPI ${SPPARKS_ENABLE_MPI})\n"
        "set(SPPARKS_HAS_JPEG ${JPEG_FOUND})\n"
        "set(SPPARKS_HAS_PNG ${PNG_FOUND})\n"
        "set(SPPARKS_HAS_HDF5_FOUND ${HDF5_FOUND})\n"
        "set(SPPARKS_HAS_STITCH ${SPPARKS_PACKAGE_STITCH})\n"
        "set(SPPARKS_HAS_STITCH_FOUND ${STITCH_LIBRARY_FOUND})\n"
    )

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
    if(NOT (SPPARKS_PACKAGE_STITCH AND STITCH_LIBRARY))
        list(APPEND _skipped_styles "stitch (dump)             needs STITCH pkg → add --package stitch")
    endif()
    set(SPPARKS_SKIPPED_STYLES "${_skipped_styles}" PARENT_SCOPE)
endfunction()

# Function to check if a header should be included based on features.
#
# should_include_header(<header> <result_var> [<reason_var>])
#
# The optional third argument receives a human-readable reason when the header is
# skipped. Callers print it, so that a build log copied off a remote machine says
# WHICH feature was missing -- a bare "missing dependencies" leaves the reader
# unable to tell a misconfigured build from a broken one, and the symptom shows up
# much later as "style not recognized" at runtime.
function(should_include_header HEADER_NAME RESULT_VAR)
    set(INCLUDE_IT TRUE)
    set(SKIP_REASON "")

    # Load feature status. When invoked from a cmake -P script (e.g.
    # regenerate_single_style.cmake), CMAKE_CURRENT_BINARY_DIR points at the working
    # directory rather than the real build dir, so prefer SPK_BINARY_DIR when the caller
    # provides it.
    if(DEFINED SPK_BINARY_DIR)
        set(FEATURE_STATUS_FILE "${SPK_BINARY_DIR}/feature_status.cmake")
    else()
        set(FEATURE_STATUS_FILE "${CMAKE_CURRENT_BINARY_DIR}/feature_status.cmake")
    endif()
    if(EXISTS "${FEATURE_STATUS_FILE}")
        include("${FEATURE_STATUS_FILE}")
    endif()
    
    # Check image-related headers
    if(HEADER_NAME MATCHES "dump_image\\.h|image\\.h")
        if(NOT (SPPARKS_HAS_JPEG OR SPPARKS_HAS_PNG))
            set(INCLUDE_IT FALSE)
            set(SKIP_REASON "needs JPEG or PNG; found JPEG=${SPPARKS_HAS_JPEG} PNG=${SPPARKS_HAS_PNG} "
                            "-- enable with -DSPPARKS_ENABLE_JPEG=ON and install libjpeg headers")
        endif()
    endif()
    
    # Check HighFive-related headers. additive/texture links against the HighFive-based
    # HDF5 temperature sources and uses parallel (MPIO) reads, so it is only registered
    # when both HighFive and MPI are enabled -- matching the source gating in
    # configure_conditional_sources() above. (HighFive implies HDF5.)
    if(HEADER_NAME MATCHES "app_additive_texture(_kokkos)?\\.h")
        if(NOT (SPPARKS_HAS_HIGHFIVE AND SPPARKS_HAS_MPI))
            set(INCLUDE_IT FALSE)
            set(SKIP_REASON "needs HighFive AND MPI; found HighFive=${SPPARKS_HAS_HIGHFIVE} "
                            "MPI=${SPPARKS_HAS_MPI} HDF5=${SPPARKS_HAS_HDF5_FOUND} "
                            "-- enable with -DSPPARKS_ENABLE_HIGHFIVE=ON -DSPPARKS_ENABLE_MPI=ON")
        endif()
    endif()

    # Check STITCH-related headers
    if(HEADER_NAME MATCHES "dump_stitch\\.h")
        if(NOT (SPPARKS_HAS_STITCH AND SPPARKS_HAS_STITCH_FOUND))
            set(INCLUDE_IT FALSE)
            set(SKIP_REASON "needs the STITCH package AND its library; package=${SPPARKS_HAS_STITCH} "
                            "library_found=${SPPARKS_HAS_STITCH_FOUND} "
                            "-- enable with -DSPPARKS_PACKAGE_STITCH=ON and build lib/stitch")
        endif()
    endif()

    set(${RESULT_VAR} ${INCLUDE_IT} PARENT_SCOPE)
    if(ARGC GREATER 2)
        string(REPLACE ";" "" SKIP_REASON "${SKIP_REASON}")
        set(${ARGV2} "${SKIP_REASON}" PARENT_SCOPE)
    endif()
endfunction()