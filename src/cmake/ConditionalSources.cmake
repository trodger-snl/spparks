# Conditional source management for SPPARKS
# This handles sources that depend on optional features

function(configure_conditional_sources)
    # Start with base sources (already set in main CMakeLists.txt)
    
    # Add additive/texture sources (needs MPI only).
    #
    # HighFive is optional. Only temperature_source_hdf5_{unstructured,csr}.cpp
    # include it (headers forward-declare the HighFive types), and those two TUs
    # compile in every build: without HighFive their HighFive-using functions are
    # #ifdef SPPARKS_HIGHFIVE'd out and setup_temperature_source() errors, so the
    # additive apps run Rosenthal/Moser sources and reject hdf5_* at input time.
    # With HighFive they read through HighFive::MPIOFileAccess (parallel HDF5).
    #
    # Gate on HighFive, NOT bare HDF5, for the HDF5 sources: an env can have HDF5
    # but not HighFive (`--hdf5` without `--highfive`).
    if(SPPARKS_ENABLE_MPI)
        list(APPEND SPPARKS_SOURCES
            app_additive_texture.cpp
            thermal_manager.cpp
            additive_params.cpp
            material_card.cpp
            temperature_source.cpp
            temperature_source_rosenthal.cpp
            temperature_source_moser.cpp
            temperature_source_conduction.cpp
            temperature_source_hdf5_unstructured.cpp
            temperature_source_hdf5_csr.cpp
        )
        # nvcc's front end cannot parse toml++ (toml.hpp:8127, '__T0' not
        # declared, nvcc 12.8); the file has no device code, so it skips nvcc
        if(SPPARKS_PACKAGE_KOKKOS AND SPPARKS_KOKKOS_BACKEND STREQUAL "CUDA")
            set_source_files_properties(material_card.cpp PROPERTIES COMPILE_OPTIONS --host-only)
        endif()
        if(SPPARKS_ENABLE_HIGHFIVE)
            message(STATUS "Added additive/texture sources (HDF5 temperature sources: ON)")
        else()
            message(STATUS "Added additive/texture sources (HDF5 temperature sources: OFF, needs HighFive)")
        endif()
    endif()

    # The HDF5 dump style (dump hdf5 / set hdf5 / reset_time hdf5) has a serial
    # fallback, so it needs HighFive but not MPI. dump_hdf5.h guards its
    # DumpStyle with SPPARKS_HIGHFIVE.
    if(SPPARKS_ENABLE_HIGHFIVE)
        list(APPEND SPPARKS_SOURCES dump_hdf5.cpp)
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
            KOKKOS/app_phasefield_potts_table_kokkos.cpp
        )
        # The Kokkos additive/texture app shares the CPU app's temperature
        # sources, so it has the same MPI-only gate.
        if(SPPARKS_ENABLE_MPI)
            list(APPEND SPPARKS_SOURCES KOKKOS/app_additive_texture_kokkos.cpp
                                        KOKKOS/conduction_solver_kokkos.cpp)
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
    if(NOT SPPARKS_ENABLE_MPI)
        list(APPEND _skipped_styles "additive/texture (app)    needs MPI        → add -m mpi")
    elseif(NOT SPPARKS_ENABLE_HIGHFIVE)
        list(APPEND _skipped_styles "hdf5_* temperature source needs HighFive   → add --highfive")
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
           AND NOT SPPARKS_ENABLE_MPI)
        set(ok FALSE)
        set(why "needs MPI -- -DSPPARKS_ENABLE_MPI=ON")
    elseif(HEADER_NAME MATCHES "dump_stitch\\.h" AND NOT SPPARKS_PACKAGE_STITCH)
        set(ok FALSE)
        set(why "needs the STITCH package -- -DSPPARKS_PACKAGE_STITCH=ON")
    endif()
    set(${RESULT_VAR} ${ok} PARENT_SCOPE)
    string(REPLACE ";" "" why "${why}")
    set(${REASON_VAR} "${why}" PARENT_SCOPE)
endfunction()
