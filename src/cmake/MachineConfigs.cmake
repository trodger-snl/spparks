# Machine-specific configurations for SPPARKS
# Equivalent to the MAKE/Makefile.* files

# Set SPPARKS_ENABLE_MPI from a machine config, warning loudly if that contradicts
# what the user asked for on the command line.
#
# The override is invisible otherwise, and actively misleading: the machine config
# sets a normal variable that shadows the cache, so CMakeCache.txt keeps reporting
# the user's value while the build uses the machine config's. Anyone diagnosing
# "I passed -DSPPARKS_ENABLE_MPI=ON and got a serial binary" from a pasted log
# needs this stated.
macro(_spk_machine_set_mpi _value _machine)
    if(DEFINED CACHE{SPPARKS_ENABLE_MPI} AND NOT "$CACHE{SPPARKS_ENABLE_MPI}" STREQUAL "${_value}")
        message(WARNING
            "Machine config '${_machine}' is overriding SPPARKS_ENABLE_MPI: "
            "requested $CACHE{SPPARKS_ENABLE_MPI}, machine config forces ${_value}.\n"
            "CMakeCache.txt will still show $CACHE{SPPARKS_ENABLE_MPI}; the build uses ${_value}.\n"
            "Pass -DSPPARKS_MACHINE=mpi (or another preset whose MPI setting matches) to avoid this.")
    endif()
    set(SPPARKS_ENABLE_MPI ${_value} PARENT_SCOPE)
endmacro()

# Function to apply machine-specific settings
function(apply_machine_config MACHINE_NAME)
    message(STATUS "Applying machine configuration: ${MACHINE_NAME}")

    # Don't override compiler for GPU backends (HIP, CUDA, SYCL)
    # These require their specific compilers (hipcc, nvcc, etc.)
    if(SPPARKS_KOKKOS_BACKEND STREQUAL "HIP" OR
       SPPARKS_KOKKOS_BACKEND STREQUAL "CUDA" OR
       SPPARKS_KOKKOS_BACKEND STREQUAL "SYCL")
        message(STATUS "GPU backend detected - keeping user-specified compiler")
        # Only apply non-compiler settings from machine config
        if(MACHINE_NAME STREQUAL "linux" OR MACHINE_NAME STREQUAL "mpi" OR
           MACHINE_NAME STREQUAL "serial" OR MACHINE_NAME STREQUAL "debug")
            add_compile_definitions(SPPARKS_GZIP)
        endif()
        return()
    endif()
    
    if(MACHINE_NAME STREQUAL "mac")
        # macOS configuration (no MPI)
        _spk_machine_set_mpi(OFF "${MACHINE_NAME}")
        set(CMAKE_CXX_COMPILER "c++" PARENT_SCOPE)
        set(CMAKE_C_COMPILER "cc" PARENT_SCOPE)
        add_compile_definitions(SPPARKS_GZIP)
        
    elseif(MACHINE_NAME STREQUAL "mac_mpi")
        # macOS with MPI
        _spk_machine_set_mpi(ON "${MACHINE_NAME}")
        set(CMAKE_CXX_COMPILER "mpicxx" PARENT_SCOPE)
        set(CMAKE_C_COMPILER "mpicc" PARENT_SCOPE)
        add_compile_definitions(SPPARKS_GZIP)
        
    elseif(MACHINE_NAME STREQUAL "mac_arm")
        # Apple Silicon Mac with MPI (matches Makefile.mac_arm)
        _spk_machine_set_mpi(ON "${MACHINE_NAME}")
        
        # Force enable HDF5 and JPEG for mac_arm unless explicitly overridden by command line
        if(NOT DEFINED CACHE{SPPARKS_ENABLE_HDF5} OR NOT CMAKE_ARGV MATCHES "SPPARKS_ENABLE_HDF5")
            set(SPPARKS_ENABLE_HDF5 ON CACHE BOOL "Enable HDF5 support" FORCE)
        endif()
        if(NOT DEFINED CACHE{SPPARKS_ENABLE_JPEG} OR NOT CMAKE_ARGV MATCHES "SPPARKS_ENABLE_JPEG")
            set(SPPARKS_ENABLE_JPEG ON CACHE BOOL "Enable JPEG support" FORCE)
        endif()
        
        message(STATUS "mac_arm: Enabled HDF5 and JPEG support (if not overridden)")
        
        set(CMAKE_CXX_COMPILER "mpic++" PARENT_SCOPE)
        set(CMAKE_C_COMPILER "mpicc" PARENT_SCOPE)
        add_compile_definitions(SPPARKS_GZIP SPPARKS_UNORDERED_MAP)
        
        # Set Homebrew paths for Apple Silicon
        if(EXISTS "/opt/homebrew")
            set(CMAKE_PREFIX_PATH "/opt/homebrew" PARENT_SCOPE)
        endif()
        
    elseif(MACHINE_NAME STREQUAL "linux")
        # Generic Linux
        _spk_machine_set_mpi(OFF "${MACHINE_NAME}")
        set(CMAKE_CXX_COMPILER "g++" PARENT_SCOPE)
        set(CMAKE_C_COMPILER "gcc" PARENT_SCOPE)
        add_compile_definitions(SPPARKS_GZIP)
        
    elseif(MACHINE_NAME STREQUAL "mpi")
        # Generic MPI
        _spk_machine_set_mpi(ON "${MACHINE_NAME}")
        set(CMAKE_CXX_COMPILER "mpicxx" PARENT_SCOPE)
        set(CMAKE_C_COMPILER "mpicc" PARENT_SCOPE)
        add_compile_definitions(SPPARKS_GZIP SPPARKS_UNORDERED_MAP)
        
    elseif(MACHINE_NAME STREQUAL "serial")
        # Serial build
        _spk_machine_set_mpi(OFF "${MACHINE_NAME}")
        set(CMAKE_CXX_COMPILER "g++" PARENT_SCOPE)
        set(CMAKE_C_COMPILER "gcc" PARENT_SCOPE)
        add_compile_definitions(SPPARKS_GZIP)
        
    elseif(MACHINE_NAME STREQUAL "debug")
        # Debug build
        set(CMAKE_BUILD_TYPE "Debug" PARENT_SCOPE)
        set(CMAKE_CXX_COMPILER "g++" PARENT_SCOPE)
        set(CMAKE_C_COMPILER "gcc" PARENT_SCOPE)
        add_compile_definitions(SPPARKS_GZIP)
        
    elseif(MACHINE_NAME STREQUAL "mpi_debug")
        # MPI Debug build
        set(CMAKE_BUILD_TYPE "Debug" PARENT_SCOPE)
        _spk_machine_set_mpi(ON "${MACHINE_NAME}")
        set(CMAKE_CXX_COMPILER "mpicxx" PARENT_SCOPE)
        set(CMAKE_C_COMPILER "mpicc" PARENT_SCOPE)
        add_compile_definitions(SPPARKS_GZIP)
        
    else()
        message(STATUS "Unknown machine configuration: ${MACHINE_NAME}")
        message(STATUS "Using default settings")
    endif()
endfunction()

# Preset configurations
macro(setup_machine_presets)
    # Create preset options
    set(SPPARKS_MACHINE_PRESETS 
        "mac" "mac_mpi" "mac_arm" "linux" "mpi" "serial" "debug" "mpi_debug"
        CACHE STRING "Available machine presets")
    
    # Allow user to select a machine preset
    set(SPPARKS_MACHINE "" CACHE STRING "Machine configuration preset to use")
    
    if(SPPARKS_MACHINE)
        apply_machine_config("${SPPARKS_MACHINE}")
    endif()
endmacro()

# Function to show available machine configurations
function(show_machine_configs)
    message(STATUS "Available machine configurations:")
    message(STATUS "  mac        - macOS, no MPI, c++")
    message(STATUS "  mac_mpi    - macOS with MPI, mpicxx")
    message(STATUS "  mac_arm    - Apple Silicon Mac, no MPI")
    message(STATUS "  linux      - Generic Linux, g++")
    message(STATUS "  mpi        - Generic MPI, mpicxx")
    message(STATUS "  serial     - Serial build, g++")
    message(STATUS "  debug      - Debug build, g++")
    message(STATUS "  mpi_debug  - MPI Debug build, mpicxx")
    message(STATUS "")
    message(STATUS "Usage: cmake -DSPPARKS_MACHINE=mac ..")
endfunction()

# Auto-detect machine if not specified
function(auto_detect_machine)
    if(NOT SPPARKS_MACHINE)
        if(APPLE)
            # Check if this is Apple Silicon
            execute_process(
                COMMAND uname -m
                OUTPUT_VARIABLE MACHINE_ARCH
                OUTPUT_STRIP_TRAILING_WHITESPACE
            )
            if(MACHINE_ARCH STREQUAL "arm64")
                set(SPPARKS_MACHINE "mac_arm" PARENT_SCOPE)
                message(STATUS "Auto-detected: Apple Silicon Mac")
            else()
                set(SPPARKS_MACHINE "mac" PARENT_SCOPE)
                message(STATUS "Auto-detected: Intel Mac")
            endif()
        elseif(UNIX)
            # Deliberately do NOT select the "linux" config here. It forces
            # SPPARKS_ENABLE_MPI=OFF, which silently discarded an explicit
            # -DSPPARKS_ENABLE_MPI=ON on every non-GPU Linux build: the cache kept
            # reporting ON while the build was serial, and additive/texture then
            # vanished from the style headers with no message naming the cause.
            # Its other settings are inert (CMAKE_CXX_COMPILER is fixed by project()
            # long before this runs) or redundant (SPPARKS_GZIP is handled in
            # CMakeLists.txt), so skipping it costs nothing.
            message(STATUS "Auto-detected: Linux (no machine config applied; "
                           "pass -DSPPARKS_MACHINE=<preset> to force one)")
        else()
            message(STATUS "Could not auto-detect machine type")
        endif()
    endif()
endfunction()

# HPC-specific configurations
function(setup_hpc_configs)
    # Add HPC system configurations
    if(MACHINE_NAME STREQUAL "redsky")
        # Sandia Red Sky
        set(CMAKE_CXX_COMPILER "mpicxx" PARENT_SCOPE)
        set(CMAKE_C_COMPILER "mpicc" PARENT_SCOPE)
        add_compile_definitions(SPPARKS_GZIP)
        
    elseif(MACHINE_NAME STREQUAL "chama")
        # Sandia Chama
        set(CMAKE_CXX_COMPILER "mpicxx" PARENT_SCOPE)
        set(CMAKE_C_COMPILER "mpicc" PARENT_SCOPE)
        add_compile_definitions(SPPARKS_GZIP)
        
    elseif(MACHINE_NAME STREQUAL "skybridge")
        # Sandia SkyBridge
        set(CMAKE_CXX_COMPILER "mpicxx" PARENT_SCOPE)
        set(CMAKE_C_COMPILER "mpicc" PARENT_SCOPE)
        add_compile_definitions(SPPARKS_GZIP)
    endif()
endfunction()