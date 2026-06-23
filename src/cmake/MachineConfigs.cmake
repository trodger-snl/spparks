# Machine-specific configurations for SPPARKS
# Equivalent to the MAKE/Makefile.* files

# Function to apply machine-specific settings
#
# NOTE: The C/C++ compiler is chosen by project()/enable_language(), which runs before
# this function is called. Setting CMAKE_CXX_COMPILER here would have no effect (CMake
# does not re-run toolchain detection mid-configure), so we no longer do it. MPI is
# provided through find_package(MPI) + the MPI::MPI_CXX imported target (see
# CMakeLists.txt), which is the modern approach and does not require swapping the
# compiler for mpicxx. To force a specific compiler, pass -DCMAKE_CXX_COMPILER=... (or
# set CC/CXX) on the cmake command line before the first configure.
function(apply_machine_config MACHINE_NAME)
    message(STATUS "Applying machine configuration: ${MACHINE_NAME}")

    if(MACHINE_NAME STREQUAL "mac")
        # macOS configuration (no MPI)
        set(SPPARKS_ENABLE_MPI OFF PARENT_SCOPE)
        add_compile_definitions(SPPARKS_GZIP)

    elseif(MACHINE_NAME STREQUAL "mac_mpi")
        # macOS with MPI
        set(SPPARKS_ENABLE_MPI ON PARENT_SCOPE)
        add_compile_definitions(SPPARKS_GZIP)

    elseif(MACHINE_NAME STREQUAL "mac_arm")
        # Apple Silicon Mac with MPI (matches Makefile.mac_arm)
        set(SPPARKS_ENABLE_MPI ON PARENT_SCOPE)

        # NOTE: JPEG defaults ON (see option() in CMakeLists.txt) and is auto-detected,
        # so it does not need forcing here. HDF5 is opt-in: request it with --hdf5
        # (i.e. -DSPPARKS_ENABLE_HDF5=ON). Feature detection currently runs *before*
        # this machine-config step, so forcing SPPARKS_ENABLE_HDF5 here would set the
        # flag but would not trigger find_package(HDF5) and therefore would not actually
        # compile in HDF5 support -- which is why the previous force block was removed.
        add_compile_definitions(SPPARKS_GZIP SPPARKS_UNORDERED_MAP)

        # Set Homebrew paths for Apple Silicon
        if(EXISTS "/opt/homebrew")
            set(CMAKE_PREFIX_PATH "/opt/homebrew" PARENT_SCOPE)
        endif()

    elseif(MACHINE_NAME STREQUAL "linux")
        # Generic Linux
        set(SPPARKS_ENABLE_MPI OFF PARENT_SCOPE)
        add_compile_definitions(SPPARKS_GZIP)

    elseif(MACHINE_NAME STREQUAL "mpi")
        # Generic MPI
        set(SPPARKS_ENABLE_MPI ON PARENT_SCOPE)
        add_compile_definitions(SPPARKS_GZIP SPPARKS_UNORDERED_MAP)

    elseif(MACHINE_NAME STREQUAL "serial")
        # Serial build
        set(SPPARKS_ENABLE_MPI OFF PARENT_SCOPE)
        add_compile_definitions(SPPARKS_GZIP)

    elseif(MACHINE_NAME STREQUAL "debug")
        # Debug build
        set(CMAKE_BUILD_TYPE "Debug" PARENT_SCOPE)
        add_compile_definitions(SPPARKS_GZIP)

    elseif(MACHINE_NAME STREQUAL "mpi_debug")
        # MPI Debug build
        set(CMAKE_BUILD_TYPE "Debug" PARENT_SCOPE)
        set(SPPARKS_ENABLE_MPI ON PARENT_SCOPE)
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
    message(STATUS "  mac_arm    - Apple Silicon Mac, with MPI")
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
            # Respect explicit MPI setting from command line (--mpi flag)
            if(SPPARKS_ENABLE_MPI)
                set(SPPARKS_MACHINE "mpi" PARENT_SCOPE)
                message(STATUS "Auto-detected: Linux with MPI")
            else()
                set(SPPARKS_MACHINE "linux" PARENT_SCOPE)
                message(STATUS "Auto-detected: Linux")
            endif()
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