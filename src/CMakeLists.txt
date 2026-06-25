cmake_minimum_required(VERSION 3.22)  # Kokkos >= 5.0 (bundled submodule) requires CMake >= 3.22

# Set project name and version
project(SPPARKS 
    VERSION 0.1.0
    DESCRIPTION "Stochastic Parallel PARticle Kinetic Simulator"
    LANGUAGES CXX C)

# Set C++ standard
# Kokkos >= 5.0 requires C++20; older SPPARKS/Kokkos 4.x built with C++17.
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Compiler version validation for C++17 support
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 7.0)
        message(WARNING "GCC ${CMAKE_CXX_COMPILER_VERSION} may not fully support C++17. "
                        "GCC 7.0 or later is recommended. Build may fail.")
    endif()
elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang|AppleClang")
    if(CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
        # Apple Clang 11+ (Xcode 11+) supports C++17
        if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 11.0)
            message(WARNING "Apple Clang ${CMAKE_CXX_COMPILER_VERSION} may not fully support C++17. "
                            "Apple Clang 11.0 (Xcode 11) or later is recommended. Build may fail.")
        endif()
    else()
        # Regular Clang 5+ supports C++17
        if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 5.0)
            message(WARNING "Clang ${CMAKE_CXX_COMPILER_VERSION} may not fully support C++17. "
                            "Clang 5.0 or later is recommended. Build may fail.")
        endif()
    endif()
endif()

# Set default build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Enhanced optimization flags for different build types
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    # Start with base optimization level
    set(OPT_FLAGS "-DNDEBUG")
    
    if(SPPARKS_AGGRESSIVE_OPTIMIZATION)
        set(OPT_FLAGS "${OPT_FLAGS} -O3")
        message(STATUS "Using aggressive optimization (-O3)")
    else()
        set(OPT_FLAGS "${OPT_FLAGS} -O2")
        message(STATUS "Using standard optimization (-O2)")
    endif()
    
    # Add native optimizations if enabled
    if(SPPARKS_NATIVE_OPTIMIZATION)
        set(OPT_FLAGS "${OPT_FLAGS} -march=native -mtune=native")
        message(STATUS "Enabled native CPU optimizations (-march=native)")
    endif()
    
    # Add Link Time Optimization (LTO) for better performance
    if(SPPARKS_AGGRESSIVE_OPTIMIZATION)
        if(APPLE AND CMAKE_SYSTEM_PROCESSOR MATCHES "arm64")
            # Apple Silicon specific LTO
            set(OPT_FLAGS "${OPT_FLAGS} -flto=thin")
            message(STATUS "Apple Silicon: Enabled thin LTO")
        elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
            # GCC LTO
            set(OPT_FLAGS "${OPT_FLAGS} -flto")
            message(STATUS "GCC: Enabled LTO")
        elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
            # Clang LTO
            set(OPT_FLAGS "${OPT_FLAGS} -flto")
            message(STATUS "Clang: Enabled LTO")
        endif()
    endif()
    
    # Additional performance flags
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|AppleClang")
        set(OPT_FLAGS "${OPT_FLAGS} -ffast-math -funroll-loops")
    elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        set(OPT_FLAGS "${OPT_FLAGS} -ffast-math -funroll-loops -fomit-frame-pointer")
    endif()

    # Note: OPT_FLAGS will be applied per-target, not globally
    # This provides better build isolation and prevents flag pollution
    message(STATUS "Release optimization flags: ${OPT_FLAGS}")
    
elseif(CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
    # Optimized with debug info
    set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2 -g -DNDEBUG")
    message(STATUS "RelWithDebInfo: Using -O2 -g optimizations")
endif()

# Automatic CPU detection and parallel build optimization
include(ProcessorCount)
ProcessorCount(N)
if(NOT N EQUAL 0)
    # On Apple Silicon, use all available cores for compilation
    if(APPLE AND CMAKE_SYSTEM_PROCESSOR MATCHES "arm64")
        message(STATUS "Apple Silicon detected: Using all ${N} cores for parallel compilation")
        set(CMAKE_BUILD_PARALLEL_LEVEL ${N} CACHE STRING "Number of parallel jobs")
        # Set make flags for parallel builds
        set(CMAKE_MAKE_PROGRAM_ARGS "-j${N}" CACHE STRING "Make parallel arguments")
    else()
        # For other systems, use most cores but leave one free
        math(EXPR PARALLEL_JOBS "${N} - 1")
        if(PARALLEL_JOBS LESS 1)
            set(PARALLEL_JOBS 1)
        endif()
        message(STATUS "Using ${PARALLEL_JOBS} of ${N} available cores for parallel compilation")
        set(CMAKE_BUILD_PARALLEL_LEVEL ${PARALLEL_JOBS} CACHE STRING "Number of parallel jobs")
        set(CMAKE_MAKE_PROGRAM_ARGS "-j${PARALLEL_JOBS}" CACHE STRING "Make parallel arguments")
    endif()
else()
    message(WARNING "Could not detect CPU count, using single-threaded build")
    set(CMAKE_BUILD_PARALLEL_LEVEL 1 CACHE STRING "Number of parallel jobs")
endif()

# WSL2 Detection (Windows Subsystem for Linux)
if(UNIX AND NOT APPLE)
    execute_process(
        COMMAND uname -r
        OUTPUT_VARIABLE UNAME_R
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
    if(UNAME_R MATCHES "microsoft|WSL")
        set(WSL_BUILD ON CACHE BOOL "Building in Windows Subsystem for Linux")
        message(STATUS "Windows Subsystem for Linux (WSL2) detected")
        message(STATUS "  Using Linux build configuration within WSL environment")
    endif()
endif()

# Build options
option(BUILD_SHARED_LIBS "Build shared libraries" OFF)
option(SPPARKS_BUILD_EXECUTABLE "Build spparks executable" ON)
option(SPPARKS_BUILD_LIB "Build spparks library" OFF)
option(SPPARKS_ENABLE_MPI "Enable MPI support" ON)
option(SPPARKS_ENABLE_GZIP "Enable GZIP support" ON)
option(SPPARKS_ENABLE_JPEG "Enable JPEG support" ON)
option(SPPARKS_ENABLE_PNG "Enable PNG support" OFF)
option(SPPARKS_ENABLE_HDF5 "Enable HDF5 support" OFF)
option(SPPARKS_ENABLE_HIGHFIVE "Enable HighFive C++ HDF5 wrapper" OFF)
option(SPPARKS_USE_EXTERNAL_HIGHFIVE "Use system-provided HighFive instead of FetchContent" OFF)
option(SPPARKS_UNORDERED_MAP "Use unordered_map" OFF)

# Kokkos options
option(SPPARKS_PACKAGE_KOKKOS "Enable KOKKOS package for GPU acceleration" OFF)
set(SPPARKS_KOKKOS_BACKEND "OPENMP" CACHE STRING "Kokkos backend (OPENMP, CUDA, HIP, SYCL, SERIAL)")
set_property(CACHE SPPARKS_KOKKOS_BACKEND PROPERTY STRINGS OPENMP CUDA HIP SYCL SERIAL)
# GPU architecture for the CUDA/HIP backends. Kokkos does NOT auto-detect this,
# so it must be set explicitly or the device build silently produces no GPU code.
# Examples: AMD_GFX1151 (Strix Halo / Radeon 8050S/8060S, RDNA 3.5) for HIP,
#           AMPERE80 (A100) or ADA89 (RTX 40xx) for CUDA.
set(SPPARKS_KOKKOS_ARCH "" CACHE STRING "Kokkos GPU architecture token (e.g. AMD_GFX1151, AMPERE80)")

# Performance options
option(SPPARKS_AGGRESSIVE_OPTIMIZATION "Enable aggressive optimizations (may increase compile time)" ON)
option(SPPARKS_NATIVE_OPTIMIZATION "Enable native CPU optimizations (-march=native)" ON)

# Integer size options
option(SPPARKS_SMALLSMALL "Use 32-bit for all integers" OFF)
option(SPPARKS_BIGBIG "Use 64-bit tagint and bigint" OFF)
# Default is SPPARKS_SMALLBIG (32-bit smallint/tagint, 64-bit bigint)

# Configure integer types
if(SPPARKS_SMALLSMALL)
    add_compile_definitions(SPPARKS_SMALLSMALL)
elseif(SPPARKS_BIGBIG)
    add_compile_definitions(SPPARKS_BIGBIG)
else()
    add_compile_definitions(SPPARKS_SMALLBIG)
endif()

# Configure features
if(SPPARKS_ENABLE_GZIP)
    add_compile_definitions(SPPARKS_GZIP)
endif()

if(SPPARKS_UNORDERED_MAP)
    add_compile_definitions(SPPARKS_UNORDERED_MAP)
endif()

# Move MPI configuration after machine config application

if(SPPARKS_ENABLE_JPEG)
    # First try standard CMake FindJPEG
    find_package(JPEG QUIET)
    
    if(NOT JPEG_FOUND)
        # Try manual detection for common paths (especially Homebrew on macOS)
        find_path(JPEG_INCLUDE_DIR_MANUAL jpeglib.h
            PATHS /opt/homebrew/include /usr/local/include
            NO_DEFAULT_PATH
        )
        find_library(JPEG_LIBRARY_MANUAL jpeg
            PATHS /opt/homebrew/lib /usr/local/lib
            NO_DEFAULT_PATH
        )
        
        if(JPEG_INCLUDE_DIR_MANUAL AND JPEG_LIBRARY_MANUAL)
            set(JPEG_FOUND TRUE)
            set(JPEG_INCLUDE_DIR ${JPEG_INCLUDE_DIR_MANUAL})
            set(JPEG_LIBRARIES ${JPEG_LIBRARY_MANUAL})
            message(STATUS "Found JPEG library (manual): ${JPEG_LIBRARIES}")
            message(STATUS "JPEG include dir (manual): ${JPEG_INCLUDE_DIR}")
        endif()
    else()
        message(STATUS "Found JPEG library: ${JPEG_LIBRARIES}")
        message(STATUS "JPEG include dir: ${JPEG_INCLUDE_DIR}")
    endif()
    
    if(JPEG_FOUND)
        add_compile_definitions(SPPARKS_JPEG)
        message(STATUS "JPEG support enabled")
    else()
        message(STATUS "JPEG library not found - disabling JPEG support")
        message(STATUS "  To install: brew install jpeg (macOS) or apt-get install libjpeg-dev (Ubuntu)")
        set(SPPARKS_ENABLE_JPEG OFF)
    endif()
endif()

if(SPPARKS_ENABLE_PNG)
    find_package(PNG)
    if(PNG_FOUND)
        add_compile_definitions(SPPARKS_PNG)
        message(STATUS "PNG support enabled")
    else()
        message(STATUS "PNG library not found - disabling PNG support")
        message(STATUS "  To install: brew install libpng (macOS) or apt-get install libpng-dev (Ubuntu)")
        set(SPPARKS_ENABLE_PNG OFF)
    endif()
endif()

if(SPPARKS_ENABLE_HDF5)
    # Add cmake module path for enhanced HDF5 finder
    list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
    
    find_package(HDF5Enhanced REQUIRED)
    if(HDF5_FOUND)
        add_compile_definitions(SPPARKS_HDF)
        # Check if HDF5 is parallel (MPI-enabled)
        if(HDF5_IS_PARALLEL AND SPPARKS_ENABLE_MPI)
            message(STATUS "Found parallel HDF5 (MPI-enabled)")
            add_compile_definitions(H5_HAVE_PARALLEL)
        elseif(SPPARKS_ENABLE_MPI AND NOT HDF5_IS_PARALLEL)
            message(WARNING "MPI is enabled but HDF5 is not parallel. Consider using parallel HDF5 for better performance.")
        endif()
    else()
        message(FATAL_ERROR "HDF5 was requested but not found")
    endif()
endif()

if(SPPARKS_ENABLE_HIGHFIVE)
    # Ensure HDF5 is also enabled for HighFive
    if(NOT SPPARKS_ENABLE_HDF5)
        message(STATUS "HighFive requires HDF5 - enabling HDF5 automatically")
        set(SPPARKS_ENABLE_HDF5 ON CACHE BOOL "Enable HDF5 support" FORCE)

        # Add cmake module path for enhanced HDF5 finder
        list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

        find_package(HDF5Enhanced REQUIRED)
        if(HDF5_FOUND)
            add_compile_definitions(SPPARKS_HDF)
            # Check if HDF5 is parallel (MPI-enabled)
            if(HDF5_IS_PARALLEL AND SPPARKS_ENABLE_MPI)
                message(STATUS "Found parallel HDF5 (MPI-enabled)")
                add_compile_definitions(H5_HAVE_PARALLEL)
            elseif(SPPARKS_ENABLE_MPI AND NOT HDF5_IS_PARALLEL)
                message(WARNING "MPI is enabled but HDF5 is not parallel. Consider using parallel HDF5 for better performance.")
            endif()
        else()
            message(FATAL_ERROR "HDF5 was requested but not found")
        endif()
    endif()

    # Use external HighFive or fetch from source
    if(SPPARKS_USE_EXTERNAL_HIGHFIVE)
        # Try to find system-provided HighFive (e.g., from Spack, Homebrew, or system package manager)
        # Add Spack installation paths as hints for find_package (cross-platform)
        file(GLOB SPACK_HIGHFIVE_DIRS
            # macOS Homebrew Spack
            "/opt/homebrew/Cellar/spack/*/opt/spack/*/highfive-*"
            "/opt/homebrew/opt/spack/opt/spack/*/highfive-*"
            # Linux/HPC common Spack locations
            "/opt/spack/opt/spack/*/highfive-*"
            "/usr/local/spack/opt/spack/*/highfive-*"
            "$ENV{HOME}/.spack/opt/spack/*/highfive-*"
            "/spack/opt/spack/*/highfive-*"
        )
        if(SPACK_HIGHFIVE_DIRS)
            list(APPEND CMAKE_PREFIX_PATH ${SPACK_HIGHFIVE_DIRS})
            message(STATUS "Added Spack HighFive paths to search: ${SPACK_HIGHFIVE_DIRS}")
        endif()

        find_package(HighFive QUIET CONFIG)

        if(HighFive_FOUND)
            message(STATUS "Using system-provided HighFive")
        else()
            message(FATAL_ERROR "SPPARKS_USE_EXTERNAL_HIGHFIVE is ON but HighFive was not found. "
                               "Please install HighFive or set SPPARKS_USE_EXTERNAL_HIGHFIVE=OFF to use FetchContent.")
        endif()
    else()
        # Fetch HighFive using FetchContent
        message(STATUS "Fetching HighFive from GitHub")
        include(FetchContent)

        # CMake 4.x compatibility: HighFive requires CMake 3.1 minimum, which is incompatible with CMake 4.0+
        # Set policy version minimum to 3.5 as a workaround for older HighFive CMakeLists.txt
        if(CMAKE_VERSION VERSION_GREATER_EQUAL "4.0")
            set(CMAKE_POLICY_VERSION_MINIMUM 3.5 CACHE STRING "Minimum policy version for FetchContent compatibility")
            message(STATUS "CMake 4.x detected - applying policy workaround for HighFive compatibility")
        endif()

        FetchContent_Declare(
            HighFive
            GIT_REPOSITORY https://github.com/BlueBrain/HighFive.git
            GIT_TAG        v2.10.1  # Latest stable version - CMake 4.x compatible
            GIT_SHALLOW    TRUE
        )

        # Configure HighFive options
        set(HIGHFIVE_EXAMPLES OFF CACHE BOOL "Enable HighFive examples")
        set(HIGHFIVE_UNIT_TESTS OFF CACHE BOOL "Enable HighFive unit tests")
        set(HIGHFIVE_USE_BOOST OFF CACHE BOOL "Enable Boost support in HighFive")
        set(HIGHFIVE_USE_EIGEN OFF CACHE BOOL "Enable Eigen support in HighFive")
        set(HIGHFIVE_USE_XTENSOR OFF CACHE BOOL "Enable xtensor support in HighFive")
        set(HIGHFIVE_USE_OPENCV OFF CACHE BOOL "Enable OpenCV support in HighFive")

        # Make HighFive available
        FetchContent_MakeAvailable(HighFive)
    endif()

    add_compile_definitions(SPPARKS_HIGHFIVE)
    message(STATUS "HighFive C++ HDF5 wrapper enabled")
endif()

# Configure Kokkos if enabled
if(SPPARKS_PACKAGE_KOKKOS)
    message(STATUS "Kokkos package enabled")
    message(STATUS "Kokkos backend: ${SPPARKS_KOKKOS_BACKEND}")

    # Set Kokkos options before adding subdirectory
    # Enable the selected backend
    if(SPPARKS_KOKKOS_BACKEND STREQUAL "OPENMP")
        set(Kokkos_ENABLE_OPENMP ON CACHE BOOL "Enable OpenMP backend" FORCE)
        set(Kokkos_ENABLE_SERIAL OFF CACHE BOOL "Disable Serial backend" FORCE)
        message(STATUS "Kokkos: OpenMP backend enabled (CPU multi-threading)")
    elseif(SPPARKS_KOKKOS_BACKEND STREQUAL "CUDA")
        set(Kokkos_ENABLE_CUDA ON CACHE BOOL "Enable CUDA backend" FORCE)
        set(Kokkos_ENABLE_SERIAL ON CACHE BOOL "Enable Serial for host execution" FORCE)
        if(NOT SPPARKS_KOKKOS_ARCH)
            message(FATAL_ERROR "CUDA backend requires SPPARKS_KOKKOS_ARCH (e.g. AMPERE80, ADA89)")
        endif()
        set(Kokkos_ARCH_${SPPARKS_KOKKOS_ARCH} ON CACHE BOOL "Kokkos GPU arch" FORCE)
        message(STATUS "Kokkos: CUDA backend enabled (NVIDIA GPUs), arch ${SPPARKS_KOKKOS_ARCH}")
    elseif(SPPARKS_KOKKOS_BACKEND STREQUAL "HIP")
        set(Kokkos_ENABLE_HIP ON CACHE BOOL "Enable HIP backend" FORCE)
        set(Kokkos_ENABLE_SERIAL ON CACHE BOOL "Enable Serial for host execution" FORCE)
        # AMD GPUs are NOT auto-detected; the gfx target must be set explicitly.
        if(NOT SPPARKS_KOKKOS_ARCH)
            message(FATAL_ERROR "HIP backend requires SPPARKS_KOKKOS_ARCH (e.g. AMD_GFX1151)")
        endif()
        set(Kokkos_ARCH_${SPPARKS_KOKKOS_ARCH} ON CACHE BOOL "Kokkos GPU arch" FORCE)
        message(STATUS "Kokkos: HIP backend enabled (AMD GPUs), arch ${SPPARKS_KOKKOS_ARCH}")
    elseif(SPPARKS_KOKKOS_BACKEND STREQUAL "SYCL")
        set(Kokkos_ENABLE_SYCL ON CACHE BOOL "Enable SYCL backend" FORCE)
        set(Kokkos_ENABLE_SERIAL ON CACHE BOOL "Enable Serial for host execution" FORCE)
        message(STATUS "Kokkos: SYCL backend enabled (Intel GPUs)")
    elseif(SPPARKS_KOKKOS_BACKEND STREQUAL "SERIAL")
        set(Kokkos_ENABLE_SERIAL ON CACHE BOOL "Enable Serial backend" FORCE)
        message(STATUS "Kokkos: Serial backend enabled (single-threaded)")
    else()
        message(FATAL_ERROR "Unknown Kokkos backend: ${SPPARKS_KOKKOS_BACKEND}")
    endif()

    # Disable backends we don't want
    if(NOT SPPARKS_KOKKOS_BACKEND STREQUAL "OPENMP")
        set(Kokkos_ENABLE_OPENMP OFF CACHE BOOL "Disable OpenMP backend" FORCE)
    endif()
    if(NOT SPPARKS_KOKKOS_BACKEND STREQUAL "CUDA")
        set(Kokkos_ENABLE_CUDA OFF CACHE BOOL "Disable CUDA backend" FORCE)
    endif()
    if(NOT SPPARKS_KOKKOS_BACKEND STREQUAL "HIP")
        set(Kokkos_ENABLE_HIP OFF CACHE BOOL "Disable HIP backend" FORCE)
    endif()
    if(NOT SPPARKS_KOKKOS_BACKEND STREQUAL "SYCL")
        set(Kokkos_ENABLE_SYCL OFF CACHE BOOL "Disable SYCL backend" FORCE)
    endif()
    # Keep Serial enabled for GPU backends (required for host execution)
    # Only disable Serial for OpenMP backend (OpenMP provides host parallel execution)
    if(SPPARKS_KOKKOS_BACKEND STREQUAL "OPENMP")
        set(Kokkos_ENABLE_SERIAL OFF CACHE BOOL "Disable Serial backend" FORCE)
    endif()

    # Disable Kokkos tests and examples
    set(Kokkos_ENABLE_TESTS OFF CACHE BOOL "Disable Kokkos tests" FORCE)
    set(Kokkos_ENABLE_EXAMPLES OFF CACHE BOOL "Disable Kokkos examples" FORCE)
    set(Kokkos_ENABLE_BENCHMARKS OFF CACHE BOOL "Disable Kokkos benchmarks" FORCE)

    # Check if bundled Kokkos exists
    set(KOKKOS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../lib/kokkos")
    if(NOT EXISTS "${KOKKOS_DIR}/CMakeLists.txt")
        message(FATAL_ERROR
            "Kokkos submodule not found at ${KOKKOS_DIR}\n"
            "Please initialize the submodule:\n"
            "  git submodule update --init --recursive\n")
    endif()

    # Kokkos 4.7.01 does not recognize RDNA 3.5 (gfx1150/gfx1151, e.g. AMD Strix
    # Halo / Radeon 8050S-8060S APUs). Apply a small bundled patch that registers
    # those archs and sets the correct 32-wide wavefront size. Idempotent: skipped
    # if the submodule already supports gfx1151 (e.g. a newer Kokkos).
    if(SPPARKS_KOKKOS_ARCH MATCHES "^AMD_GFX115")
        file(READ "${KOKKOS_DIR}/cmake/kokkos_arch.cmake" _kokkos_arch_contents)
        if(NOT _kokkos_arch_contents MATCHES "GFX1151")
            find_package(Git REQUIRED)
            set(_gfx1151_patch "${CMAKE_CURRENT_SOURCE_DIR}/../patches/kokkos-gfx1151.patch")
            message(STATUS "Applying RDNA 3.5 (gfx1150/gfx1151) support patch to bundled Kokkos")
            execute_process(
                COMMAND ${GIT_EXECUTABLE} apply "${_gfx1151_patch}"
                WORKING_DIRECTORY "${KOKKOS_DIR}"
                RESULT_VARIABLE _patch_result
                ERROR_VARIABLE _patch_error)
            if(NOT _patch_result EQUAL 0)
                message(FATAL_ERROR
                    "Failed to apply gfx1151 Kokkos patch (${_gfx1151_patch}):\n${_patch_error}"
                    "\nApply it manually with: cd ${KOKKOS_DIR} && git apply ${_gfx1151_patch}")
            endif()
        else()
            message(STATUS "Bundled Kokkos already supports gfx1151; skipping patch")
        endif()
    endif()

    # Add bundled Kokkos
    message(STATUS "Using bundled Kokkos from: ${KOKKOS_DIR}")
    add_subdirectory(${KOKKOS_DIR} kokkos)

    # Add Kokkos compile definition
    add_compile_definitions(SPK_KOKKOS)
    message(STATUS "Kokkos configured successfully")
endif()

# Include helper modules
include(cmake/MachineConfigs.cmake)
include(cmake/PackageManager.cmake)

# Apply machine configuration if specified
setup_machine_presets()
setup_hpc_configs()  # Enable HPC system configurations
auto_detect_machine()

if(SPPARKS_MACHINE)
    apply_machine_config("${SPPARKS_MACHINE}")
endif()

# Command line options should override machine config
# (This happens automatically since CMake cache variables take precedence)

# Configure MPI after machine configuration
# Note: This ordering is intentional - machine configs can set SPPARKS_ENABLE_MPI,
# but the actual find_package(MPI) happens here to use the updated settings.
# This allows machine presets to control MPI enablement while keeping the
# MPI detection logic centralized.
if(SPPARKS_ENABLE_MPI)
    find_package(MPI REQUIRED)
    if(MPI_CXX_FOUND)
        add_compile_definitions(SPPARKS_MPI)
    endif()
else()
    # Use MPI stubs for serial builds
    set(MPI_STUB_DIR "${CMAKE_CURRENT_SOURCE_DIR}/STUBS")
    if(NOT EXISTS "${MPI_STUB_DIR}/libmpi.a")
        message(STATUS "Building MPI stubs...")
        execute_process(
            COMMAND make clean
            WORKING_DIRECTORY ${MPI_STUB_DIR}
            RESULT_VARIABLE CLEAN_RESULT
        )
        execute_process(
            COMMAND make
            WORKING_DIRECTORY ${MPI_STUB_DIR}
            RESULT_VARIABLE BUILD_RESULT
        )
        if(NOT BUILD_RESULT EQUAL 0)
            message(FATAL_ERROR "Failed to build MPI stubs")
        endif()
    endif()
    
    # Create MPI stub library target
    add_library(mpi_stubs STATIC IMPORTED)
    set_target_properties(mpi_stubs PROPERTIES
        IMPORTED_LOCATION "${MPI_STUB_DIR}/libmpi.a"
        INTERFACE_INCLUDE_DIRECTORIES "${MPI_STUB_DIR}"
    )
    message(STATUS "Using MPI stubs from: ${MPI_STUB_DIR}/libmpi.a")
    # Add to global include directories as fallback
    include_directories("${MPI_STUB_DIR}")
endif()

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# Source files (base list - conditional sources added below)
set(SPPARKS_SOURCES
    app.cpp
    app_am_ellipsoid.cpp
    app_chemistry.cpp
    app_diffusion.cpp
    app_diffusion_multiphase.cpp
    app_erbium.cpp
    app_ising.cpp
    app_ising_single.cpp
    app_lattice.cpp
    app_membrane.cpp
    app_off_lattice.cpp
    app_phasefield_potts.cpp
    app_potts.cpp
    app_potts_am_bezier.cpp
    app_potts_am_path_gen.cpp
    app_potts_am_weld.cpp
    app_potts_grad.cpp
    app_potts_neigh.cpp
    app_potts_neighonly.cpp
    app_potts_pin.cpp
    app_potts_quaternion.cpp
    app_potts_strain.cpp
    app_potts_strain_pin.cpp
    app_potts_weld.cpp
    app_potts_weld_jom.cpp
    app_relax.cpp
    app_sinter.cpp
    app_sos.cpp
    app_test_group.cpp
    am_bezier.cpp
    cluster.cpp
    comm_lattice.cpp
    comm_off_lattice.cpp
    create_box.cpp
    create_sites.cpp
    diag.cpp
    diag_array.cpp
    diag_cluster.cpp
    diag_diffusion.cpp
    diag_energy.cpp
    diag_erbium.cpp
    diag_propensity.cpp
    diag_sinter_density.cpp
    diag_sinter_free_energy.cpp
    diag_sinter_free_energy_pore.cpp
    domain.cpp
    dump.cpp
    dump_sites.cpp
    dump_text.cpp
    dump_vtk.cpp
    error.cpp
    finish.cpp
    fourth_order_bezier.cpp
    groups.cpp
    input.cpp
    irregular.cpp
    lattice.cpp
    library.cpp
    math_extra.cpp
    memory.cpp
    output.cpp
    pair.cpp
    pair_lj_cut.cpp
    potential.cpp
    potts_am_path_parser.cpp
    random_mars.cpp
    random_park.cpp
    read_sites.cpp
    region.cpp
    region_block.cpp
    region_cylinder.cpp
    region_intersect.cpp
    region_sphere.cpp
    region_union.cpp
    set.cpp
    shell.cpp
    solve.cpp
    solve_group.cpp
    solve_linear.cpp
    solve_tree.cpp
    spparks.cpp
    timer.cpp
    universe.cpp
    variable.cpp
)

# Configure conditional sources and generate style headers after all dependencies are resolved
include(cmake/ConditionalSources.cmake)
configure_conditional_sources()

# Add package-specific sources
if(SPPARKS_PACKAGE_STITCH AND STITCH_LIBRARY)
    list(APPEND SPPARKS_SOURCES STITCH/dump_stitch.cpp)
    message(STATUS "Added STITCH/dump_stitch.cpp to sources")
elseif(SPPARKS_PACKAGE_STITCH)
    message(WARNING "STITCH package enabled but library not found at '${STITCH_LIBRARY}' - dump_stitch not added to sources")
endif()

# Generate style headers after all dependencies are resolved
include(cmake/generate_styles.cmake)
generate_style_headers()

# Header files for style generation
file(GLOB SPPARKS_HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/*.h")

# Create library targets
if(SPPARKS_BUILD_LIB OR NOT SPPARKS_BUILD_EXECUTABLE)
    add_library(spparks_lib ${SPPARKS_SOURCES})
    set_target_properties(spparks_lib PROPERTIES
        OUTPUT_NAME "spparks"
        VERSION ${PROJECT_VERSION}
        SOVERSION ${PROJECT_VERSION_MAJOR}
    )
    
    # Link dependencies to library
    if(SPPARKS_ENABLE_MPI AND MPI_CXX_FOUND)
        target_link_libraries(spparks_lib PUBLIC MPI::MPI_CXX)
    elseif(NOT SPPARKS_ENABLE_MPI)
        target_link_libraries(spparks_lib PUBLIC mpi_stubs)
    endif()
    
    if(SPPARKS_ENABLE_JPEG AND JPEG_FOUND)
        target_link_libraries(spparks_lib PUBLIC ${JPEG_LIBRARIES})
        target_include_directories(spparks_lib PUBLIC ${JPEG_INCLUDE_DIR})
    endif()
    
    if(SPPARKS_ENABLE_PNG AND PNG_FOUND)
        target_link_libraries(spparks_lib PUBLIC PNG::PNG)
    endif()
    
    if(SPPARKS_ENABLE_HDF5 AND HDF5_FOUND)
        target_link_libraries(spparks_lib PUBLIC ${HDF5_LIBRARIES})
        target_include_directories(spparks_lib PUBLIC ${HDF5_INCLUDE_DIRS})
        target_compile_definitions(spparks_lib PUBLIC ${HDF5_DEFINITIONS})
    endif()
    
    if(SPPARKS_ENABLE_HIGHFIVE)
        target_link_libraries(spparks_lib PUBLIC HighFive)
    endif()

    # Add Kokkos package support
    if(SPPARKS_PACKAGE_KOKKOS)
        target_link_libraries(spparks_lib PUBLIC Kokkos::kokkoscore)
        message(STATUS "Linked Kokkos::kokkoscore to spparks_lib")
    endif()

    # Add STITCH package support
    if(SPPARKS_PACKAGE_STITCH AND STITCH_LIBRARY)
        target_link_libraries(spparks_lib PUBLIC ${STITCH_LIBRARY})
        target_include_directories(spparks_lib PUBLIC ${STITCH_INCLUDE_DIR})
        target_compile_definitions(spparks_lib PUBLIC STITCH_PARALLEL SPK_STITCH)
    endif()

    # Add KOKKOS package support
    if(SPPARKS_PACKAGE_KOKKOS)
        if(TARGET Kokkos::kokkos)
            target_link_libraries(spparks_lib PUBLIC Kokkos::kokkos)
            target_compile_definitions(spparks_lib PUBLIC SPK_KOKKOS)
            message(STATUS "Linked Kokkos::kokkos to spparks_lib for KOKKOS package")
        elseif(TARGET Kokkos::kokkoscore)
            target_link_libraries(spparks_lib PUBLIC Kokkos::kokkoscore)
            target_compile_definitions(spparks_lib PUBLIC SPK_KOKKOS)
            message(STATUS "Linked Kokkos::kokkoscore to spparks_lib for KOKKOS package")
        else()
            message(FATAL_ERROR "KOKKOS package enabled but Kokkos library not found")
        endif()
    endif()

    # Add GZIP support
    if(SPPARKS_ENABLE_GZIP)
        find_package(ZLIB)
        if(ZLIB_FOUND)
            target_link_libraries(spparks_lib PUBLIC ZLIB::ZLIB)
        endif()
    endif()
    
    # Set include directories for library
    target_include_directories(spparks_lib PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<INSTALL_INTERFACE:include>
    )
    
    # Add dependency on style headers and validation
    if(TARGET style_headers)
        add_dependencies(spparks_lib style_headers)
    endif()
    if(TARGET validate_styles)
        add_dependencies(spparks_lib validate_styles)
    endif()
    
    # Add alias for easier linking
    add_library(SPPARKS::spparks ALIAS spparks_lib)
endif()

# Create executable target
if(SPPARKS_BUILD_EXECUTABLE)
    add_executable(spparks main.cpp)
    
    # Add dependency on style headers and validation  
    if(TARGET style_headers)
        add_dependencies(spparks style_headers)
    endif()
    if(TARGET validate_styles)
        add_dependencies(spparks validate_styles)
    endif()
    
    if(TARGET spparks_lib)
        target_link_libraries(spparks spparks_lib)
    else()
        # Link sources directly if not building library
        target_sources(spparks PRIVATE ${SPPARKS_SOURCES})
        
        # Link dependencies to executable
        if(SPPARKS_ENABLE_MPI AND MPI_CXX_FOUND)
            target_link_libraries(spparks MPI::MPI_CXX)
        elseif(NOT SPPARKS_ENABLE_MPI)
            target_link_libraries(spparks mpi_stubs)
        endif()
        
        if(SPPARKS_ENABLE_JPEG AND JPEG_FOUND)
            target_link_libraries(spparks ${JPEG_LIBRARIES})
            target_include_directories(spparks PRIVATE ${JPEG_INCLUDE_DIR})
        endif()
        
        if(SPPARKS_ENABLE_PNG AND PNG_FOUND)
            target_link_libraries(spparks PNG::PNG)
        endif()
        
        if(SPPARKS_ENABLE_HDF5 AND HDF5_FOUND)
            target_link_libraries(spparks ${HDF5_LIBRARIES})
            target_include_directories(spparks PRIVATE ${HDF5_INCLUDE_DIRS})
            target_compile_definitions(spparks PRIVATE ${HDF5_DEFINITIONS})
        endif()
        
        if(SPPARKS_ENABLE_HIGHFIVE)
            target_link_libraries(spparks HighFive)
        endif()

        # Add Kokkos package support
        if(SPPARKS_PACKAGE_KOKKOS)
            target_link_libraries(spparks Kokkos::kokkoscore)
        endif()

        # Add STITCH package support
        if(SPPARKS_PACKAGE_STITCH AND STITCH_LIBRARY)
            target_link_libraries(spparks ${STITCH_LIBRARY})
            target_include_directories(spparks PRIVATE ${STITCH_INCLUDE_DIR})
            target_compile_definitions(spparks PRIVATE STITCH_PARALLEL SPK_STITCH)
        endif()

        # Add KOKKOS package support
        if(SPPARKS_PACKAGE_KOKKOS)
            if(TARGET Kokkos::kokkos)
                target_link_libraries(spparks Kokkos::kokkos)
                target_compile_definitions(spparks PRIVATE SPK_KOKKOS)
                message(STATUS "Linked Kokkos::kokkos to spparks executable for KOKKOS package")
            elseif(TARGET Kokkos::kokkoscore)
                target_link_libraries(spparks Kokkos::kokkoscore)
                target_compile_definitions(spparks PRIVATE SPK_KOKKOS)
                message(STATUS "Linked Kokkos::kokkoscore to spparks executable for KOKKOS package")
            endif()
        endif()

        if(SPPARKS_ENABLE_GZIP)
            find_package(ZLIB)
            if(ZLIB_FOUND)
                target_link_libraries(spparks ZLIB::ZLIB)
            endif()
        endif()
    endif()
endif()

# Compiler flags - applied per-target for better isolation
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        target_compile_options(spparks PRIVATE -g -Wall -Wextra)
        if(TARGET spparks_lib)
            target_compile_options(spparks_lib PRIVATE -g -Wall -Wextra)
        endif()
    else()
        # Apply all optimization flags from OPT_FLAGS (defined earlier)
        # Convert string to list for target_compile_options
        separate_arguments(OPT_FLAGS_LIST UNIX_COMMAND "${OPT_FLAGS}")
        target_compile_options(spparks PRIVATE ${OPT_FLAGS_LIST})
        if(TARGET spparks_lib)
            target_compile_options(spparks_lib PRIVATE ${OPT_FLAGS_LIST})
        endif()
    endif()
endif()

# Installation
if(SPPARKS_BUILD_LIB)
    install(TARGETS spparks_lib
        EXPORT SpparksTargets
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION bin
    )

    # Install public headers
    file(GLOB SPPARKS_PUBLIC_HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/*.h")
    install(FILES ${SPPARKS_PUBLIC_HEADERS}
        DESTINATION include/spparks
    )
endif()

if(SPPARKS_BUILD_EXECUTABLE)
    install(TARGETS spparks
        RUNTIME DESTINATION bin
    )
endif()

# Package configuration
include(CMakePackageConfigHelpers)
if(SPPARKS_BUILD_LIB)
    write_basic_package_version_file(
        "${CMAKE_CURRENT_BINARY_DIR}/SpparksConfigVersion.cmake"
        VERSION ${PROJECT_VERSION}
        COMPATIBILITY AnyNewerVersion
    )
    
    export(EXPORT SpparksTargets
        FILE "${CMAKE_CURRENT_BINARY_DIR}/SpparksTargets.cmake"
        NAMESPACE Spparks::
    )
    
    configure_file(cmake/SpparksConfig.cmake.in SpparksConfig.cmake @ONLY)
    
    install(EXPORT SpparksTargets
        FILE SpparksTargets.cmake
        NAMESPACE Spparks::
        DESTINATION lib/cmake/spparks
    )
    
    install(FILES
        "${CMAKE_CURRENT_BINARY_DIR}/SpparksConfig.cmake"
        "${CMAKE_CURRENT_BINARY_DIR}/SpparksConfigVersion.cmake"
        DESTINATION lib/cmake/spparks
    )
endif()

# Testing support
enable_testing()
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/CMakeLists.txt")
    add_subdirectory(tests)
endif()

# Print configuration summary
message(STATUS "=== SPPARKS Configuration Summary ===")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Build executable: ${SPPARKS_BUILD_EXECUTABLE}")
message(STATUS "Build library: ${SPPARKS_BUILD_LIB}")
message(STATUS "MPI support: ${SPPARKS_ENABLE_MPI}")
message(STATUS "GZIP support: ${SPPARKS_ENABLE_GZIP}")
message(STATUS "JPEG support: ${SPPARKS_ENABLE_JPEG}")
message(STATUS "PNG support: ${SPPARKS_ENABLE_PNG}")
message(STATUS "HDF5 support: ${SPPARKS_ENABLE_HDF5}")
if(SPPARKS_ENABLE_HDF5 AND HDF5_FOUND)
    message(STATUS "HDF5 version: ${HDF5_VERSION}")
    message(STATUS "HDF5 is parallel: ${HDF5_IS_PARALLEL}")
endif()
message(STATUS "HighFive support: ${SPPARKS_ENABLE_HIGHFIVE}")
message(STATUS "Kokkos support: ${SPPARKS_PACKAGE_KOKKOS}")
if(SPPARKS_PACKAGE_KOKKOS)
    message(STATUS "  Kokkos backend: ${SPPARKS_KOKKOS_BACKEND}")
endif()
if(SPPARKS_SMALLSMALL)
    message(STATUS "Integer types: 32-bit (SMALLSMALL)")
elseif(SPPARKS_BIGBIG)
    message(STATUS "Integer types: 64-bit tagint/bigint (BIGBIG)")
else()
    message(STATUS "Integer types: 64-bit bigint (SMALLBIG)")
endif()
message(STATUS "=====================================")