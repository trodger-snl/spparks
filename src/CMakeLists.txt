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

# Set default build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Performance options. These MUST be declared before the Release block below,
# which reads them. Declared after it, a fresh configure evaluated them as empty
# (option() had not run yet) and silently produced -O2 with no -march=native and
# no LTO, while re-configuring the same directory picked the values up from the
# cache and produced -O3 -march=native -flto. The optimization level therefore
# depended on whether the build directory was new.
option(SPPARKS_AGGRESSIVE_OPTIMIZATION "Enable aggressive optimizations (may increase compile time)" ON)
option(SPPARKS_NATIVE_OPTIMIZATION "Enable native CPU optimizations (-march=native)" ON)

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

# Build options
option(BUILD_SHARED_LIBS "Build shared libraries" OFF)
option(SPPARKS_ENABLE_MPI "Enable MPI support" ON)
option(SPPARKS_ENABLE_GZIP "Enable GZIP support" ON)
option(SPPARKS_ENABLE_JPEG "Enable JPEG support" ON)
option(SPPARKS_ENABLE_PNG "Enable PNG support" OFF)
option(SPPARKS_ENABLE_HDF5 "Enable HDF5 support" OFF)
option(SPPARKS_ENABLE_HIGHFIVE "Enable HighFive C++ HDF5 wrapper" OFF)
option(SPPARKS_USE_EXTERNAL_HIGHFIVE "Use system-provided HighFive instead of FetchContent" OFF)
option(SPPARKS_UNORDERED_MAP "Use unordered_map" OFF)

# Packages: optional components whose sources live in src/KOKKOS/ and src/STITCH/
# and compile from there (no copying into src/).
option(SPPARKS_PACKAGE_STITCH "Enable STITCH package (dump stitch)" OFF)
option(SPPARKS_PACKAGE_KOKKOS "Enable KOKKOS package for GPU acceleration" OFF)
set(SPPARKS_KOKKOS_BACKEND "OPENMP" CACHE STRING "Kokkos backend (OPENMP, CUDA, HIP, SYCL, SERIAL)")
set_property(CACHE SPPARKS_KOKKOS_BACKEND PROPERTY STRINGS OPENMP CUDA HIP SYCL SERIAL)
# GPU architecture for the CUDA/HIP backends. Kokkos does NOT auto-detect this,
# so it must be set explicitly or the device build silently produces no GPU code.
# Examples: AMD_GFX1151 (Strix Halo / Radeon 8050S/8060S, RDNA 3.5) for HIP,
#           AMPERE80 (A100) or ADA89 (RTX 40xx) for CUDA.
set(SPPARKS_KOKKOS_ARCH "" CACHE STRING "Kokkos GPU architecture token (e.g. AMD_GFX1151, AMPERE80)")

# Enable OpenMP as the Kokkos HOST execution space alongside a GPU DEVICE backend
# (HIP/CUDA/SYCL). With this ON, Kokkos::DefaultExecutionSpace stays the GPU while
# Kokkos::DefaultHostExecutionSpace becomes OpenMP instead of Serial -- so
# host-side Kokkos parallel_for loops run multithreaded. (The non-Kokkos
# #pragma omp host loops in the HDF5 source are already threaded via the
# find_package(OpenMP) link below regardless of this option; this option adds
# threading for the Kokkos host space itself.) Ignored for the OPENMP/SERIAL
# backends. Intended for multi-GPU HPC nodes with many host cores per GPU
# (e.g. 4xH100 + 112 cores: 1 rank/GPU x N host threads) and usable on this APU
# (HIP + OpenMP) for the same host-threading benefit.
option(SPPARKS_KOKKOS_HOST_OPENMP "Use OpenMP (not Serial) as the Kokkos host space for GPU backends" OFF)

# Performance options are declared near the top of this file, before the
# Release flag block that reads them.

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

# Additive/texture device temperature-eval precision. Default OFF => FP64
# (double), the HPC production precision and bit-for-bit CPU-validation
# reference. ON => FP32 (float) for the additive/texture device temperature
# work, at the cost of bit-exact validation. Two distinct wins, per source:
#   * Rosenthal/Moser: compute-bound transcendental kernels, where
#     RDNA/consumer GPUs (gfx1151) run FP64 at heavily reduced rate.
#   * HDF5 unstructured: a gather-and-blend fill that is MEMORY-bound,
#     so the win is halved traffic on the per-node temperature slice, the
#     barycentric weights and the raw sample temperatures. Measured on the ct15
#     10-layer decks, that fill is ~33% of per-box wall and is provably not
#     host-threading-bound (flat across OMP_NUM_THREADS 1/4/16).
# Absolute simulation times and the k_T field stay FP64 in both cases --
# see src/KOKKOS/am_temp_precision.h for why narrowing them is not safe.
option(SPPARKS_AM_TEMP_FP32 "Use FP32 for additive/texture device temperature kernels (Rosenthal/Moser + HDF5)" OFF)
if(SPPARKS_AM_TEMP_FP32)
    add_compile_definitions(SPK_AM_TEMP_FP32)
endif()

# potts/quaternion/kk disorientation + Read-Shockley precision. Default OFF =>
# FP64, unchanged arithmetic. ON => FP32 (float), a win on RDNA/consumer GPUs
# whose FP64 rate is heavily reduced: gfx1151, 256^3 sweep color/strict, loop
# 59.5 -> 32.4 s (cubic), 40.9 -> 21.1 s (hcp); disorientation agrees with
# FP64 to ~2e-5 deg. See src/KOKKOS/quat_precision.h.
option(SPPARKS_QUAT_FP32 "Use FP32 for potts/quaternion/kk disorientation and energy math" OFF)
if(SPPARKS_QUAT_FP32)
    add_compile_definitions(SPK_QUAT_FP32)
endif()

# Configure features
if(SPPARKS_ENABLE_GZIP)
    add_compile_definitions(SPPARKS_GZIP)
endif()

if(SPPARKS_UNORDERED_MAP)
    add_compile_definitions(SPPARKS_UNORDERED_MAP)
endif()

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

# HighFive is a C++ wrapper over HDF5, so it implies HDF5.
if(SPPARKS_ENABLE_HIGHFIVE AND NOT SPPARKS_ENABLE_HDF5)
    message(STATUS "HighFive requires HDF5 - enabling HDF5 automatically")
    set(SPPARKS_ENABLE_HDF5 ON CACHE BOOL "Enable HDF5 support" FORCE)
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
            # Not a performance note. The HDF5 temperature sources call
            # HighFive::MPIOFileAccess(...) unconditionally, so serial HDF5 with
            # HighFive+MPI is a compile failure in
            # temperature_source_hdf5_unstructured.cpp, not a slow build.
            if(SPPARKS_ENABLE_HIGHFIVE)
                message(FATAL_ERROR
                    "MPI is enabled but the HDF5 found at ${HDF5_INCLUDE_DIRS} is serial.\n"
                    "The additive/texture temperature sources call HighFive::MPIOFileAccess() "
                    "unconditionally, so this will fail to compile rather than run slowly.\n"
                    "Load a parallel HDF5 (on a Cray system: module load cray-hdf5-parallel) "
                    "and point at it with -DHDF5_ROOT=<prefix>, or configure with "
                    "-DSPPARKS_ENABLE_HIGHFIVE=OFF to drop the additive/texture styles.")
            else()
                message(WARNING
                    "MPI is enabled but the HDF5 found at ${HDF5_INCLUDE_DIRS} is serial. "
                    "Parallel I/O will be unavailable.")
            endif()
        endif()
    else()
        message(FATAL_ERROR "HDF5 was requested but not found")
    endif()
endif()

if(SPPARKS_ENABLE_HIGHFIVE)
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
        # HighFive requires a FULL MPI when HDF5 is parallel, including the C
        # component we deliberately do not require ourselves. Its
        # CMake/HighFiveTargetDeps.cmake:111-112 does
        #
        #   if(HIGHFIVE_PARALLEL_HDF5 OR HDF5_IS_PARALLEL)
        #     find_package(MPI REQUIRED)        # all enabled languages
        #
        # and then uses only MPI_CXX_*. So on a machine where FindMPI cannot
        # validate the C wrapper, configure dies inside third-party code with
        # "Could NOT find MPI_C (missing: MPI_C_WORKS)" and a trace into
        # HighFiveTargetDeps.cmake, which reads like a HighFive problem and is
        # not. The same file is included by the installed HighFiveConfig.cmake,
        # so SPPARKS_USE_EXTERNAL_HIGHFIVE does not avoid it either.
        #
        # Say so before it happens, with the fix.
        if(SPPARKS_ENABLE_MPI AND HDF5_IS_PARALLEL)
            find_package(MPI QUIET COMPONENTS C)
            if(NOT MPI_C_FOUND)
                message(WARNING
                    "Parallel HDF5 is in use and MPI's C component was not usable "
                    "(MPI_C_FOUND is false). SPPARKS itself does not need MPI C -- "
                    "it has no C sources -- but HighFive requires it unconditionally "
                    "in CMake/HighFiveTargetDeps.cmake, so configure is likely to "
                    "fail there next with 'Could NOT find MPI_C (missing: "
                    "MPI_C_WORKS)'.\n"
                    "Fix by pointing FindMPI at a working MPI C wrapper, e.g.\n"
                    "  -DMPI_C_COMPILER=cc              (Cray PE C wrapper)\n"
                    "  -DMPI_C_COMPILER=mpiamdclang     (LLNL magic-module wrapper)\n"
                    "  -DMPI_C_COMPILER=mpicc\n"
                    "Pick whichever exists in your environment: `which cc mpiamdclang mpicc`.")
            endif()
        endif()

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
            message(FATAL_ERROR
                "HIP backend requires SPPARKS_KOKKOS_ARCH.\n"
                "  AMD_GFX942_APU  MI300A APU (El Capitan and similar; unified memory)\n"
                "  AMD_GFX942      MI300X, or MI300A with unified memory disabled\n"
                "  AMD_GFX90A      MI200 series\n"
                "  AMD_GFX1151     Strix Halo / RDNA 3.5 APU\n"
                "Set it with -DSPPARKS_KOKKOS_ARCH=<token>, or use a preset from "
                "src/CMakePresets.json (cmake --list-presets).")
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

    # Disable backends we don't want.
    # OpenMP is special: besides being a standalone device backend, it can serve
    # as the HOST execution space for a GPU device backend. Keep it enabled when
    # SPPARKS_KOKKOS_HOST_OPENMP is requested (Serial stays enabled too, but
    # Kokkos picks OpenMP as DefaultHostExecutionSpace -- OpenMP outranks Serial).
    if(SPPARKS_KOKKOS_BACKEND STREQUAL "OPENMP")
        # OpenMP is the device backend; leave it enabled.
    elseif(SPPARKS_KOKKOS_HOST_OPENMP)
        set(Kokkos_ENABLE_OPENMP ON CACHE BOOL "OpenMP as Kokkos host space" FORCE)
        message(STATUS "Kokkos: OpenMP host execution space enabled alongside "
                       "${SPPARKS_KOKKOS_BACKEND} device (DefaultHostExecutionSpace=OpenMP)")
    else()
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
            "Kokkos not found at ${KOKKOS_DIR}\n"
            "\n"
            "Normally:\n"
            "  git submodule update --init --recursive\n"
            "\n"
            "If that fails -- no submodule metadata (source copied rather than\n"
            "cloned), or the submodule fetch is blocked -- clone it directly. This\n"
            "check only requires the directory to exist with a CMakeLists.txt; a\n"
            "registered submodule is not needed:\n"
            "  rm -rf ${KOKKOS_DIR}\n"
            "  git clone --depth 1 --branch 5.1.1 \\\n"
            "      https://github.com/kokkos/kokkos.git ${KOKKOS_DIR}\n"
            "\n"
            "Tag 5.1.1 is the pinned commit\n"
            "(267ebc25fc5c8b96bb321f34d78f97b1c30f8830). If the machine has no\n"
            "outbound network, clone elsewhere and copy the tree in.\n")
    endif()

    # Validate the arch token against the bundled Kokkos before handing off. We set
    # Kokkos_ARCH_${SPPARKS_KOKKOS_ARCH} unconditionally above, so a typo produces a
    # cache entry Kokkos simply never reads: the configure succeeds, no arch is
    # selected, and the failure surfaces as an unrelated compile or link error much
    # later. Catch it here, where we can name the valid tokens.
    if(SPPARKS_KOKKOS_ARCH)
        file(READ "${KOKKOS_DIR}/cmake/kokkos_arch.cmake" _kokkos_arch_list)
        if(NOT _kokkos_arch_list MATCHES "[ \t(]${SPPARKS_KOKKOS_ARCH}[ \t)\n]")
            string(REGEX MATCHALL "SUPPORTED_AMD_ARCHS ([A-Z0-9_ ]+)" _amd_arch_lines "${_kokkos_arch_list}")
            string(REGEX REPLACE "SUPPORTED_AMD_ARCHS " "" _amd_tokens "${_amd_arch_lines}")
            string(REPLACE ";" " " _amd_tokens "${_amd_tokens}")
            string(REGEX REPLACE " +" " " _amd_tokens "${_amd_tokens}")
            message(FATAL_ERROR
                "SPPARKS_KOKKOS_ARCH='${SPPARKS_KOKKOS_ARCH}' is not a token known to the "
                "bundled Kokkos at ${KOKKOS_DIR}.\n"
                "This would otherwise configure cleanly and fail later with an unrelated error.\n"
                "AMD tokens this Kokkos accepts:\n  ${_amd_tokens}\n"
                "Run 'cmake --list-presets' in src/ for the configurations already defined here.")
        endif()
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

if(SPPARKS_ENABLE_MPI)
    # COMPONENTS CXX, not a bare find_package(MPI REQUIRED).
    #
    # project() above declares LANGUAGES CXX C, and a bare find_package(MPI)
    # requires EVERY enabled language, so it demanded a working MPI C as well.
    # SPPARKS has no C sources (src/ contains zero .c files) and links only
    # MPI::MPI_CXX, so MPI C was never used -- it was purely an extra way to
    # fail. On a Cray machine whose C wrapper FindMPI cannot exercise, that
    # showed up as
    #
    #   Could NOT find MPI_C (missing: MPI_C_WORKS)
    #   Could NOT find MPI (missing: MPI_C_FOUND) (found version "4.0")
    #
    # -- note MPI CXX had been found fine; only the unused C component failed.
    find_package(MPI REQUIRED COMPONENTS CXX)
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

# STITCH package: libstitch (stitch.c + bundled sqlite3) is a plain static
# library target; dump_stitch compiles from src/STITCH/. stitch.c includes mpi.h
# under STITCH_PARALLEL, so it borrows the include path FindMPI found for C++
# (the C component is deliberately not required -- see the MPI note above).
if(SPPARKS_PACKAGE_STITCH)
    set(_stitch_dir "${CMAKE_CURRENT_SOURCE_DIR}/../lib/stitch/src/stitch/libstitch")
    add_library(stitch STATIC "${_stitch_dir}/stitch.c" "${_stitch_dir}/sqlite3.c")
    target_include_directories(stitch PUBLIC "${_stitch_dir}")
    target_compile_definitions(stitch PUBLIC STITCH_PARALLEL SPK_STITCH)
    set_target_properties(stitch PROPERTIES C_STANDARD 99 C_EXTENSIONS ON
                                            POSITION_INDEPENDENT_CODE ON)
    find_package(Threads REQUIRED)
    target_link_libraries(stitch PUBLIC ${CMAKE_DL_LIBS} Threads::Threads)
    if(SPPARKS_ENABLE_MPI)
        target_include_directories(stitch PRIVATE ${MPI_CXX_INCLUDE_DIRS})
    endif()
    include_directories(${CMAKE_CURRENT_SOURCE_DIR}/STITCH)
endif()

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
if(SPPARKS_PACKAGE_KOKKOS)
    include_directories(${CMAKE_CURRENT_SOURCE_DIR}/KOKKOS)
endif()

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

# Generate style headers after all dependencies are resolved
include(cmake/generate_styles.cmake)


# One static library holding every source; the executable is main.cpp on top
# of it. Dependencies are PUBLIC so the executable inherits them.
add_library(spparks_lib STATIC ${SPPARKS_SOURCES})
set_target_properties(spparks_lib PROPERTIES OUTPUT_NAME spparks)
add_dependencies(spparks_lib style_headers)
target_include_directories(spparks_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

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

# Host OpenMP for the non-Kokkos per-site layer-cache loops in the HDF5
# temperature source (build_site_element_cache / compute_site_thermal_intervals,
# ~40 s/layer serial on a 5M-site box). The OpenMP Kokkos backend already
# compiles with -fopenmp; for the HIP/CUDA backends this adds host threading for
# those loops. Harmless (loops just run serially) if OpenMP is unavailable.
find_package(OpenMP QUIET)
if(OpenMP_CXX_FOUND)
    target_link_libraries(spparks_lib PUBLIC OpenMP::OpenMP_CXX)
else()
    message(STATUS "OpenMP not found; HDF5 layer-cache loops will run serially")
endif()

if(SPPARKS_PACKAGE_KOKKOS)
    target_link_libraries(spparks_lib PUBLIC Kokkos::kokkos)
endif()
if(SPPARKS_PACKAGE_STITCH)
    target_link_libraries(spparks_lib PUBLIC stitch)
endif()
if(SPPARKS_ENABLE_GZIP)
    find_package(ZLIB)
    if(ZLIB_FOUND)
        target_link_libraries(spparks_lib PUBLIC ZLIB::ZLIB)
    endif()
endif()

add_executable(spparks main.cpp)
target_link_libraries(spparks PRIVATE spparks_lib)

# Compiler flags, per target
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        set(_spk_flags -g -Wall -Wextra)
    else()
        separate_arguments(_spk_flags UNIX_COMMAND "${OPT_FLAGS}")
    endif()
    target_compile_options(spparks_lib PRIVATE ${_spk_flags})
    target_compile_options(spparks PRIVATE ${_spk_flags})
endif()

install(TARGETS spparks RUNTIME DESTINATION bin)

# Print configuration summary
message(STATUS "=== SPPARKS Configuration Summary ===")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
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
if(SPPARKS_ENABLE_HIGHFIVE)
    if(SPPARKS_USE_EXTERNAL_HIGHFIVE)
        message(STATUS "  HighFive source: external (${HighFive_DIR})")
    else()
        message(STATUS "  HighFive source: FetchContent from github.com (needs network at configure time)")
    endif()
endif()
message(STATUS "Kokkos support: ${SPPARKS_PACKAGE_KOKKOS}")
if(SPPARKS_PACKAGE_KOKKOS)
    message(STATUS "  Kokkos backend: ${SPPARKS_KOKKOS_BACKEND}")
    if(SPPARKS_KOKKOS_ARCH)
        message(STATUS "  Kokkos arch: ${SPPARKS_KOKKOS_ARCH}")
    endif()
    if(SPPARKS_KOKKOS_HOST_OPENMP)
        message(STATUS "  Kokkos host space: OpenMP")
    endif()
    # Unified memory is the single behavioural difference between the MI300A APU
    # token and the plain gfx942 token, and it changes whether DualView keeps a
    # distinct host mirror. State it explicitly: it is otherwise invisible in the
    # log, and it decides whether HSA_XNACK=1 is needed at runtime.
    if(SPPARKS_KOKKOS_ARCH STREQUAL "AMD_GFX942_APU")
        message(STATUS "  Unified memory: ON (KOKKOS_IMPL_HIP_UNIFIED_MEMORY) -- export HSA_XNACK=1 to run")
    elseif(SPPARKS_KOKKOS_BACKEND STREQUAL "HIP")
        message(STATUS "  Unified memory: OFF -- HIPSpace is host-inaccessible, deep_copy does real work")
    endif()
endif()
if(SPPARKS_SMALLSMALL)
    message(STATUS "Integer types: 32-bit (SMALLSMALL)")
elseif(SPPARKS_BIGBIG)
    message(STATUS "Integer types: 64-bit tagint/bigint (BIGBIG)")
else()
    message(STATUS "Integer types: 64-bit bigint (SMALLBIG)")
endif()

# Optional styles that were NOT registered, with the flag that would enable each.
# SPPARKS_SKIPPED_STYLES is built in configure_conditional_sources(). A style that
# is silently absent here surfaces much later as "style not recognized" at runtime,
# with nothing in the C++ to explain it, so it is worth stating at configure time.
if(SPPARKS_SKIPPED_STYLES)
    message(STATUS "")
    message(STATUS "Styles NOT built (each needs the feature shown):")
    foreach(_skipped IN LISTS SPPARKS_SKIPPED_STYLES)
        message(STATUS "  ${_skipped}")
    endforeach()
endif()

# Environment block. This exists so that a build log copied off a remote machine
# carries everything needed to diagnose it without an interactive session.
message(STATUS "")
message(STATUS "--- Environment (include this when reporting a build problem) ---")
message(STATUS "CMake: ${CMAKE_VERSION} (${CMAKE_GENERATOR})")
message(STATUS "System: ${CMAKE_SYSTEM_NAME} ${CMAKE_SYSTEM_VERSION} ${CMAKE_SYSTEM_PROCESSOR}")
message(STATUS "CXX: ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "CXX path: ${CMAKE_CXX_COMPILER}")
message(STATUS "CXX standard: ${CMAKE_CXX_STANDARD}")
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    message(STATUS "Release flags: ${OPT_FLAGS}")
endif()
if(SPPARKS_ENABLE_MPI)
    if(MPI_CXX_FOUND)
        # MPI_CXX_LIBRARY_VERSION_STRING is only populated when FindMPI could run a
        # probe binary, which does not happen on machines that forbid launching MPI
        # from a login node. Fall back to the standard version so the line is never
        # blank in a log someone pastes back.
        if(MPI_CXX_LIBRARY_VERSION_STRING)
            string(REGEX REPLACE "\n.*" "" _mpi_impl "${MPI_CXX_LIBRARY_VERSION_STRING}")
            message(STATUS "MPI: ${_mpi_impl}")
        else()
            message(STATUS "MPI: implementation string unavailable (MPI standard ${MPI_CXX_VERSION})")
        endif()
        message(STATUS "MPI compiler: ${MPI_CXX_COMPILER}")
    else()
        message(STATUS "MPI: requested but MPI_CXX_FOUND is false")
    endif()
endif()
if(SPPARKS_ENABLE_HDF5 AND HDF5_FOUND)
    message(STATUS "HDF5 include: ${HDF5_INCLUDE_DIRS}")
endif()
message(STATUS "Source dir: ${CMAKE_CURRENT_SOURCE_DIR}")
message(STATUS "Binary dir: ${CMAKE_CURRENT_BINARY_DIR}")
message(STATUS "=====================================")