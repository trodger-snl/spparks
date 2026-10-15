#!/bin/bash

# CMake build script for SPPARKS
# Provides convenient interface similar to Makefile system

set -e

# Default values
BUILD_TYPE="Release"
MACHINE=""
BUILD_DIR="build"
INSTALL_PREFIX=""
ENABLE_MPI="AUTO"
ENABLE_PACKAGES=""
SHARED_LIBS="OFF"
BUILD_LIB="OFF"
ENABLE_HDF5="AUTO"
ENABLE_HIGHFIVE="AUTO"
ENABLE_JPEG="AUTO"
ENABLE_KOKKOS="OFF"
KOKKOS_BACKEND="OPENMP"
KOKKOS_ARCH=""
EXTRA_CMAKE_ARGS=()
CLEAN=false
VERBOSE=false
DEV_MODE=false
PARALLEL_JOBS="AUTO"
AGGRESSIVE_OPT="AUTO"
NATIVE_OPT="AUTO"

# Function to show usage
show_usage() {
    cat << EOF
SPPARKS CMake Build Script

Usage: $0 [options]

Options:
    -m, --machine MACHINE       Machine configuration (mac, linux, mpi, serial, etc.)
    -b, --build-type TYPE       Build type (Release, Debug, RelWithDebInfo)
    -d, --build-dir DIR         Build directory (default: build)
    -p, --prefix DIR            Installation prefix
    --mpi                       Enable MPI support
    --no-mpi                    Disable MPI support
    --hdf5                      Enable HDF5 support
    --no-hdf5                   Disable HDF5 support
    --highfive                  Enable HighFive C++ HDF5 wrapper (adds the hdf5_unstructured /
                                  hdf5_csr temperature sources; the additive apps themselves
                                  need only MPI)
    --no-highfive               Disable HighFive C++ HDF5 wrapper (default)
    --jpeg                      Enable JPEG support
    --no-jpeg                   Disable JPEG support
    --kokkos                    Enable Kokkos for GPU acceleration
    --no-kokkos                 Disable Kokkos
    --kokkos-backend BACKEND    Kokkos backend (OPENMP, CUDA, HIP, SYCL, SERIAL)
    --kokkos-arch ARCH          Kokkos GPU arch token; REQUIRED by the HIP and CUDA
                                  backends (e.g. AMD_GFX942_APU for MI300A,
                                  AMD_GFX1151 for Strix Halo, AMPERE80)
    -DVAR=VALUE                 Passed straight through to cmake (e.g.
                                  -DHDF5_ROOT=..., -DMPI_C_COMPILER=...)
    --shared                    Build shared libraries
    --lib                       Build as library only
    --package PKG               Enable package (can be used multiple times)
                                  Available: stitch, kokkos (kokkos may not be available on all branches)
    -j, --jobs N                Number of parallel build jobs (default: auto-detect)
    --aggressive-opt            Enable aggressive optimizations (-O3, LTO)
    --no-aggressive-opt         Disable aggressive optimizations (use -O2)
    --native-opt                Enable native CPU optimizations (-march=native)
    --no-native-opt             Disable native CPU optimizations
    --clean                     Clean build directory before building
    --dev-mode                  Development mode: use symlinks instead of copies (Unix only)
    -v, --verbose               Verbose output
    -h, --help                  Show this help

Package Management:
    IMPORTANT: Package files live in two locations:
      • src/PACKAGE/  (reference version, tracked in git) ← EDIT HERE!
      • src/          (installed copy, ignored by git)  ← DON'T EDIT!

    GOLDEN RULE: Always edit src/PACKAGE/ files, never src/ installed copies!
    Changes to installed files will be LOST on package reinstall.

    Check package sync status:
      cmake --build build --target package-status   # Show which packages differ
      cmake --build build --target package-diff     # Show detailed differences

Machine configurations:
    mac        macOS, no MPI, c++
    mac_mpi    macOS with MPI, mpicxx
    mac_arm    Apple Silicon Mac, no MPI
    linux      Generic Linux, g++
    mpi        Generic MPI, mpicxx
    serial     Serial build, g++
    debug      Debug build, g++
    mpi_debug  MPI Debug build, mpicxx

Examples:
    Basic builds:
      $0 -m mac                           # Build for macOS (auto-optimized)
      $0 -m mpi --package stitch          # Build with MPI and STITCH package
      $0 -m mac_arm --hdf5                # Build for Apple Silicon with HDF5+JPEG
      $0 --no-jpeg -j 8                   # Build without JPEG using 8 parallel jobs
      $0 --clean -b Debug                 # Clean build in debug mode

    Kokkos GPU acceleration:
      $0 --kokkos                         # Build with Kokkos (OpenMP backend)
      $0 --kokkos --kokkos-backend cuda   # Build with Kokkos CUDA (NVIDIA GPUs)
      $0 --kokkos --kokkos-backend hip    # Build with Kokkos HIP (AMD GPUs)

    Additive (additive/texture, additive/texture/kk):
      $0 -m mpi --highfive                # with HDF5 temperature sources (parallel HDF5)
      $0 -m mpi --no-highfive --no-hdf5   # Rosenthal/Moser only, no HDF5 dependency

    Performance tuning:
      $0 --aggressive-opt --native-opt    # Maximum performance (O3, LTO, native CPU)
      $0 --no-aggressive-opt              # Conservative optimization (O2, no LTO)

    Library builds:
      $0 --lib --shared                   # Build shared library

    Package management:
      $0 --package stitch                 # Enable STITCH package
      cmake --build build --target package-status  # Check if installed files match source

EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -m|--machine)
            MACHINE="$2"
            shift 2
            ;;
        -b|--build-type)
            BUILD_TYPE="$2"
            shift 2
            ;;
        -d|--build-dir)
            BUILD_DIR="$2"
            shift 2
            ;;
        -p|--prefix)
            INSTALL_PREFIX="$2"
            shift 2
            ;;
        --mpi)
            ENABLE_MPI="ON"
            shift
            ;;
        --no-mpi)
            ENABLE_MPI="OFF"
            shift
            ;;
        --hdf5)
            ENABLE_HDF5="ON"
            shift
            ;;
        --no-hdf5)
            ENABLE_HDF5="OFF"
            shift
            ;;
        --highfive)
            ENABLE_HIGHFIVE="ON"
            shift
            ;;
        --no-highfive)
            ENABLE_HIGHFIVE="OFF"
            shift
            ;;
        --jpeg)
            ENABLE_JPEG="ON"
            shift
            ;;
        --no-jpeg)
            ENABLE_JPEG="OFF"
            shift
            ;;
        --kokkos)
            ENABLE_KOKKOS="ON"
            shift
            ;;
        --no-kokkos)
            ENABLE_KOKKOS="OFF"
            shift
            ;;
        --kokkos-backend)
            KOKKOS_BACKEND="$2"
            shift 2
            ;;
        --kokkos-arch)
            KOKKOS_ARCH="$2"
            shift 2
            ;;
        --shared)
            SHARED_LIBS="ON"
            shift
            ;;
        --lib)
            BUILD_LIB="ON"
            shift
            ;;
        --package)
            ENABLE_PACKAGES="$ENABLE_PACKAGES $2"
            shift 2
            ;;
        -j|--jobs)
            PARALLEL_JOBS="$2"
            shift 2
            ;;
        --aggressive-opt)
            AGGRESSIVE_OPT="ON"
            shift
            ;;
        --no-aggressive-opt)
            AGGRESSIVE_OPT="OFF"
            shift
            ;;
        --native-opt)
            NATIVE_OPT="ON"
            shift
            ;;
        --no-native-opt)
            NATIVE_OPT="OFF"
            shift
            ;;
        --clean)
            CLEAN=true
            shift
            ;;
        --dev-mode)
            DEV_MODE=true
            shift
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -h|--help)
            show_usage
            exit 0
            ;;
        -D*)
            # Forward raw -DVAR=VALUE straight to cmake. Needed for anything the
            # script has no flag for -- HDF5_ROOT, MPI_C_COMPILER, CMAKE_C_COMPILER
            # and so on -- which previously made this script unusable on machines
            # that need them.
            EXTRA_CMAKE_ARGS+=("$1")
            shift
            ;;
        *)
            echo "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Get the source directory (where this script is located)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_DIR="$SCRIPT_DIR"

echo "=== SPPARKS CMake Build ==="
echo "Source directory: $SOURCE_DIR"
echo "Build directory: $BUILD_DIR"
echo "Build type: $BUILD_TYPE"
echo "Machine: ${MACHINE:-auto-detect}"

# Clean build directory if requested
if [ "$CLEAN" = true ]; then
    echo "Cleaning build directory..."
    rm -rf "$BUILD_DIR"
fi

# Create build directory
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# Prepare CMake arguments
CMAKE_ARGS=()
CMAKE_ARGS+=("-DCMAKE_BUILD_TYPE=$BUILD_TYPE")
CMAKE_ARGS+=("-DBUILD_SHARED_LIBS=$SHARED_LIBS")
CMAKE_ARGS+=("-DSPPARKS_BUILD_LIB=$BUILD_LIB")

if [ -n "$MACHINE" ]; then
    CMAKE_ARGS+=("-DSPPARKS_MACHINE=$MACHINE")
fi

if [ -n "$INSTALL_PREFIX" ]; then
    CMAKE_ARGS+=("-DCMAKE_INSTALL_PREFIX=$INSTALL_PREFIX")
fi

if [ "$ENABLE_MPI" != "AUTO" ]; then
    CMAKE_ARGS+=("-DSPPARKS_ENABLE_MPI=$ENABLE_MPI")
fi

if [ "$ENABLE_HDF5" != "AUTO" ]; then
    CMAKE_ARGS+=("-DSPPARKS_ENABLE_HDF5=$ENABLE_HDF5")
fi

if [ "$ENABLE_HIGHFIVE" != "AUTO" ]; then
    CMAKE_ARGS+=("-DSPPARKS_ENABLE_HIGHFIVE=$ENABLE_HIGHFIVE")
fi

if [ "$ENABLE_JPEG" != "AUTO" ]; then
    CMAKE_ARGS+=("-DSPPARKS_ENABLE_JPEG=$ENABLE_JPEG")
fi

# Add Kokkos settings
if [ "$ENABLE_KOKKOS" = "ON" ]; then
    CMAKE_ARGS+=("-DSPPARKS_PACKAGE_KOKKOS=ON")
    # Normalise case: CMakeLists.txt compares the backend with STREQUAL "HIP"
    # etc., so the lowercase form this script's own --help examples use ("hip")
    # was rejected with "Unknown Kokkos backend: hip". Same for the arch token.
    KOKKOS_BACKEND=$(echo "$KOKKOS_BACKEND" | tr '[:lower:]' '[:upper:]')
    CMAKE_ARGS+=("-DSPPARKS_KOKKOS_BACKEND=$KOKKOS_BACKEND")
    if [ -n "$KOKKOS_ARCH" ]; then
        KOKKOS_ARCH=$(echo "$KOKKOS_ARCH" | tr '[:lower:]' '[:upper:]')
        CMAKE_ARGS+=("-DSPPARKS_KOKKOS_ARCH=$KOKKOS_ARCH")
    fi
    # GPU backends need a device-capable C++ compiler; left alone CMake picks
    # c++/g++, which rejects the device flags at Kokkos' compile-options check.
    # HIP -> hipcc from PATH; CUDA -> Kokkos' bundled nvcc_wrapper (host compiler
    # via NVCC_WRAPPER_DEFAULT_COMPILER, default g++). An explicit
    # -DCMAKE_CXX_COMPILER=... or $CXX still wins.
    if [ -z "${CXX:-}" ] && \
       ! printf '%s\n' ${EXTRA_CMAKE_ARGS[@]+"${EXTRA_CMAKE_ARGS[@]}"} | grep -q '^-DCMAKE_CXX_COMPILER[:=]'; then
        GPU_CXX=""
        case "$KOKKOS_BACKEND" in
            HIP)  GPU_CXX=$(command -v hipcc || true) ;;
            CUDA) [ -x "$SCRIPT_DIR/../lib/kokkos/bin/nvcc_wrapper" ] && \
                      GPU_CXX="$SCRIPT_DIR/../lib/kokkos/bin/nvcc_wrapper" ;;
        esac
        if [ -n "$GPU_CXX" ]; then
            CMAKE_ARGS+=("-DCMAKE_CXX_COMPILER=$GPU_CXX")
        elif [ "$KOKKOS_BACKEND" = "HIP" ] || [ "$KOKKOS_BACKEND" = "CUDA" ]; then
            echo "Warning: $KOKKOS_BACKEND backend but no default compiler found" \
                 "(hipcc on PATH / lib/kokkos/bin/nvcc_wrapper); pass -DCMAKE_CXX_COMPILER=..." >&2
        fi
    fi
fi

# Add optimization settings
if [ "$AGGRESSIVE_OPT" != "AUTO" ]; then
    CMAKE_ARGS+=("-DSPPARKS_AGGRESSIVE_OPTIMIZATION=$AGGRESSIVE_OPT")
fi

if [ "$NATIVE_OPT" != "AUTO" ]; then
    CMAKE_ARGS+=("-DSPPARKS_NATIVE_OPTIMIZATION=$NATIVE_OPT")
fi

# Enable packages
for PKG in $ENABLE_PACKAGES; do
    PKG_UPPER=$(echo "$PKG" | tr '[:lower:]' '[:upper:]')
    CMAKE_ARGS+=("-DSPPARKS_PACKAGE_$PKG_UPPER=ON")
done

# Development mode
if [ "$DEV_MODE" = true ]; then
    CMAKE_ARGS+=("-DSPPARKS_DEV_MODE=ON")
    echo "Development mode enabled: Package files will be symlinked instead of copied"
fi

# Run CMake configuration
echo ""
echo "Running CMake configuration..."
if [ "$VERBOSE" = true ]; then
    echo "CMake command: cmake ${CMAKE_ARGS[@]} ${EXTRA_CMAKE_ARGS[@]} $SOURCE_DIR"
fi

# EXTRA_CMAKE_ARGS last so a raw -D overrides anything the flags produced.
cmake "${CMAKE_ARGS[@]}" ${EXTRA_CMAKE_ARGS[@]+"${EXTRA_CMAKE_ARGS[@]}"} "$SOURCE_DIR"

# Build
echo ""
echo "Building SPPARKS..."

# Determine number of parallel build jobs
if [ "$PARALLEL_JOBS" = "AUTO" ]; then
    # Auto-detect CPU cores for parallel build
    if command -v nproc >/dev/null 2>&1; then
        # Linux
        CORES=$(nproc)
    elif command -v sysctl >/dev/null 2>&1; then
        # macOS
        CORES=$(sysctl -n hw.ncpu)
    else
        # Fallback
        CORES=4
    fi

    echo "System has $CORES CPU cores detected"

    # On Apple Silicon, use all cores; otherwise use smart defaults
    if [[ "$OSTYPE" == "darwin"* ]] && [[ "$(uname -m)" == "arm64" ]]; then
        BUILD_JOBS=$CORES
        echo "Apple Silicon detected: Using all $CORES cores for parallel build"
    else
        # Linux/other systems: default to 8 cores, or 50% if system has <= 8 cores
        if [ $CORES -le 8 ]; then
            BUILD_JOBS=$((CORES / 2))
            if [ $BUILD_JOBS -lt 1 ]; then
                BUILD_JOBS=1
            fi
            echo "Using $BUILD_JOBS of $CORES available cores for parallel build (50%)"
        else
            BUILD_JOBS=8
            echo "Using $BUILD_JOBS of $CORES available cores for parallel build"
        fi
    fi
else
    BUILD_JOBS=$PARALLEL_JOBS
    echo "Using manually specified $BUILD_JOBS parallel build jobs"
fi

MAKE_ARGS=("-j$BUILD_JOBS")
if [ "$VERBOSE" = true ]; then
    MAKE_ARGS+=("--verbose")
fi

cmake --build . "${MAKE_ARGS[@]}"

echo ""
echo "=== Build Complete ==="
echo "Executable: $BUILD_DIR/spparks"
if [ "$BUILD_LIB" = "ON" ]; then
    echo "Library: $BUILD_DIR/libspparks.*"
fi
echo ""
echo "To install: cmake --install $BUILD_DIR"
echo "To run tests: ctest"