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
ENABLE_JPEG="AUTO"
CLEAN=false
VERBOSE=false
PARALLEL_JOBS="AUTO"

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
    --jpeg                      Enable JPEG support
    --no-jpeg                   Disable JPEG support
    --shared                    Build shared libraries
    --lib                       Build as library only
    --package PKG               Enable package (can be used multiple times)
    -j, --jobs N                Number of parallel build jobs (default: auto-detect)
    --clean                     Clean build directory before building
    -v, --verbose               Verbose output
    -h, --help                  Show this help

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
    $0 -m mac                           # Build for macOS (auto-enables JPEG if found)
    $0 -m mpi --package stitch          # Build with MPI and STITCH package
    $0 -m mac_arm --hdf5                # Build for Apple Silicon with HDF5+JPEG (auto-detects all cores)
    $0 --no-jpeg -j 8                  # Build without JPEG using 8 parallel jobs
    $0 --lib --shared                   # Build shared library
    $0 --clean -b Debug                 # Clean build in debug mode

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
        --jpeg)
            ENABLE_JPEG="ON"
            shift
            ;;
        --no-jpeg)
            ENABLE_JPEG="OFF"
            shift
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
        --clean)
            CLEAN=true
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

if [ "$ENABLE_JPEG" != "AUTO" ]; then
    CMAKE_ARGS+=("-DSPPARKS_ENABLE_JPEG=$ENABLE_JPEG")
fi

# Enable packages
for PKG in $ENABLE_PACKAGES; do
    PKG_UPPER=$(echo "$PKG" | tr '[:lower:]' '[:upper:]')
    CMAKE_ARGS+=("-DSPPARKS_PACKAGE_$PKG_UPPER=ON")
done

# Run CMake configuration
echo ""
echo "Running CMake configuration..."
if [ "$VERBOSE" = true ]; then
    echo "CMake command: cmake ${CMAKE_ARGS[@]} $SOURCE_DIR"
fi

cmake "${CMAKE_ARGS[@]}" "$SOURCE_DIR"

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
    
    # On Apple Silicon, use all cores; otherwise leave one free
    if [[ "$OSTYPE" == "darwin"* ]] && [[ "$(uname -m)" == "arm64" ]]; then
        BUILD_JOBS=$CORES
        echo "Apple Silicon detected: Using all $CORES cores for parallel build"
    else
        BUILD_JOBS=$((CORES > 1 ? CORES - 1 : 1))
        echo "Using $BUILD_JOBS of $CORES available cores for parallel build"
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
echo "Executable: $BUILD_DIR/spk"
if [ "$BUILD_LIB" = "ON" ]; then
    echo "Library: $BUILD_DIR/libspparks.*"
fi
echo ""
echo "To install: cmake --install $BUILD_DIR"
echo "To run tests: ctest"