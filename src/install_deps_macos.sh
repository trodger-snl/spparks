#!/bin/bash

# Script to install common SPPARKS dependencies on macOS

set -e

echo "=== SPPARKS Dependency Installer for macOS ==="

# Check if Homebrew is installed
if ! command -v brew &> /dev/null; then
    echo "Homebrew not found. Please install Homebrew first:"
    echo "  /bin/bash -c \"\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\""
    exit 1
fi

echo "Updating Homebrew..."
brew update

echo "Installing SPPARKS dependencies..."

# Core dependencies
echo "Installing MPI..."
brew install open-mpi

echo "Installing HDF5 with MPI support..."
brew install hdf5-mpi

echo "Installing JPEG library..."
brew install jpeg

echo "Installing PNG library..."
brew install libpng

echo "Installing ZLIB (usually already present)..."
brew install zlib

echo ""
echo "=== Installation Complete ==="
echo "You can now build SPPARKS with:"
echo "  ./build_cmake.sh -m mac_arm --hdf5"
echo ""
echo "Or for a full-featured build:"
echo "  ./build_cmake.sh -m mac_arm --hdf5 --package stitch"
echo ""

# Test that key libraries can be found
echo "=== Dependency Check ==="
echo "MPI compiler: $(which mpicc || echo 'NOT FOUND')"
echo "JPEG library: $(brew --prefix jpeg)/lib/libjpeg.dylib"
echo "HDF5 library: $(brew --prefix hdf5-mpi)/lib/libhdf5.dylib"
echo "HDF5 includes: $(brew --prefix hdf5-mpi)/include"
echo ""