# Spack Environment Setup for SPPARKS

This guide explains how to use Spack to manage SPPARKS dependencies across different systems, including macOS (ARM/Intel), Linux (x86/ARM), and Windows (WSL2).

## Table of Contents
- [Supported Platforms](#supported-platforms)
- [Why Spack?](#why-spack)
- [Prerequisites](#prerequisites)
- [Initial Setup](#initial-setup)
- [Creating the Environment](#creating-the-environment)
- [Platform-Specific Workflows](#platform-specific-workflows)
- [Using Modules](#using-modules)
- [Building SPPARKS](#building-spparks)
- [Advanced Topics](#advanced-topics)

## Supported Platforms

| Platform | Architecture | Status | Notes |
|----------|-------------|---------|-------|
| macOS | ARM (M1/M2/M3) | ✅ Tested | Apple Silicon |
| macOS | Intel (x86_64) | ✅ Tested | Intel-based Macs |
| Linux | x86_64 | ✅ Tested | RHEL, Ubuntu, Debian, etc. |
| Linux | ARM64 | ⚠️ Experimental | Not extensively tested |
| Windows | WSL2 (Ubuntu) | ✅ Tested | Via Windows Subsystem for Linux |
| Windows | Native | ❌ Not Supported | Use WSL2 instead |

## Why Spack?

Spack provides:
- **Consistent dependencies** across development and HPC systems
- **Reproducible builds** via lockfiles
- **Easy installation** of complex dependency chains (MPI, HDF5, etc.)
- **Module generation** for HPC environments (no admin permissions needed)
- **Version control friendly** configurations

## Prerequisites

Before using Spack with SPPARKS, ensure you have:

**Required:**
- **Git** - For cloning Spack and SPPARKS repositories
- **C++17 Compatible Compiler:**
  - macOS: Xcode Command Line Tools (Clang 11+) or Homebrew GCC 7+
  - Linux: GCC 7+ or Clang 5+
  - Windows: Use WSL2 with Ubuntu (includes GCC)
- **Basic build tools** - make, cmake (Spack can install these if missing)

**Optional:**
- **MPI implementation** - If not using Spack-provided MPI (OpenMPI/MPICH)
- **Python 3** - For Spack itself (usually pre-installed on most systems)

**Installing Prerequisites:**

```bash
# macOS (Xcode Command Line Tools)
xcode-select --install

# Ubuntu/Debian
sudo apt-get update
sudo apt-get install build-essential git

# RHEL/CentOS
sudo yum groupinstall "Development Tools"
sudo yum install git

# Windows: Install WSL2 with Ubuntu first
# Then run Ubuntu commands above
```

## Initial Setup

### Installing Spack (One-Time)

```bash
# Clone Spack repository
git clone --depth=2 https://github.com/spack/spack.git ~/spack

# Add to your shell configuration (~/.bashrc, ~/.zshrc, etc.)
source ~/spack/share/spack/setup-env.sh

# Detect available compilers
spack compiler find

# Verify installation
spack --version
```

### Configuring Spack for SPPARKS

No additional configuration needed! The `spack.yaml` file in this repository contains all settings.

## Creating the Environment

### Quick Start

```bash
# Navigate to SPPARKS directory
cd /path/to/spparks

# Create environment from spack.yaml
spack env create spparks-dev spack.yaml

# Activate environment
spack env activate spparks-dev

# Install all dependencies (this may take 30-60 minutes on first run)
spack install -j$(nproc)

# Verify installation
spack find
```

### What Gets Installed

- **OpenMPI** (4.1+): MPI implementation with C++ support
- **HDF5** (1.12+): Parallel HDF5 with high-level API and compression
- **HighFive** (2.8+): Modern C++ wrapper for HDF5
- **libjpeg**: JPEG image support
- **libpng**: PNG image support
- **zlib**: Compression library (gzip)
- **CMake** (3.20+): Build system
- **gmake**: GNU Make

## Platform-Specific Workflows

### macOS (ARM - Apple Silicon)

```bash
# Initial setup on macOS
cd ~/spparks
spack env create spparks-dev spack.yaml
spack env activate spparks-dev

# Concretize for macOS ARM
spack concretize -f

# Install (uses Apple Clang by default)
spack install

# Create platform-specific lockfile for version control
cp spack.lock spack-darwin-arm.lock
git add spack-darwin-arm.lock
git commit -m "Add Spack lockfile for macOS ARM"
```

**Note for macOS users:**
- Spack will use Apple Clang by default
- You can also use Homebrew GCC: `spack install gcc` then use `spack compiler add`
- The environment will work with both `build_cmake.sh` and traditional Makefiles

### RHEL Linux (x86)

```bash
# Initial setup on RHEL
cd ~/spparks
spack env create spparks-dev spack.yaml
spack env activate spparks-dev

# Concretize for Linux x86
spack concretize -f

# Install (uses GCC by default)
spack install

# Create platform-specific lockfile for version control
cp spack.lock spack-linux-x86.lock
git add spack-linux-x86.lock
git commit -m "Add Spack lockfile for RHEL x86"
```

**Note for RHEL users:**
- Uses system GCC by default (or Spack-provided GCC if system version is too old)
- Compatible with existing module systems
- Can generate user-space modules (no admin permissions needed)

### Windows (WSL2)

SPPARKS can be built on Windows using Windows Subsystem for Linux 2 (WSL2). Native Windows builds are not currently supported.

```bash
# Prerequisites: Install WSL2 with Ubuntu
# From Windows PowerShell (Admin):
# wsl --install -d Ubuntu

# Inside WSL2 Ubuntu, follow Linux instructions:
cd ~/spparks
spack env create spparks-dev spack.yaml
spack env activate spparks-dev

# Concretize and install
spack concretize -f
spack install

# CMake will automatically detect WSL2 and use Linux configuration
cd src
./build_cmake.sh -m mpi
```

**Note for WSL2 users:**
- SPPARKS will be built using Linux toolchain within WSL2
- Access files from Windows at `\\wsl$\Ubuntu\home\username\spparks`
- Performance is comparable to native Linux
- Native Windows builds (without WSL2) are not supported

### Using Existing Lockfiles

If lockfiles already exist in the repository:

```bash
# Create environment from platform-specific lockfile
spack env create spparks-dev spack-darwin-arm.lock  # On macOS
# OR
spack env create spparks-dev spack-linux-x86.lock   # On Linux

# Activate and install
spack env activate spparks-dev
spack install

# All packages will be installed with exact versions from lockfile
```

This ensures **exact reproducibility** of the build environment.

### Lockfile Management Best Practices

Lockfiles ensure reproducibility but are **platform-specific**. Follow these practices for version control:

**Do's:**
- ✅ Use platform-suffixed lockfiles: `spack-darwin-arm.lock`, `spack-linux-x86.lock`
- ✅ Commit platform-specific lockfiles to version control
- ✅ Update lockfiles when dependencies change
- ✅ Document which lockfile corresponds to which platform

**Don'ts:**
- ❌ Never commit `spack.lock` directly (it's platform-specific)
- ❌ Don't try to use a macOS lockfile on Linux (won't work)
- ❌ Don't share lockfiles across different architectures

**Updating Lockfiles:**

```bash
# When updating dependencies in spack.yaml
spack env activate spparks-dev
spack concretize -f  # Force re-concretization

# Create platform-specific lockfile
cp spack.lock spack-$(uname -s)-$(uname -m).lock

# Add to version control
git add spack-*.lock
git commit -m "Update Spack lockfile for $(uname -s)/$(uname -m)"
```

**Multi-Platform Development:**

If developing on multiple platforms:
```bash
# On macOS ARM
spack concretize -f
cp spack.lock spack-darwin-arm64.lock
git add spack-darwin-arm64.lock

# On Linux x86_64
spack concretize -f
cp spack.lock spack-linux-x86_64.lock
git add spack-linux-x86_64.lock

# Commit both
git commit -m "Update lockfiles for macOS ARM and Linux x86_64"
```

## Using Modules

### Generating Module Files

Module files are automatically generated in user space (no admin permissions needed).

```bash
# Activate environment
spack env activate spparks-dev

# Generate module files (one-time)
spack module tcl refresh

# List available modules
module use $(spack location -e)/modules
module avail

# Load a specific package module
module load openmpi
module load hdf5
module load cmake

# Or load with dependencies
source <(spack module tcl loads --dependencies hdf5)
```

### Module Workflow

**Option 1: Use Spack activation (simplest)**
```bash
spack env activate spparks-dev
# All environment variables are set automatically
cd spparks/src
./build_cmake.sh -m mpi
```

**Option 2: Use modules without activation**
```bash
module use ~/spack/var/spack/environments/spparks-dev/modules
module load openmpi hdf5 cmake
cd spparks/src
./build_cmake.sh -m mpi
```

**Option 3: Permanent module configuration**

Add to your `~/.bashrc` or `~/.zshrc`:
```bash
# Make SPPARKS modules available
module use ~/spack/var/spack/environments/spparks-dev/modules
```

Then load modules as needed:
```bash
module load openmpi hdf5
```

### Module Commands Reference

```bash
# Add environment modules to search path
module use $(spack location -e spparks-dev)/modules

# List available modules
module avail

# Load a module
module load openmpi

# List loaded modules
module list

# Unload a module
module unload openmpi

# Show module details
module show openmpi

# Generate load script for dependencies
spack module tcl loads --dependencies hdf5
```

## Building SPPARKS

### With CMake (Recommended)

```bash
# Activate Spack environment
spack env activate spparks-dev

# Environment variables are automatically set:
# - CMAKE_PREFIX_PATH points to all dependencies
# - PATH includes MPI compilers

# Build SPPARKS
cd src
./build_cmake.sh -m mpi --package stitch

# Or manual CMake
mkdir -p build && cd build
cmake -DSPPARKS_MACHINE=mpi -DSPPARKS_PACKAGE_STITCH=ON ..
cmake --build . -j$(nproc)
```

### With Traditional Makefile

```bash
# Activate Spack environment
spack env activate spparks-dev

# Build
cd src
make -j12 mac_mpi  # macOS
# OR
make -j12 mpi      # Linux

# Test
cd ../examples/ReducedTempAM/QuatTest
mpirun -np 6 ../../../src/spk_mac_mpi < in.additive
```

### Verifying Dependencies

```bash
# Check that CMake finds dependencies
spack env activate spparks-dev
cd src/build
cmake .. -DSPPARKS_MACHINE=mpi

# Look for successful finds:
# -- Found MPI_CXX: ...
# -- Found HDF5: ...
# -- Found JPEG: ...
# -- Found PNG: ...
# -- Found ZLIB: ...
```

## Advanced Topics

### Updating Dependencies

```bash
# Activate environment
spack env activate spparks-dev

# Update to newer versions (within spec constraints)
spack concretize -f
spack install

# Update lockfile
cp spack.lock spack-$(uname -s)-$(uname -m).lock
```

### Adding New Dependencies

Edit `spack.yaml` and add to `specs:` section:
```yaml
specs:
  - openmpi@4.1:+cxx
  - hdf5@1.12:+mpi+hl+szip ^openmpi
  # Add new package here:
  - boost@1.80:+mpi
```

Then:
```bash
spack concretize -f
spack install
```

### Using Binary Caches

Speed up installation by using pre-built binaries:

```bash
# Add official Spack binary mirror
spack mirror add binary_mirror https://binaries.spack.io/releases/v0.21

# Trust GPG keys
spack buildcache keys --install --trust

# Install will now use binaries when available
spack install
```

### Creating Your Own Binary Cache

Share builds across machines:

```bash
# On build machine: create cache
mkdir -p /shared/spack-cache
spack env activate spparks-dev
spack buildcache push --mirror-url /shared/spack-cache --unsigned spparks-dev

# On other machines: use cache
spack mirror add lab-cache /shared/spack-cache
spack install  # Will use cached binaries
```

### Cleaning Up

```bash
# Remove specific environment
spack env remove spparks-dev

# Clean up old package versions
spack gc

# Clean up build stage directories
spack clean -a
```

### Troubleshooting

**Issue**: CMake can't find MPI
```bash
# Solution: Verify environment is activated
spack env status

# Check CMAKE_PREFIX_PATH
echo $CMAKE_PREFIX_PATH

# Should include: .../.spack-env/view
```

**Issue**: HDF5 is not parallel
```bash
# Solution: Verify HDF5 variant
spack find -v hdf5

# Should show: hdf5@1.14.0+mpi+hl+szip
# If not, edit spack.yaml and reinstall
```

**Issue**: HighFive FetchContent conflicts
```bash
# Solution: Use external HighFive option
cd src/build
cmake .. -DSPPARKS_USE_EXTERNAL_HIGHFIVE=ON
```

**Issue**: Module command not found
```bash
# Solution: Ensure Spack environment is sourced
source ~/spack/share/spack/setup-env.sh
```

## Integration with Existing HPC Systems

### Using System MPI

If your HPC system provides MPI via modules:

```yaml
# Edit spack.yaml, add to packages section:
packages:
  openmpi:
    externals:
      - spec: openmpi@4.1.4
        modules: [openmpi/4.1.4]
    buildable: false
```

Then rebuild:
```bash
spack concretize -f
spack install
```

### Hybrid Approach

Use system modules for MPI, Spack for everything else:

```bash
# Load system MPI
module load openmpi/4.1.4

# Activate Spack environment
spack env activate spparks-dev

# Build SPPARKS - will use system MPI
cd src
./build_cmake.sh -m mpi
```

## Quick Reference

```bash
# Environment management
spack env list                          # List environments
spack env activate spparks-dev          # Activate
spack env deactivate                    # Deactivate
spack env status                        # Show active environment

# Package management
spack find                              # List installed packages
spack find -v                           # List with variants
spack spec openmpi                      # Show package details
spack info hdf5                         # Show package information

# Installation
spack install                           # Install all specs
spack install -j12                      # Parallel build (12 cores)
spack uninstall <package>               # Remove package

# Modules
spack module tcl refresh                # Generate modules
spack module tcl find openmpi           # Find module file
spack module tcl loads openmpi          # Show load commands

# Maintenance
spack gc                                # Garbage collect unused packages
spack clean -a                          # Clean build stages
```

## Support

For Spack-specific issues:
- Documentation: https://spack.readthedocs.io
- GitHub: https://github.com/spack/spack

For SPPARKS build issues:
- Check `CLAUDE.md` in this repository
- Verify dependencies with `spack find -v`
- Ensure environment is activated: `spack env status`
