# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build System

SPPARKS supports both traditional Makefile and modern CMake build systems:

## Repository Location

Primary repository: `/Users/Tron/spparks`

**Related repositories:**
- **Microstructure Analysis Scripts**: `/Users/Tron/spparks-micro-analysis`
  - Previously located in `micro_analysis_scripts/` (extracted November 2025)
  - Contains Python tools for analyzing SPPARKS output (IPF, KAM, GND, thermal analysis)
  - Requires SPPARKS STITCH library
  - See separate CLAUDE.md in that repository for analysis-specific guidance

## Building to test Claude Code changes

**Quick build for testing (Homebrew dependencies):**
```bash
cd /Users/Tron/spparks/src
make -j12 mac_arm
```

**Full-featured build with all packages:**
```bash
cd /Users/Tron/spparks/src
./build_cmake.sh --highfive --package stitch --hdf5 --jpeg -m mpi
```

**Test changes:**
Use mpirun with 6 cores and the `/Users/Tron/spparks/examples/ReducedTempAM/QuatTest/in.additive` input file. Let tests run for at least 30 seconds.

### CMake Build (Recommended)

```bash
# Quick start - build with auto-detected settings
./build_cmake.sh

# Machine-specific builds
./build_cmake.sh -m mac                    # macOS
./build_cmake.sh -m mpi                    # MPI build
./build_cmake.sh -m linux                  # Linux

# Advanced options
./build_cmake.sh -m mpi --package stitch   # With packages
./build_cmake.sh --lib --shared            # Shared library
./build_cmake.sh --clean -b Debug          # Debug build

# Manual CMake
mkdir build && cd build
cmake -DSPPARKS_MACHINE=mpi -DSPPARKS_PACKAGE_STITCH=ON ..
cmake --build .
```

### Traditional Makefile Build

```bash
# Build for a specific machine configuration
make machine

# Common configurations:
make mac          # macOS with no MPI
make mac_mpi      # macOS with MPI
make linux        # Linux
make mpi          # Generic MPI build
make serial       # Serial (no MPI) build

# Build as library
make mode=lib machine
make mode=shlib machine    # shared library

# Clean builds
make clean-all             # delete all object files
make clean-machine         # delete specific machine objects
```

### Build System Features

Both systems support:
- Auto-generation of `style_*.h` headers for dynamic class registration
- Machine-specific compiler and library configurations
- Package management (install/remove optional components)
- Static and shared library builds
- MPI and serial configurations

CMake advantages:
- Modern dependency management
- Better IDE integration
- Cross-platform compatibility
- Package config file generation
- Easier customization

### Spack Environment Management

SPPARKS includes `spack.yaml` for reproducible dependency management across systems.

**Quick Start:**
```bash
# One-time setup (if Spack not installed)
git clone https://github.com/spack/spack.git ~/spack
source ~/spack/share/spack/setup-env.sh

# Create and activate environment
cd /Users/Tron/spparks
spack env create spparks-dev spack.yaml
spack env activate spparks-dev
spack install -j 4  # Install all dependencies

# Build SPPARKS with Spack dependencies
cd src
./build_cmake.sh --highfive --package stitch --hdf5 --jpeg -m mpi
```

**Python Compatibility:**
- Spack 1.0.2 (Homebrew): Requires Python ≤ 3.12 (incompatible with 3.13+)
- Spack develop (source): Compatible with Python 3.13+
- Recommended: Use Python 3.10-3.12 for stability

**Common Issues:**
- If `berkeley-db` download stalls: `brew install berkeley-db && spack external find berkeley-db`
- Increase download timeout: Add `connect_timeout: 120` to `spack config edit config`
- Use Homebrew fallback: Install dependencies via Homebrew, then use `spack external find`

**Deactivate environment:**
```bash
spack env deactivate
```

## Package Development Workflow

SPPARKS uses a dual-directory package system inherited from LAMMPS:

### ⚠️ CRITICAL: Two-Location System

**Package files exist in TWO locations:**

1. **`src/PACKAGE/`** - **Reference version** (tracked in git) ← **EDIT HERE!**
2. **`src/`** - **Installed copy** (ignored by git) ← **DON'T EDIT!**

### 🔑 GOLDEN RULE

**ALWAYS edit `src/PACKAGE/` files, NEVER edit `src/` installed copies!**

Changes to installed files (`src/*.cpp`, `src/*.h`) will be **LOST** on package reinstall.

### How It Works

1. **Package source** lives in `src/STITCH/`, `src/KOKKOS/`, etc.
2. **Installation** copies files to `src/` for compilation
3. **Git tracks** only `src/PACKAGE/` (installed files ignored via `.gitignore`)
4. **Compilation** uses files from `src/`

### Common Mistake (AVOID!)

```bash
# ❌ WRONG - editing installed file
vim src/app_potts_kokkos.cpp

# ✅ CORRECT - editing source file
vim src/KOKKOS/app_potts_kokkos.cpp
```

### Package Management Commands

**CMake:**
```bash
# Install package (copies src/PACKAGE/ → src/)
./build_cmake.sh --package stitch

# Check if files are in sync
cmake --build build --target package-status

# Show detailed differences
cmake --build build --target package-diff

# Reinstall package (updates installed copies)
cmake ..
```

**Makefile:**
```bash
# Install package
make yes-stitch
make yes-kokkos

# Uninstall package
make no-stitch
make no-kokkos

# Check sync status
make package-status

# Show differences (after implementing Phase 1)
make package-diff
```

### Development Workflow

**Typical development cycle:**

1. Edit source file: `src/KOKKOS/app_potts_kokkos.cpp`
2. Reinstall package: `cmake ..` or `make yes-kokkos`
3. Build: `cmake --build build` or `make`
4. Test
5. Commit (only `src/KOKKOS/` files are tracked)

**If you accidentally edited `src/` instead:**

1. Copy your changes to `src/PACKAGE/`
2. Reconfigure to reinstall: `cmake ..`
3. Verify with `cmake --build build --target package-status`

### Package Sync Detection

CMake automatically warns if installed files differ from package source:

```
╔═══════════════════════════════════════════════════════════════
║ PACKAGE SYNC WARNING: kokkos package files differ!
╠═══════════════════════════════════════════════════════════════
║ The following installed files differ from package source:
║   • app_potts_kokkos.cpp
║   • app_potts_kokkos.h
║
║ GOLDEN RULE: Always edit src/KOKKOS/ files, not src/ copies!
╚═══════════════════════════════════════════════════════════════
```

## Architecture Overview

SPPARKS is a **Stochastic Parallel PARticle Kinetic Simulator** with a modular, object-oriented C++ design:

### Core Class Hierarchy

```
SPPARKS (main simulation object)
├── Universe (MPI communication management)  
├── Input (command parsing)
├── Memory (memory allocation)
├── Error (error handling)
├── Domain (spatial decomposition, regions, lattices)
├── App (physics applications - pure virtual base)
│   ├── AppLattice (lattice-based simulations)
│   │   ├── AppPotts (Potts model grain growth)
│   │   ├── AppIsing (Ising model)
│   │   ├── AppDiffusion (diffusion processes)
│   │   └── App*AM* (additive manufacturing models)
│   └── AppOffLattice (off-lattice particle simulations)
├── Solve (kinetic Monte Carlo solvers)
├── Potential (interatomic potentials)
├── Output (diagnostics and dump files)
└── Timer (performance timing)
```

### Key Design Patterns

1. **Plugin Architecture**: Apps, solvers, diagnostics registered via macros in `style_*.h` files
2. **MPI Parallelization**: Domain decomposition with ghost site communication
3. **Polymorphic Apps**: Three app classes (GENERAL, LATTICE, OFF_LATTICE) with virtual interfaces
4. **KMC/rKMC Methods**: Both kinetic Monte Carlo and rejection kinetic Monte Carlo algorithms

### App Types

- **AppLattice**: Fixed lattice sites with neighbor connectivity
  - Supports sectors, coloring, masking for parallel efficiency
  - KMC and rejection KMC methods
  - Examples: Potts, Ising, diffusion, additive manufacturing

- **AppOffLattice**: Free particles in continuous space
  - Spatial binning for neighbor finding
  - Dynamic site creation/deletion
  - Examples: molecular dynamics-like simulations

## Key Concepts

### Kinetic Monte Carlo (KMC)
- **Propensity**: Event rates computed per site via `site_propensity()`
- **Events**: State changes via `site_event()` 
- **Solvers**: Tree, linear, group-based KMC algorithms
- **Sectors**: Domain partitioning for parallel KMC

### Site Management
- **Sites**: Fundamental simulation entities with coordinates and properties
- **Ghost Sites**: Boundary sites from neighboring processors
- **Site Arrays**: `iarray` (integers), `darray` (doubles) per site

### Time Integration
- `dt_sweep`: Rejection KMC time per sweep
- `dt_kmc`: KMC time per sector update  
- `dt_step`: Time per global KMC step

## Development Guidelines

### Adding New Apps
1. Inherit from `AppLattice` or `AppOffLattice`
2. Implement pure virtual functions: `site_energy()`, `site_propensity()`, `site_event()`
3. Register via `AppStyle(name,Class)` macro
4. Override `init_app()`, `setup_app()` as needed

### File Naming Conventions
- Apps: `app_*.cpp/.h`
- Solvers: `solve_*.cpp/.h` 
- Diagnostics: `diag_*.cpp/.h`
- Dumps: `dump_*.cpp/.h`
- Regions: `region_*.cpp/.h`

### Testing
```bash
# Run test suite (undocumented feature)
make test
```

### Python environment

Use ```conda activate WORK``` to activate the correct Python environment.

cppyy can be used to test c++ code with Python if applicable.

### Tool usage

If a tool is not currently installed in the environment, ask the user to install it.

### Common Commands

**CMake Package Management:**
```bash
# Install/remove packages via CMake
cmake -DSPPARKS_PACKAGE_STITCH=ON ..     # Enable STITCH
cmake -DSPPARKS_PACKAGE_STITCH=OFF ..    # Disable STITCH

# Using build script
./build_cmake.sh --package stitch        # Enable package
```

**HighFive Usage:**
```bash
# CMake build with HighFive C++ HDF5 wrapper for improved chunk handling
./build_cmake.sh --highfive              # Enable HighFive
./build_cmake.sh --highfive --hdf5       # Enable both HDF5 and HighFive
cmake -DSPPARKS_ENABLE_HIGHFIVE=ON ..    # Direct CMake option

# Traditional Makefile build with HighFive (local installation)
./install_highfive_local.sh              # Install HighFive locally (recommended)
make mac_arm_highfive_local               # Build with local HighFive installation
```

**Makefile Package Management:**
```bash
# Package installation/removal
make yes-stitch    # install STITCH package
make no-stitch     # remove STITCH package

# Style header generation (run automatically during build)
sh Make.sh style

# MPI stubs for serial builds
make mpi-stubs
```

**CMake Configuration Options:**
```bash
# Feature toggles (ON = enabled by default if library found)
-DSPPARKS_ENABLE_MPI=ON/OFF              # MPI support (default: ON)
-DSPPARKS_ENABLE_GZIP=ON/OFF             # GZIP compression (default: ON)
-DSPPARKS_ENABLE_JPEG=ON/OFF             # JPEG image support (default: ON)
-DSPPARKS_ENABLE_PNG=ON/OFF              # PNG image support (default: OFF)
-DSPPARKS_ENABLE_HDF5=ON/OFF             # HDF5 data format support (default: OFF)
-DSPPARKS_ENABLE_HIGHFIVE=ON/OFF         # HighFive C++ HDF5 wrapper (default: OFF)

# Integer size options
-DSPPARKS_SMALLSMALL=ON                  # 32-bit all integers
-DSPPARKS_BIGBIG=ON                      # 64-bit tagint/bigint
# Default: SMALLBIG (32-bit smallint/tagint, 64-bit bigint)

# Build options
-DBUILD_SHARED_LIBS=ON/OFF               # Shared vs static
-DSPPARKS_BUILD_EXECUTABLE=ON/OFF        # Build spk executable
-DSPPARKS_BUILD_LIB=ON/OFF               # Build library

# Feature detection notes:
# - JPEG: Automatically enabled if libjpeg found, disable with --no-jpeg
# - HDF5: Automatically detects parallel vs serial HDF5
# - HighFive: Modern C++ wrapper for HDF5, automatically enables HDF5
# - MPI: Warns if MPI is enabled but HDF5 is not parallel
# - Used by additive manufacturing applications for temperature data
```

## Important Files

**Core Framework:**
- `spparks.h/.cpp`: Main simulation driver
- `app.h`: Base application class
- `solve.h`: Base KMC solver class  
- `domain.h`: Spatial domain management
- `pointers.h`: Base class providing access to SPPARKS object
- `spktype.h`: Type definitions (tagint, bigint, etc.)

**Build System:**
- `Makefile`: Traditional multi-machine build system
- `CMakeLists.txt`: Modern CMake build configuration
- `build_cmake.sh`: Convenient CMake build script
- `Make.sh`: Dynamic header generation (Makefile)
- `cmake/generate_styles.cmake`: Dynamic header generation (CMake)
- `cmake/MachineConfigs.cmake`: Machine-specific settings
- `cmake/PackageManager.cmake`: Package installation system
- `style_*.h`: Auto-generated class registration headers

**Package & Dependency Management:**
- `STITCH/`: Optional STITCH package for external coupling
- `Makefile.package*`: Package configuration for Makefile system
- `spack.yaml`: Spack environment specification for reproducible dependency management
- `README_SPACK.md`: Detailed Spack usage guide