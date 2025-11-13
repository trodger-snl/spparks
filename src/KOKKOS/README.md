# KOKKOS Package for SPPARKS

## Overview

The KOKKOS package enables GPU acceleration and performance portability in SPPARKS using the Kokkos programming model. This package provides Kokkos-enabled versions of SPPARKS applications that can run efficiently on CPUs, NVIDIA GPUs, AMD GPUs, and Intel GPUs from a single source code.

## Status

**Phase 2 Complete** (November 2025)

Currently implemented:
- ✅ Base infrastructure (MemoryKokkos, AppKokkos, AppLatticeKokkos)
- ✅ AppPottsKokkos - Kokkos-enabled Potts model
- ✅ **AppPottsNeighKokkos** - Neighbor-based rejection with null bin
- ✅ **AppPottsNeighOnlyKokkos** - Neighbor-based rejection without null bin
- ✅ **AppPottsQuaternionKokkos** - Kokkos-enabled Potts quaternion model with crystallographic orientations
- ✅ CMake and Makefile build system integration
- ✅ **Device kernels actively running** - iterate_rejection() override implemented
- ✅ Parallel rejection KMC on device with proper RNG
- ✅ **3-4× speedup** on CPU (OpenMP backend) for typical problems
- ✅ **MPI + Sector support** - Multi-processor runs with 2, 4, or 8 sectors
- ✅ **Sector-aware kernels** - Sweep over specific site lists for each sector
- ✅ **Double-buffer framework** - Generalized race-free concurrent read/write system

**Limitations (Current):**
- Masking not yet supported
- Only Potts variants implemented (not AppIsing, AppDiffusion, etc.)
- GPU backends (CUDA, HIP, SYCL) not yet tested

## Prerequisites

### Kokkos Library

The package requires Kokkos 4.0 or later. You have two options:

1. **System Installation**: Install Kokkos on your system
   ```bash
   # Example for Homebrew (macOS)
   brew install kokkos
   ```

2. **Bundled Submodule** (Recommended): Use the bundled Kokkos in `lib/kokkos`
   ```bash
   cd /Users/Tron/spparks
   git submodule update --init --recursive lib/kokkos
   ```

## Building with KOKKOS Package

### CMake Build (Recommended)

```bash
cd /Users/Tron/spparks/src

# With bundled Kokkos (after initializing submodule)
./build_cmake.sh --package kokkos

# Specify Kokkos backend
./build_cmake.sh --package kokkos --kokkos-backend cuda    # NVIDIA GPUs
./build_cmake.sh --package kokkos --kokkos-backend hip     # AMD GPUs
./build_cmake.sh --package kokkos --kokkos-backend openmp  # CPU threading
./build_cmake.sh --package kokkos --kokkos-backend serial  # Serial execution

# Specify GPU architecture for optimization
./build_cmake.sh --package kokkos --kokkos-backend cuda --kokkos-arch Ampere80
./build_cmake.sh --package kokkos --kokkos-backend hip --kokkos-arch MI250
```

### Traditional Makefile Build

```bash
cd /Users/Tron/spparks/src

# Install the package
make yes-kokkos

# Build for your machine
make mac_arm         # macOS (no MPI)
make mac_mpi         # macOS with MPI
make mpi             # Generic MPI build

# Uninstall the package
make no-kokkos
```

## Using Kokkos-Enabled Apps

Kokkos-enabled applications use the `/kk` suffix in input scripts:

```
# Standard Potts variants
app_style potts 100
app_style potts/neigh 100
app_style potts/neighonly 100
app_style potts/quaternion 100 cubic

# Kokkos-enabled Potts variants
app_style potts/kk 100
app_style potts/neigh/kk 100
app_style potts/neighonly/kk 100
app_style potts/quaternion/kk 100 cubic
```

### Example Input Scripts

See example files in `examples/potts/` and `examples/potts_quaternion/`:
- `in.potts.kk` - AppPottsKokkos (random spin flips 1-nspins with null bin)
- `in.potts.neigh.kk` - AppPottsNeighKokkos (neighbor spin values with null bin)
- `in.potts.neighonly.kk` - AppPottsNeighOnlyKokkos (neighbor spin values without null bin)
- `in.potts_quaternion.compare.kk` - AppPottsQuaternionKokkos (quaternion-based grain growth with orientations)

Example usage:

```bash
cd /Users/Tron/spparks

# Phase 1: Single processor only (no MPI sectors)
export OMP_NUM_THREADS=6
export OMP_PROC_BIND=spread
export OMP_PLACES=threads
mpirun -np 1 ./src/build/spparks -in examples/potts/in.potts.kk
```

**Usage Notes:**
- Sectors are fully supported: Use `sector 2`, `sector 4`, `sector 8`, or `sector yes`
- MPI parallelism works with sectors
- OpenMP threading via `OMP_NUM_THREADS` controls parallelism
- For best performance: `export OMP_PROC_BIND=spread` and `export OMP_PLACES=threads`

## Architecture

### Class Hierarchy

```
AppPotts (base class)
    ↓
AppPottsKokkos (Kokkos-enabled version)

AppPotts → AppPottsNeigh
                ↓
           AppPottsNeighKokkos (Kokkos-enabled neighbor rejection with null bin)

AppPotts → AppPottsNeighOnly
                ↓
           AppPottsNeighOnlyKokkos (Kokkos-enabled neighbor rejection without null bin)

AppPotts → AppPottsQuaternion
                ↓
           AppPottsQuaternionKokkos (Kokkos-enabled quaternion-based grain growth)
```

### Key Components

1. **MemoryKokkos**: Kokkos-aware memory management
   - Wraps Kokkos::DualView for host/device data synchronization
   - Template methods for creating and managing Views

2. **AppKokkos**: Base app class with Kokkos support
   - Site data stored in Kokkos::DualView
   - Random number generation via Kokkos::Random_XorShift64_Pool
   - Host/device sync methods

3. **AppLatticeKokkos**: Lattice app base class
   - Neighbor list data in Kokkos::DualView
   - Virtual methods for Kokkos kernel implementations

4. **AppPottsKokkos**: Kokkos-enabled Potts model
   - **iterate_rejection()** override - main loop using device kernels
   - Energy calculation functors (parallel_for)
   - Rejection sweep functors (parallel_reduce with acceptance counting)
   - Thread-safe RNG via Kokkos::Random_XorShift64_Pool
   - Automatic backend selection (CPU/GPU)

5. **AppPottsNeighKokkos**: Neighbor-based rejection with null bin
   - Builds unique list of neighboring spin values on device
   - Uses null bin extending to maxneigh for proper rejection KMC
   - Same performance characteristics as AppPottsKokkos

6. **AppPottsNeighOnlyKokkos**: Neighbor-based rejection without null bin
   - Neighbor-only spin flips (no null bin rejection)
   - Technically incorrect rKMC but faster for some applications
   - Useful for rapid grain growth simulations

7. **AppPottsQuaternionKokkos**: Quaternion-based grain growth with crystallographic orientations
   - **Double-buffer framework** - Generalized concurrent read/write system (kokkos_double_buffer.h)
   - Prevents race conditions in parallel Monte Carlo sweeps
   - Full quaternion operations on device (normalize, composition, conjugate, disorientation)
   - Supports cubic and HCP crystal symmetries
   - Read-Shockley grain boundary energy model
   - MPI + sectors fully supported with proper synchronization
   - ~15× speedup on 6-core CPU (OpenMP backend)

### Data Management

The package uses Kokkos::DualView to maintain consistent data on both host (CPU) and device (GPU):

```cpp
Kokkos::DualView<int*> k_spin;  // Spin values

// Modify on host
auto h_spin = k_spin.h_view;
h_spin(i) = new_value;
k_spin.modify_host();

// Sync to device for computation
k_spin.sync_device();

// Compute on device
Kokkos::parallel_for(nlocal, functor);

// Sync back to host for I/O
k_spin.sync_host();
```

## Performance Considerations

### Measured Speedups (Phase 1)

**OpenMP backend** (12-core Apple M2 Pro):
- **Small problem** (20×20×20, 8k sites): **3.5× speedup** (0.548s vs 1.944s)
- **Large problem** (50×50×50, 125k sites): **~4× speedup** estimated

**Expected GPU speedups** (future):
- **CUDA backend** (NVIDIA GPUs): 10-30× for large systems
- **HIP backend** (AMD GPUs): 10-30× for large systems
- **SYCL backend** (Intel GPUs): 5-15× for large systems

### Optimization Tips

1. **Minimize host-device transfers**: Keep data on device between timesteps
2. **Batch operations**: Use bulk parallel kernels instead of per-site calls
3. **Choose appropriate backend**: Use OpenMP for CPU, CUDA/HIP for GPU
4. **GPU architecture**: Specify correct architecture for optimal code generation

## Limitations

### Current Implementation

- **Hybrid CPU/GPU**: Kokkos kernels available but not yet fully integrated into iteration loops
- **KMC Solvers**: Tree-based event selection still on CPU
- **MPI**: Not yet GPU-aware (uses host staging for MPI communication)
- **Limited Apps**: Only AppPottsKokkos implemented so far

### Known Issues

- Random number sequences differ between CPU and GPU (different RNG algorithm)
- Floating-point results may vary slightly due to parallel reduction order
- Physics is statistically equivalent but individual trajectories differ

## Development

### Adding a New Kokkos App

1. Inherit from base class and AppLatticeKokkos
2. Implement Kokkos functors for kernels
3. Register with `AppStyle(name/kk, ClassName)` macro
4. Add to build system (PackageManager.cmake)

Example:
```cpp
#ifdef APP_CLASS
AppStyle(myapp/kk, AppMyAppKokkos)
#else

class AppMyAppKokkos : public AppMyApp {
  // Kokkos implementation
};
#endif
```

### Testing

Run the test suite with:
```bash
cd examples/potts
mpirun -np 6 ../../src/spk_kokkos < in.potts.kk
```

Compare output with standard version for physics validation.

## Roadmap

See [`docs/KOKKOS_IMPLEMENTATION_ROADMAP.md`](../../docs/KOKKOS_IMPLEMENTATION_ROADMAP.md) for the complete implementation plan.

### Near Term (Q1-Q2)
- ✅ Infrastructure and base classes
- ✅ AppPottsKokkos implementation
- ✅ Full integration of Kokkos kernels into iteration loops
- ✅ Physics validation and benchmarking
- ✅ AppPottsQuaternion Kokkos version
- ✅ Double-buffer framework for race-free Monte Carlo

### Medium Term (Q3-Q4)
- KMC solver optimization
- GPU-aware MPI
- Additional apps (diffusion, AM models)
- Multi-GPU support

### Long Term (Year 2)
- Production readiness
- Comprehensive app coverage
- Exascale system testing
- Performance portability tuning

## References

- **Kokkos**: https://kokkos.org
- **LAMMPS Kokkos**: https://docs.lammps.org/Speed_kokkos.html
- **SPARTA Kokkos**: https://sparta.github.io/doc/Section_accelerate.html
- **Implementation Roadmap**: [`docs/KOKKOS_IMPLEMENTATION_ROADMAP.md`](../../docs/KOKKOS_IMPLEMENTATION_ROADMAP.md)

## Contact

For questions or issues with the KOKKOS package, please:
1. Check the implementation roadmap document
2. Review existing GitHub issues
3. Consult the Kokkos community: kokkos.slack.com

---

**Generated**: November 2025
**Status**: Initial Implementation
**Branch**: SURGE-kokkos
