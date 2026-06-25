# Kokkos Installation Status

## AMD GPU (HIP / gfx1151) — verified working

**Date:** June 24, 2026
**Branch:** SURGE-kokkos
**System:** Linux (Fedora), AMD Ryzen AI Max+ 395 / Radeon 8060S APU (gfx1151, RDNA 3.5), ROCm/HIP 7.1

✅ **SPPARKS builds and runs on the GPU with the Kokkos HIP backend.** A `potts/kk`
run was confirmed on the GPU (`Kokkos::HIP gcnArch gfx1151 : Selected`, Wavefront
Size 32, energy relaxes correctly).

**Kokkos version:** 5.1.1 (submodule tag `5.1.1`). Requires C++20 and CMake >= 3.22.

**Key point — gfx1151 support:** No released Kokkos through 5.1.1 lists gfx1150/gfx1151
in `SUPPORTED_AMD_ARCHS` (it is in Kokkos `develop`, headed for a future release).
The bundled `patches/kokkos-gfx1151.patch` backports the arch + the correct 32-wide
RDNA wavefront size, and `src/CMakeLists.txt` auto-applies it to the submodule when
`SPPARKS_KOKKOS_ARCH` matches `AMD_GFX115x` and Kokkos lacks it.

**Note on 5.1.1 vs 4.7.01:** Benchmarking `potts/kk` on gfx1151 showed *identical* GPU
solve time between Kokkos 4.7.01 and 5.1.1 (no performance change for this workload);
the bump was taken for future-proofing. Moving to 5.x required C++20, CMake >= 3.22,
and migrating the KOKKOS package off the removed `DualView::h_view`/`d_view` and
`View::HostMirror` APIs (now `view_host()`/`view_device()` and `host_mirror_type`).

**Build recipe (from repo root):**
```bash
git submodule update --init lib/kokkos          # CMake auto-applies the gfx1151 patch
spack env activate spparks-dev                   # provides MPI/HDF5/HighFive/JPEG/PNG
cmake -S src -B build-hip \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=hipcc \
  -DSPPARKS_ENABLE_MPI=ON \
  -DSPPARKS_ENABLE_HDF5=ON -DSPPARKS_ENABLE_HIGHFIVE=ON \
  -DSPPARKS_ENABLE_JPEG=ON -DSPPARKS_ENABLE_PNG=ON \
  -DSPPARKS_PACKAGE_KOKKOS=ON \
  -DSPPARKS_KOKKOS_BACKEND=HIP -DSPPARKS_KOKKOS_ARCH=AMD_GFX1151
make -C build-hip -j
./build-hip/spparks -in examples/kokkos/in.potts_kk    # smoke test on the GPU
```

**Notes / gotchas:**
- KOKKOS apps require `sweep color` (not `sweep random`). Styles: `potts/kk`,
  `potts/neigh/kk`, `potts/neighonly/kk`, `potts/quaternion/kk`,
  `additive_temperature_texture/kk`.
- Disabling HDF5/JPEG/PNG causes link errors (`DumpImage`,
  `AppAdditiveExtTempTexture*`) because the style-header generator registers
  those styles even when their sources are excluded — keep them enabled.
- Memory model: gfx1151 supports managed memory (`Kokkos::SharedSpace`,
  effectively zero-copy on this shared-DRAM APU) but NOT XNACK system-allocation
  access, so raw host pointers cannot be dereferenced in kernels — allocate
  through Kokkos.

---

## macOS ARM (OpenMP) — historical

**Date:** November 12, 2025
**Branch:** SURGE-kokkos
**System:** macOS ARM (Apple Silicon)

## Installation Summary

✅ **Kokkos 4.7.01 bundled as git submodule (recommended approach)**

## Installation Details

**Version:** 4.7.01 (stable release)
**Installation Method:** Git submodule (bundled with SPPARKS)
**Location:** `/Users/Tron/spparks/lib/kokkos`
**Commit:** `b601b82d0` (tag 4.7.01)

**Previous Installation:**
- ~~Homebrew (`/opt/homebrew/Cellar/kokkos/4.7.01`)~~ - Uninstalled
- Removed due to CMake SDK path conflicts on macOS

**For New Clones:**
Other users cloning SPPARKS with Kokkos need to initialize the submodule:
```bash
git clone <spparks-repo> --branch SURGE-kokkos
cd spparks
git submodule update --init --recursive
```

Or if already cloned:
```bash
cd /Users/Tron/spparks
git submodule update --init --recursive
```

**Enabled Backends:**
- ✅ **OpenMP** (CPU multi-threading)
- ❌ CUDA (NVIDIA GPUs - not available on macOS)
- ❌ HIP (AMD GPUs - not available on macOS)
- ❌ SYCL (Intel GPUs - not available on macOS)
- ❌ Serial (single-threaded fallback - not enabled)

**Configuration:**
- C++ Standard: C++17
- OpenMP Threads: 12 (matches system cores)
- Thread Pool Topology: 1 x 12 x 1

**Libraries Installed:**
```
/opt/homebrew/Cellar/kokkos/4.7.01/lib/
├── libkokkoscore.a        (1.0 MB - main library)
├── libkokkoscontainers.a  (7 KB - container utilities)
├── libkokkosalgorithms.a  (< 1 KB - algorithms)
└── libkokkossimd.a        (< 1 KB - SIMD support)
```

**Include Files:**
- Headers: `/opt/homebrew/Cellar/kokkos/4.7.01/include/`
- Symlink: `/opt/homebrew/include/` (via Homebrew)

**CMake Configuration:**
- Config files: `/opt/homebrew/lib/cmake/Kokkos/`
- Package name: `Kokkos` (use `find_package(Kokkos REQUIRED)`)

## Verification Test Results

A simple Kokkos test program was compiled and executed successfully:

**Test Program:**
```cpp
// Parallel computation test
Kokkos::parallel_for("initialize", N, KOKKOS_LAMBDA(const int i) {
    a(i) = i * 2.0;
});

// Parallel reduction test
Kokkos::parallel_reduce("sum", N, KOKKOS_LAMBDA(...) {...}, sum);
```

**Results:**
```
✓ Kokkos is working correctly!
✓ Parallel computations executing on 12 OpenMP threads
✓ Parallel reductions producing correct results
```

## Compilation Instructions

### Method 1: Direct Compilation with g++

```bash
g++ -std=c++17 \
    -I/opt/homebrew/include \
    -L/opt/homebrew/lib \
    -I/opt/homebrew/opt/libomp/include \
    -L/opt/homebrew/opt/libomp/lib \
    -Xpreprocessor -fopenmp \
    your_code.cpp \
    -lkokkoscore -lkokkoscontainers -lomp \
    -o your_program
```

**Notes:**
- macOS uses Clang as default compiler (not GCC)
- `-Xpreprocessor -fopenmp` required for Clang on macOS
- Must link with `-lomp` (Homebrew libomp)

### Method 2: CMake (Has Issues on macOS)

⚠️ **Note:** CMake's `find_package(Kokkos)` has SDK path conflicts on macOS with Homebrew Kokkos.

**The issue:**
- CMake can find Kokkos package ✅
- Compilation fails due to SDK header search path ordering ❌

**Workaround - Manual Linking:**
```cmake
cmake_minimum_required(VERSION 3.16)
project(YourProject CXX)

set(CMAKE_CXX_STANDARD 17)

# Manually specify Kokkos (works on macOS)
include_directories(/opt/homebrew/include)
include_directories(/opt/homebrew/opt/libomp/include)
link_directories(/opt/homebrew/lib)
link_directories(/opt/homebrew/opt/libomp/lib)

add_executable(your_program your_code.cpp)
target_compile_options(your_program PRIVATE -Xpreprocessor -fopenmp)
target_link_libraries(your_program kokkoscore kokkoscontainers omp)
```

## Recommendations for SPPARKS Integration

### Recommended Approach: Bundle Kokkos (Like LAMMPS)

To avoid macOS SDK path issues, **bundle Kokkos** as a git submodule (recommended for SPPARKS):

```bash
# Add Kokkos as submodule (do this once)
cd /Users/Tron/spparks
git submodule add https://github.com/kokkos/kokkos.git lib/kokkos
git submodule update --init --recursive
```

**In `CMakeLists.txt`:**
```cmake
option(SPPARKS_PACKAGE_KOKKOS "Enable KOKKOS package" OFF)

if(SPPARKS_PACKAGE_KOKKOS)
    # Set Kokkos options before adding subdirectory
    set(Kokkos_ENABLE_OPENMP ON CACHE BOOL "Enable OpenMP backend")
    set(Kokkos_ENABLE_SERIAL OFF CACHE BOOL "Disable Serial backend")

    # Add bundled Kokkos
    add_subdirectory(lib/kokkos)

    # Link to SPPARKS
    target_link_libraries(spparks_lib PUBLIC Kokkos::kokkoscore)
    target_compile_definitions(spparks_lib PUBLIC SPK_KOKKOS)
endif()
```

**Benefits:**
- ✅ No SDK path conflicts
- ✅ Works on all platforms (macOS, Linux, Windows)
- ✅ Consistent Kokkos version across builds
- ✅ Users don't need to install Kokkos separately
- ✅ This is exactly what LAMMPS does

### Alternative: Use System Kokkos (Linux Only)

For Linux clusters where Kokkos is pre-installed, allow using system Kokkos:

```cmake
if(SPPARKS_PACKAGE_KOKKOS)
    option(SPPARKS_KOKKOS_EXTERNAL "Use external Kokkos" OFF)

    if(SPPARKS_KOKKOS_EXTERNAL)
        # Use system Kokkos (Linux)
        find_package(Kokkos REQUIRED)
        target_link_libraries(spparks_lib PUBLIC Kokkos::kokkos)
    else()
        # Use bundled Kokkos (all platforms)
        add_subdirectory(lib/kokkos)
        target_link_libraries(spparks_lib PUBLIC Kokkos::kokkoscore)
    endif()

    target_compile_definitions(spparks_lib PUBLIC SPK_KOKKOS)
endif()
```

### 1. Build System Updates

**In `build_cmake.sh`:**
```bash
# Add --kokkos flag support
if [[ "$*" == *"--kokkos"* ]]; then
    CMAKE_OPTIONS+="-DSPPARKS_PACKAGE_KOKKOS=ON "

    # On macOS, set OpenMP paths
    if [[ "$OSTYPE" == "darwin"* ]]; then
        export OpenMP_ROOT=/opt/homebrew/opt/libomp
        export OpenMP_CXX_FLAGS="-Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include"
        export OpenMP_CXX_LIB_NAMES="omp"
        export OpenMP_omp_LIBRARY=/opt/homebrew/opt/libomp/lib/libomp.dylib
    fi
fi
```

**In `CMakeLists.txt`:**
```cmake
option(SPPARKS_PACKAGE_KOKKOS "Enable KOKKOS package" OFF)

if(SPPARKS_PACKAGE_KOKKOS)
    # Find OpenMP first (required dependency)
    find_package(OpenMP REQUIRED COMPONENTS CXX)

    # Find Kokkos
    find_package(Kokkos REQUIRED)

    # Link to SPPARKS library
    target_link_libraries(spparks_lib PUBLIC Kokkos::kokkos OpenMP::OpenMP_CXX)
    target_compile_definitions(spparks_lib PUBLIC SPK_KOKKOS)
endif()
```

### 2. Performance Optimization

**Set OpenMP Environment Variables:**
```bash
# For best performance (recommended by Kokkos)
export OMP_PROC_BIND=spread
export OMP_PLACES=threads
export OMP_NUM_THREADS=12  # Match your core count

# For testing/debugging
export OMP_PROC_BIND=false
```

### 3. macOS-Specific Considerations

**Limitations on macOS:**
- ❌ No GPU support (CUDA/HIP/SYCL require Linux/Windows)
- ✅ OpenMP backend only (sufficient for CPU development/testing)
- ✅ Can develop Kokkos code on macOS, deploy to GPU clusters

**Development Strategy:**
1. Develop and test Kokkos integration on macOS using OpenMP backend
2. Validate correctness with CPU-only execution
3. Test GPU backends on Linux cluster when ready
4. Kokkos code will work on both CPU and GPU without changes

## Dependencies

**Kokkos requires:**
- ✅ libomp (Homebrew OpenMP) - installed
- ✅ CMake 3.16+ - available
- ✅ C++17 compiler - Apple Clang 15.0+

**SPPARKS additional requirements:**
- MPI (for multi-node parallelism) - independent of Kokkos
- OpenMP + MPI can coexist (hybrid parallelism)

## Next Steps for SPPARKS Integration

Based on the roadmap in `docs/KOKKOS_IMPLEMENTATION_ROADMAP.md`:

### Phase 1: Infrastructure (Ready to Start)

1. **Add Kokkos package to CMake build system** ✅ Kokkos installed
2. **Create MemoryKokkos class** - wrapper for Kokkos::DualView
3. **Create AppKokkos base classes** - extend existing App hierarchy
4. **Set up build scripts** - add `--kokkos` flag to `build_cmake.sh`

### Testing Command

```bash
# Build SPPARKS with Kokkos support (once implemented)
cd /Users/Tron/spparks/src
./build_cmake.sh --kokkos -m mac

# Run test (OpenMP backend)
export OMP_PROC_BIND=spread OMP_PLACES=threads
./spk_kokkos < ../examples/potts/in.potts.kk
```

## Troubleshooting

### Common Issues

**Issue: CMake can't find OpenMP**
```bash
# Solution: Set environment variables before CMake
export OpenMP_ROOT=/opt/homebrew/opt/libomp
export CMAKE_PREFIX_PATH=/opt/homebrew
```

**Issue: Linking errors with libomp**
```bash
# Solution: Explicitly link with OpenMP
target_link_libraries(your_target OpenMP::OpenMP_CXX)
```

**Issue: Runtime OpenMP warnings**
```bash
# Solution: Set OMP_PROC_BIND environment variable
export OMP_PROC_BIND=spread
export OMP_PLACES=threads
```

## Resources

**Kokkos Documentation:**
- Official: https://kokkos.org
- GitHub: https://github.com/kokkos/kokkos
- Tutorials: https://github.com/kokkos/kokkos-tutorials

**SPPARKS Kokkos Documentation:**
- Roadmap: `docs/KOKKOS_IMPLEMENTATION_ROADMAP.md`
- Main docs: `CLAUDE.md` (see "GPU Acceleration" section)

## Summary

✅ Kokkos 4.7.01 bundled as git submodule (commit b601b82d0)
✅ Homebrew version uninstalled to avoid SDK conflicts
✅ OpenMP backend available (12 threads on this system)
✅ Bundled approach tested and working (direct compilation verified)
✅ CMakeLists.txt ready for add_subdirectory() integration
✅ Ready for SPPARKS CMake integration (Phase 1)

**Bundled vs External:**
- ✅ **Bundled (current)**: Works on all platforms, no SDK issues
- ⚠️ External (Homebrew/system): CMake SDK path conflicts on macOS

⚠️ macOS limitations:
- OpenMP backend only (no GPU support)
- Sufficient for development and CPU testing
- GPU deployment requires Linux cluster

**Status: Ready to proceed with Phase 1 - Update SPPARKS CMakeLists.txt**
