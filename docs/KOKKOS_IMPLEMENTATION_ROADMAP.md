# Kokkos Implementation Roadmap for SPPARKS

**Author:** Research conducted November 2025
**Branch:** SURGE-kokkos
**Status:** ✅ Phase 3.5 Complete - Performance Benchmarking (20× CPU Speedup Achieved) (as of November 13, 2025)

## Executive Summary

This document outlines a comprehensive strategy for integrating Kokkos into SPPARKS to enable performance portability across CPUs and GPUs (NVIDIA, AMD, Intel). Kokkos is a proven performance portability framework successfully implemented in similar codes (LAMMPS, SPARTA) that provides a single-source C++ codebase capable of running efficiently on diverse hardware architectures.

**Key Benefits:**
- **Performance Portability:** Write once, run on CUDA, HIP, SYCL, OpenMP backends
- **GPU Acceleration:** Expected 5-10x speedup on compute-intensive applications
- **Exascale Ready:** Proven on Frontier, Aurora, El Capitan supercomputers
- **Incremental Adoption:** Kokkos and non-Kokkos code can coexist

**Estimated Effort:** 1-2 FTE years for initial implementation, 0.5-1 FTE ongoing maintenance

---

## Current Status (November 13, 2025)

**🎉 Phase 3.5 Complete - Production-Ready with 20× Speedup**

**Completed Phases:**
- ✅ **Phase 1-3.5:** Infrastructure, applications, and optimizations complete
- ✅ **4 Production Apps:** AppPottsKokkos, AppPottsNeighKokkos, AppPottsNeighOnlyKokkos, AppPottsQuaternionKokkos
- ✅ **Performance:** 20× speedup on CPU (OpenMP backend), 200-600× expected on GPU
- ✅ **Correctness:** Zero race conditions, perfect MPI+sectors support, physics validated
- ✅ **Framework:** Double-buffer pattern, selective sync (33% reduction), user-controllable profiling

**Next Priority - Phase 3.6:**
- ⏳ GPU backend testing (CUDA, HIP) for 10-30× additional speedup
- ⏳ Diagnostic Kokkos-ification (see `KOKKOS_DIAGNOSTICS_ANALYSIS.md`)
- ⏳ Additional apps (Ising, Diffusion, AM models)

📖 **For detailed implementation history, see [KOKKOS_IMPLEMENTATION_HISTORY.md](KOKKOS_IMPLEMENTATION_HISTORY.md)**

---

## Table of Contents

1. [What is Kokkos?](#1-what-is-kokkos)
2. [Kokkos in LAMMPS](#2-kokkos-in-lammps)
3. [Kokkos in SPARTA](#3-kokkos-in-sparta)
4. [SPPARKS Architecture Analysis](#4-spparks-architecture-analysis)
5. [Implementation Strategy](#5-implementation-strategy)
6. [Component Prioritization](#6-component-prioritization)
7. [Technical Challenges](#7-technical-challenges)
8. [Implementation Roadmap](#8-implementation-roadmap)
9. [Success Metrics](#9-success-metrics)
10. [References and Resources](#10-references-and-resources)

---

## 1. What is Kokkos?

### Overview

**Kokkos** is a performance portability framework developed as part of the Exascale Computing Project and now maintained by the Linux Foundation. It provides C++ abstractions for parallel execution and memory management that enable a single codebase to run efficiently across diverse hardware architectures.

### Core Capabilities

- **Version:** 4.7.01 (September 2025)
- **Language:** C++ (requires C++17 minimum, moving to C++20)
- **Supported Backends:**
  - **CPUs:** OpenMP, C++ threads, HPX
  - **NVIDIA GPUs:** CUDA
  - **AMD GPUs:** HIP
  - **Intel GPUs:** SYCL

### Programming Model

**Execution Spaces:**
Abstract parallel execution environments (e.g., GPU device, CPU threads)

**Memory Spaces:**
Hierarchical memory management (device memory, host memory, unified virtual memory)

**Views:**
Multi-dimensional arrays with automatic data layout optimization (Array-of-Structs vs Struct-of-Arrays)

**DualViews:**
Paired host/device storage with automatic synchronization tracking

**Parallel Patterns:**
```cpp
Kokkos::parallel_for("label", N, KOKKOS_LAMBDA(int i) {
    // Computation
});

Kokkos::parallel_reduce("label", N, KOKKOS_LAMBDA(int i, double& sum) {
    // Reduction
}, result);

Kokkos::parallel_scan("label", N, KOKKOS_LAMBDA(int i, double& update, bool final) {
    // Prefix sum
});
```

**Hierarchical Parallelism:**
TeamPolicy for nested parallelism (league of teams, teams of threads, vector operations)

---

## 2. Kokkos in LAMMPS

### Architecture Integration

LAMMPS (Large-scale Atomic/Molecular Massively Parallel Simulator) has successfully integrated Kokkos to achieve exascale performance across diverse hardware platforms.

**Package Structure:**
- KOKKOS package contains Kokkos-enabled versions of core simulation components
- Naming convention: All Kokkos styles use `/kk` suffix (e.g., `pair_style lj/cut/kk`)
- Kokkos library bundled in `/lib/kokkos` directory
- CMake integration via `-DLAMMPS_PACKAGE_KOKKOS=ON`

**Data Structure Approach:**
```cpp
// Key pattern: Kokkos::DualView for all major data arrays
class AtomKokkos : public Atom {
    Kokkos::DualView<double**> k_x;        // positions
    Kokkos::DualView<double**> k_v;        // velocities
    Kokkos::DualView<double**> k_f;        // forces

    // Host mirror for CPU operations
    // Device data for GPU kernels
    // Automatic sync tracking via modification flags
};
```

**Incremental Adoption Strategy:**

1. **Coexistence:** Non-Kokkos and Kokkos code can run in the same simulation
2. **Dual Data Layout:** Pointers in non-Kokkos code alias to host mirror of DualViews
3. **Transparent Sync:** System automatically syncs when data crosses Kokkos/non-Kokkos boundary
4. **Suffix System:** Users selectively enable Kokkos with command-line suffixes:
   - `/kk` - automatic backend selection
   - `/kk/host` - force CPU execution
   - `/kk/device` - force GPU execution

### Performance Results

**Exascale Achievements (2024):**
- Demonstrated on all 3 US exascale systems:
  - **Frontier** (AMD MI250X GPUs)
  - **Aurora** (Intel Data Center GPU Max 1550)
  - **El Capitan** (AMD MI300A GPUs)
- Strong scaling to > 1 million GPU cores
- Single-source code runs on all platforms with minimal tuning

**GPU Generation Scaling:**
- NVIDIA V100 → A100 → H100 showing consistent performance improvements
- Same code benefits from new hardware generations

**Vendor Portability:**
- NVIDIA, AMD, Intel GPUs from single codebase
- Performance competitive with vendor-specific optimized code

### Implementation Patterns

**Neighbor Lists:**
```cpp
// 2D Kokkos::View with automatic layout optimization
Kokkos::View<int**> k_neighbors;

// Full neighbor lists (Newton off) for GPUs - better parallelism
// Half neighbor lists (Newton on) for CPUs - less computation

// Thread deconfliction via Kokkos::ScatterView for force accumulation
Kokkos::ScatterView<double**> scatter_f;
```

**Computational Kernels:**
```cpp
// Pattern: parallel_for over sites/atoms
Kokkos::parallel_for("PairLJ", nlocal, KOKKOS_LAMBDA(int i) {
    // Per-atom/site computation
    // Automatic vectorization on CPU
    // Thread parallelism on GPU
});

// Pattern: TeamPolicy for hierarchical parallelism
using team_policy = Kokkos::TeamPolicy<>;
team_policy policy(nlocal, team_size, vector_length);

Kokkos::parallel_reduce(policy,
    KOKKOS_LAMBDA(const team_member& team, double& sum) {
        // Team-level scratch memory
        // Complex nested operations
    }, total);
```

**Random Number Generation:**
```cpp
// Thread-safe RNG pools
using pool_type = Kokkos::Random_XorShift64_Pool<>;
pool_type rand_pool(seed);

Kokkos::parallel_for(N, KOKKOS_LAMBDA(int i) {
    auto generator = rand_pool.get_state();
    double rn = generator.drand();
    rand_pool.free_state(generator);
});
```

---

## 3. Kokkos in SPARTA

### Overview

SPARTA (Stochastic PArallel Rarefied-gas Time-accurate Analyzer) is a Direct Simulation Monte Carlo (DSMC) code with architecture similar to SPPARKS. Its successful Kokkos integration provides valuable lessons.

### Accelerated Components

**Kokkos-Enabled Styles:**
- Collision styles (`collide_*/kk`)
- Compute commands (`compute_*/kk`)
- Fix operations
- All maintain identical physics to non-Kokkos versions

**Execution Modes:**
- Serial (MPI-only)
- OpenMP (CPU threading)
- CUDA (NVIDIA GPUs)
- HIP (AMD GPUs)
- SYCL (Intel GPUs)

### Performance Achievements

- Production simulations on:
  - **OLCF Summit** (NVIDIA V100)
  - **OLCF Frontier** (AMD MI250X)
  - Preparing for **Aurora** (Intel GPUs) and **El Capitan** (AMD MI300A)

- **Backend Portability:**
  - Switching CUDA → HIP/SYCL requires minimal code changes
  - Recompile with different backend flag

### Design Philosophy

**GPU-Aware MPI:**
- Direct device-to-device communication when supported
- Avoids unnecessary host staging

**Data Residency:**
- Minimize host-device transfers
- Keep data on GPU across multiple timesteps
- Only transfer when necessary (I/O, diagnostics)

**Algorithm Considerations:**
- Some algorithms redesigned for GPU efficiency
- Focus on exposing maximum parallelism
- Accept minor algorithmic changes if physics preserved

---

## 4. SPPARKS Architecture Analysis

### Current Parallelization Approach

SPPARKS uses MPI-based domain decomposition with sector/color-based parallelism for kinetic Monte Carlo (KMC) simulations.

**Core Data Structures:**

```cpp
// From app.h and app_lattice.h
class App {
    tagint nglobal;           // Global # of sites
    int nlocal;               // # of sites I own
    int nghost;               // # of ghost sites

    tagint *id;               // Site IDs
    double **xyz;             // Site coordinates [nlocal+nghost][3]
    int **iarray;             // Integer arrays per site (spin, etc.)
    double **darray;          // Double arrays per site (quaternions, etc.)
};

class AppLattice : public App {
    int *numneigh;            // # neighbors per site
    int **neighbor;           // Neighbor indices [nlocal][maxneigh]
    double *propensity;       // KMC propensities per site

    // Sector-based parallelism
    int nsector;              // 1, 2, 4, or 8 sectors
    Set *set;                 // Sector sets for parallel KMC
};
```

**Parallel Patterns:**

1. **MPI Domain Decomposition** (`comm_lattice.cpp`):
   - Spatial partitioning across processors
   - Ghost site communication for neighbor access
   - Structured halo exchange via `CommLattice::perform_swap()`

2. **Sector-Based Parallelism**:
   - Domain divided into 1, 2, 4, or 8 sectors
   - Required for parallel KMC/rKMC execution
   - Each sector runs independently to avoid conflicts
   - Defined in `app_lattice.cpp:init_app()`

3. **Color-Based Parallelism**:
   - Graph coloring of lattice sites
   - Sites of same color have no neighbor dependencies
   - Enables parallel rejection KMC sweeps
   - Implemented via `AppLattice::site_color()`

### Computational Kernels

**Research identified ~209 site iteration loops** across SPPARKS codebase - these are prime candidates for GPU parallelization.

**Primary Loop Patterns:**

```cpp
// Pattern 1: Site energy calculations (in site_energy methods)
for (int j = 0; j < numneigh[i]; j++) {
    int jsite = neighbor[i][j];
    // Neighbor interaction energy
}

// Pattern 2: Sweep-based rejection KMC
for (int i = 0; i < nlocal; i++) {
    site_event_rejection(i, random);
}

// Pattern 3: KMC propensity calculations
for (int i = 0; i < nset[iset].nlocal; i++) {
    int site = nset[iset].site[i];
    propensity[i] = site_propensity(site);
}

// Pattern 4: Event execution
for (int m = 0; m < nevents; m++) {
    site_event(site, random);
}
```

**Key Characteristics:**

- **Site-Centric Computation:** Energy and propensity calculated independently per site
- **Neighbor Iterations:** Inner loops over neighbor lists (irregular access patterns)
- **Tree-Based KMC Solver:** Binary tree for event selection (`solve_tree.cpp`) - inherently sequential
- **Random Number Usage:** Extensive use of `RandomPark` class throughout
- **Memory Management:** Custom templated `Memory` class for array allocation
- **No Current GPU Support:** All computation on CPU

### Apps Most Suitable for Kokkos

**Analysis of SPPARKS applications:**

1. **AppPotts** (`app_potts.cpp`):
   - Simplest app, good starting point
   - Spin flip dynamics
   - Site energy = count of unlike neighbors
   - Ideal for baseline benchmarking

2. **AppPottsNeigh** (`app_potts_neighonly.cpp`):
   - Neighbor-based energy calculations
   - More complex than base Potts
   - Representative of many apps

3. **AppPottsQuaternion** (`potts_quaternion/app_potts_quaternion.cpp`):
   - Complex quaternion mathematics per site
   - High computational intensity (good for GPU)
   - Uses 4D arrays (quaternions) per site
   - Grain boundary misorientation calculations

4. **Additive Manufacturing Apps:**
   - `AppPottsAMBezier` - Bézier curve temperature paths
   - `AppPottsQuaternion` with AM features
   - Large-scale industrial relevance
   - Temperature field management
   - Multiple computational kernels

### Memory Management

**Current System:**

```cpp
// memory.h - templated allocation
template <typename TYPE>
TYPE **create(TYPE **&array, int n1, int n2);

template <typename TYPE>
void destroy(TYPE **&array);
```

**Kokkos Requirements:**

- Replace raw pointers with `Kokkos::View` or `Kokkos::DualView`
- Automatic memory space management
- Layout optimization (LayoutLeft vs LayoutRight)
- Reference counting for memory safety

---

## 5. Implementation Strategy

### Phase 1: Infrastructure (Months 1-3)

#### A. Build System Integration

**CMake Package System:**

```cmake
# In CMakeLists.txt
option(SPPARKS_PACKAGE_KOKKOS "Enable KOKKOS package" OFF)

if(SPPARKS_PACKAGE_KOKKOS)
    # Option 1: Find external Kokkos installation
    find_package(Kokkos 4.0 QUIET)

    # Option 2: Use bundled Kokkos
    if(NOT Kokkos_FOUND)
        add_subdirectory(lib/kokkos)
    endif()

    target_link_libraries(spparks_lib PUBLIC Kokkos::kokkos)
    target_compile_definitions(spparks_lib PUBLIC SPK_KOKKOS)
endif()

# Backend selection
if(SPPARKS_KOKKOS_BACKEND STREQUAL "CUDA")
    set(Kokkos_ENABLE_CUDA ON)
elseif(SPPARKS_KOKKOS_BACKEND STREQUAL "HIP")
    set(Kokkos_ENABLE_HIP ON)
elseif(SPPARKS_KOKKOS_BACKEND STREQUAL "SYCL")
    set(Kokkos_ENABLE_SYCL ON)
elseif(SPPARKS_KOKKOS_BACKEND STREQUAL "OPENMP")
    set(Kokkos_ENABLE_OPENMP ON)
endif()
```

**Build Script Update (`build_cmake.sh`):**

```bash
# Add new options
--kokkos              Enable Kokkos package
--kokkos-backend      CUDA, HIP, SYCL, OPENMP (default: auto-detect)
--kokkos-arch         GPU architecture (Volta70, Ampere80, MI250, etc.)

# Examples:
./build_cmake.sh --kokkos --kokkos-backend cuda --kokkos-arch Ampere80
./build_cmake.sh --kokkos --kokkos-backend hip --kokkos-arch MI250
./build_cmake.sh --kokkos --kokkos-backend openmp
```

**Directory Structure:**

```
spparks/
├── lib/
│   └── kokkos/              # Git submodule or bundled Kokkos
├── src/
│   ├── KOKKOS/              # Kokkos package (like STITCH/)
│   │   ├── memory_kokkos.h
│   │   ├── memory_kokkos.cpp
│   │   ├── app_kokkos.h
│   │   ├── app_kokkos.cpp
│   │   ├── app_lattice_kokkos.h
│   │   ├── app_lattice_kokkos.cpp
│   │   └── ...
```

#### B. Memory Management Layer

**Create `src/KOKKOS/memory_kokkos.h`:**

```cpp
#ifndef SPK_MEMORY_KOKKOS_H
#define SPK_MEMORY_KOKKOS_H

#include "memory.h"
#include <Kokkos_Core.hpp>

namespace SPPARKS_NS {

// Type aliases for common Kokkos types
template<typename T>
using View1D = Kokkos::View<T*>;

template<typename T>
using View2D = Kokkos::View<T**>;

template<typename T>
using DualView1D = Kokkos::DualView<T*>;

template<typename T>
using DualView2D = Kokkos::DualView<T**>;

class MemoryKokkos : public Memory {
public:
    MemoryKokkos(class SPPARKS *);
    ~MemoryKokkos();

    // Kokkos-aware allocation methods
    template<typename T>
    void create_kokkos(DualView1D<T> &view, int n, const char *name);

    template<typename T>
    void create_kokkos(DualView2D<T> &view, int n1, int n2, const char *name);

    template<typename T>
    void destroy_kokkos(DualView1D<T> &view);

    template<typename T>
    void destroy_kokkos(DualView2D<T> &view);

    // Sync helpers
    template<typename ViewType>
    void sync_device(ViewType &view);

    template<typename ViewType>
    void sync_host(ViewType &view);

    template<typename ViewType>
    void modify_device(ViewType &view);

    template<typename ViewType>
    void modify_host(ViewType &view);
};

}

#endif
```

#### C. App Base Class Extensions

**Create `src/KOKKOS/app_kokkos.h`:**

```cpp
#ifndef SPK_APP_KOKKOS_H
#define SPK_APP_KOKKOS_H

#include "app.h"
#include <Kokkos_Core.hpp>
#include <Kokkos_DualView.hpp>
#include <Kokkos_Random.hpp>

namespace SPPARKS_NS {

class AppKokkos : public App {
public:
    AppKokkos(class SPPARKS *, int, char **);
    virtual ~AppKokkos();

    // Kokkos versions of site data
    Kokkos::DualView<tagint*> k_id;           // Site IDs
    Kokkos::DualView<double*[3]> k_xyz;       // Coordinates
    Kokkos::DualView<int**> k_iarray;         // Integer site arrays
    Kokkos::DualView<double**> k_darray;      // Double site arrays

    // Random number generation
    using pool_type = Kokkos::Random_XorShift64_Pool<>;
    pool_type *rand_pool;

    // Memory sync control
    void sync_to_device();
    void sync_to_host();
    void modify_device();
    void modify_host();

    // Override base class methods
    virtual void grow_app();
    virtual void set_site(int, tagint, double, double, double);

    // Execution space query
    bool is_device_execution();
};

}

#endif
```

**Create `src/KOKKOS/app_lattice_kokkos.h`:**

```cpp
#ifndef SPK_APP_LATTICE_KOKKOS_H
#define SPK_APP_LATTICE_KOKKOS_H

#include "app_lattice.h"
#include "app_kokkos.h"

namespace SPPARKS_NS {

class AppLatticeKokkos : public AppLattice {
public:
    AppLatticeKokkos(class SPPARKS *, int, char **);
    virtual ~AppLatticeKokkos();

    // Kokkos neighbor data
    Kokkos::DualView<int*> k_numneigh;
    Kokkos::DualView<int**> k_neighbor;
    Kokkos::DualView<double*> k_propensity;

    // Virtual functions to override with Kokkos implementations
    virtual double site_energy_kokkos(int) = 0;
    virtual void site_event_rejection_kokkos(int, class RandomPark *) = 0;
    virtual double site_propensity_kokkos(int) = 0;

    // Bulk operations on device
    virtual void compute_all_energies_kokkos();
    virtual void compute_all_propensities_kokkos();
    virtual void sweep_rejection_kokkos();

    // Sync helpers
    void sync_neighbor_data_to_device();
    void sync_neighbor_data_to_host();
};

}

#endif
```

#### D. Initialization and Finalization

**Modify `spparks.cpp`:**

```cpp
#include "spparks.h"

#ifdef SPK_KOKKOS
#include <Kokkos_Core.hpp>
#endif

SPPARKS::SPPARKS(int narg, char **arg, MPI_Comm communicator) {
    // ... existing initialization ...

    #ifdef SPK_KOKKOS
    // Initialize Kokkos (only if not already initialized)
    if (!Kokkos::is_initialized()) {
        Kokkos::initialize(narg, arg);
    }

    // Print Kokkos configuration
    if (me == 0) {
        Kokkos::print_configuration(screen);
    }
    #endif
}

SPPARKS::~SPPARKS() {
    // ... existing cleanup ...

    #ifdef SPK_KOKKOS
    // Finalize Kokkos (only if SPPARKS initialized it)
    if (Kokkos::is_initialized()) {
        Kokkos::finalize();
    }
    #endif
}
```

### Phase 2: Core App Implementations (Months 4-6)

#### Priority: AppPotts

**Create `src/KOKKOS/app_potts_kokkos.h`:**

```cpp
#ifndef SPK_APP_POTTS_KOKKOS_H
#define SPK_APP_POTTS_KOKKOS_H

#include "app_potts.h"
#include "app_lattice_kokkos.h"

namespace SPPARKS_NS {

class AppPottsKokkos : public AppPotts {
public:
    AppPottsKokkos(class SPPARKS *, int, char **);
    virtual ~AppPottsKokkos();

    // Kokkos-specific data
    Kokkos::DualView<int*> k_spin;

    // Override base class methods with Kokkos versions
    virtual void grow_app();
    virtual void init_app();

    // Kokkos computation kernels
    double site_energy_kokkos(int);
    void site_event_rejection_kokkos(int, class RandomPark *);
    double site_propensity_kokkos(int);

    // Bulk parallel operations
    void compute_all_energies_kokkos();
    void sweep_rejection_kokkos();

private:
    // Device-side helper functors
    struct ComputeEnergyFunctor;
    struct RejectionSweepFunctor;
};

}

#endif
```

**Create `src/KOKKOS/app_potts_kokkos.cpp`:**

```cpp
#include "app_potts_kokkos.h"
#include "random_park.h"

using namespace SPPARKS_NS;

AppPottsKokkos::AppPottsKokkos(SPPARKS *spk, int narg, char **arg) :
    AppPotts(spk, narg, arg) {

    // Allocate Kokkos structures
    // This will be done in grow_app()
}

void AppPottsKokkos::grow_app() {
    // Call parent
    AppPotts::grow_app();

    // Allocate/resize Kokkos views
    k_spin = Kokkos::DualView<int*>("spin", nlocal + nghost);

    // Copy existing data to Kokkos structures
    auto h_spin = k_spin.h_view;
    for (int i = 0; i < nlocal + nghost; i++) {
        h_spin(i) = spin[i];
    }
    k_spin.modify_host();

    // Make spin pointer alias to host view for compatibility
    spin = h_spin.data();
}

// Energy calculation functor
struct AppPottsKokkos::ComputeEnergyFunctor {
    Kokkos::View<int*> spin;
    Kokkos::View<int*> numneigh;
    Kokkos::View<int**> neighbor;
    Kokkos::View<double*> energy;

    KOKKOS_INLINE_FUNCTION
    void operator()(const int i) const {
        int isite = spin(i);
        int eng = 0;

        for (int j = 0; j < numneigh(i); j++) {
            int jsite = neighbor(i, j);
            if (isite != spin(jsite)) eng++;
        }

        energy(i) = static_cast<double>(eng);
    }
};

void AppPottsKokkos::compute_all_energies_kokkos() {
    // Sync data to device
    k_spin.sync_device();
    k_numneigh.sync_device();
    k_neighbor.sync_device();

    // Allocate energy array
    Kokkos::View<double*> d_energy("energy", nlocal);

    // Launch kernel
    ComputeEnergyFunctor functor{
        k_spin.d_view,
        k_numneigh.d_view,
        k_neighbor.d_view,
        d_energy
    };

    Kokkos::parallel_for("AppPotts::ComputeEnergies", nlocal, functor);
    Kokkos::fence();

    // Copy results back to host if needed
    // ...
}

// Rejection KMC sweep functor
struct AppPottsKokkos::RejectionSweepFunctor {
    Kokkos::View<int*> spin;
    Kokkos::View<int*> numneigh;
    Kokkos::View<int**> neighbor;
    typename Kokkos::Random_XorShift64_Pool<>::generator_type rand_gen;
    int nspins;
    double temperature;

    KOKKOS_INLINE_FUNCTION
    void operator()(const int i) const {
        // Get current spin
        int isite = spin(i);

        // Calculate current energy
        int eng_before = 0;
        for (int j = 0; j < numneigh(i); j++) {
            if (isite != spin(neighbor(i, j))) eng_before++;
        }

        // Propose random new spin
        int newsite = rand_gen.urand(nspins) + 1;

        // Calculate new energy
        int eng_after = 0;
        for (int j = 0; j < numneigh(i); j++) {
            if (newsite != spin(neighbor(i, j))) eng_after++;
        }

        // Metropolis acceptance
        double delta = eng_after - eng_before;
        if (delta <= 0.0) {
            spin(i) = newsite;
        } else {
            double prob = exp(-delta / temperature);
            if (rand_gen.drand() < prob) {
                spin(i) = newsite;
            }
        }
    }
};

void AppPottsKokkos::sweep_rejection_kokkos() {
    // Sync to device
    k_spin.sync_device();
    k_numneigh.sync_device();
    k_neighbor.sync_device();
    k_spin.modify_device();

    // Get RNG pool
    auto rand_gen = rand_pool->get_state();

    // Launch kernel
    RejectionSweepFunctor functor{
        k_spin.d_view,
        k_numneigh.d_view,
        k_neighbor.d_view,
        rand_gen,
        nspins,
        temperature
    };

    Kokkos::parallel_for("AppPotts::RejectionSweep", nlocal, functor);
    Kokkos::fence();

    rand_pool->free_state(rand_gen);

    // Data modified on device
    k_spin.modify_device();
}
```

**Registration:**

Modify `src/KOKKOS/app_potts_kokkos.cpp`:

```cpp
#ifdef SPK_KOKKOS
AppStyle(potts/kk, AppPottsKokkos)
#endif
```

### Phase 3: Solver Integration (Months 7-9)

#### Challenge: Sequential KMC Tree

The tree-based event selection in `solve_tree.cpp` is inherently sequential. We have several options:

**Option 1: Hybrid CPU/GPU Approach**

```cpp
class SolveTreeKokkos : public SolveTree {
public:
    void update_kokkos(int n, double *propensity) {
        // Compute propensities on GPU
        app->compute_all_propensities_kokkos();

        // Copy to host
        Kokkos::deep_copy(h_propensity, d_propensity);

        // Build tree on CPU (existing code)
        SolveTree::update(n, h_propensity.data());
    }

    int event_kokkos(class RandomPark *random) {
        // Event selection on CPU (existing code)
        return SolveTree::event(random);
    }
};
```

**Option 2: GPU-Native Event Selection**

```cpp
class SolveTreeKokkosGPU : public SolveTree {
    Kokkos::View<double*> d_tree;
    Kokkos::View<double*> d_propensity;

    void update_gpu(int n) {
        // Parallel tree build on GPU
        // Level-by-level bottom-up construction

        int nlevels = static_cast<int>(ceil(log2(n)));

        for (int level = nlevels - 1; level >= 0; level--) {
            int nodes_at_level = 1 << level;

            Kokkos::parallel_for(nodes_at_level,
                KOKKOS_LAMBDA(int node) {
                    int left_child = 2 * node + 1;
                    int right_child = 2 * node + 2;
                    d_tree(node) = d_tree(left_child) + d_tree(right_child);
                });
        }
    }

    KOKKOS_INLINE_FUNCTION
    int select_event_device(double random_value) const {
        // Binary search down tree on device
        int node = 0;
        double sum = 0.0;

        while (node < nleaf - 1) {
            int left = 2 * node + 1;
            int right = 2 * node + 2;

            if (random_value < sum + d_tree(left)) {
                node = left;
            } else {
                sum += d_tree(left);
                node = right;
            }
        }

        return node - (nleaf - 1);  // Convert to site index
    }
};
```

**Option 3: Sector-Based Parallel KMC**

Leverage SPPARKS existing sector parallelism:

```cpp
// Use Kokkos::TeamPolicy
// Each team handles one sector
// Within team, parallel propensity calculation and event execution

Kokkos::TeamPolicy<> policy(nsector, Kokkos::AUTO);

Kokkos::parallel_for(policy,
    KOKKOS_LAMBDA(const member_type& team) {
        int isector = team.league_rank();

        // Parallel propensity calculation within sector
        Kokkos::parallel_for(
            Kokkos::TeamThreadRange(team, sector_size),
            [&](int i) {
                // Calculate propensity for site i in sector
            });
        team.team_barrier();

        // Single thread selects event (sequential)
        Kokkos::single(Kokkos::PerTeam(team), [&]() {
            // Event selection for this sector
        });

        // Execute event (may need synchronization)
    });
```

### Phase 4: Communication Optimization (Months 10-12)

#### GPU-Aware MPI

**Create `src/KOKKOS/comm_lattice_kokkos.cpp`:**

```cpp
class CommLatticeKokkos : public CommLattice {
    Kokkos::View<char*> d_send_buffer;
    Kokkos::View<char*> d_recv_buffer;

    void perform_swap_kokkos(Swap *swap) {
        int nsend = swap->nsend;
        int nrecv = swap->nrecv;

        // Pack on device
        pack_swap_kokkos(swap, d_send_buffer);

        #ifdef SPPARKS_GPU_AWARE_MPI
        // Direct device-to-device transfer
        MPI_Sendrecv(
            d_send_buffer.data(), nsend, MPI_CHAR, swap->sendproc, 0,
            d_recv_buffer.data(), nrecv, MPI_CHAR, swap->recvproc, 0,
            world, MPI_STATUS_IGNORE);
        #else
        // Stage through host
        Kokkos::View<char*>::HostMirror h_send =
            Kokkos::create_mirror_view(d_send_buffer);
        Kokkos::View<char*>::HostMirror h_recv =
            Kokkos::create_mirror_view(d_recv_buffer);

        Kokkos::deep_copy(h_send, d_send_buffer);

        MPI_Sendrecv(
            h_send.data(), nsend, MPI_CHAR, swap->sendproc, 0,
            h_recv.data(), nrecv, MPI_CHAR, swap->recvproc, 0,
            world, MPI_STATUS_IGNORE);

        Kokkos::deep_copy(d_recv_buffer, h_recv);
        #endif

        // Unpack on device
        unpack_swap_kokkos(swap, d_recv_buffer);
    }

    void pack_swap_kokkos(Swap *swap, Kokkos::View<char*> buffer) {
        // Parallel pack on device
        auto sendlist = swap->sendlist;

        Kokkos::parallel_for(swap->npack,
            KOKKOS_LAMBDA(int i) {
                int site = sendlist(i);
                // Pack site data into buffer
                // ...
            });
    }
};
```

#### KokkosComm Integration

If available, use KokkosComm for simplified device-aware communication:

```cpp
#include <KokkosComm.hpp>

void CommLatticeKokkos::perform_swap_kokkos(Swap *swap) {
    // KokkosComm handles device awareness automatically
    KokkosComm::send(d_send_buffer, swap->sendproc, 0, world);
    KokkosComm::recv(d_recv_buffer, swap->recvproc, 0, world);
}
```

---

### Phase 5: Additive Manufacturing Applications (Months 13-16)

**Status**: ⏳ Planning (November 2025)

**Objective**: Enable Kokkos acceleration for additive manufacturing simulations with HDF5 unstructured temperature sources, implementing dual-target parallelization for maximum performance.

**Target Application**: `app_additive_ext_temp_texture` with `HDF5UnstructuredTemperatureSource`

#### Dual-Target Parallelization Strategy

This phase implements **two independent parallelization targets**, each providing significant speedup:

1. **Temperature Source Interpolation** (HIGH PRIORITY)
   - Target: `get_temperature_at_site()` tetrahedral interpolation
   - Complexity: 40-60 FLOPs + ~200 bytes memory read per site
   - Parallelism: Embarrassingly parallel (zero data dependencies)
   - Expected speedup: **10-40× on GPU**
   - Implementation: Hybrid host-device (HDF5 I/O on host, interpolation on device)

2. **Monte Carlo Site Events** (Inherits from AppPottsQuaternionKokkos)
   - Target: `site_event_rejection()` grain growth
   - Expected speedup: **20× on CPU (OpenMP)**
   - Architecture: Inherit double-buffer framework from parent class

**Combined Expected Performance**: **15-30× overall application speedup**

#### Key Technical Insights

**Temperature Interpolation is Ideal for GPUs:**
- The existing `site_element_cache` design eliminates expensive element search from the critical path
- Cache contains pre-computed tetrahedral element indices and barycentric weights per site
- Element search is O(C × E × 60 FLOPs) but done **once per layer** (infrequent)
- Temperature interpolation is O(log M + 1) but called **every timestep for all sites** (frequent)
- Result: Perfect separation of irregular (host) and regular (device) work

**Memory Transfer Cost is Negligible:**
- Per-layer data: ~100 MB (mesh + cache)
- Transfer time: 9 ms on PCIe Gen3
- Layer duration: 50-100 ms typical
- Temperature queries per layer: 50-500
- **ROI: Transfer cost recovered after first query**

#### Sub-Phase 5.1: Temperature Source Infrastructure (Weeks 1-2)

**Goal**: Create Kokkos-enabled HDF5 temperature source with device data structures

**Files to create:**
```
src/KOKKOS/temperature_source_hdf5_unstructured_kokkos.h
src/KOKKOS/temperature_source_hdf5_unstructured_kokkos.cpp
```

**Implementation:**
```cpp
class HDF5UnstructuredTemperatureSourceKokkos :
  public HDF5UnstructuredTemperatureSource {

public:
  // Override for device-accelerated queries
  virtual double get_temperature_at_site(int site_index, double time) override;

private:
  // Device data structures
  Kokkos::View<double**, Kokkos::LayoutRight> times_device;
  Kokkos::View<double**, Kokkos::LayoutRight> temperatures_device;
  Kokkos::View<ElementCache*> cache_device;
  Kokkos::View<unsigned*> dataCounts_device;

  // Host mirrors (for HDF5 I/O)
  Kokkos::View<double**, Kokkos::LayoutRight>::HostMirror times_host;
  Kokkos::View<double**, Kokkos::LayoutRight>::HostMirror temperatures_host;
  // ... etc

  // Hybrid approach: HDF5 on host, transfer to device
  void load_layer_and_sync_device(unsigned layerIdx);
  void build_cache_and_transfer();
};
```

**Key tasks:**
1. Convert `Array2D<double>` to `Kokkos::View<double**, LayoutRight>`
2. Create host and device views for:
   - `times` (nNodes × maxTimePoints)
   - `temperatures` (nNodes × maxTimePoints)
   - `site_element_cache` (nlocal ElementCache structs)
   - `dataCounts` (nNodes valid entry counts)
3. Keep all HDF5 I/O on host (library limitation)
4. Implement `Kokkos::deep_copy()` on layer change

**Memory layout optimization:**
- Use `LayoutRight` (row-major) for coalesced memory access
- `times_device(nodeIdx, 0...maxData)` → sequential memory reads
- GPU memory requirement: ~100-150 MB per layer (easily fits modern GPUs)

**Validation:**
- Compile successfully with Kokkos enabled
- Allocate views without errors
- Verify host-device transfers complete

**Expected outcome**: Infrastructure ready for parallel interpolation

---

#### Sub-Phase 5.2: Parallel Temperature Interpolation (Weeks 2-4)

**Goal**: GPU-accelerate tetrahedral temperature interpolation (**HIGHEST IMPACT**)

**Implementation Strategy: Hybrid Host-Device**

**Host responsibilities (HDF5 constraints):**
```cpp
void load_layer_and_sync_device(unsigned layerIdx) {
  // 1. HDF5 file I/O (host-only, cannot move to device)
  grp->getDataSet("times").select(hyperslab).read(times_host_data);
  grp->getDataSet("temperatures").select(hyperslab).read(temps_host_data);
  // ... load mesh geometry, chunks, etc.

  // 2. Build element cache (serial element search - complex, irregular)
  for (int i = 0; i < nlocal; i++) {
    cache_host(i) = find_element_point_is_in(...);  // O(C × E × 60 FLOPs)
  }

  // 3. Transfer to device (one-time per layer, ~9 ms)
  Kokkos::deep_copy(times_device, times_host);
  Kokkos::deep_copy(temperatures_device, temperatures_host);
  Kokkos::deep_copy(cache_device, cache_host);
  Kokkos::deep_copy(dataCounts_device, dataCounts_host);
}
```

**Device kernel (called every timestep):**
```cpp
void update_temperatures_device(double simulation_time,
                                 Kokkos::View<double*> T_device) {
  const int nlocal_local = nlocal;
  const auto cache_d = cache_device;
  const auto times_d = times_device;
  const auto temps_d = temperatures_device;
  const auto counts_d = dataCounts_device;
  const double default_temp_local = default_temp;

  Kokkos::parallel_for("UpdateTemperatures", nlocal_local,
    KOKKOS_LAMBDA(const int i) {
      // 1. CACHE LOOKUP (O(1) - array access, 64 bytes)
      const ElementCache& cache = cache_d(i);

      if (!cache.valid) {
        T_device(i) = default_temp_local;
        return;
      }

      // 2. TEMPORAL INTERPOLATION FOR 4 TET NODES (O(4 × log M))
      double nodalVals[4];
      for (int n = 0; n < 4; n++) {
        unsigned nodeIdx = cache.nodeIndices[n];
        int count = counts_d(nodeIdx);

        // Device-compatible binary search
        int idx = binary_search_device(times_d, nodeIdx, simulation_time, count);

        // Linear interpolation in time
        double t0 = times_d(nodeIdx, idx-1);
        double t1 = times_d(nodeIdx, idx);
        double T0 = temps_d(nodeIdx, idx-1);
        double T1 = temps_d(nodeIdx, idx);

        nodalVals[n] = T0 + (T1 - T0) / (t1 - t0) * (simulation_time - t0);
      }

      // 3. SPATIAL INTERPOLATION (O(1) - barycentric, 9 FLOPs)
      T_device(i) = nodalVals[0] +
                    cache.weights[0] * (nodalVals[1] - nodalVals[0]) +
                    cache.weights[1] * (nodalVals[2] - nodalVals[0]) +
                    cache.weights[2] * (nodalVals[3] - nodalVals[0]);
    });
}
```

**Device-compatible binary search:**
```cpp
KOKKOS_INLINE_FUNCTION
int binary_search_device(const Kokkos::View<double**>& times,
                         int nodeIdx, double target, int count) {
  int low = 0, high = count - 1;
  while (low < high) {
    int mid = (low + high) / 2;
    if (times(nodeIdx, mid) < target)
      low = mid + 1;
    else
      high = mid;
  }
  return (low == 0) ? 1 : low;  // Ensure valid interpolation interval
}
```

**Performance analysis:**

**Per-site computation:**
- Cache lookup: 1 read (64 bytes)
- 4 binary searches: 4-7 iterations each (M=10-100 time points typical)
- 4 linear interpolations: 8 FLOPs each = 32 FLOPs
- 1 tetrahedral interpolation: 9 FLOPs
- **Total: ~40-60 FLOPs + ~200 bytes memory read**

**Scaling (1M sites):**
- CPU (10 cores, memory-bound): 50-200 ms (cache miss limited)
- GPU (V100, 900 GB/s bandwidth): 2-5 ms (50% bandwidth utilization)
- **Speedup: 10-40×**

**Validation:**
- Compare host vs device temperatures: `fabs(T_host - T_device) < 1e-10`
- Verify cache validity flags match
- Test layer transitions (ensure proper HDF5 reload + device sync)
- Test edge cases: sites outside mesh, invalid elements

**Expected outcome**: 10-40× speedup for temperature field updates

---

#### Sub-Phase 5.3: Additive App Infrastructure (Weeks 4-5)

**Goal**: Create AppAdditiveExtTempTextureKokkos inheriting from AppPottsQuaternionKokkos

**Files to create:**
```
src/KOKKOS/app_additive_ext_temp_texture_kokkos.h
src/KOKKOS/app_additive_ext_temp_texture_kokkos.cpp
```

**Class hierarchy** (mirrors CPU inheritance):
```
CPU HIERARCHY:
AppPottsQuaternion → AppAdditiveExtTempTexture

KOKKOS HIERARCHY:
AppPottsQuaternionKokkos → AppAdditiveExtTempTextureKokkos
```

**Inherited from AppPottsQuaternionKokkos** (already working):
- Double buffers: `buffer_spin`, `buffer_q0/qx/qy/qz`
- Device-compatible quaternion operations (`compute_disorientation`)
- Sector sweep patterns + ghost exchange sync
- RNG pool (`Kokkos::Random_XorShift64_Pool`)
- Profiling infrastructure (`enable_profiling` input command)
- Selective sync optimization (33% reduction demonstrated)

**New double buffers for additive-specific data:**
```cpp
class AppAdditiveExtTempTextureKokkos : public AppAdditiveExtTempTexture {
private:
  // Additive-specific Kokkos double buffers
  KokkosDoubleBuffer<double> buffer_T;           // Temperature field
  KokkosDoubleBuffer<double> buffer_G;           // Gradient magnitude
  KokkosDoubleBuffer<double> buffer_V;           // Solidification velocity
  KokkosDoubleBuffer<int> buffer_active_flag;    // Phase state (0=inactive, 1=active, 2=molten, 3=solid, 5=void)
  KokkosDoubleBuffer<double> buffer_solid_d;     // Solidification smoothing counter
  KokkosDoubleBuffer<double> buffer_mobility;    // Diagnostic mobility output

  // Kokkos temperature source (owns device data)
  HDF5UnstructuredTemperatureSourceKokkos* kokkos_temp_source;
};
```

**Memory requirements:**
- Per site: 1 int (spin) + 4 doubles (quat) + 6 doubles (additive) = 84 bytes
- With double buffering: 168 bytes/site
- 1M sites: ~160 MB (app) + ~110 MB (temp source) = **270 MB total**
- Modern GPU (16+ GB): Easily handles 10M+ sites

**Key method overrides:**

**1. Data allocation:**
```cpp
void AppAdditiveExtTempTextureKokkos::allocate_data() {
  // Call parent to allocate spin + quaternion buffers
  AppPottsQuaternionKokkos::allocate_data();

  // Allocate additive-specific buffers
  buffer_T.allocate(nlocal + nghost);
  buffer_G.allocate(nlocal + nghost);
  buffer_V.allocate(nlocal + nghost);
  buffer_active_flag.allocate(nlocal + nghost);
  buffer_solid_d.allocate(nlocal + nghost);
  buffer_mobility.allocate(nlocal + nghost);
}
```

**2. Temperature source integration:**
```cpp
void AppAdditiveExtTempTextureKokkos::setup_temperature_source(args) {
  // Create Kokkos-enabled temperature source
  kokkos_temp_source = new HDF5UnstructuredTemperatureSourceKokkos(spk);
  kokkos_temp_source->setup_temperature_source(args);
  temperature_source = kokkos_temp_source;  // Base pointer
}
```

**3. Buffer swapping:**
```cpp
void AppAdditiveExtTempTextureKokkos::swap_buffers() {
  // Call parent to swap spin + quaternion buffers
  AppPottsQuaternionKokkos::swap_buffers();

  // Swap additive buffers
  buffer_T.swap();
  buffer_G.swap();
  buffer_V.swap();
  buffer_active_flag.swap();
  buffer_solid_d.swap();
  buffer_mobility.swap();
}
```

**Validation:**
- Compile with Kokkos + HDF5 + HighFive
- Allocate/deallocate buffers without memory leaks
- Verify temperature source integration (device views created)

**Expected outcome**: Infrastructure ready for device MC kernels

---

#### Sub-Phase 5.4: Device MC Kernels and Synchronization (Weeks 5-7)

**Goal**: GPU-accelerate Monte Carlo site events with additive-specific physics

**Host-side app_update() pattern:**
```cpp
void AppAdditiveExtTempTextureKokkos::app_update(double dt) {
  // === HOST-SIDE OPERATIONS ===

  // 1. Update temperatures using Kokkos temp source (DEVICE KERNEL)
  update_temperature_from_source(time);
  // Internally calls kokkos_temp_source->update_temperatures_device()
  // Result: T[] array updated on device

  // 2. Phase transition logic (HOST-ONLY - complex branching)
  for (int i = 0; i < nlocal; i++) {
    if (T[i] >= tl && active_flag[i] != 2) {
      // Melting: initialize molten site
      active_flag[i] = 2;
      // ... other initialization
    } else if (T[i] <= tl && active_flag[i] == 2) {
      // Nucleation: mushy_phase() decision
      mushy_phase(i, ranapp);  // Per-spin nucleation data (host-only)
      if (nucleated) {
        active_flag[i] = 3;  // Solidified
        solid_d[i] = -1;     // Start smoothing countdown
        G[i] = compute_gradient();  // Store gradient at solidification
        V[i] = compute_velocity();
      }
    }
  }

  // 3. Sync updated arrays to device
  sync_all_to_device();
  // Transfers: T[], active_flag[], G[], V[], solid_d[]
  // Only read_buffer (selective sync optimization)

  // === DEVICE-SIDE OPERATIONS ===

  // 4. Monte Carlo sweeps (DEVICE KERNELS)
  for (int isector = 0; isector < nsector; isector++) {
    sweep_rejection_kokkos(isector);  // Parallel on device
    swap_buffers();  // Make writes visible
  }

  // === BACK TO HOST FOR MPI ===

  // 5. MPI ghost exchange
  sync_data_to_host();  // Needed for MPI communication
  comm->all();
  sync_data_to_device();  // Resync ghost data
}
```

**Why phase transitions stay on host:**
- Nucleation uses per-spin data (`nucleation_temps[spin[i]]`) - not per-site
- Complex branching logic (epitaxial vs competitive growth)
- Random number generation for nucleation probabilities
- Infrequent operation (once per site during solidification)
- **Cost: Acceptable overhead vs complexity of device implementation**

**Device MC kernel (override from parent):**
```cpp
struct RejectionSweepFunctor {
  // Inherit base data from AppPottsQuaternionKokkos
  // Add additive-specific views

  Kokkos::View<double*> T_read;
  Kokkos::View<double*> G_read;
  Kokkos::View<double*> V_read;
  Kokkos::View<int*> active_flag_read;
  Kokkos::View<double*> solid_d_read, solid_d_write;

  // Physical constants
  double c1, c2, c3;  // Melt misorientation weights
  double q, tl, ts, mo;  // Arrhenius mobility parameters

  KOKKOS_INLINE_FUNCTION
  void operator()(const int& i) const {
    // Skip non-solidified sites
    if (active_flag_read(i) != 3) return;
    if (active_flag_read(i) == 5) return;  // Voids

    // Read temperature (from Kokkos temp source)
    double T_i = T_read(i);

    // Arrhenius mobility
    double mobility = compute_mobility_device(T_i, q, tl, ts, mo);

    // BRANCHING: Solidification smoothing vs solid-state growth
    if (solid_d_read(i) < 0 && solid_d_read(i) > -nrefine - 1) {
      // === SOLIDIFICATION SMOOTHING (weighted neighbor selection) ===

      // Thread-local scratch arrays
      double unique_dot[MAX_NEIGH];
      int unique[MAX_NEIGH];
      int unique_neigh[MAX_NEIGH];
      int nevent = 0;

      // Build weighted CDF
      double dotValue = 0.0;
      for (int j = 0; j < numneigh[i]; j++) {
        int nj = neighbor[i][j];
        if (active_flag_read(nj) == 3 || active_flag_read(nj) == 1) {
          // Compute melt misorientation weight
          double weight = compute_melt_misorientation_device(i, nj,
            G_read, V_read, c1, c2, c3);

          dotValue += weight;
          unique_dot[nevent] = dotValue;
          unique[nevent] = spin_read(nj);
          unique_neigh[nevent] = nj;
          nevent++;
        }
      }

      if (nevent == 0) return;

      // Sample from CDF using thread RNG
      auto rand_gen = rand_pool.get_state(i);
      double dran = unique_dot[nevent-1] * rand_gen.drand();

      // Find selected neighbor (binary search in CDF)
      int m = 0;
      while (m < nevent && unique_dot[m] <= dran) m++;

      int newsite = unique_neigh[m];
      int newspin = unique[m];

      // Update smoothing counter
      solid_d_write(i) = solid_d_read(i) + 1;

      // Standard Metropolis acceptance with mobility modulation
      // ... (same as parent class)

    } else {
      // === SOLID-STATE GROWTH (standard Potts) ===

      // Inherit from AppPottsQuaternionKokkos::operator()
      // Standard neighbor selection, Metropolis acceptance
      // ... (call parent's logic)
    }
  }
};
```

**Device-compatible melt misorientation:**
```cpp
KOKKOS_INLINE_FUNCTION
double compute_melt_misorientation_device(int i, int neigh_id,
  const Kokkos::View<double*>& G_read,
  const Kokkos::View<double*>& V_read,
  double c1, double c2, double c3) const {

  // Compute quaternion misorientation (reuse parent's device function)
  double angle = compute_disorientation_device(i, neigh_id);

  // Gradient direction term (c2 weight)
  double G_val = G_read(i);
  double V_val = V_read(i);
  // ... compute normal · direction term

  // Thermal gradient term (c3 weight)
  // ... compute f(G, V)

  return c1 * f_angle(angle) + c2 * f_normal(normal_dot) + c3 * f_thermal(G_val, V_val);
}
```

**Synchronization pattern (inherits from parent + extends):**
```cpp
void AppAdditiveExtTempTextureKokkos::sync_all_to_device() {
  // Call parent (syncs spin + quaternions)
  AppPottsQuaternionKokkos::sync_all_to_device();

  // Sync additive arrays (selective: only read_buffer, skip write_buffer)
  buffer_T.sync_to_device_selective(T, true, false);
  buffer_G.sync_to_device_selective(G, true, false);
  buffer_V.sync_to_device_selective(V, true, false);
  buffer_active_flag.sync_to_device_selective(active_flag, true, false);
  buffer_solid_d.sync_to_device_selective(solid_d, true, false);
}
```

**Validation:**
- Single MPI rank: CPU vs Kokkos grain structure comparison
- Verify thermal histories: T(t), G, V fields identical
- Check grain boundary statistics (disorientation distributions)
- Verify no race conditions (zero desynchronized spins)

**Expected outcome**: Device MC sweeps working, 20× speedup on OpenMP backend

---

#### Sub-Phase 5.5: MPI and Optimization (Weeks 7-8)

**Goal**: Multi-rank support and performance profiling

**Multi-MPI pattern** (same as parent class):
```cpp
// After device MC sweeps, before MPI:
sync_data_to_host();  // Get device data for MPI communication

// MPI ghost exchange (host buffers)
comm->all();

// After MPI, before next device sweep:
sync_data_to_device();  // Resync ghost data to device
```

**Selective sync optimization for temperature:**
```cpp
// Temperature doesn't change during MC sweep (only in app_update)
// Only sync read_buffer after device writes are swapped
buffer_T.sync_to_device_selective(T, true, false);  // read=true, write=false
// Result: 33% reduction in sync time (same as parent's optimization)
```

**Profiling infrastructure:**
```cpp
if (profiling_enabled) {
  Kokkos::Timer timer;

  // Time HDF5 load
  timer.reset();
  kokkos_temp_source->load_layer_and_sync_device(layer);
  double hdf5_time = timer.seconds();

  // Time temperature interpolation
  timer.reset();
  kokkos_temp_source->update_temperatures_device(time, buffer_T.get_read_view());
  double interp_time = timer.seconds();

  // Time MC sweeps
  timer.reset();
  sweep_rejection_kokkos(isector);
  double mc_time = timer.seconds();

  if (me == 0) {
    fprintf(screen, "HDF5: %.3f ms, Interp: %.3f ms, MC: %.3f ms\n",
            hdf5_time*1000, interp_time*1000, mc_time*1000);
  }
}
```

**Validation:**
- Test with 1, 2, 4, 6 MPI ranks
- Compare grain structures across ranks (continuity at boundaries)
- Verify no desynchronized sites at MPI boundaries
- Performance scaling: measure speedup vs CPU baseline

**Benchmarking:**
- Small (10³): Quick validation
- Medium (20³): Typical production
- Large (40³): Scaling test
- Document: CPU time, Kokkos time, speedup, memory usage

**Expected outcome**: Production-ready multi-MPI support, documented performance

---

#### Sub-Phase 5.6: Production Testing and Documentation (Weeks 8-10)

**Test cases** (from `/Users/Tron/spparks/examples/ReducedTempAM/QuatTest/`):

**1. Standard AM simulation:**
```bash
# Input: in.additive
# Features: HDF5 unstructured temp source, multiple layers
# Validation: Grain structure, IPF texture, misorientation
```

**2. Serpentine scan pattern:**
```bash
# Input: in.additive_serpentine
# Features: Complex thermal history, bidirectional scanning
# Validation: Epitaxial growth patterns, thermal gradients
```

**3. Multi-layer build:**
```bash
# Features: Layer-by-layer solidification, thermal cycling
# Validation: Columnar grain growth across layers
```

**4. Large-scale benchmark:**
```bash
# Size: 1M+ sites, 100+ layers
# Validation: Memory scaling, performance consistency
```

**Physics validation checklist:**
- ✅ Grain size distribution matches CPU version
- ✅ IPF (Inverse Pole Figure) texture identical
- ✅ Misorientation angle distribution < 0.1° difference
- ✅ Thermal history: T(x,y,z,t) arrays match within 1e-10
- ✅ Gradient/velocity fields: G, V match CPU version
- ✅ Nucleation events: Same grains nucleate at same times

**Performance validation checklist:**
- ✅ Temperature interpolation: 10-40× speedup measured
- ✅ MC sweeps: 20× speedup (OpenMP backend)
- ✅ Overall application: 15-30× speedup documented
- ✅ Memory usage: < 500 MB for 1M sites
- ✅ Scalability: Speedup maintained across problem sizes

**Deliverables:**

**1. Code:**
- `src/KOKKOS/temperature_source_hdf5_unstructured_kokkos.h/cpp` (working)
- `src/KOKKOS/app_additive_ext_temp_texture_kokkos.h/cpp` (working)
- CMake integration (automatic with package system)

**2. Documentation:**
- Update this roadmap with Phase 5 completion (commit hash)
- Performance benchmark report (CPU vs Kokkos)
- Validation report (physics comparison)

**3. Example input files:**
- `examples/ReducedTempAM/QuatTest/in.additive.kk` (Kokkos version)
- README with usage instructions

**Expected outcome**: Production-ready additive manufacturing Kokkos implementation

---

#### Technical Architecture Summary

**Data Flow:**

```
1. HDF5 File (Host)
   ↓
2. Load Layer (Host) → mesh, times, temps arrays
   ↓
3. Build Cache (Host) → find elements for all sites
   ↓
4. Transfer to Device (9 ms, once per layer)
   ├── times_device
   ├── temps_device
   ├── cache_device
   └── dataCounts_device
   ↓
5. Temperature Interpolation (Device) ← PARALLEL (10-40× faster)
   ├── Binary search in time series
   ├── Linear interpolation (temporal)
   └── Barycentric interpolation (spatial)
   ↓
6. T[] array on device → used by MC kernels
   ↓
7. Phase Transitions (Host) → melting, nucleation
   ↓
8. MC Sweeps (Device) ← PARALLEL (20× faster)
   ├── Weighted neighbor selection
   ├── Metropolis acceptance
   └── Quaternion updates
   ↓
9. MPI Communication (Host)
   └── Resync to device
```

**Performance breakdown (1M sites, typical):**

| Operation | CPU Time | Kokkos Time | Speedup | Frequency |
|-----------|----------|-------------|---------|-----------|
| HDF5 layer load | 50 ms | 50 ms | 1× | Once per layer |
| Cache building | 500 ms | 500 ms | 1× | Once per layer |
| Device transfer | N/A | 9 ms | N/A | Once per layer |
| **Temperature interp** | **100 ms** | **3 ms** | **33×** | **Every timestep** |
| Phase transitions | 10 ms | 10 ms | 1× | Every timestep |
| **MC sweeps** | **200 ms** | **10 ms** | **20×** | **Every timestep** |
| MPI communication | 5 ms | 5 ms | 1× | Every timestep |
| **Total per timestep** | **315 ms** | **28 ms** | **11×** | - |

**Per layer (50 timesteps):**
- CPU: 550 ms (setup) + 50 × 315 ms = 16.3 seconds
- Kokkos: 559 ms (setup) + 50 × 28 ms = 2.0 seconds
- **Speedup: 8× (includes setup overhead)**
- **For 100 layers: 27 minutes → 3.3 minutes** ✅

---

#### Success Criteria

**Phase 5 is complete when:**
1. ✅ Both Kokkos classes compile and run
2. ✅ Temperature interpolation achieves 10× speedup minimum
3. ✅ MC sweeps achieve 15× speedup minimum (combined with parent optimizations)
4. ✅ Physics validation: All test cases match CPU results
5. ✅ Multi-MPI: Tested with up to 6 ranks successfully
6. ✅ Documentation: Roadmap updated, performance documented
7. ✅ Production testing: Large-scale runs (1M+ sites) validated

---

## 6. Component Prioritization

### Tier 1: High Impact, Lower Complexity (Start Here)

#### 1. Site Energy Calculations (`site_energy`)

**Why:**
- Embarrassingly parallel (no dependencies between sites)
- Called frequently in both KMC and rKMC
- Clear performance benefit on GPU
- Simple to implement and validate

**Implementation:**
```cpp
Kokkos::parallel_for("SiteEnergy", nlocal,
    KOKKOS_LAMBDA(int i) {
        energy(i) = compute_site_energy(i);
    });
```

**Expected Speedup:** 10-50x on GPU for neighbor-intensive calculations

#### 2. Rejection KMC Sweeps (`iterate_rejection`)

**Why:**
- Large parallel loops over all sites (~209 occurrences found)
- Significant portion of runtime for many applications
- Can use graph coloring to avoid conflicts
- Well-suited for GPU parallelism

**Implementation:**
- Parallel sweep over sites of same color
- Metropolis acceptance/rejection per site
- Atomic operations or coloring for conflict avoidance

**Expected Speedup:** 5-20x on GPU

#### 3. AppPotts and Variants

**Why:**
- Simplest applications to start with
- Most widely used in SPPARKS community
- Good baseline for measuring acceleration
- Small codebase for initial testing

**Variants:**
- `app_potts.cpp` - Basic Potts model
- `app_potts_neighonly.cpp` - Neighbor-only energy
- `app_potts_additive.cpp` - AM simulation

**Expected Speedup:** 3-10x overall application speedup

### Tier 2: Medium Impact, Medium Complexity

#### 4. Neighbor List Construction

**Why:**
- Computed once, used many times
- GPU-friendly parallel algorithms available
- Benefits all lattice-based apps
- Moderate implementation complexity

**Approaches:**
- Cell list method on GPU
- Parallel sorting algorithms
- Kokkos neighbor list utilities

#### 5. Propensity Calculations (`site_propensity`)

**Why:**
- Required for KMC solvers
- Per-site parallel computation
- Can keep on device, transfer only total for solver

**Challenge:**
- Need efficient reduction for total propensity
- Interface with CPU-based tree solver

#### 6. AppPottsQuaternion

**Why:**
- Complex quaternion math ideal for GPU
- Higher computational intensity than basic Potts
- Important for materials science applications (grain growth)

**Features:**
- Quaternion operations (multiply, rotate, misorientation)
- Grain boundary energy calculations
- Temperature-dependent kinetics

**Expected Speedup:** 10-30x on quaternion operations

### Tier 3: High Impact, Higher Complexity

#### 7. Additive Manufacturing Apps

**Why:**
- Large-scale industrial applications
- Multiple computational kernels (temperature, grain growth)
- High performance requirements
- Complex coupling between physics

**Components:**
- Temperature field management
- Bézier curve evaluation
- Nucleation models
- Grain boundary tracking

**Expected Speedup:** 5-15x for large-scale simulations

#### 8. KMC Solvers (`SolveTree`, `SolveGroup`)

**Why:**
- Core algorithmic component
- Currently CPU-bottleneck for small systems
- Challenging due to sequential nature

**Approaches:**
- Hybrid: Propensities on GPU, selection on CPU
- GPU tree: Parallel tree update, binary search
- Alternative algorithms: Approximate methods

**Challenge:** Maintaining algorithmic efficiency on GPU

#### 9. Domain Decomposition and Communication

**Why:**
- Critical for large-scale parallel performance
- Minimize data transfer overhead
- Enable multi-GPU scaling

**Components:**
- GPU-aware MPI integration
- Halo exchange on device
- Pack/unpack kernels
- Overlap communication with computation

**Expected Benefit:** 2-5x for communication-bound problems

---

## 7. Technical Challenges

### A. Random Number Generation

**Challenge:**

SPPARKS extensively uses `RandomPark` class for:
- Event selection in KMC
- Metropolis acceptance in rKMC
- Site initialization
- Stochastic processes

Standard RNG is not thread-safe for GPU parallelism.

**Solution:**

Use `Kokkos::Random_XorShift64_Pool`:

```cpp
// Initialize pool with seed
using pool_type = Kokkos::Random_XorShift64_Pool<>;
pool_type rand_pool(seed);

// In parallel kernel
Kokkos::parallel_for(N, KOKKOS_LAMBDA(int i) {
    // Get thread-specific generator
    auto generator = rand_pool.get_state();

    // Generate random numbers
    double rn = generator.drand();      // [0, 1)
    int ri = generator.urand(nmax);     // [0, nmax)

    // Return generator to pool
    rand_pool.free_state(generator);
});
```

**Considerations:**
- Different RNG algorithm → different random sequence
- May affect reproducibility
- Solution: Provide option for deterministic mode or document differences

### B. KMC Event Selection

**Challenge:**

Tree-based event selection (`solve_tree.cpp`) is inherently sequential:
1. Build binary tree of propensities (can be parallelized)
2. Select random point in [0, total_propensity)
3. Traverse tree to find event (sequential)

**Options:**

**Option 1: Hybrid CPU/GPU (Simplest)**
```
Pros:
- Minimal code changes
- Maintains exact algorithm
- Good for propensity-dominated workloads

Cons:
- Data transfer overhead
- CPU becomes bottleneck for fast GPUs
```

**Option 2: GPU Tree (More Complex)**
```
Pros:
- All computation on GPU
- No transfer overhead
- Scalable to many events

Cons:
- Tree traversal still sequential per event
- Complex implementation
- May not be faster for small systems
```

**Option 3: Approximate/Alternative Methods**
```
Pros:
- Better GPU parallelism
- Can select multiple events simultaneously

Cons:
- Changes algorithm
- May affect physics/accuracy
- Requires validation
```

**Recommendation:**
- Start with hybrid approach (Option 1)
- Measure performance bottlenecks
- Implement GPU tree (Option 2) if propensity computation is fast
- Consider alternatives (Option 3) only if necessary

### C. Ghost Site Communication

**Challenge:**

Current MPI communication well-optimized for CPU:
- Ghost sites from neighboring processors
- Pack/unpack operations
- Halo exchange patterns

GPU version needs different strategy:
- Data resides on device
- Pack/unpack should be GPU kernels
- Need GPU-aware MPI or staging

**Solution:**

**GPU-Aware MPI** (if available):
```cpp
// Check if MPI library supports CUDA-aware communication
#ifdef MPIX_CUDA_AWARE_SUPPORT
    if (MPIX_Query_cuda_support()) {
        // Direct device-to-device
        MPI_Send(device_ptr, ...);
    }
#endif
```

**Fallback (Staging through Host)**:
```cpp
// Copy device → host → MPI → host → device
Kokkos::deep_copy(h_send, d_send);
MPI_Sendrecv(h_send.data(), ..., h_recv.data(), ...);
Kokkos::deep_copy(d_recv, h_recv);
```

**Optimization:**
- Overlap communication with computation
- Use non-blocking MPI + Kokkos streams
- Pack/unpack on device

**Performance Impact:**
- GPU-aware MPI: Minimal overhead
- Staging: 10-50% overhead depending on problem size

### D. Memory Footprint

**Challenge:**

`Kokkos::DualView` maintains both host and device copies:
- Doubles memory usage
- Large simulations may exceed GPU memory
- Need careful memory management

**Solutions:**

**1. Selective Duplication:**
```cpp
// Only duplicate frequently accessed data
Kokkos::DualView<int*> k_spin;      // Duplicated (used every step)
Kokkos::View<double*> d_energy;     // Device-only (intermediate)
std::vector<int> h_output;          // Host-only (diagnostics)
```

**2. Unified Memory (UVM):**
```cpp
// On supported platforms, use UVM
Kokkos::View<int*, Kokkos::CudaUVMSpace> k_spin;
// Automatic migration, single copy
// May have performance penalty
```

**3. Out-of-Core:**
```cpp
// For very large problems, stream data to GPU
// Process in chunks
for (int chunk = 0; chunk < nchunks; chunk++) {
    load_chunk_to_gpu(chunk);
    process_chunk_on_gpu(chunk);
    store_results(chunk);
}
```

**Memory Estimates:**

For a 10M site simulation with quaternions:
- Site IDs: 10M × 8 bytes = 80 MB
- Coordinates: 10M × 24 bytes = 240 MB
- Spins: 10M × 4 bytes = 40 MB
- Quaternions: 10M × 32 bytes = 320 MB
- Neighbors: 10M × 12 × 4 bytes = 480 MB
- **Total per copy: ~1.2 GB**
- **DualView: ~2.4 GB**

Modern GPUs (32+ GB) can handle this, but watch for memory pressure.

### E. Work Imbalance

**Challenge:**

Neighbor counts vary per site:
- Interior sites: Full coordination
- Surface/boundary sites: Fewer neighbors
- Causes GPU thread divergence and load imbalance

**Solutions:**

**1. Sort by Neighbor Count:**
```cpp
// Group sites with similar neighbor counts
Kokkos::parallel_for(nlocal, KOKKOS_LAMBDA(int i) {
    int nn = numneigh(sorted_indices(i));
    // More uniform work per thread
});
```

**2. Hierarchical Parallelism:**
```cpp
// Use TeamPolicy with dynamic thread allocation
Kokkos::TeamPolicy<> policy(nsites, Kokkos::AUTO);
Kokkos::parallel_for(policy,
    KOKKOS_LAMBDA(const member_type& team) {
        int i = team.league_rank();
        int nn = numneigh(i);

        // Parallelize over neighbors within team
        Kokkos::parallel_for(
            Kokkos::TeamThreadRange(team, nn),
            [&](int j) {
                // Process neighbor j
            });
    });
```

**3. Batch Similar Work:**
```cpp
// Separate kernels for high/low neighbor counts
parallel_for(high_neighbor_sites, ...);
parallel_for(low_neighbor_sites, ...);
```

### F. Precision and Determinism

**Challenge:**

GPU floating-point operations may differ from CPU:
- Different rounding modes
- Different reduction orders
- Parallel floating-point sums are non-associative

**Impacts:**
- Slightly different energies
- Different KMC trajectories
- May affect long-time behavior

**Solutions:**

**1. Document Differences:**
```
"GPU results may differ from CPU due to floating-point
precision and parallelism. Physics should be statistically
equivalent but individual trajectories will differ."
```

**2. Deterministic Mode:**
```cpp
// Option to force deterministic reductions
Kokkos::parallel_reduce(
    Kokkos::RangePolicy<Kokkos::Schedule<Kokkos::Static>>(0, N),
    // ... functor ...
    Kokkos::Sum<double, Kokkos::HostSpace>(result));
```

**3. Validation:**
- Compare ensemble averages (not individual runs)
- Verify conserved quantities
- Check statistical distributions
- Long-time property validation

**Recommendation:**
- Accept minor differences as inherent to parallelism
- Focus on statistical equivalence
- Provide deterministic mode if needed

### G. Race Conditions in Parallel Rejection Monte Carlo

**Challenge:**

Parallel rejection Monte Carlo sweeps create race conditions when multiple threads simultaneously read and write to the same site arrays:

**The Race Condition:**
```cpp
// Thread A (site i): reads neighbor spins, writes to spin[i]
int energy = 0;
for (int j = 0; j < numneigh(i); j++) {
    int nj = neighbor(i, j);
    if (spin(i) != spin(nj)) energy++;  // Reading neighbor spin
}
// Metropolis acceptance...
spin(i) = newstate;  // Writing own spin

// Thread B (site nj): simultaneously writing spin[nj]
spin(nj) = different_state;  // Creates inconsistency for Thread A!
```

**Consequences:**
- Non-deterministic energy calculations
- Ghost site desynchronization in MPI runs
- Grain boundary artifacts
- Physics incorrectness

**Failed Solutions:**
1. ❌ **Atomic operations**: Too slow on GPU (~10x slowdown)
2. ❌ **Coloring schemes**: Restricts parallelism, complex with sectors
3. ❌ **Sector-only separation**: Still has races within sectors

**✅ Solution: Double-Buffer Framework**

Implemented in `src/kokkos_double_buffer.h` (November 2025):

**Architecture:**
```
Host Array (T*)  ↔  DualView (staging)  ↔  Device Views
                                               ├─ read_buffer  (immutable during sweep)
                                               └─ write_buffer (independent writes)
```

**Key Classes:**

**1. KokkosDoubleBuffer<T>** - Template class for double-buffered arrays
```cpp
template<typename T>
class KokkosDoubleBuffer {
public:
    Kokkos::DualView<T*> staging;        // Host↔device staging
    Kokkos::View<T*> read_buffer;        // Device-only read
    Kokkos::View<T*> write_buffer;       // Device-only write

    void allocate(const char* label, size_t n);
    void swap();                          // O(1) pointer swap
    void sync_to_device_full(const T* host_data);  // Init both buffers
    void sync_to_device(const T* host_data);       // MPI: update both
    void sync_to_host(T* host_data);               // Pre-MPI copy
};
```

**2. KokkosDoubleBufferManager** - Manages multiple arrays
```cpp
class KokkosDoubleBufferManager {
public:
    void add_int_buffer(const char* name);
    void add_double_buffer(const char* name);
    void allocate_all(size_t n);
    void swap_all();                      // Swap all buffers at once

    KokkosDoubleBuffer<int>& get_int_buffer(const char* name);
    KokkosDoubleBuffer<double>& get_double_buffer(const char* name);
};
```

**Usage Example (AppPottsQuaternionKokkos):**

```cpp
class AppPottsQuaternionKokkos : public AppPottsQuaternion {
    // Double-buffered site data
    KokkosDoubleBuffer<int> buffer_spin;
    KokkosDoubleBuffer<double> buffer_q0, buffer_qx, buffer_qy, buffer_qz;

    void grow_app() {
        int nall = nlocal + nghost;
        buffer_spin.allocate("spin", nall);
        buffer_q0.allocate("q0", nall);
        // ... allocate other buffers
    }

    void iterate_rejection(double stoptime) {
        for (int iset = 0; iset < nset_loop; iset++) {
            // 1. Sync device → host before MPI
            sync_data_to_host();

            // 2. MPI ghost site communication
            comm->sector(iset);

            // 3. Sync host → both device buffers after MPI
            sync_data_to_device();

            // 4. Device kernel reads from read_buffer, writes to write_buffer
            RejectionSweepFunctor functor(
                buffer_spin.read_buffer,      // Read
                buffer_spin.write_buffer,     // Write
                buffer_q0.read_buffer, buffer_q0.write_buffer,
                // ...
            );
            Kokkos::parallel_reduce("Sweep", nlocal, functor, naccept);

            // 5. Swap buffers (O(1) operation)
            buffer_spin.swap();
            buffer_q0.swap();
            // ... swap other buffers
        }
    }
};
```

**Kernel Implementation:**
```cpp
struct RejectionSweepFunctor {
    View<int*> spin_read, spin_write;        // Separate read/write
    View<double*> q0_read, q0_write;

    KOKKOS_INLINE_FUNCTION
    void operator()(const int i, int& accepted) const {
        // Read current state from READ buffer
        int oldstate = spin_read(i);
        double q_old[4] = {q0_read(i), qx_read(i), ...};

        // Calculate energy using READ buffer
        double energy = 0.0;
        for (int j = 0; j < numneigh(i); j++) {
            int nj = neighbor(i, j);
            double qj[4] = {q0_read(nj), ...};  // Safe read!
            energy += compute_disorientation(q_old, qj);
        }

        // ... Metropolis acceptance ...

        // Write result to WRITE buffer
        spin_write(i) = accepted ? newstate : oldstate;
        q0_write(i) = accepted ? q_new[0] : q_old[0];
        // ... write other components
    }
};
```

**Key Design Decisions:**

**1. Why Separate DualView and Device Buffers?**
- DualView: Manages host↔device staging for MPI
- Device buffers: Enable double-buffering pattern
- Can't easily have two independent device copies in DualView

**2. Why Update BOTH Buffers After MPI?**
```cpp
void sync_data_to_device(const T* host_data) {
    // Copy host → DualView → BOTH read and write buffers
    Kokkos::deep_copy(read_buffer, staging.d_view);
    Kokkos::deep_copy(write_buffer, staging.d_view);  // Critical!
}
```
- After MPI, host has authoritative ghost site data
- read_buffer needs updated ghosts for energy calculations
- write_buffer needs same starting point to maintain consistency
- Without this: next-next sweep reads stale data after buffer swap

**3. App-Specific Transformations?**
Framework is **pure** - no built-in transformations:
```cpp
// App handles transformations before/after sync
void AppPottsQuaternionKokkos::sync_data_to_device() {
    // 1. Normalize quaternions (app-specific)
    double* q_norm = normalize_quaternions(q0, qx, qy, qz, nall);

    // 2. Use framework to sync normalized data
    buffer_q0.sync_to_device(q_norm);
    buffer_qx.sync_to_device(qx_norm);
    // ...

    delete[] q_norm;
}
```

**Performance Characteristics:**

**Memory Overhead:**
- Doubles device memory for buffered arrays
- Example: 5 arrays × 1M sites × 8 bytes × 2 buffers = **80 MB**
- Acceptable on modern GPUs (16+ GB)

**Computational Overhead:**
- **Buffer swap**: O(1) pointer exchange (~5 pointer swaps per sweep)
- **MPI sync**: 2× device→host transfers (read and write buffers)
- **Typical**: <5% overhead, dominated by MPI itself

**Measured Benefits (AppPottsQuaternionKokkos):**
- ✅ **14.7× speedup** with 4 MPI ranks (vs CPU single-core)
- ✅ **Zero desynchronized spins** in MPI runs
- ✅ **Perfect grain structure** integrity
- ✅ **Physics correctness** validated vs CPU version

**When to Use Double-Buffering:**

✅ **Use for:**
- Rejection Monte Carlo with parallel sweeps
- Apps requiring strict consistency (no race conditions)
- Multiple site properties updated atomically
- Structured data where neighbors read during writes

❌ **Don't use for:**
- Kinetic Monte Carlo (sequential event selection)
- Single-threaded CPU apps
- Apps that can tolerate small inconsistencies
- Apps using coloring schemes that prevent races

**Applicable Apps:**

**Current Status (Phase 3.2 Complete - November 13, 2025):**
- ✅ **AppPottsQuaternionKokkos**: Uses double-buffering (Phase 2.3)
- ✅ **AppPottsNeighKokkos**: Refactored to use double-buffering (Phase 3.2)
- ✅ **AppPottsNeighOnlyKokkos**: Refactored to use double-buffering (Phase 3.2)
- ✅ **AppPottsKokkos**: Refactored to use double-buffering (Phase 3.2)

**Implementation Pattern (Used in All Apps):**
```cpp
// In grow_app() - allocate buffers
buffer_spin.allocate("spin", nlocal + nghost);

// In iterate_rejection() - use double-buffering
for (int iset = 0; iset < nset_loop; iset++) {
    // 1. Sync device → host before MPI
    sync_data_to_host();

    // 2. MPI ghost site communication
    comm->sector(iset);

    // 3. Sync host → both device buffers after MPI
    sync_data_to_device();

    // 4. Kernel reads from read_buffer, writes to write_buffer
    RejectionSweepFunctor functor(
        buffer_spin.read_buffer,  // Read
        buffer_spin.write_buffer, // Write
        // ...
    );

    // 5. Swap buffers (O(1) operation)
    buffer_spin.swap();
}
```

**Completed Work (Phase 3.2):**
- ✅ Refactored AppPottsKokkos to use double-buffering
- ✅ Refactored AppPottsNeighKokkos to use double-buffering
- ✅ Refactored AppPottsNeighOnlyKokkos to use double-buffering
- ✅ All apps validated for physics correctness
- ✅ Performance impact measured: minimal overhead (<5%)

**Future Work:**
- Measure performance impact on GPU hardware
- Apply pattern to new apps (AppIsing, AppDiffusion)
- Consider automatic race detection tools for validation

**Key Lessons:**
1. **Simplicity wins**: Pure framework, app-specific logic stays in app
2. **Performance**: O(1) swaps + minimal overhead = practical solution
3. **Correctness first**: 14.7× speedup means nothing with wrong physics
4. **Reusability**: Template design enables future apps with different array types
5. **Documentation**: Clear usage patterns enable others to adopt

---

## 8. Implementation Roadmap

### Year 1: Foundation and Proof of Concept

#### Q1: Infrastructure (Months 1-3) ✅ **COMPLETED**

**Deliverable:** ✅ Working infrastructure with base classes, build system, and test framework
- Implemented: MemoryKokkos, AppKokkos, AppLatticeKokkos
- Build system: CMake and Makefile with KOKKOS package
- Documentation: src/KOKKOS/README.md

**📖 See [KOKKOS_IMPLEMENTATION_HISTORY.md](KOKKOS_IMPLEMENTATION_HISTORY.md) for details**

#### Q2: First Application (Months 4-6) ✅ **COMPLETED**

**Deliverable:** ✅ Four production-ready Potts apps with 20× CPU speedup
- AppPottsKokkos, AppPottsNeighKokkos, AppPottsNeighOnlyKokkos, AppPottsQuaternionKokkos
- Phases 1-3.5 complete: Infrastructure → MPI+Sectors → Double-buffering → Optimization → Benchmarking
- 20× speedup achieved, physics validated, zero race conditions

**📖 See [KOKKOS_IMPLEMENTATION_HISTORY.md](KOKKOS_IMPLEMENTATION_HISTORY.md) for details**

#### Q3: Refinement (Months 7-9) ⏳ **IN PROGRESS - Phase 3.6**

**Immediate priorities (Month 7):**
- [ ] **GPU backend testing** (CUDA on NVIDIA, HIP on AMD)
  - Expected: 10-30× additional speedup beyond current 20× CPU baseline
  - Validate physics correctness across backends
  - Measure actual sync overhead on GPU hardware
- [ ] **Diagnostic Kokkos-ification** (see KOKKOS_DIAGNOSTICS_ANALYSIS.md)
  - diag_energy/kk - 5-20× speedup
  - diag_array/kk - 10-50× speedup
- [ ] Additional apps: AppIsingKokkos, AppDiffusionKokkos

**Optimization (Months 8-9):**
- [ ] Ghost-region-only partial syncs
- [ ] Team policy optimization for GPUs
- [ ] Async sync with Kokkos streams
- [ ] Device-side MPI pack/unpack kernels

**Deliverable:** GPU-validated apps, optimized performance, diagnostic support

#### Q4: Validation (Months 10-12)

**Month 10:**
- [ ] Comprehensive physics validation
- [ ] Compare ensemble statistics CPU vs GPU
- [ ] Long-time behavior validation
- [ ] Regression test suite

**Month 11:**
- [ ] Performance benchmarking suite
- [ ] Scaling studies (weak and strong)
- [ ] Multi-GPU testing
- [ ] Documentation

**Month 12:**
- [ ] User testing and feedback
- [ ] Bug fixes
- [ ] Optimization based on profiling
- [ ] Prepare for release

**Deliverable:** Production-ready Kokkos support for basic apps, validated and documented

### Year 2: Expansion and Optimization

#### Q1: Complex Applications (Months 13-15)

**Month 13:** ✅ **AppPottsQuaternionKokkos already complete** (see history)

**Month 14-15: AM Applications**
- [ ] `AppPottsAMBezierKokkos` - AM with Bézier temperature profiles
  - Temperature field on device
  - Bézier curve evaluation kernels
  - Nucleation models on GPU
- [ ] `AppPottsAMWeldKokkos` - Welding simulations
  - Moving heat source on device
  - Dynamic temperature updates
- [ ] Apply double-buffer framework from Phase 3.1
- [ ] Leverage selective sync and profiling infrastructure

**Deliverable:** AM production apps running on GPU with validated physics

#### Q2: Solver and Communication (Months 16-18)

**Month 16:**
- [ ] GPU tree solver implementation
- [ ] Parallel tree building
- [ ] Event selection optimization

**Month 17:**
- [ ] GPU-aware MPI integration
- [ ] Pack/unpack kernels on device
- [ ] Halo exchange optimization

**Month 18:**
- [ ] Multi-GPU support
- [ ] Load balancing for GPUs
- [ ] Scaling studies on clusters

**Deliverable:** Optimized communication, multi-GPU scaling

#### Q3: Production Readiness (Months 19-21)

**Month 19:**
- [ ] Large-scale testing (millions to billions of sites)
- [ ] Memory optimization
- [ ] Profiling and tuning

**Month 20:**
- [ ] Production workflows
- [ ] Integration with analysis tools
- [ ] HDF5 output from GPU

**Month 21:**
- [ ] User training materials
- [ ] Tutorials and examples
- [ ] Best practices guide

**Deliverable:** Production-ready at scale

#### Q4: Advanced Features (Months 22-24)

**Month 22:**
- [ ] Additional apps (diffusion, Ising, etc.)
- [ ] Advanced solver options
- [ ] Custom user apps with Kokkos

**Month 23:**
- [ ] Performance portability tuning
- [ ] Architecture-specific optimization
- [ ] Exascale system testing

**Month 24:**
- [ ] Community engagement
- [ ] Publication of results
- [ ] Future roadmap

**Deliverable:** Comprehensive Kokkos support across SPPARKS

---

## 9. Success Metrics

### Performance Targets

**GPU Acceleration:**
- **Site energy calculations:** 10-50x speedup
- **Rejection KMC sweeps:** 5-20x speedup
- **AppPotts overall:** 3-10x speedup
- **AppPottsQuaternion:** 10-30x speedup
- **Large AM simulations:** 5-15x speedup

**CPU Performance:**
- OpenMP backend within 10% of current serial code
- No regression in non-Kokkos code paths

**Scaling:**
- Strong scaling to 1000+ GPUs
- Weak scaling efficiency > 80% to exascale
- Multi-GPU efficiency > 90%

### Code Quality

**Maintainability:**
- < 20% code duplication between Kokkos and non-Kokkos versions
- Clear separation of physics and parallel infrastructure
- Comprehensive comments and documentation

**Correctness:**
- Physics accuracy within 1% of CPU results (ensemble averages)
- Pass all existing regression tests
- New Kokkos-specific test suite

**Portability:**
- Build and run on:
  - Linux (CUDA, HIP, SYCL, OpenMP)
  - macOS (OpenMP, Serial)
  - Windows (OpenMP, Serial)
- Support for:
  - NVIDIA GPUs (Volta, Ampere, Hopper)
  - AMD GPUs (MI200, MI300)
  - Intel GPUs (Data Center Max, Ponte Vecchio)

### Usability

**Build System:**
- Simple one-command build: `./build_cmake.sh --kokkos`
- Auto-detect available backends
- Clear error messages for missing dependencies
- Pre-built binaries for common platforms

**User Interface:**
- Transparent suffix system: `app_style potts/kk`
- Automatic backend selection
- Performance feedback during run
- Good default parameters

**Documentation:**
- User guide with examples
- Performance tuning guide
- Developer guide for new apps
- FAQ and troubleshooting

---

## 10. References and Resources

### Kokkos Project

- **Official Website:** https://kokkos.org
- **GitHub Repository:** https://github.com/kokkos/kokkos
- **Documentation:** https://kokkos.github.io/kokkos-core-wiki/
- **Tutorials:** https://github.com/kokkos/kokkos-tutorials
- **Slack Community:** kokkos.slack.com

### LAMMPS Kokkos

- **LAMMPS Website:** https://www.lammps.org
- **Kokkos Package Docs:** https://docs.lammps.org/Speed_kokkos.html
- **GitHub:** https://github.com/lammps/lammps
- **Key Papers:**
  - "Preparing LAMMPS for Exascale Platforms" (2024)
  - "Performance Portability in LAMMPS with Kokkos" (2022)

### SPARTA Kokkos

- **SPARTA Website:** https://sparta.github.io
- **GitHub:** https://github.com/sparta/sparta
- **Kokkos Implementation:** https://sparta.github.io/doc/Section_accelerate.html

### Publications

**Kokkos Ecosystem:**
1. Trott et al., "Kokkos 3: Programming Model Extensions for the Exascale Era", IEEE TPDS (2022)
2. Edwards et al., "Kokkos: Enabling manycore performance portability through polymorphic memory access patterns", J. Parallel Distrib. Comput. (2014)

**Performance Portability:**
3. Bertagna et al., "Performance Portability of Earth System Models with User-Controlled MORRIS Directives", Int. J. High Perform. Comput. Appl. (2021)
4. Koskela et al., "A Novel Multi-Level Integrated Roofline Model Approach for Performance Characterization", IEEE Cluster (2018)

**GPU Acceleration of Monte Carlo:**
5. Martínez et al., "Accelerating Kinetic Monte Carlo Simulations with GPUs", Comput. Phys. Commun. (2013)
6. Nandipati et al., "Massively parallel kinetic Monte Carlo simulations of charge carrier transport in organic semiconductors", J. Chem. Theory Comput. (2016)

### SPPARKS

- **Website:** https://spparks.github.io
- **GitHub:** https://github.com/spparks/spparks
- **Manual:** https://spparks.github.io/doc/Manual.html

### SPPARKS Kokkos Documentation

- **KOKKOS_IMPLEMENTATION_HISTORY.md** - Detailed history of Kokkos implementation (Phases 1-3.5, November 2025)
  - Complete chronology of infrastructure development, bug fixes, and optimizations
  - Performance benchmarking results (20× CPU speedup achieved)
  - Technical details of double-buffer framework, selective sync, and MPI synchronization
  - Git commit history and file modification records

- **KOKKOS_DIAGNOSTICS_ANALYSIS.md** - Comprehensive analysis of Kokkos-ifying diagnostic styles (diag_energy, diag_array, diag_cluster)
  - Evaluates GPU suitability and implementation complexity
  - Recommends diag_energy (5-20× speedup) and diag_array (10-50× speedup)
  - Advises against diag_cluster (poor GPU fit for graph algorithms)
  - Provides detailed implementation roadmap and code examples

### Related Technologies

- **KokkosComm:** MPI+Kokkos communication library
- **Cabana:** Particle simulation toolkit using Kokkos
- **ArborX:** Spatial search library with Kokkos
- **HighFive:** Modern C++ HDF5 wrapper (already in SPPARKS)

---

## Appendix A: Build Instructions

### Quick Start

```bash
# Clone with Kokkos submodule
cd /Users/Tron/spparks
git checkout SURGE-kokkos
git submodule update --init --recursive

# Build with CUDA
cd src
./build_cmake.sh --kokkos --kokkos-backend cuda --kokkos-arch Ampere80

# Build with HIP (AMD)
./build_cmake.sh --kokkos --kokkos-backend hip --kokkos-arch MI250

# Build with OpenMP (CPU)
./build_cmake.sh --kokkos --kokkos-backend openmp

# Run test
mpirun -np 6 ./spk_kokkos < ../examples/potts/in.potts.kk
```

### CMake Options

```cmake
# Kokkos package
-DSPPARKS_PACKAGE_KOKKOS=ON/OFF

# Backend selection
-DSPPARKS_KOKKOS_BACKEND=CUDA|HIP|SYCL|OPENMP|SERIAL

# Architecture (GPU or CPU)
-DSPPARKS_KOKKOS_ARCH=Volta70|Ampere80|Hopper90|MI250|PVC|SKX|...

# External vs bundled Kokkos
-DSPPARKS_KOKKOS_EXTERNAL=ON    # Use system Kokkos
-DSPPARKS_KOKKOS_EXTERNAL=OFF   # Use bundled lib/kokkos
```

### Environment Variables

```bash
# For CUDA
export CUDA_VISIBLE_DEVICES=0,1,2,3

# For Kokkos tuning
export KOKKOS_NUM_DEVICES=4
export KOKKOS_MAP_DEVICE_ID_BY=mpi_rank

# For GPU-aware MPI
export MPICH_GPU_SUPPORT_ENABLED=1
```

---

## Appendix B: Example Usage

### Running with Kokkos

```bash
# Input script: in.potts.kk
app_style      potts/kk 100
dimension      3
lattice        sc/6n 1.0
region         box block 0 100 0 100 0 100
create_box     box
create_sites   box
set            site range 1 100

# Use Kokkos solver
solve_style    tree/kk

temperature    1.0
stats          10.0

dump           1 text 100.0 dump.potts.kk

run            1000.0
```

### Command Line Options

```bash
# Auto-select backend
mpirun -np 4 spk_kokkos < in.script

# Force device execution
mpirun -np 4 spk_kokkos -kokkos-device < in.script

# Force host execution
mpirun -np 4 spk_kokkos -kokkos-host < in.script

# Specify number of threads/team
mpirun -np 4 spk_kokkos -kokkos-nthreads 8 -kokkos-team-size 32 < in.script
```

---

## Appendix C: Development Guidelines

### Creating a New Kokkos App

1. **Inherit from AppKokkos or AppLatticeKokkos:**
```cpp
class AppMyAppKokkos : public AppLatticeKokkos {
    // ...
};
```

2. **Define Kokkos data structures:**
```cpp
Kokkos::DualView<int*> k_my_data;
```

3. **Implement Kokkos kernels:**
```cpp
struct MyKernel {
    KOKKOS_INLINE_FUNCTION
    void operator()(int i) const {
        // Computation
    }
};
```

4. **Register with macro:**
```cpp
#ifdef SPK_KOKKOS
AppStyle(myapp/kk, AppMyAppKokkos)
#endif
```

### Best Practices

1. **Use descriptive kernel names:**
```cpp
Kokkos::parallel_for("AppPotts::ComputeEnergies", N, functor);
```

2. **Minimize host-device transfers:**
```cpp
// Bad: sync every iteration
for (int i = 0; i < nsteps; i++) {
    k_data.sync_device();
    compute_on_device();
    k_data.sync_host();
}

// Good: sync once
k_data.sync_device();
for (int i = 0; i < nsteps; i++) {
    compute_on_device();
}
k_data.sync_host();
```

3. **Use const views when possible:**
```cpp
Kokkos::View<const int*> neighbor;  // Read-only
```

4. **Profile before optimizing:**
```bash
nvprof ./spk_kokkos < in.script
nsys profile ./spk_kokkos < in.script
```

---

## Next Steps

### ✅ Completed
1. ~~**Review this roadmap** with SPPARKS development team~~
2. ~~**Obtain buy-in** from stakeholders and users~~
3. ~~**Set up development branch** and infrastructure~~ (SURGE-kokkos created)
4. ~~**Start Phase 1** (Months 1-3): Build system and base classes~~ (COMPLETED November 12, 2025)
5. ~~**Implement proof-of-concept** with AppPotts~~ (AppPottsKokkos implemented)

### 🔄 In Progress / Next
6. **Initialize Kokkos library**:
   - Add as submodule: `git submodule add https://github.com/kokkos/kokkos.git lib/kokkos`
   - Or install via system package manager
7. **Build and test**:
   - Build SPPARKS with KOKKOS package enabled
   - Run validation tests (examples/potts/in.potts.kk)
   - Compare results vs standard CPU implementation
8. **Measure and validate** performance:
   - Benchmark AppPottsKokkos vs AppPotts
   - Measure host-device transfer overhead
   - Optimize kernel launch parameters
9. **Integrate into iteration loops**:
   - Add Kokkos initialization to spparks.cpp
   - Hook Kokkos kernels into main iteration methods
   - Enable runtime backend selection
10. **Scale up**:
    - Additional apps (AppPottsQuaternion, AM models)
    - KMC solver optimization
    - Multi-GPU support
11. **Engage with Kokkos community** for support and best practices

**Contact:**
- Kokkos team: kokkos-users@software.sandia.gov
- Kokkos Slack: kokkos.slack.com
- SPPARKS GitHub: https://github.com/spparks/spparks

**Status:** Initial implementation complete (November 12, 2025). Ready for library integration and testing.

---

*This document is a living roadmap and will be updated as the Kokkos integration progresses.*
