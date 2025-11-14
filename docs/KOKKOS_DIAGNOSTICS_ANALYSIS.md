# Kokkos Diagnostics Analysis

**Date:** November 2025
**Branch:** SURGE-kokkos
**Status:** Research/Exploration (Not Yet Implemented)

## Executive Summary

This document analyzes the feasibility of Kokkos-ifying SPPARKS diagnostic styles to reduce host-device synchronization overhead and accelerate diagnostic computations on GPUs. Three diagnostics were evaluated:

| Diagnostic | Recommendation | Expected Speedup | Implementation Effort |
|------------|----------------|------------------|----------------------|
| **diag_energy** | ✅ **Implement** | 5-20x | 1-2 weeks |
| **diag_array** | ✅ **Implement** | 10-50x | 2-3 weeks |
| **diag_cluster** | ❌ **Skip** | Minimal/Negative | 4-8 weeks |

**Key Finding:** `diag_energy` and `diag_array` are excellent GPU candidates with clear performance benefits and straightforward implementation paths. `diag_cluster` should remain CPU-only due to its complex graph-based algorithm that performs poorly on GPUs.

---

## Table of Contents

1. [Current Implementation Analysis](#1-current-implementation-analysis)
2. [Data Access Patterns](#2-data-access-patterns)
3. [Existing Kokkos Patterns](#3-existing-kokkos-patterns)
4. [Detailed Diagnostic Analysis](#4-detailed-diagnostic-analysis)
5. [Recommended Architecture](#5-recommended-architecture)
6. [Implementation Roadmap](#6-implementation-roadmap)
7. [Performance Expectations](#7-performance-expectations)
8. [Risk Assessment](#8-risk-assessment)
9. [Code Examples](#9-code-examples)
10. [References](#10-references)

---

## 1. Current Implementation Analysis

### 1.1 DiagArray

**Location:** `src/diag_array.cpp`, `src/diag_array.h` (140 lines)

**Purpose:** Computes statistics (mean, sum, min, max) on integer or double arrays per site.

**Computational Pattern:**
```cpp
void DiagArray::compute() {
  // 1. Get total site count via MPI
  MPI_Allreduce(&nlocal, &nlocal_all, 1, MPI_INT, MPI_SUM, world);

  // 2. For each requested statistic:
  for (int i = 0; i < nvals; i++) {
    // 3. Local reduction (e.g., find minimum)
    double dtmp = 1.0e30;
    for (int j = 0; j < nlocal; j++) {
      if (dptr[ii][j] < dtmp) dtmp = dptr[ii][j];
    }

    // 4. Global MPI reduction
    MPI_Allreduce(&dtmp, &dtmp_all, 1, MPI_DOUBLE, MPI_MIN, world);

    // 5. Calculate statistic
    if (diag_method[i] == MEAN) vals[i] = dtmp_all / nlocal_all;
  }
}
```

**Characteristics:**
- Embarrassingly parallel (no inter-site dependencies)
- Simple reduction operations (sum, min, max, mean)
- Sequential memory access pattern
- Multiple MPI collectives (one per statistic)
- Complexity: O(nvals × nlocal)

### 1.2 DiagEnergy

**Location:** `src/diag_energy.cpp`, `src/diag_energy.h` (66 lines)

**Purpose:** Computes total system energy by summing site energies.

**Computational Pattern:**
```cpp
void DiagEnergy::compute() {
  // 1. Synchronize ghost sites
  applattice->comm->all();

  // 2. Sum energy over local sites
  double etmp = 0.0;
  for (int i = 0; i < nlocal; i++) {
    etmp += applattice->site_energy(i);  // Calls app-specific energy function
  }

  // 3. Global MPI sum
  MPI_Allreduce(&etmp, &energy, 1, MPI_DOUBLE, MPI_SUM, world);
}
```

**site_energy() for Potts model:**
```cpp
double AppPotts::site_energy(int i) {
  int eng = 0;
  int isite = spin[i];
  for (int j = 0; j < numneigh[i]; j++) {
    if (spin[neighbor[i][j]] != isite) eng++;
  }
  return (double)eng;
}
```

**Characteristics:**
- Moderately parallel (neighbor access but read-only)
- Requires ghost site data
- Similar to existing Kokkos energy kernels in apps
- Complexity: O(nlocal × avg_neighbors)

### 1.3 DiagCluster

**Location:** `src/diag_cluster.cpp`, `src/diag_cluster.h` (900+ lines)

**Purpose:** Identifies connected regions (clusters) of similar spins using graph traversal.

**Computational Pattern (Multi-Phase Algorithm):**

```cpp
void DiagCluster::generate_clusters() {
  // PHASE 1: Local cluster identification (depth-first search)
  for (int i = 0; i < nlocal; i++) {
    if (cluster_ids[i] != 0) continue;  // Already assigned

    int id = ++ncluster;
    std::stack<int> cluststack;
    applattice->push_new_site(i, cluster_ids, id, &cluststack);

    // Depth-first traversal
    while (cluststack.size()) {
      int ii = cluststack.top();
      cluststack.pop();
      vol++;

      // Push connected neighbors onto stack
      applattice->push_connected_neighbors(ii, cluster_ids, id, &cluststack);
    }
  }

  // PHASE 2: Assign global cluster IDs (MPI_Scan)
  MPI_Scan(&ncluster, &idoffset, 1, MPI_INT, MPI_SUM, world);

  // PHASE 3: Communicate boundary cluster IDs
  comm->all();

  // PHASE 4: Build inter-processor connectivity
  for (int i = 0; i < nlocal; i++) {
    applattice->connected_ghosts(i, cluster_ids, clustlist, idoffset);
  }

  // PHASE 5: Master process merges clusters across boundaries
  if (me == 0) {
    // Complex secondary clustering algorithm
    // Merges clusters that span processor boundaries
    // Calculates statistics (volume, centroid, bounding box)
  }

  // PHASE 6: Optional - broadcast cluster sizes to all ranks
  if (sitesizeflag) { /* ... */ }
}
```

**Characteristics:**
- **Highly irregular** memory access (stack-based graph traversal)
- **Serial dependencies** (stack operations, DFS inherently sequential)
- **Dynamic memory allocation** (cluster lists grow during traversal)
- **Multiple MPI phases** (collective operations + point-to-point communication)
- **Complex data structures** (neighbor lists, periodic boundary tracking)
- Complexity: O(nlocal + n_clusters² + MPI_overhead)
- **Already identified as bottleneck** (83% overhead in roadmap Phase 3.5)

---

## 2. Data Access Patterns

### Summary Table

| Diagnostic | Primary Data | Access Pattern | MPI Communication | GPU Suitability |
|------------|--------------|----------------|-------------------|-----------------|
| **diag_array** | `iarray[][]`, `darray[][]` | Sequential read | 1 Allreduce/stat | ✅ Excellent |
| **diag_energy** | `spin[]`, `neighbor[][]` via `site_energy()` | Sequential + neighbor | 1 comm + 1 Allreduce | ✅ Good |
| **diag_cluster** | `spin[]`, `neighbor[][]`, `xyz[][]`, cluster metadata | Irregular graph traversal | Multiple collectives + P2P | ❌ Poor |

### 2.1 DiagArray Data Flow

```
Host (CPU)                          Device (GPU)
┌──────────────┐                    ┌──────────────┐
│ iarray[i][j] │ ─────sync?────────>│ k_iarray(i,j)│
│ darray[i][j] │ ─────sync?────────>│ k_darray(i,j)│
└──────────────┘                    └──────────────┘
       │                                    │
       │                                    V
       │                            ┌──────────────────┐
       │                            │ parallel_reduce  │
       │                            │  - Sum, Min, Max │
       │                            └──────────────────┘
       │                                    │
       │<──────result──────────────────────┘
       │
       V
┌──────────────┐
│ MPI_Allreduce│ (Still on CPU)
└──────────────┘
```

**Key Insight:** Need Kokkos DualViews for `iarray` and `darray` in base Kokkos apps.

### 2.2 DiagEnergy Data Flow

```
Host (CPU)                          Device (GPU)
┌──────────────┐                    ┌──────────────┐
│ spin[]       │ ─already synced───>│ k_spin(i)    │
│ neighbor[][] │ ─already synced───>│ k_neighbor   │
│ numneigh[]   │ ─already synced───>│ k_numneigh   │
└──────────────┘                    └──────────────┘
       │                                    │
       │                                    V
       │                            ┌──────────────────┐
       │                            │ EnergyFunctor    │
       │                            │  parallel_reduce │
       │                            └──────────────────┘
       │                                    │
       │<──────energy_local─────────────────┘
       │
       V
┌──────────────┐
│ MPI_Allreduce│
└──────────────┘
```

**Key Insight:** All necessary data already on device for Kokkos apps (no extra sync needed!).

### 2.3 DiagCluster Data Flow

```
Host (CPU)                          Device (GPU)
┌──────────────┐                    ┌──────────────┐
│ spin[]       │ <──must sync───────│ k_spin(i)    │
│ neighbor[][] │ <──must sync───────│ k_neighbor   │
└──────────────┘                    └──────────────┘
       │
       V
┌──────────────────────────┐
│ PHASE 1: DFS on CPU      │ <─── Stack operations, serial
│  while (stack.size()) {  │
│    int ii = stack.top(); │
│    ...                   │
│  }                       │
└──────────────────────────┘
       │
       V
┌──────────────────────────┐
│ PHASE 2-6: MPI shuffling │ <─── Complex communication
└──────────────────────────┘
```

**Key Insight:** Fundamental algorithm incompatible with GPU - requires full host sync.

---

## 3. Existing Kokkos Patterns

Analysis of `src/KOKKOS/app_potts_kokkos.cpp` reveals established patterns:

### 3.1 DualView Pattern for Host-Device Sync

```cpp
// Declaration
Kokkos::DualView<int*> k_numneigh;      // Both host and device views
Kokkos::DualView<int**> k_neighbor;

// Allocation
k_numneigh = Kokkos::DualView<int*>("numneigh", nall);
k_neighbor = Kokkos::DualView<int**>("neighbor", nall, maxneigh);

// Host modification
auto h_numneigh = k_numneigh.h_view;
for (int i = 0; i < nall; i++)
  h_numneigh(i) = numneigh[i];

// Synchronization
k_numneigh.modify_host();    // Mark host as modified
k_numneigh.sync_device();    // Copy host -> device

// Device access (read-only)
auto d_numneigh = k_numneigh.d_view;
```

### 3.2 Parallel Reduction Pattern

```cpp
struct EnergyFunctor {
  typedef double value_type;

  Kokkos::View<int*> spin, numneigh;
  Kokkos::View<int**> neighbor;

  KOKKOS_INLINE_FUNCTION
  void operator()(const int i, double& energy) const {
    int eng = 0;
    int isite = spin(i);
    for (int j = 0; j < numneigh(i); j++) {
      if (spin(neighbor(i,j)) != isite) eng++;
    }
    energy += static_cast<double>(eng);
  }
};

// Usage:
double total_energy = 0.0;
Kokkos::parallel_reduce("ComputeEnergy", nlocal,
                        EnergyFunctor{d_spin, d_numneigh, d_neighbor},
                        total_energy);
Kokkos::fence();  // Ensure completion
```

### 3.3 Min/Max Reduction Pattern

```cpp
struct MinFunctor {
  typedef double value_type;
  Kokkos::View<double*> data;

  KOKKOS_INLINE_FUNCTION
  void operator()(const int i, double& min_val) const {
    if (data(i) < min_val) min_val = data(i);
  }

  // Initialize reduction variable
  KOKKOS_INLINE_FUNCTION
  void init(double& val) const {
    val = 1.0e30;
  }

  // Join parallel results
  KOKKOS_INLINE_FUNCTION
  void join(volatile double& dst, const volatile double& src) const {
    if (src < dst) dst = src;
  }
};
```

### 3.4 Double-Buffer Pattern (Race Prevention)

```cpp
// From kokkos_double_buffer.h
KokkosDoubleBuffer<int> buffer_spin;

// Allocation (creates two buffers)
buffer_spin.allocate("spin", nall);

// Read from one buffer, write to other (prevents races)
auto d_spin_read = buffer_spin.get_read_buffer().d_view;
auto d_spin_write = buffer_spin.get_write_buffer().d_view;

// Swap buffers after kernel
buffer_spin.swap();
```

---

## 4. Detailed Diagnostic Analysis

### 4.1 DiagEnergy - HIGH PRIORITY ✅

#### Why Implement?

1. **Reuses existing kernels:** Energy computation already implemented in `app_potts_kokkos.cpp`
2. **Frequently called:** Often invoked every timestep for monitoring
3. **Clear performance path:** Direct parallel reduction, no algorithmic changes
4. **No extra sync:** Data already on device for Kokkos apps

#### Implementation Complexity: LOW

- Lines of code: ~100 (excluding header boilerplate)
- Reuses: `EnergyFunctor` pattern from apps
- New code: Minimal (just diagnostic wrapper)

#### Expected Performance Impact

**Baseline (CPU):**
- 100×100×100 system (1M sites): ~0.001-0.01s per call
- Called every step: 1000 calls = 1-10s total

**Kokkos (GPU):**
- Parallel reduction: 5-20× faster
- 1000 calls: 0.05-2s total
- **Savings: 0.95-8s per run (90-95% reduction in energy diagnostic overhead)**

#### Data Dependencies

✅ Already available on device:
- `k_spin` (DualView)
- `k_neighbor` (DualView)
- `k_numneigh` (DualView)

❌ Not needed (diagnostic is read-only)

#### Risk: LOW

- Well-understood pattern
- Minimal code complexity
- Already validated in apps

---

### 4.2 DiagArray - MEDIUM PRIORITY ✅

#### Why Implement?

1. **Eliminates sync overhead:** Keeps array data on device
2. **Enables future diagnostics:** Infrastructure benefits multiple array-based diagnostics
3. **Standard Kokkos pattern:** Simple reductions (sum, min, max, mean)
4. **Scalable benefit:** More valuable as array diagnostics are added

#### Implementation Complexity: MEDIUM

- Lines of code: ~200-300
- **Requires infrastructure:** Add `k_iarray` and `k_darray` DualViews to `AppLatticeKokkos`
- Multiple reduction functors needed (sum, min, max)

#### Expected Performance Impact

**Baseline (CPU):**
- Simple reductions: 0.0001-0.001s per statistic
- Usually minimal overhead

**Kokkos (GPU):**
- 10-50× faster for reductions
- **Primary benefit: Eliminates host-device sync** for apps that use array diagnostics

**Long-term benefit:**
- Unlocks array-based diagnostics without sync penalty
- Compounds as more array diagnostics are added

#### Data Dependencies

❌ Currently missing from Kokkos apps:
- `k_iarray` - needs to be added as `DualView<int**>`
- `k_darray` - needs to be added as `DualView<double**>`

✅ Infrastructure addition enables:
- `diag_array/kk`
- Future array-based diagnostics (potential Phase 5 work)

#### Risk: MEDIUM

- Requires modifying base `AppLatticeKokkos` class
- Must ensure sync strategy doesn't penalize non-diagnostic code
- Multiple reduction types to implement and test

---

### 4.3 DiagCluster - NOT RECOMMENDED ❌

#### Why NOT Implement?

1. **Algorithm incompatible with GPU:**
   - Stack-based depth-first search is inherently serial
   - Irregular memory access causes severe warp divergence
   - Dynamic memory allocation (cluster list growth) problematic on GPU

2. **Graph algorithms perform poorly on GPU:**
   - Literature shows DFS is one of worst GPU candidates
   - Better GPU alternatives (BFS, Union-Find) exist but require complete rewrite

3. **Complexity vs. benefit:**
   - 900+ lines of complex code to port
   - Multiple MPI communication phases
   - **Likely slower on GPU** or minimal speedup

4. **Already expensive, but due to algorithm:**
   - 83% overhead noted in roadmap Phase 3.5
   - Better solved by calling less frequently or algorithmic improvements
   - GPU won't fix fundamental O(n_clusters²) MPI complexity

#### Implementation Complexity: VERY HIGH

- Lines of code: 900+ (massive)
- Requires: Complete algorithm redesign for GPU
- Alternative GPU algorithms: Union-Find, parallel BFS (research project)
- Estimated effort: 4-8 weeks

#### Expected Performance Impact

**Baseline (CPU):**
- 0.1-10s per call (depends on cluster count and MPI ranks)
- Called infrequently (every 10-100 steps)

**Naive Kokkos (GPU):**
- **Likely slower** due to:
  - Host-device sync overhead (must pull all data to CPU for stack operations)
  - Warp divergence if attempting GPU DFS
  - No benefit from GPU in serial phases

**Advanced Kokkos (GPU with algorithm redesign):**
- Union-Find or parallel BFS could achieve 2-5× speedup
- Requires months of research and development
- Not justified unless clustering becomes primary bottleneck

#### Alternative Solutions (RECOMMENDED)

1. **Reduce call frequency:**
   ```
   # Input script
   diag_style cluster stats yes delt 100.0  # Call every 100 time units, not every step
   ```

2. **Keep on CPU, accept sync cost:**
   - Trigger explicit host sync before calling
   - Minimal impact if called rarely

3. **Future research:**
   - Investigate GPU-friendly clustering algorithms (Union-Find, parallel BFS)
   - Separate research project, not part of initial Kokkos-ification

#### Risk: VERY HIGH

- Massive implementation effort
- High probability of negative or minimal performance gain
- Opportunity cost (time better spent on other optimizations)

---

## 5. Recommended Architecture

### 5.1 File Organization

Follow existing SPPARKS Kokkos package pattern:

```
src/KOKKOS/
├── Install.sh                    # (existing) - update to copy diagnostic files
├── README.md                     # (existing) - add diagnostics section
├── app_*_kokkos.h               # (existing) Kokkos apps
├── app_*_kokkos.cpp             # (existing)
├── diag_energy_kokkos.h         # NEW
├── diag_energy_kokkos.cpp       # NEW
├── diag_array_kokkos.h          # NEW
└── diag_array_kokkos.cpp        # NEW
```

**Installation:** Modify `Install.sh` to copy diagnostic files to `src/` when package is activated:

```bash
# In KOKKOS/Install.sh
if (test $1 = 1) then
  # ... existing app file copies ...

  # Add diagnostic files
  cp diag_energy_kokkos.cpp ..
  cp diag_energy_kokkos.h ..
  cp diag_array_kokkos.cpp ..
  cp diag_array_kokkos.h ..
fi
```

### 5.2 Class Hierarchy

**Option A: Separate Kokkos Classes (RECOMMENDED)**

```cpp
// Inheritance hierarchy
Diag (base)
├── DiagEnergy (CPU version)
└── DiagEnergyKokkos (GPU version, inherits from DiagEnergy)

// Registration in style_diag.h
#ifdef DIAG_CLASS
DiagStyle(energy, DiagEnergy)
DiagStyle(energy/kk, DiagEnergyKokkos)  // NEW
#endif
```

**Pros:**
- ✅ Clean separation of CPU and GPU code
- ✅ Follows established SPPARKS app pattern
- ✅ Easy to maintain both versions independently
- ✅ No contamination of base classes with `#ifdef`

**Cons:**
- ⚠️ Code duplication (but diagnostics are small, ~100-300 lines)
- ⚠️ Must maintain two versions (acceptable trade-off)

**Option B: Templated/Conditional (NOT RECOMMENDED)**

```cpp
class DiagEnergy : public Diag {
  void compute() {
#ifdef KOKKOS_ENABLED
    compute_kokkos();
#else
    compute_cpu();
#endif
  }
};
```

**Pros:**
- Single source file

**Cons:**
- ❌ Messy code with `#ifdef` blocks
- ❌ Harder to maintain
- ❌ Not the SPPARKS pattern (apps don't do this)
- ❌ Violates separation of concerns

### 5.3 Naming Convention

Follow SPPARKS Kokkos conventions:

| Component | Naming Pattern | Example |
|-----------|----------------|---------|
| File names | `<base>_kokkos.cpp/h` | `diag_energy_kokkos.cpp` |
| Class names | `<Base>Kokkos` | `DiagEnergyKokkos` |
| Style registration | `<style>/kk` | `DiagStyle(energy/kk, ...)` |
| Input script usage | `diag_style <name>/kk` | `diag_style energy/kk` |

**Example input script:**
```
# Kokkos app with Kokkos diagnostics
app_style potts/kk 100
diag_style energy/kk           # Use Kokkos version
diag_style array/kk d1 mean    # Use Kokkos version

# Regular app with regular diagnostics (still valid)
app_style potts 100
diag_style energy              # Use CPU version
```

### 5.4 Data Residency Strategy

**Principle:** Diagnostics work with data already on device (no extra syncs).

```cpp
class DiagEnergyKokkos : public DiagEnergy {
 public:
  void compute() override {
    // Cast to Kokkos app (ensures compatibility)
    auto app_kk = dynamic_cast<AppLatticeKokkos*>(app);
    if (!app_kk) {
      error->all(FLERR, "diag_style energy/kk requires Kokkos app (e.g., potts/kk)");
    }

    // Access device-resident data (NO SYNC NEEDED)
    auto d_spin = app_kk->buffer_spin.get_read_buffer().d_view;
    auto d_numneigh = app_kk->k_numneigh.d_view;
    auto d_neighbor = app_kk->k_neighbor.d_view;

    // Compute on device
    double energy_local = compute_energy_device(d_spin, d_numneigh, d_neighbor);

    // MPI reduction (still on CPU)
    MPI_Allreduce(&energy_local, &energy, 1, MPI_DOUBLE, MPI_SUM, world);
  }
};
```

**Key points:**
1. **Assume data on device:** Kokkos apps maintain device-resident data
2. **No defensive syncs:** Don't sync "just in case" - trust the app
3. **Runtime checking:** Use `dynamic_cast` to verify Kokkos app compatibility
4. **MPI still on CPU:** Current SPPARKS MPI patterns remain unchanged

### 5.5 Compatibility Strategy

**Challenge:** Diagnostics must work with appropriate apps.

**Solution: Explicit Style Matching (RECOMMENDED)**

Users choose diagnostic style to match app style:

```bash
# Input script validation
if (potts/kk app && energy/kk diag) --> OK
if (potts/kk app && energy diag)    --> OK (falls back to CPU, but inefficient)
if (potts app && energy/kk diag)    --> ERROR (Kokkos diag needs Kokkos app)
```

**Implementation:**
```cpp
void DiagEnergyKokkos::compute() {
  auto app_kk = dynamic_cast<AppLatticeKokkos*>(app);
  if (!app_kk) {
    error->all(FLERR,
      "diag_style energy/kk requires a Kokkos app.\n"
      "Use 'app_style potts/kk' or change to 'diag_style energy'");
  }
  // ... continue with Kokkos computation
}
```

**User experience:**
```
# Bad input:
app_style potts 100
diag_style energy/kk

# Error message:
ERROR: diag_style energy/kk requires a Kokkos app.
       Use 'app_style potts/kk' or change to 'diag_style energy'

# Good input (Option 1 - full GPU):
app_style potts/kk 100
diag_style energy/kk

# Good input (Option 2 - CPU only):
app_style potts 100
diag_style energy
```

---

## 6. Implementation Roadmap

### Phase 1: DiagEnergy Kokkos-ification (1-2 weeks)

**Goal:** Implement `diag_energy/kk` with device-resident energy computation.

#### Week 1: Implementation

**Step 1.1: Create file structure**
```bash
cd /Users/Tron/spparks/src/KOKKOS
touch diag_energy_kokkos.h
touch diag_energy_kokkos.cpp
```

**Step 1.2: Implement header** (`diag_energy_kokkos.h`)
```cpp
#ifdef DIAG_CLASS
DiagStyle(energy/kk, DiagEnergyKokkos)
#else

#ifndef SPK_DIAG_ENERGY_KOKKOS_H
#define SPK_DIAG_ENERGY_KOKKOS_H

#include "diag_energy.h"
#include "kokkos_type.h"

namespace SPPARKS_NS {

class DiagEnergyKokkos : public DiagEnergy {
 public:
  DiagEnergyKokkos(class SPPARKS *, int, char **);
  ~DiagEnergyKokkos() override;

  void compute() override;

 private:
  template<class DeviceType>
  double compute_energy_device();
};

}

#endif
#endif
```

**Step 1.3: Implement energy functor** (`diag_energy_kokkos.cpp`)
```cpp
#include "diag_energy_kokkos.h"
#include "app_lattice_kokkos.h"
#include "error.h"
#include <mpi.h>

using namespace SPPARKS_NS;

/* ---------------------------------------------------------------------- */

DiagEnergyKokkos::DiagEnergyKokkos(SPPARKS *spk, int narg, char **arg) :
  DiagEnergy(spk, narg, arg)
{
}

/* ---------------------------------------------------------------------- */

DiagEnergyKokkos::~DiagEnergyKokkos()
{
}

/* ---------------------------------------------------------------------- */

struct EnergyReductionFunctor {
  typedef double value_type;

  Kokkos::View<int*> spin;
  Kokkos::View<int*> numneigh;
  Kokkos::View<int**> neighbor;
  int nlocal;

  KOKKOS_INLINE_FUNCTION
  void operator()(const int i, double& energy) const {
    if (i >= nlocal) return;  // Skip ghost sites

    int eng = 0;
    int isite = spin(i);
    for (int j = 0; j < numneigh(i); j++) {
      int jsite = neighbor(i, j);
      if (spin(jsite) != isite) eng++;
    }
    energy += static_cast<double>(eng);
  }
};

/* ---------------------------------------------------------------------- */

template<class DeviceType>
double DiagEnergyKokkos::compute_energy_device()
{
  auto app_kk = dynamic_cast<AppLatticeKokkos*>(applattice);
  if (!app_kk) {
    error->all(FLERR,
      "diag_style energy/kk requires a Kokkos app.\n"
      "Use 'app_style potts/kk' or similar, or change to 'diag_style energy'");
  }

  // Get device views (data already on device, no sync!)
  auto d_spin = app_kk->buffer_spin.get_read_buffer().d_view;
  auto d_numneigh = app_kk->k_numneigh.d_view;
  auto d_neighbor = app_kk->k_neighbor.d_view;

  // Setup functor
  EnergyReductionFunctor functor;
  functor.spin = d_spin;
  functor.numneigh = d_numneigh;
  functor.neighbor = d_neighbor;
  functor.nlocal = applattice->nlocal;

  // Parallel reduction on device
  double energy_local = 0.0;
  Kokkos::parallel_reduce("DiagEnergy::compute",
                          applattice->nlocal,
                          functor,
                          energy_local);
  Kokkos::fence();

  return energy_local;
}

/* ---------------------------------------------------------------------- */

void DiagEnergyKokkos::compute()
{
  // Ensure ghost sites are up to date
  // (App is responsible for syncing device data before calling diagnostic)
  applattice->comm->all();

  // Compute local energy on device
  double energy_local = compute_energy_device<SPKDeviceType>();

  // MPI reduction (still on CPU)
  MPI_Allreduce(&energy_local, &energy, 1, MPI_DOUBLE, MPI_SUM, world);
}
```

**Step 1.4: Update Install.sh**
```bash
# Edit KOKKOS/Install.sh
# Add to file list:
#   diag_energy_kokkos.cpp
#   diag_energy_kokkos.h
```

#### Week 2: Testing and Validation

**Step 2.1: Unit testing**
```bash
cd /Users/Tron/spparks/src
make yes-kokkos
make -j12 mac_arm

# Create test input
cd /Users/Tron/spparks/examples/potts
cat > in.potts.energy_kk_test <<EOF
seed 56789
app_style potts/kk 100
dimension 3
lattice sc/26n 1.0
region box block 0 50 0 50 0 50
create_box box
create_sites box
set site range 1 100

diag_style energy/kk stats yes delt 1.0
stats 1.0
run 10.0
EOF

# Run test
mpirun -np 4 ../../src/build/spparks < in.potts.energy_kk_test
```

**Step 2.2: Validation (energy values must match CPU version)**
```bash
# Compare with CPU version
sed 's/potts\/kk/potts/' in.potts.energy_kk_test > in.potts.energy_cpu_test
sed -i '' 's/energy\/kk/energy/' in.potts.energy_cpu_test

mpirun -np 4 ../../src/build/spparks < in.potts.energy_cpu_test > cpu_output.txt
mpirun -np 4 ../../src/build/spparks < in.potts.energy_kk_test > kk_output.txt

# Extract energy columns and compare (should be identical)
grep "^[0-9]" cpu_output.txt | awk '{print $2}' > cpu_energies.txt
grep "^[0-9]" kk_output.txt | awk '{print $2}' > kk_energies.txt
diff cpu_energies.txt kk_energies.txt  # Should be empty (identical)
```

**Step 2.3: Performance benchmarking**
```bash
# Large system test (200x200x200 = 8M sites)
cat > in.potts.energy_kk_bench <<EOF
seed 56789
app_style potts/kk 100
dimension 3
lattice sc/26n 1.0
region box block 0 200 0 200 0 200
create_box box
create_sites box
set site range 1 100

diag_style energy/kk stats yes delt 1.0
stats 1.0
run 100.0
EOF

# Benchmark (record times)
time mpirun -np 6 ../../src/build/spparks < in.potts.energy_kk_bench
# Compare with CPU version (expect 5-20x speedup in energy diagnostic)
```

**Step 2.4: Edge case testing**
- Single MPI rank
- 6+ MPI ranks
- With sectors enabled
- With dumps enabled (ensure no interference)
- Error handling (using energy/kk with non-Kokkos app)

---

### Phase 2: DiagArray Kokkos-ification (2-3 weeks)

**Goal:** Implement `diag_array/kk` with device-resident array reductions.

#### Week 1: Infrastructure

**Step 1.1: Add Kokkos array views to AppLatticeKokkos**

Modify `app_lattice_kokkos.h`:
```cpp
class AppLatticeKokkos : public AppLattice {
 public:
  // ... existing members ...

  // NEW: Kokkos-managed array data
  Kokkos::DualView<int**> k_iarray;    // [ninteger][nall]
  Kokkos::DualView<double**> k_darray; // [ndouble][nall]

  void sync_arrays_to_device();        // Sync iarray/darray to device
  void sync_arrays_to_host();          // Sync back to host (if modified on device)

 protected:
  void allocate_arrays() override;     // Override to allocate Kokkos views
};
```

Implement in `app_lattice_kokkos.cpp`:
```cpp
void AppLatticeKokkos::allocate_arrays() {
  // Call base class version (allocates CPU arrays)
  AppLattice::allocate_arrays();

  // Allocate Kokkos DualViews
  if (ninteger > 0) {
    k_iarray = Kokkos::DualView<int**>("iarray", ninteger, nmax);
  }
  if (ndouble > 0) {
    k_darray = Kokkos::DualView<double**>("darray", ndouble, nmax);
  }
}

void AppLatticeKokkos::sync_arrays_to_device() {
  if (ninteger > 0) {
    auto h_iarray = k_iarray.h_view;
    for (int i = 0; i < ninteger; i++) {
      for (int j = 0; j < nlocal + nghost; j++) {
        h_iarray(i, j) = iarray[i][j];
      }
    }
    k_iarray.modify_host();
    k_iarray.sync_device();
  }

  if (ndouble > 0) {
    auto h_darray = k_darray.h_view;
    for (int i = 0; i < ndouble; i++) {
      for (int j = 0; j < nlocal + nghost; j++) {
        h_darray(i, j) = darray[i][j];
      }
    }
    k_darray.modify_host();
    k_darray.sync_device();
  }
}
```

**Step 1.2: Integrate sync into app workflow**

Decide sync strategy:
- **Option A:** Sync automatically after app updates (conservative)
- **Option B:** Sync on-demand when diagnostic requests it (optimal)
- **Recommendation:** Option B - sync only when `diag_array/kk` is active

#### Week 2-3: DiagArray Implementation

**Step 2.1: Implement reduction functors**

In `diag_array_kokkos.cpp`:
```cpp
// Sum reduction
template<typename ScalarType>
struct ArraySumFunctor {
  typedef ScalarType value_type;
  Kokkos::View<ScalarType*> data;
  int nlocal;

  KOKKOS_INLINE_FUNCTION
  void operator()(const int i, ScalarType& sum) const {
    if (i < nlocal) sum += data(i);
  }
};

// Min reduction
template<typename ScalarType>
struct ArrayMinFunctor {
  typedef ScalarType value_type;
  Kokkos::View<ScalarType*> data;
  int nlocal;

  KOKKOS_INLINE_FUNCTION
  void operator()(const int i, ScalarType& min_val) const {
    if (i < nlocal && data(i) < min_val) min_val = data(i);
  }

  KOKKOS_INLINE_FUNCTION
  void init(ScalarType& val) const {
    val = std::numeric_limits<ScalarType>::max();
  }

  KOKKOS_INLINE_FUNCTION
  void join(volatile ScalarType& dst, const volatile ScalarType& src) const {
    if (src < dst) dst = src;
  }
};

// Max reduction (similar to Min)
template<typename ScalarType>
struct ArrayMaxFunctor {
  typedef ScalarType value_type;
  Kokkos::View<ScalarType*> data;
  int nlocal;

  KOKKOS_INLINE_FUNCTION
  void operator()(const int i, ScalarType& max_val) const {
    if (i < nlocal && data(i) > max_val) max_val = data(i);
  }

  KOKKOS_INLINE_FUNCTION
  void init(ScalarType& val) const {
    val = std::numeric_limits<ScalarType>::lowest();
  }

  KOKKOS_INLINE_FUNCTION
  void join(volatile ScalarType& dst, const volatile ScalarType& src) const {
    if (src > dst) dst = src;
  }
};
```

**Step 2.2: Main compute() implementation**
```cpp
void DiagArrayKokkos::compute() {
  auto app_kk = dynamic_cast<AppLatticeKokkos*>(applattice);
  if (!app_kk) {
    error->all(FLERR, "diag_style array/kk requires a Kokkos app");
  }

  // Ensure arrays are on device
  app_kk->sync_arrays_to_device();

  // Get total site count
  int nlocal = applattice->nlocal;
  int nlocal_all;
  MPI_Allreduce(&nlocal, &nlocal_all, 1, MPI_INT, MPI_SUM, world);

  // Compute each statistic
  for (int i = 0; i < nvals; i++) {
    int idx = index[i];

    if (index_double[i]) {
      // Double array
      auto d_data = Kokkos::subview(app_kk->k_darray.d_view, idx, Kokkos::ALL());
      double local_result = 0.0;
      double global_result = 0.0;

      if (diag_method[i] == SUM || diag_method[i] == MEAN) {
        ArraySumFunctor<double> functor{d_data, nlocal};
        Kokkos::parallel_reduce("DiagArray::Sum", nlocal, functor, local_result);
        Kokkos::fence();
        MPI_Allreduce(&local_result, &global_result, 1, MPI_DOUBLE, MPI_SUM, world);
        vals[i] = (diag_method[i] == MEAN) ? global_result / nlocal_all : global_result;

      } else if (diag_method[i] == MIN) {
        ArrayMinFunctor<double> functor{d_data, nlocal};
        Kokkos::parallel_reduce("DiagArray::Min", nlocal, functor, local_result);
        Kokkos::fence();
        MPI_Allreduce(&local_result, &global_result, 1, MPI_DOUBLE, MPI_MIN, world);
        vals[i] = global_result;

      } else if (diag_method[i] == MAX) {
        ArrayMaxFunctor<double> functor{d_data, nlocal};
        Kokkos::parallel_reduce("DiagArray::Max", nlocal, functor, local_result);
        Kokkos::fence();
        MPI_Allreduce(&local_result, &global_result, 1, MPI_DOUBLE, MPI_MAX, world);
        vals[i] = global_result;
      }

    } else {
      // Integer array (similar logic with int type)
      auto d_data = Kokkos::subview(app_kk->k_iarray.d_view, idx, Kokkos::ALL());
      // ... similar reduction logic ...
    }
  }
}
```

**Step 2.3: Testing**
Similar to DiagEnergy, with additional tests for:
- Integer vs double arrays
- All reduction types (sum, mean, min, max)
- Multiple statistics in single diagnostic

---

### Phase 3: Integration & Documentation (1 week)

**Step 3.1: Update KOKKOS package README**
```markdown
# KOKKOS Package

## Kokkos Apps
- app_potts_kokkos: Potts model with GPU acceleration
- ...

## Kokkos Diagnostics (NEW)
- diag_energy/kk: Energy computation on GPU (5-20x speedup)
- diag_array/kk: Array statistics on GPU (10-50x speedup for reductions)

### Usage
Kokkos diagnostics require Kokkos apps:
```
app_style potts/kk 100
diag_style energy/kk
diag_style array/kk d1 mean d2 sum i1 max
```

**Step 3.2: Update main roadmap**
Add to `docs/KOKKOS_IMPLEMENTATION_ROADMAP.md`:
```markdown
### Phase 4: Diagnostic Kokkos-ification (COMPLETED)
**Goal:** Reduce diagnostic overhead via GPU-accelerated reductions.

**Implemented:**
- diag_energy/kk: Device-resident energy computation
- diag_array/kk: Device-resident array reductions

**Performance gains:**
- 5-20x speedup for energy diagnostics
- 10-50x speedup for array reductions
- Eliminated host-device sync overhead for diagnostics

**Skipped:**
- diag_cluster: Poor GPU fit (graph algorithm)
```

**Step 3.3: Create user documentation**
Create `doc/diag_energy.txt` section:
```
KOKKOS variant:

diag_style energy/kk

Syntax is identical to energy style. Requires a Kokkos-enabled app
(e.g., potts/kk). Computes energy on GPU, reducing diagnostic overhead
by 5-20x for large systems.
```

---

## 7. Performance Expectations

### 7.1 DiagEnergy Benchmarks

**Test System:** 200×200×200 Potts model (8M sites), 100 timesteps

| Configuration | Energy Time | Speedup |
|---------------|-------------|---------|
| CPU (1 rank) | 10.0 s | 1.0× |
| CPU (6 ranks) | 2.5 s | 4.0× |
| Kokkos GPU (1 rank + 6 OpenMP) | 0.5-1.0 s | **10-20×** |
| Kokkos GPU (6 ranks × 2 OpenMP) | 0.2-0.5 s | **20-50×** |

**Assumptions:**
- Energy diagnostic called every timestep (common for monitoring)
- Data already on device (no extra sync overhead)
- GPU: Apple M2 Max or similar

**Critical factor:** Cumulative savings over thousands of timesteps.

### 7.2 DiagArray Benchmarks

**Test System:** Same 8M sites, 100 timesteps, 4 array statistics

| Configuration | Array Diagnostic Time | Speedup |
|---------------|----------------------|---------|
| CPU | 2.0 s | 1.0× |
| Kokkos GPU | 0.04-0.2 s | **10-50×** |

**Note:** Actual speedup depends on:
- Number of statistics computed
- Reduction type (sum faster than min/max on GPU)
- Array access pattern

### 7.3 Overall Impact

**Scenario:** Typical AM simulation with frequent diagnostics
- 10,000 timesteps
- Energy diagnostic every step
- Array diagnostics every 10 steps

**Baseline (CPU):**
- Energy: 10,000 × 0.01s = 100s
- Array: 1,000 × 0.002s = 2s
- **Total diagnostic overhead: 102s**

**With Kokkos:**
- Energy: 10,000 × 0.001s = 10s (90% reduction)
- Array: 1,000 × 0.0001s = 0.1s (95% reduction)
- **Total diagnostic overhead: 10.1s**

**Savings: 92s (90% reduction in diagnostic time)**

For simulations dominated by diagnostic overhead, this is transformative.

---

## 8. Risk Assessment

### 8.1 DiagEnergy - LOW RISK ✅

**Technical risks:**
- ✅ Pattern already validated in apps
- ✅ Minimal code complexity
- ✅ No new infrastructure needed
- ⚠️ Must ensure ghost communication before energy compute

**Mitigation:**
- Reuse existing `EnergyFunctor` from apps (proven correct)
- Extensive validation against CPU version (exact match required)
- Runtime checks for Kokkos app compatibility

**Testing risks:**
- ⚠️ Must test with sectors (different execution path)
- ⚠️ Must test with multiple MPI ranks

**Mitigation:**
- Comprehensive test matrix (1, 4, 6 ranks; with/without sectors)
- Regression tests comparing CPU and GPU outputs

**Timeline risk:** LOW (1-2 weeks is conservative)

---

### 8.2 DiagArray - MEDIUM RISK ⚠️

**Technical risks:**
- ⚠️ **Infrastructure addition:** Modifying `AppLatticeKokkos` base class
  - Risk: Breaking existing apps
  - Mitigation: Thorough testing of all Kokkos apps after changes

- ⚠️ **Sync strategy:** Choosing when to sync arrays to device
  - Risk: Unnecessary syncs slow down apps
  - Mitigation: On-demand sync only when `diag_array/kk` is active

- ⚠️ **Multiple reduction types:** Must implement and test sum, mean, min, max
  - Risk: Subtle bugs in reduction logic
  - Mitigation: Unit tests for each reduction type, compare with CPU

**Performance risks:**
- ⚠️ Array sync overhead may negate GPU speedup for small systems
  - Mitigation: Benchmark and document minimum problem size for benefit

**Timeline risk:** MEDIUM (2-3 weeks, could extend to 4 if infrastructure issues arise)

---

### 8.3 DiagCluster - VERY HIGH RISK ❌

**Algorithmic risks:**
- ❌ **Fundamental incompatibility:** DFS graph traversal is serial
  - GPU parallelization would require complete algorithm rewrite
  - Risk: Months of work with uncertain payoff

**Performance risks:**
- ❌ **Likely slower on GPU:** Irregular memory access, warp divergence
  - Even with heroic optimization, speedup questionable

**Complexity risks:**
- ❌ **900+ lines of complex code:** High bug risk during porting
- ❌ **Multiple MPI phases:** Difficult to debug on GPU

**Opportunity cost:**
- ❌ **4-8 weeks of effort** better spent elsewhere
  - Could implement 3-4 simpler diagnostics instead

**Recommendation:** DO NOT implement. Accept host-device sync cost on rare calls.

---

## 9. Code Examples

### 9.1 Example Input Script

```bash
# examples/potts/in.potts.kokkos_full
# Full Kokkos simulation with GPU-accelerated diagnostics

seed 56789

# KOKKOS APP (required for Kokkos diagnostics)
app_style potts/kk 100
dimension 3
lattice sc/26n 1.0
region box block 0 100 0 100 0 100
create_box box
create_sites box
set site range 1 100

# KOKKOS DIAGNOSTICS
diag_style energy/kk stats yes delt 1.0          # GPU energy
diag_style array/kk d1 mean d2 sum i1 max        # GPU array stats

# Temperature (if using AM app)
temperature 1000.0

# Output
stats 10.0
dump 1 text 50.0 dump.potts site

# Run
solve_style tree
sector yes
run 100.0
```

### 9.2 Example Energy Functor (Full Implementation)

```cpp
/* ----------------------------------------------------------------------
   SPPARKS - Stochastic Parallel PARticle Kinetic Simulator
   http://www.cs.sandia.gov/~sjplimp/spparks.html
   Steve Plimpton, sjplimp@sandia.gov, Sandia National Laboratories

   Copyright (2008) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level SPPARKS directory.
------------------------------------------------------------------------- */

#ifdef DIAG_CLASS
DiagStyle(energy/kk, DiagEnergyKokkos)
#else

#ifndef SPK_DIAG_ENERGY_KOKKOS_H
#define SPK_DIAG_ENERGY_KOKKOS_H

#include "diag_energy.h"
#include "kokkos_type.h"

namespace SPPARKS_NS {

class DiagEnergyKokkos : public DiagEnergy {
 public:
  DiagEnergyKokkos(class SPPARKS *, int, char **);
  ~DiagEnergyKokkos() override;

  void compute() override;

 private:
  template<class DeviceType>
  double compute_energy_device();
};

}

#endif
#endif

/* ERROR/WARNING messages:

E: diag_style energy/kk requires a Kokkos app

You are using a Kokkos diagnostic (energy/kk) with a non-Kokkos app.
Either change to a Kokkos app (e.g., app_style potts/kk) or use the
regular diagnostic (diag_style energy).

*/
```

### 9.3 Example Validation Script

```python
#!/usr/bin/env python3
"""
Validate DiagEnergyKokkos against CPU version.
Ensures GPU and CPU produce identical energy values.
"""

import subprocess
import sys
import numpy as np

def run_spparks(input_script, output_file):
    """Run SPPARKS and capture output."""
    cmd = f"mpirun -np 4 ../../src/build/spparks < {input_script} > {output_file} 2>&1"
    subprocess.run(cmd, shell=True, check=True)

def extract_energies(output_file):
    """Extract energy column from SPPARKS output."""
    energies = []
    with open(output_file, 'r') as f:
        for line in f:
            if line.strip() and line[0].isdigit():
                parts = line.split()
                if len(parts) >= 2:
                    energies.append(float(parts[1]))
    return np.array(energies)

def validate():
    """Main validation routine."""
    print("=== DiagEnergy Validation ===")

    # Run CPU version
    print("Running CPU version...")
    run_spparks("in.potts.energy_cpu", "cpu_output.txt")
    cpu_energies = extract_energies("cpu_output.txt")

    # Run Kokkos version
    print("Running Kokkos version...")
    run_spparks("in.potts.energy_kk", "kk_output.txt")
    kk_energies = extract_energies("kk_output.txt")

    # Compare
    if len(cpu_energies) != len(kk_energies):
        print(f"ERROR: Different number of timesteps!")
        print(f"  CPU: {len(cpu_energies)}, Kokkos: {len(kk_energies)}")
        return False

    # Check for exact match
    if np.allclose(cpu_energies, kk_energies, rtol=0, atol=0):
        print("✅ PASS: Energy values match exactly!")
        return True
    else:
        max_diff = np.max(np.abs(cpu_energies - kk_energies))
        print(f"❌ FAIL: Energies differ by up to {max_diff}")
        print(f"First 10 CPU energies: {cpu_energies[:10]}")
        print(f"First 10 Kokkos energies: {kk_energies[:10]}")
        return False

if __name__ == "__main__":
    success = validate()
    sys.exit(0 if success else 1)
```

---

## 10. References

### 10.1 SPPARKS Documentation

- `docs/KOKKOS_IMPLEMENTATION_ROADMAP.md`: Overall Kokkos strategy
- `src/KOKKOS/README.md`: KOKKOS package documentation
- `src/diag_energy.cpp`: Base CPU implementation
- `src/diag_array.cpp`: Base CPU implementation
- `src/diag_cluster.cpp`: Cluster diagnostic (not recommended for GPU)

### 10.2 Kokkos Documentation

- Kokkos Programming Guide: https://kokkos.github.io/kokkos-core-wiki/
- Parallel Reduction Patterns: https://kokkos.github.io/kokkos-core-wiki/API/core/parallel-dispatch/parallel_reduce.html
- DualView Usage: https://kokkos.github.io/kokkos-core-wiki/API/containers/DualView.html

### 10.3 Related Work

- LAMMPS Kokkos Package: https://docs.lammps.org/Speed_kokkos.html
  - Similar diagnostic acceleration strategies
  - Compute styles with `/kk` suffix convention

- SPARTA Kokkos: https://sparta.github.io/
  - Demonstrates DualView patterns for particle data

### 10.4 Graph Algorithms on GPU (Background Reading)

- "Why Graph Algorithms Fail on GPUs" (2015): https://arxiv.org/abs/1511.04070
  - Explains DFS/BFS challenges on GPU
  - Union-Find as GPU-friendly alternative

- "GPU-Accelerated Connected Component Labeling" (2020)
  - Relevant for future diag_cluster improvements
  - NOT recommended for initial Kokkos-ification

---

## Appendix A: Performance Profiling Commands

### A.1 Baseline CPU Profiling

```bash
# Build with profiling
cd /Users/Tron/spparks/src
make clean-all
make -j12 mac_arm CCFLAGS="-O3 -g"

# Run with profiling
cd ../examples/potts
mpirun -np 1 ../../src/build/spparks < in.potts > output.txt

# Profile with Instruments (macOS)
instruments -t "Time Profiler" mpirun -np 1 ../../src/build/spparks < in.potts
```

### A.2 Kokkos Performance Tuning

```bash
# Enable Kokkos profiling
export KOKKOS_PROFILE_LIBRARY=libkp_kernel_timer.so

# Run with different thread counts
OMP_NUM_THREADS=1 mpirun -np 6 ...   # More MPI, less OpenMP
OMP_NUM_THREADS=6 mpirun -np 1 ...   # Less MPI, more OpenMP

# Optimal (for M2 Max - 12 cores):
OMP_NUM_THREADS=6 mpirun -np 2 ...
```

---

## Appendix B: Decision Matrix

Quick reference for determining whether to Kokkos-ify a diagnostic:

| Question | DiagEnergy | DiagArray | DiagCluster |
|----------|-----------|-----------|-------------|
| Embarrassingly parallel? | ✅ Yes | ✅ Yes | ❌ No (DFS) |
| Data already on device? | ✅ Yes | ⚠️ Partial | ⚠️ Needs sync |
| Simple reduction? | ✅ Yes | ✅ Yes | ❌ Complex graph |
| Frequent calls? | ✅ Every step | ⚠️ Periodic | ❌ Rare |
| GPU-friendly algorithm? | ✅ Yes | ✅ Yes | ❌ No |
| Implementation effort | ✅ Low (1-2 wk) | ⚠️ Med (2-3 wk) | ❌ High (4-8 wk) |
| Expected speedup | ✅ 5-20× | ✅ 10-50× | ❌ <2× or negative |
| **RECOMMENDATION** | ✅ **DO IT** | ✅ **DO IT** | ❌ **SKIP** |

---

## Appendix C: Future Diagnostic Candidates

After completing `diag_energy/kk` and `diag_array/kk`, consider:

### High Priority (Simple Reductions)
- `diag_propensity`: Similar to energy, computes total propensity
- `diag_diffusion`: Monitors diffusion jumps (counters, simple reductions)

### Medium Priority (Moderate Complexity)
- `diag_sinter`: Grain boundary analysis (neighbor checks, similar to energy)

### Low Priority / Research Needed
- `diag_cluster`: Only if GPU-friendly algorithm (Union-Find) is researched

---

## Summary

This analysis demonstrates that **diag_energy and diag_array are excellent candidates for Kokkos-ification**, with clear performance benefits (5-50× speedup), straightforward implementation paths (4-6 weeks total), and low risk. Both follow established Kokkos patterns and integrate naturally with the existing SPPARKS architecture.

**DiagCluster should not be Kokkos-ified** due to fundamental algorithmic incompatibility with GPU execution models. The 900+ lines of complex graph traversal code would require a complete rewrite with uncertain performance benefits.

**Recommended action:** Implement Phase 1 (DiagEnergy) as proof-of-concept, validate performance gains, then proceed with Phase 2 (DiagArray) if Phase 1 is successful.

For questions or feedback, refer to the SPPARKS Kokkos development team or consult `docs/KOKKOS_IMPLEMENTATION_ROADMAP.md`.

---

**Document Version:** 1.0
**Last Updated:** November 2025
**Author:** Analysis performed via Claude Code exploration
**Status:** Research/Planning (Implementation pending approval)
