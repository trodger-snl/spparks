# Kokkos Coloring Implementation Plan (AppLatticeKokkos Base Class)

## Problem Statement

The current Kokkos implementation uses a double-buffer approach where all sites are updated in parallel within a sweep. This causes race conditions when adjacent sites simultaneously read from the same frozen snapshot and both accept moves (especially when ΔE ≤ 0), leading to:

- Spin/quaternion desynchronization
- Noisy, non-physical grain patterns
- Results that differ significantly from CPU versions

## Solution: Restructure Inheritance + 8-Color Checkerboard

1. **Restructure inheritance**: `AppPottsKokkos` inherits from `AppLatticeKokkos` (not `AppPotts`)
2. **Centralize Kokkos logic**: All shared GPU infrastructure lives in `AppLatticeKokkos`
3. **8-color parallelization**: Process sites by color to eliminate race conditions

## New Class Hierarchy

```
AppLattice (CPU base - has coloring infrastructure)
│
├── AppPotts, AppPottsNeigh, AppPottsQuaternion, etc. (CPU apps)
│
└── AppLatticeKokkos (NEW: Kokkos base with shared GPU infrastructure)
    │
    ├── AppPottsKokkos (inherits ONLY from AppLatticeKokkos)
    ├── AppPottsNeighKokkos
    ├── AppPottsNeighOnlyKokkos
    └── AppPottsQuaternionKokkos
        └── AppAdditiveExtTempTextureKokkos
```

**Key change**: Kokkos apps no longer inherit from their CPU counterparts. They inherit from `AppLatticeKokkos` and re-implement the small amount of app-specific logic.

## Why This Works

`AppPotts` is small (~130 lines of logic):
- Constructor flag setup (~10 lines)
- `nspins`, `spin` data members
- `grow_app()` (~3 lines)
- `init_app()` validation (~10 lines)
- `site_energy()` - replaced by Kokkos functor anyway
- `site_event_rejection()` - replaced by Kokkos functor anyway
- KMC methods - not needed for rejection KMC

Re-implementing this in each Kokkos app is trivial compared to the benefit of shared infrastructure.

---

## Phase 1: Extend AppLatticeKokkos as Full Base Class

### File: `src/KOKKOS/app_lattice_kokkos.h`

```cpp
#ifndef SPK_APP_LATTICE_KOKKOS_H
#define SPK_APP_LATTICE_KOKKOS_H

#include "app_lattice.h"
#include <Kokkos_Core.hpp>
#include <Kokkos_DualView.hpp>
#include <Kokkos_Random.hpp>

namespace SPPARKS_NS {

class AppLatticeKokkos : public AppLattice {
public:
  AppLatticeKokkos(class SPPARKS *, int, char **);
  virtual ~AppLatticeKokkos();

  // === Kokkos type aliases ===
  using pool_type = Kokkos::Random_XorShift64_Pool<>;
  using view_int = Kokkos::View<int*, Kokkos::DefaultExecutionSpace::memory_space>;
  using view_double = Kokkos::View<double*, Kokkos::DefaultExecutionSpace::memory_space>;

  // === Shared Kokkos data (all derived classes use these) ===
  Kokkos::DualView<int*> k_numneigh;
  Kokkos::DualView<int**> k_neighbor;
  pool_type *rand_pool;

  // === Color-based site lists for GPU ===
  int kokkos_ncolors;                    // Number of colors (8 for 26-neighbor)
  Kokkos::View<int*> d_color_offsets;    // CSR offsets [ncolors+1]
  Kokkos::View<int*> d_color_sites;      // Site indices by color [nlocal]
  bool color_lists_initialized;

  // === Shared methods ===
  virtual void init_app() override;
  virtual void grow_app() override;
  void setup_color_lists_kokkos();
  virtual void iterate_rejection(double stoptime) override;

  // === Virtual methods derived classes MUST implement ===
  // Sync app-specific data (spin, quaternions, etc.)
  virtual void sync_app_data_to_device() = 0;
  virtual void sync_app_data_to_host() = 0;

  // Allocate app-specific Kokkos views
  virtual void allocate_app_views() = 0;

  // The actual sweep kernel for one color
  virtual void sweep_color_kokkos(int color, int start, int count) = 0;

protected:
  // Shared neighbor sync
  void sync_neighbors_to_device();
};

}
#endif
```

### File: `src/KOKKOS/app_lattice_kokkos.cpp`

```cpp
#include "app_lattice_kokkos.h"
#include "error.h"
#include "timer.h"
#include "output.h"
#include "comm_lattice.h"
#include "domain.h"

using namespace SPPARKS_NS;

enum{NOSWEEP,RANDOM,RASTER,COLOR,COLOR_STRICT};

AppLatticeKokkos::AppLatticeKokkos(SPPARKS *spk, int narg, char **arg) :
  AppLattice(spk, narg, arg), rand_pool(nullptr),
  kokkos_ncolors(0), color_lists_initialized(false)
{
  // Initialize RNG pool with a seed using world rank
  int me;
  MPI_Comm_rank(world, &me);
  rand_pool = new pool_type(12345 + me);
}

AppLatticeKokkos::~AppLatticeKokkos()
{
  if (rand_pool) delete rand_pool;
}

void AppLatticeKokkos::grow_app()
{
  // Call parent
  AppLattice::grow_app();

  int nall = nlocal + nghost;

  // Allocate neighbor arrays
  k_numneigh = Kokkos::DualView<int*>("numneigh", nall);
  if (maxneigh > 0) {
    k_neighbor = Kokkos::DualView<int**>("neighbor", nall, maxneigh);
  }

  // Derived class allocates app-specific views
  allocate_app_views();
}

void AppLatticeKokkos::init_app()
{
  // Call parent initialization (sets up sweepflag, ncolors, sets, etc.)
  AppLattice::init_app();

  // Kokkos apps require coloring for correct parallel GPU execution
  if (sweepflag != NOSWEEP && sweepflag != COLOR && sweepflag != COLOR_STRICT)
    error->all(FLERR,"Kokkos app requires 'sweep color' or 'sweep color/strict' "
               "for correct GPU execution");

  // Sync neighbor arrays to device
  sync_neighbors_to_device();

  // Setup color lists for GPU
  setup_color_lists_kokkos();
}

void AppLatticeKokkos::sync_neighbors_to_device()
{
  int nall = nlocal + nghost;

  auto h_numneigh = k_numneigh.h_view;
  for (int i = 0; i < nall; i++) {
    h_numneigh(i) = numneigh[i];
  }
  k_numneigh.modify_host();
  k_numneigh.sync_device();

  if (maxneigh > 0) {
    auto h_neighbor = k_neighbor.h_view;
    for (int i = 0; i < nall; i++) {
      for (int j = 0; j < numneigh[i]; j++) {
        h_neighbor(i, j) = neighbor[i][j];
      }
    }
    k_neighbor.modify_host();
    k_neighbor.sync_device();
  }
}

void AppLatticeKokkos::setup_color_lists_kokkos()
{
  // Use existing coloring from AppLattice (computed in AppLattice::init_app)

  if (ncolors <= 1) {
    // No coloring - single color with all local sites
    kokkos_ncolors = 1;

    d_color_offsets = Kokkos::View<int*>("color_offsets", 2);
    auto h_offsets = Kokkos::create_mirror_view(d_color_offsets);
    h_offsets(0) = 0;
    h_offsets(1) = nlocal;
    Kokkos::deep_copy(d_color_offsets, h_offsets);

    d_color_sites = Kokkos::View<int*>("color_sites", nlocal);
    auto h_sites = Kokkos::create_mirror_view(d_color_sites);
    for (int i = 0; i < nlocal; i++) {
      h_sites(i) = i;
    }
    Kokkos::deep_copy(d_color_sites, h_sites);

    color_lists_initialized = true;
    return;
  }

  kokkos_ncolors = ncolors;

  // Build CSR offsets from set[] data
  d_color_offsets = Kokkos::View<int*>("color_offsets", ncolors + 1);
  auto h_offsets = Kokkos::create_mirror_view(d_color_offsets);

  h_offsets(0) = 0;
  for (int c = 0; c < ncolors; c++) {
    // When bothflag=0 and ncolors>1, set[c] contains sites of color c
    // When bothflag=1, set[nsector + isector*ncolors + icolor] has color sites per sector
    // For now, handle simple case (no sectors or single sector)
    int set_idx = c;
    h_offsets(c + 1) = h_offsets(c) + set[set_idx].nlocal;
  }
  Kokkos::deep_copy(d_color_offsets, h_offsets);

  // Build flattened site list
  int total_sites = h_offsets(ncolors);
  d_color_sites = Kokkos::View<int*>("color_sites", total_sites);
  auto h_sites = Kokkos::create_mirror_view(d_color_sites);

  int offset = 0;
  for (int c = 0; c < ncolors; c++) {
    int set_idx = c;
    for (int j = 0; j < set[set_idx].nlocal; j++) {
      h_sites(offset++) = set[set_idx].site2i[j];
    }
  }
  Kokkos::deep_copy(d_color_sites, h_sites);

  color_lists_initialized = true;
}

void AppLatticeKokkos::iterate_rejection(double stoptime)
{
  // Ensure color lists are ready
  if (!color_lists_initialized) {
    setup_color_lists_kokkos();
  }

  // Cache color offsets on host for the loop
  auto h_offsets = Kokkos::create_mirror_view(d_color_offsets);
  Kokkos::deep_copy(h_offsets, d_color_offsets);

  int done = 0;
  while (!done) {
    // For each sector (or single pass if no sectors)
    int nset_loop = sectorflag ? nsector : 1;

    for (int iset = 0; iset < nset_loop; iset++) {

      // MPI communication
      if (nprocs > 1) {
        timer->stamp();
        sync_app_data_to_host();  // Virtual - derived class implements
        timer->stamp(TIME_SOLVE);

        timer->stamp();
        if (sectorflag) comm->sector(iset);
        else comm->all();
        timer->stamp(TIME_COMM);
      }

      // Sync to device after MPI
      timer->stamp();
      sync_app_data_to_device();  // Virtual - derived class implements
      timer->stamp(TIME_SOLVE);

      // === CORE: Process each color sequentially ===
      for (int icolor = 0; icolor < kokkos_ncolors; icolor++) {
        int color_start = h_offsets(icolor);
        int color_count = h_offsets(icolor + 1) - color_start;

        if (color_count > 0) {
          // Derived class implements the actual sweep kernel
          sweep_color_kokkos(icolor, color_start, color_count);

          // CRITICAL: Fence before next color to ensure writes are visible
          Kokkos::fence();
        }
      }
      nattempt += nlocal;

      // Sync back for reverse communication
      timer->stamp();
      sync_app_data_to_host();
      timer->stamp(TIME_SOLVE);

      if (nprocs > 1) {
        timer->stamp();
        if (sectorflag) comm->reverse_sector(iset);
        else comm->all_reverse();
        timer->stamp(TIME_COMM);
      }
    }

    // Time advancement
    if (allow_app_update) app_update(dt_rkmc);
    nsweeps++;
    time += dt_rkmc;

    if (time >= stoptime) done = 1;
    if (done || time >= nextoutput) nextoutput = output->compute(time, done);
    timer->stamp(TIME_OUTPUT);
  }
}
```

---

## Phase 2: Implement AppPottsKokkos (Inherits from AppLatticeKokkos)

### File: `src/KOKKOS/app_potts_kokkos.h`

```cpp
#ifdef APP_CLASS
AppStyle(potts/kk,AppPottsKokkos)
#else

#ifndef SPK_APP_POTTS_KOKKOS_H
#define SPK_APP_POTTS_KOKKOS_H

#include "app_lattice_kokkos.h"

namespace SPPARKS_NS {

class AppPottsKokkos : public AppLatticeKokkos {
public:
  AppPottsKokkos(class SPPARKS *, int, char **);
  virtual ~AppPottsKokkos();

  // App-specific data
  int nspins;
  int *spin;                              // Host pointer (from iarray[0])
  Kokkos::View<int*> d_spin;              // Device view

  // Override base class methods
  virtual void grow_app() override;
  virtual void init_app() override;

  // Implement required virtual methods from AppLatticeKokkos
  virtual void sync_app_data_to_device() override;
  virtual void sync_app_data_to_host() override;
  virtual void allocate_app_views() override;
  virtual void sweep_color_kokkos(int color, int start, int count) override;

  // Site methods (for compatibility, though mostly replaced by kernels)
  virtual double site_energy(int) override;
  virtual void site_event_rejection(int, class RandomPark *) override;

private:
  // Sweep functor
  struct ColorSweepFunctor;
};

}
#endif
#endif
```

### File: `src/KOKKOS/app_potts_kokkos.cpp`

```cpp
#include "app_potts_kokkos.h"
#include "random_park.h"
#include "error.h"

using namespace SPPARKS_NS;

AppPottsKokkos::AppPottsKokkos(SPPARKS *spk, int narg, char **arg) :
  AppLatticeKokkos(spk, narg, arg)
{
  // Potts-specific setup (from AppPotts constructor)
  ninteger = 1;
  ndouble = 0;
  delpropensity = 1;
  delevent = 0;
  allow_kmc = 1;
  allow_rejection = 1;
  allow_masking = 1;
  numrandom = 1;

  create_arrays();

  // Parse arguments
  if (narg != 2) error->all(FLERR,"Illegal app_style command");

  nspins = atoi(arg[1]);
  if (nspins <= 0) error->all(FLERR,"Illegal app_style command");
  dt_sweep = 1.0/nspins;
}

AppPottsKokkos::~AppPottsKokkos() {}

void AppPottsKokkos::grow_app()
{
  // Set spin pointer to iarray[0]
  spin = iarray[0];

  // Call parent (allocates neighbors and calls allocate_app_views)
  AppLatticeKokkos::grow_app();
}

void AppPottsKokkos::allocate_app_views()
{
  int nall = nlocal + nghost;
  d_spin = Kokkos::View<int*>("spin", nall);
}

void AppPottsKokkos::init_app()
{
  // Validate spin values
  int flag = 0;
  for (int i = 0; i < nlocal; i++)
    if (spin[i] < 1 || spin[i] > nspins) flag = 1;
  int flagall;
  MPI_Allreduce(&flag, &flagall, 1, MPI_INT, MPI_SUM, world);
  if (flagall) error->all(FLERR,"One or more sites have invalid values");

  // Call parent (validates coloring, sets up color lists, syncs neighbors)
  AppLatticeKokkos::init_app();

  // Initial sync of spin data to device
  sync_app_data_to_device();
}

void AppPottsKokkos::sync_app_data_to_device()
{
  int nall = nlocal + nghost;
  auto h_spin = Kokkos::create_mirror_view(d_spin);
  for (int i = 0; i < nall; i++) {
    h_spin(i) = spin[i];
  }
  Kokkos::deep_copy(d_spin, h_spin);
}

void AppPottsKokkos::sync_app_data_to_host()
{
  auto h_spin = Kokkos::create_mirror_view(d_spin);
  Kokkos::deep_copy(h_spin, d_spin);
  for (int i = 0; i < nlocal; i++) {
    spin[i] = h_spin(i);
  }
}

double AppPottsKokkos::site_energy(int i)
{
  int isite = spin[i];
  int eng = 0;
  for (int j = 0; j < numneigh[i]; j++)
    if (isite != spin[neighbor[i][j]]) eng++;
  return (double) eng;
}

void AppPottsKokkos::site_event_rejection(int i, RandomPark *random)
{
  // Fallback CPU implementation (not used in Kokkos sweeps)
  int oldstate = spin[i];
  double einitial = site_energy(i);

  int iran = (int) (nspins*random->uniform()) + 1;
  if (iran > nspins) iran = nspins;
  spin[i] = iran;
  double efinal = site_energy(i);

  if (efinal <= einitial) {
  } else if (temperature == 0.0) {
    spin[i] = oldstate;
  } else if (random->uniform() > exp((einitial-efinal)*t_inverse)) {
    spin[i] = oldstate;
  }

  if (spin[i] != oldstate) naccept++;
}

// === Kokkos Sweep Functor ===

struct AppPottsKokkos::ColorSweepFunctor {
  Kokkos::View<int*> spin;
  Kokkos::View<int*> numneigh;
  Kokkos::View<int**> neighbor;
  Kokkos::View<int*> color_sites;
  int color_start;
  pool_type rand_pool;
  double temperature;
  double t_inverse;
  int nspins;

  KOKKOS_INLINE_FUNCTION
  void operator()(const int idx, int& accepted) const {
    int i = color_sites(color_start + idx);
    auto rand_gen = rand_pool.get_state();

    int oldstate = spin(i);

    // Compute initial energy (count unlike neighbors)
    int einitial = 0;
    for (int j = 0; j < numneigh(i); j++) {
      if (spin(neighbor(i,j)) != oldstate) einitial++;
    }

    // Pick random neighbor's spin as new state
    int neigh_idx = static_cast<int>(rand_gen.drand() * numneigh(i));
    if (neigh_idx >= numneigh(i)) neigh_idx = numneigh(i) - 1;
    int newstate = spin(neighbor(i, neigh_idx));

    if (newstate == oldstate) {
      rand_pool.free_state(rand_gen);
      return;
    }

    // Compute final energy
    int efinal = 0;
    for (int j = 0; j < numneigh(i); j++) {
      if (spin(neighbor(i,j)) != newstate) efinal++;
    }

    // Metropolis acceptance
    bool accept = false;
    if (efinal <= einitial) {
      accept = true;
    } else if (temperature > 0.0) {
      double prob = Kokkos::exp((einitial - efinal) * t_inverse);
      if (rand_gen.drand() < prob) accept = true;
    }

    if (accept) {
      spin(i) = newstate;  // Safe - no same-color neighbors!
      accepted++;
    }

    rand_pool.free_state(rand_gen);
  }
};

void AppPottsKokkos::sweep_color_kokkos(int color, int start, int count)
{
  ColorSweepFunctor functor{
    d_spin,
    k_numneigh.d_view,
    k_neighbor.d_view,
    d_color_sites,
    start,
    *rand_pool,
    temperature,
    t_inverse,
    nspins
  };

  int naccept_color = 0;
  Kokkos::parallel_reduce("PottsSweep", count, functor, naccept_color);
  naccept += naccept_color;
}
```

---

## Phase 3: Port AppPottsQuaternionKokkos

Same pattern - inherit from `AppLatticeKokkos`, add quaternion-specific data and sweep functor.

```cpp
class AppPottsQuaternionKokkos : public AppLatticeKokkos {
public:
  // Quaternion-specific data
  int nspins;
  int *spin;
  double *q0, *qx, *qy, *qz;

  Kokkos::View<int*> d_spin;
  Kokkos::View<double*> d_q0, d_qx, d_qy, d_qz;
  Kokkos::View<double*> d_symmetries;
  int num_symmetries;
  double theta_cut;

  // Implement virtual methods
  void sync_app_data_to_device() override;
  void sync_app_data_to_host() override;
  void allocate_app_views() override;
  void sweep_color_kokkos(int color, int start, int count) override;
};
```

---

## Mandatory Coloring Enforcement

The coloring check is now in `AppLatticeKokkos::init_app()`, automatically enforced for ALL derived Kokkos apps:

```cpp
void AppLatticeKokkos::init_app()
{
  AppLattice::init_app();

  // Enforced for all Kokkos lattice apps
  if (sweepflag != NOSWEEP && sweepflag != COLOR && sweepflag != COLOR_STRICT)
    error->all(FLERR,"Kokkos app requires 'sweep color' or 'sweep color/strict' "
               "for correct GPU execution");
  // ...
}
```

---

## Summary of Changes

| File | Changes |
|------|---------|
| `app_lattice_kokkos.h` | Full Kokkos base class with coloring, iterate_rejection, virtual methods |
| `app_lattice_kokkos.cpp` | Implement shared iterate_rejection, color list setup, neighbor sync |
| `app_potts_kokkos.h` | Inherit from AppLatticeKokkos, add spin data + sweep functor |
| `app_potts_kokkos.cpp` | Implement virtual methods, sweep_color_kokkos |
| `app_potts_quaternion_kokkos.h/cpp` | Same pattern with quaternion data |
| `app_potts_neigh_kokkos.h/cpp` | Same pattern |
| `app_potts_neighonly_kokkos.h/cpp` | Same pattern |
| `kokkos_double_buffer.h` | Can be removed (no longer needed) |

## Benefits of This Architecture

1. **Single location for shared code**: iterate_rejection, color lists, neighbor sync all in `AppLatticeKokkos`
2. **Mandatory coloring enforced once**: In `AppLatticeKokkos::init_app()`
3. **No double-buffering**: Coloring eliminates races, simplifies data management
4. **Easy to add new Kokkos apps**: Just inherit from `AppLatticeKokkos`, implement 4 virtual methods
5. **Clean separation**: Base class handles GPU infrastructure, derived classes handle physics

## Timeline

| Phase | Description | Effort |
|-------|-------------|--------|
| 1 | Implement AppLatticeKokkos base class | 3-4 hrs |
| 2 | Rewrite AppPottsKokkos | 2-3 hrs |
| 3 | Port AppPottsQuaternionKokkos | 2-3 hrs |
| 4 | Port AppPottsNeighKokkos, AppPottsNeighOnlyKokkos | 2-3 hrs |
| 5 | Update input files, testing | 3-4 hrs |
| 6 | Cleanup (remove old code, double-buffer) | 1-2 hrs |
| **Total** | | **13-19 hours** |
