# Additive Manufacturing Kokkos Implementation Plan

**Status:** Phase 3.3 Complete - Smoothing Implementation Next
**Date Updated:** November 16, 2025
**Target:** AppAdditiveExtTempTextureKokkos extending AppPottsQuaternionKokkos
**Expected Speedup:** 30-70x for full AM simulations
**Progress:**
- Phase 1 (Quaternion Foundation) ✅ Complete
- Phase 2 (Temperature/Gradients) ✅ Substantially Complete (timing issue documented)
- Phase 3.1 (Melting Kernel + RNG) ✅ Complete
- Phase 3.2 (Nucleation + Fast-Forward) ✅ Complete
- Phase 3.3 (Solidification + Texture) ✅ Complete
- Phase 3.4 (Smoothing) 🔄 Next

---

## Executive Summary

This document outlines the plan for adding Kokkos GPU acceleration to `app_additive_ext_temp_texture`, building on the successful `AppPottsQuaternionKokkos` implementation that achieved 20-33x CPU speedup.

**Current Status:**
- ✅ AppPottsQuaternionKokkos fully implemented and validated (Phase 3.5 complete)
- ✅ Double-buffer framework prevents race conditions
- ✅ MPI + sectors support working correctly
- ✅ **Phase 1 Complete:** Advanced quaternion operations implemented (Nov 14, 2025)
- ✅ **Phase 2 Substantially Complete:** Temperature and gradient kernels on device (Nov 16, 2025)
  - ✅ Temperature field loading and device sync
  - ✅ Thermal gradient computation kernel
  - ✅ Data structures for phase transitions
  - ⚠️  Known timing issue documented (see `KOKKOS_TIMING_ISSUE.md`)
- ✅ **Phase 3.1 Complete:** Melting kernel and RNG infrastructure (Nov 16, 2025)
  - ✅ Kokkos Random_XorShift64_Pool for thread-safe device RNG
  - ✅ Per-spin nucleation parameters on device
  - ✅ Melting kernel (T >= liquidus transition)
  - ✅ Random quaternion generation (Marsaglia method)
  - ✅ Double buffering for phase state updates
- 🔄 **Phase 3.2 In Progress:** Nucleation kernel

**Key Challenge:** Converting 450+ lines of thermal gradient computation and texture modeling to device-compatible code while maintaining physics accuracy.

**Estimated Effort:** 3-4 months for full GPU-accelerated AM capability

---

## 1. Current Implementation Status

### 1.1 What's Already Working

**AppPottsQuaternionKokkos** (`src/KOKKOS/app_potts_quaternion_kokkos.{h,cpp}`):

**Existing KOKKOS_INLINE Functions** (lines 121-385):
```cpp
struct QuaternionOps {
  // Basic operations (original)
  KOKKOS_INLINE_FUNCTION
  static void normalize(double* q);

  KOKKOS_INLINE_FUNCTION
  static void composition(const double* a, const double* b, double* result);

  KOKKOS_INLINE_FUNCTION
  static void conjugate(const double* q, double* result);

  KOKKOS_INLINE_FUNCTION
  static double trace_of_rotation_matrix(const double* q);

  KOKKOS_INLINE_FUNCTION
  static double compute_disorientation(/* 24 symmetries */);

  // ✅ Advanced operations for AM applications (Phase 1 - Nov 14, 2025)
  KOKKOS_INLINE_FUNCTION
  static void vector_cross_product(const double* a, const double* b, double* result);

  KOKKOS_INLINE_FUNCTION
  static void rotate_vector(const double* q, const double* r, double* result);

  KOKKOS_INLINE_FUNCTION
  static double get_cosine_angle_q_u(const double* q, const double* u);

  KOKKOS_INLINE_FUNCTION
  static void rotate_q_towards_u(const double* q_in, const double* u,
                                 double angle, double* q_out);

  template<typename RandGenType>
  KOKKOS_INLINE_FUNCTION
  static void generate_random_quaternion(double* q, RandGenType& rand_gen);
};
```

**Functors Implemented:**
- `ComputeEnergyFunctor` (lines 354-406) - Parallel energy calculation
- `RejectionSweepFunctor` (lines 412-596) - Main grain growth kernel
- `SectorSweepFunctor` (lines 768-948) - Sector-based parallel sweep

**Framework Features:**
- `KokkosDoubleBuffer<T>` class - Race-free read/write buffer separation
- Sync profiling infrastructure - User-controllable via `sync_profiling yes`
- Random number generation - `Kokkos::Random_XorShift64_Pool`

**Performance Achievements:**
- 33.5x speedup (single core, small test)
- 20x speedup (production benchmark, 1M sites)
- 22.2x solve speedup (6 MPI × 2 OpenMP)

**Double-Buffer Arrays:**
```cpp
KokkosDoubleBuffer<int> buffer_spin;       // Grain ID
KokkosDoubleBuffer<double> buffer_q0;      // Quaternion w component
KokkosDoubleBuffer<double> buffer_qx;      // Quaternion x component
KokkosDoubleBuffer<double> buffer_qy;      // Quaternion y component
KokkosDoubleBuffer<double> buffer_qz;      // Quaternion z component
```

### 1.2 Inheritance Strategy

**Planned Class Hierarchy:**

```
CPU:
AppPottsQuaternion → AppAdditiveExtTempTexture

KOKKOS:
AppPottsQuaternionKokkos → AppAdditiveExtTempTextureKokkos
```

**Inherited from AppPottsQuaternionKokkos:**
- ✅ Double buffers: buffer_spin, buffer_q0/qx/qy/qz
- ✅ Device quaternion operations (normalize, composition, etc.)
- ✅ Sector sweep patterns + ghost exchange sync
- ✅ RNG pool for parallel random number generation
- ✅ Profiling infrastructure

**Must Add for AM:**
- ❌ Temperature field management (buffer_T, buffer_G, buffer_V)
- ❌ Active flag tracking (buffer_active_flag)
- ❌ Thermal gradient computation on device
- ❌ Texture-weighted grain selection
- ❌ Phase transition kernels (melt, nucleation, solidification)

---

## 2. ✅ Advanced Quaternion Operations (COMPLETED - Phase 1)

### 2.1 Advanced Quaternion Operations - Implementation Details

**Status:** ✅ All functions implemented and validated (November 14, 2025)

These functions have been successfully ported to device-compatible versions:

#### A. rotate_q_towards_u
**Source:** `potts_quaternion/quaternion.h` lines 316-419
**Usage:** Adds thermal gradient-driven misorientation in AM simulations
**Called by:** `apply_misorientation()` in app_additive_ext_temp_texture.cpp:930

```cpp
KOKKOS_INLINE_FUNCTION
static void rotate_q_towards_u(
    const double* q_in,      // Input quaternion
    const double* u,         // Target vector (thermal gradient)
    double angle,            // Rotation angle (radians)
    double* q_out)           // Output quaternion
{
  // Algorithm:
  // 1. Rotate basis vectors e1, e2, e3 by quaternion q_in
  // 2. Find which rotated vector is closest to u
  // 3. Compute rotation axis perpendicular to both
  // 4. Create rotation quaternion around axis by angle
  // 5. Compose with original quaternion

  // Complexity: ~100 lines of device code
  // Dependencies: rotate_vector(), vector_cross_product(), composition()
}
```

**Implementation Notes:**
- Requires 3 calls to `rotate_vector()` (one per basis vector)
- Uses `vector_cross_product()` to find rotation axis
- Must handle parallel vectors (cross product = 0)
- Final quaternion normalization critical for stability

#### B. get_cosine_angle_q_u
**Source:** `crystalline_orientations/quaternion.h` lines 306-360
**Usage:** Computes angle between crystal orientation and thermal gradient
**Called by:** `melt_misorientation()` in app_additive_ext_temp_texture.cpp:1607

```cpp
KOKKOS_INLINE_FUNCTION
static double get_cosine_angle_q_u(
    const double* q,    // Quaternion (orientation)
    const double* u)    // Unit vector (gradient direction)
{
  // Algorithm:
  // 1. Compute rotation matrix R from quaternion q
  // 2. Extract column vectors (rotated basis vectors)
  // 3. Dot each column with u
  // 4. Return maximum absolute value

  // Rotation matrix formula:
  // R[0][0] = 1 - 2*(qy*qy + qz*qz)
  // R[0][1] = 2*(qx*qy - q0*qz)
  // ... (9 components total)

  double max_cos = 0.0;
  // Compute R * [1,0,0], dot with u
  // Compute R * [0,1,0], dot with u
  // Compute R * [0,0,1], dot with u
  // Return max absolute value

  return max_cos;
}
```

**Performance Impact:** Called for every solidification event (~10^6 times per simulation)

#### C. rotate_vector
**Source:** `crystalline_orientations/quaternion.h` lines 103-125
**Usage:** Utility for rotate_q_towards_u

```cpp
KOKKOS_INLINE_FUNCTION
static void rotate_vector(
    const double* q,      // Quaternion (rotation)
    const double* r,      // Input vector
    double* result)       // Output vector
{
  // Formula: r' = r + 2*q0*(q_vec × r) + 2*(q_vec × (q_vec × r))
  // where q = (q0, q_vec) = (q0, qx, qy, qz)

  double q_vec[3] = {q[1], q[2], q[3]};
  double cross1[3], cross2[3];

  vector_cross_product(q_vec, r, cross1);
  vector_cross_product(q_vec, cross1, cross2);

  result[0] = r[0] + 2.0 * q[0] * cross1[0] + 2.0 * cross2[0];
  result[1] = r[1] + 2.0 * q[0] * cross1[1] + 2.0 * cross2[1];
  result[2] = r[2] + 2.0 * q[0] * cross1[2] + 2.0 * cross2[2];
}
```

#### D. vector_cross_product
**Source:** `potts_quaternion/quaternion.h` lines 58-66
**Usage:** Basic utility for many operations

```cpp
KOKKOS_INLINE_FUNCTION
static void vector_cross_product(
    const double* a,
    const double* b,
    double* result)
{
  result[0] = a[1]*b[2] - a[2]*b[1];
  result[1] = -(a[0]*b[2] - a[2]*b[0]);
  result[2] = a[0]*b[1] - a[1]*b[0];
}
```

#### E. generate_random_quaternion
**Source:** `potts_quaternion/quaternion.h` lines 262-304
**Usage:** Nucleation and melting events
**Challenge:** Requires normal distribution RNG

```cpp
KOKKOS_INLINE_FUNCTION
static void generate_random_quaternion(
    double* q,
    RandGenType& rand_gen)
{
  // Box-Muller transform for normal distribution on device
  auto normal_rand = [&]() {
    double u1 = rand_gen.drand();
    double u2 = rand_gen.drand();
    return Kokkos::sqrt(-2.0 * Kokkos::log(u1)) *
           Kokkos::cos(2.0 * M_PI * u2);
  };

  // Generate 4 independent normal variates
  double mag = 0.0;
  for (int i = 0; i < 4; i++) {
    q[i] = normal_rand();
    mag += q[i] * q[i];
  }

  // Normalize to unit quaternion
  mag = Kokkos::sqrt(mag);
  for (int i = 0; i < 4; i++) {
    q[i] /= mag;
  }
}
```

**Note:** Box-Muller transform generates pairs of normal variates - optimize to use both

### 2.2 Implementation Location

**Recommendation:** Extend existing `QuaternionOps` struct in `app_potts_quaternion_kokkos.h`

```cpp
// In src/KOKKOS/app_potts_quaternion_kokkos.h
struct QuaternionOps {
  // Existing functions (normalize, composition, etc.)
  // ...

  // NEW: Advanced operations for AM applications
  KOKKOS_INLINE_FUNCTION
  static void rotate_q_towards_u(/*...*/);

  KOKKOS_INLINE_FUNCTION
  static double get_cosine_angle_q_u(/*...*/);

  KOKKOS_INLINE_FUNCTION
  static void rotate_vector(/*...*/);

  KOKKOS_INLINE_FUNCTION
  static void vector_cross_product(/*...*/);

  KOKKOS_INLINE_FUNCTION
  static void generate_random_quaternion(/*...*/);
};
```

---

## 3. Temperature Field Integration

### 3.1 Required Data Structures

**Add to AppAdditiveExtTempTextureKokkos:**

```cpp
// Read-only arrays (updated from HDF5, not modified during sweeps)
Kokkos::DualView<double*> k_T;                  // Temperature field (from HDF5)
Kokkos::View<double*> d_G;                      // Gradient magnitude (computed once/timestep)
Kokkos::View<double*> d_V;                      // Solidification velocity (computed once/timestep)

// Read-write arrays (modified during parallel sweeps - need double buffering)
KokkosDoubleBuffer<double> buffer_mobility_out; // Arrhenius mobility
KokkosDoubleBuffer<double> buffer_solid_d;      // Solid front distance
KokkosDoubleBuffer<int> buffer_active_flag;     // 0=inactive, 1=active,
                                                 // 2=molten, 3=solidified, 5=void

// Nucleation data (read-only on device)
Kokkos::View<int*> d_nucleation_flags;          // Per-spin nucleation enabled
Kokkos::View<double*> d_nucleation_temps;       // Nucleation temperature
Kokkos::View<double*> d_nucleation_sizes;       // Nucleation particle size
```

**Why DualView for temperature instead of KokkosDoubleBuffer?**
- Temperature is read-only during all device kernels (never modified by parallel sweeps)
- Only updated on CPU from HDF5 source between sweeps
- No race conditions possible since writes happen before device sync
- Uses 50% less memory than double buffering (important for large simulations)
- Matches established pattern for read-only data (k_numneigh, k_neighbor)

**Why simple View for gradients instead of KokkosDoubleBuffer?**
- Gradients are computed once per timestep on device, then read multiple times
- No neighbor-dependent writes during sweeps (unlike spin, quaternions)
- Device-only data (no host-device sync needed after initial computation)
- Simpler structure for pure device-to-device data flow

### 3.2 Synchronization Strategy

**Key Principle:** Temperature updates come from HDF5 (CPU) → must sync to device

**Per-timestep workflow:**

```
1. [CPU] Update temperature from HDF5: T[i] = hdf5_source.get_temp(xyz[i], time)
2. [CPU→GPU] Sync temperature to device: k_T.modify_host(); k_T.sync_device()
3. [GPU] Compute gradients: d_G, d_V (one-time calculation, cached for sweep)
4. [GPU] Phase transition kernel: identify molten sites (T > T_melt)
5. [GPU] Nucleation kernel: create new grains (undercooled liquid)
6. [CPU↔CPU] MPI sector communication:
   a. [GPU→CPU] buffer_*.sync_to_host() (double-buffered arrays only)
   b. [CPU] MPI exchange ghost sites
   c. [CPU→GPU] buffer_*.sync_to_device()
7. [GPU] Grain growth kernel: solidification with texture weighting
8. [GPU] Swap buffers: buffer_*.swap() (double-buffered arrays only)
```

**Key differences from double-buffered arrays:**
- Temperature (`k_T`): DualView synced once per timestep, no swap needed
- Gradients (`d_G`, `d_V`): Device-only, no sync needed after initial computation
- Active flag, spin, quaternions: KokkosDoubleBuffer with sync + swap after MPI

**Note:** Only double-buffered arrays need swap() after sweeps. Read-only arrays like temperature are updated between sweeps and never participate in buffer swapping.

### 3.3 Temperature Source Handling

**Current CPU implementation:**

```cpp
// In app_additive_ext_temp_texture.cpp, app_update():
for (int i = 0; i < nlocal; i++) {
  T[i] = temperature_source->get_temperature_at_time(
      xyz[i][0], xyz[i][1], xyz[i][2], time);
}
```

**Kokkos strategy:**

**Option A: Host update + sync (simpler, recommended for Phase 2)**
```cpp
// Update on CPU (HDF5 library is CPU-only)
auto h_T = k_T.h_view;  // Get host mirror
for (int i = 0; i < nlocal + nghost; i++) {
  h_T(i) = temperature_source->get_temperature_at_time(...);
}
// Mark host as modified and sync to device
k_T.modify_host();
k_T.sync_device();
```

**Option B: Device interpolation (future optimization)**
```cpp
// Pre-load HDF5 data to device memory
Kokkos::View<double***> d_temp_field;  // [time][z][y][x]
// Interpolate on device in parallel
auto d_T = k_T.d_view;  // Get device view
Kokkos::parallel_for(nlocal, KOKKOS_LAMBDA(int i) {
  d_T(i) = interpolate_temperature(xyz(i,0), xyz(i,1), xyz(i,2), time);
});
k_T.modify_device();  // Mark device as modified
```

**Recommendation:** Option A for initial implementation (minimize code changes)

---

## 4. Thermal Gradient Computation (Critical Path)

### 4.1 Current CPU Implementation

**Function:** `normal_finder()` in app_additive_ext_temp_texture.cpp line 1129
**Size:** ~450 lines
**Called by:** Every nucleation and solidification event (~10^6 calls per simulation)
**Performance impact:** Converting this alone could yield 10-20x speedup

**Algorithm:**
1. Identify neighbors in +x, -x, +y, -y, +z, -z directions
2. Weight each neighbor by distance
3. Compute least-squares gradient from weighted temperatures
4. Return gradient vector and magnitude

**Simplified pseudocode:**

```cpp
vector<double> normal_finder(int site) {
  // Count neighbors in each direction
  int n_px = 0, n_mx = 0, n_py = 0, n_my = 0, n_pz = 0, n_mz = 0;

  for (j = 0; j < numneigh[site]; j++) {
    int nbr = neighbor[site][j];
    double dx = xyz[nbr][0] - xyz[site][0];
    double dy = xyz[nbr][1] - xyz[site][1];
    double dz = xyz[nbr][2] - xyz[site][2];

    // Categorize by direction
    if (dx > 0.5) n_px++;
    else if (dx < -0.5) n_mx++;
    // ... similar for y, z
  }

  // Compute weighted gradient
  double grad_x = (avg_temp_px - avg_temp_mx) / dx;
  double grad_y = (avg_temp_py - avg_temp_my) / dy;
  double grad_z = (avg_temp_pz - avg_temp_mz) / dz;

  double magnitude = sqrt(grad_x*grad_x + grad_y*grad_y + grad_z*grad_z);

  return {grad_x, grad_y, grad_z, magnitude};
}
```

### 4.2 Device-Compatible Version

**Add to AppAdditiveExtTempTextureKokkos:**

```cpp
struct ThermalOps {
  KOKKOS_INLINE_FUNCTION
  static void compute_temperature_gradient(
      int i,
      const Kokkos::View<const double*>& d_T,  // DualView device view (read-only)
      const view_int_type& numneigh,
      const Kokkos::View<int**>& neighbor,
      const Kokkos::View<double**>& xyz,
      double* grad_out,       // [nx, ny, nz]
      double* magnitude_out)
  {
    // Cache neighbor temperatures in registers (optimization)
    const int nn = numneigh(i);
    double T_neigh[26];  // Max FCC neighbors
    for (int j = 0; j < nn; j++) {
      T_neigh[j] = d_T(neighbor(i,j));
    }

    // Direction counters
    int count[6] = {0, 0, 0, 0, 0, 0};  // +x,-x,+y,-y,+z,-z
    double sum_T[6] = {0, 0, 0, 0, 0, 0};
    double sum_dist[6] = {0, 0, 0, 0, 0, 0};

    const double xi = xyz(i,0);
    const double yi = xyz(i,1);
    const double zi = xyz(i,2);

    for (int j = 0; j < nn; j++) {
      const int nbr = neighbor(i,j);
      const double dx = xyz(nbr,0) - xi;
      const double dy = xyz(nbr,1) - yi;
      const double dz = xyz(nbr,2) - zi;
      const double dist = Kokkos::sqrt(dx*dx + dy*dy + dz*dz);

      // Categorize by direction (optimized with early continue)
      int dir = -1;
      if (Kokkos::abs(dx) > 0.5) {
        dir = (dx > 0) ? 0 : 1;  // +x : -x
      } else if (Kokkos::abs(dy) > 0.5) {
        dir = (dy > 0) ? 2 : 3;  // +y : -y
      } else if (Kokkos::abs(dz) > 0.5) {
        dir = (dz > 0) ? 4 : 5;  // +z : -z
      } else {
        continue;  // Diagonal neighbor, skip
      }

      count[dir]++;
      sum_T[dir] += T_neigh[j];
      sum_dist[dir] += dist;
    }

    // Compute average temperatures and distances
    double avg_T[6], avg_dist[6];
    for (int d = 0; d < 6; d++) {
      avg_T[d] = (count[d] > 0) ? sum_T[d] / count[d] : 0.0;
      avg_dist[d] = (count[d] > 0) ? sum_dist[d] / count[d] : 1.0;
    }

    // Gradient components
    grad_out[0] = (avg_T[0] - avg_T[1]) / (avg_dist[0] + avg_dist[1]);  // x
    grad_out[1] = (avg_T[2] - avg_T[3]) / (avg_dist[2] + avg_dist[3]);  // y
    grad_out[2] = (avg_T[4] - avg_T[5]) / (avg_dist[4] + avg_dist[5]);  // z

    *magnitude_out = Kokkos::sqrt(
        grad_out[0]*grad_out[0] +
        grad_out[1]*grad_out[1] +
        grad_out[2]*grad_out[2]);
  }
};
```

**Optimization notes:**
- Cache neighbor temperatures in registers before loop
- Use early `continue` to skip diagonal neighbors
- Avoid repeated array accesses (save xyz values in locals)
- Use `Kokkos::abs`, `Kokkos::sqrt` for device math

### 4.3 Usage in Kernels

```cpp
// In NucleationFunctor or SolidificationFunctor
struct SomeFunctor {
  Kokkos::View<const double*> d_T;  // From k_T.d_view
  // ... other views

  KOKKOS_INLINE_FUNCTION
  void operator()(const int i) const {
    double gradient[3], magnitude;

    ThermalOps::compute_temperature_gradient(
        i, d_T, numneigh, neighbor, xyz,
        gradient, &magnitude);

    // Use gradient for texture-weighted selection
    double texture_weight = melt_misorientation_device(
        i, gradient[0], gradient[1], gradient[2]);

    // ...
  }
};

// Launch example:
auto d_T = k_T.d_view;  // Get device view from DualView
SomeFunctor functor;
functor.d_T = d_T;
// ... set other views
Kokkos::parallel_for("SomeFunctor", nlocal, functor);
```

---

## 5. Texture Model Integration

### 5.1 Melt Misorientation Function

**Current CPU implementation:** Line 1592 of app_additive_ext_temp_texture.cpp

```cpp
double AppAdditiveExtTempTexture::melt_misorientation(int site,
                                                       double c1,
                                                       double c2,
                                                       double c3) {
  // Get thermal gradient
  vector<double> normal_vector = normal_finder(site);
  double grad[3] = {normal_vector[0], normal_vector[1], normal_vector[2]};

  // Get quaternion at site
  double q_site[4] = {q0[site], qx[site], qy[site], qz[site]};

  // Compute angle between crystal orientation and gradient
  double cosine_theta = quaternion::get_cosine_of_minumum_angle_between_q_and_u(
      q_site, grad);

  // Mobility factor based on texture model
  double mobility = c1 * exp(-c2 / (cosine_theta + c3));

  return mobility;
}
```

### 5.2 Device-Compatible Version

```cpp
struct TextureOps {
  KOKKOS_INLINE_FUNCTION
  static double melt_misorientation_device(
      int i,
      const view_double_type& q0_read,
      const view_double_type& qx_read,
      const view_double_type& qy_read,
      const view_double_type& qz_read,
      const Kokkos::View<const double*>& d_T,  // DualView device view
      const view_int_type& numneigh,
      const Kokkos::View<int**>& neighbor,
      const Kokkos::View<double**>& xyz,
      double c1, double c2, double c3)
  {
    // Compute thermal gradient
    double gradient[3], magnitude;
    ThermalOps::compute_temperature_gradient(
        i, d_T, numneigh, neighbor, xyz,
        gradient, &magnitude);

    // Normalize gradient
    if (magnitude > 1e-10) {
      gradient[0] /= magnitude;
      gradient[1] /= magnitude;
      gradient[2] /= magnitude;
    } else {
      return 1.0;  // No gradient, no texture effect
    }

    // Get quaternion at site
    double q_site[4] = {q0_read(i), qx_read(i), qy_read(i), qz_read(i)};

    // Compute angle between orientation and gradient
    double cosine_theta = QuaternionOps::get_cosine_angle_q_u(
        q_site, gradient);

    // Texture model: mobility ~ exp(-c2 / (cos(theta) + c3))
    double mobility = c1 * Kokkos::exp(-c2 / (cosine_theta + c3));

    return mobility;
  }
};
```

### 5.3 Texture-Weighted Neighbor Selection

**Current CPU approach:** Compute probabilities, weighted random selection

**Device approach:** Parallel reduction to find weighted cumulative distribution

```cpp
KOKKOS_INLINE_FUNCTION
int select_texture_weighted_neighbor(
    int i,
    const view_double_type& q0_read,
    const view_double_type& qx_read,
    const view_double_type& qy_read,
    const view_double_type& qz_read,
    const Kokkos::View<const double*>& d_T,  // DualView device view
    const view_int_type& numneigh,
    const Kokkos::View<int**>& neighbor,
    const Kokkos::View<double**>& xyz,
    double c1, double c2, double c3,
    RandGenType& rand_gen)
{
  const int nn = numneigh(i);

  // Compute mobility for each unique neighbor spin
  double mobilities[26];  // Max neighbors
  int unique_neighbors[26];
  int n_unique = 0;

  // Find unique spins (similar to existing rejection kernel)
  for (int j = 0; j < nn; j++) {
    int nbr = neighbor(i,j);
    int nbr_spin = spin_read(nbr);

    // Check if already in list
    bool found = false;
    for (int k = 0; k < n_unique; k++) {
      if (unique_neighbors[k] == nbr) {
        found = true;
        break;
      }
    }

    if (!found) {
      unique_neighbors[n_unique] = nbr;

      // Compute texture weight for this neighbor
      double q_nbr[4] = {q0_read(nbr), qx_read(nbr),
                         qy_read(nbr), qz_read(nbr)};

      // Get gradient at neighbor site
      double grad_nbr[3], mag_nbr;
      ThermalOps::compute_temperature_gradient(
          nbr, d_T, numneigh, neighbor, xyz,
          grad_nbr, &mag_nbr);

      // Normalize
      if (mag_nbr > 1e-10) {
        grad_nbr[0] /= mag_nbr;
        grad_nbr[1] /= mag_nbr;
        grad_nbr[2] /= mag_nbr;
      }

      // Compute mobility
      double cos_theta = QuaternionOps::get_cosine_angle_q_u(
          q_nbr, grad_nbr);
      mobilities[n_unique] = c1 * Kokkos::exp(-c2 / (cos_theta + c3));

      n_unique++;
    }
  }

  // Weighted random selection
  double total_mobility = 0.0;
  for (int k = 0; k < n_unique; k++) {
    total_mobility += mobilities[k];
  }

  double rand_val = rand_gen.drand() * total_mobility;
  double cumsum = 0.0;
  for (int k = 0; k < n_unique; k++) {
    cumsum += mobilities[k];
    if (rand_val <= cumsum) {
      return unique_neighbors[k];
    }
  }

  // Fallback (should never reach here)
  return unique_neighbors[n_unique - 1];
}
```

**Note:** This is computationally expensive (gradient calculation per neighbor). Consider caching gradients in buffer_G.

---

## 6. Phase Transition Kernels

### 6.1 Mushy Phase Functor (Liquid → Solid)

**Purpose:** Handle phase transitions in undercooled liquid

**Current CPU implementation:** `mushy_phase()` line 801 of app_additive_ext_temp_texture.cpp

```cpp
struct MushyPhaseFunctor {
  // Views (read-only)
  Kokkos::View<const double*> d_T;  // Temperature from DualView
  view_int_type active_flag_read;
  view_int_type spin_read;

  // Views (write buffers)
  view_int_type active_flag_write;
  view_int_type spin_write;
  view_double_type q0_write, qx_write, qy_write, qz_write;

  // Parameters
  double T_melt;
  double T_nucleation;
  int nspins;

  // Nucleation data
  Kokkos::View<int*> d_nucleation_flags;
  Kokkos::View<double*> d_nucleation_temps;

  // RNG
  rand_pool_type rand_pool;

  KOKKOS_INLINE_FUNCTION
  void operator()(const int i) const {
    // Skip non-molten sites
    if (active_flag_read(i) != 2) {
      // Copy data unchanged
      active_flag_write(i) = active_flag_read(i);
      spin_write(i) = spin_read(i);
      q0_write(i) = q0_read(i);
      qx_write(i) = qx_read(i);
      qy_write(i) = qy_read(i);
      qz_write(i) = qz_read(i);
      return;
    }

    double T_site = d_T(i);

    // Check if solidified
    if (T_site < T_melt) {
      int old_spin = spin_read(i);

      // Check if nucleation is enabled for this spin
      if (old_spin < nspins && d_nucleation_flags(old_spin) == 1) {
        double T_nuc = d_nucleation_temps(old_spin);

        // Check undercooling threshold
        if (T_site < T_nuc) {
          // Nucleate new grain
          auto rand_gen = rand_pool.get_state();

          // Generate random quaternion
          double q_new[4];
          QuaternionOps::generate_random_quaternion(q_new, rand_gen);

          // Assign new spin and quaternion
          spin_write(i) = old_spin;  // Keep same spin ID
          q0_write(i) = q_new[0];
          qx_write(i) = q_new[1];
          qy_write(i) = q_new[2];
          qz_write(i) = q_new[3];
          active_flag_write(i) = 3;  // Solidified

          rand_pool.free_state(rand_gen);
          return;
        }
      }
    }

    // No change, copy data
    active_flag_write(i) = active_flag_read(i);
    spin_write(i) = spin_read(i);
    q0_write(i) = q0_read(i);
    qx_write(i) = qx_read(i);
    qy_write(i) = qy_read(i);
    qz_write(i) = qz_read(i);
  }
};
```

**Launch:**

```cpp
void AppAdditiveExtTempTextureKokkos::mushy_phase_kernel() {
  MushyPhaseFunctor functor;
  functor.d_T = k_T.d_view;  // Get device view from DualView (read-only, no buffer)
  functor.active_flag_read = buffer_active_flag.get_read_view();
  functor.spin_read = buffer_spin.get_read_view();
  // ... set write views, parameters

  Kokkos::parallel_for("MushyPhase", nlocal, functor);
  Kokkos::fence();

  // Swap buffers (only for double-buffered arrays, NOT temperature)
  buffer_active_flag.swap();
  buffer_spin.swap();
  buffer_q0.swap();
  buffer_qx.swap();
  buffer_qy.swap();
  buffer_qz.swap();
  // Note: k_T does NOT need swap() - it's a DualView, not a double buffer
}
```

### 6.2 Solidification Functor (Epitaxial Growth)

**Purpose:** Solidify liquid sites from neighboring solid grains with texture weighting

**Current CPU implementation:** `site_event_solidification()` line 849

```cpp
struct SolidificationFunctor {
  // Views (read-only)
  Kokkos::View<const double*> d_T;  // Temperature from DualView
  // ... other views similar to MushyPhaseFunctor

  // Texture parameters
  double texture_c1, texture_c2, texture_c3;

  KOKKOS_INLINE_FUNCTION
  void operator()(const int i) const {
    // Skip if not in mushy zone
    if (active_flag_read(i) != 2) {
      // Copy unchanged
      return;
    }

    // Check if temperature is in solidification range
    double T_site = d_T(i);
    if (T_site >= T_melt || T_site < T_lower) {
      return;
    }

    // Find solid neighbors
    int solid_neighbors[26];
    int n_solid = 0;

    for (int j = 0; j < numneigh(i); j++) {
      int nbr = neighbor(i,j);
      if (active_flag_read(nbr) == 3) {  // Solidified
        solid_neighbors[n_solid++] = nbr;
      }
    }

    if (n_solid == 0) {
      return;  // No solid neighbors, can't grow
    }

    // Select neighbor with texture weighting
    auto rand_gen = rand_pool.get_state();

    int selected = select_texture_weighted_neighbor(
        i, q0_read, qx_read, qy_read, qz_read, d_T,
        numneigh, neighbor, xyz,
        texture_c1, texture_c2, texture_c3,
        rand_gen);

    // Copy spin and quaternion from selected neighbor
    spin_write(i) = spin_read(selected);
    q0_write(i) = q0_read(selected);
    qx_write(i) = qx_read(selected);
    qy_write(i) = qy_read(selected);
    qz_write(i) = qz_read(selected);
    active_flag_write(i) = 3;  // Now solidified

    rand_pool.free_state(rand_gen);
  }
};
```

### 6.3 Nucleation Particle Flipper

**Challenge:** Multi-site update (flip all sites within radius)

**Current CPU approach:** Loop over neighbors, flip sequentially

**Device approach:** Two-pass algorithm

**Pass 1: Mark sites for nucleation**
```cpp
struct NucleationMarkFunctor {
  Kokkos::View<int*> nucleation_list;  // Sites to nucleate
  Kokkos::View<int> nucleation_count;   // Atomic counter

  KOKKOS_INLINE_FUNCTION
  void operator()(const int i) const {
    // Check if eligible for nucleation
    if (should_nucleate(i)) {
      int idx = Kokkos::atomic_fetch_add(&nucleation_count(), 1);
      nucleation_list(idx) = i;
    }
  }
};
```

**Pass 2: Flip neighbors (parallel over nucleation sites)**
```cpp
struct NucleationFlipFunctor {
  Kokkos::View<int*> nucleation_list;
  int n_nucleation;
  int particle_radius;

  KOKKOS_INLINE_FUNCTION
  void operator()(const int idx) const {
    int center_site = nucleation_list(idx);

    // Flip all neighbors within radius
    // This is tricky - may require atomic operations
    // or accept race conditions (last write wins)

    // ... implementation
  }
};
```

**Alternative:** Keep nucleation on CPU initially (less frequent operation)

---

## 7. Performance Optimization Strategies

### 7.1 Memory Layout (Current: Optimal ✅)

**Structure-of-Arrays (SoA) vs Array-of-Structures (AoS):**

**Current (SoA):**
```cpp
buffer_q0, buffer_qx, buffer_qy, buffer_qz  // Separate arrays
```

**Alternative (AoS):**
```cpp
Kokkos::View<double[4]*> quaternions;  // [site][component]
```

**Recommendation:** Keep SoA layout
- Better GPU memory coalescing (threads access same component)
- Each kernel typically operates on one component at a time
- Consistent with existing LAMMPS/SPARTA Kokkos implementations

### 7.2 Neighbor Access Optimization

**Current:** 2D array `k_neighbor(nall, maxneigh)`

**Pros:**
- Simple indexing: `neighbor(i,j)`
- Fixed stride for coalescing

**Cons:**
- Wastes memory if maxneigh varies widely
- Fixed size limits very connected sites

**Alternative:** Compressed Sparse Row (CSR)

```cpp
Kokkos::View<int*> neighbor_offsets;  // Size: nall+1
Kokkos::View<int*> neighbor_list;     // Size: total_edges

// Access: for (int j = neighbor_offsets(i); j < neighbor_offsets(i+1); j++)
int nbr = neighbor_list(j);
```

**Recommendation:** Keep current layout for Phase 2, consider CSR in Phase 5 if memory is limiting

### 7.3 Gradient Caching

**Problem:** `compute_temperature_gradient()` called multiple times per site per timestep

**Solution:** Cache results in device Views (d_G, d_grad_x, d_grad_y, d_grad_z)

```cpp
// Declare device-only gradient caches
Kokkos::View<double*> d_G("gradient_magnitude", nall);
Kokkos::View<double*> d_grad_x("gradient_x", nall);
Kokkos::View<double*> d_grad_y("gradient_y", nall);
Kokkos::View<double*> d_grad_z("gradient_z", nall);

// In app_update(), before kernels:
auto d_T = k_T.d_view;  // Get temperature from DualView
Kokkos::parallel_for("ComputeGradients", nlocal,
  KOKKOS_LAMBDA(int i) {
    double gradient[3], magnitude;
    ThermalOps::compute_temperature_gradient(
        i, d_T, numneigh, neighbor, xyz,
        gradient, &magnitude);

    // Store for later use (device-only, no sync needed)
    d_G(i) = magnitude;
    d_grad_x(i) = gradient[0];
    d_grad_y(i) = gradient[1];
    d_grad_z(i) = gradient[2];
  });
Kokkos::fence();  // Ensure gradients computed before next kernel
```

Then in solidification kernel:
```cpp
// Just read cached gradient (no recomputation)
double grad[3] = {d_grad_x(i), d_grad_y(i), d_grad_z(i)};
double magnitude = d_G(i);
```

**Key advantage over double buffering:**
- No swap() needed (device-only data, computed once per timestep)
- No MPI sync needed (local computation, not exchanged)
- Uses 50% less memory than KokkosDoubleBuffer

**Expected impact:** 5-10x speedup for texture calculations

### 7.4 Unique Neighbor Selection

**Current issue:** Fixed stack arrays limit maxneigh

```cpp
int unique[100];  // Hard limit
```

**Solution 1: Kokkos scratch space**

```cpp
// Declare scratch space size
size_t scratch_size = maxneigh * sizeof(int) * 2;

Kokkos::parallel_for(
    "Rejection",
    Kokkos::TeamPolicy<>(nlocal, Kokkos::AUTO)
        .set_scratch_size(0, Kokkos::PerTeam(scratch_size)),
    KOKKOS_LAMBDA(const member_type& team) {
      int i = team.league_rank();

      // Get scratch space
      int* unique = (int*)team.team_shmem().get_shmem(maxneigh * sizeof(int));
      int* unique_neigh = unique + maxneigh;

      // ... rest of kernel
    });
```

**Solution 2: Pre-allocate per-thread storage**

```cpp
Kokkos::View<int**> unique_storage("unique", nlocal, maxneigh);

Kokkos::parallel_for("Rejection", nlocal,
  KOKKOS_LAMBDA(int i) {
    // Use unique_storage(i, 0..maxneigh-1)
  });
```

**Recommendation:** Scratch space for optimal memory usage

### 7.5 Random Number Generation

**Current:** Kokkos::Random_XorShift64_Pool works well

**Additions needed:**

1. **Normal distribution** for quaternion generation (Box-Muller)
2. **Weighted selection** for texture model

Both can be implemented with existing `drand()` uniform generator.

### 7.6 Kernel Launch Configuration

**Auto-tuning parameters:**

```cpp
// Team size for scratch space kernels
int team_size = Kokkos::AUTO;  // Let Kokkos decide

// Vector length (GPU: warp size)
int vector_length = Kokkos::AUTO;

// League size (number of teams)
int league_size = (nlocal + team_size - 1) / team_size;
```

**Manual tuning for production:**

```cpp
// CUDA: warp size = 32, optimal team size = 256 or 512
int team_size = 256;

// HIP (AMD): wavefront = 64
int team_size = 256;
```

**Recommendation:** Use AUTO initially, profile and tune in Phase 5

---

## 8. Implementation Roadmap

### Phase 1: Quaternion Foundation ✅ COMPLETE (November 14, 2025)

**Goal:** Add missing device-compatible quaternion operations

**Tasks:**
1. ✅ Review existing QuaternionOps struct
2. ✅ Implement `vector_cross_product()` (simple, test first)
3. ✅ Implement `rotate_vector()` (depends on cross product)
4. ✅ Implement `get_cosine_angle_q_u()` (rotation matrix calculation)
5. ✅ Implement `rotate_q_towards_u()` (complex, depends on above)
6. ✅ Implement `generate_random_quaternion()` (Box-Muller transform)

**Testing:**
- ✅ Physics validation with in.potts_quaternion.compare.kk
- ✅ Energy decreases correctly (103,853 → 20,674 = 80% reduction)
- ✅ Grain growth confirmed (clusters: 7,011 → 13; avg size: 1.14 → 615.4)
- ✅ No compilation errors, device functions working

**Deliverables:**
- ✅ Extended QuaternionOps struct in `app_potts_quaternion_kokkos.h` (+181 lines)
- ✅ Commit a0f59ef: "Add advanced quaternion operations for AM texture modeling"

**Success criteria:**
- ✅ All functions implemented as KOKKOS_INLINE_FUNCTION
- ✅ Physics validation passed
- ✅ Build successful with CMake + Kokkos package

---

### Phase 2: Temperature Integration (Weeks 3-5)

**Goal:** Enable temperature field on device

**Tasks:**
1. ⬜ Add data structures:
   - Read-only: `k_T` (DualView), `d_G`, `d_V` (device Views)
   - Read-write: `buffer_mobility_out`, `buffer_solid_d`, `buffer_active_flag` (double buffers)

2. ⬜ Update `grow_app()` to allocate arrays

3. ⬜ Implement `ThermalOps::compute_temperature_gradient()`
   - Device-compatible version of `normal_finder()`
   - Optimize with neighbor temperature caching
   - Test against CPU version

4. ⬜ Modify `app_update()`:
   - Update temperature on CPU (HDF5)
   - Sync `k_T` to device (modify_host() + sync_device())
   - Compute gradients on device, cache in d_G/d_grad_x/d_grad_y/d_grad_z
   - Update MPI ghost exchange to include double-buffered arrays

**Testing:**
- Small test case: 10x10x10 voxels, simple temperature field
- Verify gradient calculation matches CPU (tolerance: 1e-6)
- Check ghost exchange correctness (MPI test)

**Deliverables:**
- Modified `app_additive_ext_temp_texture_kokkos.{h,cpp}` (create these files)
- Test input: `in.additive_temp_test`
- Validation script: `validate_temperature_sync.py`

**Success criteria:**
- Temperature gradient accuracy within 0.1%
- MPI correctness test passes

---

### Phase 3: AM Functors (Weeks 6-11)

**Goal:** Implement phase transition kernels

#### Substep 3A: Mushy Phase (Weeks 6-7)

**Tasks:**
1. ⬜ Copy nucleation data to device:
   ```cpp
   d_nucleation_flags = Kokkos::create_mirror_copy(
       Kokkos::DefaultExecutionSpace(), h_nucleation_flags);
   ```

2. ⬜ Implement `MushyPhaseFunctor`
   - Check active_flag == 2 (molten)
   - Check T < T_nucleation
   - Generate random quaternion
   - Update spin, quaternion, active_flag

3. ⬜ Launch kernel in `app_update()`

**Testing:**
- Test case: single molten site, verify nucleation
- Check random quaternion is normalized
- Verify active_flag transition 2→3

#### Substep 3B: Texture Model (Weeks 8-9)

**Tasks:**
1. ⬜ Implement `TextureOps::melt_misorientation_device()`
   - Call `compute_temperature_gradient()`
   - Call `get_cosine_angle_q_u()`
   - Compute mobility factor

2. ⬜ Implement `select_texture_weighted_neighbor()`
   - Find unique neighbor spins
   - Compute mobility for each
   - Weighted random selection

**Testing:**
- Known orientation + gradient → verify mobility calculation
- Compare weighted selection distribution vs CPU
- Statistical test (1000 trials, chi-square test)

#### Substep 3C: Solidification (Weeks 10-11)

**Tasks:**
1. ⬜ Implement `SolidificationFunctor`
   - Check active_flag == 2, T in mushy zone
   - Find solid neighbors
   - Select with texture weighting
   - Copy spin and quaternion

2. ⬜ Integrate with sector sweeps
   - Combine with existing grain growth kernel?
   - Or separate pass?

3. ⬜ Apply misorientation (if needed)
   - Call `rotate_q_towards_u()` for gradient alignment

**Testing:**
- Test case: liquid site with 1 solid neighbor → verify copying
- Test case: liquid site with multiple solid neighbors → verify weighting
- Large-scale test: 100x100x100 voxels, full solidification

**Deliverables:**
- Complete `AppAdditiveExtTempTextureKokkos` class
- Three functors: Mushy, Solidification, Nucleation
- Test suite: `test_am_kernels.cpp`

**Success criteria:**
- Physics matches CPU version (grain structure comparison)
- No race conditions (repeat runs give identical results)
- 10x speedup vs CPU on test problem

---

### Phase 4: Nucleation & Advanced (Weeks 12-14)

**Goal:** Complete nucleation logic

**Tasks:**
1. ⬜ Decide on nucleation strategy:
   - Option A: Keep on CPU (simple, infrequent operation)
   - Option B: Two-pass device kernel (mark + flip)
   - Option C: Atomic operations (race conditions?)

2. ⬜ Implement chosen approach

3. ⬜ Void generation (CPU initialization, device filtering)

4. ⬜ Solid front distance calculation (if needed for kinetics)

**Testing:**
- Nucleation event test: verify particle size, shape
- Multi-site test: check no conflicting updates
- Full AM simulation: 1000 layers

**Deliverables:**
- Complete AM application
- User documentation: `AM_KOKKOS_USAGE.md`

**Success criteria:**
- Nucleation events match CPU frequency and distribution
- 20-50x speedup vs CPU on full AM simulation

---

### Phase 5: Optimization & Validation (Weeks 15-18)

**Goal:** Performance tuning and validation

**Tasks:**
1. ⬜ Profiling:
   - Identify kernel bottlenecks (Nsight, rocprof)
   - Measure memory bandwidth utilization
   - Find host-device sync overhead

2. ⬜ Optimization:
   - Gradient caching (buffer_G)
   - Tune kernel launch parameters
   - Minimize synchronization points
   - Consider CSR neighbor format (if memory limited)

3. ⬜ Validation:
   - Physics comparison: grain size distribution, texture index
   - Conservation laws: total volume, grain count
   - Statistical tests: repeat 10 runs, verify consistency

4. ⬜ Scaling studies:
   - Weak scaling: constant work per GPU
   - Strong scaling: constant total work
   - Multi-GPU (if available)

**Testing:**
- Production test: 200x200x1000 voxels, 100 layers
- Compare vs CPU: microstructure images, IPF plots
- Performance benchmarks: CPU vs GPU, 1-8 GPUs

**Deliverables:**
- Performance report: `AM_KOKKOS_PERFORMANCE.md`
- Validation report: `AM_KOKKOS_VALIDATION.md`
- Benchmark suite: `benchmark_am_kokkos.sh`

**Success criteria:**
- 30-70x overall speedup on production problem
- Physics matches CPU within statistical uncertainty
- Scales to 8 GPUs with >50% efficiency

---

## 9. Testing Strategy

### 9.1 Unit Tests

**Quaternion operations:**

```cpp
// test_quaternion_ops_kokkos.cpp
void test_rotate_vector() {
  double q[4] = {0.707, 0, 0, 0.707};  // 90° rotation around z
  double r[3] = {1, 0, 0};              // Vector along x
  double result[3];

  // Call device function via wrapper
  Kokkos::parallel_for(1, KOKKOS_LAMBDA(int i) {
    QuaternionOps::rotate_vector(q, r, result);
  });

  // Expected: r' = (0, 1, 0)
  assert(abs(result[0] - 0.0) < 1e-10);
  assert(abs(result[1] - 1.0) < 1e-10);
  assert(abs(result[2] - 0.0) < 1e-10);
}
```

**Temperature gradient:**

```cpp
void test_gradient_simple() {
  // Create 3x3x3 lattice with linear temperature gradient
  // T(x,y,z) = x

  // Compute gradient at center site
  // Expected: (1, 0, 0)

  double gradient[3], magnitude;
  ThermalOps::compute_temperature_gradient(...);

  assert(abs(gradient[0] - 1.0) < 0.01);
  assert(abs(gradient[1] - 0.0) < 0.01);
  assert(abs(gradient[2] - 0.0) < 0.01);
}
```

### 9.2 Integration Tests

**Mushy phase:**

```cpp
// Test nucleation in undercooled liquid
// Setup: 10x10x10 molten region, T < T_nuc
// Run mushy_phase_kernel()
// Check: all sites now solidified (active_flag = 3)
// Check: all quaternions normalized
```

**Solidification:**

```cpp
// Setup: 10x10x10 liquid with one solid seed at center
// Run solidification kernel multiple times
// Check: solid region grows outward
// Check: all sites have spin matching seed
```

**Texture weighting:**

```cpp
// Setup: liquid site with 6 solid neighbors (cubic lattice)
// Each neighbor has different orientation
// Known gradient direction
// Run 1000 trials of texture-weighted selection
// Check: distribution favors grains aligned with gradient
// Statistical test: chi-square, p > 0.05
```

### 9.3 Physics Validation

**Grain structure comparison:**

```python
# validate_am_physics.py

# Run CPU version
os.system("mpirun -np 4 spk_cpu < in.additive_test")

# Run Kokkos version
os.system("mpirun -np 4 spk_kokkos < in.additive_test")

# Load microstructure dumps
grains_cpu = load_vtk("dump.cpu.*.vtk")
grains_kokkos = load_vtk("dump.kokkos.*.vtk")

# Compare grain size distributions
assert ks_test(grains_cpu.sizes, grains_kokkos.sizes).pvalue > 0.05

# Compare texture indices
assert abs(grains_cpu.texture_index - grains_kokkos.texture_index) < 0.05

# Compare IPF images (image similarity metric)
assert ssim(grains_cpu.ipf, grains_kokkos.ipf) > 0.95
```

**Conservation laws:**

```python
# Check total volume preserved
assert abs(sum(grains_cpu.volumes) - sum(grains_kokkos.volumes)) < 1e-6

# Check grain count matches
assert len(grains_cpu) == len(grains_kokkos)
```

### 9.4 Performance Benchmarks

**Weak scaling (constant work per GPU):**

```bash
# 100^3 per GPU
mpirun -np 1 spk_kokkos < in.additive_100  # 1 GPU
mpirun -np 2 spk_kokkos < in.additive_100  # 2 GPUs
mpirun -np 4 spk_kokkos < in.additive_100  # 4 GPUs
mpirun -np 8 spk_kokkos < in.additive_100  # 8 GPUs

# Expected: constant time per timestep
```

**Strong scaling (constant total work):**

```bash
# 200^3 total
mpirun -np 1 spk_kokkos < in.additive_200
mpirun -np 2 spk_kokkos < in.additive_200
mpirun -np 4 spk_kokkos < in.additive_200
mpirun -np 8 spk_kokkos < in.additive_200

# Expected: time decreases by ~2x per doubling
```

**CPU vs GPU:**

```bash
# CPU baseline (6 cores)
OMP_NUM_THREADS=6 mpirun -np 1 spk_cpu < in.additive_200

# GPU (1 device)
mpirun -np 1 spk_kokkos < in.additive_200

# Expected: 30-70x speedup
```

---

## 10. Risk Assessment & Mitigation

### 10.1 Technical Risks

**Risk 1: Race conditions in multi-site updates (nucleation)**

**Impact:** High - incorrect physics
**Probability:** Medium
**Mitigation:**
- Use atomic operations for conflicting writes
- Or use two-pass algorithm (mark, then update)
- Or keep nucleation on CPU (low frequency operation)

**Risk 2: Numerical instability in quaternion operations**

**Impact:** Medium - incorrect orientations
**Probability:** Low (CPU version is stable)
**Mitigation:**
- Normalize quaternions after every operation
- Add epsilon checks for division by zero
- Extensive unit testing

**Risk 3: Memory bandwidth bottleneck**

**Impact:** Medium - lower than expected speedup
**Probability:** Medium
**Mitigation:**
- Cache frequently accessed data (gradients)
- Optimize memory access patterns (coalescing)
- Profile memory bandwidth utilization
- Consider texture memory for read-only data

**Risk 4: HDF5 I/O becomes bottleneck**

**Impact:** High - limits overall speedup
**Probability:** High (HDF5 is CPU-only)
**Mitigation:**
- Batch temperature updates (update every N timesteps)
- Pre-load HDF5 data to device (if fits in memory)
- Use asynchronous I/O (overlap with computation)
- Consider GPU-compatible I/O library (ADIOS2)

### 10.2 Physics Validation Risks

**Risk 5: Subtle differences in random number sequences**

**Impact:** Medium - different results vs CPU
**Probability:** High (different RNG implementation)
**Mitigation:**
- Accept statistical equivalence, not bitwise identical
- Use statistical tests (KS test, chi-square)
- Document expected variability in user guide

**Risk 6: Texture model gives different grain structure**

**Impact:** High - physics correctness questioned
**Probability:** Low (if implemented correctly)
**Mitigation:**
- Validate intermediate calculations (gradients, angles)
- Compare weighted selection distributions
- Use tight tolerances in unit tests
- Visualize microstructures side-by-side

### 10.3 Performance Risks

**Risk 7: Synchronization overhead dominates**

**Impact:** High - low speedup
**Probability:** Medium (many host-device transfers)
**Mitigation:**
- Minimize sync points (batch operations)
- Profile to quantify overhead
- Consider CUDA-aware MPI (zero-copy)

**Risk 8: Kernel launch overhead**

**Impact:** Low - small absolute time
**Probability:** Low
**Mitigation:**
- Fuse kernels where possible
- Use async launches with streams
- Profile kernel launch times

---

## 11. Code Organization

### 11.1 File Structure

**New files to create:**

```
src/
├── KOKKOS/
│   └── app_additive_ext_temp_texture_kokkos.cpp  [NEW]
│   └── app_additive_ext_temp_texture_kokkos.h    [NEW]
│
├── app_potts_quaternion_kokkos.h  [MODIFY - add quaternion ops]
│
tests/
├── test_quaternion_ops_kokkos.cpp     [NEW]
├── test_thermal_ops_kokkos.cpp        [NEW]
├── test_am_kernels_kokkos.cpp         [NEW]
│
examples/
├── additive_kokkos/
│   ├── in.additive_test               [NEW]
│   ├── in.additive_100                [NEW]
│   ├── in.additive_200                [NEW]
│   └── validate_am_physics.py         [NEW]
│
docs/
├── AM_KOKKOS_PLAN.md                  [THIS FILE]
├── AM_KOKKOS_USAGE.md                 [NEW - user guide]
├── AM_KOKKOS_PERFORMANCE.md           [NEW - benchmarks]
└── AM_KOKKOS_VALIDATION.md            [NEW - physics tests]
```

### 11.2 Class Organization

**app_additive_ext_temp_texture_kokkos.h:**

```cpp
#ifndef SPK_APP_ADDITIVE_EXT_TEMP_TEXTURE_KOKKOS_H
#define SPK_APP_ADDITIVE_EXT_TEMP_TEXTURE_KOKKOS_H

#include "app_potts_quaternion_kokkos.h"

namespace SPPARKS_NS {

class AppAdditiveExtTempTextureKokkos : public AppPottsQuaternionKokkos {
 public:
  AppAdditiveExtTempTextureKokkos(class SPPARKS *, int, char **);
  ~AppAdditiveExtTempTextureKokkos() override;

  void grow_app() override;
  void init_app() override;
  void app_update(double dt) override;

 private:
  // Read-only arrays (updated from HDF5 or computed once/timestep)
  Kokkos::DualView<double*> k_T;          // Temperature field
  Kokkos::View<double*> d_G;              // Gradient magnitude (device-only)
  Kokkos::View<double*> d_grad_x;         // Gradient x-component
  Kokkos::View<double*> d_grad_y;         // Gradient y-component
  Kokkos::View<double*> d_grad_z;         // Gradient z-component
  Kokkos::View<double*> d_V;              // Solidification velocity

  // Read-write arrays (modified during parallel sweeps - need double buffering)
  KokkosDoubleBuffer<double> buffer_mobility_out;
  KokkosDoubleBuffer<double> buffer_solid_d;
  KokkosDoubleBuffer<int> buffer_active_flag;

  // Nucleation data (device, read-only)
  Kokkos::View<int*> d_nucleation_flags;
  Kokkos::View<double*> d_nucleation_temps;
  Kokkos::View<double*> d_nucleation_sizes;

  // Temperature source (CPU-only)
  class TemperatureSource *temperature_source;

  // AM parameters
  double T_melt;
  double texture_c1, texture_c2, texture_c3;

  // Kernel launches
  void update_temperature_from_hdf5(double time);  // CPU
  void mushy_phase_kernel();                        // GPU
  void solidification_kernel();                     // GPU
  void nucleation_kernel();                         // GPU (or CPU)

  // Device operations (nested structs)
  struct ThermalOps;
  struct TextureOps;

  // Functors
  struct MushyPhaseFunctor;
  struct SolidificationFunctor;
  struct NucleationMarkFunctor;
  struct NucleationFlipFunctor;
};

// Thermal operations
struct AppAdditiveExtTempTextureKokkos::ThermalOps {
  KOKKOS_INLINE_FUNCTION
  static void compute_temperature_gradient(/* ... */);
};

// Texture operations
struct AppAdditiveExtTempTextureKokkos::TextureOps {
  KOKKOS_INLINE_FUNCTION
  static double melt_misorientation_device(/* ... */);
};

// Functors (defined in .cpp file)
struct AppAdditiveExtTempTextureKokkos::MushyPhaseFunctor { /* ... */ };
struct AppAdditiveExtTempTextureKokkos::SolidificationFunctor { /* ... */ };
struct AppAdditiveExtTempTextureKokkos::NucleationMarkFunctor { /* ... */ };
struct AppAdditiveExtTempTextureKokkos::NucleationFlipFunctor { /* ... */ };

}  // namespace SPPARKS_NS

#endif
```

### 11.3 Coding Standards

**Follow SPPARKS/Kokkos conventions:**

1. **Indentation:** 2 spaces (no tabs)
2. **Naming:**
   - Classes: `CamelCase`
   - Functions: `snake_case`
   - Member variables: `snake_case`
   - Constants: `UPPER_CASE`
3. **Kokkos patterns:**
   - Use `KOKKOS_INLINE_FUNCTION` for all device functions
   - Use `KOKKOS_LAMBDA` for small lambdas
   - Use `Kokkos::parallel_for`, not raw CUDA/HIP
   - Fence after kernels that write data read by next kernel
4. **Comments:**
   - Document all public functions
   - Explain non-obvious algorithms
   - Add TODO for future optimizations
5. **Error handling:**
   - Check View sizes before access
   - Validate inputs in constructors
   - Use `error->all()` for fatal errors

---

## 12. User Documentation Plan

### 12.1 AM_KOKKOS_USAGE.md

**Contents:**

1. **Building with KOKKOS package:**
   ```bash
   cd src
   ./build_cmake.sh --package kokkos --package stitch --hdf5 -m mpi
   ```

2. **Input file syntax:**
   ```
   app_style additive/ext/temp/texture/kk nspins temp_style

   # Same as CPU version, automatically uses GPU
   # No special flags needed
   ```

3. **Performance tuning:**
   ```
   # Enable profiling
   sync_profiling yes

   # Check buffer sync overhead
   # Output: sync times to log.spparks
   ```

4. **Multi-GPU usage:**
   ```bash
   # 1 MPI rank per GPU
   export CUDA_VISIBLE_DEVICES=0,1,2,3
   mpirun -np 4 spk < in.additive
   ```

5. **Troubleshooting:**
   - Out of memory → reduce domain size or use multi-GPU
   - Slow performance → check sync_profiling, reduce MPI ranks
   - Different results → statistical variation, check seed

### 12.2 Code Comments

**Example documented function:**

```cpp
// Compute temperature gradient at site i using weighted least-squares
//
// Algorithm: For each of 6 cardinal directions (+x,-x,+y,-y,+z,-z),
// average temperature of neighbors in that direction, then compute
// centered difference: grad_x = (avg_T_px - avg_T_mx) / (avg_dist)
//
// Args:
//   i: Site index (0 <= i < nlocal)
//   T_read: Temperature field (read buffer)
//   numneigh: Number of neighbors per site
//   neighbor: Neighbor list [site][neighbor_index] -> neighbor_site
//   xyz: Site coordinates [site][dimension]
//   grad_out: Output gradient vector [3] (modified)
//   magnitude_out: Output gradient magnitude (modified)
//
// Returns: void (modifies grad_out and magnitude_out)
//
// Complexity: O(numneigh[i]) ≈ O(26) for FCC lattice
//
// Thread safety: Read-only access to all inputs, writes to separate
// output pointers per thread (safe for parallel_for)
//
KOKKOS_INLINE_FUNCTION
static void compute_temperature_gradient(
    int i,
    const view_double_type& T_read,
    const view_int_type& numneigh,
    const Kokkos::View<int**>& neighbor,
    const Kokkos::View<double**>& xyz,
    double* grad_out,
    double* magnitude_out)
{
  // Implementation...
}
```

---

## 13. Performance Predictions

### 13.1 Expected Speedups by Phase

**Phase 2 (Temperature on device):**
- Eliminates 2x full-array sync per sector sweep
- Expected: **5-10x** vs current CPU

**Phase 3A (Mushy phase kernel):**
- Parallel nucleation checks
- Expected: **50-100x** for nucleation events (rare, low overall impact)

**Phase 3B (Texture model):**
- Parallel gradient calculations
- Parallel misorientation calculations
- Expected: **20-50x** for texture calculations (high impact)

**Phase 3C (Solidification kernel):**
- Parallel grain growth with texture
- Expected: **10-30x** for solidification sweeps

**Overall (all phases):**
- Bottleneck: HDF5 I/O (unavoidable)
- Expected: **30-70x** for full AM simulation
- Best case: **100x** if HDF5 data pre-loaded

### 13.2 Scaling Predictions

**Weak scaling (constant work per GPU):**

| GPUs | Problem Size | Expected Time | Efficiency |
|------|--------------|---------------|------------|
| 1    | 100³        | 1.0x          | 100%       |
| 2    | 100³ each   | 1.1x          | 91%        |
| 4    | 100³ each   | 1.2x          | 83%        |
| 8    | 100³ each   | 1.3x          | 77%        |

**Strong scaling (constant total work):**

| GPUs | Problem Size | Expected Time | Speedup | Efficiency |
|------|--------------|---------------|---------|------------|
| 1    | 200³        | 1.0x          | 1.0x    | 100%       |
| 2    | 200³        | 0.55x         | 1.8x    | 91%        |
| 4    | 200³        | 0.30x         | 3.3x    | 83%        |
| 8    | 200³        | 0.17x         | 5.9x    | 74%        |

**Scaling limited by:**
- MPI communication overhead
- Load imbalance (sectors may have different work)
- HDF5 I/O (shared file access)

### 13.3 Memory Requirements

**Per site data:**

| Array | Type | Bytes | Notes |
|-------|------|-------|-------|
| spin | int | 4 | Grain ID |
| q0,qx,qy,qz | double×4 | 32 | Quaternion |
| T,G,V | double×3 | 24 | Temperature, gradient, velocity |
| active_flag | int | 4 | Phase state |
| xyz | double×3 | 24 | Coordinates |
| neighbor | int×26 | 104 | Neighbor list (max) |
| **Total** | | **192** | **Bytes per site** |

**Double-buffer overhead:** 2x for all modified arrays
**Effective:** ~250 bytes per site (including read + write buffers)

**Example:**

| Problem Size | Sites | Memory (device) | GPU Needed |
|--------------|-------|-----------------|------------|
| 100³         | 1M    | 250 MB          | Any modern GPU |
| 200³         | 8M    | 2 GB            | Any modern GPU |
| 500³         | 125M  | 31 GB           | A100 (40GB) |
| 1000³        | 1B    | 250 GB          | Multi-GPU required |

**Note:** Add ~30% for Kokkos overhead (Views, RNG pool, etc.)

---

## 14. Future Enhancements (Beyond Phase 5)

### 14.1 GPU-Compatible I/O

**Problem:** HDF5 library is CPU-only, limits speedup

**Solution:** Use GPU-compatible I/O library

**Options:**
1. **ADIOS2** - Supports GPU buffers, MPI-IO
2. **H5Z-ZFP** - GPU compression plugin for HDF5
3. **CuPy** - GPU arrays with HDF5 interface

**Expected impact:** Eliminate I/O bottleneck, enable 100-200x total speedup

### 14.2 Advanced Optimizations

**Texture memory (CUDA):**
```cpp
// Read-only temperature data in texture memory (cached, interpolated)
cudaResourceDesc resDesc;
cudaCreateTextureObject(&tex_temperature, &resDesc);

// Access in kernel (faster than global memory for spatial locality)
double T_site = tex3D<double>(tex_temperature, x, y, z);
```

**Shared memory for neighbors:**
```cpp
// Load all neighbors into shared memory (team collaboration)
__shared__ double shared_T[256];  // One per thread
shared_T[threadIdx.x] = T_read(neighbor(i, threadIdx.x));
__syncthreads();

// Now all threads can access neighbors without global memory
```

**Warp-level primitives:**
```cpp
// Reduce unique neighbor spins using warp shuffle
int spin = spin_read(neighbor(i,j));
unsigned mask = __match_any_sync(0xFFFFFFFF, spin);
int unique = __ffs(mask) - 1;  // First thread with this spin
```

**Expected impact:** Additional 2-5x speedup

### 14.3 Multi-Resolution Time Integration

**Problem:** Liquid sites change fast, solid sites change slow

**Solution:** Use different timesteps for different regions

```cpp
// Fast timestep for molten zone
if (active_flag(i) == 2) dt_local = dt_fast;

// Slow timestep for solidified zone
if (active_flag(i) == 3) dt_local = dt_slow;

// Adaptive timestep ratio
dt_fast / dt_slow = 100
```

**Expected impact:** 10-100x speedup by skipping unnecessary updates

### 14.4 Machine Learning Texture Model

**Idea:** Replace expensive `melt_misorientation()` with neural network

**Training data:**
- Input: (q_site, gradient) → quaternion + 3D vector
- Output: mobility factor

**Inference on GPU:**
- Small network (~1000 parameters)
- Fast forward pass (~10 FLOPs vs 1000 FLOPs for current model)

**Expected impact:** 10-50x speedup for texture calculations

---

## 15. Summary & Next Steps

### 15.1 Key Takeaways

1. **Solid foundation:** AppPottsQuaternionKokkos provides proven infrastructure (20x CPU speedup)

2. **Clear path forward:** 4-phase roadmap with well-defined tasks

3. **High impact targets:** Temperature gradient and texture model offer 10-20x speedup each

4. **Manageable risk:** Most risks have clear mitigation strategies

5. **Expected outcome:** 30-70x overall speedup for production AM simulations

### 15.2 Critical Success Factors

1. **Correctness first, performance second**
   - Validate physics at every step
   - Unit test all KOKKOS_INLINE functions
   - Statistical equivalence, not bitwise identical

2. **Incremental development**
   - Complete each phase before moving to next
   - Maintain working CPU version as reference
   - Frequent testing on small problems

3. **Minimize synchronization**
   - Batch operations
   - Profile to quantify overhead

4. **Leverage existing code**
   - Extend QuaternionOps struct
   - Reuse KokkosDoubleBuffer framework
   - Follow LAMMPS/SPARTA patterns

### 15.3 Immediate Next Steps

**✅ Phase 1 Completed (November 14, 2025):**

1. ✅ Review this document
2. ✅ Set up test environment
3. ✅ Implement `vector_cross_product()`
4. ✅ Implement `rotate_vector()`
5. ✅ Implement `get_cosine_angle_q_u()`
6. ✅ Implement `rotate_q_towards_u()`
7. ✅ Implement `generate_random_quaternion()`
8. ✅ Physics validation passed
9. ✅ Commit a0f59ef

**Starting Phase 2 (Temperature Integration):**

1. ⬜ Add double-buffer arrays for temperature fields
2. ⬜ Implement `ThermalOps::compute_temperature_gradient()`
3. ⬜ Update `grow_app()` to allocate temperature arrays
4. ⬜ Modify `app_update()` for temperature sync
5. ⬜ Test gradient calculation against CPU version

**Updated Timeline:**
- Phase 1: ✅ Complete (November 14, 2025)
- Phase 2: Weeks 3-5 (temperature) ← **NEXT**
- Phase 3: Weeks 6-11 (AM functors)
- Phase 4: Weeks 12-14 (nucleation)
- Phase 5: Weeks 15-18 (optimization)

**Estimated completion:** 3-4 months from Phase 1 completion

---

## 16. References

### 16.1 Code References

**Existing implementations:**
- `src/KOKKOS/app_potts_quaternion_kokkos.{h,cpp}` - Base class, quaternion ops
- `src/kokkos_double_buffer.h` - Double-buffer framework
- `src/app_additive_ext_temp_texture.{h,cpp}` - Target CPU implementation
- `src/potts_quaternion/quaternion.h` - Quaternion utilities (CPU)
- `src/crystalline_orientations/quaternion.h` - Rotation utilities (CPU)

**Documentation:**
- `docs/KOKKOS_IMPLEMENTATION_ROADMAP.md` - Overall Kokkos strategy
- `docs/KOKKOS_IMPLEMENTATION_HISTORY.md` - Completed work chronicle
- `examples/potts_quaternion/KOKKOS_PHYSICS_VALIDATION.md` - Validation results

### 16.2 External Resources

**Kokkos:**
- [Kokkos Programming Guide](https://kokkos.org/kokkos-core-wiki/ProgrammingGuide/ProgrammingModel.html)
- [Kokkos Tutorials](https://github.com/kokkos/kokkos-tutorials)
- [LAMMPS Kokkos Package](https://docs.lammps.org/Speed_kokkos.html) - Similar implementation

**Quaternion Math:**
- Shoemake, K. (1985). "Animating rotation with quaternion curves"
- Kuipers, J. B. (1999). "Quaternions and Rotation Sequences"

**Additive Manufacturing:**
- Rodgers, T. M., et al. (2021). "Simulation of metal additive manufacturing microstructures"
- SPPARKS Additive Manufacturing Documentation

**GPU Optimization:**
- [NVIDIA CUDA Best Practices](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- [AMD ROCm Performance Tuning](https://rocm.docs.amd.com/en/latest/how-to/tuning-guides.html)
- [Kokkos Performance Tuning](https://kokkos.org/kokkos-core-wiki/ProgrammingGuide/Tuning.html)

---

## 17. Appendices

### Appendix A: Quaternion Math Reference

**Unit quaternion:** q = (q0, qx, qy, qz) with ||q|| = 1

**Rotation by quaternion:**
```
r' = q * r * q^(-1)
   = r + 2*q0*(q_vec × r) + 2*(q_vec × (q_vec × r))
```

**Composition:**
```
q3 = q1 * q2
q3.q0 = q1.q0*q2.q0 - q1.qx*q2.qx - q1.qy*q2.qy - q1.qz*q2.qz
q3.qx = q1.q0*q2.qx + q1.qx*q2.q0 + q1.qy*q2.qz - q1.qz*q2.qy
q3.qy = q1.q0*q2.qy - q1.qx*q2.qz + q1.qy*q2.q0 + q1.qz*q2.qx
q3.qz = q1.q0*q2.qz + q1.qx*q2.qy - q1.qy*q2.qx + q1.qz*q2.q0
```

**Rotation matrix from quaternion:**
```
R[0][0] = 1 - 2*(qy² + qz²)
R[0][1] = 2*(qx*qy - q0*qz)
R[0][2] = 2*(qx*qz + q0*qy)
R[1][0] = 2*(qx*qy + q0*qz)
R[1][1] = 1 - 2*(qx² + qz²)
R[1][2] = 2*(qy*qz - q0*qx)
R[2][0] = 2*(qx*qz - q0*qy)
R[2][1] = 2*(qy*qz + q0*qx)
R[2][2] = 1 - 2*(qx² + qy²)
```

### Appendix B: Texture Model Details

**Melt misorientation model:** (Rappaz, 1989)

```
M(θ) = M₀ * exp(-c2 / (cos(θ) + c3))
```

Where:
- M(θ): Mobility (growth rate)
- θ: Angle between crystal orientation and thermal gradient
- c1 = M₀: Baseline mobility
- c2: Texture strength parameter
- c3: Offset to avoid singularity

**Physical meaning:**
- θ = 0° (aligned): Maximum growth rate
- θ = 90° (perpendicular): Minimum growth rate
- Creates columnar grain structure along build direction

### Appendix C: Active Flag States

| Value | State | Description | Temperature Range |
|-------|-------|-------------|-------------------|
| 0 | Inactive | Outside build region | Any |
| 1 | Active | In active layer (not yet molten) | T < T_melt |
| 2 | Molten | Liquid state | T >= T_melt |
| 3 | Solidified | Solid with defined grain | T < T_melt |
| 5 | Void | Empty space (pores) | Any |

**Transitions:**
- 0 → 1: Layer activation (part of build)
- 1 → 2: Melting (laser passes)
- 2 → 3: Solidification (cooling)
- Any → 5: Void creation (porosity)

### Appendix D: Benchmark Problem Specifications

**Small test (development):**
- Size: 20³ = 8,000 sites
- Layers: 10
- Time: ~10 seconds (CPU)
- Purpose: Quick validation

**Medium test (validation):**
- Size: 100³ = 1,000,000 sites
- Layers: 50
- Time: ~1 hour (CPU)
- Purpose: Physics validation

**Large test (production):**
- Size: 200³ = 8,000,000 sites
- Layers: 100
- Time: ~10 hours (CPU)
- Purpose: Performance benchmarking

**Expected GPU times:**
- Small: <1 second
- Medium: <2 minutes
- Large: <10 minutes (30-70x speedup)

---

## Document History

| Date | Version | Author | Changes |
|------|---------|--------|---------|
| 2025-11-14 | 1.0 | Claude Code | Initial document creation |
| 2025-11-14 | 1.1 | Claude Code | Phase 1 completion update - all quaternion operations implemented and validated |

---

**End of AM_KOKKOS_PLAN.md**
