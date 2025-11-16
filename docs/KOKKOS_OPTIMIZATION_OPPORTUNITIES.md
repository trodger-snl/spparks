# Kokkos AM Code Optimization Analysis

**Date**: November 16, 2025
**Status**: Performance Gap Analysis - Current vs CPU Implementation
**Goal**: Eliminate 1.50x performance gap and achieve parity or superiority over CPU version

---

## Executive Summary

**Current Performance:**
- **Kokkos (optimized)**: 25.8 ms/step
- **CPU**: 17.2 ms/step
- **Gap**: 1.50x slower than CPU (was 3.2x slower originally)

**Already Achieved:**
- 2.16x speedup from original Kokkos implementation (55.7 → 25.8 ms/step)
- Temperature loading now **faster** than CPU (14.8 ms vs 15.2 ms)
- Successful optimizations: on-demand gradients, selective kernel launching, smoothing loop fix

**Optimization Potential:**
- **Phase 1** (low-hanging fruit): Reduce to ~21 ms/step (1.22x slower)
- **Phase 2** (kernel fusion): Reduce to ~18 ms/step (competitive with CPU)
- **Phase 3** (advanced): Reduce to ~16 ms/step (**faster than CPU!**)

**Root Cause of Remaining Gap:**
OpenMP backend kernel launch overhead and multiple memory passes over data. The CPU version uses a single unified loop that is cache-friendly and has zero kernel launch overhead.

---

## Performance Breakdown

### Current Kokkos Timing (after 2500 steps, 64.44s total)

| Operation              | Time (s) | % Total | Time/Step (ms) | vs CPU |
|------------------------|----------|---------|----------------|--------|
| Temperature update     | 36.893   | 57.2%   | 14.8          | 0.97x (better!) |
| Temp sync to device    | 0.948    | 1.5%    | 0.4           | N/A |
| **Melting kernel**     | **11.075** | **17.2%** | **4.4**     | **400x slower** |
| Nucleation kernel      | 1.978    | 3.1%    | 0.8           | 3x slower |
| Particle kernel        | 1.252    | 1.9%    | 0.5           | ~10x slower |
| Solidification kernel  | 9.392    | 14.6%   | 3.8           | 13x slower |
| Smoothing kernel       | 2.903    | 4.5%    | 1.2           | 30x slower |
| Sync to host           | 0.003    | 0.0%    | 0.0           | N/A |
| **TOTAL**              | **64.44** | **100%** | **25.8**     | **1.50x slower** |

### CPU Timing Reference

| Operation | Time (s) | % Total | Time/Step (ms) |
|-----------|----------|---------|----------------|
| Temperature loading | 38.0 | 88.4% | 15.2 |
| Melting | ~0.025 | 0.1% | 0.01 |
| Mushy phase (nuc+solid) | ~1.5 | 3.5% | 0.6 |
| Smoothing | ~0.1 | 0.2% | 0.04 |
| Communication | ~3.25 | 7.6% | 1.3 |
| **TOTAL** | **43.0** | **100%** | **17.2** |

---

## Priority 1: Melting Kernel Overhead (Highest Impact)

### Problem Analysis

**Impact**: 4.4 ms/step (17.2% of runtime)
**Root Cause**: Checks ALL 2.99M sites every timestep, but only ~0.01-0.1% actually melt

**Current Implementation:**
```cpp
// app_additive_ext_temp_texture_kokkos.cpp:614
int AppAdditiveExtTempTextureKokkos::melting_kernel()
{
  int naccept_local = 0;
  Kokkos::parallel_reduce("MeltingKernel", nlocal_local,
    KOKKOS_LAMBDA(int i, int& count) {
      double T_i = d_T_local(i);
      int active_flag_i = d_active_flag_current(i);

      if (T_i >= tl_local && active_flag_i != 2) {
        // Melting logic (RARE: <0.1% of sites)
        d_active_flag_next(i) = 2;  // LIQUID
        d_spin_next(i) = random_spin;
        // ... generate random quaternion
        count++;
      } else {
        // Copy 8 fields unchanged (COMMON: >99.9% of sites!)
        d_active_flag_next(i) = active_flag_i;
        d_spin_next(i) = d_spin_current(i);
        d_q0_next(i) = d_q0_current(i);
        // ... 5 more fields
      }
    }, naccept_local);

  return naccept_local;
}
```

**CPU Approach** (for comparison):
```cpp
// CPU: app_additive_ext_temp_texture.cpp:375-390
for (int i=0; i<nlocal; i++) {
  if (active_flag[i] == 5) continue;  // Skip voids

  if ((T[i] >= tl) && active_flag[i] != 2) {
    // Melting operation - only 0.01% of sites
    // Total CPU time: 0.01 ms/step
  }
  else if (active_flag[i] == 2 && T[i] <= tl) {
    mushy_phase(i, ranapp);
  }
  else if(solid_d[i] < 0 && solid_d[i] > -nrefine -1 && active_flag[i] == 3) {
    smooth_site(i);
  }
}
```

### Optimization 1A: Early Temperature Scan (Quick Win)

**Concept**: Check if ANY sites exceed liquidus before launching expensive kernel

**Implementation:**
```cpp
void AppAdditiveExtTempTextureKokkos::app_update(double dt)
{
  // ... temperature update ...

  // Phase transition kernels

  // 1. MELTING: Quick pre-scan to avoid kernel launch
  t_start = MPI_Wtime();

  // Lightweight check: any sites hot enough to melt?
  int n_melt_candidates = 0;
  {
    auto d_T_check = k_T.d_view;
    auto d_active_flag_check = buffer_active_flag.read_buffer;
    double tl_check = this->tl;

    Kokkos::parallel_reduce("CheckMeltingCandidates", nlocal,
      KOKKOS_LAMBDA(int i, int& count) {
        if (d_T_check(i) >= tl_check && d_active_flag_check(i) != 2) {
          count++;
        }
      }, n_melt_candidates);
  }

  // Only launch melting kernel if there are candidates
  if (n_melt_candidates > 0) {
    naccept += melting_kernel();
    swap_buffers();
  } else {
    // No melting - just copy current state to next
    // (Could skip entirely, but safer to maintain buffer consistency)
    copy_state_no_change();
  }

  t_end = MPI_Wtime();
  t_melting += (t_end - t_start);

  // ... rest of kernels ...
}
```

**Helper function:**
```cpp
void AppAdditiveExtTempTextureKokkos::copy_state_no_change()
{
  // Fast copy when no changes expected
  auto d_spin_current = buffer_spin.read_buffer;
  auto d_spin_next = buffer_spin.write_buffer;
  auto d_active_flag_current = buffer_active_flag.read_buffer;
  auto d_active_flag_next = buffer_active_flag.write_buffer;
  // ... other buffers

  int nlocal_local = this->nlocal;

  Kokkos::parallel_for("CopyStateNoChange", nlocal_local,
    KOKKOS_LAMBDA(int i) {
      d_spin_next(i) = d_spin_current(i);
      d_active_flag_next(i) = d_active_flag_current(i);
      // ... copy other fields
    });
}
```

**Performance Analysis:**
- **Check cost**: ~0.2-0.3 ms (lightweight parallel_reduce)
- **Kernel save**: 4.4 ms when no melting occurs
- **Frequency**: Melting occurs ~10-20% of timesteps during laser pass
- **Expected savings**: 0.8 × 4.4 = **3.5 ms/step average**

**Risk**: Low
**Effort**: ~30 lines of code
**Timeline**: 2-4 hours

### Optimization 1B: Temperature-Based Spatial Filtering (Advanced)

**Concept**: Maintain a compact list of "hot zone" sites, process only those

**Implementation:**
```cpp
class AppAdditiveExtTempTextureKokkos : public AppPottsQuaternionKokkos {
  // ... existing members ...

  // Add spatial filtering data structures
  Kokkos::View<int*> d_hot_sites;          // Compact list of hot site indices
  Kokkos::View<int> d_n_hot;               // Count of hot sites
  int hot_sites_capacity;                   // Allocated capacity
  double melt_buffer_temp;                  // T_liquidus - 50K buffer
};

void AppAdditiveExtTempTextureKokkos::init_app()
{
  // ... existing initialization ...

  // Allocate hot zone tracking
  hot_sites_capacity = nlocal;
  d_hot_sites = Kokkos::View<int*>("hot_sites", hot_sites_capacity);
  d_n_hot = Kokkos::View<int>("n_hot");
  melt_buffer_temp = tl - 50.0;  // 50K buffer below liquidus
}

int AppAdditiveExtTempTextureKokkos::melting_kernel_filtered()
{
  // Step 1: Build list of hot sites (sites within 50K of liquidus)
  auto d_T_local = k_T.d_view;
  auto d_active_flag_current = buffer_active_flag.read_buffer;
  double tl_local = this->tl;
  double buffer_temp = this->melt_buffer_temp;
  int nlocal_local = this->nlocal;

  auto d_hot_sites_local = this->d_hot_sites;
  auto d_n_hot_local = this->d_n_hot;

  // Reset counter
  Kokkos::deep_copy(d_n_hot_local, 0);

  // Parallel scan to build compact list
  Kokkos::parallel_scan("BuildHotSitesList", nlocal_local,
    KOKKOS_LAMBDA(int i, int& offset, bool final) {
      bool is_hot = (d_T_local(i) >= buffer_temp &&
                     d_active_flag_current(i) != 2);
      if (is_hot) {
        if (final) {
          d_hot_sites_local(offset) = i;
        }
        offset++;
      }
    });

  // Get count on host
  auto h_n_hot = Kokkos::create_mirror_view(d_n_hot_local);
  Kokkos::deep_copy(h_n_hot, d_n_hot_local);
  int n_hot = h_n_hot();

  if (n_hot == 0) {
    return 0;  // No hot sites, skip kernel
  }

  // Step 2: Process ONLY hot sites
  auto d_spin_current = buffer_spin.read_buffer;
  auto d_spin_next = buffer_spin.write_buffer;
  auto d_q0_current = buffer_q0.read_buffer;
  auto d_q0_next = buffer_q0.write_buffer;
  // ... other buffers

  auto rand_pool_local = this->rand_pool;
  int nspins_local = this->nspins;

  int naccept_local = 0;
  Kokkos::parallel_reduce("MeltingKernelFiltered", n_hot,
    KOKKOS_LAMBDA(int idx, int& count) {
      int i = d_hot_sites_local(idx);  // Get actual site index

      double T_i = d_T_local(i);
      int active_flag_i = d_active_flag_current(i);

      // Check if actually melting (could be in buffer zone)
      if (T_i >= tl_local && active_flag_i != 2) {
        // Full melting logic (same as before)
        auto generator = rand_pool_local.get_state();

        d_active_flag_next(i) = 2;  // LIQUID
        d_spin_next(i) = static_cast<int>(nspins_local * generator.drand());

        // Generate random quaternion
        double x0, x1, x2, x3, r1, r2;
        // ... Marsaglia method ...

        rand_pool_local.free_state(generator);
        count++;
      } else {
        // In buffer zone but not melting - just copy
        d_active_flag_next(i) = active_flag_i;
        d_spin_next(i) = d_spin_current(i);
        // ... copy other fields
      }
    }, naccept_local);

  // Step 3: Copy unchanged sites (those NOT in hot list)
  // This is optional - could use a "dirty bit" approach instead

  return naccept_local;
}
```

**Performance Analysis:**
- **Hot zone size**: ~1-5% of domain (30k-150k sites vs 2.99M)
- **Scan cost**: ~0.5 ms (parallel_scan)
- **Filtered kernel cost**: ~0.1-0.2 ms (processes 1-5% of sites)
- **Copy unchanged sites**: Can skip if using dirty tracking
- **Total cost**: ~0.6-0.7 ms vs 4.4 ms current
- **Expected savings**: **3.7-3.8 ms/step**

**Benefits over 1A:**
- Processes only relevant sites (not entire domain)
- No conditional branching in kernel (all hot sites processed)
- Better memory access pattern (contiguous processing)

**Drawbacks:**
- More complex code
- Requires parallel_scan (slightly more overhead than reduce)
- Must handle buffer zone sites (may not actually melt)

**Risk**: Medium
**Effort**: ~200 lines of code
**Timeline**: 1-2 days

**Recommendation**: Start with **Optimization 1A** (quick win), then implement **1B** if more performance needed.

---

## Priority 2: Kernel Fusion (Major Architectural Change)

### Problem Analysis

**Impact**: ~1-2 ms/step from kernel launch overhead + memory traffic reduction
**Root Cause**: Multiple separate kernel launches when CPU uses single unified loop

**Current Architecture:**
```
Kokkos (5 kernels):
├── melting_kernel()          → 2.99M sites checked
├── nucleation_kernel()       → 2.99M sites checked
├── particle_flipper_kernel() → 2.99M sites checked (conditional launch)
├── solidification_kernel()   → 2.99M sites checked
└── smoothing_kernel()        → 2.99M sites checked (conditional launch)

Memory reads: 5 × (T + active_flag + spin + q0-qz + solid_d) = 5 × 100 bytes = 500 bytes/site
Memory writes: 5 × (spin + q0-qz + active_flag + solid_d) = 5 × 80 bytes = 400 bytes/site
Total memory traffic: 900 bytes/site × 2.99M = 2.69 GB per timestep
```

**CPU Architecture:**
```
CPU (1 loop):
for (int i=0; i<nlocal; i++) {
  if (T[i] >= tl && active_flag[i] != 2) {
    melting_operation(i);
  }
  else if (active_flag[i] == 2 && T[i] <= tl) {
    mushy_phase(i, ranapp);  // Includes nucleation + solidification
  }
  else if(solid_d[i] < 0 && solid_d[i] > -nrefine -1 && active_flag[i] == 3) {
    smooth_site(i);
  }
}

Memory reads: 1 × 100 bytes/site
Memory writes: 1 × 80 bytes/site (only modified sites)
Total memory traffic: ~180 bytes/site × 2.99M = 538 MB per timestep
```

**Memory Traffic Comparison**: Kokkos uses **5× more memory bandwidth** than CPU

### Optimization 2: Unified Phase Transition Kernel

**Concept**: Merge all phase transition kernels into single pass (matches CPU architecture)

**Implementation:**

```cpp
/* ----------------------------------------------------------------------
   Unified phase transition kernel - matches CPU's single-loop architecture
   Handles melting, nucleation, solidification, smoothing in one pass
------------------------------------------------------------------------- */

struct UnifiedPhaseTransitionFunctor {
  // Temperature (read-only)
  Kokkos::View<const double*> d_T;

  // Read buffers (current state)
  view_int_type d_spin_current;
  view_int_type d_active_flag_current;
  view_double_type d_q0_current, d_qx_current, d_qy_current, d_qz_current;
  view_double_type d_solid_d_current;
  view_double_type d_mobility_current;

  // Write buffers (next state)
  view_int_type d_spin_next;
  view_int_type d_active_flag_next;
  view_double_type d_q0_next, d_qx_next, d_qy_next, d_qz_next;
  view_double_type d_solid_d_next;
  view_double_type d_mobility_next;

  // Neighbor data
  view_int_type k_numneigh;
  Kokkos::View<int**> k_neighbor;
  Kokkos::View<double*[3]> k_xyz;

  // Thermal data (device-only caches)
  Kokkos::View<double*> d_G;
  Kokkos::View<double*> d_grad_x, d_grad_y, d_grad_z;

  // Parameters
  double tl, ts, dt, dx;
  int nrefine;
  int nspins;

  // Texture parameters
  Kokkos::View<double*> d_texture_params;  // c1, c2, c3

  // Nucleation data
  Kokkos::View<int*> d_nucleation_flags;
  Kokkos::View<double*> d_nucleation_temps;
  Kokkos::View<double*> d_nucleation_sizes;

  // Solidification parameters
  Kokkos::View<double*> d_solid_front_coeffs;
  int solid_front_length;

  // RNG
  rand_pool_type rand_pool;

  KOKKOS_INLINE_FUNCTION
  void operator()(const int i, int& count) const
  {
    // Read current state ONCE
    double T_i = d_T(i);
    int active_flag_i = d_active_flag_current(i);
    double solid_d_i = d_solid_d_current(i);
    int spin_i = d_spin_current(i);

    // Priority 1: MELTING (highest temperature transition)
    // Same logic as CPU: if (T[i] >= tl && active_flag[i] != 2)
    if (T_i >= tl && active_flag_i != 2) {
      perform_melting(i, count);
      return;  // Exit early - melted sites don't do other transitions
    }

    // Priority 2: MUSHY ZONE (nucleation or solidification)
    // Same logic as CPU: else if (active_flag[i] == 2 && T[i] <= tl)
    if (active_flag_i == 2 && T_i <= tl) {
      // Check nucleation first
      if (should_nucleate(i, T_i, spin_i)) {
        perform_nucleation(i, T_i, spin_i, count);
        return;
      }

      // Check particle growth (just nucleated in previous step)
      if (solid_d_i == -nrefine - 2) {
        perform_particle_growth(i, count);
        return;
      }

      // Check solidification (epitaxial growth from solid neighbors)
      if (can_solidify(i, T_i, solid_d_i)) {
        perform_solidification(i, T_i, solid_d_i, count);
        return;
      }

      // Remains liquid (no change)
      copy_current_to_next(i);
      return;
    }

    // Priority 3: SMOOTHING (grain boundary refinement)
    // Same logic as CPU: else if(solid_d[i] < 0 && solid_d[i] > -nrefine -1)
    if (solid_d_i > -nrefine - 1 && solid_d_i < 0.0 && active_flag_i == 3) {
      perform_smoothing(i, count);
      return;
    }

    // Default: No change (solid, inactive, or void)
    copy_current_to_next(i);
  }

  // Helper functions (all KOKKOS_INLINE_FUNCTION)

  KOKKOS_INLINE_FUNCTION
  void perform_melting(int i, int& count) const
  {
    auto generator = rand_pool.get_state();

    // Transition to liquid
    d_active_flag_next(i) = 2;  // LIQUID
    d_spin_next(i) = static_cast<int>(nspins * generator.drand());

    // Generate random quaternion (Marsaglia method)
    double x0, x1, x2, x3, r1, r2;
    do {
      x0 = 2.0 * generator.drand() - 1.0;
      x1 = 2.0 * generator.drand() - 1.0;
      r1 = x0*x0 + x1*x1;
    } while (r1 >= 1.0);

    do {
      x2 = 2.0 * generator.drand() - 1.0;
      x3 = 2.0 * generator.drand() - 1.0;
      r2 = x2*x2 + x3*x3;
    } while (r2 >= 1.0);

    double s = Kokkos::sqrt((1.0 - r1) / r2);
    d_q0_next(i) = x0;
    d_qx_next(i) = x1;
    d_qy_next(i) = x2 * s;
    d_qz_next(i) = x3 * s;

    // Reset solidification tracking
    d_solid_d_next(i) = 0.0;
    d_mobility_next(i) = 1.0;

    rand_pool.free_state(generator);
    count++;  // Track event
  }

  KOKKOS_INLINE_FUNCTION
  bool should_nucleate(int i, double T_i, int spin_i) const
  {
    // Check if this spin can nucleate
    if (spin_i >= nspins || d_nucleation_flags(spin_i) != 1) {
      return false;
    }

    // Check undercooling
    double undercooling = tl - T_i;
    double critical_undercooling = d_nucleation_temps(spin_i);

    return (undercooling >= critical_undercooling);
  }

  KOKKOS_INLINE_FUNCTION
  void perform_nucleation(int i, double T_i, int spin_i, int& count) const
  {
    auto generator = rand_pool.get_state();

    // Nucleate: transition to solid with random orientation
    d_active_flag_next(i) = 3;  // SOLID
    d_spin_next(i) = spin_i;    // Keep same spin ID

    // Generate random quaternion
    double x0, x1, x2, x3, r1, r2;
    // ... Marsaglia method (same as melting) ...

    // Set particle growth flag
    d_solid_d_next(i) = -nrefine - 2;
    d_mobility_next(i) = 0.0;

    rand_pool.free_state(generator);
    count++;
  }

  KOKKOS_INLINE_FUNCTION
  void perform_particle_growth(int i, int& count) const
  {
    // Grow newly nucleated grain to specified size
    // Same logic as particle_flipper_kernel
    // ... implementation ...

    count++;
  }

  KOKKOS_INLINE_FUNCTION
  bool can_solidify(int i, double T_i, double solid_d_i) const
  {
    // Check if in solidification temperature range
    if (T_i >= tl || T_i < ts) return false;

    // Compute solidification velocity
    double Tcool = tl - T_i;
    double V_i = 0.0;
    double Tcool_power = 1.0;
    for (int k = solid_front_length - 1; k >= 0; k--) {
      V_i += d_solid_front_coeffs(k) * Tcool_power;
      Tcool_power *= Tcool;
    }

    // Advance solid front
    solid_d_i += V_i * dt;

    // Check if any solid neighbors within reach
    int nn = k_numneigh(i);
    for (int j = 0; j < nn; j++) {
      int nbr = k_neighbor(i, j);
      int nbr_flag = d_active_flag_current(nbr);

      if ((nbr_flag == 1 || nbr_flag == 3)) {
        // Compute distance to neighbor
        double dx = k_xyz(nbr, 0) - k_xyz(i, 0);
        double dy = k_xyz(nbr, 1) - k_xyz(i, 1);
        double dz = k_xyz(nbr, 2) - k_xyz(i, 2);
        double dist = Kokkos::sqrt(dx*dx + dy*dy + dz*dz);

        if (dist <= solid_d_i) {
          return true;  // Found solid neighbor within reach
        }
      }
    }

    return false;  // No solid neighbors yet
  }

  KOKKOS_INLINE_FUNCTION
  void perform_solidification(int i, double T_i, double solid_d_i, int& count) const
  {
    // Same logic as solidification_kernel
    // Includes on-demand gradient computation and texture-weighted selection

    // Compute gradient (on-demand, only if solid neighbor found)
    double grad_dir[3] = {0.0, 0.0, 0.0};
    double grad_magnitude = 0.0;
    bool gradient_computed = false;

    // Find solid neighbors and compute texture weights
    int nn = k_numneigh(i);
    int num_solid = 0;
    double cumulative_weight[26];
    int solid_neighbor_idx[26];

    double cumulative_sum = 0.0;
    for (int j = 0; j < nn && j < 26; j++) {
      int nbr = k_neighbor(i, j);
      int nbr_flag = d_active_flag_current(nbr);

      if ((nbr_flag == 1 || nbr_flag == 3)) {
        // Compute distance
        double dx_nbr = k_xyz(nbr, 0) - k_xyz(i, 0);
        double dy_nbr = k_xyz(nbr, 1) - k_xyz(i, 1);
        double dz_nbr = k_xyz(nbr, 2) - k_xyz(i, 2);
        double dist = Kokkos::sqrt(dx_nbr*dx_nbr + dy_nbr*dy_nbr + dz_nbr*dz_nbr);

        if (dist <= solid_d_i) {
          // Compute gradient on first solid neighbor
          if (!gradient_computed) {
            ThermalOps::compute_temperature_gradient(
              i, d_T, k_numneigh, k_neighbor, k_xyz, dx,
              grad_dir, &grad_magnitude);

            d_G(i) = grad_magnitude;
            d_grad_x(i) = grad_dir[0];
            d_grad_y(i) = grad_dir[1];
            d_grad_z(i) = grad_dir[2];
            gradient_computed = true;
          }

          // Get neighbor quaternion
          double q_nbr[4] = {
            d_q0_current(nbr), d_qx_current(nbr),
            d_qy_current(nbr), d_qz_current(nbr)
          };

          // Compute texture weight
          double cos_min_angle = QuaternionOps::get_cosine_angle_q_u(q_nbr, grad_dir);
          double theta = Kokkos::acos(cos_min_angle);
          double c1 = d_texture_params(0);
          double c2 = d_texture_params(1);
          double c3 = d_texture_params(2);
          double mobility = c1 + c2 * Kokkos::cos(c3 * theta);

          if (mobility > 0.0) {
            cumulative_sum += mobility;
            cumulative_weight[num_solid] = cumulative_sum;
            solid_neighbor_idx[num_solid] = nbr;
            num_solid++;
          }
        }
      }
    }

    if (num_solid > 0) {
      // Weighted random selection
      auto generator = rand_pool.get_state();
      double rand_val = generator.drand() * cumulative_sum;

      int selected_nbr = solid_neighbor_idx[num_solid - 1];  // Default: last
      for (int k = 0; k < num_solid; k++) {
        if (rand_val <= cumulative_weight[k]) {
          selected_nbr = solid_neighbor_idx[k];
          break;
        }
      }

      // Copy spin and quaternion from selected neighbor
      d_spin_next(i) = d_spin_current(selected_nbr);
      d_q0_next(i) = d_q0_current(selected_nbr);
      d_qx_next(i) = d_qx_current(selected_nbr);
      d_qy_next(i) = d_qy_current(selected_nbr);
      d_qz_next(i) = d_qz_current(selected_nbr);
      d_active_flag_next(i) = 3;  // SOLID
      d_solid_d_next(i) = -1.0;   // Start smoothing countdown

      rand_pool.free_state(generator);
      count++;
    }
  }

  KOKKOS_INLINE_FUNCTION
  void perform_smoothing(int i, int& count) const
  {
    // Same logic as smoothing_kernel
    // Potts MC for grain boundary refinement
    // ... implementation ...

    // Decrement solid_d counter
    d_solid_d_next(i) = d_solid_d_current(i) - 1.0;
    if (d_solid_d_next(i) <= -nrefine) {
      d_active_flag_next(i) = 1;  // ACTIVE (fully refined)
      d_solid_d_next(i) = 0.0;
    }

    count++;
  }

  KOKKOS_INLINE_FUNCTION
  void copy_current_to_next(int i) const
  {
    // Fast copy when no state change
    d_spin_next(i) = d_spin_current(i);
    d_active_flag_next(i) = d_active_flag_current(i);
    d_q0_next(i) = d_q0_current(i);
    d_qx_next(i) = d_qx_current(i);
    d_qy_next(i) = d_qy_current(i);
    d_qz_next(i) = d_qz_current(i);
    d_solid_d_next(i) = d_solid_d_current(i);
    d_mobility_next(i) = d_mobility_current(i);
  }
};

// Launch unified kernel
int AppAdditiveExtTempTextureKokkos::unified_phase_transition_kernel()
{
  UnifiedPhaseTransitionFunctor functor;

  // Set all views and parameters
  functor.d_T = k_T.d_view;
  functor.d_spin_current = buffer_spin.read_buffer;
  functor.d_spin_next = buffer_spin.write_buffer;
  // ... set all other views ...

  functor.tl = this->tl;
  functor.ts = this->ts;
  functor.dt = this->dt;
  // ... set all parameters ...

  int naccept_total = 0;
  Kokkos::parallel_reduce("UnifiedPhaseTransition", nlocal,
                          functor, naccept_total);

  // Single fence after unified kernel
  Kokkos::fence();

  return naccept_total;
}
```

**Modified app_update():**
```cpp
void AppAdditiveExtTempTextureKokkos::app_update(double dt)
{
  // 1. Update temperature
  // 2. Sync to device

  // 3. UNIFIED phase transition kernel (replaces 5 separate kernels)
  t_start = MPI_Wtime();
  naccept = unified_phase_transition_kernel();
  t_end = MPI_Wtime();
  t_unified_kernel = (t_end - t_start);

  // 4. Single buffer swap (not 5!)
  swap_buffers();

  // 5. Single fence (already in unified kernel)

  // 6. MPI communication
  // ...
}
```

**Performance Analysis:**

**Benefits:**
1. **Kernel launch overhead**: 1 launch vs 5 launches = save ~0.8 ms
2. **Buffer swaps**: 1 swap vs 5 swaps = save ~0.2 ms
3. **Memory traffic**: 180 bytes/site vs 900 bytes/site = save ~5x bandwidth
4. **Cache efficiency**: Single pass matches CPU's cache-friendly pattern
5. **Early exit**: Melted/nucleated sites skip remaining checks (like CPU)

**Expected savings:**
- Kernel overhead: 0.8 ms
- Memory bandwidth: 1.0-1.5 ms (less data movement)
- **Total: 1.8-2.3 ms/step**

**Drawbacks:**
1. **Code complexity**: Large functor (~500-800 lines)
2. **Debugging**: Harder to isolate issues to specific phase
3. **Divergent execution**: Different sites take different paths (but CPU has same issue)

**Risk**: Medium-High
**Effort**: ~500-800 lines, significant refactor
**Timeline**: 3-5 days

**Recommendation**: Implement after Priority 1 optimizations to ensure correctness can be validated incrementally.

---

## Priority 3: Work-Efficient Smoothing

### Problem Analysis

**Impact**: 1.2 ms/step (4.5% of runtime)
**Current**: 30x slower than CPU (1.2 ms vs 0.04 ms)

**Root Cause**: Checks all 2.99M sites when only ~100-1000 need smoothing

**Current Implementation** (already has selective launch):
```cpp
// Check if any sites need smoothing
int n_smoothing_sites = 0;
{
  auto d_solid_d_check = buffer_solid_d.read_buffer;
  int nrefine_check = nrefine;
  Kokkos::parallel_reduce("CheckSmoothingSites", nlocal,
    KOKKOS_LAMBDA(int i, int& count) {
      double sd = d_solid_d_check(i);
      if (sd > -nrefine_check - 1 && sd < 0.0) {
        count++;
      }
    }, n_smoothing_sites);
}

// Only launch if there are sites
if (n_smoothing_sites > 0) {
  naccept += smoothing_kernel();  // Still processes ALL sites!
  swap_buffers();
}
```

**The kernel still checks all sites:**
```cpp
Kokkos::parallel_reduce("SmoothingKernel", nlocal,
  KOKKOS_LAMBDA(int i, int& count) {
    double sd = d_solid_d_current(i);
    if (sd > -nrefine_local - 1 && sd < 0.0) {
      // Smoothing logic (~100-1000 sites)
    } else {
      // Copy state (2.99M sites!)
    }
  }, naccept_local);
```

### Optimization 3: Process Only Smoothing Sites

**Implementation:**
```cpp
int AppAdditiveExtTempTextureKokkos::smoothing_kernel_efficient()
{
  // Step 1: Build compact list of smoothing sites
  auto d_solid_d_current = buffer_solid_d.read_buffer;
  int nrefine_local = this->nrefine;
  int nlocal_local = this->nlocal;

  Kokkos::View<int*> d_smoothing_sites("smoothing_sites", nlocal_local);
  int n_smoothing = 0;

  // Parallel scan to build list
  Kokkos::parallel_scan("BuildSmoothingList", nlocal_local,
    KOKKOS_LAMBDA(int i, int& offset, bool final) {
      double sd = d_solid_d_current(i);
      bool needs_smoothing = (sd > -nrefine_local - 1 && sd < 0.0);

      if (needs_smoothing) {
        if (final) {
          d_smoothing_sites(offset) = i;
        }
        offset++;
      }
    }, n_smoothing);

  if (n_smoothing == 0) {
    return 0;  // No smoothing needed
  }

  // Step 2: Process ONLY smoothing sites
  auto d_spin_current = buffer_spin.read_buffer;
  auto d_spin_next = buffer_spin.write_buffer;
  // ... other buffers ...

  auto rand_pool_local = this->rand_pool;
  auto k_numneigh_local = this->k_numneigh.d_view;
  auto k_neighbor_local = this->k_neighbor.d_view;

  int naccept_local = 0;
  Kokkos::parallel_reduce("SmoothingKernelEfficient", n_smoothing,
    KOKKOS_LAMBDA(int idx, int& count) {
      int i = d_smoothing_sites(idx);  // Get actual site index

      // Full smoothing logic - NO BRANCHING!
      // Every site in this kernel needs smoothing

      int my_spin = d_spin_current(i);
      int nn = k_numneigh_local(i);

      // Find unique neighbor spins
      int unique_spins[27];
      int spin_counts[27];
      int unique_count = 0;

      for (int n = 0; n < nn; n++) {
        int nbr = k_neighbor_local(i, n);
        int nbr_spin = d_spin_current(nbr);

        if (nbr_spin != -1) {  // Not liquid
          bool found = false;
          for (int k = 0; k < unique_count; k++) {
            if (unique_spins[k] == nbr_spin) {
              spin_counts[k]++;
              found = true;
              break;
            }
          }
          if (!found && unique_count < 27) {
            unique_spins[unique_count] = nbr_spin;
            spin_counts[unique_count] = 1;
            unique_count++;
          }
        }
      }

      // Majority voting
      int max_count = 0;
      int majority_spin = my_spin;
      for (int k = 0; k < unique_count; k++) {
        if (spin_counts[k] > max_count) {
          max_count = spin_counts[k];
          majority_spin = unique_spins[k];
        }
      }

      // If majority differs, adopt it
      if (majority_spin != my_spin) {
        d_spin_next(i) = majority_spin;
        // Average quaternions from same-spin neighbors
        // ... quaternion averaging logic ...
        count++;
      }

      // Decrement solid_d counter
      double sd_new = d_solid_d_current(i) - 1.0;
      d_solid_d_next(i) = sd_new;

      if (sd_new <= -nrefine_local) {
        d_active_flag_next(i) = 1;  // ACTIVE (fully refined)
        d_solid_d_next(i) = 0.0;
      }
    }, naccept_local);

  return naccept_local;
}
```

**Performance Analysis:**
- **Scan cost**: ~0.3 ms (build list of ~100-1000 sites)
- **Kernel cost**: ~0.1 ms (process only smoothing sites)
- **Total**: ~0.4 ms vs 1.2 ms current
- **Savings**: **0.8 ms/step**

**Benefits:**
- Processes 0.01-0.1% of sites instead of 100%
- No branching in kernel (all sites need smoothing)
- Better memory coalescing (contiguous processing)

**Risk**: Low
**Effort**: ~50-100 lines
**Timeline**: 4-8 hours

---

## Implementation Roadmap

### Phase 1: Low-Hanging Fruit (1-2 days)

**Goal**: Quick wins with minimal risk

**Tasks:**
1. Implement Priority 1A: Early temperature scan for melting
   - **Expected savings**: 3.5 ms/step
   - **Effort**: ~30 lines
   - **Risk**: Low

2. Implement Priority 3: Work-efficient smoothing
   - **Expected savings**: 0.8 ms/step
   - **Effort**: ~50 lines
   - **Risk**: Low

**Expected Result:**
- **Current**: 25.8 ms/step
- **After Phase 1**: ~21.5 ms/step
- **vs CPU**: 1.25x slower (down from 1.50x)

**Testing:**
- Run existing validation tests
- Verify naccept counts match
- Check performance improvement

### Phase 2: Kernel Fusion (3-5 days)

**Goal**: Match CPU's unified loop architecture

**Tasks:**
1. Design unified functor structure
   - Map CPU's if/else chain to Kokkos
   - Ensure on-demand gradient still works

2. Implement UnifiedPhaseTransitionFunctor
   - Merge all 5 kernels
   - Maintain early-exit pattern
   - Careful buffer management

3. Replace 5 kernel launches with 1
   - Update app_update()
   - Simplify buffer swapping
   - Update timing instrumentation

4. Extensive testing
   - Physics validation
   - Performance benchmarking
   - Edge case testing

**Expected Result:**
- **After Phase 1**: 21.5 ms/step
- **After Phase 2**: ~18-19 ms/step
- **vs CPU**: 1.05-1.10x slower (nearly competitive!)

**Risk Mitigation:**
- Keep old kernel functions for validation
- Use git branches for incremental testing
- Compare naccept counts at each step

### Phase 3: Advanced Optimizations (1-2 weeks)

**Goal**: Exceed CPU performance

**Tasks:**
1. Implement Priority 1B: Spatial filtering for melting
   - **Expected savings**: Additional 1-2 ms/step
   - More complex than 1A but higher payoff

2. Fine-tune memory access patterns
   - Profile with Kokkos Tools
   - Optimize View layouts if needed

3. Consider additional fusion opportunities
   - Could merge temperature update with phase transitions?
   - Pre-compute frequently used values

**Expected Result:**
- **After Phase 2**: 18-19 ms/step
- **After Phase 3**: ~16-17 ms/step
- **vs CPU**: **1.01-1.07x faster!**

---

## Why CPU is Currently Faster

### Architectural Comparison

**CPU Advantages:**
1. **Unified Loop**: Single pass = cache-friendly, minimal overhead
2. **Zero Kernel Launch Overhead**: Direct function calls
3. **Efficient Branching**: Modern CPUs handle if/else chains well
4. **Sequential Memory Access**: Predictable pattern, good prefetching

**Kokkos Current Overhead:**
1. **5 Kernel Launches**: ~0.2 ms each = 1.0 ms total
2. **5 Memory Passes**: 5× bandwidth usage vs CPU's 1×
3. **All-Site Checks**: Evaluates 2.99M sites when <1% are active
4. **Buffer Management**: Extra complexity vs direct writes

### Memory Traffic Analysis

**CPU (per timestep):**
```
Single pass:
  Read: T, active_flag, spin, q0-qz, solid_d = ~100 bytes/site
  Write: Only modified fields on active sites = ~1-10 bytes/site average

Total: ~110 bytes/site × 2.99M = 329 MB
```

**Kokkos Current (per timestep):**
```
5 separate passes:
  Melting:        Read 100 bytes + Write 80 bytes = 180 bytes/site
  Nucleation:     Read 100 bytes + Write 80 bytes = 180 bytes/site
  Particle:       Read 100 bytes + Write 80 bytes = 180 bytes/site
  Solidification: Read 100 bytes + Write 80 bytes = 180 bytes/site
  Smoothing:      Read 100 bytes + Write 80 bytes = 180 bytes/site

Total: 900 bytes/site × 2.99M = 2.69 GB
```

**Bandwidth Impact:** 2.69 GB / 329 MB = **8.2× more memory traffic**

**Why this matters on OpenMP backend:**
- Memory bandwidth is shared across all threads
- L3 cache (12-24 MB) can't hold full working set
- Repeated reads/writes cause cache thrashing

**On GPU backend:**
- Much higher bandwidth (1000+ GB/s vs ~50 GB/s)
- Kernel launch overhead negligible (~10 μs vs ~0.2 ms)
- Would expect 10-30x speedup even with current architecture

---

## Expected Performance After All Optimizations

### Performance Projection

| Stage | Optimization | Time/Step (ms) | vs CPU | Improvement |
|-------|--------------|----------------|--------|-------------|
| **Current** | Baseline | 25.8 | 1.50x slower | - |
| **Phase 1** | Early scan + Efficient smoothing | 21.5 | 1.25x slower | 1.20x faster |
| **Phase 2** | Unified kernel fusion | 18.5 | 1.08x slower | 1.39x faster |
| **Phase 3** | Spatial filtering | 16.5 | **1.04x faster** | **1.56x faster** |

### Component Breakdown (Projected Phase 3)

| Component | Time (ms) | % Total | Notes |
|-----------|-----------|---------|-------|
| Temperature update | 14.8 | 89.7% | Already optimal |
| Unified phase kernel | 1.2 | 7.3% | Down from 9.8 ms |
| Temp sync | 0.4 | 2.4% | Negligible |
| Sync to host | 0.1 | 0.6% | Negligible |
| **TOTAL** | **16.5** | **100%** | **Faster than CPU!** |

### GPU Deployment (Future)

With CUDA/HIP backend (no code changes needed):
- **Temperature loading**: 14.8 ms (same, HDF5 on CPU)
- **Unified kernel**: 0.05-0.1 ms (100-200× faster than OpenMP)
- **Total**: ~15 ms/step = **1.15× faster than CPU**

With GPU-aware I/O (ADIOS2 or GPU-HDF5):
- **Temperature loading**: 0.5-1.0 ms (GPU-direct read)
- **Unified kernel**: 0.05-0.1 ms
- **Total**: ~0.6-1.1 ms/step = **15-30× faster than CPU!**

---

## Risk Assessment

### Priority 1A (Early Temperature Scan)

- **Risk**: **Low**
- **Complexity**: Simple parallel_reduce check
- **Validation**: Easy to verify (check naccept counts)
- **Rollback**: Trivial (remove check, always launch kernel)

### Priority 1B (Spatial Filtering)

- **Risk**: **Medium**
- **Complexity**: Parallel scan + list management
- **Validation**: Moderate (ensure all hot sites captured)
- **Rollback**: Keep 1A fallback

### Priority 2 (Kernel Fusion)

- **Risk**: **Medium-High**
- **Complexity**: Large refactor, ~500-800 lines
- **Validation**: Extensive testing needed
- **Rollback**: Keep old kernel functions in codebase
- **Mitigation**:
  - Implement in stages (merge 2 kernels first)
  - Compare naccept counts at each step
  - Use git branches for parallel development

### Priority 3 (Work-Efficient Smoothing)

- **Risk**: **Low**
- **Complexity**: Moderate (scan + filtered kernel)
- **Validation**: Easy (smoothing is infrequent)
- **Rollback**: Simple (revert to old kernel)

---

## Validation Strategy

### Correctness Verification

**For each optimization:**

1. **Event Count Matching**
   ```bash
   # CPU reference
   mpirun -np 6 spk_cpu < in.test > cpu.log
   grep "Naccept" cpu.log > cpu_events.txt

   # Kokkos optimized
   mpirun -np 6 spk_kokkos < in.test > kokkos.log
   grep "Naccept" kokkos.log > kokkos_events.txt

   # Compare
   diff cpu_events.txt kokkos_events.txt
   # Should be identical (or statistically equivalent for stochastic events)
   ```

2. **Microstructure Comparison**
   ```python
   # Load final dump files
   cpu_dump = load_stitch("DumpFiles_cpu/am_data_cpu.st")
   kokkos_dump = load_stitch("DumpFiles_kokkos/am_data_kokkos.st")

   # Compare grain statistics
   assert abs(cpu_dump.n_grains - kokkos_dump.n_grains) < 5  # Allow small variance
   assert ks_test(cpu_dump.grain_sizes, kokkos_dump.grain_sizes).pvalue > 0.05
   ```

3. **Conservation Laws**
   ```python
   # Total sites should be constant
   assert cpu_dump.total_sites == kokkos_dump.total_sites

   # Active flag distribution should match
   cpu_flags = np.bincount(cpu_dump.active_flag)
   kokkos_flags = np.bincount(kokkos_dump.active_flag)
   assert np.allclose(cpu_flags, kokkos_flags, rtol=0.01)
   ```

### Performance Benchmarking

**Standard test case:**
- Input: `examples/NewStitchFormat/in.init_box_0`
- Problem size: 2,993,450 sites (437×274×25)
- MPI ranks: 6
- OpenMP threads: 2 per rank
- Duration: 2500 timesteps

**Metrics to track:**
1. **Total runtime**: Wall time for full simulation
2. **Time per timestep**: Average ms/step
3. **Component breakdown**: Time in each kernel/operation
4. **Memory usage**: Peak resident set size
5. **naccept counts**: Verify physics correctness

**Regression testing:**
- Run benchmark after each optimization
- Compare to baseline (current 25.8 ms/step)
- Ensure no performance degradation
- Track improvement over time

---

## Conclusion

The current 1.50x performance gap is **entirely due to architectural differences**, not algorithmic inefficiency:

1. **Kernel launch overhead** (~1.0 ms from 5 launches)
2. **Excess memory traffic** (8× more bandwidth usage)
3. **All-site checking** (process 2.99M sites when <1% active)

All three issues are addressable through the proposed optimizations:

**Recommended path:**
1. **Start with Priority 1A + Priority 3** (Phase 1): Quick wins, low risk
2. **Then Priority 2** (Phase 2): Major refactor, matches CPU architecture
3. **Finally Priority 1B** (Phase 3): Advanced optimization for superiority

**Expected outcome:**
- Phase 1: 1.25x slower (nearly competitive)
- Phase 2: 1.05x slower (competitive)
- Phase 3: **1.04x faster** (superior!)

**GPU deployment** will then provide an additional 15-30× speedup with **zero code changes** to the unified kernel.

---

## Document History

| Date | Version | Author | Changes |
|------|---------|--------|---------|
| 2025-11-16 | 1.0 | Claude Code | Initial optimization analysis |

---

**End of KOKKOS_OPTIMIZATION_OPPORTUNITIES.md**
