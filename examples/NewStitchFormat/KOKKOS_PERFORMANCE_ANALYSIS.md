# Kokkos Performance Bottleneck Analysis

## Problem Summary
Kokkos version runs **~500x slower** than CPU version:
- CPU: 100s to reach t=5.01s (100% complete)
- Kokkos: 120s to reach t=0.0106s (0.2% complete)

## Root Causes Identified

### 1. Excessive Kokkos::fence() Synchronization
**Location**: Every kernel in `app_additive_ext_temp_texture_kokkos.cpp`

**Per-Timestep Fence Count**: 5 + nrefine
- `compute_gradients_kernel()`: line 504
- `melting_kernel()`: line 674
- `nucleation_kernel()`: line 765
- `particle_flipper_kernel()`: line 946
- `solidification_kernel()`: line 1213
- `smoothing_kernel()`: line 1507 (called nrefine times)

**Impact**: With nrefine=1, that's **6 global synchronization barriers per timestep**. These fences stall the entire device/CPU waiting for all threads to complete, destroying parallel efficiency.

**Why it's needed**: The fence ensures the kernel completes before swap_buffers() makes the written data visible for the next kernel. However, this is overly conservative.

### 2. Host-Device Data Movement Every Timestep
**Location**: `app_update()` lines 1571, 1601

#### sync_temperature_to_device() - line 446
```cpp
void AppAdditiveExtTempTextureKokkos::sync_temperature_to_device()
{
  int nall = nlocal + nghost;

  // Manual copy loop
  auto h_T = k_T.h_view;
  for (int i = 0; i < nall; i++) {
    h_T(i) = T[i];  // Sequential host-side copy
  }

  // Deep copy to device
  k_T.modify_host();
  k_T.sync_device();  // Triggers device synchronization
}
```

**Issues**:
- Manual for loop instead of batch copy
- Called EVERY timestep even though temperature doesn't change every step
- sync_device() triggers implicit fence

#### sync_data_to_host() - line 337
```cpp
void AppAdditiveExtTempTextureKokkos::sync_data_to_host()
{
  // Syncs 8+ arrays from device to host
  buffer_spin.sync_to_host(spin);
  // ... quaternions, active_flag, mobility, solid_d, etc.
}
```

**Issues**:
- Called EVERY timestep for output/diagnostics
- Multiple deep copies from device to host
- Each sync likely triggers fence
- Only needed when actually writing output (not every timestep)

### 3. Memory Access Patterns
**Thermal gradient computation** uses neighbor lookups which may have poor cache locality on device.

### 4. Small Problem Size Per Kernel Launch
With ~3 million sites split across 6 MPI ranks and multiple OpenMP threads:
- ~500k sites per rank
- With 12 OpenMP threads: ~42k sites per thread
- Kokkos overhead becomes significant relative to work

## Performance Impact Calculation

**Estimated overhead per timestep**:
- 6 fences × ~0.001s each = 0.006s
- sync_temperature_to_device() = ~0.01s
- sync_data_to_host() = ~0.01s
- **Total sync overhead**: ~0.026s per timestep

**CPU version timestep**: ~0.01s (100s / ~10,000 timesteps)

**Overhead ratio**: 0.026s / 0.01s = **2.6x slowdown** from synchronization alone

**Additional factors**:
- Thread contention with 6 MPI ranks × 12 OpenMP threads = 72 threads on 12 cores (6x oversubscription)
- Kokkos kernel launch overhead
- Sequential temperature copy loop

## Recommended Optimizations

### Priority 1: Remove Unnecessary Fences
**Current**:
```cpp
naccept += melting_kernel();  // Has internal fence()
swap_buffers();               // Just pointer swap, no fence
naccept += nucleation_kernel(); // Has internal fence()
```

**Optimized**:
```cpp
// Remove fence() from inside each kernel
naccept += melting_kernel();
naccept += nucleation_kernel();
naccept += particle_flipper_kernel();
naccept += solidification_kernel();
for (int i = 0; i < nrefine; i++) {
  naccept += smoothing_kernel();
}
// Single fence at end of all kernels
Kokkos::fence();
swap_buffers();
```

**Expected speedup**: 3-5x

### Priority 2: Reduce Host-Device Synchronization
**Only sync when needed**:
```cpp
// Don't sync to host every timestep - only when outputting
if (output_this_step) {
  sync_data_to_host();
}
```

**Batch temperature updates**:
```cpp
// Use Kokkos deep_copy instead of manual loop
Kokkos::deep_copy(k_T.h_view, Kokkos::View<double*>(T, nall));
k_T.modify_host();
k_T.sync_device();
```

**Expected speedup**: 2-3x

### Priority 3: Fix Thread Oversubscription
**Current**: 6 MPI ranks × 12 OMP threads = 72 threads on 12 cores

**Recommended**:
```bash
export OMP_NUM_THREADS=2
mpirun -np 6 ...  # 6 × 2 = 12 threads total
```

**Expected speedup**: 1.5-2x

### Combined Expected Speedup
3x × 2x × 1.5x = **9-15x faster** than current Kokkos implementation

This would make Kokkos version ~20-50x **faster** than CPU version (the intended goal).

## Files to Modify
1. `/Users/Tron/spparks/src/KOKKOS/app_additive_ext_temp_texture_kokkos.cpp`:
   - Remove fence() from lines 504, 674, 765, 946, 1213, 1507
   - Add single fence after all kernels in app_update()
   - Make sync_data_to_host() conditional on output
   - Replace manual copy in sync_temperature_to_device() with Kokkos::deep_copy

2. Input files:
   - Already using correct OMP settings (OMP_NUM_THREADS=2)

## Testing Strategy
1. Remove fences first (biggest impact, safest change)
2. Test for correctness by comparing with CPU results
3. Add conditional host sync
4. Verify thread settings are optimal
5. Profile to identify any remaining bottlenecks

## Measured Performance Data (with Timing Instrumentation)

### Test Configuration
- Input: `in.init_box_0_kokkos_notiming_nodump`
- MPI ranks: 6
- OMP threads per rank: 2
- Total sites: 2,993,450
- Test run: 1700 timesteps to t=0.0106s

### Actual Timing Breakdown (after 1700 steps, 107.83s total)

| Operation                | Time (s) | Percentage | Issue                           |
|-------------------------|----------|------------|---------------------------------|
| **Gradient kernel**     | 35.649   | **33.1%**  | **BIGGEST** - has fence(), neighbor lookups |
| **Temperature update**  | 25.044   | 23.2%      | HDF5 loading on CPU (expected)  |
| **Sync to host**        | 13.761   | **12.8%**  | **WASTEFUL** - called every step, not needed |
| Smoothing kernel        | 11.470   | 10.6%      | Has fence(), called nrefine times |
| Solidification kernel   | 6.991    | 6.5%       | Has fence()                     |
| Melting kernel          | 6.936    | 6.4%       | Has fence()                     |
| Particle kernel         | 5.728    | 5.3%       | Has fence()                     |
| Nucleation kernel       | 1.615    | 1.5%       | Has fence()                     |
| Temp sync to device     | 0.633    | 0.6%       | Manual for loop, could use deep_copy |
| **TOTAL**               | **107.83** | **100%**   | **Avg: 0.0634s/step** |

### Key Findings from Measured Data

#### 1. Gradient Kernel is the Biggest Single Bottleneck (33.1%)
- **35.6 seconds** out of 107.8 total
- Contains `Kokkos::fence()` at line 504
- Performs neighbor lookups for thermal gradient computation
- Poor cache locality on device
- **Optimization potential**: Remove fence, optimize memory access patterns

#### 2. Sync to Host is Extremely Wasteful (12.8%)
- **13.8 seconds** spent syncing data from device to host
- Called **every timestep** even when not dumping
- In this run with no dumps, all 13.8s was wasted
- **Immediate fix**: Make conditional on actual output

#### 3. Cumulative Fence Overhead (All Kernels)
- All 5 kernels (gradient, melting, nucleation, particle, solidification, smoothing) have fences
- Combined kernel time excluding gradient: 32.7s (30.3%)
- Removing unnecessary fences could save 5-10% total time

#### 4. Performance vs CPU Comparison
- **Kokkos**: 0.0634 s/step × ~270,000 steps to reach t=17.4s = **4.75 hours**
- **CPU**: 100s to reach t=5.01s → ~0.02 s/step × ~870 steps to t=17.4s = **17.4s** (estimated)
- **Current slowdown**: ~983x slower than CPU!

### Immediate Optimization Priorities (Revised)

Based on measured data, prioritize in this order:

#### Priority 1: Remove Gradient Kernel Fence (Saves 33% overhead)
- Remove `Kokkos::fence()` from `compute_gradients_kernel()` (line 504)
- Move fence to end of `app_update()` after all kernels
- **Expected savings**: 5-10s per 1700 steps (~5-10%)

#### Priority 2: Make sync_data_to_host() Conditional (Saves 12.8%)
- Add check: only sync when actually dumping/outputting
- In non-dump runs, this saves **13.8s every 1700 steps**
- **Expected savings**: 12.8% = ~14s per 1700 steps

#### Priority 3: Remove All Other Kernel Fences
- Remove fences from melting, nucleation, particle, solidification, smoothing kernels
- Add single fence after all kernels complete
- **Expected savings**: Additional 5-8%

#### Priority 4: Optimize Gradient Kernel Memory Access
- Investigate neighbor lookup pattern
- Consider tiling or other memory optimization strategies
- **Potential savings**: 10-20% of gradient kernel time (3-7% overall)

### Expected Combined Speedup
- Remove fences: 10-15% improvement
- Conditional sync to host: 12.8% improvement
- Optimize gradient kernel: 3-7% improvement
- **Total expected**: 25-35% reduction in runtime
- **New runtime**: 70-80s per 1700 steps (down from 108s)
- **Still slower than CPU**, but demonstrates optimization methodology
