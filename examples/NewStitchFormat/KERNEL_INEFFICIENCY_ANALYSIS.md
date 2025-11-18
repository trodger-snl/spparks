# Kokkos Kernel Inefficiency Analysis

## Executive Summary

After implementing on-demand gradients (1.66x speedup), investigation reveals **additional optimization opportunities** in how kernels are applied. The Kokkos version calls kernels unconditionally for all sites, while the CPU version uses conditional logic to process only relevant sites.

## Key Findings

### 1. Smoothing Kernel Inefficiency (HIGHEST IMPACT)

**Problem**: Kokkos calls smoothing kernel multiple times per timestep for ALL sites

**Kokkos Implementation** (`app_additive_ext_temp_texture_kokkos.cpp:1639-1642`):
```cpp
for (int sweep = 0; sweep < nrefine; sweep++) {  // nrefine = 2
  naccept += smoothing_kernel();  // Iterates over ALL 2.99M sites
  swap_buffers();
}
```

**CPU Implementation** (`app_additive_ext_temp_texture.cpp:401-409`):
```cpp
for (int i=0; i<nlocal; i++) {
  // Only process sites in smoothing range once per timestep
  else if(solid_d[i] < 0 && solid_d[i] > -nrefine -1 && active_flag[i] == 3) {
    smooth_site(i);
    solid_d[i]--;  // Counts down over multiple timesteps
  }
}
```

**Impact**:
- **Kokkos**: 2 full passes over 2.99M sites = **5.98M site checks/timestep**
- **CPU**: 1 pass with conditional check = **~few thousand active sites/timestep**
- **Performance**: 18.7% of Kokkos runtime (6.3 ms) vs 0.2% CPU runtime (0.04 ms)
- **Overhead**: **150x slower than CPU**

**Root Cause**: Misunderstanding of smoothing semantics
- **Kokkos interprets**: "Do nrefine sweeps of smoothing per timestep"
- **CPU implements**: "Each site smooths for nrefine timesteps total" (solid_d counts -1 → -nrefine)

**Optimization Potential**: ~6 ms/step → could reduce to ~0.1 ms/step

---

### 2. Kernel Launch Pattern Inefficiency

All Kokkos kernels are launched unconditionally every timestep:

```cpp
// Phase transition kernels (lines 1609-1644)
naccept += melting_kernel();          // Line 1610
naccept += nucleation_kernel();       // Line 1617
naccept += particle_flipper_kernel(); // Line 1624
naccept += solidification_kernel();   // Line 1631
for (int sweep = 0; sweep < nrefine; sweep++) {
  naccept += smoothing_kernel();      // Line 1640
}
```

**CPU conditional logic** (lines 368-410):
```cpp
for (int i=0; i<nlocal; i++) {
  if (active_flag[i] == 5) continue;  // Skip void

  if (T[i] >= tl && active_flag[i] != 2) {
    // MELTING: Only for sites crossing liquidus
    melting_operation(i);
  }
  else if (active_flag[i] == 2 && T[i] <= tl) {
    // MUSHY PHASE: nucleation + solidification
    // Only for liquid sites below liquidus
    mushy_phase(i, ranapp);
  }
  else if (solid_d[i] < 0 && solid_d[i] > -nrefine-1 && active_flag[i] == 3) {
    // SMOOTHING: Only for sites in refinement range
    smooth_site(i);
    solid_d[i]--;
  }
}
```

**Analysis**:
- **CPU**: Single pass with branching (cache-friendly, minimal overhead)
- **Kokkos**: Multiple kernel launches with early-exit logic inside each kernel

**Why Kokkos can't easily match CPU**:
- Separate kernels enable GPU parallelism and avoid race conditions
- Early-exit in kernel is correct approach for Kokkos
- BUT: smoothing loop is incorrect

---

## Detailed Kernel Comparison

### Melting Kernel

| Aspect | CPU | Kokkos |
|--------|-----|--------|
| **When called** | Sites with `T[i] >= tl && active_flag[i] != 2` | Every timestep, all sites |
| **How filtered** | Outer loop conditional | Early return inside kernel |
| **Performance** | 0.1% (0.01 ms/step) | 13.1% (4.4 ms/step) |
| **Status** | ⚠️ Kokkos 400x slower | Kernel overhead + check all sites |

### Nucleation + Solidification (Mushy Phase in CPU)

| Aspect | CPU | Kokkos |
|--------|-----|--------|
| **When called** | Sites with `active_flag[i] == 2 && T[i] <= tl` | Nucleation + solidification separate, every timestep |
| **How filtered** | Outer loop conditional | Early return inside each kernel |
| **Performance (total)** | 3.5% (0.6 ms/step) | Nucleation: 2.7% (0.9 ms) + Solidification: 12.0% (4.0 ms) = **14.7%** |
| **Status** | ⚠️ Kokkos 8.5x slower | Higher overhead from 2 kernels vs 1 function |

### Smoothing

| Aspect | CPU | Kokkos |
|--------|-----|--------|
| **When called** | Sites with `solid_d[i] ∈ (-nrefine-1, 0)` once/timestep | **ALL sites, nrefine times/timestep** |
| **How filtered** | Outer loop conditional | Early return inside kernel |
| **Performance** | 0.2% (0.04 ms/step) | **18.7% (6.3 ms/step)** |
| **Status** | 🔴 **CRITICAL: Kokkos 150x slower** | **WRONG: 2x too many kernel launches** |

---

## Root Cause Analysis

### Smoothing Loop Misinterpretation

**Physical meaning**:
- When a site solidifies, it needs "refinement" to smooth grain boundaries
- `nrefine` = number of MC steps for smoothing
- `solid_d` tracks smoothing progress: -1 (just solidified) → -nrefine (fully refined) → 0 (active grain)

**CPU implementation**:
```cpp
// Each timestep, decrement solid_d ONCE for sites in smoothing range
if (solid_d[i] < 0 && solid_d[i] > -nrefine-1) {
  smooth_site(i);
  solid_d[i]--;  // -1 → -2 → -3 → ... → 0 (over nrefine timesteps)
}
```
**Behavior**: Site smooths over `nrefine` **timesteps**

**Kokkos implementation**:
```cpp
// Every timestep, call smoothing nrefine TIMES
for (int sweep = 0; sweep < nrefine; sweep++) {
  smoothing_kernel();  // Each call processes ALL sites
}
```
**Behavior**: Every site checked `nrefine` times **per timestep**

**Impact**: **2x too many kernel launches**, plus each launch checks ALL sites instead of just active ones

---

## Optimization Recommendations

### Priority 1: Fix Smoothing Loop (HIGHEST IMPACT)

**Change** (in `app_additive_ext_temp_texture_kokkos.cpp:1639-1644`):

**Current (WRONG)**:
```cpp
for (int sweep = 0; sweep < nrefine; sweep++) {
  naccept += smoothing_kernel();
  swap_buffers();
}
```

**Corrected**:
```cpp
// Call smoothing kernel ONCE per timestep (like CPU)
// The kernel itself handles solid_d decrement over multiple timesteps
naccept += smoothing_kernel();
swap_buffers();
```

**AND** modify smoothing kernel to match CPU behavior:
```cpp
// In smoothing_kernel(), after MC move logic (around line 1309-1320):
// Decrement solid_d ONCE per timestep (not per sweep)
if (solid_d_i < 0.0 && solid_d_i > -nrefine_local - 1) {
  d_solid_d_next(i) = solid_d_i - 1.0;  // Count down: -1 → -2 → ... → -nrefine
  // If reached -nrefine, next timestep goes to 0 (becomes ACTIVE)
  if (d_solid_d_next(i) <= -nrefine_local) {
    d_active_flag_next(i) = 1;  // ACTIVE
    d_solid_d_next(i) = 0.0;
  }
}
```

**Expected Impact**:
- Remove 1 of 2 smoothing kernel calls per timestep
- Reduce smoothing time: 6.3 ms → ~3 ms (still higher than CPU due to all-site iteration)
- **Estimated speedup**: ~10% overall (6.3 ms / 2 ÷ 33.6 ms ≈ 9.4%)

### Priority 2: Investigate Other Kernels

All other kernels show significant overhead vs CPU:

| Kernel | CPU (ms) | Kokkos (ms) | Slowdown | Reason |
|--------|----------|-------------|----------|---------|
| Melting | 0.01 | 4.4 | 400x | Checks all sites vs few crossing liquidus |
| Nucleation | ~0.3* | 0.9 | 3x | Kernel overhead |
| Solidification | ~0.3* | 4.0 | 13x | More complex than CPU mushy_phase? |

*Estimated: CPU's "mushy phase" (0.6 ms) includes nucleation + solidification + gradient

**Analysis needed**:
1. How many sites actually undergo melting per timestep?
2. Can melting be skipped if no sites above liquidus?
3. Is solidification doing extra work vs CPU's mushy_phase?

### Priority 3: Kernel Fusion (Future Work)

Combine related kernels to reduce launch overhead:
- Merge nucleation + solidification (they're both mushy-zone operations)
- Consider melting + mushy-zone fusion if branching overhead is acceptable

---

## Expected Performance After Fixes

### After Smoothing Fix:
- **Current**: 33.6 ms/step
- **Remove 1 smoothing call**: 33.6 - 3.15 = **30.45 ms/step**
- **Speedup**: 1.10x faster
- **vs CPU**: 30.45 / 17.2 = **1.77x slower** (improved from 1.95x)

### After All Optimizations (Hypothetical):
If we could match CPU overhead for melting (4.4 → 0.01 ms) and other kernels:
- Temperature: 14.1 ms (comparable to CPU)
- Kernels: ~1 ms (if matching CPU's 0.6 ms mushy + 0.04 smoothing)
- **Total**: ~15 ms/step → **competitive with CPU!**

But this is unrealistic on CPU backend due to inherent kernel launch overhead.

**Realistic target**: **25-28 ms/step** after smoothing fix + minor optimizations

---

## Conclusions

1. ✅ **Gradient optimization successful** (1.66x speedup achieved)
2. 🔴 **Smoothing loop is WRONG** - calls kernel 2x per timestep instead of 1x
3. ⚠️ **All kernels have overhead** vs CPU's conditional single-pass approach
4. **Next action**: Fix smoothing loop (Priority 1) for ~10% speedup
5. **Long-term**: GPU backends will overcome kernel overhead with massive parallelism

---

**Recommendation**: Fix the smoothing loop first (simple change, big impact), then investigate why other kernels are slower than expected.
