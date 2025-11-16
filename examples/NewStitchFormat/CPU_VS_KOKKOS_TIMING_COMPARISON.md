# CPU vs Kokkos Performance Comparison

## Test Configuration
- **MPI ranks**: 6
- **OMP threads per rank**: 2
- **Total sites**: 2,993,450
- **Lattice**: sc/26n (simple cubic, 26 neighbors)
- **Input**: Reduced temperature AM simulation with HDF5 temperature source

## Timing Results Summary

### CPU Version (after 4300 steps, 73.88s total)
| Operation              | Time (s) | Percentage | Time/Step (ms) |
|------------------------|----------|------------|----------------|
| **Temperature update** | 65.330   | **88.4%**  | 15.19          |
| Fast-forward logic     | 0.000    | 0.0%       | 0.00           |
| Melting                | 0.057    | 0.1%       | 0.01           |
| **Mushy phase**        | 2.565    | **3.5%**   | 0.60           |
| Smoothing              | 0.172    | 0.2%       | 0.04           |
| Communication          | 5.752    | 7.8%       | 1.34           |
| **TOTAL**              | **73.88**| **100%**   | **17.2**       |

### Kokkos Version (after 900 steps, 50.10s total - optimized)
| Operation              | Time (s) | Percentage | Time/Step (ms) |
|------------------------|----------|------------|----------------|
| **Temperature update** | 23.2     | ~46%       | 25.8           |
| Temp sync to device    | 0.3      | 0.6%       | 0.3            |
| **Gradient kernel**    | 14.0     | ~28%       | 15.6           |
| Melting kernel         | 3.4      | ~7%        | 3.8            |
| Nucleation kernel      | 0.8      | ~2%        | 0.9            |
| Particle kernel        | 2.8      | ~6%        | 3.1            |
| Solidification kernel  | 3.4      | ~7%        | 3.8            |
| Smoothing kernel       | 2.2      | ~4%        | 2.4            |
| Sync to host           | 0.001    | 0.0%       | 0.00           |
| **TOTAL**              | **50.1** | **100%**   | **55.7**       |

**Note**: Kokkos percentages estimated from earlier 1700-step run scaled to 900 steps.

## Key Performance Insights

### Overall Speed Comparison
- **CPU**: 17.2 ms/step
- **Kokkos (optimized)**: 55.7 ms/step
- **Kokkos slowdown**: **3.2x slower** than CPU

### Why is Kokkos Slower?

#### 1. **Gradient Computation is the Main Bottleneck (28% of Kokkos time)**
   - **CPU**: Gradients computed **on-demand** only when needed in `mushy_phase()` via `normal_finder()`
     - Only called for sites undergoing nucleation/solidification
     - Minimal overhead: ~0.6ms/step total for all mushy phase operations

   - **Kokkos**: Gradients computed **for ALL sites** every timestep in `compute_gradients_kernel()`
     - Line 504 in app_additive_ext_temp_texture_kokkos.cpp
     - ~15.6ms/step (26x more expensive than CPU's mushy phase)
     - **Root cause**: Unnecessary work - most sites don't need gradients

   **Optimization potential**: Make gradient computation on-demand like CPU version.

#### 2. **Temperature Update Overhead (46% vs 88%)**
   - **CPU**: 88.4% spent in HDF5 loading (15.2ms/step)
   - **Kokkos**: 46% spent in HDF5 loading (25.8ms/step)
   - **Kokkos is 1.7x slower** at temperature loading
   - Likely due to extra DualView sync overhead

#### 3. **Kernel Overhead from Fine-Grained Parallelism**
   - CPU version has simple sequential loops with tight cache locality
   - Kokkos version has 6+ kernel launches per timestep with overhead:
     - Kernel launch latency
     - Thread synchronization
     - Memory fence operations (reduced but still present)

#### 4. **What CPU Does Well**
   - **On-demand computation**: Only compute what's needed, when it's needed
   - **Cache locality**: Sequential loops over contiguous arrays
   - **Minimal synchronization**: No global fences, just local function calls
   - **Combined operations**: Mushy phase combines nucleation + solidification + gradient in one pass

#### 5. **What Kokkos Could Improve**
   1. **Make gradient computation on-demand** (could save ~15ms/step = 27% speedup)
   2. **Combine kernels**: Merge nucleation + solidification into single kernel
   3. **Optimize temperature loading**: Reduce DualView sync overhead
   4. **Conditional gradient updates**: Only compute gradients for sites in mushy zone

## Expected Performance After On-Demand Gradients

If we make gradients on-demand in Kokkos:
- Remove: 15.6 ms/step (gradient kernel)
- Add: ~0.6 ms/step (on-demand gradient in solidification kernel)
- **New Kokkos time**: 55.7 - 15.6 + 0.6 = **40.7 ms/step**
- **Speedup vs current**: 1.37x faster
- **Comparison to CPU**: 40.7/17.2 = **2.4x slower** (still slower, but much better)

## Why Kokkos Can't Match CPU Speed (Yet)

Even with on-demand gradients, Kokkos will likely remain ~2x slower because:

1. **Kernel launch overhead**: Each kernel has setup/teardown cost
2. **Memory access patterns**: Kokkos views have bounds checking and indirection
3. **Thread overhead**: Managing 12 OpenMP threads has overhead vs simple sequential loops
4. **HDF5 loading**: Current implementation is slower (25.8ms vs 15.2ms)

**Future optimization directions**:
- GPU acceleration: Move to CUDA/HIP backend (could be 5-50x faster)
- Fused kernels: Combine multiple operations into single kernel
- Memory layout optimization: Improve cache locality in Kokkos views
- Asynchronous HDF5: Overlap temperature loading with computation

## Testing Methodology

### CPU Test
```bash
cd /Users/Tron/spparks/examples/NewStitchFormat
export OMP_PROC_BIND=spread OMP_PLACES=threads OMP_NUM_THREADS=2
mpirun -np 6 ../../src/build/spparks -in in.init_box_0_cpu_timing
```

### Kokkos Test
```bash
cd /Users/Tron/spparks/examples/NewStitchFormat
export OMP_PROC_BIND=spread OMP_PLACES=threads OMP_NUM_THREADS=2
mpirun -np 6 ../../src/build/spparks -in in.init_box_0_kokkos_timing
```

Both tests run to t=0.02s with identical physics parameters, no dumps.

## Conclusions

1. **CPU version is well-optimized** - 88% time in unavoidable HDF5 I/O
2. **Kokkos gradient kernel is wasteful** - computes for all sites, not just active ones
3. **On-demand gradients would close the gap significantly** - from 3.2x to 2.4x slower
4. **Kokkos overhead is inherent** - kernel launches, synchronization, memory indirection
5. **Future work**: GPU backends could make Kokkos 10-100x faster than CPU

**Recommendation**: Implement on-demand gradient computation in Kokkos version to match CPU architecture.
