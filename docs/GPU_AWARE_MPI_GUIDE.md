# GPU-Aware MPI Implementation Guide for SPPARKS Kokkos

## Overview

This document provides a complete roadmap for implementing GPU-aware MPI in SPPARKS Kokkos, from initial benchmarking through full production implementation. The implementation targets:

- **AMD MI250X** (Frontier, Crusher) - ROCm + Cray MPICH with GTL
- **NVIDIA H100** clusters - CUDA + OpenMPI/MVAPICH with UCX or NCCL

---

# Stage 1: Benchmarking Current Implementation

## Objective
Quantify the current host-staging overhead to establish baseline and justify GPU-aware MPI development.

## 1.1 Instrumentation Points

Add timing instrumentation to `src/KOKKOS/app_lattice_kokkos.cpp`:

```cpp
// Add these timing variables to AppLatticeKokkos class
double time_sync_to_host = 0.0;
double time_sync_to_device = 0.0;
double time_mpi_comm = 0.0;
double time_gpu_compute = 0.0;
int sync_count = 0;

// In iterate_rejection(), instrument the sync calls:
void AppLatticeKokkos::iterate_rejection(double stoptime) {
  // Reset timers
  time_sync_to_host = time_sync_to_device = time_mpi_comm = time_gpu_compute = 0.0;
  sync_count = 0;

  while (!done) {
    for (int iset = 0; iset < nset_loop; iset++) {
      if (nprocs > 1) {
        // Time sync to host
        Kokkos::fence();
        double t0 = MPI_Wtime();
        sync_app_data_to_host();
        double t1 = MPI_Wtime();
        time_sync_to_host += (t1 - t0);

        // Time MPI communication
        if (sectorflag) comm->sector(iset);
        else comm->all();
        double t2 = MPI_Wtime();
        time_mpi_comm += (t2 - t1);

        // Time sync to device
        sync_app_data_to_device();
        Kokkos::fence();
        double t3 = MPI_Wtime();
        time_sync_to_device += (t3 - t2);

        sync_count++;
      }

      // Time GPU computation
      double t4 = MPI_Wtime();
      for (int icolor = 0; icolor < kokkos_ncolors; icolor++) {
        sweep_color_kokkos(icolor, color_start, color_count);
        Kokkos::fence();
      }
      double t5 = MPI_Wtime();
      time_gpu_compute += (t5 - t4);
    }
  }

  // Report timing breakdown
  if (me == 0) {
    printf("\n=== GPU-MPI Timing Breakdown ===\n");
    printf("Sync to host:   %.4f s (%.1f%%)\n", time_sync_to_host,
           100.0 * time_sync_to_host / (time_sync_to_host + time_sync_to_device + time_mpi_comm + time_gpu_compute));
    printf("MPI comm:       %.4f s (%.1f%%)\n", time_mpi_comm,
           100.0 * time_mpi_comm / (time_sync_to_host + time_sync_to_device + time_mpi_comm + time_gpu_compute));
    printf("Sync to device: %.4f s (%.1f%%)\n", time_sync_to_device,
           100.0 * time_sync_to_device / (time_sync_to_host + time_sync_to_device + time_mpi_comm + time_gpu_compute));
    printf("GPU compute:    %.4f s (%.1f%%)\n", time_gpu_compute,
           100.0 * time_gpu_compute / (time_sync_to_host + time_sync_to_device + time_mpi_comm + time_gpu_compute));
    printf("Total syncs:    %d\n", sync_count);
    printf("================================\n\n");
  }
}
```

## 1.2 Benchmark Test Cases

Create benchmark input files with varying problem sizes:

**Small (baseline, communication-bound):**
```
# in.potts.benchmark.small
dimension 3
lattice sc/26n 1.0
region box block 0 100 0 100 0 100
create_box box
create_sites box
set site range 1 100
sweep raster
sector yes
temperature 0.5
solve_style tree
run 1000
```

**Medium (balanced):**
```
# in.potts.benchmark.medium
dimension 3
lattice sc/26n 1.0
region box block 0 200 0 200 0 200
create_box box
create_sites box
set site range 1 100
sweep raster
sector yes
temperature 0.5
solve_style tree
run 1000
```

**Large (compute-bound):**
```
# in.potts.benchmark.large
dimension 3
lattice sc/26n 1.0
region box block 0 400 0 400 0 400
create_box box
create_sites box
set site range 1 100
sweep raster
sector yes
temperature 0.5
solve_style tree
run 500
```

## 1.3 Benchmark Execution Scripts

**For Frontier (AMD MI250X):**
```bash
#!/bin/bash
#SBATCH -A <project>
#SBATCH -J spparks_bench
#SBATCH -t 00:30:00
#SBATCH -N 2
#SBATCH -p batch

module load PrgEnv-amd
module load rocm
module load cray-mpich

# 8 GPUs per node (4 GCDs × 2 nodes)
export MPICH_GPU_SUPPORT_ENABLED=1

for size in small medium large; do
  for ngpu in 1 2 4 8 16; do
    echo "=== $size, $ngpu GPUs ==="
    srun -n $ngpu --gpus-per-task=1 --gpu-bind=closest \
      ./spparks -k on g 1 -sf kk < in.potts.benchmark.$size
  done
done
```

**For NVIDIA H100 cluster:**
```bash
#!/bin/bash
#SBATCH -J spparks_bench
#SBATCH -t 00:30:00
#SBATCH -N 2
#SBATCH --gres=gpu:4

module load cuda/12
module load openmpi/4.1

export UCX_TLS=rc,cuda_copy,cuda_ipc,gdr_copy
export UCX_RNDV_SCHEME=put_zcopy

for size in small medium large; do
  for ngpu in 1 2 4 8; do
    echo "=== $size, $ngpu GPUs ==="
    mpirun -n $ngpu --map-by ppr:4:node:pe=1 --bind-to core \
      ./spparks -k on g 1 -sf kk < in.potts.benchmark.$size
  done
done
```

## 1.4 Expected Baseline Results

| Problem Size | Sites | Sync Overhead (est.) | Target Improvement |
|-------------|-------|---------------------|-------------------|
| Small (100³) | 1M | 30-50% | 2-3x iteration time |
| Medium (200³) | 8M | 15-25% | 1.3-1.5x |
| Large (400³) | 64M | 5-15% | 1.1-1.2x |

**Key metrics to record:**
- `time_sync_to_host` / total time (%)
- `time_sync_to_device` / total time (%)
- Bytes transferred per sync
- Sync frequency (syncs per second)

---

# Stage 2: Proof of Concept

## Objective
Implement minimal GPU-aware MPI for `comm->all()` to validate approach and measure speedup.

## 2.1 GPU-Aware MPI Detection

**Create: `src/KOKKOS/kokkos_mpi.h`**

```cpp
#ifndef SPPARKS_KOKKOS_MPI_H
#define SPPARKS_KOKKOS_MPI_H

#include <mpi.h>
#include <Kokkos_Core.hpp>

namespace SPPARKS_NS {

class KokkosMPI {
public:
  static bool gpu_aware_mpi_available;
  static bool gpu_aware_mpi_enabled;

  static void detect_gpu_aware_mpi() {
    gpu_aware_mpi_available = false;
    gpu_aware_mpi_enabled = false;

#if defined(KOKKOS_ENABLE_CUDA)
    // CUDA-aware MPI detection
    #if defined(MPIX_CUDA_AWARE_SUPPORT)
      if (MPIX_Query_cuda_support() == 1) {
        gpu_aware_mpi_available = true;
      }
    #endif

    // OpenMPI detection
    #if defined(OMPI_HAVE_MPI_EXT_CUDA) && OMPI_HAVE_MPI_EXT_CUDA
      gpu_aware_mpi_available = true;
    #endif

#elif defined(KOKKOS_ENABLE_HIP)
    // ROCm/HIP GPU-aware MPI detection
    // Cray MPICH with GTL
    #if defined(MPICH_VERSION) && defined(__HIP_PLATFORM_AMD__)
      // Check MPICH_GPU_SUPPORT_ENABLED at runtime
      const char* env = getenv("MPICH_GPU_SUPPORT_ENABLED");
      if (env && strcmp(env, "1") == 0) {
        gpu_aware_mpi_available = true;
      }
    #endif

    // OpenMPI with UCX ROCm support
    #if defined(OMPI_MAJOR_VERSION) && defined(__HIP_PLATFORM_AMD__)
      const char* ucx_tls = getenv("UCX_TLS");
      if (ucx_tls && strstr(ucx_tls, "rocm") != nullptr) {
        gpu_aware_mpi_available = true;
      }
    #endif
#endif

    // User override via environment variable
    const char* override_env = getenv("SPPARKS_GPU_AWARE_MPI");
    if (override_env) {
      if (strcmp(override_env, "0") == 0) {
        gpu_aware_mpi_enabled = false;
      } else if (strcmp(override_env, "1") == 0) {
        gpu_aware_mpi_enabled = gpu_aware_mpi_available;
      }
    } else {
      gpu_aware_mpi_enabled = gpu_aware_mpi_available;
    }
  }

  static void print_status(int me) {
    if (me == 0) {
      printf("GPU-aware MPI: %s (available: %s, enabled: %s)\n",
             gpu_aware_mpi_enabled ? "ACTIVE" : "DISABLED",
             gpu_aware_mpi_available ? "yes" : "no",
             gpu_aware_mpi_enabled ? "yes" : "no");
    }
  }
};

// Static member initialization
bool KokkosMPI::gpu_aware_mpi_available = false;
bool KokkosMPI::gpu_aware_mpi_enabled = false;

} // namespace SPPARKS_NS

#endif
```

## 2.2 Device Buffer Pack/Unpack Kernels

**Create: `src/KOKKOS/comm_kokkos_kernels.h`**

```cpp
#ifndef SPPARKS_COMM_KOKKOS_KERNELS_H
#define SPPARKS_COMM_KOKKOS_KERNELS_H

#include <Kokkos_Core.hpp>

namespace SPPARKS_NS {

// Pack integer data from scattered sites into contiguous buffer
template<class DeviceType>
struct PackIntFunctor {
  using view_type = Kokkos::View<int*, DeviceType>;

  view_type src;      // Source array (e.g., spin values)
  view_type indices;  // Site indices to pack
  view_type buffer;   // Destination buffer
  int offset;         // Offset in buffer

  PackIntFunctor(view_type src_, view_type indices_, view_type buffer_, int offset_)
    : src(src_), indices(indices_), buffer(buffer_), offset(offset_) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(const int i) const {
    buffer(offset + i) = src(indices(i));
  }
};

// Unpack integer data from contiguous buffer to scattered sites
template<class DeviceType>
struct UnpackIntFunctor {
  using view_type = Kokkos::View<int*, DeviceType>;

  view_type dest;     // Destination array
  view_type indices;  // Site indices to unpack to
  view_type buffer;   // Source buffer
  int offset;         // Offset in buffer

  UnpackIntFunctor(view_type dest_, view_type indices_, view_type buffer_, int offset_)
    : dest(dest_), indices(indices_), buffer(buffer_), offset(offset_) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(const int i) const {
    dest(indices(i)) = buffer(offset + i);
  }
};

// Pack multiple integer arrays (ninteger arrays per site)
template<class DeviceType>
struct PackMultiIntFunctor {
  using view_type = Kokkos::View<int**, DeviceType>;
  using idx_view = Kokkos::View<int*, DeviceType>;
  using buf_view = Kokkos::View<int*, DeviceType>;

  view_type src;      // Source arrays [narray][nsite]
  idx_view indices;   // Site indices to pack
  buf_view buffer;    // Destination buffer
  int narray;         // Number of arrays
  int nsites;         // Number of sites to pack
  int offset;         // Offset in buffer

  KOKKOS_INLINE_FUNCTION
  void operator()(const int i) const {
    int isite = i % nsites;
    int iarray = i / nsites;
    buffer(offset + i) = src(iarray, indices(isite));
  }
};

// Unpack multiple integer arrays
template<class DeviceType>
struct UnpackMultiIntFunctor {
  using view_type = Kokkos::View<int**, DeviceType>;
  using idx_view = Kokkos::View<int*, DeviceType>;
  using buf_view = Kokkos::View<int*, DeviceType>;

  view_type dest;
  idx_view indices;
  buf_view buffer;
  int narray;
  int nsites;
  int offset;

  KOKKOS_INLINE_FUNCTION
  void operator()(const int i) const {
    int isite = i % nsites;
    int iarray = i / nsites;
    dest(iarray, indices(isite)) = buffer(offset + i);
  }
};

// Double precision versions
template<class DeviceType>
struct PackDoubleFunctor {
  using view_type = Kokkos::View<double*, DeviceType>;

  view_type src;
  Kokkos::View<int*, DeviceType> indices;
  view_type buffer;
  int offset;

  KOKKOS_INLINE_FUNCTION
  void operator()(const int i) const {
    buffer(offset + i) = src(indices(i));
  }
};

template<class DeviceType>
struct UnpackDoubleFunctor {
  using view_type = Kokkos::View<double*, DeviceType>;

  view_type dest;
  Kokkos::View<int*, DeviceType> indices;
  view_type buffer;
  int offset;

  KOKKOS_INLINE_FUNCTION
  void operator()(const int i) const {
    dest(indices(i)) = buffer(offset + i);
  }
};

} // namespace SPPARKS_NS

#endif
```

## 2.3 Proof-of-Concept CommLatticeKokkos

**Create: `src/KOKKOS/comm_lattice_kokkos.h`**

```cpp
#ifndef SPPARKS_COMM_LATTICE_KOKKOS_H
#define SPPARKS_COMM_LATTICE_KOKKOS_H

#include "comm_lattice.h"
#include "kokkos_mpi.h"
#include <Kokkos_Core.hpp>

namespace SPPARKS_NS {

class CommLatticeKokkos : public CommLattice {
public:
  CommLatticeKokkos(class SPPARKS *);
  ~CommLatticeKokkos();

  void init(int, int, int, int*, int*) override;
  void all() override;
  void all_reverse() override;
  void sector(int) override;
  void reverse_sector(int) override;

private:
  bool use_gpu_aware;

  // Device buffers for GPU-aware MPI
  Kokkos::View<int*> d_send_buf_int;
  Kokkos::View<int*> d_recv_buf_int;
  Kokkos::View<double*> d_send_buf_double;
  Kokkos::View<double*> d_recv_buf_double;

  // Device-side send/recv index lists
  Kokkos::View<int*> d_send_indices;
  Kokkos::View<int*> d_send_offsets;
  Kokkos::View<int*> d_recv_indices;
  Kokkos::View<int*> d_recv_offsets;

  // Maximum buffer sizes
  int max_send_int, max_recv_int;
  int max_send_double, max_recv_double;

  void setup_device_buffers();
  void perform_swap_gpu_aware(Swap* swap);
  void perform_swap_host_staged(Swap* swap);  // Fallback
};

} // namespace SPPARKS_NS

#endif
```

**Create: `src/KOKKOS/comm_lattice_kokkos.cpp`** (core implementation)

```cpp
#include "comm_lattice_kokkos.h"
#include "app_lattice.h"
#include "domain.h"
#include "memory.h"
#include "error.h"
#include "comm_kokkos_kernels.h"

using namespace SPPARKS_NS;

CommLatticeKokkos::CommLatticeKokkos(SPPARKS *spk) : CommLattice(spk)
{
  use_gpu_aware = KokkosMPI::gpu_aware_mpi_enabled;
  max_send_int = max_recv_int = 0;
  max_send_double = max_recv_double = 0;
}

CommLatticeKokkos::~CommLatticeKokkos()
{
  // Kokkos Views automatically deallocate
}

void CommLatticeKokkos::init(int ninteger_caller, int ndouble_caller,
                             int delpropensity_caller,
                             int* iarray, int* darray)
{
  // Call parent init to set up standard swap structures
  CommLattice::init(ninteger_caller, ndouble_caller,
                    delpropensity_caller, iarray, darray);

  if (use_gpu_aware) {
    setup_device_buffers();
  }
}

void CommLatticeKokkos::setup_device_buffers()
{
  Swap* swap = allswap;
  if (!swap) return;

  // Calculate total send/recv sizes
  int total_send = 0, total_recv = 0;
  for (int i = 0; i < swap->nsend; i++) total_send += swap->scount[i];
  for (int i = 0; i < swap->nrecv; i++) total_recv += swap->rcount[i];

  // Allocate device buffers
  if (ninteger > 0) {
    max_send_int = total_send * ninteger;
    max_recv_int = total_recv * ninteger;
    d_send_buf_int = Kokkos::View<int*>("send_buf_int", max_send_int);
    d_recv_buf_int = Kokkos::View<int*>("recv_buf_int", max_recv_int);
  }

  if (ndouble > 0) {
    max_send_double = total_send * ndouble;
    max_recv_double = total_recv * ndouble;
    d_send_buf_double = Kokkos::View<double*>("send_buf_double", max_send_double);
    d_recv_buf_double = Kokkos::View<double*>("recv_buf_double", max_recv_double);
  }

  // Copy send indices to device
  d_send_indices = Kokkos::View<int*>("send_indices", total_send);
  d_send_offsets = Kokkos::View<int*>("send_offsets", swap->nsend + 1);

  auto h_send_indices = Kokkos::create_mirror_view(d_send_indices);
  auto h_send_offsets = Kokkos::create_mirror_view(d_send_offsets);

  int offset = 0;
  for (int i = 0; i < swap->nsend; i++) {
    h_send_offsets(i) = offset;
    for (int j = 0; j < swap->scount[i]; j++) {
      h_send_indices(offset + j) = swap->sindex[i][j];
    }
    offset += swap->scount[i];
  }
  h_send_offsets(swap->nsend) = offset;

  Kokkos::deep_copy(d_send_indices, h_send_indices);
  Kokkos::deep_copy(d_send_offsets, h_send_offsets);

  // Copy recv indices to device
  total_recv = 0;
  for (int i = 0; i < swap->nrecv; i++) total_recv += swap->rcount[i];

  d_recv_indices = Kokkos::View<int*>("recv_indices", total_recv);
  d_recv_offsets = Kokkos::View<int*>("recv_offsets", swap->nrecv + 1);

  auto h_recv_indices = Kokkos::create_mirror_view(d_recv_indices);
  auto h_recv_offsets = Kokkos::create_mirror_view(d_recv_offsets);

  offset = 0;
  for (int i = 0; i < swap->nrecv; i++) {
    h_recv_offsets(i) = offset;
    for (int j = 0; j < swap->rcount[i]; j++) {
      h_recv_indices(offset + j) = swap->rindex[i][j];
    }
    offset += swap->rcount[i];
  }
  h_recv_offsets(swap->nrecv) = offset;

  Kokkos::deep_copy(d_recv_indices, h_recv_indices);
  Kokkos::deep_copy(d_recv_offsets, h_recv_offsets);
}

void CommLatticeKokkos::all()
{
  if (use_gpu_aware && allswap) {
    perform_swap_gpu_aware(allswap);
  } else {
    // Fallback to host-staged communication
    CommLattice::all();
  }
}

void CommLatticeKokkos::perform_swap_gpu_aware(Swap* swap)
{
  // This is the core GPU-aware MPI implementation

  // Get device views from AppLatticeKokkos
  // (Will need to add accessor methods to AppLatticeKokkos)
  AppLattice* applattice = (AppLattice*) app;
  // auto d_spin = applattice_kokkos->get_d_spin();  // Example

  auto h_send_offsets = Kokkos::create_mirror_view(d_send_offsets);
  auto h_recv_offsets = Kokkos::create_mirror_view(d_recv_offsets);
  Kokkos::deep_copy(h_send_offsets, d_send_offsets);
  Kokkos::deep_copy(h_recv_offsets, d_recv_offsets);

  // 1. Post non-blocking receives (directly into device buffers)
  for (int i = 0; i < swap->nrecv; i++) {
    int recv_offset = h_recv_offsets(i) * ninteger;  // For int arrays
    int recv_count = swap->rcount[i] * ninteger;

    MPI_Irecv(d_recv_buf_int.data() + recv_offset, recv_count, MPI_INT,
              swap->rproc[i], 0, world, &swap->request[i]);
  }

  // 2. Pack send buffer on GPU
  // Parallel over all sites being sent
  int total_send = h_send_offsets(swap->nsend);

  // For single integer array (spin) - simplified example
  // Full implementation would handle ninteger arrays
  Kokkos::parallel_for("pack_send", total_send,
    KOKKOS_LAMBDA(const int i) {
      // d_send_buf_int(i) = d_spin(d_send_indices(i));
    });
  Kokkos::fence();

  // 3. Send from device buffers
  for (int i = 0; i < swap->nsend; i++) {
    int send_offset = h_send_offsets(i) * ninteger;
    int send_count = swap->scount[i] * ninteger;

    MPI_Send(d_send_buf_int.data() + send_offset, send_count, MPI_INT,
             swap->sproc[i], 0, world);
  }

  // 4. Wait for all receives
  if (swap->nrecv > 0) {
    MPI_Waitall(swap->nrecv, swap->request, swap->status);
  }

  // 5. Unpack receive buffer on GPU
  int total_recv = h_recv_offsets(swap->nrecv);
  Kokkos::parallel_for("unpack_recv", total_recv,
    KOKKOS_LAMBDA(const int i) {
      // d_spin(d_recv_indices(i)) = d_recv_buf_int(i);
    });
  Kokkos::fence();
}

// ... implement all_reverse(), sector(), reverse_sector() similarly
```

## 2.4 POC Validation Test

```cpp
// Add to test file or main
void validate_gpu_aware_mpi() {
  // Run same simulation with and without GPU-aware MPI
  // Compare final state bit-for-bit

  // Method 1: Environment variable toggle
  // SPPARKS_GPU_AWARE_MPI=0 ./spparks < input  # Host staged
  // SPPARKS_GPU_AWARE_MPI=1 ./spparks < input  # GPU aware

  // Compare dump files or final energy values
}
```

## 2.5 POC Performance Measurement

Add timing comparison:

```cpp
void benchmark_comm_methods() {
  const int niter = 100;
  double t_host_staged = 0, t_gpu_aware = 0;

  // Warm up
  for (int i = 0; i < 10; i++) comm->all();

  // Host-staged timing
  KokkosMPI::gpu_aware_mpi_enabled = false;
  Kokkos::fence();
  MPI_Barrier(world);
  double t0 = MPI_Wtime();
  for (int i = 0; i < niter; i++) {
    sync_app_data_to_host();
    comm->all();
    sync_app_data_to_device();
    Kokkos::fence();
  }
  t_host_staged = MPI_Wtime() - t0;

  // GPU-aware timing
  KokkosMPI::gpu_aware_mpi_enabled = true;
  Kokkos::fence();
  MPI_Barrier(world);
  t0 = MPI_Wtime();
  for (int i = 0; i < niter; i++) {
    comm_kokkos->all();  // GPU-aware path
    Kokkos::fence();
  }
  t_gpu_aware = MPI_Wtime() - t0;

  if (me == 0) {
    printf("Host-staged: %.4f s for %d iterations (%.2f ms/iter)\n",
           t_host_staged, niter, 1000.0 * t_host_staged / niter);
    printf("GPU-aware:   %.4f s for %d iterations (%.2f ms/iter)\n",
           t_gpu_aware, niter, 1000.0 * t_gpu_aware / niter);
    printf("Speedup:     %.2fx\n", t_host_staged / t_gpu_aware);
  }
}
```

---

# Stage 3: Full Implementation

## Objective
Production-quality GPU-aware MPI with all communication patterns, robust error handling, and comprehensive testing.

## 3.1 Complete CommLatticeKokkos Class

### 3.1.1 Header File (Full Version)

**File: `src/KOKKOS/comm_lattice_kokkos.h`**

```cpp
#ifndef SPPARKS_COMM_LATTICE_KOKKOS_H
#define SPPARKS_COMM_LATTICE_KOKKOS_H

#include "comm_lattice.h"
#include <Kokkos_Core.hpp>

namespace SPPARKS_NS {

// Forward declarations
class AppLatticeKokkos;

// Device-side swap structure
template<class DeviceType>
struct SwapKokkos {
  int nsend, nrecv;

  // Host arrays (for MPI calls)
  int *sproc, *rproc;
  int *scount, *rcount;
  MPI_Request *request;
  MPI_Status *status;

  // Device arrays for indices
  Kokkos::View<int*, DeviceType> d_sindex_flat;
  Kokkos::View<int*, DeviceType> d_sindex_offsets;
  Kokkos::View<int*, DeviceType> d_rindex_flat;
  Kokkos::View<int*, DeviceType> d_rindex_offsets;

  // Device buffers for data
  Kokkos::View<int*, DeviceType> d_sbuf_int;
  Kokkos::View<int*, DeviceType> d_rbuf_int;
  Kokkos::View<double*, DeviceType> d_sbuf_double;
  Kokkos::View<double*, DeviceType> d_rbuf_double;
};

template<class DeviceType>
class CommLatticeKokkos : public CommLattice {
public:
  using device_type = DeviceType;
  using execution_space = typename DeviceType::execution_space;
  using memory_space = typename DeviceType::memory_space;

  CommLatticeKokkos(class SPPARKS *);
  ~CommLatticeKokkos();

  void init(int, int, int, int*, int*) override;
  void all() override;
  void all_reverse() override;
  void sector(int) override;
  void reverse_sector(int) override;

  // Set the app pointer for accessing device data
  void set_app_kokkos(AppLatticeKokkos* app_kk) { applattice_kokkos = app_kk; }

  // Query GPU-aware MPI status
  bool is_gpu_aware() const { return use_gpu_aware; }

private:
  AppLatticeKokkos* applattice_kokkos;
  bool use_gpu_aware;

  // Kokkos swap structures
  SwapKokkos<DeviceType>* allswap_kk;
  SwapKokkos<DeviceType>* reverseswap_kk;
  SwapKokkos<DeviceType>** sectorswap_kk;
  SwapKokkos<DeviceType>** sectorreverseswap_kk;

  // Setup methods
  void setup_swap_kokkos(Swap* swap, SwapKokkos<DeviceType>*& swap_kk);
  void free_swap_kokkos(SwapKokkos<DeviceType>*& swap_kk);

  // Core communication methods
  void perform_swap_gpu_aware(SwapKokkos<DeviceType>* swap_kk);
  void perform_swap_reverse_gpu_aware(SwapKokkos<DeviceType>* swap_kk);

  // Pack/unpack using app-specific arrays
  void pack_site_data(SwapKokkos<DeviceType>* swap_kk);
  void unpack_site_data(SwapKokkos<DeviceType>* swap_kk);
  void pack_site_data_reverse(SwapKokkos<DeviceType>* swap_kk);
  void unpack_site_data_reverse(SwapKokkos<DeviceType>* swap_kk);
};

} // namespace SPPARKS_NS

#endif
```

### 3.1.2 Full Implementation Patterns

**Sector Communication:**
```cpp
template<class DeviceType>
void CommLatticeKokkos<DeviceType>::sector(int isector)
{
  if (!use_gpu_aware || !sectorswap_kk || !sectorswap_kk[isector]) {
    // Fallback to host-staged
    CommLattice::sector(isector);
    return;
  }

  perform_swap_gpu_aware(sectorswap_kk[isector]);
}
```

**Reverse Communication (ghost → owner):**
```cpp
template<class DeviceType>
void CommLatticeKokkos<DeviceType>::perform_swap_reverse_gpu_aware(
    SwapKokkos<DeviceType>* swap_kk)
{
  // Reverse: send from recv indices, receive to send indices
  // Used to accumulate changes from ghosts back to owners

  // 1. Post receives at original send locations
  for (int i = 0; i < swap_kk->nsend; i++) {
    int offset = /* get from offsets */;
    int count = swap_kk->scount[i] * ninteger;
    MPI_Irecv(swap_kk->d_sbuf_int.data() + offset, count, MPI_INT,
              swap_kk->sproc[i], 0, world, &swap_kk->request[i]);
  }

  // 2. Pack from ghost sites (recv indices in forward direction)
  pack_site_data_reverse(swap_kk);
  Kokkos::fence();

  // 3. Send ghost data back to owners
  for (int i = 0; i < swap_kk->nrecv; i++) {
    int offset = /* get from offsets */;
    int count = swap_kk->rcount[i] * ninteger;
    MPI_Send(swap_kk->d_rbuf_int.data() + offset, count, MPI_INT,
             swap_kk->rproc[i], 0, world);
  }

  // 4. Wait and unpack to owned sites
  if (swap_kk->nsend > 0) {
    MPI_Waitall(swap_kk->nsend, swap_kk->request, swap_kk->status);
  }

  unpack_site_data_reverse(swap_kk);
  Kokkos::fence();
}
```

## 3.2 Integration with AppLatticeKokkos

**Modify: `src/KOKKOS/app_lattice_kokkos.h`**

```cpp
class AppLatticeKokkos : public AppLattice {
  // ...

  // Accessor methods for device arrays (used by CommLatticeKokkos)
  Kokkos::View<int*>& get_d_spin() { return d_spin; }
  Kokkos::View<double*[4]>& get_d_quaternion() { return d_quaternion; }
  Kokkos::View<int**>& get_d_iarray() { return d_iarray; }
  Kokkos::View<double**>& get_d_darray() { return d_darray; }

  // Communication object
  CommLatticeKokkos<DeviceType>* comm_kokkos;

  // ...
};
```

**Modify: `src/KOKKOS/app_lattice_kokkos.cpp` iteration loop**

```cpp
void AppLatticeKokkos::iterate_rejection(double stoptime)
{
  while (!done) {
    for (int iset = 0; iset < nset_loop; iset++) {

      if (nprocs > 1) {
        timer->stamp();

        if (comm_kokkos && comm_kokkos->is_gpu_aware()) {
          // GPU-aware MPI: no host sync needed
          if (sectorflag) comm_kokkos->sector(iset);
          else comm_kokkos->all();
        } else {
          // Fallback: host-staged communication
          sync_app_data_to_host();
          if (sectorflag) comm->sector(iset);
          else comm->all();
          sync_app_data_to_device();
        }

        timer->stamp(TIME_COMM);
      }

      // GPU computation (unchanged)
      timer->stamp();
      for (int icolor = 0; icolor < kokkos_ncolors; icolor++) {
        sweep_color_kokkos(icolor, color_start, color_count);
        Kokkos::fence();
      }
      timer->stamp(TIME_SOLVE);

      // Reverse communication
      if (nprocs > 1) {
        timer->stamp();

        if (comm_kokkos && comm_kokkos->is_gpu_aware()) {
          if (sectorflag) comm_kokkos->reverse_sector(iset);
          else comm_kokkos->all_reverse();
        } else {
          sync_app_data_to_host();
          if (sectorflag) comm->reverse_sector(iset);
          else comm->all_reverse();
          sync_app_data_to_device();
        }

        timer->stamp(TIME_COMM);
      }
    }
  }
}
```

## 3.3 Build System Integration

**Modify: `src/cmake/kokkos.cmake`**

```cmake
# GPU-aware MPI detection
if(SPPARKS_ENABLE_KOKKOS AND MPI_FOUND)
  message(STATUS "Checking for GPU-aware MPI support...")

  # Try to detect CUDA-aware MPI
  if(Kokkos_ENABLE_CUDA)
    include(CheckCXXSourceCompiles)
    set(CMAKE_REQUIRED_INCLUDES ${MPI_CXX_INCLUDE_DIRS})
    set(CMAKE_REQUIRED_LIBRARIES ${MPI_CXX_LIBRARIES})

    check_cxx_source_compiles("
      #include <mpi.h>
      #ifdef OPEN_MPI
      #include <mpi-ext.h>
      #endif
      int main() {
        #if defined(MPIX_CUDA_AWARE_SUPPORT) || (defined(OMPI_HAVE_MPI_EXT_CUDA) && OMPI_HAVE_MPI_EXT_CUDA)
        return 0;
        #else
        #error No CUDA-aware MPI support detected
        #endif
      }
    " HAVE_CUDA_AWARE_MPI)

    if(HAVE_CUDA_AWARE_MPI)
      message(STATUS "  CUDA-aware MPI: FOUND")
      target_compile_definitions(spparks PRIVATE HAVE_GPU_AWARE_MPI)
    else()
      message(STATUS "  CUDA-aware MPI: NOT FOUND (will use host staging)")
    endif()
  endif()

  # HIP/ROCm with Cray MPICH (Frontier)
  if(Kokkos_ENABLE_HIP)
    # Cray MPICH with GTL is detected at runtime via MPICH_GPU_SUPPORT_ENABLED
    message(STATUS "  HIP GPU-aware MPI: Runtime detection via MPICH_GPU_SUPPORT_ENABLED")
    target_compile_definitions(spparks PRIVATE HAVE_GPU_AWARE_MPI_RUNTIME)
  endif()

  # User override option
  option(SPPARKS_FORCE_GPU_AWARE_MPI
         "Force GPU-aware MPI (override detection)" OFF)
  if(SPPARKS_FORCE_GPU_AWARE_MPI)
    target_compile_definitions(spparks PRIVATE HAVE_GPU_AWARE_MPI)
  endif()
endif()
```

## 3.4 Testing Framework

### Unit Tests

```cpp
// test_comm_kokkos.cpp

void test_pack_unpack_correctness() {
  // Create known data pattern
  // Pack with GPU kernel
  // Verify buffer contents match expected
}

void test_mpi_exchange_gpu_vs_host() {
  // Run identical exchange with both paths
  // Compare results bit-for-bit
}

void test_sector_communication() {
  // Test each sector independently
  // Verify ghost data correct for each sector
}

void test_reverse_communication() {
  // Set values on ghost sites
  // Reverse comm back to owners
  // Verify accumulation correct
}
```

### Integration Tests

**File: `examples/potts/in.potts.kk.mpi_test`**
```
# Test GPU-aware MPI correctness
dimension 3
lattice sc/26n 1.0
region box block 0 50 0 50 0 50

create_box box
create_sites box
set site range 1 100

sweep raster
sector yes
temperature 0.5

# Dump state for comparison
dump 1 text 100 dump.potts.* id site

solve_style tree
run 1000
```

Compare outputs:
```bash
# Host-staged reference
SPPARKS_GPU_AWARE_MPI=0 mpirun -n 4 ./spparks -k on g 1 -sf kk < in.potts.kk.mpi_test
mv dump.potts.1000 dump.host.1000

# GPU-aware test
SPPARKS_GPU_AWARE_MPI=1 mpirun -n 4 ./spparks -k on g 1 -sf kk < in.potts.kk.mpi_test
mv dump.potts.1000 dump.gpu.1000

diff dump.host.1000 dump.gpu.1000  # Should be identical
```

---

# Stage 4: Platform-Specific Optimization

## 4.1 Frontier (AMD MI250X) Specifics

**Environment setup:**
```bash
module load PrgEnv-amd
module load rocm/5.4
module load cray-mpich/8.1.23

# Enable GPU-aware MPI (GTL)
export MPICH_GPU_SUPPORT_ENABLED=1

# Optimize for MI250X
export HSA_ENABLE_SDMA=0  # Better for small transfers
export MPICH_OFI_NIC_POLICY=GPU  # Route through GPU-attached NIC
```

**Slurm job script:**
```bash
#!/bin/bash
#SBATCH -A <project>
#SBATCH -J spparks_gpu_mpi
#SBATCH -t 01:00:00
#SBATCH -N 4
#SBATCH -p batch

# 8 GCDs per node, 32 total
srun -n 32 --gpus-per-task=1 --gpu-bind=closest \
  ./spparks -k on g 1 -sf kk < in.additive.kk
```

## 4.2 NVIDIA H100 Specifics

**Environment setup:**
```bash
module load cuda/12.2
module load openmpi/4.1.5+cuda

# UCX GPU-aware settings
export UCX_TLS=rc,cuda_copy,cuda_ipc,gdr_copy
export UCX_RNDV_SCHEME=put_zcopy
export UCX_RNDV_THRESH=0  # Always use rendezvous for GPU buffers

# NCCL alternative (if using collective operations)
export NCCL_P2P_LEVEL=NVL  # NVLink preferred
export NCCL_IB_DISABLE=0   # Enable InfiniBand
```

**Slurm job script:**
```bash
#!/bin/bash
#SBATCH -J spparks_gpu_mpi
#SBATCH -t 01:00:00
#SBATCH -N 2
#SBATCH --gres=gpu:h100:4

mpirun -n 8 --map-by ppr:4:node --bind-to core \
  -x UCX_TLS -x UCX_RNDV_SCHEME \
  ./spparks -k on g 1 -sf kk < in.additive.kk
```

## 4.3 Runtime Configuration

Add runtime options to SPPARKS:

```cpp
// In input script:
// kokkos gpu_aware_mpi yes/no/auto

void KokkosInput::command(int narg, char **arg) {
  // ... existing parsing ...

  if (strcmp(arg[iarg], "gpu_aware_mpi") == 0) {
    if (strcmp(arg[iarg+1], "yes") == 0) {
      KokkosMPI::force_gpu_aware = true;
    } else if (strcmp(arg[iarg+1], "no") == 0) {
      KokkosMPI::force_disable = true;
    } else if (strcmp(arg[iarg+1], "auto") == 0) {
      // Default: auto-detect
    }
  }
}
```

---

# Summary: Implementation Phases

| Phase | Effort | Risk | Value |
|-------|--------|------|-------|
| **Stage 1: Benchmarking** | 1-2 days | Low | Establishes baseline, justifies work |
| **Stage 2: POC** | 1 week | Medium | Validates approach, measures potential |
| **Stage 3: Full Implementation** | 2-3 weeks | Medium | Production-ready GPU-aware MPI |
| **Stage 4: Platform Optimization** | 1 week | Low | Maximizes performance on target systems |

**Total estimated effort:** 5-6 weeks

**Key files to create:**
- `src/KOKKOS/kokkos_mpi.h` - GPU-aware MPI detection
- `src/KOKKOS/comm_kokkos_kernels.h` - Pack/unpack kernels
- `src/KOKKOS/comm_lattice_kokkos.h` - Class declaration
- `src/KOKKOS/comm_lattice_kokkos.cpp` - Full implementation

**Key files to modify:**
- `src/KOKKOS/app_lattice_kokkos.h/cpp` - Integration
- `src/cmake/kokkos.cmake` - Build detection
- `src/KOKKOS/kokkos.cpp` - Initialization
