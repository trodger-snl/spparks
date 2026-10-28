# Running the Kokkos additive app on multi-GPU nodes

## Rank → GPU binding (automatic)

`main()` binds each MPI rank to a GPU by **node-local rank**:
`device = (node_local_rank) % (visible GPUs)`, set via `KOKKOS_DEVICE_ID`
before `Kokkos::initialize()` (see `src/spparks.cpp`,
`spk_bind_gpu_by_local_rank`). The local rank is found with
`MPI_Comm_split_type(MPI_COMM_TYPE_SHARED)`, so it works on **any** launcher
(srun, mpirun, mpiexec) — not only the few whose env vars Kokkos recognizes.

It is a **no-op** when:
- only one GPU is visible (single-GPU box, or a scheduler that already pins one
  GPU per rank via `ROCR_/HIP_/CUDA_VISIBLE_DEVICES`), or
- the build is host-only (Serial/OpenMP), or
- you set `KOKKOS_DEVICE_ID`, `KOKKOS_MAP_DEVICE_ID_BY`, or `--kokkos-device-id`
  yourself (those win).

Rank 0 prints a one-line summary, e.g.
`SPPARKS Kokkos: 4 visible GPU(s); binding MPI ranks to GPUs by node-local rank
(this node: 8 rank(s) over 4 GPU(s)).`

## How many ranks to launch

The GPU does the physics; a GPU code wants **few ranks per GPU**, not one per
CPU core. Oversubscribing a GPU serializes its kernels (no speedup + overhead;
expensive sources like Moser stall hardest).

- **Best default: 1 rank per GPU** (e.g. 4 ranks on a 4-GPU node). Each rank owns
  1/N of the domain and drives one GPU.
- **2 ranks/GPU** is usually fine for cheap-fill sources (Rosenthal, HDF5
  interpolation); reserve 1/GPU for Moser (heavy Green's-function quadrature).
- Do **not** run one rank per core (e.g. 112 ranks on 4 GPUs = 28:1
  oversubscription). The extra cores are for host-side work, not feeding the GPU.

Caveat: the per-layer HDF5 element-cache build is host-serial per rank, so very
few ranks make it slower (e.g. ~57 s at 1 rank vs ~7 s at 8). If that dominates,
prefer 2 ranks/GPU, or thread the cache build (future work).

## Launch examples

4×H100 (CUDA) or 4×MI300 (HIP) node, 1 rank/GPU:

```bash
# SLURM
srun -N1 -n4 --gpus-per-node=4 ./spparks -in in.foo
# OpenMPI
mpirun -np 4 ./spparks -in in.foo
```

2 ranks/GPU on a 4-GPU node:

```bash
mpirun -np 8 ./spparks -in in.foo      # ranks 0,4->GPU0  1,5->GPU1  2,6->GPU2  3,7->GPU3
```

Let the scheduler pin GPUs instead (binding then auto-no-ops, count==1/rank):

```bash
srun -N1 -n4 --gpu-bind=closest ./spparks -in in.foo
```

Override the device for one rank (debugging):

```bash
KOKKOS_DEVICE_ID=2 ./spparks -in in.foo
# or:  ./spparks -in in.foo --kokkos-device-id=2
```

## Verifying the binding

- The rank-0 summary line above confirms it is active.
- `rocm-smi` / `nvidia-smi` during a run should show **all** GPUs busy (not just
  device 0). On NVIDIA, MPS lets >1 rank/GPU share more efficiently.
- Temperature is decomposition-invariant: multi-rank `diag d6 sum/max` matches
  single-rank (the ghost-halo design fills T positionally), so it is a quick
  correctness check that the decomposition is sound.

## Discrete GPUs (H100/H200, MI250X/MI300X)

Development ran on APUs, where host and GPU share memory and copies are
nearly free. On a discrete GPU every host<->device copy crosses PCIe, so:

- **Halo and ghost exchanges move only boundary data.** The conduction halo
  and the lattice ghost exchange pack on the device and move only the packed
  buffers. They go through `MPIStagingSpace` (`src/KOKKOS/gpu_aware_mpi.h`):
  pinned host memory on CUDA, ordinary host memory on HIP, whose pinned space
  is fine-grained and measured 1.8x slower.
- **GPU-aware MPI**: `export SPPARKS_GPU_AWARE_MPI=1` hands MPI the device
  buffers and skips the staging. Only with an MPI built with GPU support
  (Open MPI: `ompi_info --parsable --all | grep
  mpi_built_with_cuda_support:value`); otherwise it crashes on the first
  exchange. Off by default.
- **Output copies everything.** Each `stats` line, dump or diagnostic step
  copies every per-site array to the host (about 9 GB at 112.5M sites, about
  3.5 s through pageable memory on an H200). Keep `stats` coarse (0.1 s
  rather than 1 ms of simulated time); a 1 ms cadence cost the 2 mm Bi2Te3
  deck about 1.6x.
- **Conduction stencil tile**: `-DSPPARKS_COND_TILE=XxY` (default `32x8`,
  tuned on gfx1151) sets the ConductionStep team tile; results do not depend
  on it. Sweep it with ncu/nsys on a new GPU. On an H200 (2 mm Bi2Te3 deck,
  3 ms) 32x8 was fastest: 64x4 +8%, 64x8 +35%, 32x16 +42%. The double kernel
  uses 76 registers on sm_90, so only 3 teams of 256 fit per SM (37.5%
  occupancy) and bigger teams fit fewer. `-DSPPARKS_COND_MINB=N` (Kokkos
  LaunchBounds, default 0 = off) asks for N teams per SM; N = 4 caps it at
  64 registers with a 24-byte spill, and was 7% SLOWER on the H200: each
  team marches a whole z column, so the 1122 teams of a 528x528 grid run
  in 3 waves on 132 SMs at 3 or 4 teams per SM. It can pay where it saves
  a wave (A100: 108 SMs, 4 -> 3 waves). Read register counts without profiler
  rights: `cuobjdump --dump-resource-usage spparks | grep -A1 ConductionSolverKokkos5stage`.
- **Precision**: `-DKOKKOS_PREC=mixed` is validated for conduction to 0.010 K
  per site (`docs/CONDUCTION_SOURCE.md`); measure it before choosing double.
  H200: 1.17x (50 registers, 50% occupancy), not the 1.6-3x of gfx1151.
- **Stage 1 kernel** (6ce12630: 13 -> 7 divisions per cell, linear
  indexing): H200, same 3 ms, back to back: double 33.8 -> 27.8 s (1.22x),
  mixed 28.3 -> 22.5 s (1.26x); mixed s1 is 1.50x the old double kernel.
