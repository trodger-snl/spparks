# Running the Kokkos additive app on multi-GPU nodes

## Rank → GPU binding (automatic)

`main()` binds each MPI rank to a GPU by **node-local rank**:
`device = (node_local_rank) % (visible GPUs)`, set via `KOKKOS_DEVICE_ID`
before `Kokkos::initialize()` (see `src/main.cpp`,
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
