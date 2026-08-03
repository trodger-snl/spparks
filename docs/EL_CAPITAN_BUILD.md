# Building on El Capitan (and other MI300A / HPE Cray EX machines)

**Status:** Build path defined and encoded as a CMake preset. **Not yet run on the
machine** — every measurement below is from the gfx1151 development APU or from
LLNL documentation, and is marked as such.
**Date:** August 3, 2026
**Applies to:** El Capitan, Tuolumne, El Dorado, RZAdams — any HPE Cray EX node
with AMD MI300A. Keyed by hardware, not by site.

## 1. Modules

```bash
module load rocmcc/6.4.3-magic    # check `module avail rocmcc` for the current -magic
module load cray-mpich            # load AFTER the compiler; it resolves against it
module load cray-hdf5-parallel
module load cmake
```

Use the `-magic` variants. For `cray-mpich/8.1.30+` they auto-link `-lxpmem` and
the GTL library (`-lmpi_gtl_hsa`), which LLNL has recommended for all
applications since August 2024.

ROCm **>= 6.4.3** matters here specifically: LLNL's known-issues page reports
that 6.4.3 patches a rare race in the header-only `rocprim`, and Kokkos pulls
`rocprim` in for its sort paths. Check the cce/rocm/cray-mpich compatibility
table on that page before pinning versions.

## 2. HighFive

`SPPARKS_ENABLE_HIGHFIVE=ON` is required — the `additive/texture*` styles are
gated on HighFive **and** MPI (`src/cmake/ConditionalSources.cmake`). If it
resolves to nothing the styles are silently dropped and you find out at runtime
as "style not recognized".

By default CMake fetches HighFive from GitHub at configure time. If the machine
can reach GitHub, nothing to do. If not, install it (header-only) and configure
with:

```bash
-DSPPARKS_USE_EXTERNAL_HIGHFIVE=ON -DHighFive_DIR=<prefix>/share/cmake/HighFive
```

The configure summary prints which path was taken.

## 3. Configure and build

```bash
cd src
cmake --preset mi300a          # -> build-mi300a/
cmake --build --preset mi300a -j 16
```

The preset sets `amdclang++` / `mpiamdclang++` (LLNL states amdclang++
outperforms CCE for C++ HIP), `SPPARKS_KOKKOS_HOST_OPENMP=ON`, and the arch
token below. Run `cmake --list-presets` from `src/` for the full list.

Do **not** hand-write `-x hip --offload-arch=... --rocm-path=...` as the LLNL C++
examples show — Kokkos generates all of it from the arch token. Ignore their
`-std=c++11` too; this tree is C++20 (`src/CMakeLists.txt:11`).

**Configure on a compute node** (`flux alloc -N 1 -t 2h -q pdev`). The preset
leaves `-march=native` on. If you must configure on a login node, add
`-DSPPARKS_NATIVE_OPTIMIZATION=OFF`.

### The arch token is the one thing you cannot get wrong

Use **`AMD_GFX942_APU`**, not `AMD_GFX942`. Both emit `--offload-arch=gfx942`,
but only the APU token defines `KOKKOS_IMPL_HIP_UNIFIED_MEMORY`
(`lib/kokkos/core/src/setup/Kokkos_Setup_HIP.hpp:39`), which flips
`MemorySpaceAccess<HostSpace,HIPSpace>::accessible` to true
(`Kokkos_HIP_Space.hpp:279-285`) and gives `HIPSpace` a host-accessible
`SharedAllocationRecord`. The allocator is `hipMalloc` either way. Without it you
get discrete-GPU semantics on unified hardware: a real host mirror and a real
copy for every `DualView`.

A wrong token is now caught at configure time with the valid list
(`src/CMakeLists.txt`, the check just after the Kokkos submodule test).

## 4. Runtime environment

```bash
export HSA_XNACK=1                    # required by the APU build
export MPICH_GPU_SUPPORT_ENABLED=1    # else "process_vm_readv: Bad address"
export MPICH_SMP_SINGLE_COPY_MODE=XPMEM
```

`HSA_XNACK=1` especially: without it Kokkos only *warns*
(`Kokkos_HIP.cpp:79`) and `HIPManaged` misbehaves quietly.

Jobs run under **Flux**, not Slurm. Four MI300A per node, so one rank per APU is
the natural mapping. `src/main.cpp` handles GPU-to-rank binding itself and
defers to the scheduler when `flux run --gpus-per-task=1` has already bound.

## 5. First runs, cheapest decisive test first

1. **Configure only.** Read the summary for `Unified memory: ON`, the integer
   width, and the "Styles NOT built" block. If `additive/texture/kk` is listed
   as skipped, stop — nothing else matters.
2. **`potts/kk` with `sweep color/strict`, np=1, small lattice.** This is the
   only CPU oracle available: strict keys each site's RNG stream to its global
   id, so the result is bit-identical to the plain CPU `potts` app. One run
   confirms wave64 codegen, unified-memory DualView semantics, and the compiler
   flags all at once.
3. **Same case at np=1/2/4 on one node.** Strict is decomposition-invariant.
   Exercises the 4-APU binding path and the per-color forward exchange.
4. **A deliberately zero-site rank** (thin geometry, no `processors` line, np=4).
   Cheapest way to surface a rank-local `error->all`. Run under a hard wall time.
5. **`additive/texture/kk` with the HDF5 source**, np=1 then np=4, temperature
   columns only (`diag_style array d6 sum d6 max`).

If step 2 fails, rebuild with `-DSPPARKS_AGGRESSIVE_OPTIMIZATION=OFF` and rerun
before debugging anything else — see below.

## 6. Things that will bite

**`-ffast-math` versus bit-for-bit validation.** Release adds `-ffast-math
-funroll-loops -flto` unconditionally for Clang (`src/CMakeLists.txt`). The
stated acceptance criterion for the AM port is a bit-for-bit temperature match
against the CPU app; with fast-math on a different ISA that will probably not
hold, and you will not be able to tell "port is broken" from "compiler
reassociated a sum". Build the first validation binary with
`-DSPPARKS_AGGRESSIVE_OPTIMIZATION=OFF`.

**AM microstructure is not comparable across machines.** The default RNG pool
keys on hardware thread index and sizes itself from `concurrency()`, which
differs between gfx942 and gfx1151. `sweep color/strict` fixes this for
`potts/kk` and `potts/quaternion/kk`, but **not** for `additive/texture/kk`,
which sets `app_update_only=1` and never runs the color sweep. AM validation
stays temperature-only, exactly as it is locally.

**32-bit site IDs.** The build defaults to `SPPARKS_SMALLBIG`, capping a run at
2^31 sites. On one node that ceiling is unreachable — at the per-site footprint
measured on this branch you exhaust 512 GB of unified HBM first — so it only
matters multi-node. Note that turning on `SPPARKS_BIGBIG` is **not currently
safe**: `src/read_sites.cpp:394` does an `MPI_Allreduce` on a `tagint` with a
hardcoded `MPI_INT`, and `src/lattice.cpp:365-366,372-373` computes
`idsite/(nx*ny)` with `nx*ny` as `int*int`, which silently corrupts the
checkerboard coloring rather than erroring. Both need fixing first.

**Multi-node HDF5 is unproven.** `load_layer()` returns early when a rank's
padded subdomain overlaps no chunks of the current layer, without clearing the
previous layer's mesh — so those sites get interpolated from stale thermal
history. The wide chunk-load pad currently keeps this latent (no rank has been
observed reaching the zero-chunk path), and there is now a warning when that pad
is wide enough to erase a decomposed axis. The two are coupled through
`grid_cell_size_multiplier`: **do not lower it to reduce I/O amplification
without fixing the early return first**, or you trade a performance problem for a
silent correctness one.

## 7. A/B-ing the unified-memory path

`cmake --preset mi300a-discrete` builds the same source on the same hardware with
`AMD_GFX942` instead, leaving unified memory off. Same ISA, same allocations —
the only difference is that `DualView` keeps a distinct host mirror and
`deep_copy` does real work, which is the path the gfx1151 development machine
takes. Useful for isolating what unified memory is worth, and as a stricter
oracle: a missing sync shows up there and is masked by unified memory.

Run it **without** `HSA_XNACK=1` to match. Keep it in its own build directory —
`Kokkos_ARCH_*` is cached with `FORCE`, so two arch tokens in one directory both
stay `ON` and Kokkos errors out.

## 8. Reference

- [Compilers and User Environment](https://hpc.llnl.gov/documentation/user-guides/using-el-capitan-systems/using-el-capitan-systems-compilers-and-user)
- [C++ Code Examples](https://hpc.llnl.gov/documentation/user-guides/using-el-capitan-systems/introduction-and-quickstart/c)
- [MPI Overview](https://hpc.llnl.gov/documentation/user-guides/using-el-capitan-systems/mpi-overview)
- [Known Issues](https://hpc.llnl.gov/documentation/user-guides/using-el-capitan-systems/known-issues) — version compatibility table
