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
module load cmake                 # prefer a newer one if offered; see below
# NOTE: there is deliberately no `module load cray-hdf5-parallel` here -- under
# -magic modules that module is not visible. See section 2.
```

**CMake version.** The tree needs **>= 3.22** (`src/CMakeLists.txt:1`, and
bundled Kokkos requires the same). `src/CMakePresets.json` is schema version 3,
needing **>= 3.21**, so the presets can never be the binding constraint on a
machine that can build this at all.

An LC default of 3.24.2 works. If a newer module is available (e.g. 3.29.2),
prefer it — later CMake has better HIP-language and Cray-wrapper handling — but
it is not required. If you ever see

```
CMake Error: Could not read presets from <dir>: Unrecognized "version" field
```

that is a presets *schema* version above what your CMake understands (3.24
tops out at schema 5, 3.25 at 6), not a problem with your environment.

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

### Parallel HDF5 on a magic-module environment

`cray-hdf5-parallel` lives in the Cray PE **Lmod hierarchy** and only becomes
visible once a `PrgEnv-*` is loaded. LLNL's `-magic` modules deliberately sit
outside that hierarchy, so with `rocmcc/<ver>-magic` you will see only the serial
`cray-hdf5` and **no parallel module at all**. That is expected, not a broken
environment — point at the install directly (LLNL's "Explicit-Paths" model):

```bash
ls /opt/cray/pe/hdf5-parallel/            # versions
ls /opt/cray/pe/hdf5-parallel/1.14.3.5/   # compiler families: amd, crayclang, gnu, ...
```

Take the family matching your compiler (`amd` for rocmcc/amdclang++). Verified
working on El Capitan: `/opt/cray/pe/hdf5-parallel/1.14.3.5/amd/6.0`.

```bash
cmake --preset mi300a -DHDF5_ROOT=/opt/cray/pe/hdf5-parallel/1.14.3.5/amd/6.0
```

**`HDF5_ROOT` is the installation prefix — do NOT append `/include`.** With the
include directory CMake searches `<root>/include/include` and `<root>/include/lib`;
headers can still resolve while libraries do not, and it fails with the
thoroughly unhelpful `Could NOT find HDF5Enhanced (missing: HDF5_LIBRARIES)
(found version "1.14.3")`. The finder now detects and corrects this, but the
prefix form is what to pass.

Do **not** use the `openmpi` / `oneapi` / `gcc` / `clang` HDF5 modules. Those are
non-Cray builds tied to a different MPI; OpenMPI and MPICH are not ABI
compatible, and cray-mpich is MPICH-derived. Everything under
`/opt/cray/pe/hdf5-parallel/` is built against cray-mpich, which is the part that
must match.

Confirm before building:

```bash
grep H5_HAVE_PARALLEL <prefix>/include/H5pubconf.h    # what our finder greps for
```

and check the configure summary says `HDF5 is parallel: ON`. Since MPI implies a
parallel requirement here (the temperature sources call
`HighFive::MPIOFileAccess()` unconditionally), the finder sets
`HDF5_PREFER_PARALLEL` for you — otherwise CMake probes the serial `h5cc`
wrapper first and a wrapper on `PATH` outranks `HDF5_ROOT`.

### HighFive

By default CMake fetches HighFive from GitHub at configure time. If the machine
can reach GitHub, nothing to do. If not, install it (header-only) and configure
with:

```bash
-DSPPARKS_USE_EXTERNAL_HIGHFIVE=ON -DHighFive_DIR=<prefix>/share/cmake/HighFive
```

The configure summary prints which path was taken.

## 3. Getting the source there (and the Kokkos submodule)

If you transfer by **archive download + copy** rather than `git clone`, note that
a forge's ZIP/tar.gz export contains only the repository's own tree. `lib/kokkos`
is a separate repository referenced by a commit pointer, so it arrives **empty**,
and `git submodule update --init --recursive` cannot fix it — the copied tree has
no `.git` directory for the command to act on. Configure then fails with
"Kokkos not found at .../lib/kokkos".

Clone it directly instead. The build only requires the directory to exist with a
CMakeLists.txt; a registered submodule is not needed:

```bash
rm -rf lib/kokkos
git clone --depth 1 --branch 5.1.1 https://github.com/kokkos/kokkos.git lib/kokkos
```

Tag `5.1.1` is the pinned commit (`267ebc25fc5c8b96bb321f34d78f97b1c30f8830`).
Nothing further is needed for gfx942 — the bundled gfx1151 Kokkos patch is gated
on `SPPARKS_KOKKOS_ARCH MATCHES "^AMD_GFX115"` and is skipped.

**Then leave it alone.** Every later archive copy will again lack `lib/kokkos`,
so refresh `src/`, `docs/` and `examples/` but do not delete the Kokkos tree.

## 4. Configure and build

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

## 5. Runtime environment

```bash
export HSA_XNACK=1                    # required by the APU build
export MPICH_GPU_SUPPORT_ENABLED=1    # else "process_vm_readv: Bad address"
export MPICH_SMP_SINGLE_COPY_MODE=XPMEM
```

`HSA_XNACK=1` especially: without it Kokkos only *warns*
(`Kokkos_HIP.cpp:79`) and `HIPManaged` misbehaves quietly.

Jobs run under **Flux**, not Slurm. Four MI300A per node, so one rank per APU is
the natural mapping. `src/spparks.cpp` handles GPU-to-rank binding itself and
defers to the scheduler when `flux run --gpus-per-task=1` has already bound.

## 6. First runs, cheapest decisive test first

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

## 7. Things that will bite

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
below you exhaust the node's HBM first — so it only matters multi-node. Past
it, configure with `-DSPPARKS_BIGBIG=ON` (64-bit site IDs). Since 2026-10-22 the
create_sites path is 64-bit clean: the simple-lattice site count, the
checkerboard colouring, read_sites' site count, set `if id`, the sorted text
dump, diag array sums and the AM apps' global counts. A BIGBIG build (CPU and
HIP) reproduces the SMALLBIG trajectories and d6 bit for bit on the
am_validation decks, potts with `sweep color` and potts/pin. It has not been
run past 2^31 sites (~340 GB even for a 2-D lattice). What stays 32-bit stops
with an error instead: the read_sites Neighbors section, `set bfile`, the
diag cluster dump and app sinter. `set ... unique` wraps at 2^31-1, and the
31-bit Park-Miller RNG gives ids that differ by 2^31-1 the same stream under
`sweep color/strict`.

**Memory per site.** Measured on gfx1151 (2D HDF5 ct15 deck, np=1, slope of
peak memory over 4M/16M/64M-site boxes, host + device counted together):

| App | B/site |
|---|---|
| CPU `additive/texture` | ~410 |
| `additive/texture/kk`, discrete-GPU path (device copies) | ~700 |
| `additive/texture/kk`, integrated-GPU path | ~510 |

The integrated-GPU path moves the per-site host arrays into pinned host memory
the GPU reads in place, instead of keeping `hipMalloc` copies
(`AppLatticeKokkos::pin_site_arrays`). It keys on
`hipDeviceAttributeIntegrated`, not on the arch token or `HSA_XNACK`, and
prints `Site arrays shared with the integrated GPU (pinned host memory)` at the
first run. **Check for that line on MI300A**; without it you are on the
~700 B/site path. The arch token's unified memory does not merge these
separate allocations by itself.

Sizing at ~510 B/site, one rank per 128 GB APU with ~10% headroom: about
225M sites per APU. Multiply by the APUs per node (four on El Capitan; some
nodes have two).

**Multi-node HDF5 is still unproven, but the stale-layer trap is fixed.**
`load_layer()` used to return early when a rank's padded subdomain overlapped no
chunks of the current layer *without clearing the previous layer's mesh*, so
those sites interpolated stale thermal history. It now calls `clear_layer_data()`
on that path: every site reads `default_temp`, the rank says so on stdout, and
the empty interval list lets the run fast-forward to the next layer. The
Kokkos fill has a matching `d_n_active_nodes > 0` guard, without which the
cleared state would send `ComputeNodalTemps` past the end of a zero-extent view.

Regression test: `examples/am/surge/hdf5/stale_layer` (generates its own
three-layer file — the production file cannot reach the path).

This **decouples the correctness risk from `grid_cell_size_multiplier`**. Lowering
it to cut I/O amplification is now a performance-only tradeoff: the pad is still
load-bearing for boundary-straddling elements, for ghost sites (the site cache
covers `nall`), and for the `nodal_smooth` halo, and measured mean element extent
in the ct15 dataset is 41·dx — so 100·dx is only ~2.5 element hops of margin.
Shrink it and you will break interpolation at subdomain boundaries, but you will
no longer silently read the wrong layer.

Note that at np=4 the pad costs nothing anyway: the default decomposition is
2×2×1, the warning stays silent, and per-rank chunk loads genuinely differ
(66/69/62/62 on the ct15 box). The amplification only appears if a deck forces a
split along the thin axis.

## 8. A/B-ing the unified-memory path

`cmake --preset mi300a-discrete` builds the same source on the same hardware with
`AMD_GFX942` instead, leaving unified memory off. Same ISA, same allocations —
the only difference is that `DualView` keeps a distinct host mirror and
`deep_copy` does real work, which is the path the gfx1151 development machine
takes. Useful for isolating what unified memory is worth, and as a stricter
oracle: a missing sync shows up there and is masked by unified memory.

Run it **without** `HSA_XNACK=1` to match. Keep it in its own build directory —
`Kokkos_ARCH_*` is cached with `FORCE`, so two arch tokens in one directory both
stay `ON` and Kokkos errors out.

## 9. Reference

- [Compilers and User Environment](https://hpc.llnl.gov/documentation/user-guides/using-el-capitan-systems/using-el-capitan-systems-compilers-and-user)
- [C++ Code Examples](https://hpc.llnl.gov/documentation/user-guides/using-el-capitan-systems/introduction-and-quickstart/c)
- [MPI Overview](https://hpc.llnl.gov/documentation/user-guides/using-el-capitan-systems/mpi-overview)
- [Known Issues](https://hpc.llnl.gov/documentation/user-guides/using-el-capitan-systems/known-issues) — version compatibility table
