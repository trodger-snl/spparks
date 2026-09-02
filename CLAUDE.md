# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this checkout is

SPPARKS (Stochastic Parallel PARticle Kinetic Simulator), a kinetic Monte Carlo
code. This checkout (`/home/tron/spparks-kokkos`, branch **SURGE-kokkos**) is the
GPU-acceleration line: it adds a Kokkos/HIP port of the additive-manufacturing
`additive/texture` app, building for an **AMD gfx1151 (Strix Halo, RDNA 3.5)**
APU with `hipcc`.

Sibling repos: `/home/tron/spparks` (the non-GPU SURGE line), plus analysis
tooling. The two SPPARKS repos share history; the CPU `additive/texture` app here
is the correctness reference for the GPU port.

## Build & run

All builds use the **spack environment `spparks-dev`** (provides openmpi,
HDF5+parallel, HighFive). Activate it first or commands fail:

```bash
. /home/tron/spack/share/spack/setup-env.sh && spack env activate spparks-dev
```

The working GPU build is configured in `build-verify/` (hipcc, HighFive+HDF5+MPI,
Kokkos HIP/gfx1151). Reconfigure + build:

```bash
cmake -S src -B build-verify          # configure (also installs packages, see below)
cmake --build build-verify -j 8       # build -> executable is build-verify/spparks
```

The executable is **`spparks`** (not `spk`). Run an input:

```bash
mpirun -np 1 build-verify/spparks < in.somefile     # -np N for multi-rank
```

`src/build_cmake.sh` is the convenience wrapper (`--highfive --kokkos
--kokkos-backend hip --package stitch -m mpi` etc.); the raw `cmake -S src -B`
form above is what's used day-to-day here. The Makefile build (`make mpi`) also
exists but the CMake/HighFive path is the supported one for the additive apps.

There is no unit-test harness for this work; correctness is verified by running
inputs and comparing output — see "Validating the GPU port" below.

## Packages and style headers

Optional components (KOKKOS, STITCH) are *packages*: their sources live in
`src/KOKKOS/` and `src/STITCH/` and compile from there when
`SPPARKS_PACKAGE_KOKKOS` / `SPPARKS_PACKAGE_STITCH` are on (the package
directory goes on the include path). Nothing is copied into `src/`; edit the
package files in place and rebuild. Adding a *new* source or style header
still needs a reconfigure (`cmake -S src -B build-verify`) because the source
list and the style headers are built at configure time.

`style_*.h` headers are auto-generated (`src/cmake/generate_styles.cmake`) and
register app/command/etc. classes via `AppStyle(...)` macros in the headers.
The `additive/texture*` styles are gated on **HighFive + MPI**
(`src/cmake/ConditionalSources.cmake`). If a style is "not recognized" at
runtime, the style header generation skipped it — check that gating, not the
C++. Any `cmake -S src -B <dir>` rewrites the shared `src/style_*.h`, so a
scratch configure with fewer features can deregister styles for a working
build; reconfigure the real build dir afterwards.

## Core SPPARKS architecture (big picture)

- `App` (pure virtual) → `AppLattice` (lattice sites + neighbors, sectors,
  coloring) → `AppPotts` → `AppPottsQuaternion` → `AppAdditiveTexture`.
  Apps implement `site_event*` / `app_update`; registered via `AppStyle`.
- Per-site data: `iarray[]` (ints) and `darray[]` (doubles). The additive app
  maps `active_flag = iarray[1]` (= `i2` in inputs) and `T = darray[5]` (= `d6`).
- Parallelism: MPI domain decomposition with ghost-site exchange (`CommLattice`;
  `comm->all_selective(...)` does index-selective ghost comm).

## The additive/texture temperature-source architecture (the part to understand)

This is where the non-obvious, multi-file design lives.

**Temperature I/O is factored out of the app into `ThermalManager`**
(`src/thermal_manager.{h,cpp}`). It owns the modular `TemperatureSource`
(Rosenthal analytic / Moser Green's-function / HDF5), the laser scan path,
fast-forward predictors, and inter-pass pause/reset scheduling. `app_update`
delegates to `thermal->advance(time, stoptime)` which returns
`{new_time, activate_powder, end_run}`.

`ThermalManager` reaches lattice state through a **`ThermalHost` POD snapshot**
(`src/thermal_host.h`), populated by each app's `refresh_thermal_host()` — NOT a
typed back-pointer. This decoupling is what lets BOTH the CPU app
(`AppAdditiveTexture`, style `additive/texture`) and the Kokkos app
(`AppAdditiveTextureKokkos`, style `additive/texture/kk`, inherits
`AppPottsQuaternionKokkos`) reuse one `ThermalManager`.

**GPU temperature evaluation.** The per-site temperature integral is the dominant
cost, so it runs on the device. The host advances cheap scalar state (laser pose,
pass scheduling) and produces a plan POD (`src/rosenthal_device.h`,
`src/moser_device.h`); the Kokkos app fills `Kokkos::DualView<double*> k_T`
directly in a `parallel_for` over all `nlocal+nghost` sites (positional, so
ghosts compute their own T — no ghost temp comm). Covered on device: Rosenthal
standard/anisotropic/keyhole; Moser standard/keyhole, stochastic fluctuations,
static + dynamic multi-pass pause/reset. Dynamic pause/reset uses a **device-fill
callback** (`set_moser_device_filler` + `moser_advance_device`): the host makes
the per-pass firing decisions (they rebuild the scan), the device does every
fill. Only `temperature_smooth` (solidification-band smoothing) deliberately
falls back to the host path.

**Performance/consistency conventions in the Kokkos app** (`app_update`):
- The app sets `app_update_only=1`, so it runs via
  `AppLattice::iterate_app_update_only` (just `app_update(dt); time += dt;`) —
  the rKMC color sweeps are skipped; all physics is in the `app_update` device
  kernels (melt/nucleate/solidify/smooth). `time` is advanced once per step by
  that loop — `app_update` must NOT also increment it.
- Lazy device→host copy-back: device fills mark `host_T_stale`; host `T` is
  synced only before a host consumer reads it (powder activation, the `d6`
  diagnostic at output steps, `stats()`).
- Global max temperature and powder activation run on the device
  (`d_active_flag`/`d_xyz`), not host loops. Cached `d_xyz` must be built
  layout-correctly (host `xyz` is row-major `[nall][3]`; a LayoutLeft device View
  misreads it — use `ensure_device_xyz()`).

## Validating the GPU port

Microstructure output is non-deterministic run-to-run (MC randomness). Only the
**temperature field is deterministic** — validate by comparing the CPU app
(`additive/texture`) against the Kokkos app (`additive/texture/kk`) on the
temperature diagnostic columns (`diag_style array d6 sum d6 max`), expecting a
bit-for-bit match. The Kokkos app requires `sweep color`. Base test inputs are in
`examples/ReducedTempAM/` and the SURGE branch's `examples/{rosenthal,moser}_am/`.

## Commits

Commit each task as it completes (don't batch — interactive `git add -p` is
unavailable here, so interleaved changes are painful to split). Exclude the
`lib/kokkos` submodule pointer and `build-*/` artifacts unless asked. End commit messages with
the `Co-Authored-By:` / `Claude-Session:` trailer the harness specifies for the
current session.

## Reference docs

`docs/AM_KOKKOS_PLAN.md` (the additive Kokkos plan/status), plus
`docs/KOKKOS_HOT_LOOP_OPTIMIZATION.md`, `docs/KOKKOS_MULTI_GPU.md`, `docs/EL_CAPITAN_BUILD.md`.
