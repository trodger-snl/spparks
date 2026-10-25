# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this checkout is

SPPARKS (Stochastic Parallel PARticle Kinetic Simulator), a kinetic Monte Carlo
code. This checkout (`/home/tron/spparks`, branch **SURGE**) carries the
additive-manufacturing `additive/texture` app and its Kokkos/HIP port
(`additive/texture/kk`), building for an **AMD gfx1151 (Strix Halo, RDNA 3.5)**
APU with `hipcc`. The GPU line (branch SURGE-kokkos, checkout
`/home/tron/spparks-kokkos`) was merged into SURGE on 2026-10-17; SURGE is now
the main line.

The CPU `additive/texture` app is the correctness reference for the GPU port.
`lib/kokkos` is a git submodule (`git submodule update --init lib/kokkos`).

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
--kokkos-backend hip --package stitch --mpi` etc.); the raw `cmake -S src -B`
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
The `additive/texture*` styles are gated on **MPI** only
(`src/cmake/ConditionalSources.cmake`); HighFive (`--highfive`) adds the
`hdf5_*` temperature sources, which otherwise error at input time
(`#ifdef SPPARKS_HIGHFIVE` in `temperature_source_hdf5_*.cpp`). If a style is "not recognized" at
runtime, the style header generation skipped it — check that gating, not the
C++. The headers are generated per build dir (`<build>/styles/`), so build
dirs with different features (e.g. `build-verify` and the no-Kokkos
`build-cpu`) coexist. Any `src/style_*.h` (left by the `make mpi` path's
`Make.sh`) would shadow them and is deleted at configure and at every CMake
build.

## Core SPPARKS architecture (big picture)

- `App` (pure virtual) → `AppLattice` (lattice sites + neighbors, sectors,
  coloring) → `AppPotts` → `AppPottsQuaternion` → `AppAdditiveTexture`.
- Kokkos apps are mixins over the CPU app (LAMMPS-style):
  `LatticeKokkos<Base>` (device neighbours, colours, syncs;
  `AppLatticeKokkos` = `LatticeKokkos<AppLattice>`) and
  `QuaternionKokkos<Base>` (device spin/quaternions), so
  `AppAdditiveTextureKokkos : QuaternionKokkos<AppAdditiveTexture>` and
  `AppPottsQuaternionKokkos : QuaternionKokkos<AppPottsQuaternion>`
  (the phase-field-only `phasefield` app, `AppPhaseField`, runs one model
  struct from `src/phasefield_models.h` -- add a PDE there -- over
  `src/fd_stencil.h`, a host/device central-difference stencil for site
  fields; `AppPhaseFieldPottsTable : AppPhaseField` is its ch_heat model
  under Potts spins; both kk twins run the field step through
  `src/KOKKOS/phasefield_device.h`).
  The CPU
  class owns the state, parameters and argument parsing; the kk class
  replaces `init_app`/`app_update`/`input_app` (its fallback is
  `AppLattice::input_app`, so CPU-only commands stay errors) and never calls
  the CPU `init_app`. Member definitions live in `app_lattice_kokkos_impl.h`,
  explicitly instantiated in each app's .cpp.
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
(`AppAdditiveTextureKokkos`, style `additive/texture/kk`) reuse one
`ThermalManager`.

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
- **CUDA portability:** a member function that defines a `KOKKOS_LAMBDA` /
  `KOKKOS_CLASS_LAMBDA` must be **public**. nvcc rejects device lambdas in
  private or protected members; HIP and OpenMP accept them, so local builds never
  catch it (it broke a colleague's CUDA build). Run
  `python3 tools/check_kokkos_lambda_access.py` before committing Kokkos code;
  it exits 1 and names any offender. Other nvcc-only rejections (a private
  nested type passed to a kernel, headers nvcc cannot parse) need a real CUDA
  compile: `build-cuda/` is a compile-and-link-only CUDA build (nvcc 12.8 + gcc
  13 from the conda env `cuda-build`, Kokkos CUDA AMPERE80; there is no NVIDIA
  GPU here, so it cannot run). `. build-cuda/env.sh && cmake --build build-cuda`.

## Validating the GPU port

Microstructure output of both apps is repeatable bit for bit at a fixed seed and
rank count (keyed RNG; the kk app on each backend at any thread count, though HIP and
OpenMP differ from each other), but the two apps
differ from each other and change with the rank count. Only the
**temperature field is deterministic across apps** — validate by comparing the CPU app
(`additive/texture`) against the Kokkos app (`additive/texture/kk`) on the
temperature diagnostic columns (`diag_style array d6 sum d6 max`), expecting a
bit-for-bit match. A kk app that sweeps needs `sweep color` (or
`color/strict`); update-only apps (`app_update_only yes`) take any sweep. Base test inputs are in
`examples/hdf5/`, `examples/{rosenthal,moser}_am/` and `examples/am_validation/`.

## Commits

Commit each task as it completes (don't batch — interactive `git add -p` is
unavailable here, so interleaved changes are painful to split). Exclude the
`lib/kokkos` submodule pointer and `build-*/` artifacts unless asked. End commit messages with
the `Co-Authored-By:` / `Claude-Session:` trailer the harness specifies for the
current session.

## Reference docs

User-manual pages (txt2html: edit the `.txt`, regenerate with `python3 doc/txt2html.py`):
`doc/app_additive_texture`, `setup_temperature_source`, `temperature_source_{rosenthal,moser,conduction,hdf5}`,
`laser_path`, `laser_fluctuations`, `material`. Developer notes:
`docs/AM_KOKKOS_PLAN.md` (the additive Kokkos plan/status),
`docs/CONDUCTION_SOURCE.md` (the nonlinear conduction temperature source), `docs/MATERIAL_CARDS.md`
(TOML material cards, `material` command, `tools/material_card.py`), plus
`docs/KOKKOS_HOT_LOOP_OPTIMIZATION.md`, `docs/KOKKOS_MULTI_GPU.md`, `docs/EL_CAPITAN_BUILD.md`.
