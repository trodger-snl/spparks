# Getting Started: Moser Temperature Source

A practical guide for running additive-manufacturing texture simulations
in SPPARKS using the **Moser / Green's-function** temperature source
(`app_style additive/texture`).

If you are new to SPPARKS or to this lab, start here and work top to
bottom. By the end you will have built SPPARKS, run a single-pass laser
scan, and viewed the result in OVITO.

---

## 1. What is the Moser source?

The Moser source is the unsteady, time-domain integral form of the
ellipsoidal Gaussian heat source against the half-space heat-conduction
Green's function (Moser et al., SAND2017, Eq. 4). Compared with the
classical Rosenthal point-source kernel:

| | Rosenthal | Moser |
|---|---|---|
| Kernel | point source | ellipsoidal Gaussian |
| Time-history | steady-state (moving frame) | unsteady (path + time) |
| Fluctuations in laser parameters | not supported | supported |
| Computational cost | very cheap | moderate (numerical integral) |

Choose Moser when you need finite source-volume effects (laser spot
shape) or stochastic per-emission fluctuations in laser power, width,
or depth. Choose Rosenthal when the laser parameters are constant in
time and you only need the steady moving-frame field.

Both sources are Green's-function methods on an insulating half-space
(image-source factor of 2). The Moser source reduces to Rosenthal in
the limit `(sx, sy, sz) → 0`.

---

## 2. Installing a compiler toolchain

You need a C++17 compiler, MPI, and CMake ≥ 3.16. HDF5 is optional —
enable it only if you plan to drive simulations from externally-prepared
FE thermal traces via the `hdf5_unstructured` / `hdf5_csr` temperature
sources. The Moser source itself does not need HDF5.

### macOS — Homebrew

```bash
# One-time install (or `brew upgrade` if already present)
brew install cmake gcc open-mpi hdf5 libpng jpeg
```

GCC from Homebrew is preferred over Apple Clang for OpenMP support.
After install, MPI is exposed as `mpicxx`.

### Linux — modules

Load the toolchain via the module system. Run `module avail` to see
what's offered, then load a compiler, MPI, and CMake (and HDF5 if
you'll use it):

```bash
module load gcc openmpi cmake
```

---

## 3. Building SPPARKS

The recommended path is the convenience script `build_cmake.sh` in
`src/`.

```bash
cd /path/to/spparks/src

# Default build (auto-detects platform, MPI, HDF5)
./build_cmake.sh

# Explicit machine target
./build_cmake.sh -m mac          # macOS, no MPI
./build_cmake.sh -m mpi          # generic MPI
./build_cmake.sh -m linux        # Linux

# With optional features
./build_cmake.sh -m mpi --hdf5 --highfive --jpeg
./build_cmake.sh -m mpi --package stitch        # enable STITCH I/O
./build_cmake.sh --clean -b Debug               # clean debug build
```

The binary lands at `src/build/spparks`. Add it to your `PATH` or
invoke with the full path.

If you'd rather drive CMake directly:

```bash
mkdir -p src/build && cd src/build
cmake -DSPPARKS_MACHINE=mpi -DSPPARKS_ENABLE_HDF5=ON ..
cmake --build . -j8
```

---

## 4. Running your first Moser simulation

The simplest end-to-end example is `in.additive.moser` in this
directory: a single 1.2 mm linear scan over an LPBF-scale box.

```bash
cd /path/to/spparks/examples/moser_am
mpirun -n 6 /path/to/spparks/src/build/spparks -in in.additive.moser
```

Expected runtime: a few minutes on a workstation. Output goes to
`log.spparks` and `DumpFiles_moser/`. As a quick sanity check, look at
the tail of `log.spparks` for a `Loop time of …` line and confirm at
least two `dump.moser.*` files were written.

For `-n`, start with 2–4 ranks on a laptop and 8–16 on a workstation.
The boxes in these examples are small; more ranks past that point hit
domain-decomposition overhead before they help.

### Anatomy of the input script

A Moser run has six required sections, in order:

```
1.  app_style additive/texture <nspins> <dx[m]> <dt[s]> <nrefine>
2.  lattice / region / create_box / create_sites / set ...
3.  Material thermo + Potts parameters (liquidus, solidus, mobility, …)
4.  setup_temperature_source moser <Q> <lambda> <k> <alpha> <T0> <cp> <sx> <sy> <sz>
5.  laser_path start <X0> <Y0> <Z0> end <X1> <Y1> speed <V> [repeats <N>]
6.  run <duration_seconds>
```

**The order of (4) and (5) matters.** The Moser source materializes
its scan tables inside `laser_path`, so `setup_temperature_source` must
come first. If you also use `laser_fluctuations` (see §6), it must sit
between them. SPPARKS will refuse to run with a clear error if the
order is wrong.

### Choosing the parameters

| Parameter | Meaning | Typical LPBF range |
|---|---|---|
| `Q` | raw laser power (W) | 100–500 |
| `lambda` | absorption efficiency | 0.1–0.5 |
| `k` | thermal conductivity (W/m·K) | 10–25 (Ni-superalloy ~ 11) |
| `alpha` | thermal diffusivity (m²/s) | 2–5 × 10⁻⁶ |
| `T0` | preheat / ambient (K) | 300–1000 |
| `cp` | specific heat (J/kg·K) | 500–700 |
| `sx, sy, sz` | ellipsoidal source widths (m) | 30–100 µm; calibrate |

`sx, sy, sz` are the standard-deviations of the ellipsoidal Gaussian
energy deposition in the moving laser frame. With `sx == sy` the
source is circular in plan view; `sz` controls the depth of energy
deposition. There is no closed-form way to set these — calibrate
against a melt-pool image or a high-fidelity FE solution if quantitative
fidelity matters.

The `laser_path` argument is given in meters and m/s — SPPARKS does
not assume any unit system internally.

---

## 5. Output files

A typical run produces three kinds of output:

### `log.spparks`
A line-by-line transcript of every command echoed plus per-step output
from `stats`. Useful for diagnosing setup errors and monitoring run
progress. Open it in any text editor.

### Text dump files (`dump.* text ... `)
Plain-text per-site state at each dump cadence, one file per timestep.
The columns you choose in the `dump` command determine what's written:

```
dump 1 text 2e-4 dump.moser.* id i1 i2 d6 x y z
```

This writes id, spin (i1), active_flag (i2), temperature (d6), and
xyz coordinates every 2×10⁻⁴ s. Field meanings for `app_style
additive/texture`:

| Field | Meaning |
|---|---|
| `i1` | spin (Potts grain id) |
| `i2` | active_flag: 0=inactive, 1=active layer, 2=molten, 3=solidified, 5=void |
| `d1`–`d4` | quaternion (q0, qx, qy, qz) for crystallographic orientation |
| `d5` | mobility_out (per-site Boltzmann factor) |
| `d6` | temperature T [K] |
| `d7` | solid_d (refinement-window counter) |
| `d8` | thermal gradient G [K/m] at solidification |
| `d9` | solidification rate V [m/s] |

For OVITO compatibility you **must** include `x y z` in the dump
command — OVITO needs explicit per-site coordinates and won't
reconstruct them from the lattice index.

A common debug-friendly dump command:

```
dump 1 text 5e-4 dump.run.* id i1 i2 d1 d2 d3 d4 d6 x y z
dump_modify 1 thresh i2 > 1   # keep only molten/solid/void sites
```

The `dump_modify thresh` line filters out inactive sites so the dump
files stay small.

### Other dump formats

The full list of dump styles available in this build is `text`, `sites`,
`stitch`, `image`, and `vtk`. There is **no** `dump hdf5` style — HDF5
in this codebase only appears on the *input* side, via the
`hdf5_unstructured` and `hdf5_csr` temperature sources that read
externally-prepared FE thermal traces.

* `dump ... stitch` — STITCH binary format, used by the
  `spparks-micro-analysis` Python tools. Fast to read/write and the
  preferred archival format, but not human-inspectable.
* `dump ... vtk` — ParaView/VTK XML. An alternative to OVITO for
  visualization.
* `dump ... sites` — minimal per-site format used by other SPPARKS
  apps; rarely the right choice for AM runs.
* `dump ... image` — direct PNG/JPEG snapshots without an external
  viewer. Useful for quick-look but limited control over the
  visualization.

For your day-to-day workflow as you learn the simulation, `text` is
the easiest format to debug. Switch to `stitch` when run sizes start
to make text dumps unwieldy.

---

## 6. Visualizing in OVITO

[OVITO](https://www.ovito.org) reads SPPARKS text dumps natively as
long as `x y z` are in the column list. Drag any `dump.moser.*` file
into OVITO and it auto-detects the wildcard and loads the full time
series; the columns become particle properties (use the `Color coding`
modifier to color by `i1` for grain id or `d6` for temperature).

If a dump opens with everything at the origin, you forgot `x y z` in
the dump command — re-run with them included.

---

## 7. Going further

### Other Moser examples in this directory

| Script | Demonstrates |
|---|---|
| `in.additive.moser` | single-pass baseline |
| `in.additive.moser_multiscan_fluct` | 5-pass scan with stochastic σ_W, σ_D, σ_P fluctuations |
| `in.additive.moser_keyhole` | overlapping double-ellipsoid (Goldak) keyhole mode with independent top/bot streams |

For Rosenthal counterparts (steady-state, no fluctuations) see
`../rosenthal_am/`.

### Setting up substrate orientations

To assign a single-crystal substrate at a chosen orientation, use the
helper `scripts/crystal_to_quat.py` to convert from crystal directions
(or Bunge Euler angles) into the scalar-first quaternion SPPARKS
expects in the d1..d4 site fields:

```bash
# [001] along build (+Z), [100] along scan (+X) — identity
python scripts/crystal_to_quat.py --normal 001 --inplane 100

# Emit a ready-to-paste SPPARKS `set` line for substrate spin 7
python scripts/crystal_to_quat.py --normal 110 --inplane 001 --as-set 7
```

See `scripts/README.md` for full options.

### Adding fluctuations

The Moser source supports per-emission stochastic variation via
`laser_fluctuations psd ...`. Order matters:

```
setup_temperature_source moser ...
laser_fluctuations psd lorentzian sigma_W 0.05 sigma_D 0.06 \
                       tau 200e-6 rho 0.5 seed 12345 dt 5e-6
laser_path start ... end ... speed ...
```

See `doc/laser_fluctuations.txt` for the four PSD shapes (white,
lorentzian, pink, narrow_band) and the keyhole top/bot/both selector.

### Documentation

* `doc/app_additive_texture.txt` — app overview and required commands
* `doc/temperature_source_moser.txt` — Moser source physics + arguments
* `doc/temperature_source_rosenthal.txt` — Rosenthal source (steady-state alternative)
* `doc/setup_temperature_source.txt` — the modular temperature-source factory
* `doc/laser_path.txt`, `doc/laser_fluctuations.txt` — path + fluctuation commands

### Common pitfalls

SPPARKS will catch these at parse time with explicit errors, but it's
useful to know them up front:

* Calling `laser_path` before `setup_temperature_source` — the active
  source has to exist first.
* Calling `laser_fluctuations` after `laser_path` — fluctuation tables
  are materialized inside `laser_path`, so any later spec is silently
  dropped (now caught with an error).
* In keyhole mode, `f_top + f_bot` must equal 1 within 1e-6 — a
  diagnostic error reports the actual sum if you're off.
* The `app_style additive_temperature_texture` (underscore form) is
  accepted as a deprecated alias for `additive/texture` — you'll see a
  warning. Use `additive/texture` in new scripts.

### Where to ask

For repo-specific questions, check `CLAUDE.md` at the repo root and
the per-section `*.md` files. The companion `spparks-micro-analysis`
repo has its own getting-started material for postprocessing.
