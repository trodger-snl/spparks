# Design doc: `additive/texture/kk` as an embeddable Kokkos library with in-memory temperature coupling

> **Status: DESIGN — not yet approved for implementation.** The temperature
> handoff contract (mesh layout, per-rank node/halo coverage, View ownership,
> update cadence) must be settled **with the thermal-app developer** before
> coding begins. This document is the basis for that conversation.

## Key difference from the HDF5 model: no time interpolation

The HDF5 source stores a per-node **time series** and, each step,
binary-searches + linearly interpolates each node's samples to the current time
(`precompute_nodal_temperatures` on the host, or the device `ComputeNodalTemps`
CSR path). **In-memory coupling does not need any of that.** The thermal app
advances its own solve and hands over the **current** node temperatures at the
current time, so the in-memory source drops the entire time dimension —
no `times`/`temps` CSR, no `nodeOffsets`, no time search, no per-step time-interp
upload. It reuses **only the spatial machinery** (point-in-tet element cache +
barycentric blend). This makes the in-memory source strictly *simpler* than the
HDF5 one: a static per-layer spatial cache + a live nodal-temperature View.

## Context

Today the additive/texture Kokkos app ingests an externally-computed temperature
field from **HDF5 files** (an FE thermal solve is run offline, dumped to disk,
then replayed). The goal is to couple it directly to a live **AM thermal
simulation** (another Kokkos/GPU code) so the temperature field is handed off
**in GPU memory each timestep** instead of round-tripping through files —
eliminating I/O, staging, and time-series storage.

The good news from exploration: SPPARKS already has ~80% of the embedding
machinery, and the temperature path already has a **data-driven, FE-mesh device
template** (the HDF5 unstructured source) that the in-memory coupling can largely
reuse. This plan turns that into a supported library + typed C++ coupling API.

**Decisions locked in (from the user):**
- Thermal grid is a **different FE/nodal mesh** → reuse the HDF5 barycentric
  interpolation (per-site element cache), not a direct per-site copy.
- **Zero-copy, in-memory** handoff of node temperatures as a `Kokkos::View<double*>`
  by reference — no host round-trip. Topology is **not** a 1:1 co-location: each
  small ROI runs on the rank(s)/GPU(s) overlapping its box (often a single GPU),
  and its temperature/mesh patch is gathered from the owning thermal rank(s). See
  "Deployment shape" below. (Thermal layout for scale: 112 CPU cores + 4 H100/node,
  4 ranks/node.)
- Coupling surface is a **typed C++ header** the thermal app includes.
- Deliver the **full path as ordered phases** so we can stop at any milestone.

## Architecture facts this plan builds on

- **Library already exists.** `src/library.{h,cpp}` = LAMMPS-style C API
  (`spparks_open(argc,argv,MPI_Comm,&ptr)`, `spparks_command`, `spparks_file`,
  `spparks_extract("darrayN"/"xyz"/...)`, `spparks_close`). `src/CMakeLists.txt`
  already has a **main-free** `spparks_lib` target (`SPPARKS::spparks`) that
  **PUBLIC-links `Kokkos::kokkos` + defines `SPK_KOKKOS`** and installs a
  `find_package(Spparks)` config (`SPPARKS_BUILD_LIB=ON`).
- **Kokkos runtime is owned only by `src/main.cpp`** (`Kokkos::initialize/finalize`
  at lines 114/122, plus the GPU rank→device binding helper
  `spk_bind_gpu_by_local_rank`). The library path never touches Kokkos — so an
  embedder that already called `Kokkos::initialize` coexists cleanly. There is no
  `KokkosSPPARKS` singleton.
- **`SPPARKS` is per-instance on its ctor `MPI_Comm`** (`world` = its sub-comm);
  core + `additive/texture/kk` hot path use `world` throughout. (A few peripheral
  paths hardcode `MPI_COMM_WORLD`: STITCH dumps, `set.cpp`, bezier AM apps — avoid
  those when embedding on a sub-comm.)
- **Temperature seam.** `TemperatureSource` (base, `src/temperature_source.h`) has
  a string factory `create_temperature_source(type, spk)` in
  `src/temperature_source.cpp` (rosenthal / moser / hdf5_unstructured / hdf5_csr).
  `ThermalManager` (`src/thermal_manager.{h,cpp}`) owns the source and reaches the
  app through a POD `ThermalHost` snapshot (`src/thermal_host.h`,
  populated by `refresh_thermal_host()`).
- **Device fill pattern.** Each source exports a Kokkos-free **plan POD**
  (`rosenthal_device.h` / `moser_device.h` / `hdf5_device.h`); the app fills
  `Kokkos::DualView<double*> k_T` on the GPU via `fill_temperature_*_device(plan)`
  over all `nall = nlocal+nghost` sites; a registered
  `std::function<double(double)>` filler (`set_{moser,hdf5}_device_filler`) is
  driven by `{moser,hdf5}_advance_device`. Dispatch is a 4-way branch in
  `AppAdditiveTextureKokkos::app_update()` (`src/KOKKOS/app_additive_texture_kokkos.cpp:2374`).
- **The HDF5 device path is the exact template.** Host does the mesh element
  search once per layer → per-site cache (`build_site_element_cache()`:
  4 tet-node indices + 3 barycentric weights + valid flag, from `App::xyz*dx`);
  device does the barycentric blend of `cached_nodal_temps` into `k_T`
  (`get_temperature_at_site_fast()`, `temperature_source_hdf5_unstructured.h:179`).
  Cache covers owned **+ ghost** via a chunk-load pad, so ghosts fill positionally
  with no T ghost comm. **The in-memory source differs only in where the mesh and
  the per-node temperature slice come from** (a live device View vs file reads +
  time interpolation) — everything downstream (melt/nucleate/solidify/smooth,
  `src/KOKKOS/app_additive_texture_kokkos.cpp:2530+`) is source-agnostic.
- **Run driving.** `run <dt>` → `App::run` → `iterate_app_update_only`
  (`src/app_lattice.cpp:731`) loops `app_update(dt); time += dt` to stoptime;
  `pre no post no` skips re-init/finish. `app_update(double)` is already **public**
  (`src/app_lattice.h:90`), but raw `app_update` does **not** advance `time` or run
  `output->compute` — those live in `iterate_app_update_only`, which is why a
  single-step wrapper is still wanted (bookkeeping, not an access-level problem).

## Deployment shape: many small, transient ROIs (not one co-located domain)

The real usage is **not** one SPPARKS domain sized to the thermal domain. It is:

- **SPPARKS domain ≪ thermal domain.** Each microstructure region-of-interest
  (ROI) is a small spatial box inside the much larger thermal domain.
- **Active over a sub-window of time**, not the whole thermal run — SPPARKS
  evolves a region only while it matters (melt/solidify passing through).
- **Many independent ROIs** at different places and times, with **no coupling
  between them**.
- **Each ROI may be tiled into subvolumes** for memory (the lattice is much
  *finer* than the thermal mesh, so a full ROI may not fit at once) — done today
  by reading/writing **STITCH** files.
- Microstructure is **cheaper per-op than the thermal solve but on a much finer
  lattice** — so memory, not flops, is the SPPARKS-side constraint.

**Proposed architecture — an ROI manager over per-ROI SPPARKS instances:**

- **ROI manager** (a coupling-layer object on the thermal-app side) owns a list of
  ROIs, each `{spatial box, time window, subvolume tiling, STITCH store}`. It is
  driven by the thermal app's own time loop: as thermal time advances it **opens**
  ROIs entering their window, **drives** active ROIs each step (hand them the
  local mesh patch + current node temps, in-memory), and **closes + persists**
  ROIs leaving their window. Only currently-active ROIs consume compute/memory —
  memory-bounded and fully file-free *for temperature*.
- **Each ROI = one (or a sequence of) SPPARKS instance(s).** Independence maps
  directly onto SPPARKS's per-instance `MPI_Comm` object model — so **multiple
  concurrent instances are now a hard requirement**, which makes the Phase 1
  statics→members cleanup mandatory (not cosmetic). ROIs run on disjoint
  rank/GPU sets concurrently, or in sequence, per the manager's schedule.
- **Local mesh patch, not the whole mesh.** Because an ROI box is small, its
  SPPARKS instance registers only the thermal elements overlapping its box (+ the
  interpolation pad) — a small, bounded gather from the thermal rank(s) owning
  that region. This is a big simplification vs a global mesh, and the barycentric
  element cache is exactly the coarse-mesh→fine-lattice mapping the ROI needs
  (already how the HDF5 source interpolates a coarse FE mesh onto fine sites).
- **Subvolumes via STITCH (kept for v1).** Within an ROI, tile into subvolumes;
  per subvolume: `set <field> stitch <file> <time>` (`src/set.cpp:839`) loads its
  state, evolve with in-memory temperature, `dump … stitch` (`src/dump_stitch.cpp`)
  persists it. STITCH owns the tiled store + inter-subvolume halo. Only
  **temperature** moves in-memory; microstructure state still round-trips STITCH
  (scoped change, per decision). **Blocker:** both `stitch_open` calls hardcode
  `MPI_COMM_WORLD` (`src/set.cpp:863`, `src/dump_stitch.cpp:118`) — wrong once
  instances live on sub-communicators; must take the instance's `world`.
- **Footprint: single-GPU-first, multi-rank-capable.** Support an ROI (streaming
  its subvolumes) on **one GPU** first — the manager just schedules single-GPU
  instances. Keep the manager + coupling API rank-agnostic so a large ROI can
  later span a sub-communicator (multi-rank patch/temperature gather). *(Footprint
  per ROI is not yet decided; design for both.)*
- **Time cadence.** SPPARKS may sub-cycle finer than the thermal step. Since the
  source does **no time interpolation**, between thermal updates the node-temp
  View is either held piecewise-constant or refreshed by the thermal app at
  SPPARKS cadence — a contract detail to settle with the thermal dev; the
  "current temps, no interp" simplification is unchanged either way.

## Target platforms & memory model (two clusters)

The coupling must work on **two different memory architectures**, which changes
the handoff data path:

1. **NVIDIA H100 cluster (discrete GPU).** Host and device memory are separate.
   The node-temperature `Kokkos::View` handoff must be **device-resident**, both
   codes must share the same CUDA device/context, and "zero-copy" means literally
   the same device pointer. Running SPPARKS on the idle CPU cores instead would
   cost an explicit device→host copy of the node temps each step (PCIe).

2. **AMD shared-memory cluster (APU: MI300A-class; also the gfx1151 Strix Halo
   box this checkout targets).** Host and GPU share **unified physical memory**.
   This **collapses the central design fork**: there is no separate device memory,
   so one buffer is cheaply accessible from both CPU and GPU. Consequences:
   - The zero-copy-device-device vs device-host-copy distinction **disappears** —
     a single shared/managed-space buffer serves both codes with no `deep_copy`.
   - SPPARKS's host↔device bookkeeping (`host_T_stale`,
     `sync_temperature_to_host`, `sync_temperature_to_device`) becomes cheap/near
     no-op.
   - Running SPPARKS microstructure on the **CPU cores (Kokkos OpenMP)** while the
     thermal solver produces temperatures on the GPU costs **no transfer** — so
     the "avoid GPU contention" fallback that is expensive on H100 is nearly free
     here, and may be the preferred layout on APUs.

**Design implications (thread these through all phases):**
- Kokkos backend is a **build-time choice** (CUDA for H100, HIP for AMD); the
  coupling code must compile for both from one source. Nothing in the coupling
  layer may hard-code CUDA/HIP.
- The coupling API (`src/spparks_am_coupling.h`, Phase 3) must be written against
  **Kokkos memory/execution-space abstractions**, not a concrete space. The
  handoff-View memory space must be one **both codes can access** — agree it with
  the thermal dev per platform: device space on H100, a **shared/managed space**
  (e.g. `SharedSpace`/HIP managed) on the APU.
- Two build+validation configurations to maintain: **CUDA/H100** and **HIP/AMD**.
  The Phase 7 temperature check is bit-for-bit **per backend** (inmemory-vs-file on
  the *same* backend) — **NOT** CUDA==HIP: the build uses `-ffast-math
  -funroll-loops` (`src/CMakeLists.txt:80`), so FMA contraction makes CUDA and HIP
  bits differ. FP precision (`SPPARKS_FP32`) is fixed identically per deployment.

## Phased implementation

### Phase 0 — Prove the library is embeddable next to an external Kokkos runtime
Goal: a standalone Kokkos exe that owns Kokkos, links `libspparks`, opens SPPARKS
on a sub-comm, runs an **existing** HDF5 additive input, and closes — with no
double Kokkos init.
- Build with `-DSPPARKS_BUILD_LIB=ON` (+ existing KOKKOS/HighFive options) and
  confirm `libspparks` includes the KOKKOS + additive sources and installs the
  `Spparks` package (`src/CMakeLists.txt`).
- Extract the Kokkos init + GPU rank→device binding out of `src/main.cpp` into a
  small reusable helper (new `src/kokkos_env.{h,cpp}`, e.g.
  `spk_bind_gpu_by_local_rank()` + a thin `spk_kokkos_scope`): the exe keeps
  calling it; an embedder that already owns Kokkos skips it. Library code still
  never calls `Kokkos::initialize`.
- Deliverable: `examples/coupling/driver_smoke.cpp` (links the lib, calls
  `Kokkos::initialize` itself, drives an existing `in.*_kokkos` via
  `spparks_command`). This validates ABI/Kokkos coexistence before any new physics.
- **Naming caveat:** in-tree the alias is `SPPARKS::spparks` (`src/CMakeLists.txt:745`)
  but the **installed** export uses namespace `Spparks::` (`:893/900`), so an
  external `find_package(Spparks)` consumer links `Spparks::spparks_lib`. Align the
  two (pick one namespace) so the documented consume path works out-of-tree.

### Phase 1 — Make the app instance-clean and externally steppable
- Convert the function-local `static` counters/timers in
  `AppAdditiveTextureKokkos::app_update` (`static int step_count`, `static double
  t_temp_update…`, ~line 2352) to **instance members** so two embedded instances
  don't share state (cosmetic today, but required for a clean library).
- Add a **single macro-step** entry: a `spparks_am_advance(handle, dt)` library
  function (+ an `AppAdditiveTextureKokkos::advance(dt)` wrapper mirroring
  `iterate_app_update_only`'s body: `app_update(dt); time += dt;` + `output->compute`).
  `app_update` is already public, so this is about the missing time/output
  bookkeeping, not access. Avoids the `init/setup/finish` cost of `run <dt>` per
  micro-step. (`run <dt> pre no post no` remains the fallback for v1.)
- **Broaden the statics fix to a tree-wide sweep**, not just the two `app_update`
  timers: also `construct_parent_args()` (`src/KOKKOS/app_additive_texture_kokkos.cpp:47`,
  function-local `static char*[]` written on every construction — a race if two
  ROIs construct concurrently) and the HDF5-source timing statics. **Mandate
  serialized ROI construction** in the manager contract so construction-time
  statics are safe.

### Phase 2 — In-memory FE-mesh temperature source (core)
- **Refactor (treat as a first-class, gated step — NOT a lift-and-shift).**
  `HDF5UnstructuredTemperatureSource` is ~2000 lines and the spatial machinery
  (`SpatialGrid`, `build_site_element_cache()`, `get_temperature_at_site_fast()`,
  the device cache path) is interwoven with HDF5-specific state: chunk loads that
  populate `nodeCoords`/`elemNode`, the time-series CSR, and a **compacted local
  node index space**. Extract a clean `UnstructuredMeshInterpolator` that takes
  mesh + node-temps from an **abstract provider in the thermal app's global node
  index space** (resolve the index-space/renumber question up front — see the
  zero-copy caveat below), used by both HDF5 and the new source. **Gate it:** land
  the refactor with the existing HDF5 device path proven **bit-identical** (same
  node order, same accumulation order) *before* any in-memory work — the Phase 7
  acceptance test depends on the extracted kernel being arithmetically identical
  to today's.
- New source `InMemoryMeshTemperatureSource` (`src/temperature_source_inmemory.{h,cpp}`),
  registered as type `"inmemory"` in the factory (`src/temperature_source.cpp:78`).
  It holds **handles** to the thermal app's mesh (node coords + tet connectivity,
  device Views, static within a layer) and the **current** per-node temperature
  `Kokkos::View<double*>` (updated each macro-step). Unlike HDF5 there is **no
  time series and no file I/O** — the slice is already "now", so
  `precompute_nodal_temperatures`/time-interp are dropped; the cache-build +
  barycentric blend are reused verbatim.
- New plan POD `src/inmemory_device.h` (mirrors `hdf5_device.h` **minus the time
  fields**: `valid`, `version` for cache regen on layer/mesh change, `nsites=nall`,
  `n_nodes`, `default_temp` — no `sim_time`, no time-series Views), plus
  `fill_temperature_inmemory_device(plan)` and a
  `set_inmemory_device_filler` / `inmemory_advance_device` pair on `ThermalManager`
  (`src/thermal_manager.{h,cpp}`), and a 5th dispatch branch in `app_update()`
  alongside the HDF5 branch (`src/KOKKOS/app_additive_texture_kokkos.cpp:2424`).
  The device filler aliases the thermal app's node-temp View as the nodal slice.
- **Zero-copy is only partial — state the contract.** Only the *per-step nodal
  slice* can be aliased. The element cache is **built on the host** over the mesh
  node coords/connectivity, so the thermal app's mesh must be staged host-side once
  per layer regardless. And aliasing the node-temp View requires the cache to index
  the thermal app's **node numbering**; if the interpolator instead uses a
  compacted local slice (as HDF5 does today), the alias needs a gather/renumber and
  is no longer zero-copy. Settle this in the interpolator's provider interface
  (Phase 2 refactor).

### Phase 3 — Typed C++ coupling header (user-facing API)
- New `src/spparks_am_coupling.h` (+ `.cpp`): a thin C++ class the thermal app
  includes. Responsibilities:
  - Construct/own a `SPPARKS` on a caller-supplied `MPI_Comm`; run the
    additive/texture/kk **setup script** internally (`app_style
    additive/texture/kk …`, `dimension/lattice sc/26n/region/create_box/create_sites`,
    `sweep color`, temperature-source `inmemory`, material params) so the thermal
    author writes ~10 lines, not a full input deck.
  - `register_mesh(node_xyz_view, tet_connectivity_view)` — the **local patch**
    overlapping this ROI's box + interpolation pad (not the global mesh); per layer
    (static within a layer). **Coordinate frame is a required contract:** the ROI is
    a *sub-box* of the thermal domain, so the API must fix the origin offset /
    translation between ROI lattice coords (`App::xyz * dx`, meters) and the thermal
    mesh's global coords. A wrong offset fails *silently* — every site lands outside
    all elements → uniform `default_temp`. Add an explicit "fraction of sites with a
    containing element" sanity check at registration.
  - **View lifetime / remesh handshake:** the caller must guarantee each registered
    View outlives every SPPARKS kernel reading it; on a thermal remesh/realloc the
    caller bumps the cache `version` to force cache rebuild + re-alias. Specify this
    invalidation handshake, not just the `version` field.
  - `set_node_temperatures(Kokkos::View<double*, Space>)` — per macro-step, **by
    reference, zero-copy**. `Space` is a template/typedef, not hard-coded: a
    device space on discrete H100, a shared/managed space on the unified-memory
    APU (see "Target platforms"). The API must not assume which.
  - `advance(dt)` — one macro-step (Phase 1 entry).
  - **Results egress (central to the close+persist model, not an afterthought):**
    define exactly what each ROI hands back before it is destroyed — spins/grain
    IDs (`iarray[0]`), orientation quaternions, phase/`active_flag`, in what
    coordinate frame, host or device, and at what cadence. Expose as device Views
    or via `spparks_extract("iarray1"/"darray…")` after an explicit
    `sync_app_data_to_host` (host `darray` copies are lazy/stale until synced). For
    subvolume-tiled ROIs the durable result is the STITCH store (Phase 6).
- The mesh↔site correspondence key is SPPARKS site physical coordinates
  (`App::xyz * dx`), exactly as the HDF5 element cache uses.

### Phase 4 — Library-mode error handling (prerequisite for multi-ROI robustness)
Goal: a failure inside one ROI must not kill the host thermal process.
- **Problem:** SPPARKS has no recoverable error path. `Error::all`
  (`src/error.cpp`) calls `MPI_Finalize(); exit(1)` and `Error::one` calls
  `MPI_Abort(world,1)` — so any bad input, missing element, or assertion inside an
  embedded ROI tears down the *entire* thermal application (and illegally finalizes
  the host's MPI). The "open/drive/close many transient ROIs" model presumes
  graceful per-ROI failure the code cannot deliver today.
- **Fix:** in library/embedded mode, make `Error::all`/`one` **throw** a SPPARKS
  exception instead of `MPI_Finalize`/`exit`/`MPI_Abort`; the coupling layer
  catches it per instance, marks that ROI failed, tears down only that instance,
  and continues. Never finalize MPI or `exit()` from library code. This is a
  prerequisite for Phase 5's "close ROIs" to be robust.

### Phase 5 — ROI manager + live scheduling (multi-instance)
Goal: drive many independent ROIs from inside the thermal app's time loop.
- New `MicrostructureRegionManager` (coupling layer): holds the ROI list
  `{box, [t_open,t_close], subvolume tiling, STITCH store}`; the thermal app calls
  `manager.advance(thermal_time)` once per thermal step. The manager **opens**
  ROIs entering their window (construct SPPARKS instance on the overlapping
  rank/GPU set, run the setup script, register the local mesh patch), **drives**
  active ROIs (push current node temps → `advance(dt)`, possibly sub-cycled), and
  **closes** ROIs past `t_close` (persist + destroy the instance to free memory).
- Requires the Phase 1 statics→members work; instances must be fully independent.
- Concurrency: independent ROIs on disjoint sub-communicators/GPUs; the manager
  schedules. Single-GPU instances first (see Phase 7 for multi-rank-per-ROI).
  **Concurrency model:** the app uses `Kokkos::DefaultExecutionSpace`/the default
  stream throughout (no per-instance stream), so two ROIs on the *same* GPU
  time-slice **serially** through the default queue — real concurrency comes only
  from mapping ROIs to **disjoint GPUs/ranks**. (A per-instance Kokkos stream is a
  later optimization if within-GPU overlap is ever wanted.)

### Phase 6 — Subvolume driving via STITCH (memory tiling)
Goal: let one ROI exceed device memory by streaming subvolumes, reusing STITCH.
- Within an ROI, tile the box into subvolumes; per subvolume, `set … stitch`
  loads state at the current time/region, evolve with the in-memory temperature
  patch, `dump … stitch` persists it; STITCH owns the tiled store + halo. Only
  temperature is in-memory (per decision). Drive this from the manager
  (subvolumes are the inner loop of an ROI step).
- **Fix `stitch_open` comm — the full audit, not two lines.** Drop
  `MPI_COMM_WORLD` for the instance's `world` at **all** STITCH/AM sites, not just
  the two originally cited: `stitch_open` at `src/set.cpp:863`,
  `src/STITCH/dump_stitch.cpp:118` (**tracked** copy — editing `src/dump_stitch.cpp`
  is silently reverted on reconfigure, per the dual-directory rule), **and
  `src/input.cpp:813`**; plus the further `MPI_COMM_WORLD` uses in `set.cpp`
  (`:900,901,974,1036`) and `dump_stitch.cpp` (`:209,210`). Confirm
  `potts_am_path_parser.cpp` (7 sites) and `app_potts_am_bezier.cpp` are off the
  `additive/texture/kk` path or fix them too. A half-converted STITCH path will
  deadlock/misroute on a sub-comm.

### Phase 7 — Multi-rank patch coverage + validation
- **Decomposition contract.** Each participating SPPARKS rank needs node/element
  coverage of its **padded** subdomain (the pad ≫ the one-cell ghost layer). v1:
  the thermal app hands each rank the local **+ halo** patch overlapping the ROI
  box; sites with no containing element fall back to `default_temp`. A
  rank-redistribution helper (and true multi-rank-per-ROI) is the add-on when an
  ROI is large enough to span several GPUs.
- **Validation (deterministic channel).** Per CLAUDE.md, only temperature is
  reproducible run-to-run. Take an existing HDF5 additive run; feed the **same**
  node temperatures/mesh through the `inmemory` source instead of the file; assert
  the temperature diagnostic (`diag_style array d6 sum d6 max`) matches the HDF5
  **device** path **bit-for-bit *per backend*** (inmemory-vs-file on the *same*
  backend — **not** CUDA==HIP; see the `-ffast-math` note under "Target
  platforms"). This is the acceptance test for Phases 2–3, then re-run at ROI scale
  for Phases 5–6.

## Critical files

- Build/embedding: `src/CMakeLists.txt`, `src/main.cpp` → new `src/kokkos_env.{h,cpp}`.
- Library API: `src/library.{h,cpp}` (add `spparks_am_advance`), new
  `src/spparks_am_coupling.{h,cpp}` (per-ROI instance API), new
  `src/microstructure_region_manager.{h,cpp}` (multi-ROI scheduling).
- Error handling: `src/error.cpp` (throw in library mode, never
  `MPI_Finalize`/`exit`/`MPI_Abort`).
- Multi-instance/STITCH: `src/set.cpp` (`:863,900,901,974,1036`),
  `src/STITCH/dump_stitch.cpp` (**tracked copy**, `:118,209,210`),
  `src/input.cpp:813` (`stitch_open` → instance `world`); confirm/handle
  `src/potts_am_path_parser.cpp` + `src/app_potts_am_bezier.cpp`.
- App: `src/KOKKOS/app_additive_texture_kokkos.{h,cpp}` (statics→members incl.
  `construct_parent_args`; `advance(dt)`; `fill_temperature_inmemory_device`;
  dispatch branch).
- Temperature: `src/temperature_source.cpp` (factory `"inmemory"`), new
  `src/temperature_source_inmemory.{h,cpp}`, new `src/inmemory_device.h`,
  `src/thermal_manager.{h,cpp}` (inmemory device hooks); refactor shared interp out
  of `src/temperature_source_hdf5_unstructured.{h,cpp}`.
- Demo: `examples/coupling/` (Phase 0 smoke driver + a mock-thermal coupling demo).

**Package dual-directory gotcha (CLAUDE.md):** edit `src/KOKKOS/…`, never the
installed `src/…` copy, and re-run `cmake -S src -B <build>` after editing a
`src/KOKKOS/` file before building, or the stale copy compiles silently.

## Risks / notes
- **ABI:** the thermal app and `libspparks` must link the **same Kokkos build +
  backend** (CUDA on H100, HIP on the AMD cluster) and matching flags. The
  temperature FP precision toggle (`SPPARKS_FP32`) must match on both sides of
  the View handoff.
- **GPU contention (platform-dependent):** SPPARKS microstructure kernels
  time-share the GPU with the thermal solver. **On H100** the fallback — run
  SPPARKS on idle CPU cores via the Kokkos **OpenMP** backend
  (`build-openmp` already works) — costs a device→host copy of node temps, so the
  primary design there is co-located device-device. **On the AMD unified-memory
  APU** that same CPU-execution fallback is nearly free (no transfer), so it may
  be the *preferred* layout — see "Target platforms & memory model".
- `additive/texture/kk` requires `sweep color` and the `sc/26n` lattice; the mesh
  physical `dx` must match the app's `dx` convention (meters).
- **Two-backend bring-up is real work, not a flag flip.** The H100/**CUDA** path is
  entirely unbuilt/untested in this checkout (which targets gfx1151/HIP and carries
  a bundled-Kokkos patch, `src/CMakeLists.txt:448`). Plan a **CI matrix** (CUDA +
  HIP) and budget the CUDA bring-up.
- **Small-kernel launch overhead.** Each ROI step launches ~10 kernels over a
  *small* active-front; for many small transient ROIs the GPU is underutilized and
  launch latency dominates on discrete H100. Consider batching / a per-instance
  stream / persistent kernels if profiling shows this bound — flagged, not designed.
- **MC determinism across instances.** RNG is per-instance (seed-based); validation
  leans entirely on the deterministic *temperature* channel. Define how seeds are
  assigned per ROI and what (if any) microstructure reproducibility is expected
  before building regression tests beyond the temperature check.
- **Piecewise-constant T is a physics-fidelity caveat, not just a contract.**
  Holding T frozen across many SPPARKS substeps within one coarse thermal step
  perturbs the solidification-band (`solid_d`) trajectory vs fine time stepping.
  Mechanically supported (each macro-step refills T); call out the fidelity
  trade-off so the thermal-step:SPPARKS-step ratio is chosen deliberately.

## Verification
1. **Phase 0:** build `-DSPPARKS_BUILD_LIB=ON`; build+run `driver_smoke` (links the
   lib, inits Kokkos itself) on an existing `in.*_kokkos` — exits clean, no
   double-init, output matches the standalone `spparks` exe.
2. **Phase 4 (error handling):** feed an ROI a deliberately bad input (e.g. mesh
   that leaves sites outside every element, or an illegal command); confirm the
   coupling layer catches it, closes only that ROI, and the host process + other
   ROIs keep running (no `MPI_Finalize`/`exit` from the library).
3. **Phases 2–3:** run the coupling demo that replays a known HDF5 field through
   the `inmemory` source; compare `d6 sum`/`d6 max` against the file-based HDF5
   **device** run on the **same backend** — require a per-backend bit-for-bit match.
4. **Phases 5–6:** a mock-thermal driver declares ≥2 ROIs with different
   space-time windows; confirm each opens/closes on schedule, evolves only in its
   window, streams subvolumes through STITCH on its own sub-communicator (no
   `MPI_COMM_WORLD` cross-talk between concurrent instances), and reproduces the
   single-instance temperature match within each ROI box.
5. **Phase 7:** repeat the match on ≥2 MPI ranks (halo coverage correct; ghost
   sites fill positionally with no T ghost comm); run the full matrix on both
   CUDA and HIP.
6. Commit per phase (CLAUDE.md), excluding `lib/kokkos`, `build-*/`, and
   `package_installer.cmake` unless asked.
