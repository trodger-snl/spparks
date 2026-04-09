# Pull Request: Full Rosenthal Integration into `additive_temperature_texture`

## Summary

This change makes the analytical Rosenthal moving-point heat source a first-class temperature source for `AppAdditiveExtTempTexture`, alongside the existing `HDF5UnstructuredTemperatureSource` / `HDF5CSRTemperatureSource`. Three solution variants from Rolchigo et al. (Mod. Sim. Mater. Sci. Eng. 2021, Appendix B) are supported:

1. **Standard Rosenthal** (Eq. B4) — classical moving point source.
2. **Anisotropic Rosenthal** (Eq. B6/B7) — replaces `R` with `R_η = √(ξ² + (η_y y)² + (η_z z)²)` to elongate, widen, or deepen the melt pool.
3. **Keyhole Rosenthal** (Eq. B8/B9) — point source plus a depth-integrated line source for keyhole penetration; the line integral is evaluated by Gauss–Legendre quadrature.

The Rosenthal source is path-agnostic: the driving app owns the laser scan path using the existing `RASTER::Layer` / `RASTER::Path` classes from `am_raster.h`, and the source is queried in the pool-local moving frame each timestep.

The HDF5 path is unchanged; this PR adds a parallel branch to the source dispatch in `update_temperature_from_source()` and a new `rosenthal_path` input command.

## Motivation

Before this change, `AppAdditiveExtTempTexture::update_temperature_from_source()` always `dynamic_cast`'d to `HDF5UnstructuredTemperatureSource*` and short-circuited the entire temperature update if the cast failed. The factory could create a `RosenthalTemperatureSource`, but it was never invoked, the class was marked **UNTESTED** in its header, several declared methods were unimplemented stubs, and it reinvented its own scan-path machinery (linear / serpentine / spiral / custom) duplicating functionality already present in `am_raster.h`.

The goal of this PR is to:

- give users an analytical alternative to large external HDF5 thermal-history datasets, especially for parametric / repeated-scan studies where running a full FEA solve is wasteful;
- support the modified Rosenthal forms used in the literature for matching keyhole melt-pool shapes that the standard solution cannot reproduce;
- reuse the existing AM scan-path code (`RASTER::Layer`) rather than carrying duplicate path-generation logic in two places.

## User-Facing Changes

### `setup_temperature_source rosenthal …` (extended)

Three modes, all SI units:

```text
setup_temperature_source rosenthal standard <Q[W]> <lambda> <k[W/(m K)]> <alpha[m^2/s]> <T0[K]>
setup_temperature_source rosenthal aniso    <Q[W]> <lambda> <k[W/(m K)]> <alpha[m^2/s]> <T0[K]> <eta_y> <eta_z>
setup_temperature_source rosenthal keyhole  <Q[W]> <lambda_p> <lambda_l> <k[W/(m K)]> <alpha[m^2/s]> <T0[K]> <d[m]> [<n_quad>]
```

`n_quad` defaults to 8 and is clamped to `[2, 64]`. `T0` is the preheat / ambient temperature used for the Rosenthal evaluation; it can differ from the app's `t_room`.

### `rosenthal_path …` (new)

```text
rosenthal_path start <X0> <Y0> <Z0> end <X1> <Y1> speed <V> [repeats <N>]
```

- All coordinates and velocity are SI (meters, m/s).
- `Z0` sets the physical Z of the scan plane (the laser stays on this plane).
- `repeats N` builds `N` copies of the same path into the underlying `RASTER::Layer`. Between repeats the laser teleports back to `(X0, Y0)`; there is no built-in inter-pass dwell.
- After the final repeat is exhausted, the laser deactivates and all sites read `T = T0`.

### Removed: `scan_path` command

The previous `scan_path linear|serpentine|spiral|custom` command and its handler `setup_scan_path_cmd()` have been deleted. They were the only entry points to the dead path-generation code in `RosenthalTemperatureSource` and were not used by any checked-in input scripts.

## Implementation Details

### Rosenthal source — rewrite (path-agnostic)

`src/temperature_source_rosenthal.h/.cpp` was rewritten end-to-end:

- Added `enum class RosenthalMode { STANDARD, ANISOTROPIC, KEYHOLE }`.
- New query API in the pool-local moving frame:

  ```cpp
  double rosenthal_pointwise(double xi, double y_rel, double z_rel,
                             double v, double T0) const;
  ```

- Common kernel (Eq. B4 form) used by all three modes, with a singularity cutoff `r_min` so the `1/R` prefactor at the laser site stays finite:

  ```cpp
  double rosenthal_kernel(double Q_eff, double v, double xi, double R) const {
    if (R < r_min) R = r_min;
    return (Q_eff / (2*pi*k_cond*R)) * exp(-v*(xi+R)/(2*alpha));
  }
  ```

- **Standard** uses `R = √(ξ² + y² + z²)` and `Q_eff = λQ`.
- **Anisotropic** substitutes `R_η = √(ξ² + (η_y y)² + (η_z z)²)` into the kernel for both prefactor and exponent (per paper B6).
- **Keyhole** sums the surface point source `kernel(λ_p Q, ..., R)` plus a discretized line integral over `D ∈ [−d, d]` of `kernel(λ_l Q, ..., R'(D))` with `R' = √(ξ² + y² + (D + z)²)`. Quadrature nodes and weights are precomputed once at setup time by Newton iteration on the Legendre polynomial roots.
- `supports_time_queries()` returns `true` so the Rosenthal source participates in the existing fast-forward block in `app_update()`. The source itself returns `+inf` from `get_next_time_with_temperature()`; the actual prediction is owned by the app (see below).
- `set_r_min(double)` lets the driving app size the cutoff to its lattice spacing.

Deleted from the prior class:

- `enum ScanPathType`, all `linear_*`, `raster_*`, `spiral_*` member state
- `setup_scan_path`, `load_custom_path`, `generate_*_path`, `find_path_segment`,
  `interpolate_position`, `interpolate_velocity`, `parse_setup_arguments`,
  `validate_scan_path`, `get_laser_position`, `get_laser_velocity`
- The `cached_time` / `cached_laser_position` / `cached_laser_velocity` mutables
- The "UNTESTED" warning header comment

### Path machinery — `RASTER::Layer` directly, no `PottsAmPathParser`

`PottsAmPathParser` is not a mixin — it `: public AppPotts`, and `AppAdditiveExtTempTexture` already `: public AppPottsQuaternion : public AppPotts`. Inheriting both would create a diamond requiring virtual inheritance and surgery on `AppPotts`, with collateral impact on the four sister apps that already use `PottsAmPathParser` (`app_potts_am_weld`, `app_potts_am_bezier`, `app_potts_am_path_gen`, `app_am_ellipsoid`).

Instead, this PR uses the underlying `RASTER::` namespace classes from `am_raster.h` directly. They are standalone, have no SPPARKS coupling, and are exactly the building blocks `PottsAmPathParser` itself wraps:

- `RASTER::Path` — one linear segment defined by start/end/speed.
- `RASTER::Layer::move(dt)` — advances the laser; auto-rewinds when paths are exhausted.
- `RASTER::Layer::get_speed()` / `get_unit_dir()` / `get_position()`.
- `RASTER::Layer::compute_position_relative_to_pool(xyz, layer_z)` — returns the site coordinates `(ξ, y_rel, z_rel)` already rotated into the pool-local frame.

The app holds:

```cpp
RASTER::Layer scan_layer;
double scan_layer_z;       // physical Z of the scan plane (meters)
bool   scan_layer_active;  // false once all repeats are exhausted
bool   rosenthal_path_set; // true after rosenthal_path is parsed
```

`rosenthal_path_cmd()` parses the new command in ~30 lines, builds `repeats` copies of one `RASTER::Path` into a `std::vector`, constructs `scan_layer` from them with thickness 0 (the integer thickness is only used by `CartesianLayerMetaData`, not by `Layer` itself), sets `scan_layer_z` to the user-provided `Z0`, and calls `set_r_min(0.5*dx)` on the source.

### Units

`RASTER::Layer::move(dt)` does literally `dv = dt * speed` with no unit assumption; `Path` start/end/speed are raw doubles. Because we only use `RASTER::Layer` (not `CartesianLayerMetaData` and not `PottsAmPathParser::initialize_layers_am`), the integer-lattice-site assumption at `potts_am_path_parser.cpp:175` (`build_layer_z += layer_thickness`, where `layer_thickness` is `int`) **never executes**. The Rosenthal path is fully SI: meters, seconds, m/s.

### Driving the Rosenthal source from `update_temperature_from_source()`

The dispatch in `update_temperature_from_source()` is now:

```cpp
if (hdf5_source && opt_use_element_cache && opt_use_nodal_precompute) { /* fast path */ }
else if (hdf5_source && opt_use_element_cache)                        { /* lazy path */ }
else if (hdf5_source)                                                 { /* xyz fallback */ }
else if (ros_source) {
    if (!rosenthal_path_set) error->all(...);
    double t0 = MPI_Wtime();
    if (scan_layer_active) {
        if (!scan_layer.move(dt)) scan_layer_active = false;
    }
    const double v_laser = scan_layer.get_speed();
    const double T0      = ros_source->get_ambient_temperature();
    for (int i = 0; i < nlocal; i++) {
        double phys[3] = { xyz[i][0]*dx, xyz[i][1]*dx, xyz[i][2]*dx };
        RASTER::Point loc = scan_layer.compute_position_relative_to_pool(phys, scan_layer_z);
        T[i] = scan_layer_active
                 ? ros_source->rosenthal_pointwise(loc[0], loc[1], loc[2], v_laser, T0)
                 : T0;
    }
    g_t_temp_site_loop += MPI_Wtime() - t0;
}
else { /* generic xyz fallback for any other future source type */ }
```

`T0` is taken from the source's own `get_ambient_temperature()` rather than the app's `t_room` — the two can legitimately differ (e.g. paper Table B1 uses `T0 = 573 K` with the app default `t_room = 300 K`).

### Melt detection and fast-forward

The `for (int i = 0; i < nlocal; i++) if (T[i] > fast_forward_threshold) ...` block now uses:

- the HDF5 source's own `get_fast_forward_threshold()` if the source is HDF5;
- the app's liquidus `tl` if the source is Rosenthal;
- `t_room` otherwise (unchanged fallback).

For fast-forward, the new helper `AppAdditiveExtTempTexture::rosenthal_next_active_time(double current_time, double threshold_temp)` walks a **copy** of `scan_layer` forward in `fast_forward_search_window` increments (default 100 ms), evaluates `rosenthal_pointwise` at the local domain box-center for each future laser position, and returns the first time the box-center temperature reaches `threshold_temp`. Returns `+inf` if the path exhausts or no segment heats the local domain. The fast-forward block in `app_update()` calls this helper for Rosenthal sources and the source's polymorphic predictor for everything else.

### Base-class accessor

`TemperatureSource` gains a public accessor:

```cpp
double get_ambient_temperature() const { return ambient_temperature; }
```

`ambient_temperature` was already a `protected` member set by every derived class at setup time; this just exposes it so the driving app can read it without additional plumbing.

### Bug fixes along the way

- The previous Path-1 fallback in `update_temperature_from_source()` dereferenced `hdf5_source->get_dx()` even when the `dynamic_cast` had returned `nullptr` — it would have segfaulted on any non-HDF5 source. The new dispatch only takes the HDF5 fallback when `hdf5_source != nullptr`, and the generic fallback uses the app's `dx`.
- The original Rosenthal source was selectable via the factory but never invoked by the app. Selecting it would have produced silently-incorrect results.

### Issues found in code review

Three bugs were caught in review and fixed before merge:

1. **Rosenthal scan-path desync after fast-forward (high)**. The original Rosenthal branch in `update_temperature_from_source()` advanced the laser by exactly one `dt` per call. After a fast-forward jump in `app_update()` (which sets `time = target_time` and re-calls `update_temperature_from_source(time)`), the laser was still one `dt` ahead of where it had been before the jump — not `target_time - old_time` ahead. All post-skip Rosenthal evaluations therefore used the wrong laser pose.

   **Fix**: track a new `scan_layer_time` member that records the simulation time at which `scan_layer`'s pose is current. On every call into the Rosenthal branch, advance the layer by `simulation_time - scan_layer_time` in `dt`-sized chunks (`RASTER::Layer::move()` does not carry remainder across path boundaries, so taking many small steps is also the safe way to step across multiple repeats of a single path). `scan_layer_time` is initialized to the current sim time when `rosenthal_path` is parsed and updated after every advance.

   **Test**: new ad-hoc input file with the laser starting at x = −50 mm (well outside the local box) scanning at 1 m/s. Fast-forward correctly identifies and skips the 50 ms dead window, then 280 sites melt immediately at `t = 0.051 s` and 690 sites total resolidify by the end of the scan — proving the laser is at the right place after the jump.

2. **Fast-forward predictor was not provably conservative (medium)**. The first cut evaluated the Rosenthal temperature at the **center** of the local bounding box, which could under-estimate the true peak T on a rank whose subdomain is heated near an edge. That can in principle return a too-late next-active time and let fast-forward step past real heating events.

   **Fix**: replace the box-center check with a true upper bound. New helper `RosenthalTemperatureSource::rosenthal_peak_at_distance(R, T0)` returns the maximum possible T at any site whose distance to the laser is ≥ R:
   - The exponent `−v(ξ + R)/(2 α)` is maximized at zero (when ξ = −R, straight downstream), so the kernel reduces to `Q_eff_total / (2 π k R)`.
   - For ANISOTROPIC mode, `R_η ≤ R / min(η_y, η_z, 1)` shrinks the effective R, so the bound divides accordingly.
   - For KEYHOLE mode, the bound sums the point source `λ_p Q` plus a coarse upper bound on the line integral (`λ_l Q × Σ w_i`).
   - The result is clamped to `r_min` so it stays finite at the laser site.

   The predictor itself now computes the closest point on the local bounding box to the world-space laser position, takes the Euclidean distance, and tests `rosenthal_peak_at_distance(R, T0) ≥ threshold`. There is also a cheap early-out: if the unmodulated peak at `r_min` doesn't reach threshold, no laser position ever will → return `+∞`.

3. **`rosenthal_path` parser rejected its own documented minimal form (medium)**. The minimal syntax `start X0 Y0 Z0 end X1 Y1 speed V` is 9 tokens after the command name (indices 0–8), but the initial guard required `narg < 10` and errored out. The optional-`repeats` branch correctly checked `narg > 9`, so adding `repeats N` worked but the bare form was rejected.

   **Fix**: relax the guard to `narg < 9`. Verified by parsing a script that omits the `repeats` clause entirely; produces 574 melted sites (matches the explicit `repeats 1` form within RNG noise).

### Second-pass review issue

A subsequent review caught one more bug:

4. **Fast-forward `next_thermal_time` was computed locally on each rank with no global reduction (high)**. `rosenthal_next_active_time()` walks the laser past the **local** subdomain bounding box, so each rank legitimately predicts a different wake-up time depending on where its subdomain sits relative to the path. The original code then used that per-rank value directly to set `target_time`, advance `time`, and re-call `update_temperature_from_source()` — desynchronizing simulation time across ranks and risking MPI hangs in subsequent collectives (the temperature site loop, the powder activation reduction, etc.).

   **Fix**: insert an `MPI_Allreduce(..., MPI_MIN, world)` immediately after the local prediction so all ranks fast-forward to the **earliest** wake-up time across the communicator. This is the only safe choice — taking the max would skip past real heating events on other ranks. The reduction is unconditional (applies to both Rosenthal and HDF5 source predictors) for safety, with negligible cost since it's only one double per `app_update()` call and only when `global_t_active == 0`.

   **Test**: re-ran the dead-window fast-forward script (`-50 mm` start, 1 m/s, 4 ranks). Single skip from `t = 5 µs` to `t = 50.005 ms` is reported once on rank 0; melting begins immediately on the post-skip step (279 sites at `t = 51 ms`, 692 sites by `t = 52 ms`); standard regression example is unchanged (585 melted sites, within RNG noise of the prior 571/583).

## Files Changed

- `src/temperature_source.h` — new `get_ambient_temperature()` accessor on the base class.
- `src/temperature_source_rosenthal.h` — full rewrite (path-agnostic, three-mode, fast-forward stub).
- `src/temperature_source_rosenthal.cpp` — full rewrite (kernel, mode dispatch, Gauss–Legendre quadrature, validation, info print, no path code).
- `src/app_additive_ext_temp_texture.h` — `#include "am_raster.h"`; new `scan_layer`, `scan_layer_z`, `scan_layer_active`, `rosenthal_path_set` members; `rosenthal_path_cmd` and `rosenthal_next_active_time` declarations; removed `setup_scan_path_cmd` declaration.
- `src/app_additive_ext_temp_texture.cpp` — constructor initializes the new state; `input_app` routes `rosenthal_path` to the new handler and no longer recognizes `scan_path`; new `rosenthal_path_cmd()` (~50 lines); new `rosenthal_next_active_time()` (~50 lines); rewritten `update_temperature_from_source()` with HDF5 / Rosenthal / generic dispatch; removed `setup_scan_path_cmd()`; melt-detection and fast-forward blocks updated for the Rosenthal source.
- `examples/ReducedTempAM/QuatTest/in.additive.rosenthal_standard` — new (Table B1 Case 3).
- `examples/ReducedTempAM/QuatTest/in.additive.rosenthal_aniso` — new (Table B1 Case 2, η_y=0.89, η_z=0.4).
- `examples/ReducedTempAM/QuatTest/in.additive.rosenthal_keyhole` — new (Table B1 Case 1, λ_p=0.1, λ_l=0.036, d=300 μm, n_quad=8).

## Testing

### Build

Clean compile of all touched translation units under the `spparks-dev` Spack environment (`cmake --build .` after activating OpenMPI 4.1.8). The pre-existing `std::iterator` deprecation and `malloc` size warnings in unrelated files are unchanged.

### Runtime — Standard Rosenthal

`mpirun -n 4 spparks -in in.additive.rosenthal_standard` on a 60×40×20 lattice (1.2 mm × 0.8 mm × 0.4 mm at `dx = 20 μm`). 1 mm linear path along the top centerline at `v = 1.2 m/s`, run length 2 ms.

- Setup banner reports `Rosenthal temperature source [STANDARD]` with the parsed parameters.
- 583 sites melt and resolidify; `i2 = 2` (molten) sites visible during the active scan window with peak `T ≈ 2554 K` near the laser and a Rosenthal-shaped cooling tail (~1750–2554 K range across 12 cells of pool length).
- After the scan exhausts, all sites read back `T = T0 = 573 K`.
- Total wall time ≈ 30 ms for the full 2 ms simulation on 4 ranks.

### Runtime — Anisotropic Rosenthal

`mpirun -n 4 spparks -in in.additive.rosenthal_aniso` (Case 2 parameters, `v = 0.8 m/s`).

- 972 active sites at the dump midpoint vs 122 for the standard case at the same relative time — confirms the deeper, wider pool.
- Pool penetrates to `z = 13` (140 μm below the surface) vs `z = 18` (one cell below the surface) for the standard case, exactly the depth elongation predicted by `η_z = 0.4`.

### Runtime — Keyhole Rosenthal

`mpirun -n 4 spparks -in in.additive.rosenthal_keyhole` (Case 1 parameters, `v = 0.6 m/s`, `d = 300 μm`, `n_quad = 8`).

- 277 active sites at the dump midpoint with cooler peak `T ≈ 1457 K` (lower because `λ_p = 0.1` is much less than the standard case's `λ = 0.155`).
- Quadrature ran without instability; line-source contribution visible as a more uniform depth distribution.

### Repeated-scan smoke test

A quick re-run of `in.additive.rosenthal_standard` with `repeats 3` confirms three sequential scan cycles with the laser teleporting back to `(X0, Y0)` between repeats (no inter-pass dwell, as documented).

### Bug discovered and fixed during testing

The first end-to-end run produced melting (583 sites) but cooled sites returned to 300 K instead of the expected 573 K. Tracked down to the app passing `t_room` (300) into `rosenthal_pointwise` instead of the Rosenthal source's setup `T0` (573). Fixed by reading `T0` from `ros_source->get_ambient_temperature()` in both the per-step update and the fast-forward predictor. After the fix the post-cool diagnostic `sum(d6)` matches `48000 × 573 ≈ 2.75e7`.

### HDF5 regression

The rewritten dispatch keeps the three HDF5 optimization tiers byte-for-byte identical when the cast to `HDF5UnstructuredTemperatureSource*` succeeds; the only intentional behavior change there is that the previously-unreachable Path-1 fallback (xyz lookup) no longer dereferences a null `hdf5_source` pointer. The existing HDF5 example at `examples/ReducedTempAM/QuatTest/in.additive` was not re-run as part of this PR because it still uses the deprecated 6-arg `app_style additive_temperature_texture …` form with an embedded HDF5 file path (the same staleness flagged in `PR_misorientation_target.md`); modernizing it is left as a follow-up.

## Notes / Follow-Up

- **Inter-pass dwell**: between repeats the laser teleports instantly back to `(X0, Y0)` — there is no built-in cool-down. For most KMC microstructure studies that's fine. If the user later needs a dwell window, the cleanest patch is a `pass_dwell <seconds>` keyword on `setup_temperature_source rosenthal …` that holds `Q_eff = 0` for `dwell` seconds whenever `scan_layer.move()` crosses a path boundary. Out of scope for this PR.
- **Multi-segment paths**: the `rosenthal_path` command currently builds one `RASTER::Path` and (optionally) repeats it. Multi-segment scan trajectories would be a small extension — push additional `RASTER::Path` objects into the same `std::vector` before constructing `scan_layer`. The plumbing in `update_temperature_from_source()` already pulls speed live from `scan_layer.get_speed()`, so per-segment speeds work for free once the parser is extended.
- **Multi-Z builds**: not in scope. If/when needed, `RASTER::CartesianLayerMetaData::thickness` would need to become a `double` (currently `int`, lattice-site units in the `PottsAmPathParser` flow).
- **Modernizing `examples/ReducedTempAM/QuatTest/in.additive`**: the HDF5 example file is stale (old 6-arg `app_style` form, hard-coded macOS path). A follow-up to update it would also let the HDF5 path get end-to-end runtime regression coverage in CI.
