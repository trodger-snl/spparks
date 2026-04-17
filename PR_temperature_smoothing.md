# PR: Solidification-Band Temperature Smoothing

## Summary

Adds an opt-in Gaussian smoothing pass over the temperature field `T[i]` for `AppAdditiveExtTempTexture`, gated to a user-specified temperature window around the solidification range. Also adds an optional in-situ isotherm-compactness diagnostic (`P / √A` at `T ≥ liquidus`) so users can tune smoothing strength quantitatively instead of by eye. Disabled by default — existing runs are bit-identical.

## Motivation

When the HDF5 CSR temperature source interpolates an unstructured tetrahedral thermal field onto the SPPARKS lattice, the linear (C⁰) tet interpolation produces visible faceting artifacts on isotherms — straight segments aligned with tet faces, chevron ripples inside the mushy band, and fragmented T ≥ liquidus regions. The artifact wavelength is ~5–10 sites at dx = 1 μm, amplitude ~±5 K, which coincidentally sits in the solidification band (1563–1623 K) and perturbs nucleation/growth dynamics.

The fix needs to:
1. Smooth only near the solidification band (not the entire field).
2. Preserve the pool's peak temperature and area.
3. Not bleed across sharp gradients into ambient powder (~300 K).
4. Be cheap enough to run every step.
5. Provide a way for the user to tell if it's "strong enough" without dumping and post-processing every run.

## Algorithm

Per timestep, after `update_temperature_from_source` fills `T[i]` from the modular source, run N passes of:

1. **Sync ghosts** — `comm->all_selective` over the existing `ghost_dindices` (T is already index 5).
2. **Site gate** — skip site `i` unless `T[i] ∈ [tmin − guard, tmax + guard]`.
3. **Neighbor gate** — for each of the 26 lattice neighbors, skip neighbor `j` unless `T[nj]` is also in the window.
4. **Gaussian weight** — `w_j = exp(−|r_j − r_i|² / 2σ²)` using actual `xyz[nj] − xyz[i]` offsets.
5. **Guard-band taper** — blend factor `α_eff` ramps linearly from 0 at the outer guard edge to `α` at the inner window edge. No discontinuity for sites crossing in/out of the window between steps.
6. **Commit** — `T[i] ← (1 − α_eff) · T[i] + α_eff · T_avg`, double-buffered so all sites see the same pre-pass field.
7. **Final ghost sync** — after the last pass's commit, `comm->all_selective` runs once more to leave ghost `T` consistent with the committed local `T`. Downstream code (phase-transition loop, diagnostic) can then read ghost `T` safely without an additional sync.

Across multiple passes, the gates re-evaluate against the **current** `T` values (not the pre-smoothing snapshot), so the active smoothing region tracks the thermal field as it evolves.

## Why Neighbor-Gating Matters

A first-cut implementation gated only the center site, not its neighbors. That caused ~150 K drops at the pool's top surface because a site at T = 1600 K on the melt top has ~9 of its 26 neighbors in the z+1 ambient layer at ~300 K. The Gaussian average was dominated by those cold neighbors, collapsing the pool peak by ~30 K globally and shrinking the T ≥ liquidus region by ~55%.

Skipping out-of-window neighbors turns the filter into an edge-preserving kernel: within-band smoothing proceeds normally, but the pool boundary is opaque to the kernel. With the fix, dT range dropped from (−157 K, +4 K) to (−12 K, +10 K) — symmetric and pool-area-preserving.

## Diagnostic: Isotherm Compactness

When `diag_interval > 0`, every N steps the app prints:

- `V` = count of 3D lattice sites with `T ≥ tl` (pool volume in sites)
- `S` = count of those sites with at least one neighbor where `T < tl` (pool surface in sites)
- `S / V^(2/3)`, compared against the 3D isoperimetric ideal `(36π)^(1/3) ≈ 4.836` for a perfect sphere

The ratio is **size-independent** — a perfect sphere scores 4.836 regardless of pool size, so the same threshold is meaningful across simulations with different laser power or pool dimensions. Higher ratios indicate a more jagged, elongated, or fragmented pool envelope.

Rule of thumb for interpretation:
- ~4.8–5.5 : near-spherical, tightly smoothed
- ~6–8    : typical elongated melt pool, convex, moderate smoothing
- > 10    : fragmented or highly faceted isotherm, smoothing likely too weak

These targets should be calibrated once per scan geometry by comparing an unsmoothed baseline against a well-smoothed run on the same input.

(An earlier iteration of this PR used the 2D ratio `P/√A` with the `2√π` ideal. That ideal only applies to a 2D slice; extended to 3D it's size-dependent and misleading. The current implementation uses `S/V^(2/3)` instead.)

Cost: O(nlocal) scan with a 26-neighbor inner loop that short-circuits on first `T < tl` neighbor, plus 4 scalar `MPI_Allreduce`s per report. Negligible vs the smoothing pass itself.

## Performance

Per smoothing pass:

| param | runtime effect |
|---|---|
| `passes` | linear multiplier (includes MPI ghost exchange per pass) |
| `tmin`/`tmax`/`guard` | controls N_in_band via early-exit; ~94% of the domain skips in one compare |
| `sigma` | **free** — appears only inside `exp()`, not in loop bounds |
| `alpha` | **free** — one multiply at blend |

The dominant cost is the 26-neighbor loop for in-band sites plus one `comm->all_selective` per pass.

## Changes

### `src/app_additive_ext_temp_texture.h`

- Declared `apply_temperature_smoothing()` and `compute_smoothing_diagnostics()`
- Added members: `temperature_smooth_enabled`, `smooth_tmin`, `smooth_tmax`, `smooth_guard`, `smooth_sigma`, `smooth_alpha`, `smooth_passes`, `smooth_diag_interval`, `smooth_buffer` (scratch vector)

### `src/app_additive_ext_temp_texture.cpp`

- **Constructor**: initializes smoothing members; disabled by default.
- **`input_app`**: new `temperature_smooth <tmin> <tmax> [sigma] [alpha] [guard] [passes] [diag_interval]` command with argument validation.
- **`update_temperature_from_source`**: calls `apply_temperature_smoothing()` after the `T[i]` fill loop (all three HDF5 paths + Rosenthal + generic xyz fallback).
- **`apply_temperature_smoothing()`**: implements the N-pass band-gated + neighbor-gated Gaussian filter with guard-band taper.
- **`compute_smoothing_diagnostics()`**: computes A and P over local sites, reduces via `MPI_Allreduce`, prints to screen on rank 0.
- **`app_update`**: after the step's ghost sync, conditionally invokes the diagnostic when `temperature_smooth_enabled && step_count % smooth_diag_interval == 0`.

## What Does NOT Change

- Default behavior: smoothing is off unless the user adds `temperature_smooth` to their input script. Existing runs are byte-identical.
- No new per-site arrays in the `darray`/`iarray` layout; `smooth_buffer` is a private scratch vector, not part of site state or ghost communication.
- `T` is already `darray[5]` and already in `ghost_dindices`, so no new ghost-communication wiring.
- All temperature sources benefit (HDF5 CSR, HDF5 unstructured, Rosenthal, Moser/Green's function) since smoothing runs after the source-specific fill loop.

## Usage

Minimal — use default `σ=1.2`, `α=0.4`, `guard=20 K`, `passes=1`:

```
temperature_smooth 1563 1623
```

Recommended after initial tuning for the user's dataset:

```
temperature_smooth 1563 1623 1.6 0.6 30 2
#                  tmin tmax σ   α  guard passes
```

With diagnostic reporting every 10 steps:

```
temperature_smooth 1563 1623 1.6 0.6 30 2 10
#                                         ^^ diag_interval
```

## Testing

1. **Regression**: run any existing input without `temperature_smooth` — `T[i]` values unchanged.
2. **Default settings**: add `temperature_smooth 1563 1623` and verify max |ΔT| stays small (< 20 K) and pool peak is preserved.
3. **Neighbor-gating correctness**: run with strong settings (`σ=2.0 α=0.7 passes=3`) and verify pool area shrinks by less than 5% relative to unsmoothed — if the neighbor gate were broken, this would shrink by >50% due to ambient bleed.
4. **Diagnostic ratio**: run with `diag_interval 10` and verify `S/V^(2/3)` decreases monotonically as smoothing strength is cranked up. Sphere-ideal is 4.836; typical convex elongated pool is 6–8; > 10 indicates fragmented isotherm / insufficient smoothing.
5. **Multi-pass consistency**: `passes 1` at `α = 0.8` should produce a similar but not identical result to `passes 2` at `α = 0.55` — validates that the per-pass gate re-evaluation isn't producing discontinuous behavior.

## Limitations and Known Tradeoffs

- Smoothing blurs real physical gradients inside the window alongside numerical artifacts. If the artifact amplitude exceeds the real cross-band gradient, smoothing will degrade fidelity. For the user's dataset, artifact amplitude (~5 K) is small vs. cross-band gradient (~60 K over ~15 sites), so the tradeoff is favorable.
- The `S/V^(2/3)` diagnostic captures *shape compactness* but not *internal* isotherm smoothness (it's a surface-only metric). A well-smoothed interior with a ragged outer contour can still score high; a very convex but internally speckled field will score well even if unsmooth.
- When smoothing is enabled, the `update_temperature_from_source` call happens *before* the `t_active` fast-forward scan. If `fast_forward_threshold` falls inside `[tmin − guard, tmax + guard]`, smoothing can shift step-level fast-forward decisions. For the default configuration (Rosenthal/Moser threshold = `tl`, smoothing window ends at `tmax = tl`), smoothing only denoises fast-forward boundary decisions — it does not flip them under normal parameter choices.
- The guard-band taper uses linear ramps, not a smooth blend. In practice this has been invisible in results because the taper is applied at site-by-site temperature bins, not spatial bins.
