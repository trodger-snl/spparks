# Plan: distance-from-origin (grain-size) decay for misorientation rotation

Status: **PLANNED, not implemented** (design agreed 2026-08-08; deferred from the
original misorientation feature round where the user chose "nucleation only for
now"). Companion to the shipped features: nucleation-shell rotation +
orientation-gradient steepness cap (`8325eb71`) and the all-neighbors cap mode
(`ad9962d0`).

## Goal

Size-dependent misorientation: smaller (younger) grains rotate more than large
ones. Grain size is approximated by a per-site **accumulated growth distance
from the grain's birth site** — cheap, race-free, works identically on CPU and
GPU, no global grain-size bookkeeping (spin IDs collide, so per-spin site
counts are not usable).

## Agreed design decisions (user-selected)

1. **Decay law: power law** `f(d) = L0 / (L0 + d)` applied to the cooling-law
   rotation angle *before* the steepness caps clamp it. Caps stay hard limits.
   Rotation halves at `d = L0`; small residual persists in large grains; no
   discontinuities; single parameter.
2. **Pre-existing solid initializes MATURE** (substrate grains, stitch-loaded
   state, powder-activated sites): a large `d_origin` (e.g. `1e3 * L0` or an
   explicit sentinel) so regrowth from carried-over grains contributes little
   rotation. They are physically large existing grains.
3. **Persist across boxes via stitch**: add `d10` to the stitch dump columns
   and a `set d10 stitch ... last` line to the continuation template.
   Without this every box boundary silently resets the size proxy (all
   carried-over grains would re-init mature — acceptable-ish, but wrong for
   grains that nucleated late in the previous box). Stitch files grow ~14%
   (one more double field). StitchReader reads fields by name, so downstream
   analysis is backward-compatible.
4. **Smoothing/cleanup flips carry `d_origin`**: when `smooth_site`,
   `smooth_greedy_multiproposal`, single-voxel cleanup, or the kk smoothing
   kernels adopt a neighbor's spin+quaternion, copy the donor's `d_origin`
   too. A smoothed site joins its new grain at the donor's growth stage.
   (Leaving it stale would be the same post-cap leak that put the 7.3% floor
   under the neighbors-cap result.)

## Mechanism

New per-site double `d_origin` = darray index 9 (`d10` in input/dump syntax;
both apps currently use indices 0–8, ndouble 9 → 10).

Write rules (deterministic, no RNG):
- Nucleation seed (`execute_nucleation_event` / `NucleationKernel`): `0`.
- Nucleus shells (both particle flippers): distance from seed (consistent with
  the shells' existing cap-rate×distance rotation rule).
- Epitaxial solidification (`site_event_solidification` /
  `SolidificationKernel`): `d_origin[i] = d_origin[parent] + hop_distance_m`
  at the moment the parent is selected (both paths already compute the hop).
- Melt: no reset needed — overwritten on re-solidification by inheritance.

Read rule: in `apply_misorientation` (CPU) and the kernel rotation block (kk),
`mis_angle *= L0 / (L0 + d_origin[i])` before the cap clamp (parent or
neighbors mode).

New command: `misorientation_distance_decay <L0_m>`; `L0 <= 0` disables
(default off). Gated behind `max_misorient > 0` like the rest of the model.

## Touch points

CPU (`src/app_additive_texture.{h,cpp}`):
- `ndouble` 9 → 10; map `d_origin = darray[9]` in `grow_app`.
- `ghost_dindices`: add 9 (parent may be a ghost — the inheritance READS the
  neighbor's value, so it must be ghost-communicated). ~+14% ghost traffic.
- init_app: initialize mature for sites solid at t=0 unless loaded via stitch.
- Write rules above + `f(d)` in `apply_misorientation`.
- SiteState-based flips (smooth_site, greedy variant, flip_single_voxel_grain,
  nucleation flipper): copy donor `d_origin` (extend SiteState or copy beside).

Kokkos (`src/KOKKOS/app_additive_texture_kokkos.{h,cpp}`):
- Device View (simple View like d_G/d_V), allocate in `allocate_app_views`,
  wire into `sync_app_data_to_{host,device}`.
- Add to the selective ghost exchange: `sel_d_dev` {0,1,2,3} → {0,1,2,3,9} and
  `sel_d_host` likewise; extend `sync_ghost_to_host/device`.
- NucleationKernel (0), particle_flipper_kernel (dist from seed; the CAS-claim
  writes are single-writer so no race), SolidificationKernel (inherit + hop,
  then scale the angle), smoothing kernels (copy donor's value).
- Parser: `misorientation_distance_decay` mirrored in both apps.

Study templates (`am10_cpu_vs_hip/templates_*`):
- dump line: append `d10`; continuation template: add `set d10 stitch ... last`
  and include `d10` in the i2<2 reset guards if appropriate (mirror how d8/d9
  are handled).

## Validation plan

- Temperature (d6 sum/max) must stay bit-identical with the feature disabled
  AND enabled (rotation-amplitude change only; no RNG, no phase change).
- With L0 large (f≈1): results must match the no-decay binary statistically.
- Functional: per-grain mean KAM vs grain size scatter — decay should tilt it
  (small grains keep spread, large grains lose it). compare_misorientation.py
  already computes per-grain spread + sizes; add a size-binned KAM report.
- np=2 OpenMP run to exercise the ghost path for d_origin (parent on another
  rank), plus the standing GPU box-0 run.

## Known approximations / risks

- Growth distance ≈ path length along the dendrite growth direction, not
  radius; for equiaxed grains it overestimates size mildly. Acceptable proxy.
- The alternative "origin-anchored" construction (store the origin QUATERNION
  per site and set orientation = rotate(q_origin, target, cap_rate*d)) gives
  radially coherent non-accumulating fans but needs 4 more doubles + heavier
  comm. Revisit only if the modulation approach proves insufficient.
- Ghost-exchange volume is the multi-rank chokepoint (see
  reference_ff_allreduce_multirank memory); the +1 double is bounded but
  should be re-measured at np=8/16 after implementation.
