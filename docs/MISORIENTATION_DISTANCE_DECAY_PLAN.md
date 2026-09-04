# Plan: distance-from-origin two-regime misorientation cap

Status: **PLANNED, not implemented.** Supersedes the 2026-08-08 "power-law
decay" version of this plan (kept at the bottom as an alternative). Companion
to the shipped features: nucleation-shell rotation + orientation-gradient
steepness cap (`8325eb71`) and the all-neighbors cap mode (`ad9962d0`).

## What the 3D-EBSD papers show

Three TriBeam studies in `~/Downloads/3DEBSD` (Lamb et al., Acta Mater 2023,
CoNi GammaPrint-700 PBF-L; the 2026 Met Trans A cellular-structure follow-up on
the same alloy; Polonsky et al., 304L LENS). All measure misorientation against
a reference at the grain's *origin* (SROD: starting reference orientation
deviation) as a function of distance grown from that origin. Two regimes:

| regime | where | mean SROD slope (Lamb Fig. 5a/5c) |
|---|---|---|
| early | first build layer (~30 µm) after the grain originates | ≈ 3.5° over the first 30 µm (steeper still in the first ~10 µm) |
| steady | everything after, and established grains from the start | 0.7–1.0° per 30 µm (5c: ≈ 0.6°/30 µm) |

Met Trans A gives bulk-grain average / max rates of 1.7° / 5.8° per 30 µm.
Polonsky adds the qualitative rule: misorientation concentrates in the last
material to solidify and keeps accumulating when a grain re-solidifies across a
melt-pool boundary; it never resets.

The model's current cap (`misorientation_cap`, default 2°/30 µm) is a single
constant rate, i.e. the steady regime only. It cannot produce the knee.

## Design

Make the steepness-cap **rate a function of the site's growth distance from
its grain origin**, piecewise-constant with one knee:

```
rate(d) = r_fast   for d <  d_knee
          r_slow   for d >= d_knee
```

Every existing rotation event already computes `cap = rate * hop`. The
per-hop cap becomes the integral of `rate` over the hop from parent `p`
(distance `d_p`) to site `i` (`d_i = d_p + hop`):

```
cap = r_slow * hop + (r_fast - r_slow) * max(0, min(d_knee, d_i) - d_p)
```

One line, exact at the knee, reduces to the current cap when
`r_fast == r_slow`. It applies unchanged to all three rotation events:

- epitaxial solidification (`apply_misorientation` / `SolidificationKernel`),
  parent mode: `p` = inheritance parent;
- neighbors mode: each same-spin neighbour `j` contributes `cap_j` computed
  with its own `d_j`;
- nucleation shells (both particle flippers): `p` = seed / growth center.
  Shells sit at `d < d_knee`, so a nucleus fans out at `r_fast` — this is the
  "originating grain" early regime in Fig. 5a and is exactly where the paper
  puts it.

The cooling-law angle (`misorientation_function`) is untouched; the cap still
clamps it. To reproduce the measured curve the cap has to be the binding
constraint, which it already is on the current decks.

Established grains (substrate, stitch-loaded, powder-activated solid) start
**mature** (`d >= d_knee`) and only ever see `r_slow` — Fig. 5c. Re-melt then
re-solidification inherits `d` from the surviving parent, so a grain crossing a
melt-pool boundary keeps accumulating at `r_slow` and never resets (Polonsky).

### Input

```
misorientation_cap <deg> <dist_m> [parent|neighbors]          # unchanged: r_slow
misorientation_cap_initial <deg> <dist_m> <knee_m>              # new: r_fast for d < knee
```

`deg <= 0` in the new command disables the early regime (default: off, so
every existing deck is unchanged). Paper-calibrated values for GammaPrint-700:

```
misorientation_cap         0.85 30e-6
misorientation_cap_initial 3.5  30e-6  30e-6
```

## Mechanism: per-site growth distance `d_origin`

New per-site double `d_origin` = `darray[9]` (`d10` in input/dump syntax;
both apps use indices 0–8, `ndouble` 9 → 10). Meters. Deterministic, no RNG.

Write rules:
- nucleation seed (`execute_nucleation_event` / `NucleationKernel`): `0`;
- nucleus shells (both flippers): hop distance from the seed (CPU) / from the
  seed (kk) — same distance the shell rotation already uses;
- epitaxial solidification: `d_origin[i] = d_origin[parent] + hop_m` at the
  moment the parent is selected (both paths already compute the hop);
- smoothing / cleanup flips (`smooth_site`, greedy multiproposal,
  `flip_single_voxel_grain`, kk smoothing kernels): copy the donor's value —
  a smoothed site joins its new grain at the donor's growth stage;
- init: sites solid at t=0 get the mature sentinel (`d_knee` or larger)
  unless loaded from stitch; melt needs no reset (overwritten on
  re-solidification).

Read rule: the `cap` formula above, in place of `mis_cap_rate * hop`.

## Touch points

CPU (`src/app_additive_texture.{h,cpp}`):
- `ndouble` 10; `d_origin = darray[9]` in `grow_app`; add 9 to
  `ghost_dindices` (the parent may be a ghost and inheritance reads it).
- `mis_cap_rate` → `mis_cap_rate` (slow) + `mis_cap_rate_fast`, `mis_cap_knee`;
  one `cap_over_hop(d_p, hop)` helper used by `apply_misorientation` and the
  flipper's `rotate_flipped` lambda.
- Parser for `misorientation_cap_initial`.
- SiteState-based flips copy the donor's `d_origin`.

Kokkos (`src/KOKKOS/app_additive_texture_kokkos.{h,cpp}`):
- `View<double*> d_origin` like `d_G`/`d_V`; allocate, wire into
  `sync_app_data_to_{host,device}`; add 9 to the selective ghost exchange
  index sets and `sync_ghost_to_{host,device}`.
- `NucleationKernel` (0), `particle_flipper_kernel` (distance from seed; the
  CAS-claim write is single-writer), `SolidificationKernel` (inherit + hop,
  then the cap formula), smoothing kernels (copy donor). Same helper as a
  `KOKKOS_INLINE_FUNCTION`.
- Parser mirrored.

Study templates (`am10_cpu_vs_hip/templates_*`): dump `d10`; continuation
template `set d10 stitch ... last`, mirroring how `d8`/`d9` are handled.
Stitch files grow one double field (~14%); StitchReader reads by name, so
downstream analysis is backward-compatible. Without it every box boundary
re-inits carried-over grains as mature — correct for established grains,
wrong only for grains that nucleated in the last ~30 µm of the previous box.

## Validation

- d6 (temperature) sum/max bit-identical with the feature off and on.
- `misorientation_cap_initial` off, or `r_fast == r_slow`: microstructure
  statistically identical to the current binary (same RNG draws — the cap
  only changes the rotation magnitude).
- Functional: per-grain SROD vs distance-from-origin plot from the dump
  (`d10` gives the x axis for free, the seed orientation is the grain's
  min-`d10` voxel). Expect the Fig. 5a knee: ~`r_fast * d_knee` at the knee,
  then slope `r_slow`. `compare_misorientation.py` already computes per-grain
  spread; add the SROD-vs-`d10` binning there.
- np=2 OpenMP run so a parent lands on another rank (ghost `d_origin` path),
  plus the standing GPU box-0 run.

## Known approximations

- `d_origin` is path length along the growth direction, not grain radius;
  overestimates size mildly for equiaxed grains. Fine as a proxy.
- The model's cap bounds rotation per hop *relative to the parent*; the
  papers' SROD is relative to the origin. Those coincide only when successive
  rotations point the same way (gradient target). With `random` /
  `random_rotation` targets the rotations random-walk and SROD grows as √d;
  calibrate `r_fast`/`r_slow` against SROD only in gradient mode.
- Ghost-exchange volume is the multi-rank chokepoint (see the
  `reference_ff_allreduce_multirank` memory); +1 double is bounded but should
  be re-measured at np=8/16.

## Superseded alternative: power-law decay (2026-08-08 design)

`mis_angle *= L0 / (L0 + d_origin)` applied to the cooling-law angle before
the cap (`misorientation_distance_decay <L0_m>`). Same `d_origin`
bookkeeping, so it can be layered on later if a smooth tail is wanted, but the
papers' curve is a knee plus a constant slope, not a 1/d decay, and the cap —
not the cooling-law angle — is what binds on the current decks, so scaling the
pre-cap angle would mostly not show. The two-regime cap encodes the data
directly with the same per-site cost.
