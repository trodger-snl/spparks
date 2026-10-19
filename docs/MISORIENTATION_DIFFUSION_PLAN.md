# Plan 1: diffusion-coefficient misorientation in `additive/texture` and `additive/texture/kk`

Status: **Part A implemented 2026-09-04** (commits fa8930f3, 63a68bad, 74c331e1);
Part B (the undercooling test matrix) is still to run. Evidence:
`spparks-micro-analysis/analysis/surge_srod/srod_random_walk.png`, `spparks-micro-analysis/analysis/surge_srod/srod_experimental_analysis.png`,
`spparks-micro-analysis/analysis/surge_srod/srod_random_walk.py`. Companion: `docs/GP700_MOSER_SROD_DECK_PLAN.md`
(Plan 2, the validation deck).

## Implementation record (Part A)

What landed, in both apps unless noted:
- `misorientation_diffusion <D_deg2_per_m> <thresh_deg> [alpha]` as in A1;
  `misorientation_function` / `misorientation_cap` / `misorientation_target
  gradient` are parse errors with a migration hint. `nucleation_misorientation_target`
  takes `on|off|none`. Defaults: rotation off, `random_rotation`, shells on.
- Nucleation shells: `sqrt(D_eff * distance_from_seed)` relative to the seed
  (CPU flipper re-rooted onto the seed; kk unchanged in form). kk
  `t_cool_max` is now `tl - ts` (was 100 K).
- Check deck `examples/misorientation_walk/in.walk.{cpu,kk}` and
  `spparks_micro_analysis.srod_analysis` (spparks-micro-analysis) (origin-referenced SROD vs distance, D fit,
  width ratio).

Validation (Rosenthal one-pass deck, dx = 5 µm, D = 4.2e5 deg²/m, 4 seeds):
- `examples/am_validation/compare.sh`: d6 sum/max bit-identical CPU vs kk
  on all three decks.
- Per-hop step exact in both apps: with smoothing off (`nrefine 0`) the
  axis-neighbour same-grain disorientation has a sharp peak in [1.4, 1.5)°
  (expected sqrt(D·dx) = 1.449°) holding 19% (CPU) / 29% (kk) of pairs; the
  rest are cousin pairs.
- Width ratio (p95−p5)/mean = 1.3–1.5 in both apps at every distance bin
  (Maxwell 1.38). SROD grows as √d.
- **Calibration factor to remember:** fitting RMS² = D_fit·(Euclidean
  distance from the lowest-z voxel) gives D_fit/D = 1.3 (CPU, no smoothing;
  0.88 with the default `nrefine 2` smoothing, which copies quaternions) and
  ≈ 2.0 (kk, HIP and OpenMP alike). The walk accumulates along the
  inheritance *path*, which is longer than the straight-line distance, and
  the lowest-z origin is only approximately the true origin. The experiment
  measures SROD against Euclidean distance, so the B-stage gate must compare
  the same quantity and the input D will need to be scaled down by the
  app's path factor (~0.8 CPU-with-smoothing, ~0.5 kk on this deck). Why kk
  paths are more tortuous than the CPU app's (parallel front vs sequential
  random sweep) is worth a look but is not a rotation-code issue.
- alpha = 5 runs clean (no NaN); on this deck nearly every site solidifies
  at Tcool ≥ tl − ts, so g ≈ 1 and the step is unchanged — the deck is too
  hot to exercise alpha, which is what the Rosenthal singularity does.
- kk np = 2 (OpenMP) runs clean.
- **kk left the pool liquid (fixed, commit after 74c331e1):** on this
  Rosenthal 400 W deck the kk app left 14 084 sites liquid at 573 K after the
  pass. Root cause: once the field is flat the gradient is zero, the cosine
  helper returns 0 (a bogus 90°), and the kk texture defaults (c1 0.1, c2 0.9)
  made the mobility c1 + c2·cos(2.5·90°) negative for every candidate, which
  the kernel rejected outright. Both apps now treat a zero gradient as "no
  preferred direction" (full mobility), clamp negative mobility to a
  zero-weight candidate, and share the CPU defaults (0.5, 0.5, 2.5). kk now
  solidifies exactly the CPU's 21 740 sites.
- **Centerline-grain / CPU-vs-kk discrepancy: RESOLVED (babbecf3, 1a9712b3).**
  Three kk kernel defects, isolated on the GP700 single-hatch deck against the
  CPU app (dt 2 µs: 732 grains, ESD 25 µm, isotropic):
  1. live neighbour reads in site-index order (x fastest) — a 144-voxel grain
     along x on the Rosenthal walk deck; now multicolour (8 parity classes,
     order shuffled per step) so reads are race-free and spatially random;
  2. the solid-front distance advanced for every undercooled liquid site (CPU:
     only while adjacent to solid) — now CPU parity;
  3. **the smoothing band counter was incremented and clamped at 0 (one pass
     regardless of nrefine) and the pass ran in the same step the voxel
     solidified, with its neighbours still liquid** — a greedy majority vote
     at the front that handed each track to a few grains 200+ µm long
     (23 grains, ESD 64 µm). Now decremented (nrefine passes) and the list is
     built before the solidification kernel. With smoothing off the two apps
     already agreed (18.5k vs 18.8k grains); after the fix with smoothing on:
     kk 494 grains / ESD 27.5 µm / extents 27-42-45 µm vs CPU 732 / 25.3 /
     24-39-45; dt 4 µs gives 391 / 32 µm.
  The phase fields of the two apps had been identical at every frame
  throughout — the defects were all in parent selection and post-capture
  smoothing, not in front kinetics. The earlier "≈2x kk" SROD path factor
  was defect 1; re-measure after the GP700 reruns.

## Findings this plan encodes

1. **SROD is a random walk.** Lamb et al. 2023 Fig. 5 (GammaPrint-700 PBF-L,
   3D EBSD) is reproduced, mean and 90% band, by rotations of uncorrelated
   axes accumulating along the growth path: RMS² = D·d with **D = 0.42
   deg²/µm**, Maxwell width (band/mean = 1.38, measured 1.35–1.45). Established
   grains follow from the same D with their initial spread added in
   quadrature. The authors' "fast first layer, then constant" is √d.
2. **Gradient-directed rotation is physically wrong.** Lamb et al. 2026 (Met
   Trans A): cells keep their lattice orientation and *switch* to another
   <100> at fusion lines; they do not rotate toward the heat flux. Gradient
   dependence belongs in growth selection (already in the mobility via
   `melt_misorientation`), not in rotation. It also produced the XZ/YZ
   diagonal texture.
3. **Undercooling dependence is at most weak.** The 3DThesis velocity map is
   flat over most of the pool with a sharp rise only at the last-to-solidify
   centerline; a strongly undercooling-driven step (α = 5) would put nearly
   all rotation there and make SROD flat-then-jump, not √d.
4. **Only the mean-square step per unit distance is observable.** Constant
   step, undercooling-dependent step, or intermittent steps with the same
   ⟨θ²⟩/hop give the same curve. The data constrain D, not the step law.
5. **The step must scale as √hop.** A per-hop angle that does not depend on
   hop length (both the cooling law and the cap) gives a D that changes with
   lattice spacing. D is the material/process quantity.

## A. Code changes

### A1. Angle law → diffusion coefficient (both apps)

New command, replacing `misorientation_function`:

```
misorientation_diffusion <D_deg2_per_m> <thresh_deg> [alpha]
```

- `D` in deg²/m as typed (0.42 deg²/µm = **4.2e5**); stored in rad²/m.
- Per-hop rotation, every rotation event:
  ```
  g(x)      = (exp(alpha*x) - 1) / (exp(alpha) - 1),  x = clamp(dT / (tl - ts), 0, 1)
  D_eff     = alpha > 0 ? D * g(x)^2 : D
  theta_hop = sqrt(D_eff * hop_m)
  ```
  `alpha` omitted or 0 ⇒ constant D. Squaring g keeps ⟨D_eff⟩ the quantity
  the walk sees and makes α = 0 the exact constant-step limit.
- `thresh_deg` keeps the `mis_thresh` new-grain check unchanged.
- `misorientation_function` and `misorientation_cap` become parse errors
  with a one-line migration hint (the am10 templates use the old command;
  they are regenerated from `_params_to_subs`, so update that too).
- `D <= 0` disables rotation (replaces the `max_misorient <= 0` gate).

CPU: `apply_misorientation(i, parent, Tcool, rng)` — delete the cooling-law
block and both cap branches (`CAP_PARENT`, `CAP_NEIGHBORS` budget loop);
compute `hop_m` from the parent as today, then `theta_hop` as above, then
`rotate_toward_target`. kk: same in the `SolidificationKernel` rotation block
(lines ~1710–1750), delete the cap branches. Delete `mis_cap_rate`,
`mis_cap_mode`, `MisCapMode`, `max_misorient`, `misorient_alpha` becomes the
new α; `t_cool_max` stays as `tl - ts`.

### A2. Nucleation shells (both apps)

Shell site at distance `r` from the seed rotates by `sqrt(D_eff * r)`, about a
random axis, relative to the **seed** orientation. This is the walk's RMS for
a path of `r/hop` hops, so shells and epitaxial growth obey one law. The CPU
flipper currently re-reads the already-rotated growth center (accumulating
fan) while kk rotates relative to the seed — align the CPU flipper to the
seed rule. `nucleation_misorientation_target` collapses to on/off (`none`
keeps the uniform nucleus); the target mode is the global one.

### A3. Target modes (both apps)

- Default `misorientation_target` → `random_rotation` in both apps (the kk
  default is currently `random`, the CPU default `gradient`; they differ
  today).
- **Remove `gradient`** from both rotation targets (finding 2). The
  `normal_finder`/`seed_grad` work in the flippers goes with it. This is a
  behaviour removal — confirm before deleting; if kept, it must carry a
  warning in the parser output.
- Keep `random` only as a test variant (B4). It is diffusive but the axis is
  confined ⟂ the nearest <100>; D_eff picks up a geometric factor.

### A4. Kokkos/CPU consistency fixes found on the way

- kk hardcodes `t_cool_max = 100 K`; CPU uses `tl - ts`. Use `tl - ts` in
  kk (needed for α to mean the same thing in both apps).
- kk's `mis_thresh` check uses the direct quaternion angle, CPU the
  symmetry-reduced disorientation. Leave, but note in the header.

### A5. What is *not* added

No per-site growth distance, no ghost/stitch changes, no new dump columns.
The origin-referenced SROD analysis uses the grain's lowest voxel as Lamb
did (B3). The per-grain persistent drift (B5) is deferred until the gate
says it is needed.

### A6. Validation of the code change

- Temperature `d6 sum/max` bit-identical to the current binary, all decks.
- RNG-identity check: old binary with `misorientation_cap c dist`,
  `random_rotation`, parent mode, cap binding, vs new binary with
  `D = (c·hop)²/hop`: same event sequence until `mis_thresh` or shell
  differences diverge it (shells: only if `nuclei` exist; use a
  `nucleation_density 0` deck for the strict check).
- np=2 OpenMP (`build-openmp`) and the GPU box-0 run.
- Resolution invariance: same deck at dx = 1.5 and 3 µm → same SROD curve
  within noise (this is the whole point of A1).

## B. Test formulation: undercooling-dependent vs constant vs other

One thermal deck (Plan 2's GP700 deck; fall back to the am10 ct15 deck if
Plan 2 lags), one seed set (≥ 3 seeds per variant), all variants at
dx = 3 µm, the winner re-run at 1.5 µm.

### B1. Variants

| id | law | settings | what it tests |
|---|---|---|---|
| C0 | constant | `misorientation_diffusion 4.2e5 T 0`, `random_rotation` | the paper fit; baseline |
| U5 | undercooling, sharp | α = 5, D chosen so ⟨D_eff⟩ over the measured ΔT distribution = 0.42 deg²/µm | does concentrating rotation at high ΔT still give √d? |
| U1 | undercooling, weak | α = 1, D matched the same way | the "at most weak" hypothesis |
| R | constant, `random` target | as C0, target `random` | anisotropic-axis walk; D_eff factor |
| G | constant, `gradient` (only if A3 keeps it) | as C0 | the negative control: zero width, diagonal texture |

Matching ⟨D_eff⟩ needs the distribution of undercooling at solidification.
It is recoverable from the existing dump: `d9` is the solidification velocity
V(ΔT) = polynomial(`solid_front_vel`), invert per voxel. Add a
`scripts/undercooling_hist.py` (20 lines) rather than a new stat.

### B2. Observables (all from the stitch dump: i1 i2 d1–d4 d8 d9)

1. **SROD vs distance from origin** — mean, 5/95%, per Lamb: grains ≥ 60 µm
   tall and ESD ≥ 40 µm; originating (origin inside the analysed volume) vs
   established (present at the analysis floor). Overlay
   `spparks-micro-analysis/analysis/surge_srod/srod_random_walk.py` curves.
2. **Band/mean ratio vs distance** — 1.38 target, the parameter-free check.
3. **Spatial SROD map on a melt-pool cross-section** — the discriminator
   between C0 and U5: U5 predicts SROD jumps at the centerline top, C0 a
   smooth increase along the growth direction (Lamb Fig. 4, Polonsky).
4. **Per-layer periodicity of the SROD residual** — fusion-line kick test
   (Met Trans: extra deformation at fusion lines). Directional kicks show as
   a 30 µm periodic excess over √d; random ones are already in D.
5. **Texture** — pole figures / ODF distance (`spparks-micro-analysis`
   `odf_distance.py`) vs the nucleation texture; no XZ/YZ build-up.
6. **KAM distribution** (`kam_analysis.py`) — sanity; Met Trans: KAM > 2° is
   the cell-wall tail, not the bulk.
7. **Resolution invariance** (winner only): 1.5 vs 3 µm.

### B3. Analysis tooling

`am10_cpu_vs_hip/srod_analysis.py` (new; reuse `compare_misorientation.py`'s
StitchReader + cubic-symmetry disorientation):
- grain = connected same-spin region (26-conn); origin voxel = lowest z,
  offset one voxel up (Lamb's rule); SROD = disorientation to the origin
  voxel; distance = Euclidean from it (or z for columnar).
- classes and size filters as above; bins of 5 µm; emit mean/5/95 and the
  ratio; JSON + PNG overlaying the digitized Lamb curves.

### B4. Decision rule

Take the simplest variant whose originating-grain SROD curve is within the
digitization error (~0.5°) of Lamb over 10–260 µm **and** whose ratio is
1.3–1.5. Expected: C0 passes. U1/U5 survive only if they reproduce something
C0 cannot — the Met Trans per-grain rate spread (1.7° avg / 5.8° max per
30 µm) or a spatial pattern — and then only at the smallest α that does.

### B5. "Other" approach, held in reserve

Per-grain persistent drift: axis hashed from the spin id (no storage), rate
`r_drift·hop` composed with the walk step. Gives mean ∝ √(D·d + (r·d)²) and
grain-to-grain rate spread. Add only if B2 item 1 shows the long-distance
under-prediction is real (data run ~1° above √d at 260 µm).

## Superseded designs (kept for the record)

- **First random-walk draft (today):** same as A1 without α and without the
  shell alignment; folded in here.
- **Two-regime cap (today):** cap rate fast for `d_origin < 30 µm`, slow
  after; needed per-site growth distance, ghost comm, stitch persistence.
  The knee is √d misread as two lines.
- **Power-law decay (2026-08-08):** `mis_angle *= L0/(L0+d_origin)`; the
  cap, not the cooling-law angle, bound on the decks, so it would not show.
