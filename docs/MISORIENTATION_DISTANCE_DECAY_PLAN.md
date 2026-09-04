# Plan: diffusion-coefficient random-walk misorientation

Status: **PLANNED, not implemented.** Third revision (2026-09-04). Supersedes
the two-regime cap (same day) and the 2026-08-08 power-law decay, both kept
as notes at the bottom. Companion to the shipped features: nucleation-shell
rotation + orientation-gradient steepness cap (`8325eb71`) and the
all-neighbors cap mode (`ad9962d0`).

## What the 3D-EBSD data actually shows

Lamb et al. (Acta Mater 2023, CoNi GammaPrint-700 PBF-L, `~/Downloads/3DEBSD`)
plot SROD — misorientation of every voxel from its grain's *origin*
orientation — against distance grown from that origin (their Fig. 5a,
originating grains) or from the dataset bottom (Fig. 5c, established grains),
as a probability density with mean and 90% band. They read Fig. 5a as two
regimes: a sharp rise over the first build layer (30 µm), then a constant
0.7–1.0° per layer.

Both panels are reproduced, mean **and** width, by a single-parameter
random-axis rotation walk (`docs/srod_random_walk.png`,
`docs/srod_random_walk.py` digitizes the published curves from the PDF):

- Composing per-hop rotations of fixed angle θ about independent random axes
  is a 3D random walk in rotation-vector space. Net angle from the origin has
  RMS² = D·d with **D = θ²/hop** (deg²/µm), a Maxwell (χ₃) distribution, and
  a spread/mean ratio fixed at 0.42.
- Fit on Fig. 5a mean, 5–260 µm: **D ≈ 0.42 deg²/µm**, rms residual 0.5°.
  The 90% band follows without any extra parameter (data 3.7–17.8° at
  260 µm, model 3.6–16.9°).
- Fig. 5c is then a *prediction*: RMS² = RMS₀² + D·d with RMS₀ from the
  bottom-layer mean alone (5.7° → RMS₀ 6.2°). Residual 0.6° over 480 µm.
- The authors' "fast then constant" shape is what √d looks like. The
  residual is mildly systematic (data slightly below √d at 40–120 µm, above
  at 240+ µm), i.e. the true curve is a little more linear than pure
  diffusion — see "if √d is not enough" below.

Neither this analysis nor any random-walk / diffusion / √d treatment appears
in the three papers; Lamb fit two straight segments and Polonsky maps
misorientation onto Scheil fraction solid.

Why the coherent alternative is wrong: in `gradient` target mode every hop
rotates a <100> toward the local thermal gradient, so hundreds of hops drive
all grains to one shared target (the rear melt-pool wall normal, on the XZ/YZ
diagonals in a raster) — the strong diagonal texture seen with the old model
— and every voxel of a grain moves the same way, so SROD has zero width. A
random walk has no shared target and produces the measured width for free.

## Design

Rotation per hop about a uniformly random axis (`random_rotation` target
mode, which already exists in both apps) with the angle sized as a
**diffusion coefficient**, not a rate:

```
theta_hop = sqrt(D * hop_m)
```

- `hop_m` is the parent→site hop in metres (already computed for the cap).
- Using rate × hop instead would make the accumulated SROD shrink as
  1/√(refinement): RMS = rate·hop·√(d/hop) = rate·√(hop·d). D is the
  lattice-independent quantity. At D = 0.42 deg²/µm the per-hop angle is
  0.65° at dx = 1 µm, 1.12° at 3 µm, 1.45° at 5 µm — about six times the
  current 2°/30 µm default per hop, because a walk accumulates inefficiently.
- Applies to all three rotation events unchanged: epitaxial solidification,
  nucleation shells (hop from the seed), and it needs **no per-site growth
  distance** — the √d law is a property of the walk itself. The `d_origin`
  bookkeeping of the earlier plans is dropped.
- Established grains, remelt, stitch continuation: nothing to do. A grain
  crossing a melt-pool boundary keeps walking from its surviving parent
  (Polonsky: misorientation never resets; Fig. 5c: quadrature of existing
  spread and new walk).

### Input

```
misorientation_diffusion <D_deg2_per_m>     # sets theta_hop = sqrt(D*hop); D <= 0 disables (default)
misorientation_target random_rotation
nucleation_misorientation_target random_rotation
misorientation_cap 0 0                       # see below
```

`misorientation_diffusion` *replaces* the cooling-law angle
(`misorientation_function`) and the cap when set: the walk step is the
rotation, and it is already the physically bounded quantity. Keeping the cap
on in parent mode is harmless only if `cap·hop >= theta_hop`; in
**neighbors mode it must be off** — two cousins that each took independent
θ steps differ by ≈√2·θ, the budget goes negative and the walk collapses to
zero. Paper-calibrated value for GammaPrint-700:

```
misorientation_diffusion 0.42e6        # deg^2/m  (0.42 deg^2/um)
```

Implementation is a few lines in each app: in `apply_misorientation` (CPU)
and the `SolidificationKernel` rotation block (kk), `if (mis_diffusion > 0)
mis_angle = sqrt(mis_diffusion * hop_m)` before the cap; same in both
particle flippers in place of `mis_cap_rate * hop`; parser mirrored. The
`mis_thresh` new-grain check stays (θ ≈ 1° is far below any sane threshold).
`random_rotation` consumes one RNG draw per event in both apps already.

### Not `random`

`random` target mode rotates the nearest <100> *toward* a random direction,
so the rotation axis is constrained perpendicular to that <100>; the walk is
anisotropic in crystal frame. `random_rotation` is the uniform-axis walk the
analysis assumes. The kk `perturb_orientation` has the same two modes.

## Validation

- d6 (temperature) sum/max bit-identical, feature off and on.
- With D chosen so that √(D·hop) equals the current cap·hop, and target
  `random_rotation` in both cases, results are RNG-identical to the current
  binary (the angle is the only change).
- Functional gate: SROD vs distance-from-origin from a dump. Origin voxel per
  grain = the same-spin voxel with the earliest solidification time (or
  lowest z for columnar grains, as Lamb did); distance = Euclidean from it.
  Bin by distance, plot mean and 5/95%: expect mean ≈ 0.921·√(D·d) and
  spread/mean ≈ 0.42, both **independent of dx** (run 1 µm and 3 µm decks).
  `compare_misorientation.py` has the per-grain spread machinery; add the
  origin-referenced binning there and overlay `docs/srod_random_walk.py`'s
  model curve.
- Texture: pole figures must stay as random as the nucleation texture (no
  XZ/YZ diagonal build-up).

## If √d is not enough

The data are slightly more linear than √d, and Lamb's distribution pools
within-grain and grain-to-grain variation (Met Trans A 2026 on the same alloy
gives per-grain average 1.7° and max 5.8° per 30 µm). If a run reproduces the
band but under-predicts the long-distance mean, add a **per-grain persistent
drift**: a random axis hashed from the spin ID (no storage; spin collisions
just share an axis) with a small rate `r_drift·hop` composed with the walk
step. Mean then grows as √(D·d + (r·d)²). Only add it if the gate above
demands it.

## Superseded designs

- **Two-regime cap (earlier today):** cap rate r_fast for `d_origin < 30 µm`,
  r_slow after, needing a per-site growth distance (`d10`), ghost comm and
  stitch persistence. Encoded the authors' reading literally; the knee is an
  artefact of reading √d as two lines, and the coherent fan it was calibrated
  in has zero SROD width.
- **Power-law decay (2026-08-08):** `mis_angle *= L0/(L0+d_origin)` on the
  cooling-law angle before the cap. Same `d_origin` cost; the cap, not the
  cooling-law angle, was what bound on the decks, so it would mostly not show.
