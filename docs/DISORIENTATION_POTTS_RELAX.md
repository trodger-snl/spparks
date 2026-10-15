# Disorientation Potts relaxation with frozen sites

Status: **report and port plan, 2026-10-14.** The method is implemented and tested in
MicroRust (`crates/microrust-core/src/potts.rs`, `relax` with `Energy::Disorientation`;
GPU: `potts/gpu.rs` and `potts/relax.wgsl`). This document records what it does, what
it has shown on real 3D EBSD so far, and how to add it to SPPARKS (CPU and `/kk`).

## Why

Two problems need Potts sweeps over a lattice in which some sites are fixed and grains
keep the orientation variation they were measured with:

1. **Serial-section infill.** Keep every k-th section of a 3D EBSD stack and rebuild the
   sections between. Interpolating grain shapes or copying the nearest section makes
   grains stretch along z and leaves cusps where they taper out. Relaxing the gaps by
   Potts, with the kept sections frozen, fixes the boundary shapes.
2. **2D → 3D reconstruction from a fixed surface.** This is the method of SAND2020-2777C
   (Rodgers, Alleman & Lim, "Relating 2D experimental information to 3D simulations
   using surface structure conserving 3D microstructure generation"): the EBSD map is a
   frozen face, every other site starts as its own spin, and a mobility M(x) that rises
   with distance from the face keeps the face's grains from growing columns.

Ordinary Potts treats each grain as one spin with one orientation. Measured grains are
not like that: they are regions of slowly varying orientation, found by a disorientation
threshold (segmentation at 5°), and their gradients matter. The disorientation energy
below works on the orientation field itself. It needs no grain labels, so gradients
inside a grain feel no force and survive relaxation.

## The method

Every site carries an orientation (a quaternion) and a phase. Spins are optional labels
that are only carried along. The neighbourhood is the 26 sites of a 3D cubic lattice
(8 in 2D), as `potts/neighonly` and `potts/quaternion` use.

**Boundary energy** between sites i and j, θ their disorientation under the phase's
Laue group:

| Case | Energy |
|---|---|
| θ < θ_like (5°) | 0: alike, the same grain; gradients feel no force |
| cubic phase, Σ3 within Brandon's criterion (15°/√3 = 8.66°) | γ_Σ3 (0.05 in the runs) |
| otherwise | Read–Shockley r(1 − ln r), r = θ/θ_cut (15°); 1 for θ ≥ θ_cut |
| different phases | 1 |
| both unindexed (phase 0) | 0 |

A site's energy is the sum over its neighbours. The 0.5 factor of
`AppPottsQuaternion::site_energy` doesn't matter for ΔE comparisons, but keep one
convention so kT means the same thing.

**Flip proposal.** Collect the distinct orientations among the neighbours that are
*unlike* the site, those with energy > 0 against it. Pick one uniformly and copy it, with
its phase and spin, from the first neighbour that has it. ΔE = Σ_j e(new, j) − Σ_j e(old, j).

This differs from `potts/quaternion`, which picks among distinct *spins*. With measured
gradients each cell's orientation is distinct, so the candidates are the unlike
neighbours, not all 26. MicroRust keys "distinct" by source cell: a site records which
original cell its orientation came from (`from[i]`), and two neighbours with the same
source are one candidate. In SPPARKS the equivalent is the bitwise quaternion, or an
integer source id carried in an `iarray` slot. The source id is cheaper and exact.

**Acceptance:** Metropolis at kT, then the site's mobility.

    accept = (ΔE ≤ 0  or  rand < exp(−ΔE/kT))  and  (M_i ≥ 1  or  rand < M_i)

M_i = 1 − exp(−(x_i/α)^β) per the slides, x_i the distance to the nearest frozen site.
It is precomputed into a site array, so any M(x) can be used. Frozen sites are never
visited; in the GPU version M_i < 0 marks them.

**Masking.** A site with no unlike neighbour has nothing to propose and is skipped until
a neighbour flips, as `sweep ... mask yes` does. This doesn't change the result, only the
speed. Frozen sites start masked and stay masked.

**Sweeping.** `sweep color/strict`-style: 8 colours by parity along each axis (a third
colour for the last layer of an odd periodic axis). Sites of one colour read only other
colours, so they update in parallel in any order. MicroRust draws from ChaCha8 per row
(CPU) or a counter hash per (seed, sweep, colour, site) (GPU), so results don't depend on
thread count. The SPPARKS port should keep that property.

**Stopping.** For infill from random gaps, stop when the gap layers have no more grains
per layer than the kept layers, checked every 2 sweeps. With disorientation energy,
"grains" must be counted as **5°-connected regions per layer**, not distinct spins. Two
alike neighbours never flip to each other's label, so a spin count barely falls even when
grains have formed. That bug ran every Grow case to the 400-sweep cap until it was fixed.
For the surface reconstruction, the slides' objective, KS of the 2D grain-size
distribution against the face as a function of depth, is the natural stop.

## Making it fast

Profiling the CPU version on IN625 (≈100 M cells) showed that the disorientations of
boundary sites dominate. Three changes, all exact:

1. **Short-cut for alike pairs.** If |⟨q_i, q_j⟩| > cos(θ_like/2), the pair is within
   θ_like without any symmetry, so it is within θ_like under the symmetry too: energy 0,
   no search. Most neighbours inside a grain take this path. (162 → 142 s.)
2. **Cubic closed form** for max over s of |(s·d).w| (m-3m): the largest |component|,
   (the two largest)/√2, or half the sum, instead of 24 products. MicroRust's CPU already
   used it (`crystal_orientation::Laue::disorientation_cos_half`); `potts/quaternion`'s
   `disorientation::compute_disorientation` loops over the symmetries.
3. **The Σ3 orbit.** Brandon's test against Σ3 = 60°⟨111⟩, done literally as in
   `boundaries::within`, is 24 disorientations of 24 products each. Random cubic
   disorientations peak near 45° and run to 62.8°, so a large share of boundaries fall
   within 8.66° of 60° and need the full test. Instead, enumerate once the distinct (up to
   sign) s·m·t over the operators s, t: **96 quaternions for m-3m**. Then d = q_i·q_j⁻¹ is
   within tol of Σ3 exactly when max over x of |⟨d, x⟩| ≥ cos(tol/2), because
   ⟨s·d·t, m⟩ = ⟨d, s⁻¹·m·t⁻¹⟩. That is 96 dot products. A test holds it to the literal
   search on 200 pairs. (142 → 70 s on the CPU.)

GPU (wgpu/WGSL on the gfx1151 iGPU, Vulkan RADV), in f32:

- **Threads only for the colour being swept.** Launching a thread per site and returning
  for the other colours left half of every wavefront idle, because colour alternates
  with x.
- **No masking:** every site of the colour evaluates every sweep.
- **Keep the state on the device.** Restarting from the host every 2 sweeps re-uploaded
  ≈ 2 GB per restart and left the GPU about 30 % busy. The stop check now reads back only
  the spins and the sources.

| IN625 as read, every 3rd section kept, 20 sweeps of relaxing the shape fill (≈100 M sites) | Time |
|---|---|
| CPU (32 threads), literal Σ3 test | 142 s |
| CPU, Σ3 orbit | 70 s |
| GPU, first version | 130 s |
| GPU, colour-compact dispatch + closed form + orbit | 29 s |

Each time includes about 10 s of harness work (segmentation, composing sources).
Sweeps from random gaps, where every site is active, are 1–2.5 s each on the GPU. CPU and
GPU give the same grain counts (1973 vs 1974 regions per gap layer) but not bit-identical
fields: the GPU uses f32 and different draws.

Environment note for this box: Mesa's `VkLayer_MESA_device_select` can hang inside
`vkEnumeratePhysicalDevices` waiting on the Wayland compositor. `NODEVICE_SELECT=1`
avoids it. HIP doesn't go through that layer, so the Kokkos port won't see it, but any
Vulkan tool here might.

## Results so far

Benchmark: MicroRust's `golden_ambench_infill_stats`.
- **Data:** AM Bench 2022 IN625 (LPBF, 450 × 450 × 510 cells of 2 µm) and Stinville
  IN718 (wrought, TriBeam, 549 × 420 × 526 cells of 1 µm, Σ3-rich).
- **Setup:** keep every k-th section, rebuild the rest, segment truth and rebuild alike
  at 5°.
- **Scoring:** compare DREAM.3D-style grain metrics in five z bands, averaged, lower is
  better; "as read" counts grains of 5+ cells.

Fills:
- **Blend:** shape-interpolated grains, orientations blended.
- **BlendP / BlendD:** Blend plus 20 sweeps at kT 0.25, label energy or disorientation
  energy.
- **Grow / GrowD:** every gap site its own spin, oriented as a random kept site nearby,
  grown until the region count matches.

The kept sections are frozen in all of them.

IN625 as read (grains small next to the gap):

| Fill (every 5th kept) | density | ESD μ | b/a KS | c/a KS | axes KS |
|---|---|---|---|---|---|
| floor (truth's two halves) | 0.534 | 0.026 | 0.045 | 0.067 | 0.036 |
| Blend | 3.730 | 0.363 | 0.347 | 0.444 | 0.244 |
| BlendP | 2.000 | 0.160 | 0.191 | 0.360 | 0.253 |
| BlendD | 1.869 | 0.180 | 0.199 | 0.390 | 0.261 |
| Grow | 0.988 | 0.143 | **0.101** | **0.227** | **0.069** |
| GrowD | **0.511** | 0.205 | 0.159 | 0.383 | 0.233 |

IN718 as read (large grains, inclined Σ3 lamellae):

| Fill (every 5th kept) | density | ESD μ | b/a KS | c/a KS | axes KS |
|---|---|---|---|---|---|
| floor | 0.686 | 0.117 | 0.168 | 0.215 | 0.148 |
| Blend | **0.322** | **0.483** | 0.135 | **0.071** | **0.155** |
| BlendP | 0.552 | 1.232 | 0.334 | 0.212 | 0.295 |
| BlendD | 0.580 | 1.278 | 0.332 | 0.156 | 0.264 |
| Grow | 0.605 | 1.338 | 0.254 | 0.101 | 0.244 |
| GrowD | 0.757 | 1.333 | 0.488 | 0.298 | 0.288 |

IN718, grains of 125+ cells (small grains merged into their neighbours in truth and
fills alike):

| Fill | k | density | ESD μ | b/a KS | c/a KS | axes KS |
|---|---|---|---|---|---|---|
| Blend | 5 | 0.146 | 0.147 | 0.122 | 0.066 | 0.080 |
| BlendP | 5 | **0.044** | 0.122 | 0.113 | 0.066 | **0.073** |
| BlendD | 5 | 0.059 | 0.175 | 0.134 | 0.080 | 0.093 |
| Grow | 5 | 0.061 | **0.115** | **0.099** | 0.068 | 0.089 |
| GrowD | 5 | 0.537 | 0.689 | 0.515 | 0.410 | 0.211 |
| Blend | 15 | 0.321 | 0.225 | 0.156 | **0.086** | 0.225 |
| BlendP | 15 | **0.041** | **0.157** | 0.144 | 0.203 | 0.122 |
| BlendD | 15 | 0.065 | 0.165 | 0.157 | 0.190 | **0.121** |
| GrowD | 15 | 0.732 | 0.668 | 0.326 | 0.358 | 0.360 |

On large twinned grains, relaxing a shape fill helps either way: density error falls
from 0.15–0.32 to about 0.05. The disorientation energy is a little behind labels here.
Its random start (GrowD) is the worst fill by far, because at these settings it neither
coarsens enough nor stops (400-sweep cap).

What the runs show:

- **Frozen-section Potts helps where grains are small next to the gap.** On IN625 as
  read, the random-start Grow cuts density error by 2–4× and the axes KS from 0.24 to
  0.07, near the floor. The vertical streaks of the shape fills disappear.
- **The disorientation energy is no better than labels on these cases yet.**
  - On IN625, BlendD ≈ BlendP.
  - GrowD gets the best grain density (0.51) but worse shapes than Grow.
  - On IN718 it doesn't rejoin twin lamellae that the sections cut into dashes. A
    lamella inclined so that its traces in adjacent kept sections don't overlap in any
    column can't be joined by local flips; that needs matching across the gap (for
    example, extruding each twin along the {111} plane it shares with its parent).
- **Low Σ3 energy at kT 0.25 makes speckle.** Moving a site across a Σ3 boundary costs
  about 0.05 per neighbour, so most such moves are accepted. The IN718 gap layers end
  with about 2.7× the kept layers' 5° regions (701 vs 255), mostly small specks at
  boundaries. In GrowD, twins in the frozen sections also grow sideways along them into
  the gap: thin horizontal streaks at the kept planes. Next to try: kT ≈ 0.05, or
  γ_Σ3 ≈ 0.3, or Σ3 energy only for boundaries near the coherent {111} plane.
- **Random-start growth plateaus on large, textured grains.** IN625 with the 125-cell
  threshold stopped 1.5–5 % above the kept region count at the 400-sweep cap
  (8–10 min on the GPU). IN718 with every 15th kept (14-layer gaps) ended at 1733 against
  233 regions per layer. Textured, low-angle neighbours have weak Read–Shockley driving
  force. The stop should also end on a plateau, and gaps that deep probably need seeds
  sized from the sections rather than one spin per site.
- **The mobility M(x) made no consistent difference in thin gaps** (k ≤ 15 layers): the
  count-matched stop already prevents runaway growth. It should matter for the surface
  reconstruction, where one face looks into a deep volume, as the slides found.

## Porting to SPPARKS

Proposed style **`potts/disorient`** (CPU) and **`potts/disorient/kk`**, deriving from
`AppPottsQuaternion` / `AppPottsQuaternionKokkos`, which already keep a per-site
quaternion in `d1..d4` (`darray[0..3]`), copy it with the spin on a flip, and use
Read–Shockley energy. Changes from `potts/quaternion`:

1. `site_energy`: add the θ_like zero band, the Σ3 term (precomputed orbit, cubic
   phases), the |⟨q_i, q_j⟩| short-cut and the m-3m closed form. Make the 0.5 factor
   explicit in the docs so kT is comparable.
2. `site_event_rejection`: candidates are distinct unlike *orientations*, keyed by a
   source id in a new `iarray` slot (initialised to the global site id), not distinct
   spins. Copy quaternion, phase (if multiphase), spin and source from the chosen
   neighbour.
3. Frozen sites: an `iarray` flag (`i2`, as `additive/texture`'s active flag), set from
   the input file, skipped in the sweep and never unmasked.
4. Mobility: a `darray` slot (`d5`), filled by a command, e.g.
   `set_mobility stretched alpha beta` computing x by a distance transform from the
   frozen sites, or read from the site file. Accept with one more uniform draw where it
   is below 1.
5. Masking: mask a site with no unlike neighbour, i.e. all neighbours alike, not
   `einitial < 0.5 numneigh` as in `potts/quaternion`, which is a spin-count test.
6. App keywords, e.g. `app_style potts/disorient nspins lattice θ_cut θ_like [sigma3 γ]`.
7. Diagnostics: per-layer 5°-region counts (or the slides' per-depth KS) as a `diag`, so
   a run can stop on it.

For `/kk`, keep the existing multicolour kernel (the 8 parity classes of babbecf3 and 1a9712b3).
Additions:
- Put the symmetry operators and the Σ3 orbit in device Views.
- Draw from a counter-based hash keyed by (seed, sweep, colour, site), so results don't
  depend on the Kokkos backend or team size.
- Do the stop diagnostic on the device; it needs only per-site source ids and phases.

Validation:
- **Unit scene** (MicroRust's `relax_by_disorientation_keeps_gradients`): a 24 × 8 × 9
  lattice, first and last z-layers frozen. A grain turns 0.5° per site along x (over 5°
  across it) and meets a grain 30° away at a boundary zigzagging ±4 sites. 30 sweeps at
  kT 0 should cut the boundary's misplaced sites by 4×, leave every site at x < 6 with
  its own orientation, and leave frozen sites unchanged.
- **Statistics, not bits, against MicroRust:** the RNGs differ. Run the IN625 and IN718
  infill decks (k = 3, 5, 15) and compare the band metrics above with MicroRust's
  BlendD/GrowD. They should agree to within the spread between seeds.
- **The slides' surface case:** the Ta EBSD face (2500 × 1250 µm at 1 µm), with M(x)
  fitted as before, reproducing the per-depth KS profile.
