# SROD analysis report: misorientation accumulation in GammaPrint-700 PBF-L, experiment vs SPPARKS

Status: 2026-09-09. **Reference version: `SROD_ANALYSIS_REPORT.tex` (PDF alongside); this file mirrors it.**
Development history with every intermediate run: `GP700_MOSER_SROD_DECK_PLAN.md`; model changes: `MISORIENTATION_DIFFUSION_PLAN.md`.
Figures: `srod_experimental_analysis.png`, `gp700_results_v10/`, `gp700_results_v8/`, `gp700_results_dx1p5/`.

## 1. Summary

- Population SROD in GP700 (Lamb et al. 2023, Fig. 5) is a random-axis rotation walk:
  RMS² = D·h with D ≈ 0.42 deg²/µm against build height, the same D for originating
  and established grains, and a 5/95 band of (p95 − p5)/mean ≈ 1.4, the Maxwell width
  of a 3-D walk. Individual grains (Fig. 4) rotate coherently along a directed path.
- Model, both SPPARKS additive apps: `misorientation_diffusion D_v` (per-voxel walk,
  θ = √(D_v·hop) inherited per hop) plus `misorientation_grain_walk D_g rho path`
  (coherent per-grain walk: one Gaussian rotation-vector increment per unit of
  inheritance path length, shared by every voxel of the grain, correlated between
  levels with persistence rho). The path length is a per-site field (d10). Orientation
  is inherited from one parent voxel and never averaged: per-grain parent selection and
  copy-mode smoothing are part of the model.
- Calibrated deck (`examples/gp700_moser/`, build v10): D_v 0.065, D_g 0.010 deg²/µm,
  rho 0.9 (correlation length ≈ 30 µm of path). Against Lamb: RMS 0.34°, bias −0.28°,
  band/mean 1.50; per-grain drift median 7.2°, spread 3.4°.
- The constants are per unit of growth path (1.8× the height here) and first-order
  resolution independent: the same deck reproduces Lamb at the experiment's 1.5 µm voxel
  (bias +0.41°).
- Open: no surviving grains originate mid-build (83 of 88 filtered grains start at the
  first remelt floor); D_g : D_v and rho need Lamb's within-slice widths and their
  population curve below 30 µm; the <001> fibre (0.15–0.19 within 15° of BD) has no
  measured target.

## 2. Experimental methods (Lamb et al. 2023, Met. Trans. A 2026)

Material and process: CoNi-base superalloy GammaPrint-700, laser powder bed fusion,
176 W, 1137 mm/s, 80 µm spot (D4σ), 80 µm hatch, 30 µm layers, 90° interlayer rotation,
200 °C plate preheat. The 2026 paper supplies the thermophysical set used for its
3DThesis simulations: k 22.5 W/m·K, ρ 8650 kg/m³, cp 600 J/kg·K (α 4.335e-6 m²/s),
absorptivity 0.9, liquidus 1646 K, solidus 1592 K.

3D dataset: TriBeam tomography (femtosecond-laser serial sectioning with EBSD on each
slice), 1.5 µm in-plane step and 1.5 µm slice thickness, 385 slices, reconstructed into
one HDF5 volume and segmented into grains at a 2° misorientation threshold before any
SROD calculation.

SROD: the reference orientation sits near the origination point, "the lowest location
of a grain's boundary along the build direction", offset one voxel layer (1.5 µm) upward
so the reference is not a boundary pixel; offsets up to 8 µm do not change the result.
Originating grains have their lowest point inside the volume; established grains are cut
by the dataset floor and referenced at the centroid of their footprint on the lowest
slice. Grains shorter than 60 µm or under 40 µm ESD are excluded, leaving > 500
originating and > 100 established grains. Each grain is sliced along the build
direction; each slice gives a SROD histogram fitted to a normal distribution, plotted
against height above the origin (Fig. 5a) or above the floor (Fig. 5c). Fig. 4 follows
four grains with IPF and SROD maps, the mean SROD per layer, and every voxel on the
build-direction IPF with an arrow for the rotation path bottom → top; one grain splits
into two clusters. The 2026 paper adds that cells switch <100> variants at fusion lines
rather than rotating continuously, so no gradient-directed rotation is modelled.

## 3. Analysis of the experimental data

Fig. 5a/5c were digitized (`srod_random_walk.py`; `lamb2023_fig5{a,c}_srod.json`) and
tested against a rotation walk: independent small rotations of angle θ about random
axes give RMS² = n·θ² = D·d and a Maxwell distribution with (p95 − p5)/mean = 1.38 at
every distance. D = 0.42 deg²/µm fitted on the originating grains predicts the
established grains without refitting; the measured width ratio is 1.3–1.5 at every
height; the apparent two-regime shape is the square root. The population statistics
do not distinguish whether the increments belong to voxel lineages or to whole grains
(each grain's coherent walk is one realisation; the Maxwell width appears across grains
either way). Fig. 4 supplies that information: the grains' mean orientation moves along
a path, so the accumulation is at least partly coherent across the grain.

## 4. Simulation methods

Code: SPPARKS, branch SURGE-kokkos, `additive/texture/kk` (Kokkos/HIP, AMD gfx1151),
one MPI rank; the CPU app `additive/texture` carries the same model and is the
correctness reference (temperature bit-identical, `examples/am_validation/`).

Thermal source: Moser Green's-function solution with the GP700 constants
(`setup_temperature_source moser 176 0.9 22.5 4.335e-6 473 600 20e-6 20e-6 10e-6`);
`laser_path ... hatch 80e-6 N serpentine pause 8.8e-3` (the real 10 mm track length
over the scan speed); the fast-forward predictor skips idle gaps. Single-hatch check:
pool 147 × 69 µm, gradient 1e7–2.9e7 K/m at the fusion boundary (paper 3e7).

Build: box-per-layer stitch continuation (`run_layers.sh`). 801 µm square (267 voxels
at 3 µm), 10 hatches at 40 + 80k µm with lines 100 µm past both edges, 12 layers of
30 µm, 90° rotation per layer, each box a 180 µm window below the laser plane, each
layer from the preheat. Substrate: 150 µm of 50 µm Voronoi grains with random
orientations (`make_substrate.py`); remelt 80–100 µm, so the analysable volume begins
at 105 µm. dt 4 µs (twice the dx/V(Tl − Ts) guard; 2 and 4 µs agree within seed noise).
Calibration used a 600 µm / 7-hatch / 8-layer proxy at 3 min of loop time per box.

Solidification kinetics: `solid_front_vel` = the 304L polynomial × 3.5 (front at
25–35 K undercooling, ~0.56 m/s); nucleation density 1.5e13 m⁻³, critical undercooling
15 K, deviation 2 K. Parent selection per grain: candidate neighbours grouped by grain,
each grain weighted by the mobility 1 + c2·cos(c3·θ) of its sign-aligned mean
orientation against the local gradient (c2 2.8, c3 2.5, negative clamped to 0), the
parent voxel drawn uniformly inside the chosen grain. Weighting among the voxels of one
grain rotated grains toward <001> with height and is not used.

Misorientation model, two terms applied at capture in the sample frame:

- Per-voxel walk, `misorientation_diffusion D_v 30`: the captured voxel takes its
  parent's orientation rotated by θ = √(D_v·hop) about a random axis; a rotation past
  30° would start a new grain (never reached). Independent lineages diverge, so this
  term supplies intragranular spread and no net drift.
- Coherent grain walk, `misorientation_grain_walk D_g rho path`: every voxel carries its
  inheritance path length s in lattice units (per-site field d10: 0 at melt and at
  nuclei, parent + hop at capture, copied with the orientation on a smoothing flip,
  exchanged with ghosts, continued through the stitch). Between path levels k and k+1
  every voxel of grain g receives the same rotation: a Gaussian rotation vector v_k of
  RMS √(D_g·dx) whose sequence in k is Ornstein–Uhlenbeck, v_k = rho·v_{k−1} +
  √(1−rho²)·v'_k, with v'_k hashed from (g, k) so the sequence is a pure function of
  grain and level, path independent, no stored state; correlation length ≈ dx/(1−rho)
  of path; long-range accumulation D_g·(1+rho)/(1−rho) per unit path. This term supplies
  the coherent drift. A device cache of the recurrence (16384 grains × 768 levels) makes
  it cost nothing measurable; overflow falls back to direct evaluation.

`height` in place of `path` indexes the coherent walk by the lattice z level; kept for
comparison, not used.

Boundary smoothing: one greedy steepest-descent pass per captured voxel
(`smooth_greedy_multiproposal on`, nrefine 1) the step after capture, with
`smooth_orientation copy`: a voxel that changes grain takes one random neighbour's
orientation and path length from that grain. The default `average` cancels the fresh
rotations of a voxel-scale mixed front, retains only 0.3–0.45 of the input D and widens
the population band to 1.8; not used.

## 5. Simulation analysis methods

`gp700_analysis.py`: read the last stitch frame; solid = `i2 == 3` (unremelted
substrate excluded); crop to the central 400 µm (300 µm on the proxy); relabel grains
as connected components; grain volume, ESD, extents; Lamb's filters (height ≥ 60 µm,
ESD ≥ 40 µm); origin = the grain's lowest solid voxel, whose quaternion is the reference;
grains crossing the analysis floor are established and referenced at their floor-slice
centroid. SROD = symmetry-reduced disorientation from the reference, binned by height
in 10 µm steps (mean, p5, p95, RMS). Score: RMS of (sim mean − Lamb Fig. 5a mean) over
25–240 µm, plus bias, (p95 − p5)/mean, the <001>-within-15° fraction and grain counts.
Sobol designs (scipy qmc) on the proxy were used for calibration.

`grain_paths.py` reproduces Lamb's Fig. 4 per grain and extends it to every filtered
grain: IPF-Z and SROD maps on the xz slice through the grain, mean SROD per 10 µm slab,
the per-slab mean orientation as a path on the build-direction IPF, the drift of the
mean orientation from the bottom three slabs to the top three, the RMS spread of the
top slabs about their mean, the maximum SROD, and the fraction of slabs in which the
mean SROD rises.

## 6. Results

### 6.1 Calibration path

| build | model | RMS / bias vs Lamb | band/mean | per-grain drift / spread (median) |
|---|---|---|---|---|
| v2 | per-voxel walk D_v 0.42, voxel-level parent selection | 2.2° / +3.6° (linear growth) | 1.4–1.8 | – |
| v5 | per-grain parent selection, average smoothing, D_v 1.15 | 0.59° / −0.16° | 1.79 | – |
| v8 | copy smoothing, D_v 0.22 | 0.49° / +0.17° | 1.41 | 4.2 / 6.1° |
| **v10** | **+ coherent path walk D_g 0.010, rho 0.9; D_v 0.065** | **0.34° / −0.28°** | **1.50** | **7.2 / 3.4°** |

v2's linear growth was selection drift among the voxels of one grain. v5–v7 retained
only 0.3–0.45 of the input D against height with a 1.8 band, both from the averaging
smoothing rule; copy-mode smoothing (v8) put the population curve, shape and band on
Lamb. v8's grains spread without drifting: the per-voxel walk lets the lineages of one
grain diverge independently, so the cloud on the IPF expands about a stationary mean,
whereas Lamb's grains move along a path. The coherent grain walk (v10) supplies the
drift. A memoryless coherent walk makes individual SROD(h) curves wander; correlating
the rotation vector between levels (rho) makes them rise along a directed path, and
with Gaussian increments the population band stays at Lamb's value in both the
ballistic and the diffusive regime. Indexing by path length rather than height removes
the build-direction assumption, gives the AR(1) amplification exactly, and produces an
intragranular spread from path dispersion alone.

### 6.2 v10 against Lamb

v10: D_v 0.065, D_g 0.010 deg²/µm, rho 0.9 path, copy smoothing, nrefine 1, Voronoi
substrate, nucleation 1.5e13, tc 15 K, c2 2.8, dt 4 µs, 3 µm.

| | v10 | v8 (per-voxel only) | Lamb 2023 |
|---|---|---|---|
| SROD RMS deviation, 25–240 µm | 0.34° | 0.49° | – |
| mean bias | −0.28° | +0.17° | – |
| mean SROD at 54 / 204 / 402 µm | 3.6 / 8.2 / 12.1° | 4.4 / 8.5 / 12.3° | 3.8 / 8.8 / – (data end at 262 µm) |
| (p95 − p5)/mean | 1.50 | 1.41 | 1.4 |
| D-fit vs height (deg²/µm) | 0.43 | 0.41 | 0.42 |
| grains passing filters (400 µm crop) | 97 | 86 | > 600 (full volume) |
| ESD p10 / p50 / p90 (µm) | 76 / 148 / 170 | 73 / 151 / 182 | – |
| <001> within 15° of BD | 0.15 | 0.19 | qualitative <001> fibre |

Figures: `gp700_results_v10/srod_height.png`, `ipf_sections.png`.

### 6.3 Per-grain orientation paths

Every filtered grain of the two full builds (`grain_paths.py`); medians, p10 / p90 for the drift:

| | v10 (96 grains) | v8 (86 grains) |
|---|---|---|
| drift of the mean, bottom → top | 7.2° (1.7 / 13.5) | 4.2° (2.0 / 7.5) |
| spread about the top-slab mean | 3.4° | 6.1° |
| drift ÷ spread | 1.66 | 0.60 |
| max SROD in the grain | 15.0° | 16.0° |
| top-slab mean SROD | 8.4° | 8.2° |
| fraction of slabs with rising mean SROD | 0.79 | 0.78 |

In v10 the drift scales with grain height (≈2° at 100 µm, 8–15° at 400 µm), nearly
every grain drifts more than it spreads, and the tallest grains trace directed paths
with max SROD 20–24°, the range Lamb report for their example grains. In v8 the grains
spread more than they drift at every height. The two builds share the population
level, band and max SROD: the population statistics cannot separate the models, the
per-grain ones can. The paths have no preferred direction on the triangle (isotropic
rotation vector). About a fifth of the grains reverse over 100 µm scales, which a
30 µm correlation length allows. Figures: `gp700_results_v10/grain_paths_grid.png`,
`grain_summary.png`, `grain_stats.csv` (and the v8 counterparts).

### 6.4 Resolution

The per-voxel model (v8 deck) at Lamb's 1.5 µm voxel on the proxy with the same D and
geometry: RMS / bias 0.49° / +0.41° vs 0.44° / +0.03° at 3 µm, band 1.43 vs 1.41, same
grain size and texture; the finer lattice accumulates 10–18 % more per unit height
(retention 2.17 vs 1.83) because it resolves a slightly longer growth path. The per-hop
rule is resolution independent to first order; an exact match at 1.5 µm would take D
about 15 % lower.

### 6.5 Crop and grain origins

Analysing v8 on the full 801 µm domain, the central 400 µm, the central 200 µm, or with
the never-remelted top layer removed changes the SROD bias by < 0.2° and the band by
< 0.03. Of the 88 filtered grains in the standard crop, 83 originate at the layer-0
remelt floor (99 % of the filtered volume) and 3 above 160 µm: the columnar grains are
epitaxial substrate continuations, so the simulated "originating" SROD is an
established-grain SROD referenced at the remelt floor. Lamb's two classes share the same
D, so the walk comparison is unaffected, but the model does not generate Lamb's
population of grains that originate mid-build.

### 6.6 Texture

Neither paper quantifies the <001> fibre. A Sobol study over c2 and nucleation with D
fixed gave <001> within 15° flat at 0.22 below c2 ≈ 5 and rising 0.0125 per unit c2
above, SROD unaffected; the full build gave 0.22 at c2 9 vs 0.19 at 2.8. The coherent
walk lowers it further (v10: 0.15) because it carries grains out of the cone. The
fraction is 1.7–2.5× random in every build.

## 7. Findings and open issues

1. Population SROD in GP700 is a rotation walk with D ≈ 0.42 deg²/µm against height and
   a Maxwell width; individual grains rotate coherently along a path.
2. The model reproduces both once orientation is inherited from one parent voxel and
   never averaged (per-grain parent selection, copy-mode smoothing) and a coherent,
   level-correlated grain walk is added to the per-voxel walk. The population statistics
   fix the total D; the per-grain drift-to-spread ratio fixes D_g : D_v; the shape of
   the individual curves and of the population curve below the correlation length fix rho.
3. Deck constants are per unit of growth path (1.8× the height here) and resolution
   independent to first order; whether the real path factor is that large is open.
4. Grain classes do not match: no surviving grains originate mid-build at nucleation
   1.5e13 with Lamb's filters, against their 5:1 originating:established ratio.
5. D_g : D_v (Lamb's within-slice widths) and rho (their population curve below 30 µm,
   the monotonicity of their grains) are constrained only loosely by the published
   figures; the direction of the per-grain paths on the IPF, isotropic in the model, is
   a further check available from their Fig. 4 data.
6. The <001> fibre is twice random, moves weakly with the mobility contrast and falls
   with coherent rotation; there is no measured target.
7. Not yet tested: the undercooling dependence of D_v (alpha > 0).

## 8. Reproduction

```bash
. /home/tron/spack/share/spack/setup-env.sh && spack env activate spparks-dev
# binary: build-verify/spparks (commit 6a47de77 or later); template = the v10 deck
E=examples/gp700_moser; R=/home/tron/gp700_runs
$E/run_layers.sh build-verify/spparks $R/wide12_v10 12 12345 267 10 4e-6 $R/substrate_800um_50um_d10.st
/home/tron/miniconda3/envs/WORK/bin/python $E/gp700_analysis.py $R/wide12_v10/gp700.st \
    --out $R/wide12_v10/analysis_floor100 --floor-um 100 --crop 200 600 200 600
/home/tron/miniconda3/envs/WORK/bin/python $E/grain_paths.py $R/wide12_v10/gp700.st \
    --out $R/wide12_v10/grain_paths --crop 200 600 200 600 --floor-um 100 --grid 30
```

Runs: `/home/tron/gp700_runs/`. Key commits: fa8930f3 (per-voxel diffusion walk),
ad23c1a5 (per-grain selection), 1a9712b3 (kk kernel parity), da2fad29
(`smooth_orientation`), b6e4ba5f / 47546146 / 60b2d711 / 6a47de77
(`misorientation_grain_walk`: coherent walk, OU persistence, path index and d10,
recurrence cache), f5d79cd3 (v10 deck).
