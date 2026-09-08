# SROD analysis report: misorientation accumulation in GammaPrint-700 PBF-L, experiment vs SPPARKS

Status: 2026-09-07. **Reference version: `SROD_ANALYSIS_REPORT.tex` (PDF alongside); this file mirrors it.** Companion documents: `MISORIENTATION_DIFFUSION_PLAN.md`
(model changes), `GP700_MOSER_SROD_DECK_PLAN.md` (deck development log).
Figures: `srod_experimental_analysis.png`, `srod_random_walk.png`,
`gp700_results_v8/`, `gp700_results_v9/`, `gp700_results_dx1p5/`.

## 1. Summary

- Lamb et al. (Acta Mater. 2023) measured the starting reference orientation
  deviation (SROD) of columnar grains in 3D-EBSD of PBF-L GammaPrint-700.
  Re-analysing their Fig. 5 shows SROD is a **random-axis rotation walk**:
  RMS² = D·h with D ≈ 0.42 deg²/µm against build height, the same D for
  grains originating inside the volume and for established grains, and a
  5/95 band of (p95 − p5)/mean ≈ 1.4, the Maxwell width of a 3-D walk. The
  authors' "fast first 30 µm then constant" reading is √h.
- SPPARKS `additive/texture` and `additive/texture/kk` now carry that model:
  `misorientation_diffusion D thresh` rotates each captured voxel by
  θ = √(D·hop) about a random axis relative to its parent. The old
  misorientation cap and gradient-target rotation are removed.
- A GP700 Moser-source deck (`examples/gp700_moser/`) reproduces the
  experiment: SROD RMS deviation from Lamb's mean 0.49° over 25–240 µm,
  bias +0.17°, band/mean 1.41 vs 1.4, with the √h shape over the full
  400 µm build. Two model corrections were needed on the way: per-grain
  parent selection (mobility weighting between grains, not between voxels,
  which had rotated grains toward <001>) and `smooth_orientation copy`
  (the smoothing pass averaged neighbour quaternions, which cancelled 60 %
  of the walk and widened the band to 1.8).
- The deck D is 0.22 deg²/µm, half Lamb's height fit, because the growth
  path is 1.8× the height. D is resolution-independent to first order: at
  Lamb's own 1.5 µm voxel the same deck stays on Lamb (bias +0.41°), with
  10–18 % more accumulation per height than at 3 µm.
- Open: the simulated columnar grains are epitaxial continuations of the
  substrate (83 of 88 filtered grains originate at the first remelt floor),
  so the "originating" class in Lamb's sense (> 500 grains, 5:1 over
  established) is essentially absent; the <001> fibre (0.19–0.24 within 15°
  of BD, 2× random) has no quantitative experimental target.

## 2. Experimental methods (Lamb et al. 2023, Met. Trans. A 2026)

Material and process: CoNi-base superalloy GammaPrint-700, laser powder bed
fusion, 176 W, 1137 mm/s, 80 µm spot (D4σ), 80 µm hatch, 30 µm layers, 90°
interlayer rotation, 200 °C plate preheat (473 K). The 2026 paper supplies
the thermophysical set used for its 3DThesis simulations: k 22.5 W/m·K,
ρ 8650 kg/m³, cp 600 J/kg·K (α 4.335e-6 m²/s), absorptivity 0.9, liquidus
1646 K, solidus 1592 K.

3D dataset: TriBeam tomography (femtosecond-laser serial sectioning with
EBSD on each slice), 1.5 µm in-plane step and 1.5 µm slice thickness, 385
slices, reconstructed into one HDF5 volume and segmented into grains with a
2° misorientation threshold before any SROD calculation.

SROD definition: for each grain the reference orientation is taken near
the grain's origination point, "the lowest location of a grain's boundary
along the build direction", offset one voxel layer (1.5 µm) upward so the
reference pixel is not a boundary pixel with a convoluted EBSD pattern.
Offsets up to 8 µm do not change the result. Two grain classes: originating
grains, whose lowest point lies inside the volume; established grains, cut
by the dataset floor, referenced at the centroid of their footprint on the
lowest slice. Grains shorter than 60 µm (two layers) or with equivalent
spherical diameter under 40 µm are excluded as overtaken during
competitive growth. > 500 originating and > 100 established grains remain.
Each grain is sliced along the build direction; each slice gives a SROD
histogram fitted to a normal distribution, plotted against height above
the origin (originating) or above the dataset floor (established) —
Fig. 5a and 5c.

The 2026 paper adds the cellular-scale finding used here: cells switch
<100> growth variants at fusion lines rather than rotating continuously, so
no gradient-directed rotation is modelled.

## 3. Analysis of the experimental data

Fig. 5a/5c were digitized from 300-dpi renders of the paper
(`srod_random_walk.py`; curves in `lamb2023_fig5{a,c}_srod.json`). The
mean, 5th and 95th percentile curves were tested against the model of a
rotation walk: each increment is a small rotation of angle θ about a random
axis; after n independent increments the misorientation from the origin
has RMS² = n·θ² = D·d, and its distribution is Maxwell (χ with 3 degrees of
freedom), for which (p95 − p5)/mean = 1.38 independent of d.

Findings (`srod_experimental_analysis.png`, `srod_random_walk.png`):
D = 0.42 deg²/µm fitted on the originating grains predicts the established
grains without refitting; the measured width ratio is 1.3–1.5 at every
height; the apparent two-regime shape is the square root. A constant per-hop
angle is not required — only that the increments are independent with a
finite variance, so D can carry an undercooling dependence without
changing the √h law.

## 4. Simulation methods

Code: SPPARKS, branch SURGE-kokkos, app `additive/texture/kk` (Kokkos/HIP
on an AMD gfx1151 APU), one MPI rank. The CPU app `additive/texture`
carries the same model and is the correctness reference. Temperature field
is deterministic and bit-identical between the two apps
(`examples/am_validation/`).

Thermal source: Moser Green's-function solution with the GP700 constants
above (`setup_temperature_source moser 176 0.9 22.5 4.335e-6 473 600 20e-6
20e-6 10e-6`), evaluated on the device for every site each step. Scan:
`laser_path ... hatch 80e-6 N serpentine pause 8.8e-3`, the pause being the
real 10 mm track length over the scan speed; the fast-forward predictor
skips idle gaps. Single-hatch check: pool 147 × 69 µm, thermal gradient
1e7–2.9e7 K/m at the fusion boundary (paper: 3e7).

Build: box-per-layer stitch continuation (`run_layers.sh`). Domain 801 µm
square (267 voxels at 3 µm), 10 hatches at 40 + 80k µm with lines 100 µm
past both edges, 12 layers of 30 µm, 90° rotation per layer, each box a
180 µm window from the laser plane down; each layer starts from the
preheat (recoat interval). Substrate: 150 µm of 50 µm Voronoi grains with
uniform random orientations (`make_substrate.py`); five overlapping
hatches remelt 80–100 µm, so the analysable volume begins at 105 µm.
Timestep 4 µs (twice the dx/V(Tl − Ts) guard; 2 and 4 µs agree within seed
noise). The 1.5 µm run uses 400 voxels over 600 µm, 7 hatches, 8 layers,
dt 2 µs.

Solidification kinetics: `solid_front_vel` = the 304L polynomial × 3.5
(front at 25–35 K undercooling, ~0.56 m/s, keeping up with the 1.137 m/s
scan); `nucleation_density 1.5e13` m⁻³, `critical_undercooling 15` K,
deviation 2 K. Parent selection per grain: candidate neighbours are grouped
by grain, each grain weighted by the mobility 1 + c2·cos(c3·θ) of its
sign-aligned mean orientation against the local thermal gradient
(c2 2.8, c3 2.5; negative clamped to 0), then the parent voxel is drawn
uniformly inside the chosen grain.

Misorientation: `misorientation_diffusion 2.2e5 30` — every captured voxel
takes its parent's orientation rotated by θ = √(D·hop) about a random axis
(hop = lattice distance to the parent, m; D in deg²/m); nucleus shells
rotate relative to the seed by the same rule; a rotation past the 30°
threshold starts a new grain (never reached in practice). No cap, no
gradient target, alpha 0 (constant D).

Boundary smoothing: one greedy steepest-descent pass per captured voxel
(`smooth_greedy_multiproposal on`, nrefine 1) the step after capture;
`smooth_orientation copy` — a voxel that changes grain takes one random
neighbour's orientation from that grain (the default `average` takes the
sign-aligned mean).

Kokkos kernel corrections made during this work, all now CPU-parity:
race-free multicolour neighbour reads (were x-index-ordered), front advance
only beside solid, nrefine passes honoured, zero-gradient mobility, the
`laser_path` time anchor after `reset_time`.

## 5. Simulation analysis methods

`gp700_analysis.py` (spparks-micro-analysis tools, WORK conda env): read
the last stitch frame; solid = `i2 == 3` (unremelted substrate, `i2 == 1`,
excluded); crop to the central 400 µm laterally (300 µm on the proxy),
optional ceiling; relabel grains as connected components of spin; grain
volume, ESD, extents; Lamb's filters (height ≥ 60 µm, ESD ≥ 40 µm); origin
= the grain's lowest solid voxel, reference orientation = that voxel's
quaternion (exact in the simulation, so no offset); grains crossing the
analysis floor are "established" and referenced at their floor-slice
centroid. SROD = symmetry-reduced disorientation of every voxel from its
grain's reference, binned by height above origin/floor (10 µm) and,
separately, by straight-line distance; per bin mean, p5, p95, RMS.
D-fit: RMS² = D·h through the origin, volume-weighted. Score: RMS of
(simulation mean − Lamb Fig. 5a mean) over 25–240 µm, plus mean bias,
(p95 − p5)/mean, the fraction of solid within 15° of <001> ∥ BD, grain
counts. Sobol designs (scipy qmc) on the 600 µm / 8-layer proxy (6 min per
box at 3 µm) were used for calibration.

## 6. Results

### 6.1 Calibration path

| build | change | SROD RMS / bias vs Lamb | band/mean | <001> |
|---|---|---|---|---|
| v2 | first full build, D 0.42, cap removed | 2.2° / +3.6° (linear growth) | 1.4–1.8 | 0.26 |
| v4 | per-grain parent selection | 2.9° low | 1.8 | 0.11 |
| v5 | Sobol: D 1.15, nucl 1.5e13, tc 15, c2 2.8 | 0.59° / −0.16° | 1.79 | 0.34 |
| v6 | 50 µm Voronoi substrate, nrefine 1, D 0.40 | 2.2° / −1.8° | 1.85 | 0.16 |
| v7a | D 0.80 | 0.90° / +0.24° | 1.85 | 0.20 |
| **v8** | **`smooth_orientation copy`, D 0.22** | **0.49° / +0.17°** | **1.41** | 0.19 |
| v9 | c2 9, nucl 2e13 | 0.52° / +0.02° | 1.45 | 0.22 |

v2's linear growth was selection drift: mobility weighting among the
randomly rotated voxels of one grain rotated grains toward <001> with
height (fixed by per-grain selection). v4–v7 then retained only 0.3–0.45 of
the input D against height in every configuration while the single-grain
walk deck retained 1.2: the smoothing pass replaced each reassigned voxel's
orientation with the average of its new neighbours, cancelling the fresh
rotations, and narrowed the intragranular distribution (band 1.8). Copy
mode restored retention to 1.6–1.9 and the band to 1.4 on the proxy at
D 0.42; scaling that curve onto Lamb gave D 0.22.

### 6.2 v8 against Lamb

| | v8 | Lamb 2023 |
|---|---|---|
| SROD RMS deviation, 25–240 µm | 0.49° | – |
| mean bias | +0.17° | – |
| mean SROD at 54 / 204 / 402 µm | 4.4° / 8.5° / 12.3° | 3.8° / 8.8° / – (ends at 262 µm) |
| (p95 − p5)/mean | 1.41 | 1.4 |
| D-fit vs height / straight line | 0.41 / 0.40 deg²/µm | 0.42 |
| grains passing filters (400 µm crop) | 86 | > 600 (full volume) |
| ESD p10/p50/p90 | 73 / 151 / 182 µm | – |
| lateral extents (median) | 114 × 99 µm | – |
| <001> within 15° of BD | 0.19 (random 0.09) | qualitative <001> fibre |

`gp700_results_v8/srod_height.png` shows the simulated mean on Lamb's curve
over the whole range with the √h shape and the 5/95 band matching Lamb's
percentile curves.

### 6.3 Resolution: 1.5 µm vs 3 µm

Same deck, geometry and seed on the proxy:

| | 3 µm | 1.5 µm |
|---|---|---|
| SROD RMS / bias vs Lamb | 0.44° / +0.03° | 0.49° / +0.41° |
| band/mean | 1.41 | 1.43 |
| retention (D-fit ÷ input), height / straight line | 1.83 / 1.73 | 2.17 / 2.35 |
| <001> within 15° | 0.21 | 0.20 |
| ESD p10/p50/p90 | 68 / 123 / 159 | 66 / 122 / 150 |
| wall time per box | 6 min | 47 min |

Bin-by-bin agreement 0.1–0.4° to 200 µm, 7–10 % higher at 1.5 µm beyond
245 µm. The per-hop rule θ = √(D·hop) is resolution independent to first
order; the finer lattice resolves a slightly longer growth path per unit
height. An exact match at 1.5 µm would take D ≈ 0.19.

### 6.4 Crop sensitivity and grain origins

Analysing v8 on the full 801 µm domain, the central 400 µm, the central
200 µm, or with the never-remelted top layer removed changes the SROD bias
by < 0.2° and the band by < 0.03; only the fibre fraction moves (and is
unmeasurable on 24 grains). Of the 88 filtered grains in the standard
crop, 83 originate at the layer-0 remelt floor (105–130 µm, 99 % of the
filtered volume), 3 above 160 µm: the columnar grains are epitaxial
continuations of substrate grains, and the simulated "originating" SROD is
an established-grain SROD referenced at the remelt floor. Because Lamb's
originating and established classes share the same D, the walk comparison
is unaffected, but the model does not generate Lamb's population of
grains that originate mid-build.

### 6.5 Texture

Sobol over c2 2–8 and nucleation 5e12–5e13 with D fixed: <001> = −0.042 +
0.0125·c2 + 0.0073·ln(nucl) (R² 0.72), flat at 0.22 below c2 ≈ 5; the SROD
score is unaffected throughout. On the full build c2 9 gives 0.22 (rising
0.16 → 0.24 with height; c2 2.8 saturates at 0.19). The random walk spreads
orientations out of the 15° cone (13° RMS over 400 µm) about as fast as
selection fills it. Neither paper quantifies the fibre.

### 6.6 Per-grain orientation paths and the grain-level walk

Lamb's Fig. 4 follows individual grains (IPF map, SROD map, mean SROD per
layer, and every voxel on the build-direction IPF with an arrow for the
rotation path bottom → top; G4 splits into two clusters). `grain_paths.py`
reproduces that layout. For v8's four tallest grains the mean orientation
drifts only 3–6° bottom → top while the voxels spread 12° RMS about the
top-slab mean: the per-voxel walk lets the lineages of one grain diverge
independently, so the cloud expands isotropically and its mean stays put.

`misorientation_grain_walk D_g` (b6e4ba5f, `src/grain_walk.h`) adds a
coherent walk: every voxel of grain g crossing lattice level k → k+1 gets
the same rotation, angle √(D_g·dx) about an axis hashed from (g, k) — path
independent, no per-grain state; the per-voxel walk adds spread on top.
Proxy runs with the v8 deck otherwise (`gp700_results_grainwalk/`):

| | voxel only (D_v 0.22) | mixed (D_g 0.30 + D_v 0.065) | grain only (D_g 0.42) |
|---|---|---|---|
| SROD RMS / bias vs Lamb | 0.44° / +0.03° | 0.54° / +0.34° | 0.47° / +0.04° |
| band/mean (population) | 1.41 | 1.46 | 1.48 |
| D-fit vs height (deg²/µm) | 0.40 | 0.43 | 0.38 |
| same-grain z-neighbour disorientation, median / p90 | 3.8° / 12.5° | 2.3° / 6.7° | 1.1° / 1.7° |
| 4 tallest grains: drift of the mean, bottom → top | 0.7–6.3° | 6.5–8.8° | 5.3–11.3° |
| 4 tallest grains: spread about the top-slab mean | 7.6–11.7° | 4.2–7.1° | 1.4–2.0° |
| 4 tallest grains: max SROD | 20–28° | 16–19° | 8–12° |

All three sit on Lamb's population curve with the same band ratio, so the
population statistics cannot distinguish coherent from per-voxel
accumulation; the per-grain analysis can, and D_g : D_v sets drift against
spread. The grain-only run also shows that a memoryless walk makes
individual SROD(h) curves non-monotonic (8° at 150 µm back to 2° at
240 µm) whereas Lamb's G1–G4 rise steadily to 20–25°, pointing to
persistent (correlated-axis) rotation; and the per-level increment shows
as horizontal bands. Lamb's within-slice widths are not published, so the
split is constrained only to drift ≳ spread; the mixed setting is the
working choice.

## 7. Findings and open issues

1. SROD in GP700 is a rotation walk with D ≈ 0.42 deg²/µm against height
   and a Maxwell width; both grain classes obey it.
2. The model reproduces level, shape and width once orientation is
   inherited from one parent voxel and never averaged: per-grain parent
   selection and copy-mode smoothing are required; averaging in either
   place damps the walk and narrows the band.
3. Deck D = 0.22 deg²/µm at 3 µm (0.19 at 1.5 µm): the walk runs along the
   inheritance path, 1.8× the height in this remelt geometry. Whether the
   real path factor is that large is an open physical question; D should be
   read as per path length.
4. Grain classes do not match: no surviving grains originate mid-build at
   nucleation 1.5e13 with these filters. Lamb's 5:1 originating:established
   ratio implies nucleation or lateral overgrowth the model does not produce
   at these settings.
5. Individual grains in the experiment rotate coherently along a path; the
   per-voxel walk alone spreads without drifting. The grain-level walk
   reproduces the drift with the same population statistics; Lamb's steady
   per-grain rise suggests persistent rather than memoryless rotation.
6. The <001> fibre is 2× random and moves only weakly with the mobility
   contrast; there is no measured target.
7. Not yet tested: the undercooling dependence of D (alpha > 0); the deck
   is hot enough that alpha barely acts on the single-hatch check deck.

## 8. Reproduction

```bash
. /home/tron/spack/share/spack/setup-env.sh && spack env activate spparks-dev
# binary: build-verify/spparks (>= commit da2fad29)
E=examples/gp700_moser; R=/home/tron/gp700_runs
$E/run_layers.sh build-verify/spparks $R/wide12_v8 12 12345 267 10 4e-6 $R/substrate_800um_50um.st
/home/tron/miniconda3/envs/WORK/bin/python $E/gp700_analysis.py $R/wide12_v8/gp700.st \
    --out $R/wide12_v8/analysis_floor100 --floor-um 100 --crop 200 600 200 600 --D 2.2e5
DX=1.5e-6 $E/run_layers.sh build-verify/spparks $R/dx1p5_v8 8 12345 400 7 2e-6 $R/substrate_600um_50um_dx1p5.st
```

Runs: `/home/tron/gp700_runs/wide12_v{2..9}`, `dx1p5_v8`, `dx3_v8_ref`,
`sobol{1,2}`, `sobol_fibre{1,2}`, `smooth_*`. Key commits: 8325eb71 (cap),
fa8930f3 (diffusion model), ad23c1a5 (per-grain selection), 1a9712b3 (kk
kernel parity), da2fad29 (`smooth_orientation`), a0fb1011 (v8 deck),
a3a66f41 (voxel-size driver), b6e4ba5f (`misorientation_grain_walk`).
