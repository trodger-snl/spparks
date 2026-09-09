# Review of "Misorientation accumulation along columnar grains in GammaPrint-700 laser powder bed fusion: 3D-EBSD (Lamb et al.) versus SPPARKS" (status 2026-09-09)

Reviewer basis: the LaTeX source `docs/SROD_ANALYSIS_REPORT.tex`, its five figures, and the two Lamb et al. papers (Scripta/"Acta" 2023 SROD paper; Met. Trans. A 57A (2026) 2430). No code, run directories, plan documents or logs were consulted.

---

## 1. Summary of the work

The report re-analyses the population SROD curves of Lamb et al.'s 3D-EBSD volume (their Fig. 5a, c) and argues that mean SROD grows as the square root of height with a constant relative 5/95 width, i.e. as a three-dimensional random rotation walk with `RMS^2 = D h`, `D ~ 0.42 deg^2/um`, common to the originating and established grain classes. From their Fig. 4 it further concludes that individual grains rotate coherently along a directed path. A two-term phenomenological orientation-inheritance model is then added to SPPARKS' `additive/texture` apps: a per-voxel random rotation per capture hop (`D_v`) and a per-grain, Ornstein-Uhlenbeck-correlated rotation-vector walk indexed by inheritance path length (`D_g`, `rho`). A Moser-source, 12-layer, 801 um GP700 deck (build "v10") is calibrated on a smaller proxy and reported to match Lamb's originating-grain mean SROD to 0.34 deg RMS over 25-240 um, with band/mean 1.50 and per-grain drifts of 8-15 deg over 400 um. Two modelling changes (per-grain parent selection, copy-mode boundary smoothing) are reported as necessary to avoid selection drift and averaging losses.

---

## 2. Assessment of the central claims

**Claim A (Sec. 3, Finding 1): population SROD is a random rotation walk, `D ~ 0.42`, Maxwell width, same `D` for both classes.**
Partly supported. The sqrt fit through the originating mean over 0-260 um (Fig. `fig:exp` a) is visually good and the band ratio 1.3-1.5 for originating grains is consistent with a chi-3 distribution. But:
- The report's own Fig. `fig:exp`(a) overlays Lamb's reading ("fast 30 um, then 0.85 deg/layer", orange dashed) and it lies essentially on top of the sqrt fit from 30 to 260 um. The published data therefore do **not** discriminate sqrt(h) from Lamb's piecewise-linear interpretation over 90% of the range; the discrimination rests entirely on the first ~30 um (a few digitized points near the origin, where EBSD/reference noise and the 1.5 um offset matter most) and on the established grains.
- "Predicts the established grains without refitting" (Sec. 3) is contradicted by the panel (b) legend, which adds a fitted "initial spread 6.2 deg RMS in quadrature". That number appears nowhere in the text. With one free parameter the established curve is fitted, not predicted, and even so the digitized mean runs visibly above the model beyond ~300 um (15 vs 14.2 deg at 480 um) and looks closer to linear than to sqrt.
- "The measured width ratio is 1.3-1.5 at every height" (Sec. 3) and the caption "flat at 1.3-1.5" are contradicted by panel (c): the established-grain ratio falls to 1.1-1.25 from 150 um onward. This is probably a consequence of the added initial spread (the distribution is no longer central chi-3), but the text should say so instead of asserting a flat 1.3-1.5.
- The 2026 paper reports an independent SROD-vs-distance dataset (its Fig. 7a, 0.1 x 0.1 x 1 um voxels, 120 x 120 x 169 um) with average/maximum accumulation rates of 1.7 and 5.8 deg per 30 um for bulk grains, and concludes "bulk grain rotations over 1 deg per build layer". That is ~2x the 2023 paper's 0.7-1.0 deg/layer and inconsistent with `D = 0.42` (which gives ~0.6 deg/layer at 260 um). The report neither uses nor mentions this second measurement, although it is exactly the sub-30 um data the report says it lacks (Open issue 5).

**Claim B (Sec. 3, Finding 1): individual grains rotate coherently along a directed path.**
Weakly supported. It rests on Lamb's four hand-selected example grains (G1-G4), one of which (G4) splits into two orientation clusters, i.e. does *not* follow a single path. No population statistic for per-grain directedness exists in either paper. The claim is a reasonable reading of Fig. 4 but should be presented as a hypothesis, not a finding.

**Claim C (Abstract, Finding 2): v10 "reproduces the experiment on both levels".**
Population level: a 0.34 deg RMS / -0.28 deg bias fit to a curve the deck was calibrated against (Sec. 6.1, Sobol on the proxy). This is an in-sample fit, not a validation; v8 with a different model fits the same curve to 0.49 deg. Per-grain level: **there is no experimental number to reproduce**. Table `tab:grains` reports drift/spread/max SROD for the simulation only; the comparison in Sec. 7.3 ("maximum SROD of 20-24 deg, the range Lamb report for their example grains") mis-cites Lamb, who report 20-25 deg for the *per-layer mean* SROD of their example grains over >300 um (2023, Fig. 4c text), a maximum SROD of 58.7 deg, and "most columnar grains containing over 25 deg of misorientation". The simulated median top-slab mean SROD is 8.4 deg and median maximum SROD 15 deg (Table `tab:grains`). On the evidence given, the per-grain level is *not* reproduced; at best the tallest simulated grains reach the lower edge of Lamb's example range.

**Claim D (Summary bullet 3, Finding 3): constants are per unit path, 1.8x height, and first-order resolution independent; "the same deck reproduces Lamb at the experiment's 1.5 um voxel".**
Overstated. The resolution test (Sec. 7.4) was run with the **v8 per-voxel-only deck**, on the proxy, not with v10. The coherent term's correlation length is explicitly `Delta_x/(1-rho)` (Sec. 5), so with `rho = 0.9` fixed it halves from 30 to 15 um at 1.5 um, and the long-range amplification `(1+rho)/(1-rho)` per unit path stays the same only because levels are per lattice spacing. The v10 deck is therefore *not* resolution independent unless `rho` is rescaled; this is never tested.

---

## 3. Major issues

**M1. Calibration and validation use the same data.** Sec. 6.1 states Sobol designs on the proxy were used for calibration; the score in Sec. 6.1 is the RMS against Lamb's Fig. 5a mean; Tables `tab:path`/`tab:v10` then report the same score as the result. There is no held-out target. *Why it matters:* a 0.34 deg RMS on a 3-11 deg signal after tuning three constants (`D_v`, `D_g`, `rho`) is not evidence for the model; v8 achieved 0.49 deg with a different model. *Resolution:* validate against something not used in calibration, e.g. Lamb's Fig. 5c established curve including the <30 um region, the 2026 Fig. 7a curve, or Lamb's 2023 grain-size / class-ratio statistics; or at minimum state clearly that the population result is a fit.

**M2. The per-grain "reproduction" has no experimental anchor and mis-cites the one number that exists.** See Claim C. Finding 2 asserts "the per-grain drift-to-spread ratio fixes `D_g : D_v`" while Open issue 5 admits the ratio is "constrained only loosely by the published figures" — both cannot be true. *Resolution:* (i) correct the 20-24 deg comparison to Lamb's actual quantities (per-layer mean 20-25 deg at >300 um; max SROD up to 58.7 deg; "most" grains >25 deg) and confront the simulation's 8.4 deg top-slab mean / 15 deg median max with them; (ii) either obtain per-grain drift/spread from Lamb's data (the authors have the HDF5 volume) or downgrade the claim to "qualitatively consistent with four example grains".

**M3. Grain definition differs from the experiment.** Lamb segment at a 2 deg misorientation threshold in DREAM.3D *before* SROD; filters (>= 60 um height, >= 40 um ESD) and the class assignment act on those segments. The simulation relabels "connected components of spin" (Sec. 6.1). With `D_v = 0.065 deg^2/um` and independent lineages, neighbouring voxels of one simulated grain routinely differ by several degrees after 100 hops, so a 2 deg segmentation would fragment the simulated grains, change the counts, the filters, the references and possibly the SROD population. *Resolution:* apply a 2 deg misorientation segmentation to the simulated volume and report whether grain counts, ESD, and the SROD curves survive.

**M4. The simulated microstructure appears to be locked to the hatch pattern.** In Fig. `fig:v10ipf` the xz grain boundaries fall at x ~ 80, 165, 240, 320 um and the yz boundaries at y ~ 85, 160, 235, 320 um: a regular 80 um period, equal to the hatch spacing (hatches at 40 + 80k um in domain coordinates, crop origin at 200 um, so hatch centrelines sit at crop x = 0, 80, 160, 240, 320). The grains are ~80 x 80 um columns spanning the full height, which also explains the compressed ESD distribution (p50 148, p90 170-182 um; an 80 x 80 x 400 um column has ESD 170 um). Lamb describe "highly irregular" serrated boundaries (2026, Sec. III.B). *Why it matters:* the SROD statistics, the grain counts, the filters and the per-grain paths are all computed on this microstructure; if grain selection is an artefact of pool-centreline geometry, the agreement of SROD may be coincidental, and the "no grains originate mid-build" result (Open issue 4) is a symptom of the same thing. *Resolution:* show a grain-width histogram and its autocorrelation against the hatch pattern; compare with Lamb's in-plane grain size (their Fig. 2f/3b); test a deck with a hatch offset per layer or a non-commensurate hatch.

**M5. Single realisation, unquantified seed noise.** Every number in Tables `tab:path`, `tab:v10`, `tab:grains` comes from one build per deck. "2 and 4 us agree within seed noise" (Sec. 5) is the only mention of seed variance and it is not quantified. The v8-v10 difference in RMS (0.49 vs 0.34 deg), in band (1.41 vs 1.50), and in grain count (86 vs 97) may all be within seed scatter. *Resolution:* at least three seeds for v8 and v10; report mean +/- spread of the score, band and per-grain medians.

**M6. Grain-class mismatch is more than an "open issue".** Lamb's Fig. 5a is >500 grains that originate mid-volume at steady state, hundreds of layers into a 10 mm block. The simulated "originating" set is 83/88 substrate continuations referenced at the first remelt floor above a random-textured 50 um Voronoi substrate, i.e. the transient regime of a build. Section 7.5 argues "because Lamb's two classes share the same D, the walk comparison is unaffected" — but that premise itself rests on the fitted 6.2 deg initial spread (see Claim A) and on an unproven assumption that nothing about origination matters. *Resolution:* compare the simulated floor-referenced grains to Fig. 5c (established, centroid reference), which the analysis script already computes but the report never shows; and discuss why nucleation at 1.5e13 /m^3 with `t_c` 15 K produces no surviving mid-build grains when the experiment shows a 5:1 originating:established ratio.

**M7. The 1.8x path factor is asserted, not measured.** "Retention against height 1.83 / 2.17" (Sec. 7.4) is a ratio of fitted `D` to input `D`; if the 1.8x path factor is inferred from that ratio, the argument is circular (it assumes the walk model to infer the path). The per-site field d10 exists; report the measured mean path length per unit height directly.

**M8. Physical premise is not stated.** Both Lamb papers attribute SROD to plastic deformation: solidification shrinkage in the mushy zone plus thermomechanical cycling in the HAZ of subsequent layers (2026, Sec. IV.C; 2023, strain estimate 3.4%). The report's model applies all rotation at capture. If HAZ cycling contributes, SROD at a given height depends on how many layers were deposited above it, which an inheritance model cannot represent and which changes what "height above origin" means in a 477 um slab cut from a 10 mm block. The report should state that the model is a phenomenological proxy for a post-solidification mechanism, and discuss what this implies for comparing a 12-layer build to steady-state material.

---

## 4. Minor issues and clarity problems

- **Citation.** The 2023 paper is Scripta Materialia (ISSN 1359-6462; the 2026 paper's ref. 12 gives Scr. Mater. 239 (2024) 115770; the manuscript's graphical-abstract page says "Scripta Materialia"). The report's title, abstract, figure caption and Sources paragraph all say "Acta Materialia".
- **Grain counts disagree.** v10: 97 (Table `tab:v10`) vs 96 (Table `tab:grains`, Fig. `fig:summary`). v8: 86 (both tables) vs 88 (Sec. 7.5, "88 filtered grains in the standard crop"). Sec. 7.5 also accounts for 83 + 3 = 86 of 88 grains.
- **Bias numbers mix proxy and full builds.** Summary bullet 4 and Sec. 7.4 quote 1.5 um "bias +0.41 deg" against "+0.03 deg at 3 um" (proxy), while Table `tab:path` gives v8 bias +0.17 deg (full build). State which is which.
- **Analysis floor.** Text: "the analysable volume begins at 105 um"; Fig. `fig:v10srod` caption and the reproduction command: floor 100 um.
- **"band/mean" is not defined as a scalar.** Sec. 6.1 defines the ratio per 10 um bin; Tables report one number (1.50, 1.41, 1.4). Averaged over which bins? Reading Fig. `fig:v10srod` at 200 um gives ~1.2 for both simulation and Lamb, well below 1.50.
- **Fig. `fig:exp` caption** says "Left/Right" for a three-panel figure and describes only two panels; the "6.2 deg initial spread" and the "authors' reading" curve in the legend are absent from the text.
- **Lamb's band is a normal-fit 90% CI, not empirical percentiles** (2023 Fig. 5 caption: "mean and 90% confidence interval of the normal distribution fit"). For a Maxwell distribution `3.29 sigma/mean = 1.39`, so the Maxwell reference value happens to be the same, but the report should state what it digitized and that the equivalence holds.
- **Solidus attribution.** Sec. 2 says the 2026 paper supplies "solidus 1592 K"; the 2026 paper gives no solidus. 1319 C = 1592 K comes from the 2023 paper (Eq. 2-3 constants). "D4sigma" is the report's interpretation; the papers say "80 um spot size" / "80 um Gaussian beam".
- **"over 100 established"** — the 2023 text says "over 500 originating and 100 established grains".
- **Random texture baseline.** "1.7-2.5x random" for the <001>-within-15-deg fraction implies a random fraction ~0.088; six <001> poles within a 15 deg cone give 6 x 2 pi (1 - cos 15 deg)/(4 pi) = 0.102, i.e. 1.5-1.9x. State the baseline used.
- **Undefined terms/quantities:** "retention", "proxy", "hop", "nrefine", `i2 == 3`, "stitch", "box-per-layer", "fast-forward predictor", the mobility law `1 + c2 cos(c3 theta)` (units of theta, what "sign-aligned mean orientation" means), "304L polynomial x 3.5", "t_c" (Table `tab:v10` caption) vs "critical undercooling" (Sec. 5), "alpha > 0" (Finding 7), "twice the Delta_x / V(T_l - T_s) guard".
- **Mean vs RMS in the fit.** `RMS^2 = D h` is stated but the fit is to Lamb's *mean* curve; for a Maxwell distribution mean = 0.921 RMS. Say whether this factor was applied when fitting `D = 0.42` and when computing the "D-fit vs height" row of Table `tab:v10`.
- **Fig. `fig:v10srod`.** The simulated p95 runs 2-3 deg below Lamb's p95 from 150 to 260 um while p5 coincides; the text calls the band "on Lamb". Also the simulated mean flattens above 300 um (12.1 deg at 402 um vs 13.1 deg from `D = 0.43`), which is not discussed.
- **Fig. `fig:grid`.** All 30 "tallest" grains have h = 399-402 um, i.e. the full analysable height; they are all substrate continuations. Say so in the caption.
- **Monotonicity statistic** (0.79 vs 0.78) does not separate v8 from v10, yet Sec. 7.3 uses "rise along a directed path" as the distinguishing feature of v10.
- **Reproduction section** uses machine-local absolute paths and commit hashes; fine for an internal note, but the digitized JSON curves and the two analysis scripts should be named as the reproducible artefacts.
- The Moser argument list `20e-6 20e-6 10e-6` is not explained (beam sigma_x, sigma_y, and ?).

---

## 5. Fidelity to the experimental papers

Checked against the two text extractions.

| Report statement | Paper | Verdict |
|---|---|---|
| 176 W, 1137 mm/s, 80 um hatch, 80 um spot, 30 um layers, 90 deg rotation, 200 C preheat | 2026 Sec. II.A | Correct. |
| "80 um spot (D4sigma)" | 2023: "80 um spot size"; 2026: "80 um Gaussian beam" | D4sigma is the report's assumption, not stated by the papers. |
| k = 22.5, rho = 8650, cp = 600, absorptivity 0.9, liquidus 1646 K, "from the 2026 paper" | 2026 Sec. II.C: 1373 C liquidus, 8650, 22.5 (extrapolated to 950 C), 600 (IN718 at 1000 C), absorption 0.9 "selected to obtain comparable melt pool dimensions" | Correct; alpha = 4.335e-6 follows. The report omits that 0.9 was tuned to match pool size in 3DThesis, a different solver. |
| Solidus 1592 K from the 2026 paper | 2023: T_solidus = 1319 C | Value correct, attributed to the wrong paper. |
| Gradient 3e7 K/m at fusion boundary | 2026 Sec. III.A | Correct. |
| 10 mm track length | 2026: "simulated raster scan ... 10 mm"; blocks are 10 x 10 x 10 mm | Correct. |
| 1.5 um voxels, 385 slices, HDF5, 2 deg segmentation | 2023 | Correct. Omits volume size 1400 x 412 x 477 um and >9000 grains, both relevant to the crop comparison. |
| SROD reference "lowest location of a grain's boundary along the build direction", 1.5 um offset, offsets >8 um matter | 2023 | Correct (quoted accurately). |
| Established grains referenced at centroid on the lowest slice | 2023 | Correct. |
| Filters 60 um (two layers), 40 um ESD | 2023 | Correct. |
| "over 500 originating and over 100 established" | 2023: "over 500 originating and 100 established" | Minor. |
| Per-slice histogram, normal fit, Fig. 5a/5c | 2023 | Correct, but the plotted band is the 90% CI of the normal fit, not percentiles (see Sec. 4). |
| Fig. 4: four grains, IPF/SROD maps, mean SROD per layer, IPF with arrow, one grain splits | 2023 | Correct. (Paper text says red arrow, caption says black.) |
| "maximum SROD of 20-24 deg, the range Lamb report for their example grains" (Sec. 7.3) | 2023: per-layer *mean* SROD 20-25 deg over >300 um; max SROD 58.7 deg; most columnar grains >25 deg | **Misstatement**: compares a simulated maximum to Lamb's per-layer mean. |
| Lamb's steady-state rate | 2023: 0.7-1.0 deg/layer; 2026: average 1.7, max 5.8 deg per 30 um; "over 1 deg per build layer" | The 2023 figure is implicit in the report's orange dashed line; the 2026 rates are not mentioned at all. |
| "cells switch <100> growth variants at fusion lines rather than rotating continuously" | 2026 Sec. III.A, conclusions | Correct. |
| "qualitative <001> fibre" attributed to Lamb (Table `tab:v10`) | Neither paper's text asserts a <001> fibre texture; the 2026 paper discusses both <001>- and <111>-oriented grains | Unsupported by the text; if taken from a figure, say which. |
| "Acta Materialia (2023)" | Scripta Materialia 239 (2024) 115770 | **Wrong journal.** |

---

## 6. Statistical and methodological concerns

1. **Digitization.** Curves were digitized from 300 dpi renders of a heatmap with overlaid white lines; no digitization uncertainty is given. A 0.34 vs 0.49 deg RMS difference between v8 and v10 is of the order of line-width on a 0-30 deg axis and cannot be called a model discrimination without an error estimate.
2. **Discriminating power of the sqrt fit.** As shown by the report's own Fig. `fig:exp`(a), sqrt(h) and "30 um transient + linear" are indistinguishable over 30-260 um. The only regions with leverage are 0-30 um (few points, largest EBSD/reference noise) and the established curve, which needed an extra fitted parameter and still departs from the model beyond 300 um.
3. **Sample sizes.** 96-97 simulated grains (about 40 of which span the full height) vs >500 originating + >100 established experimental grains, in a 400 x 400 x 400 um crop vs a 1400 x 412 x 477 um volume. Per-grain percentiles (p10/p90 of drift) from ~40 tall grains are noisy; no confidence intervals are given anywhere.
4. **Single realisation per deck** (see M5). The Sobol design's size, ranges and objective are not stated; "seed noise" is invoked but never measured.
5. **Calibration = validation** (M1).
6. **The 400 um build vs the experiment.** The simulation is the first 12 layers above an equiaxed random substrate; the experiment is a slab from steady-state columnar material hundreds of layers deep, with a 5.7 deg initial spread in its established grains. The report compares transient-regime substrate continuations to steady-state originating grains and argues the walk is unaffected; this is an assumption whose test (floor-referenced simulation vs Fig. 5c) is available but not shown.
7. **What the population data can and cannot distinguish.** The report is right that Lamb's Fig. 5 cannot separate coherent from per-voxel accumulation; the same logic applies to the direction of causality: any process with independent finite-variance increments per unit height (plastic bending in the HAZ, cell torsion, inheritance noise) gives the same sqrt(h) and Maxwell width. The random-walk analysis constrains the *statistics* of the process, not its *mechanism*; Finding 1 should say so.
8. **Grain-size distributions** are never compared to the experiment although they control the filters, the counts and the class ratio; the simulated ESD distribution (p10-p90 = 76-170 um) is suspiciously narrow and hatch-commensurate (M4). Also: 97 grains with median ESD 148 um cannot fit into a 400 x 400 x 410 um crop (6.6e7 um^3) unless ESD is computed on the uncropped grain — clarify how grains cut by the crop boundary are treated.
9. **Small-angle composition.** `RMS^2 = n theta^2` treats rotations as additive vectors; at 15-25 deg the non-commutativity error is a few percent, comparable to the reported bias. Worth one sentence.
10. **Per-grain metrics.** Drift/spread (1.66 vs 0.60) is the only statistic that distinguishes v8 from v10, and it has no experimental counterpart. By construction `D_g (1+rho)/(1-rho) / D_v = 0.19/0.065 ~ 2.9`, whose square root (1.7) is the reported ratio; Table `tab:grains` therefore confirms the code does what the equations say, nothing more.

---

## 7. Questions for the authors

1. How was `D = 0.42` fitted — least squares on the mean, on the mean squared, with or without the Maxwell mean/RMS factor, over which height range, and with what weighting of the dense 0-30 um points?
2. What is the "6.2 deg initial spread" in Fig. `fig:exp`(b), how was it obtained, and why is it not in the text?
3. Why does the established-grain band ratio in panel (c) fall to 1.1-1.2 while the text says 1.3-1.5 at every height?
4. Is the 1.8x path-to-height factor measured from the d10 field, or inferred from the fitted-to-input `D` ratio?
5. Was the resolution test ever run with the v10 deck (coherent walk + `rho`)? If `rho` is per lattice level, what is the intended rule for changing `Delta_x`?
6. How many seeds were run for v8 and v10, and what is the seed-to-seed scatter of the RMS score, band ratio, grain count and per-grain medians?
7. Sobol calibration: how many samples, which parameters and ranges, which objective, and was any target withheld?
8. Are the grain boundaries in Fig. `fig:v10ipf` at 80 um intervals coincident with the hatch centrelines? What is the in-plane grain-width distribution and how does it compare with Lamb's Fig. 2f/3b?
9. What happens to grain counts, ESD, and the SROD curves if the simulated volume is segmented at a 2 deg misorientation threshold like the experiment?
10. How are grains intersecting the 400 um crop faces treated, and are ESD/height computed on the cropped or the full grain?
11. What is the floor-referenced (established-type) simulated SROD against Lamb Fig. 5c? The script computes it; please show it.
12. How do the simulated top-slab mean SROD (median 8.4 deg) and maximum SROD (median 15 deg) compare with Lamb's per-layer means of 20-25 deg for G1-G4, "most columnar grains over 25 deg", and 58.7 deg maximum?
13. Why is the 2026 paper's Fig. 7a accumulation rate (1.7 deg / 30 um average) not used, given it is the sub-30 um, sub-voxel measurement the report says it needs?
14. Which figure supports the "qualitative <001> fibre" attributed to Lamb, and what random baseline underlies "1.7-2.5x random"?
15. Nucleation at 1.5e13 /m^3 yields no surviving mid-build grains. Was the nucleation density or undercooling scanned specifically for the originating:established ratio (5:1 in Lamb)?
16. Given that Lamb attribute SROD to post-solidification plasticity (HAZ cycling), what is the argument that an at-capture inheritance model is the right proxy, and does the model predict any dependence of SROD on the number of layers subsequently deposited?

---

## 8. Overall recommendation: **major revision**

The re-analysis of Lamb's Fig. 5 is a useful and largely correct observation — the population SROD is consistent with a finite-variance random rotation walk of chi-3 width — and the modelling effort is substantial and transparently reported, including its dead ends. But the report's stronger claims are not supported by the evidence it presents: sqrt(h) is not distinguishable from Lamb's own reading over most of the height range and the established-grain "prediction" involves an undisclosed fitted parameter; the per-grain "reproduction" has no quantitative experimental target and the one comparison offered mis-cites Lamb's per-layer means as maxima, while the simulated grains fall well short of the 20-25 deg per-layer means and ">25 deg in most grains" that Lamb actually report; the population fit is in-sample; every number comes from one realisation; the simulated microstructure appears hatch-locked and uses a different grain definition from the experiment; and the resolution-independence claim was tested only on the model variant without the resolution-dependent term. There are also factual slips (journal, solidus attribution, grain counts, floor height, band ratio statements contradicted by the report's own figure). The work would be publishable as "a phenomenological inheritance model fitted to Lamb's population SROD, with an untested per-grain extension" once the claims are scaled to the evidence, seed scatter and digitization error are quantified, the 2 deg segmentation and floor-referenced comparisons are shown, the hatch-periodicity of the grain structure is examined, and the two Lamb papers are cited accurately.
