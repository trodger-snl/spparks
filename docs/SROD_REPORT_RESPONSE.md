# Response to the review of the SROD report (revision 2, 2026-09-09)

Review: `SROD_REPORT_REVIEW.md` (reviewer saw the report, its figures and the two Lamb papers only).
Analyses added for this revision are in `gp700_results_review/`; the development log is
`GP700_MOSER_SROD_DECK_PLAN.md`. Items marked *pending* wait for runs queued on the GPU
(`/home/tron/gp700_runs/chain_review.sh`).

## Major issues

**M1 Calibration = validation.** Agreed. The Fig. 5a score is now labelled in-sample, and two
targets that were never used in calibration are added (report §7.2): the established-grain curve
of the same paper (Fig. 5c, floor-referenced simulated grains): v10 RMS 0.28°, bias −0.11°,
band 1.39; v8 (per-voxel only) RMS 2.99°, bias +2.98°. And the 2026 Fig. 7a heatmap, digitized
(`lamb2026_fig7a_srod.json`): v10 reproduces its modal trajectory (0.8 → 4.8° vs 1.6 → 4.7° over
5–115 µm); neither model produces its secondary 10–15° band. The established test discriminates
the models; the population curve alone does not, as the reviewer said.

**M2 Per-grain reproduction / mis-citation.** Corrected. The "20–24° max SROD vs Lamb's 20–25°"
sentence is replaced by a comparison against what Lamb report (report Table 4): per-layer mean at
300 µm (v10 median 10.8°, max 18.0° vs G1–G4 20–25°), fraction of grains with max SROD > 25°
(5 % vs "most"), maximum SROD (33° vs 58.7°), and the 2026 accumulation rates (v10 mean 0.84,
max 1.92°/30 µm vs 1.7 / 5.8). Conclusion stated: the model produces directed per-grain paths
but accumulates at half Lamb's average rate with far smaller extremes; the experiment has a
heavy tail the single-Gaussian D_g does not give. The drift-to-spread ratio is acknowledged as
confirming the construction, not an experimental value.

**M3 Grain definition.** Done (report §7.6): 2° re-segmentation with
`spparks_micro_analysis.grain_segmentation.segment_grains`. v10: 411 k labels, 139 filtered
grains, ESD p50 29 µm, SROD score within 0.6°; v8: 758 k labels, 54 filtered, ESD 8 µm, no grain
taller than 165 µm. The simulated field carries voxel-scale orientation noise the EBSD does not,
so the segmentations are not equivalent; stated as a limitation.

**M4 Hatch-locked microstructure.** Confirmed and quantified (report §7.5, `hatch_periodicity.py`):
boundary-density autocorrelation 0.6–0.7 at the 80 µm lag, 43–62 % of spectral power at the hatch
period, in-plane widths 30–51 µm median. Cause: the deck registers the hatches of layers n and
n+2. The papers specify bidirectional, 90° rotation repeating every two layers, 45° to the block
edges, but not the registration. A per-layer hatch offset was added to the deck driver
(`HATCH_OFFSET=random`) and a sensitivity run at the v10 settings is *pending*.

**M5 Seed scatter.** *Pending*: three seeds each of the v8 and v10 settings on the proxy and a
second full v10 build are queued; tables will carry the scatter.

**M6 Grain classes.** The floor-referenced (established) comparison is now shown (M1). The
origin census over 26 existing runs (nucleation 6.5e12–8.5e13 m⁻³, t_c 3–23 K) finds 0–7 filtered
grains per run originating > 60 µm above the remelt floor; no tested setting approaches Lamb's
5:1. Stated as an open issue; nucleated grains are overgrown at every tested density.

**M7 Path factor.** Measured from the d10 field: median 1.47 (report §7.4). The 1.8 quoted before
was the ratio of fitted to input D, which includes the walk's own excess over the height index;
the text now separates the two.

**M8 Physical premise.** Added (report §5, "Misorientation model"): the model is an at-capture
proxy for the plasticity Lamb attribute to mushy-zone shrinkage and HAZ cycling, cannot represent
dependence on layers deposited above, and compares a 12-layer build with block-interior material.

**Claim D, resolution.** The 1.5 µm test used the per-voxel deck only; stated. The rescaling rule
for the coherent term (rho' = 1 − (1 − rho)·dx'/dx, D_g rescaled to keep the long-range
accumulation) is given, and a 1.5 µm run of the v10 deck with rho 0.95, D_g 0.0049 is *pending*.

## Minor issues

- Journal: Scripta Materialia 239 (2024) 115770 throughout (confirmed from the 2026 paper's ref. 12).
- Grain counts: both scripts share the labelling; the residual 97/96 and 88/86 differences are
  grains with fewer than 20 voxels in their bottom or top slabs, dropped by `grain_paths.py`; stated.
- Proxy vs full-build biases are now labelled where quoted.
- Floor: analysable solid begins at 105 µm; the analysis floor is 100 µm; both stated.
- Band ratio defined as the average of the per-bin (p95 − p5)/mean over 25–240 µm.
- Fig. 1 caption rewritten for three panels; the 6.2° fitted initial spread and the established
  width ratio (1.1–1.25 above 150 µm) are in the text; "predicts without refitting" removed.
- Lamb's band is the 90 % interval of a normal fit; equivalence to the Maxwell 1.38 stated.
- Solidus attributed to the 2024 paper; D4σ marked as the deck's reading; absorptivity 0.9
  noted as tuned in 3DThesis; "over 500 originating and 100 established"; dataset volume and
  grain count added; "<001> fibre" no longer attributed to Lamb.
- Random baseline computed by Monte Carlo: 0.102 within 15°, 0.045 within 10°.
- Glossary items (hop, nrefine, stitch, i2 codes, t_c, the mobility law, the Moser arguments,
  the timestep guard) defined where first used.
- The Maxwell mean/RMS factor 0.921 and the fitting range are stated.
- Fig. 4 grid caption notes that all 30 grains span the full height.
- Reproduction section names the JSON curves, scripts and CSV tables as the artefacts.

## Questions

1. D fitted by least squares of RMS² = D·h to the digitized mean over 0–265 µm, mean converted
   to RMS with the Maxwell factor 0.921; the dense 0–30 µm points enter unweighted.
2. The 6.2° RMS initial spread is a fitted quadrature offset for the established grains (their
   orientation scatter at the dataset floor); now in the text and caption.
3. With that offset the established distribution is non-central, so its width ratio falls; the
   text now says so instead of "1.3–1.5 at every height".
4. Measured from d10: 1.47 (was inferred).
5. Not before; the rescaling rule is stated and the run is pending.
6. One seed each so far; the seed study is pending.
7. Sobol: two 16- and 8-point designs over D 0.2–2.0, nucleation 1e13–1e14, t_c 3–25 K, c2 1–4,
   objective RMS to Fig. 5a over 25–240 µm; no target withheld (hence the held-out tests added).
8. Yes, at the hatch centrelines; widths reported; the comparison with Lamb's Fig. 2f is
   qualitative because they give no in-plane size.
9. Reported (M3).
10. Grains cut by the crop faces are kept with their cropped volume; ESD and height are computed
    on the cropped grain, so ESD is underestimated for cut grains and the count can exceed what
    the crop volume would hold for uncut grains.
11. Shown (M1).
12. Reported (M2): 10.8° median per-layer mean at 300 µm, 5 % of grains above 25°.
13. Digitized and used (M1); the paper's average rate corresponds to the modal ridge near the start.
14. Attribution removed; baseline 0.102.
15. Census over 26 runs (M6): no.
16. The model is a phenomenological proxy (M8); it predicts no dependence on later layers, which
    is a stated limitation.
