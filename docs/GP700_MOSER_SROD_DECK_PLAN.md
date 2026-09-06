# Plan 2: GammaPrint-700 Moser deck to test against the published thermal simulation and SROD data

Status: **IN PROGRESS** (2026-09-04): §2 hatch extension, deck, driver and
gate 5.1 done (see "Progress log" at the end); 5.2/5.3 running. Depends on Plan 1
(`docs/MISORIENTATION_DIFFUSION_PLAN.md`) for the rotation law and on one
small scan-path extension (§2). Targets Lamb et al., Acta Mater 2023 (SROD,
Fig. 5) and Lamb et al., Met Trans A 2026 (3DThesis thermal simulation,
Fig. 3), both on the same print of GammaPrint-700.

## 1. Target process and material (from the papers)

| quantity | value | source |
|---|---|---|
| laser power / speed | 176 W / 1137 mm/s | Met Trans §II |
| spot | 80 µm (Gaussian) | both |
| hatch | 80 µm, bidirectional, 90° interlayer rotation | Met Trans §II |
| layer | 30 µm | both |
| preheat | 200 °C = 473 K | both |
| liquidus / solidus | 1373 °C = 1646 K / 1319 °C = 1592 K | Met Trans sim / Acta eq. 2 |
| k, ρ, cp | 22.5 W/m·K, 8650 kg/m³, 600 J/kg·K → α = k/(ρcp) = **4.335e-6 m²/s** | Met Trans sim |
| absorptivity | 0.9 (chosen by them to match pool size) | Met Trans sim |
| their thermal sim | 3DThesis, 1 µm / 1 µs, single 10 mm raster: G_max ≈ 3e7 K/m at the fusion boundary, V slow rise from the periphery then > 1 m/s at the very end (top centerline) | Met Trans Fig. 3 |
| EBSD volume / voxel | 1400 × 412 × 477 µm, 1.5 µm | Acta §2 |
| SROD targets | Fig. 5a/5c digitized in `docs/srod_random_walk.py`; D = 0.42 deg²/µm | Acta |
| grain filters | height ≥ 60 µm, ESD ≥ 40 µm; > 500 originating, > 100 established grains | Acta §3 |

Not given: melt pool width/depth (Fig. 3 has no scale bar; only the
half-ellipse aspect ≈ 2:1 width:depth is readable). Treat sz (§3) as the
knob that sets depth and match the aspect, not absolute size.

## 2. Prerequisite code: hatched, layered scan paths

`laser_path` today is one line repeated in place (`repeats N`, `pause`).
`MoserTemperatureSource::rebuild_scan_from_pass_list_` already emits one
GREENAM waypoint pair per pass from a single (x0,y0)→(x1,y1); GREENAM itself
takes arbitrary waypoint lists. Extension (~50 lines, Moser only):

```
laser_path start X0 Y0 Z0 end X1 Y1 speed V [hatch <spacing> <n>] [serpentine] [pause T]
```

- `hatch s n`: n passes, pass k offset by k·s along the in-plane normal of
  the line; `serpentine` reverses direction on odd passes. Store a per-pass
  (x0,y0,x1,y1) list beside `pass_start_times_`; the waypoint builder reads
  it instead of the shared endpoints. Fast-forward/pause logic is untouched
  (pass timing is unchanged).
- Layers stay **one box per layer** via stitch continuation (the am10
  workflow): each box's deck issues its own `laser_path` with `Z0` raised
  30 µm and the line direction rotated 90°. Moser's thermal history does not
  cross boxes; the source restarts at T0 = preheat, which is what a real
  interlayer recoat interval (seconds) gives anyway. Powder activation
  (`activate_powder_sites`: everything below the melt-pool top becomes
  powder) already handles the new layer.

No Rosenthal changes; Rosenthal rejects pauses and is not time-resolved
enough for this test.

## 3. Deck (per-layer box, kk app)

```
variable DX equal 3e-6
app_style additive/texture/kk 1000000 ${DX} 1e-6 1        # dt 1 µs like 3DThesis
seed ${SEED}
dimension 3
lattice sc/26n 1.0
region box block ${XMIN} ${XMAX} ${YMIN} ${YMAX} ${ZMIN} ${ZMAX}
boundary n n n
create_box box
create_sites box
# init box: set site range 1 1000000 ; set i2 value 1 ; set d6 value 473
# continuation boxes: set i1/i2/d1-d4/d8/d9 stitch ${STITCH} last, then the am10 reset lines
sweep color
diag_style array d6 sum d6 max
temperature 0.0

liquidus 1646
solidus  1592
solid_front_vel 4 -1.0302e-7 1.0533e-4 2.2196e-3 0   # placeholder: refit for GP700, see §5
nucleation_density      ${NUCL}                         # calibration knob, §5
critical_undercooling   5
undercooling_deviation  2
mean_nuclei_volume      ${NUCVOL}
nuclei_volume_deviation 1.5625e-17
texture_parameters 0 1.0
smooth_greedy_multiproposal on
nuclei_smooth off

misorientation_diffusion 4.2e5 2 0          # D = 0.42 deg²/µm, thresh 2°, constant (Plan 1 B1 varies this)
misorientation_target random_rotation
nucleation_misorientation_target on

setup_temperature_source moser 176 0.9 22.5 4.335e-6 473 600 ${SX} ${SY} ${SZ}
laser_path start ${X0} ${Y0} ${Z0} end ${X1} ${Y1} speed 1.137 hatch 80e-6 ${NH} serpentine pause 1e-4
fast_forward_search_window 1e-4
stats 1e-4
dump dump.id.1 stitch ${TOUT} ${STITCH} i1 i2 d1 d2 d3 d4 d8 d9
dump_modify dump.id.1 delay ${TOUT}
reset_time ${T0}
run ${T1} upto
```

Gaussian widths: the 80 µm spot is a D4σ diameter → **sx = sy = 20 µm**.
sz sets the effective depth of the surface source; start at 10 µm, sweep
{5, 10, 20} µm in §4.1 to match the Fig. 3 aspect. Even-layer boxes hatch
along x (line direction x, offsets in y); odd-layer boxes along y.

## 4. Domain and cost

- Lateral 400 × 400 µm → 5 hatches per layer at 80 µm; a 40 µm margin each
  side so the outer hatches are not at the box edge. Build: 60 µm substrate
  + 12 layers × 30 µm = 420 µm. At dx = 3 µm: 133 × 133 × 140 ≈ 2.5 M sites
  per full box; a per-layer box only needs the substrate-to-current-top
  slab (as am10 does), ~1 M sites. am10 boxes of 5 M sites ran 2–4 min each
  on the GPU (`run_kk/box_timing_log.csv`), so ~12 boxes × ~2 min.
- Sim time per layer: 5 hatches × 0.35 ms + 4 turnarounds ≈ 2 ms; the
  predictor skips the rest.
- Resolution check (Plan 1 A6): a 200 × 200 µm sub-box at 1.5 µm, 6 layers.
- Substrate: single-voxel random grains (`set i1 range`, random quaternions).
  The first three layers (90 µm) become the "build plate" and are excluded
  from analysis; the analysis floor at 90 µm plays Lamb's "dataset bottom"
  for the established-grain class. Lamb's volume sat inside a 10 mm block,
  so its established grains were likewise older layers, not the steel plate.

## 5. Calibration order (each step gated before the next)

### 5.1 Thermal only (no microstructure statistics needed; ~1 box)

Single hatch on a flat solid box, `nucleation_density 0`, `D 0`. Compare
against Met Trans Fig. 3:
- G at the fusion boundary (from `d8`): target ≈ 3e7 K/m, decreasing toward
  the pool center.
- V profile (from `d9`): flat and low over most of the pool, > 1 m/s only at
  the tail's centerline top.
- Pool aspect ≈ 2:1 from the i2 = 2 footprint at mid-pass. Knobs: sz first,
  absorptivity second (they used 0.9; hold it unless the aspect cannot be
  met).
`solid_front_vel` (V(ΔT) polynomial) is the ct15 alloy's; refit so that the
front velocity at the pool tail reaches ~1 m/s at the undercooling the Moser
field produces there — otherwise the front cannot keep up and the app's V
map cannot match theirs. This also fixes the undercooling distribution Plan 1
B1 uses to match ⟨D_eff⟩.

### 5.2 Grain structure (6-layer run, D = 0)

Columnar grains spanning > 300 µm and > 400 µm tall in Lamb; grain widths
of tens of µm; ~9000 grains in 0.27 mm³ → ~3.3e4 grains/mm³. Knob:
`nucleation_density` (GP700 grows overwhelmingly epitaxially; expect it
very low, 1e12–1e13, possibly 0 with substrate-supplied grains) and
`mean_nuclei_volume`. Gate: originating:established grain count ratio near
Lamb's 5:1 under their size filters, and grain-count density within 2×.
Texture check: <001> columnar grains along the build direction with a
<111>-ish minority (Met Trans Fig. 2).

### 5.3 SROD (12-layer run, Plan 1 variants)

Run Plan 1 B1 variants on this deck; gates are Plan 1 B2/B4. The deck is
"done" when C0 (constant D = 0.42 deg²/µm) reproduces Fig. 5a within ~0.5°
and the 1.38 ratio, with no further tuning of D — D is taken from the data,
not fitted to the simulation.

## 6. Deliverables

- `examples/gp700_moser/`: `in.gp700_init`, `in.gp700_layer` (templated as
  the am10 pair), `run_layers.sh` (box driver, adapted from am10), README with
  the table in §1.
- `am10_cpu_vs_hip/srod_analysis.py` (Plan 1 B3) run on the stitch file;
  figure overlaying the digitized Fig. 5 curves.
- Thermal validation figure: G/V/pool footprint vs Met Trans Fig. 3.

## 7. Risks and unknowns

- Absolute pool size is unverified (no scale bar in Fig. 3, none stated);
  match aspect and G/V magnitudes only. If a pool width is needed, it has to
  come from the authors or from a separate Lamb/Kirka reference.
- `solid_front_vel` for GP700 is a guess until 5.1; the ct15 polynomial
  saturates far below 1 m/s.
- Moser + `hatch`: pass-to-pass thermal overlap at 80 µm hatch and 0.35 ms
  passes is real and is the point of using Moser; the fast-forward predictor
  must not skip across hatches (`fast_forward_search_window 1e-4` and
  `pause 1e-4` keep it honest — verify with `d6 max` per stat line).
- The substrate-as-plate approximation: Lamb's lowest layers were GP700
  from earlier layers, so established-grain texture there is columnar, not
  random. Excluding the bottom 90 µm from analysis is the mitigation; if the
  established-grain class looks wrong, seed the substrate from a prior
  6-layer run's top instead.

## Progress log

**2026-09-04.** Implemented: `laser_path ... hatch <s> <n> [serpentine]`
(440f3be0, Moser + Rosenthal; CPU/kk d6 bit-identical on a 3-pass serpentine
deck), `examples/gp700_moser/{in.gp700_layer, run_layers.sh, in.gp700_thermal,
thermal_check.py, gp700_analysis.py}`. Two bugs found on the way, both fixed:
the template's placeholder comment expanded into live `set` lines before
`app_style` (segfault), and `laser_path` anchored the scan at the stale
ThermalHost time so `reset_time` was ignored and continuation boxes ran cold
(56bbcc86: both apps refresh the snapshot first; `reset_time` precedes
`laser_path` in the template).

Gate 5.1 (single hatch, `in.gp700_thermal`, 540k sites, 18 s on the GPU):
- pool 147 µm wide × 69 µm deep, **W/D = 2.1** through the pass (Fig. 3
  aspect ≈ 2:1 ✓); G median 1.0e7 K/m, 2.6–2.9e7 at the fusion boundary
  (Fig. 3: 3e7 at the boundary, decreasing inward ✓).
- the pool tail is long: liquid spans the full 400 µm box behind the laser
  and takes ~0.75 ms to solidify after the laser exits; the interface normal
  velocity is 0.4–0.7 m/s along the tail sides, > 1 m/s only at the tip —
  consistent with the paper's V map and with the isotherm geometry.
- `solid_front_vel` refit: the ct15 polynomial capped V at 0.43 m/s with every
  site solidifying at 43–53 K undercooling (the front could not keep up with
  the isotherms). Scaled 3.5× (adopted in the template) the front runs at
  **25–35 K undercooling** with V median 0.56 m/s; pool geometry unchanged
  (it is set by the thermal field, not the kinetics).
- peak T ~3e4 K at the source (no evaporation cap in the Green's function);
  harmless for the melt criterion, but do not read T > T_l values as physical.

Gate 5.2 first finding: five 147 µm-wide hatches at 80 µm spacing remelt to
**80–100 µm below the surface** (heat accumulation across the layer), so the
original 60 µm substrate melted through entirely in box 0 and the layer
re-solidified from a few surviving islands as a radial "starburst" of giant
grains (383 grains, volume-weighted median ESD 88 µm). Substrate raised to
150 µm (z_top = 50 + 10·layer voxels), boxes stay 180 µm deep.

**Later the same day.** Domain widened to 801 µm square with 10 hatches
(analyse `--crop 200 600 200 600`); inter-hatch pause set to the real 10 mm
track time (8.8 ms) so tracks solidify before the next hatch; timestep is a
driver argument. A dt 2 vs 4 µs comparison of box 0 disagreed wildly (161 vs
3443 grains, ESD 99 vs 35 µm, grains 378 µm long along x at 2 µs) and led to
the kk kernel-ordering bug (babbecf3, see Plan 1): neighbours were read live,
making the kernels Gauss-Seidel in x-index order. Box 0 is being rerun at
both steps with the Jacobi kernels; gates 5.2/5.3 restart from there.

**Timestep decision (16:20).** With the three kk kernel fixes (1a9712b3) box 0
of the wide build agrees between dt 2 and 4 µs within seed noise on the
central 400 µm crop: 2330 vs 2174 grains, ESD p50 37.5 vs 39.4 µm, extents
33/72/69 vs 36/69/72 µm, <001> within 15° of BD 0.20 vs 0.18, same-grain
z-neighbour disorientation 2.20 vs 2.27°, SROD D-fit (height) 1.22 vs 1.42×
input, (straight line) 0.99 vs 1.03×. Single-hatch check vs the CPU app at
2 µs: CPU 732 grains / ESD 25.3 µm, kk 494 / 27.5 (2 µs), 391 / 32.3 (4 µs).
The CPU app's guard dt ≤ dx/V(Tl−Ts) = 2.09 µs would reject 4 µs; the front
actually runs at 25–35 K (V ≤ 0.71 m/s, limit 4.2 µs) and the capture reach is
√3 dx (3.6 µs). The kk app now prints the same guard as a warning. Production
build `wide12_v2` launched at 4 µs on a frozen binary copy (~12 min/box).

## First full build: wide12_v2 (2026-09-04 18:52)

800 µm square, 10 hatches, 12 layers, dt 4 µs, fixed kernels (1a9712b3),
D = 0.42 deg²/µm, nucleation_density 1e12. Figures in `docs/gp700_results_v2/`;
stitch file /home/tron/gp700_runs/wide12_v2/gp700.st (3 GB). Analysis on the
central 400 µm crop with the floor at 240 µm (150 µm substrate + 3 layers).

| quantity | simulation | Lamb et al. 2023 |
|---|---|---|
| grain density (full domain) | 10 136 grains in 0.29 mm³ ≈ 3.5e4 /mm³ | > 9000 in 0.275 mm³ ≈ 3.3e4 /mm³ |
| columnar height | volume-weighted median 400 µm, 289 grains > 300 µm | > 400 µm columnar grains |
| extents x/y (µm, median, vol-weighted) | 87 / 87 | tens of µm to ~100 µm |
| <001> within 15° of BD | 0.24–0.28 (random 0.09) | <001> columnar fibre (Met Trans Fig. 2) |
| originating : established grains (Lamb filters) | ≈ 0 : 189 in the crop | 500 : 100 |
| SROD, established, mean at 0 / 265 µm | 7.6° / 18.9° | 5.6° / 11° |
| SROD band/mean (established) | 1.4–1.8 | ≈ 1.4 |
| SROD D-fit vs height (established) | 3.6 × input | 1 × (by construction) |
| same-grain z-neighbour disorientation | median 3.6°, p90 13° | KAM > 2° only at cell walls |

Reads: grain density, columnar morphology and the <001> fibre are already
right; the top-view xy section shows an 80 µm checkerboard following the
hatch grid of the two last (90°-rotated) layers, as the remelt depth is ~2
layers. Two things to change for the next build:
1. **D is ~3× too high as input.** The walk accumulates along the growth path
   and through smoothing copies, so SROD vs height runs 1.6× Lamb in angle
   (3.6× in D). Set `misorientation_diffusion` to ≈ 0.42/3 = **0.14 deg²/µm
   (1.4e5 deg²/m)** and re-measure (Plan 1's "path factor"; on this build it
   is 3.6 with smoothing, 1.2–1.4 on a single layer).
2. **Nucleation is an order of magnitude low.** The originating class is
   empty (5 k voxels); Lamb has 5 originating per established grain after
   the size filters. Raise `nucleation_density` to 1e13–1e14 (the 50-nuclei
   per layer estimate at 1e12 was right) and re-check the ratio and the
   centreline.
Cost: 12 × 12 min at dt 4 µs.

**Second build launched 19:03 (wide12_v3):** D 0.14 deg²/µm, nucleation_density
3e13, everything else as wide12_v2. If more tuning is needed after it, switch
to a formal scheme rather than hand steps: the free parameters are
`misorientation_diffusion` D, `nucleation_density`, and possibly
`critical_undercooling` / the smoothing pass count; the targets are the
Lamb SROD mean curves (originating + established, height convention), the
band/mean ratio, the originating:established grain ratio under Lamb's
filters, grain density and the <001> fraction. Each evaluation is one
12-layer build (~2.5 h) or, for a cheaper proxy, a 4-layer build on the
same deck. `spparks-micro-analysis/scripts/generate_calibration_sobol.py`
and `score_sobol_runs.py` already implement a Sobol design + scoring loop
for this kind of calibration and are the natural starting point.

**All-originating analysis of wide12_v2 (floor 100 µm, below the first melt;
189 grains, 2.2 M voxels; `docs/gp700_results_v2/srod_height_all_originating_floor100.png`).**
Up to ~100 µm the simulated SROD mean sits on Lamb's curve (2.9/3.8/5.3/6.4° at
24/45/84/105 µm vs 2.9/3.8/5.0/5.9°), i.e. the input D = 0.42 deg²/µm is
right for the walk itself. Beyond 100 µm the simulation grows *linearly*
(~0.9° per 30 µm; 15.9° at 264 µm vs Lamb 10.8°, 22° at 400 µm) where Lamb
stays √d. The linear part is a persistent drift, and it is orientation
selection: grains whose origin <001> is > 40° off the build direction rotate
toward it by 12° at 240 µm and 16° at 360 µm of height; grains 20–40° off by
5–6°; grains already within 20° drift away by +1.5/+5°. Mean <001>-to-BD
angle over the 189 grains falls from 28° at the origin to 20° at 240 µm.
Mechanism: the mobility weights c1 + c2·cos(c3·θ) select among candidate
parent VOXELS of the same grain, so among randomly rotated voxels the ones
rotated toward gradient alignment propagate — the grain rotates as a whole
(intragranular Lamarckian selection). Lamb's data show no such drift, and
Met Trans 2026 reports cells switching <100> variants rather than rotating.
Proposed model change (both apps, before any formal calibration, otherwise the
optimizer fits D to a drift of the wrong functional form): apply the
mobility weight per candidate GRAIN (spin), e.g. the mean mobility of that
spin's candidate voxels, choose the spin by roulette, then the parent voxel
uniformly within that spin. Intergranular competition (the <001> fibre)
survives; intragranular selection disappears. Second consequence: the D
correction to 0.14 in wide12_v3 will likely undershoot the first 100 µm.

**Per-grain parent selection implemented (ad23c1a5, both apps):** candidates
grouped by spin, each grain weighted by the mobility of the sign-aligned mean
orientation of its candidate voxels against the gradient at the solidifying
site, grain drawn by roulette, parent voxel uniform within it. Single-hatch
statistics unchanged (CPU 748 vs 732 grains, kk 500 vs 494); d6 identity
intact. Template D restored to 0.42 deg²/µm. `wide12_v4` (per-grain
selection, D 0.42, nucleation 3e13) is chained to start when v3 finishes; its
tilt-vs-height table should come out flat and the all-originating SROD should
follow Lamb beyond 100 µm.

**v3 and v4 (night of 2026-09-04).** v3 (D 0.14, nucl 3e13, per-voxel
selection): SROD undershoots Lamb by 2.1° (RMS 2.16°), 5180 grains on the crop
(2× v2) but nucleated grains do not pass the 60 µm / 40 µm filters — the
originating class stays empty. v4 (per-grain selection, D 0.42, nucl 3e13):
drift gone, but SROD accumulates only **0.33× the input D** against height
(RMS 2.9° below Lamb), band/mean 1.8, and the <001> fraction falls to 0.11
(random 0.09) — the fibre in v2/v3 was intragranular rotation, and
grain-vs-grain competition at c2 = 1 does not build it. The walk is damped by
the nrefine 2 smoothing pass (quaternion averaging over neighbours), which
the drift used to mask. Overnight: `sobol1` (16 points, D 0.2–0.8, nucl
1e13–1e14, tc 3–25 K) started by the chain at 23:54, and `sobol2` (8 points,
D 0.5–2.0, same nucl/tc, **texture c2 1–4** as a 4th parameter) started at
23:56 alongside it; both on the 600 µm / 7-hatch / 8-layer proxy, no new
points after 08:30. Results: /home/tron/gp700_runs/sobol{1,2}/results.jsonl.

## Sobol studies (night of 2026-09-04/05)

Proxy geometry 600 µm / 7 hatches / 8 layers / dt 4 µs, all-originating SROD
(floor 100 µm) on the central 300 µm crop, scored as RMS of (sim − Lamb Fig. 5a
mean) over 25–240 µm. 16 scored points (study 1: D 0.2–0.8, c2 = 1; study 2:
D 0.5–2.0 and texture c2 1–4; both nucl 1e13–1e14, tc 3–25 K). Results:
/home/tron/gp700_runs/sobol{1,2}/results.jsonl. Linear fits over all points:

- **bias (°) = 10.4 + 2.31·ln D − 0.352·ln(nucl) − 0.003·tc + 0.002·c2, R² 0.97**
  → D and nucleation density are the only levers on the SROD level; zero bias
  at D ≈ 1.13 deg²/µm for nucl 1.5e13 (1.25 at 3e13). The walk accumulates
  ~0.35× its input against height once per-grain selection and the nrefine-2
  smoothing are in.
- **<001> within 15° of BD = 0.72 − 0.025·ln D − 0.020·ln(nucl) + 0.034·c2**
  (R² 0.29, noisy): c2 and low nucleation build the fibre; tc does nothing.
- **band/mean = 1.73–1.85 at every point** (Lamb ≈ 1.4): structural, not
  reachable with these parameters — the smoothing pass (quaternion averaging)
  and/or the per-grain uniform parent draw widen the distribution; next lever
  is nrefine / the smoothing rule, not D.
- Best points: D 1.25 / nucl 1.3e13 / tc 22 / c2 2.6 → RMS 0.52°, bias +0.3°,
  <001> 0.20; D 1.63 / 8.5e13 / 11 / 2.1 → 0.83°, +0.3°, 0.22.

**v5 launched 06:55** (full 12-layer wide build, binary spparks_texguard):
D 1.15 deg²/µm, nucleation 1.5e13, tc 15 K, c2 2.8, dt 4 µs. Expected: bias
≈ 0 ± 0.5°, <001> ≈ 0.2–0.25, band ratio ≈ 1.8 (open issue).

**Smoothing finding (2026-09-05 morning).** All builds and Sobol points ran
`smooth_greedy_multiproposal on` with nrefine 2 (the ct15 templates pair
greedy with nrefine 1). Tests at v5's parameters on the proxy and on the walk
deck: with greedy OFF the structure is the raw solidification structure —
331 k grains, ESD 6.8 µm on the proxy; 2524 (CPU) / 2303 (kk) grains on the
walk deck vs 2375 / 2908 with no smoothing at all — while greedy ON coarsens
it 4–5× (626 / 443 on the walk deck; ~2000 columnar grains on the proxy).
Because the substrate is seeded with single-voxel random grains and the
mobility-weighted voxel capture thins them only slowly, the greedy pass is
what produces the columnar morphology; two passes double the effect and make
the result timestep-sensitive (the pass runs one step after capture, at the
front, mostly against solid neighbours behind the voxel). Consequences: (1)
use greedy on with **nrefine 1** (test queued: smooth_greedyon_nr1); (2) the
band/mean excess (1.8 vs 1.4) and the walk damping (~0.35×) come from these
passes averaging neighbour quaternions — expect both to move with nrefine;
(3) a physically seeded substrate (coarse grains, e.g. a short Potts anneal
or Voronoi seeding at ~50 µm) would let growth competition, not smoothing,
carry the columnar selection — worth doing before the next calibration.

**v5 result (10:26, `docs/gp700_results_v5/`):** D 1.15 deg²/µm, nucl 1.5e13,
tc 15 K, c2 2.8, greedy on / nrefine 2, dt 4 µs. All-originating SROD vs
Lamb Fig. 5a: **RMS 0.59°, bias −0.16°** over 25–240 µm; D-fit vs height
0.87× input; band/mean 1.79 (Lamb 1.4, open); <001> within 15° of BD **0.34**;
3955 grains on the crop, 164 pass Lamb's filters, 51 taller than 300 µm,
ESD p50 138 µm. The SROD level and the fibre are now at Lamb's values with
the smoothing configuration that produced them (nrefine 2 greedy); the
nrefine-1 test will say whether the same parameters hold with one pass.

**Smoothing / substrate tests at v5's parameters (proxy, 2026-09-05):**

| configuration | grains (crop) | filtered | SROD bias | D-fit/input | band/mean | <001> |
|---|---|---|---|---|---|---|
| greedy on, nrefine 2, single-voxel substrate (= v5) | 2088 | 92 | −0.42° | ≈0.35 | 1.75 | 0.29 |
| greedy off, nrefine 2 | 331 k (ESD 6.8 µm) | 0 | — | — | — | 0.26 |
| no smoothing | 507 k (ESD 4.7 µm) | 0 | — | — | — | 0.28 |
| greedy on, nrefine 1, single-voxel substrate | 2730 | 102 | +0.02° | ≈0.40 | 1.76 | 0.24 |
| greedy on, nrefine 1, **50 µm Voronoi substrate** | 1042 | 57 | +1.34° | 1.22 | 1.84 | 0.21 |

With single-voxel seeds the raw structure never coarsens; the greedy pass
made the columnar grains, and the pass count (1 vs 2) barely changes the walk
damping (0.40 vs 0.35×) — the damping comes from building grains out of a
voxel-scale mixture by majority votes, not from the number of passes. With a 50 µm-grained substrate and one pass, growth
carries the substrate grains up as 50–90 µm columns through the full height,
the walk keeps ~1.2× its input (as in the single-layer tests) so the input D
returns to ≈ 0.35–0.4 deg²/µm, i.e. Lamb's own fit. The band/mean stays ≈ 1.8
in every configuration that has grains — it is not a smoothing artefact but
heterogeneity of the effective D between voxels/grains (path-length
differences), a genuine model-vs-experiment difference to keep in view.

**v6 launched 11:38:** 800 µm build, substrate_800um_50um.st (1470 grains),
greedy on / nrefine 1, D 0.40, nucl 1.5e13, tc 15, c2 2.8, dt 4 µs.

**v6 result (14:09, coarse Voronoi substrate, nrefine 1, D 0.40):** misses low.
All-originating SROD vs Lamb Fig. 5a: RMS 2.18°, bias −1.76° over 25–240 µm;
D-fit 1.64e5 deg²/m = **0.41× input**; band/mean 1.85; <001> within 15° of
BD 0.16 (v5: 0.34); 2572 grains on the crop, 95 pass Lamb's filters, 47 taller
than 300 µm, ESD p50 148 µm, lateral extents 117 × 105 µm (v5: 96 × 96).
Established class (floor 240): D-fit 3.9e5 = 0.93×, but over only 165 µm of
range.

**Correction to the table above.** `gp700_analysis.py` divided every D-fit
by a default 4.2e5 instead of the deck's D (no chain or Sobol point passed
`--D`), so the "D-fit/input" column was wrong for every run with D ≠ 0.42.
Corrected ratios (D-fit ÷ the actual input):

| configuration | input D | D-fit (deg²/µm) | ratio |
|---|---|---|---|
| v5 full build, greedy on nrefine 2, single-voxel substrate | 1.15 | 0.367 | 0.32 |
| proxy, greedy on nrefine 2, single-voxel (= v5 config) | 1.15 | 0.147 | 0.13 |
| proxy, greedy on nrefine 1, single-voxel | 1.15 | 0.168 | 0.15 |
| proxy, greedy on nrefine 1, 50 µm Voronoi substrate | 1.15 | 0.511 | 0.44 |
| v6 full build, greedy on nrefine 1, 50 µm Voronoi substrate | 0.40 | 0.164 | 0.41 |

So the coarse substrate does **not** restore the walk to its input D (the
"1.22×" was the artefact); it roughly triples the retained fraction relative
to single-voxel seeding on the proxy (0.15 → 0.44), and the full build keeps
≈0.4× of the input in both v5-like and v6-like configurations. The
single-hatch walk deck (1.2×, `scripts/srod_analysis.py`, correct D) remains
the only configuration without damping, so the loss is a multi-layer /
remelt / smoothing effect, still unexplained. The fix is committed
(26732622); the chain scripts now pass `--D`.

**v7 launched 20:31** (two builds sharing the GPU, ~1.4× slower each):
D = 0.80 deg²/µm from the measured bias slope (v6 −1.76° at 0.40, the
Voronoi proxy +1.34° at 1.15 → zero bias at ≈0.73–0.86), everything else as
v6 (nrefine 1, nucl 1.5e13, tc 15, c2 2.8, dt 4 µs).
- `wide12_v7a`: 50 µm Voronoi substrate (as v6).
- `wide12_v7b`: substrate = the top 150 µm of v6 (`substrate_from_top.py`,
  1235 competed columnar grains) — a stand-in for Lamb's sample being far
  above the baseplate; tests whether the <001> fibre (0.16 on the random
  substrate vs 0.34 in v5 and Lamb) is a substrate-texture effect.
Chain: `/home/tron/gp700_runs/chain_v7.{sh,log}`; analysis with `--D 8e5`.

**v7 results (01:10, both `--D 8e5`):**

| | v7a Voronoi substrate | v7b substrate = top of v6 | Lamb |
|---|---|---|---|
| SROD RMS vs Fig. 5a, 25–240 µm | **0.90°** | 3.34° | 0 |
| mean bias | **+0.24°** | +3.08° | 0 |
| D-fit ÷ input (floor 100, all originating) | 0.42 | 0.75 | 1 |
| band/mean | 1.85 | 1.73 | 1.4 |
| <001> within 15° of BD | 0.20 | 0.16 | ≈0.3 (v5: 0.34) |
| grains on crop / Lamb filters / >300 µm tall | 2515 / 108 / 46 | 2072 / 37 / 32 | – |
| ESD p10/p50/p90 (µm) | 70 / 150 / 169 | 146 / 171 / 196 | – |

v7a is calibrated on the SROD level (D 0.80 with ~0.4× retention ≈ Lamb's
0.42), but the *shape* still differs: RMS² vs height is 15 → 34 → 56 → 84 →
108 → 120 deg² at 6 / 54 / 105 / 204 / 305 / 354 µm, i.e. a 3.3° offset at
zero height (lateral lineage spread inside 100 µm-wide grains) and a local
slope of 0.24–0.33 deg²/µm = 0.3–0.4× the input that falls with height,
whereas Lamb keeps rising as √h. The curve crosses Lamb near 150 µm — above
it and below it are both off by ~1°. The fibre did not come back with the
competed substrate (0.16), so the 0.34 of v5 was the nrefine-2 greedy
coarsening, not substrate texture.

v7b's +3° bias is a reference artefact: the substrate grains carry v6's
intragranular spread (same-grain z-neighbour disorientation median 4.0°,
p90 13.6°), so the regrown grain differs from its single lowest reference
voxel by 7.5° already at 6 µm; its established-class fit from floor 240 is
1.29× input. Not a usable substrate for SROD unless the substrate
quaternions are reset per grain (mean orientation) first.

**Where the ~0.4× retention comes from (hypothesis, from the kernel):** the
smoothing pass, on every accepted spin flip, replaces the voxel's quaternion
with the *sign-aligned average* of its neighbours in the new grain
(`get_average_neighbor_quaternion` on the CPU, the kk SmoothingKernel). The
raw front is a voxel-scale mixture of grains (per-grain selection interleaves
them), so most captured voxels are reassigned at least once and each
reassignment averages out the fresh per-hop rotations of ~n siblings
(variance ÷ n). The single-hatch walk deck has one grain, no flips and no
damping (1.2×); every competing-grain build damps to 0.3–0.45× regardless of
D, substrate or pass count. The band/mean excess may come from the same
averaging (heterogeneous damping). Test: a `smooth_orientation copy` mode
(take ONE random same-spin neighbour's quaternion instead of the average)
should restore ≈1× retention at D ≈ 0.42.

**Smoothing-average damping CONFIRMED (proxy, 02:05, D 0.42, Voronoi
substrate, greedy on / nrefine 1, binary spparks_smoothcopy = da2fad29):**

| smooth_orientation | retention (height / straight line) | band/mean | bias vs Lamb | <001> | same-grain z-nbr disorientation median |
|---|---|---|---|---|---|
| average (default) | 0.46 / 0.40 | 1.8 | −1.53° | 0.23 | 2.3° |
| **copy** | **1.85 / 1.62** | **1.34–1.44** | +2.67° | 0.17 | 5.0° |

Copy mode restores the walk (retention > 1 because the growth path is
longer than the height, as on the walk deck) AND brings band/mean to Lamb's
1.4 — the long-standing "structural" discrepancy was the quaternion
averaging. Scaling the copy-mode curve onto Lamb Fig. 5a over 25–240 µm
gives D = 0.216 deg²/µm (RMS 0.41° on the proxy shape), i.e. about half of
Lamb's own √h fit, consistent with the 1.6–1.85× path factor.

**v8 launched 02:06:** full 800 µm / 12-layer build, `smooth_orientation
copy`, D 0.22, Voronoi 50 µm substrate, nrefine 1, nucl 1.5e13, tc 15,
c2 2.8, dt 4 µs; chain `chain_v8.sh`, analysis `--D 2.2e5`. Open after
this: the <001> fibre (0.17–0.23 vs Lamb ≈ 0.3) — c2 / nucleation are the
levers per the Sobol surface.

**v8 result (04:30, `docs/gp700_results_v8/`):** `smooth_orientation copy`,
D 0.22 deg²/µm, Voronoi 50 µm substrate, greedy on / nrefine 1, nucl 1.5e13,
tc 15, c2 2.8, dt 4 µs. All-originating SROD vs Lamb Fig. 5a: **RMS 0.49°,
bias +0.17°, band/mean 1.41 (Lamb 1.4)**; the curve follows Lamb's √h shape
over the whole 25–260 µm range (mean 4.4° at 54 µm, 8.5° at 204 µm, 12.3° at
402 µm; retention 1.88× vs height, 1.80× straight-line). 2496 grains on the
crop, 86 pass Lamb's filters, 41 taller than 300 µm, ESD p50 151 µm,
extents 114 × 99 µm, same-grain z-neighbour disorientation median 4.2°.
Remaining gap: <001> within 15° of BD **0.19** vs Lamb ≈ 0.3.

**Fibre calibration launched 04:40** (two Sobol studies in parallel on the
proxy, copy mode, D 0.22 and tc 15 fixed): texture c2 2–8, nucleation
5e12–5e13; results `/home/tron/gp700_runs/sobol_fibre{1,2}/results.jsonl`.

**Fibre Sobol result (07:57, proxy, copy mode, D 0.22, tc 15):**

| c2 | nucleation | <001> within 15° | SROD RMS | band/mean |
|---|---|---|---|---|
| 2.65 | 6.5e12 | 0.22 | 0.43° | 1.45 |
| 3.45 | 1.9e13 | 0.22 | 0.42° | 1.45 |
| 3.75 | 1.2e13 | 0.22 | 0.42° | 1.43 |
| 4.85 | 3.9e13 | 0.23 | 0.48° | 1.45 |
| 5.60 | 7.1e12 | 0.22 | 0.47° | 1.42 |
| 5.76 | 3.7e13 | 0.27 | 0.63° | 1.45 |
| 7.22 | 1.1e13 | 0.28 | 0.52° | 1.44 |
| 7.66 | 2.1e13 | 0.28 | 0.59° | 1.45 |

Fit: **<001> = −0.042 + 0.0125·c2 + 0.0073·ln(nucl)** (R² 0.72); SROD RMS
= −0.91 + 0.031·c2 + 0.041·ln(nucl). The fibre is flat at 0.22 up to c2 ≈ 5
and then rises ~0.0125 per unit c2; nucleation barely matters; the SROD
score stays under 0.65° everywhere and the band ratio stays at Lamb's 1.4.
The full build sits ~0.03 below the proxy at equal settings (v8 0.19 vs
proxy 0.215).

**v9 launched 08:00:** v8 settings + `texture_parameters 9 2.5`,
`nucleation_density 2e13` (predicted proxy fibre 0.29, SROD RMS ≈ 0.6°);
`chain_v9.sh`, run `wide12_v9`. Note c2 9 zeroes the mobility of grains
more than ~40° off the gradient (mobility 1 + c2·cos(2.5θ), clamped at 0).

**v9 result (10:22, `docs/gp700_results_v9/`):** v8 + c2 9, nucleation 2e13.
SROD vs Lamb: RMS 0.52°, bias +0.02°, band/mean 1.45 — unchanged from v8, as
the Sobol predicted. <001> within 15° of BD **0.22** (v8 0.19; proxy
prediction 0.29). 3068 grains on the crop, 97 pass Lamb's filters, ESD p50
147 µm. Fibre vs height (central crop, 45 µm slabs): v8 saturates at
0.19–0.20 from 150 µm up; v9 climbs 0.16 → 0.24 over the 400 µm and is
still rising slowly at the top. So c2 9 keeps selecting with height but the
walk (D_eff ≈ 0.4 deg²/µm → 13° RMS spread over 400 µm) erodes the 15° cone
about as fast as selection fills it; the full-build ceiling with these
mechanics looks like ≈0.25.
