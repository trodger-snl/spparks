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

**Correction on the fibre target.** Neither Lamb paper quantifies the <001>
texture; the "Lamb ≈ 0.3" used above was carried over from v5's 0.34 and has
no experimental basis. The only reference is qualitative (Met Trans 2026
Fig. 2: <001> columnar fibre along BD). v8/v9 give 0.19–0.24 within 15° of
BD, 2–2.5× random, i.e. a clear but moderate fibre; pushing it further
needs a measured target first (e.g. Lamb's IPF/pole-figure MUD, if
obtainable) rather than more c2. **Template reset to the v8 settings**
(c2 2.8, nucleation 1.5e13), which is the calibrated deck; v9 (c2 9, 2e13)
is recorded as the stronger-fibre variant with identical SROD.

## Status (2026-09-06 10:40)

Calibrated GP700 deck = `examples/gp700_moser/in.gp700_layer` as committed:
Moser source with the GP700 constants, 3.5× front law, dt 4 µs, 50 µm
Voronoi substrate, `smooth_greedy_multiproposal on` / nrefine 1,
`smooth_orientation copy`, `misorientation_diffusion 2.2e5 30`, per-grain
parent selection, nucleation 1.5e13 / tc 15 K, texture 2.8 / 2.5. Against
Lamb 2023 Fig. 5a (all-originating SROD, height): RMS 0.49°, bias +0.17°,
band/mean 1.41 vs 1.4; grain density and columnar morphology as before.
Binary: `/home/tron/gp700_runs/spparks_smoothcopy` (= da2fad29 kernels).

## Resolution test: dx 1.5 µm (Lamb's voxel size) vs 3 µm — 2026-09-07

Same deck (v8: copy mode, D 0.22, c2 2.8, nucl 1.5e13, tc 15), same proxy
geometry (600 µm / 7 hatches / 8 layers, 50 µm Voronoi substrate, seed
12345), dt 2 µs at 1.5 µm vs 4 µs at 3 µm (both 2× the dx/V(Tl−Ts) guard).
`run_layers.sh` now takes `DX=` from the environment (a3a66f41). Runs:
`/home/tron/gp700_runs/dx1p5_v8` (47 min/box, 6.2 h) and `dx3_v8_ref`
(6 min/box). Figures: `docs/gp700_results_dx1p5/`.

| | dx 3 µm | dx 1.5 µm | Lamb |
|---|---|---|---|
| SROD RMS vs Fig. 5a, 25–240 µm | 0.44° | 0.49° | 0 |
| mean bias | +0.03° | +0.41° | 0 |
| band/mean | 1.41 | 1.43 | 1.4 |
| retention, height / straight line | 1.83 / 1.73 | 2.17 / 2.35 | – |
| <001> within 15° of BD | 0.21 | 0.20 | – |
| ESD p10/p50/p90 (µm) | 68 / 123 / 159 | 66 / 122 / 150 | – |
| grains passing Lamb's filters | 51 | 62 | – |
| same-grain z-neighbour disorientation median | 3.8° | 3.7° | – |

Bin-by-bin the mean SROD agrees to 0.1–0.4° up to 200 µm (3.5/3.6° at
35 µm, 5.7/5.9° at 95 µm, 7.8/8.4° at 185 µm) and the finer lattice runs
7–10 % higher by 245–275 µm (9.3/10.0°, 9.9/10.9°); the D-fit against height
is 18 % higher. Grain size, texture, band ratio and the intragranular
neighbour disorientation are unchanged. Conclusion: the per-hop
formulation θ = √(D·hop) is resolution-independent to first order — the
same deck D reproduces Lamb at both voxel sizes — with a second-order
increase of accumulation per unit height at finer dx (the finer lattice
resolves a slightly longer growth path per height). An exact match at
1.5 µm would take D ≈ 0.19 deg²/µm. The 1.5 µm run resolves Lamb's first
bins directly: 1.8° mean at 5.8 µm vs Lamb's ~1.5°.

## Crop sensitivity and grain-origin audit (v8, 2026-09-07)

`gp700_analysis.py --top-um` added (ceiling crop). v8 through four windows,
floor 100 µm, D 2.2e5:

| window | grains (filters) | SROD RMS / bias | band | <001> | ESD p50 |
|---|---|---|---|---|---|
| full 801 µm domain | 9299 (368) | 0.43° / +0.12° | 1.42 | 0.17 | 147 |
| central 400 µm (standard) | 2496 (86) | 0.49° / +0.17° | 1.41 | 0.19 | 151 |
| central 200 µm | 616 (24) | 0.58° / 0.00° | 1.39 | 0.06 | 140 |
| central 400 µm, z < 420 µm (drops the never-remelted top layer) | 1818 (84) | 0.49° / +0.16° | 1.41 | 0.19 | 134 |

SROD level, band and D-fit are insensitive to the crop (bias within ±0.2°,
D-fit 1.83–1.95× everywhere); the fibre fraction is the only quantity that
moves, and the 200 µm crop (24 grains) is too small to measure it.

**Origin audit.** With the floor at 100 µm the "established" panel is empty
because the analysis excludes the unremelted substrate (i2 = 1) and the
solid starts at 105 µm — no grain crosses the floor, so every grain is
classed originating, with its origin at its lowest solid voxel. Those
origins are almost all the layer-0 remelt floor: of the 88 filtered grains
in the standard crop, 83 originate at 105–130 µm (99 % of the filtered
volume) and only 3 above 160 µm. I.e. the columnar grains are epitaxial
continuations of substrate grains, and the "all-originating" SROD is in
Lamb's terms an **established**-grain SROD referenced at the remelt floor.
Lamb had > 500 originating vs > 100 established grains (their originating
grains include lateral overgrowth and nucleation inside the volume); the
simulation produces essentially none that survive the 60 µm / 40 µm
filters at nucleation 1.5e13. Since Lamb's Fig. 5a and 5c follow the same
√h law with the same D, the walk calibration stands, but the grain-class
comparison does not: the model has no mechanism generating new large
grains mid-build at these settings.

## Grain-level (coherent) walk — 2026-09-08

Lamb Fig. 4d shows each grain tracing a directed path on the build-direction
IPF (their red arrow) and G4 splitting into two clusters; the per-voxel walk
gives an isotropic cloud whose mean does not move (v8, four tallest grains:
drift 3–6° vs 12° spread). `misorientation_grain_walk D_g` (b6e4ba5f,
`src/grain_walk.h`): every voxel of grain g crossing lattice level k → k+1
gets the same rotation, angle √(D_g·dx) about an axis hashed from (g, k) —
path independent, no per-grain state; per-voxel walk on top. Three proxy
runs (600 µm / 8 layers / 3 µm, v8 deck otherwise), analysed with
`grain_paths.py` (`docs/gp700_results_grainwalk/`):

| | voxel only (D_v 0.22) | mixed (D_g 0.30 + D_v 0.065) | grain only (D_g 0.42) |
|---|---|---|---|
| SROD RMS / bias vs Lamb | 0.44° / +0.03° | 0.54° / +0.34° | 0.47° / +0.04° |
| band/mean (population) | 1.41 | 1.46 | 1.48 |
| D-fit vs height (deg²/µm) | 0.40 | 0.43 | 0.38 |
| <001> within 15° | 0.21 | 0.22 | 0.18 |
| same-grain z-neighbour disorientation, median / p90 | 3.8° / 12.5° | 2.3° / 6.7° | 1.1° / 1.7° |
| 4 tallest grains: drift of the mean orientation, bottom → top | 0.7–6.3° | 6.5–8.8° | 5.3–11.3° |
| 4 tallest grains: spread about the top-slab mean (RMS) | 7.6–11.7° | 4.2–7.1° | 1.4–2.0° |
| 4 tallest grains: max SROD | 20–28° | 16–19° | 8–12° |

All three sit on Lamb's population curve with the same band ratio: the
population SROD statistics cannot distinguish coherent from per-voxel
accumulation (each grain's coherent walk is one realisation; the Maxwell
width appears across grains either way). The per-grain analysis does: the
coherent walk produces the directed IPF path and a small within-slab
spread; the ratio D_g : D_v sets drift against spread. Two things the
grain-only run shows that Lamb's G1–G4 do not: (1) a memoryless walk makes
individual SROD(h) curves non-monotonic (grain 703: 8° at 150 µm, back to
2° at 240 µm) whereas Lamb's grains rise steadily to 20–25°, which points to
persistent (correlated-axis) rotation, i.e. lattice bending, rather than a
memoryless walk; (2) the per-level increment (1.1° at dx 3 µm) shows as
horizontal bands in the IPF map. Lamb's within-slice widths are not
published numerically, so the D_g : D_v split is unconstrained beyond
"drift ≳ spread"; the mixed run (0.30 : 0.065) is the working choice.

## Persistent (correlated-axis) grain walk — 2026-09-08 afternoon

`misorientation_grain_walk D_g [rho]` (f38c2839, 47546146): with rho > 0 the
per-level rotation vector is an AR(1)/Ornstein–Uhlenbeck sequence in the
level index (correlation height ≈ dx/(1−rho)), hashed and evaluated from
level 0 so it stays a pure function of (g, z). First form (fixed angle,
persistent axis only) made every grain rotate at the same rate and
collapsed the population band (0.77 at rho 0.9, 0.35 at rho 0.97); replaced
by a Gaussian rotation vector (Maxwell magnitudes). D_g was scaled by
(1−rho)/(1+rho) for a target long-range D of 0.42. Proxy, v8 deck otherwise,
`grain_paths` on the four tallest grains (`docs/gp700_results_grainwalk/*_ou.png`):

| | grain only, rho 0.9 (D_g 0.022) | mixed, rho 0.9 (D_g 0.016 + D_v 0.065) | grain only, rho 0.97 (D_g 0.0064) |
|---|---|---|---|
| SROD RMS / bias vs Lamb | 1.15° / −0.94° | 0.53° / −0.40° | 1.95° / −1.90° |
| band/mean (population) | 1.52 | 1.39 | 1.49 |
| D-fit vs height (deg²/µm) | 0.29 | 0.36 | 0.24 |
| mean SROD at 6 / 45 / 124 / 244 µm | 0.4 / 2.7 / 5.9 / 7.5° | 1.1 / 3.1 / 6.2 / 8.9° | 0.2 / 1.7 / 4.7 / 7.2° |
| same-grain z-neighbour disorientation, median | 0.24° | 2.0° | 0.14° |
| per-grain drift bottom → top / spread about the top-slab mean | 3.5–9.9° / 0.5–1.0° | 3.4–7.0° / 4.6–5.9° | 3.3–8.7° / 0.1–0.4° |
| per-grain max SROD ÷ top-slab mean | ≈1.0–1.1 (monotonic) | ≈2 | ≈1.0 (monotonic) |

Persistence does what it should for individual grains — monotonic SROD
rise, a directed IPF path, the population band back at Lamb's 1.4–1.5 — and
the population curve becomes ballistic below the correlation height (mean
SROD ∝ h up to ~45 µm at rho 0.9, ~100 µm at rho 0.97, then √h), which is
the "fast first layer" shape Lamb describe. The level came out low: the
measured long-range amplification is ≈13× at rho 0.9 (theory 19×, 17× with
the finite-height correction), so D_g rescaled ×1.3 (rho 0.9), ×1.8
(rho 0.97), ×1.13 (mixed) for a second round (`gwo2_*`).

**Second OU round (17:20, D_g rescaled):**

| | grain only, rho 0.9 (D_g 0.029) | mixed, rho 0.9 (D_g 0.018 + D_v 0.065) | grain only, rho 0.97 (D_g 0.0115) |
|---|---|---|---|
| SROD RMS / bias vs Lamb | 0.77° / +0.64° | **0.42° / +0.11°** | 0.59° / −0.50° |
| band/mean | 1.35 | 1.44 | 1.66 |
| D-fit vs height (deg²/µm) | 0.51 | 0.40 | 0.40 |
| mean SROD at 6 / 45 / 124 / 244 µm | 0.5 / 3.2 / 7.3 / 10.9° | 1.1 / 3.3 / 6.8 / 9.3° | 0.3 / 2.4 / 6.1 / 9.4° |
| per-grain drift / spread about the top-slab mean | 4.0–11.3° / 0.5–1.1° | 1.9–8.6° / 5.0–6.3° | 3.0–14.1° / 0.5–1.0° |
| per-grain max SROD ÷ top-slab mean | 1.0–1.3 | ≈2 | 1.0–1.2 |

Zero-bias interpolation: grain-only rho 0.9 at D_g ≈ 0.026, rho 0.97 at
≈ 0.013 (empirical amplification ≈ 15× and ≈ 32× against the AR(1)
(1+rho)/(1−rho) of 19× and 66×). The mixed rho 0.9 run is on Lamb
(0.42°, +0.11°, band 1.44) with directed per-grain paths and a 5–6°
intragranular spread; it is the working candidate for the next full build,
pending a measured within-slice width from the experiment to fix D_g : D_v.
Lamb's population curve below 30 µm (the ballistic regime, mean ∝ h) is
the observable that would fix rho; the current digitization does not
resolve it.

## Path-indexed grain walk (d10) — 2026-09-08 evening

`misorientation_grain_walk D [rho] [path|height]` (60b2d711, 82afc6ba): new
per-site double **d10** = inheritance path length in lattice units (0 at
melt and at nuclei, parent + hop at capture, copied with the orientation on
a smoothing flip, exchanged with ghosts, dumped and continued through the
stitch; substrates regenerated with the field, grains identical). In `path`
mode (default) the coherent increment index is floor(d10): one increment
per unit of path in any growth direction, no build-direction assumption.
`height` keeps the z-level indexing. Proxy runs, v8 deck otherwise, D_g
÷ 1.8 from the height-mode values (`*_path.png`):

| | grain only, rho 0 (D_g 0.23) | grain only, rho 0.9 (D_g 0.0145) | mixed, rho 0.9 (D_g 0.010 + D_v 0.065) |
|---|---|---|---|
| SROD RMS / bias vs Lamb | 0.51° / +0.28° | 0.58° / +0.40° | **0.38° / +0.28°** |
| band/mean | 1.49 | 1.63 | 1.49 |
| D-fit vs height (deg²/µm) | 0.45 | 0.51 | 0.50 |
| mean SROD at 6 / 45 / 124 / 244 µm | 1.2 / 3.7 / 6.8 / 10.0° | 0.6 / 3.0 / 7.1 / 10.5° | 1.2 / 3.3 / 6.8 / 10.4° |
| same-grain z-neighbour disorientation, median / p90 | 1.3° / 3.7° | 0.6° / 3.2° | 2.2° / 7.2° |
| per-grain drift / spread about the top-slab mean | 4–18° / 2.1–3.8° | 7–12.5° / 1.8–4.1° | 5–10.5° / 4.3–6.4° |

Path mode lands on Lamb with the 1/1.8 rescaling, so the coherent walk is
now parameterised per unit of growth path like the per-voxel walk (deck D_g
is per path length; the height factor is the same 1.8). The persistence
amplification comes out at the AR(1) value in path mode (D-fit/D_g = 35 ≈
19 × 1.8), where height mode gave 13–15 instead of 19. Path dispersion
inside a grain now produces an intragranular spread by itself (2–4° in the
grain-only cases vs 0.5–1° in height mode), so the per-voxel term is less
necessary; the mixed rho 0.9 path run is the best score so far (0.38°,
band 1.49, drift 5–10°, spread 4–6°). Cost: the persistent recurrence is
evaluated from level 0 per capture (≤ ~250 path levels), which made the
rho > 0 path runs ~1.4× slower than memoryless; a per-grain cache would
remove that if it matters for full builds.

**Recurrence cache (6a47de77) and v10 (2026-09-08/09).** Device table of the
OU rotation vector per (compact grain id, level), 16384 × 768 doubles
(302 MB), filled lazily in the capture/smoothing kernels; overflow falls
back to the direct recurrence; CPU app keeps the direct form. Cached vs
direct at the mixed rho 0.9 path settings on the proxy: RMS 0.49° /
bias +0.36° / band 1.46 vs 0.38° / +0.28° / 1.49, D-fit 0.46 vs 0.50,
per-grain drift 3–10.5° vs 5–10.5°, spread 5–6° vs 4–6° — seed-level
agreement. Timing under sharing was inconclusive (direct persistent ran
1.5× the memoryless proxy under equal 3-way sharing; the cached run shared
with the full build); a solo cached-vs-direct pair is queued after v10
(`chain_timing.sh`). **v10** (`wide12_v10`, launched 23:36): v8 deck +
`misorientation_grain_walk 1.0e4 0.9 path` + `misorientation_diffusion
6.5e4`, substrate_800um_50um_d10.st, cached binary.

**v10 result (02:24, `docs/gp700_results_v10/`):** persistent path-indexed
grain walk (D_g 0.010, rho 0.9) + per-voxel D_v 0.065, Voronoi substrate,
nrefine 1, nucl 1.5e13, tc 15, c2 2.8, dt 4 µs, cached recurrence. Full
12-layer build, central 400 µm: **SROD RMS 0.34°, bias −0.28°, band/mean
1.50** (best full-build score; v8 0.49° / +0.17° / 1.41); mean SROD 3.6 /
8.2 / 12.1° at 54 / 204 / 402 µm; D-fit vs height 0.43 (established class
from floor 240: 0.56); 2504 grains, 97 pass Lamb's filters, ESD p50 148 µm,
extents 102 × 105 µm; same-grain z-neighbour disorientation median 2.4°,
p90 8.8°. Four tallest grains: drift 8.4–9.6°, spread about the top-slab
mean 6.2–7.7°, max SROD 20–24° (v8: drift 3–6°, spread 12°, max 26–33°).
<001> within 15° of BD 0.15 (v8 0.19): the coherent rotation carries
grains out of the cone. Solo box time 13 min (v8 12 min) with the cache.

**Cache timing (solo, proxy, mixed rho 0.9 path):** loop time per box —
cached 189–190 s flat across the 8 boxes, equal to the per-voxel-only
reference (dx3_v8_ref, 189 s); direct recurrence 197 → 222 s, rising with
the build as the recurrence length grows with path level. v10's solo boxes
ran at 698 s, the same as v8's 697 s. With the cache the persistent walk
costs nothing measurable; the earlier "13 vs 12 min" was wall time under
GPU sharing.

**All-grain paths (2026-09-09, `grain_paths.py --grid`):** per-grain
drift / spread / max SROD / monotonicity for every filtered grain
(`grain_stats.csv`), a 30-grain IPF-path grid and a summary figure in
`docs/gp700_results_v{8,10}/`.

| | v10 (96 grains) | v8 (86 grains) |
|---|---|---|
| drift of the mean, median (p10 / p90) | 7.2° (1.7 / 13.5) | 4.2° (2.0 / 7.5) |
| spread about the top-slab mean, median | 3.4° | 6.1° |
| drift / spread, median | 1.66 | 0.60 |
| max SROD, median | 15.0° | 16.0° |
| top-slab mean SROD, median | 8.4° | 8.2° |
| fraction of slabs with rising mean SROD, median | 0.79 | 0.78 |

Drift scales with grain height in v10 (≈2° at 100 µm, 8–15° at 400 µm)
and the tall grains trace directed IPF paths; a minority still reverse
(monotonic fraction 0.6–0.7) because the 30 µm correlation height allows
direction changes over 100 µm scales. v8's grains spread more than they
drift at every height.

## Review response analyses — 2026-09-09 (existing data)

Reviewer: `docs/SROD_REPORT_REVIEW.md`. Results in `docs/gp700_results_review/`.

- **A1 path factor (M7), measured from d10:** median path length per unit height over the
  filtered grains of v10 = **1.47** (2° segmentation: 1.54). The 1.8 quoted before was the
  D-fit ratio; the remaining factor (1.8/1.47 ≈ 1.2) is the walk's own excess over the height
  index (diagonal hops etc.), i.e. "retention" ≠ path factor.
- **A2 held-out test (M1/M6): established class, floor 240 µm, vs Lamb Fig. 5c over
  25–233 µm:** v10 RMS **0.28°**, bias −0.11°, band/mean 1.39; v8 RMS 2.99°, bias +2.98°,
  band 1.49. The coherent model passes a target it was not tuned on; the per-voxel model fails it
  (its floor-referenced grains start 3° too high because the lineages have already diverged).
- **A3 2° segmentation (M3):** v10 → 411 k labels, 139 pass Lamb's filters (spin labels: 96),
  ESD p50 29 µm, SROD RMS 0.60° / bias −0.45° / band 1.49, 2 grains span > 300 µm; per-grain drift
  5.5°, spread 1.9°, drift/spread 2.4. v8 → 758 k labels, 54 filtered, ESD p50 8 µm, max height
  165 µm, RMS 1.19° / −0.98°. The per-voxel walk alone (v8) does not survive a 2° segmentation as
  columnar grains; the coherent walk (v10) keeps the SROD curve within 0.6° but the grain
  population changes (finer, fewer tall grains): the same-grain neighbour disorientation (median
  2.4°) sits at the threshold. A 2° segmentation of the simulation is not equivalent to Lamb's
  because the simulated field has voxel-scale noise the EBSD does not.
- **A4 hatch periodicity (M4):** boundary-density autocorrelation at the 80 µm lag 0.61 (x) /
  0.70 (y) in v10, 0.58 / 0.71 in v8; 43–62 % of the spectral power at the hatch period;
  in-plane grain width medians 30 / 39 / 51 µm at 200 / 300 / 405 µm height (p90 72–84 µm).
  Confirmed: the structure is locked to the registered hatch pattern (layers n and n+2 identical).
  Lamb: bidirectional, 90° rotation, repeats every 2 layers, 45° to the block edges, SLM125; the
  layer-to-layer hatch registration is not stated. Hatch-offset sensitivity run queued (B3).
- **A5 originating-grain census (M6):** over 26 existing runs (nucleation 6.5e12–8.5e13 m⁻³,
  tc 3–23 K) grains originating > 60 µm above the remelt floor and passing Lamb's filters number
  0–7 per run (0–4 % of the filtered volume); no region of the tested space approaches Lamb's
  5:1. Nucleated grains do not survive competitive growth at these settings.
- **A6 Lamb-style per-grain numbers (M2):** v10: grains with max SROD > 25°: 5 % (Lamb: "most");
  max SROD p90 / max 23 / 33° (Lamb max 58.7°); per-layer mean SROD at 300 µm (grains > 315 µm,
  n = 50): median 10.8°, p90 15.6°, max 18.0° (Lamb G1–G4: 20–25°); accumulation rate (drift per
  30 µm) mean 0.84°, max 1.92° (2026 paper, bulk grains: average 1.7°, maximum 5.8° per 30 µm).
  v8: 17 % > 25°, rate mean 0.57 / max 1.47. The model's per-grain accumulation is about half
  of Lamb's and its extremes are far smaller, while the population mean matches: the
  experimental population must contain a heavier tail (a few fast-rotating grains) than a
  Gaussian OU walk with one D_g produces.
- Random baseline for <001> within 15° (10°) of BD computed by Monte Carlo: **0.102 (0.045)**,
  not 0.09.

**B1 seed scatter (proxy, 3 seeds each, 12:24):** v8 settings RMS 0.50±0.05°, bias +0.17±0.19°,
band 1.45±0.05, filtered 52±2, fibre 0.23±0.02, drift 3.9±0.3°, spread 6.7±0.7°, max SROD 16.6±0.1°;
v10 settings RMS 0.35±0.05°, bias +0.05±0.11°, band 1.53±0.07, filtered 55±3, fibre 0.23±0.02,
drift 6.6±0.5°, spread 3.9±0.4°, max SROD 13.7±0.3°. Population score differs at ~2σ; per-grain
drift/spread at > 5σ.

**B3 hatch-offset sensitivity (proxy, v10 settings, HATCH_OFFSET=random, 13:37):** SROD 0.30° / −0.14°,
band 1.50, 55 filtered, ESD p50 120 µm (registered seed_v10_s1: 0.35°, 123 µm); autocorrelation at
the 80 µm lag 0.34/0.48 vs 0.46/0.45 registered; power at the period 0.52/0.39 vs 0.47/0.57; widths
27–36 vs 30–33 µm; drift/spread 5.3/3.6 vs 6.0/3.8°; <001> 0.30 vs 0.21–0.25. The pool spacing
imprints the 80 µm period on every layer regardless of registration; SROD unaffected.

**B1 second full v10 build (wide12_v10_s2, seed 24680, 15:11):** RMS 0.21°, bias 0.00°, band 1.46,
97 filtered, ESD p50 156 µm, <001> 0.20; established vs Fig 5c 0.47° / +0.25°; drift 6.6°, spread 3.2°,
max SROD median 12.7 (p90 24.5, max 49.3 — one grain at Lamb's extreme), per-layer mean at 300 µm median
9.9°, max 18.7°; rate mean 0.83, max 5.39°/30 µm; path factor 1.49.

**B2 1.5 µm v10 (rho 0.95, D_g 0.0049, D_v 0.065; 21:40):** RMS 0.74°, bias +0.64°, band 1.42, 65 filtered,
ESD p50 115 µm, <001> 0.18; drift 7.8°, spread 4.4°, max SROD median 15.6°, rate 1.00°/30 µm, path
factor 1.68. Mean SROD 5–15 % above the 3 µm proxies at every height (same excess as the per-voxel-only
test): first-order resolution independent under the rho/D_g rescaling.

**Review 2 (66f114c6): minor revision.** Follow-ups: ESD in the tables is the volume-weighted median over all
grains (number-weighted median of the filtered grains: v10 75 µm, v8 72 µm; filtered grains hold 96 % of
the crop volume) — definition, not an error. Established floor sensitivity (v10): floor 200 / 240 / 300 →
bias −0.65 / −0.11 / +0.44°. Origin census with 2° segmentation: v10 55 of 141 filtered grains originate
> 60 µm above the floor (13 % of volume), v8 0 of 54 — the coherent walk creates low-angle boundaries that a
2° segmentation reads as mid-build origins; spin labels: 3 of 100 (v10), 3 of 94 (v8). Rescaled 1.5 µm run
(D_v, D_g × 0.875, `dx1p5_v10r`) launched 21:53 for the path-factor attribution.

**B4 rescaled 1.5 µm v10 (D_v 0.057, D_g 0.0043, rho 0.95; 04:24):** RMS 0.30°, bias +0.18° (unscaled +0.64°),
band 1.49, drift/spread 7.1/4.4°, path factor 1.76; excess over 3 µm −4 / +7 / +12 % at 45 / 205 / 282 µm.
The path-factor ratio accounts for most of the resolution dependence. Report revision 2b complete.

**Segmentation threshold sweep (2026-09-10):** v10 spin/2°/3°/5° → filtered 97/139/97/91, >300 µm 48/2/37/39,
ESD p50 148/29/128/144 µm, score 0.34/0.60/0.25/0.57°; v8 → 86/54/126/94, >300 µm 41/0/11/30, ESD
151/8/16/128 µm, score 0.49/1.19/1.51/0.93°. With the simulation's voxel noise (same-grain neighbour
disorientation median 2.4° v10, 3.8° v8) a 3° threshold on v10 is the noise-matched equivalent of Lamb's 2°;
v8 needs 5°.

**Segmentation vs voxel size (2026-09-10):** 1.5 µm v10 (rescaled) spin/2°/3°/5° → filtered 63/59/60/59, ESD p50
116/79/102/115 µm, score 0.30/0.16/0.24/0.47°, neighbour disorientation median 1.9°; 3 µm proxy seed_v10_s1 →
57/63/53/54, ESD 123/45/101/119 µm, score 0.35/0.69/0.39/0.43°, neighbour median 2.1°. The 2° threshold behaves
at 1.5 µm as 3° does at 3 µm (per-voxel step ∝ √dx). No orientation smoothing before segmentation.
