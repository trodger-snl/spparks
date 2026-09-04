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
