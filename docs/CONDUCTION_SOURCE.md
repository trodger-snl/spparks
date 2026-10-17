# Conduction temperature source (`setup_temperature_source conduction`)

Nonlinear transient heat conduction for `additive/texture/kk`: finite volume
in enthalpy form on a graded, cell-centred tensor-product grid, on the device.
It complements the analytic sources: temperature-dependent properties, latent
heat, powder, surface losses, and a thermal domain larger than the lattice box.

Code: `src/temperature_source_conduction.{h,cpp}` (configuration, tables,
laser pose; a `MoserTemperatureSource` subclass) and
`src/KOKKOS/conduction_solver_kokkos.{h,cpp}` (grid, kernels, halo).
Examples in `examples/conduction_am/`: `in.single_track`; `in.thin_wall`
(four layers, one `laser_path` per layer, ported from am/finitediff's
ThinWall); `in.large_rectangle` (am/finitediff's LargerRectangle raster path
through `laser_path file`).

## Syntax

```
setup_temperature_source conduction Q lambda k alpha T0 cp sx sy sz [keyword value ...]
```

The nine positional values are Moser's standard form and mean the same thing:
absorbed power `lambda*Q`, a Gaussian `exp(-x²/sx² - y²/sy² - z²/sz²)` deposited
into the half space below the laser plane, `rho = k/(alpha*cp)`, ambient `T0`.
`laser_path` (incl. `hatch`, `serpentine`, `pause`, `pause_below`,
`reset_temperature`) works unchanged; `reset_temperature` resets the whole
thermal field to that temperature (phases are kept).

### Path files (`laser_path file`, Moser and conduction)

```
laser_path file F speed V [scale S] [shift DX DY DZ] [jump_speed V] [jump_time T] [recoat T] [layers N]
```

`F` has one waypoint per row, `x y z p [v]` (whitespace or commas, `#`
comments): a move from the previous point to (x, y, z) at power fraction `p`
of `lambda*Q` (0 = laser off) and speed `v` (default `V`). Off moves take
`jump_time` when given, else distance / `jump_speed` (default `V`).
Coordinates are file units × `scale` + `shift` (m), so a path in lattice
units takes `scale dx`. A change of z starts a new layer at that point;
`layers N` keeps the first N. Each layer starts once the previous one is
finished (below, *Layers*) plus `recoat` seconds (default 0), all within one
`run`. `pause`, `pause_below` and `reset_temperature` do not apply.

`tools/laser_path_convert.py` writes the format from am/finitediff path
files (`fd`) and from G-code (`gcode`: G0/G1 X Y Z with E or S marking the
laser on, `--travel-feed` for travel moves, `--feed-scale` to keep F as the
per-move speed).

| Keyword | Meaning | Default |
|---|---|---|
| `k_table n T1 k1 ...` | dense k(T), piecewise linear, clamped | constant `k` |
| `cp_table n T1 c1 ...` | cp(T), piecewise linear, clamped | constant `cp` |
| `latent L` | latent heat (J/kg), released linearly over [solidus, liquidus] | 0 |
| `k_powder k` | powder conductivity | k/100 |
| `convection h T_inf` | convective loss on gas faces and the top face | h = 0 |
| `emissivity e` | radiative loss (to `T_inf`) on the same faces | 0 |
| `margin xlo xhi ylo yhi` | thermal domain beyond the lattice box (m) | 0 |
| `plate d` | thermal domain below the lattice box (m) | 0 |
| `stretch r` | width ratio of successive margin/plate cells | 1.1 |
| `max_cell w` | cap on margin/plate cell width (m) | none |
| `part regionID` | outside the box: dense where the region matches, else powder | all dense |
| `layer_thickness t` | outside the box: powder within `t` of the laser plane | 0 |
| `plate_top z` | everything below `z` (m) is dense | box bottom face |
| `far_bc fixed\|adiabatic` | outer faces of the thermal domain | fixed at `T0` |
| `sts s` | RKL2 stages per laser-off super-step; 0 = explicit only | 32 |

Not supported (errors): the CPU app (`additive/texture`), `keyhole`,
`laser_fluctuations`, `temperature_smooth`, periodic boundaries, splitting z
in `processors`.

## Model

- **Grid.** Inside the lattice box the cells *are* the lattice sites (spacing
  dx), so sites read their cell directly. Outside, cell widths grow by `stretch`
  per cell out to `margin`/`plate`. Example: a 1×0.2×0.15 mm box at 5 µm with
  1 mm margins and plate is 262×102×61 cells (1.6 M), against 61 M uniform.
- **Top.** The grid's top face is half a cell above the highest site layer at
  or below the laser plane. Sites above it read `T0`. A laser plane above that
  face (e.g. GP700's `ztop·dx`, half a cell up) is moved down onto it for the
  source, so all absorbed power is deposited; put the plane on the face
  (z = (top layer + 0.5)·dx) to match Moser's surface exactly.
- **Phase per cell.** GAS / POWDER / DENSE. Inside the box it is taken from
  `i2` once (0 → powder, 1 → powder, 2/3 → dense, 5 → gas), so stitch-loaded
  powder is kept. `i2 = 1` means "not yet melted", which substrates also use:
  with `layer_thickness` set, only sites within it of the laser plane count as
  powder and deeper `i2 = 1` sites are dense. After that the solver tracks it itself: powder becomes dense
  the first time it reaches the liquidus.
- **Fluxes.** Between two cells, a harmonic mean of k over the centre distance.
  A face next to gas, and the top face, get convection plus radiation. Far
  faces are fixed or adiabatic.
- **Source.** The Gaussian is integrated over each cell (it separates into
  per-axis erf differences, computed once per sub-step), so the energy is
  right on coarse cells too. Power is scaled by the fraction of the
  sub-step the laser is on.
- **Time.** Explicit, sub-cycled at 0.9× the global stable step: the minimum
  over cells of ρ·cp_min·V / Σ conductance at k_max. The step is the same on
  every rank. T(H) comes from a 1 K table.
- **MPI.** x/y follow the lattice processor grid, with a 1-cell halo of T and
  phase exchanged in x then y every sub-step. Edge ranks also own the margin
  cells; that load imbalance is not rebalanced.
- **Laser-off intervals: RKL2 super-time-stepping** (Meyer, Balsara & Aslam
  2014). An interval with the laser off for its whole length is advanced in
  super-steps of up to `sts` stages, each stable for
  τ ≤ dt_fd·(s² + s − 2)/4 and second order, whenever that takes fewer stage
  evaluations than explicit steps. The phase updates once per super-step.
- **Fast-forward.** Once max T drops below the liquidus with no pass active,
  the app jumps to the next pass start (Moser's predictor role, with
  `next_laser_on` instead of the Green's-function estimate). The jump itself
  is integrated with RKL2, so an inter-hatch pause costs about
  pause / (dt_fd·sts²/4) super-steps instead of pause / dt_fd steps.
- **Layers.** A layer is finished when its last pass has emitted, max T is
  below the solidus and no site is liquid. A path file then starts its next
  layer after `recoat`; otherwise the field is reset to `T0` and the rest of
  the run is skipped, and the next `laser_path` + `run` is the next layer
  (the recoat, as in am/finitediff). A new plane rebuilds the grid up to it
  at `T0` and re-reads phases from `i2`; a deposited layer can stay at
  `i2 = 0`, which below the laser plane counts as powder and melts like
  `i2 = 1`. The liquid test keeps a lagging solidification front from being
  frozen by the reset (a very slow `solid_front_vel` can hold it off
  indefinitely).

## Validation (2026-10-16, gfx1151 + OpenMP backend)

| Check | Result |
|---|---|
| vs `moser`, constant properties, `k_powder = k`, adiabatic, 1 mm margins | box energy 0.01–0.04 %, peak T ≤ 0.07 % |
| Energy, adiabatic box, latent 0 vs 2.7e5 | stored energy equal to 2.5e-8 (print limit) |
| `stretch 1.0` (12.5 M cells) vs `1.1` (1.2 M), 0.5 mm margins | box energy 1e-4, peak ≤ 0.08 % |
| np 1 / 2 / 4 (4×1×1 and 2×2×1) | per-site d6 identical |
| HIP vs OpenMP (including all-features deck at np = 2) | per-site d6 identical |
| `pause_below` + `reset_temperature`, `pause` + `reset_temperature` | fire and reset as with Moser |
| `in.thin_wall` (4 layers), np 1 vs 2×2×1 | d6 identical; continuous wall z 16–39, each layer fused to the one below; 28 s |

| RKL2 vs explicit, oracle deck (0.19 ms skip) | per-site max 1.0 K after the skip, 0.4 K at end of next pass |
| RKL2, np 1 vs 2×2×1 | per-site d6 identical |
| GP700 proxy box 0 (400 µm, 5 hatches, 8.8 ms pauses): `sts` 0 / 32 / 128 | pool, fusion depth, G/V quartiles equal to 3 digits; loop 573 / 141 / 99 s (Moser 52 s) |

**vs am/finitediff (github/am_finitedifference f6a84fe0, `make mpi`).** Same
decks, spot, substrate and fixed box faces (no margins), `/home/tron/fd_runs`:

| Deck | Solid above substrate (sites) | Pool width / depth | Peak T | Time |
|---|---|---|---|---|
| ThinWall, am/finitediff | 58,300 | 100 / 60 µm, melts to z 11 | 2,150 K | 26 s, 8 CPU ranks |
| ThinWall, conduction | 16,600 (layer bottoms unfused) | 60 / 25 µm | 1,800 K | 33 s, 1 GPU |
| ThinWall, conduction, `latent 0` | 58,100 | 105 / 30 µm | 1,980 K | |
| LargerRectangle (2 layers), am/finitediff | 600,600 | | | 881 s, 16 CPU ranks |
| LargerRectangle, conduction | 570,800 | | | 696 s, 1 GPU |

am/finitediff deposits more heat than its laser: with constant properties
its ΣT rises 1.3–3× the source energy. Sites are updated in place, so
each neighbour pair exchanges r·ΔT of the earlier site's step (laser
heating included), and dt = 7e-7 s is above the explicit limit
(r = αdt/dx² = 0.31 > 1/6; it diverges when powder k = solid k). Its
surface term also has the wrong sign (convection and radiation heat), and
its clock stops while the MC step is shorter than the thermal step. Ours
at 2× power overshoots it, so the gap is its energy excess.

Identity is checked on `%g` text dumps (6 significant digits). GP700 runs and
`compare.py`: `/home/tron/gp700_runs/cond_cmp`.

**Precision.** The stage kernel is FP64-bound on gfx1151 (about 20 FP64
divisions per cell). Built with `-DSPPARKS_FP32=ON` (`build-fp32`) the
solver runs entirely in `fastreal` = float, state included; H is stored
relative to H(T0), which keeps cold cells (small H increments) resolved.
The default FP64 build is unchanged bit for bit.

| Matched decks | FP64 | FP32 |
|---|---|---|
| LargerRectangle 2 ms: `ConductionStep` / loop | 19.2 / 20.5 s | 4.3 / 5.9 s |
| per-site d6 vs FP64: max / mean / cold (< 310 K) mean | | 0.010 K (print) / +3e-7 / −2e-7 K |
| adiabatic energy deck vs FP64 | | −2.0e-6 |
| ThinWall 4 layers (mostly RKL2): loop; ΣT, max T vs FP64 | 20.1 s | 3.8 s; 5.4e-5, 0.08 K |

Without the offset, float H lost increments below about 2e-5 K in cold
cells (mean −3.5e-4 K after 8,778 steps). Keeping H double with float
fluxes only gives 7.9 s on the same step.

Cost: the example deck (1.6 M cells, about 4,000 sub-steps) runs in about 11 s
on one gfx1151. With RKL2 the remaining cost is mostly the explicit steps
while the laser is on.

## Possible next steps
- Keyhole lobes and fluctuations: sample the second lobe and `fluct_*` per sub-step.
- Rebalancing margin cells across ranks.
- Carrying thermal state across stitch boxes. Today each box starts at `T0`.
