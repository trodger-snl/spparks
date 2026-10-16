# Conduction temperature source (`setup_temperature_source conduction`)

Nonlinear transient heat conduction for `additive/texture/kk`: finite volume
in enthalpy form on a graded, cell-centred tensor-product grid, on the device.
It complements the analytic sources: temperature-dependent properties, latent
heat, powder, surface losses, and a thermal domain larger than the lattice box.

Code: `src/temperature_source_conduction.{h,cpp}` (configuration, tables,
laser pose; a `MoserTemperatureSource` subclass) and
`src/KOKKOS/conduction_solver_kokkos.{h,cpp}` (grid, kernels, halo).
Example: `examples/conduction_am/in.single_track`.

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
  `i2` once (0 → gas, 1 → powder, 2/3 → dense, 5 → gas), so stitch-loaded
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

## Validation (2026-10-16, gfx1151 + OpenMP backend)

| Check | Result |
|---|---|
| vs `moser`, constant properties, `k_powder = k`, adiabatic, 1 mm margins | box energy 0.01–0.04 %, peak T ≤ 0.07 % |
| Energy, adiabatic box, latent 0 vs 2.7e5 | stored energy equal to 2.5e-8 (print limit) |
| `stretch 1.0` (12.5 M cells) vs `1.1` (1.2 M), 0.5 mm margins | box energy 1e-4, peak ≤ 0.08 % |
| np 1 / 2 / 4 (4×1×1 and 2×2×1) | per-site d6 identical |
| HIP vs OpenMP (including all-features deck at np = 2) | per-site d6 identical |
| `pause_below` + `reset_temperature`, `pause` + `reset_temperature` | fire and reset as with Moser |

| RKL2 vs explicit, oracle deck (0.19 ms skip) | per-site max 1.0 K after the skip, 0.4 K at end of next pass |
| RKL2, np 1 vs 2×2×1 | per-site d6 identical |
| GP700 proxy box 0 (400 µm, 5 hatches, 8.8 ms pauses): `sts` 0 / 32 / 128 | pool, fusion depth, G/V quartiles equal to 3 digits; loop 573 / 141 / 99 s (Moser 52 s) |

Identity is checked on `%g` text dumps (6 significant digits). GP700 runs and
`compare.py`: `/home/tron/gp700_runs/cond_cmp`.

Cost: the example deck (1.6 M cells, about 4,000 sub-steps) runs in about 11 s
on one gfx1151. With RKL2 the remaining cost is mostly the explicit steps
while the laser is on.

## Possible next steps
- Keyhole lobes and fluctuations: sample the second lobe and `fluct_*` per sub-step.
- Rebalancing margin cells across ranks.
- Carrying thermal state across stitch boxes. Today each box starts at `T0`.
