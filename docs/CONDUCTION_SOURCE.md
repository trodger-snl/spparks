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

Not supported (errors): the CPU app (`additive/texture`), `keyhole`,
`laser_fluctuations`, `temperature_smooth`, periodic boundaries, splitting z
in `processors`.

## Model

- **Grid.** Inside the lattice box the cells *are* the lattice sites (spacing
  dx), so sites read their cell directly. Outside, cell widths grow by `stretch`
  per cell out to `margin`/`plate`. Example: a 1×0.2×0.15 mm box at 5 µm with
  1 mm margins and plate is 262×102×61 cells (1.6 M), against 61 M uniform.
- **Top.** The grid's top face is half a cell above the highest site layer at
  or below the laser plane. Sites above it read `T0`. Put the laser plane on
  that face (z = (top layer + 0.5)·dx) to match Moser's surface exactly.
- **Phase per cell.** GAS / POWDER / DENSE. Inside the box it is taken from
  `i2` once (0 → gas, 1 → powder, 2/3 → dense, 5 → gas), so stitch-loaded
  powder is kept. After that the solver tracks it itself: powder becomes dense
  the first time it reaches the liquidus.
- **Fluxes.** Between two cells, a harmonic mean of k over the centre distance.
  A face next to gas, and the top face, get convection plus radiation. Far
  faces are fixed or adiabatic.
- **Source.** The Gaussian is integrated over each cell (erf products), so the
  energy is right on coarse cells too. Power is scaled by the fraction of the
  sub-step the laser is on.
- **Time.** Explicit, sub-cycled at 0.9× the global stable step: the minimum
  over cells of ρ·cp_min·V / Σ conductance at k_max. The step is the same on
  every rank. T(H) comes from a 1 K table.
- **MPI.** x/y follow the lattice processor grid, with a 1-cell halo of T and
  phase exchanged in x then y every sub-step. Edge ranks also own the margin
  cells; that load imbalance is not rebalanced.
- **Fast-forward** is off for this source. It integrates through pauses. With
  `pause`/`pause_below` and no `reset_temperature`, an inter-pass pause costs
  pause/dt_fd sub-steps.

## Validation (2026-10-16, gfx1151 + OpenMP backend)

| Check | Result |
|---|---|
| vs `moser`, constant properties, `k_powder = k`, adiabatic, 1 mm margins | box energy 0.01–0.04 %, peak T ≤ 0.07 % |
| Energy, adiabatic box, latent 0 vs 2.7e5 | stored energy equal to 2.5e-8 (print limit) |
| `stretch 1.0` (12.5 M cells) vs `1.1` (1.2 M), 0.5 mm margins | box energy 1e-4, peak ≤ 0.08 % |
| np 1 / 2 / 4 (4×1×1 and 2×2×1) | per-site d6 identical |
| HIP vs OpenMP (including all-features deck at np = 2) | per-site d6 identical |
| `pause_below` + `reset_temperature`, `pause` + `reset_temperature` | fire and reset as with Moser |

Identity is checked on `%g` text dumps (6 significant digits).

Cost: the example deck (1.6 M cells, about 4,000 sub-steps) runs in about 16 s
on one gfx1151.

## Possible next steps
- Super-time-stepping (RKL2) if pause sub-steps dominate a run.
- Keyhole lobes and fluctuations: sample the second lobe and `fluct_*` per sub-step.
- Rebalancing margin cells across ranks.
- Carrying thermal state across stitch boxes. Today each box starts at `T0`.
