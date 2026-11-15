# Conduction temperature source (`setup_temperature_source conduction`)

Nonlinear transient heat conduction for `additive/texture/kk`: finite volume
in enthalpy form on a graded, cell-centred tensor-product grid, on the device.
It complements the analytic sources: temperature-dependent properties, latent
heat, powder, surface losses, and a thermal domain larger than the lattice box.

Code: `src/temperature_source_conduction.{h,cpp}` (configuration, tables,
laser pose; a `MoserTemperatureSource` subclass) and
`src/KOKKOS/conduction_solver_kokkos.{h,cpp}` (grid, kernels, halo).
Examples in `examples/am/surge/conduction/`: `in.single_track`; `in.thin_wall`
(four layers, one `laser_path` per layer, ported from am/finitediff's
ThinWall); `in.large_rectangle` (am/finitediff's LargerRectangle raster path
through `laser_path file`); `in.thin_wall.ded` (`in.thin_wall` as directed
energy deposition, see *DED*); `in.thin_wall.ded.graded` (that wall graded
from 316L to IN625, see *Composition*).

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
laser_path file F speed V [scale S] [shift DX DY DZ] [jump_speed V] [jump_time T] [recoat T] [pause_below Tk] [layers N]
```

`F` has one waypoint per row, `x y z p [v]` (whitespace or commas, `#`
comments): a move from the previous point to (x, y, z) at power fraction `p`
of `lambda*Q` (0 = laser off) and speed `v` (default `V`). Off moves take
`jump_time` when given, else distance / `jump_speed` (default `V`).
Coordinates are file units × `scale` + `shift` (m), so a path in lattice
units takes `scale dx`. A change of z starts a new layer at that point;
`layers N` keeps the first N. Each layer starts once the previous one is
finished (below, *Layers*) plus `recoat` seconds (default 0), all within one
`run`. `pause_below Tk` (2026-11-14) makes "finished" also need max T < Tk:
an interlayer temperature instead of a fixed dwell (DED with `carry yes`);
not together with `recoat`.
`pause` and `reset_temperature` do not apply.

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
| `stretch r` | width ratio of successive margin/plate cells | 1.25 |
| `max_cell w` | cap on margin/plate cell width (m) | none |
| `part regionID` | outside the box: dense where the region matches, else powder (`region voxel` takes a footprint mask, `tools/insitu_footprint.py`) | all dense |
| `layer_thickness t` | outside the box: powder within `t` of the laser plane | 0 |
| `packing f` | powder (and powder melted in place) holds f × the dense mass | 1 |
| `plate_top z` | everything below `z` (m) is dense | box bottom face |
| `plate_material card` | baseplate material (see Phase per cell) | none |
| `plate_growth d` | `i2 = 6` may reach `d` above `plate_top` (m) | 0 |
| `far_bc fixed\|adiabatic` | outer faces of the thermal domain | fixed at `T0` |
| `sts s` | RKL2 stages per laser-off super-step; 0 = explicit only | 32 |
| `far R` | laser > R from the box: super-steps with a path-averaged source, app fast-forward while the box is cold, run ends when no pass comes within R again | 0 (off) |
| `ded [radius r] [T_birth T] [depth d]` | directed energy deposition (see *DED*) | off; r = 2 sx, T = T0, d = layer step |

Not supported (errors): the CPU app (`additive/texture`), `keyhole`,
`laser_fluctuations`, `temperature_smooth`, periodic boundaries, splitting z
in `processors`.

## Model

- **Grid.** Inside the lattice box the cells *are* the lattice sites (spacing
  dx), so sites read their cell directly. Outside, cell widths grow by `stretch`
  per cell out to `margin`/`plate`. Example: a 1×0.2×0.15 mm box at 5 µm with
  1 mm margins and plate is 234×74×47 cells (0.8 M) at the default 1.25,
  against 61 M uniform.
- **Top.** The grid's top face is half a cell above the highest site layer at
  or below the laser plane. Sites above it read `T0`. A laser plane above that
  face (e.g. GP700's `ztop·dx`, half a cell up) is moved down onto it for the
  source, so all absorbed power is deposited; put the plane on the face
  (z = (top layer + 0.5)·dx) to match Moser's surface exactly.
- **Phase per cell.** GAS / POWDER / DENSE. Inside the box it is taken from
  `i2` once (0 → powder, 1 → powder, 2/3 → dense, 5 → gas) at every depth,
  so stitch-loaded powder is kept and the part's walls are wherever the pools
  melted, rough as they are. A substrate must be loaded as `i2 = 3`. Outside
  the box, `part`, `layer_thickness` and `plate_top` decide. After that the
  solver tracks it itself: powder becomes dense the first time it reaches the
  liquidus.
- **Baseplate.** `plate_material <card>` adds a second dense material
  (flag bit `PLATE` in the phase byte, so it rides the existing halo
  exchange). Plate cells are `i2 = 6` sites and, outside the box, cells below
  `plate_top`, which is then required. The card gives ρ, k(T), cp(T),
  latent, solidus and liquidus; each material has its own k table and H(T)
  table on the shared 1 K grid, and the face flux is the same harmonic mean,
  so the interface needs nothing extra. Emissivity, convection and k_powder
  stay the part's. Material is fixed per cell at grid build (once per layer);
  plate still liquid at a rebuild would be meshed as part.
  In the kk app, `i2 = 6` is solid wherever 3 is (growth parent, smoothing,
  nuclei). Below `plate_top + plate_growth` a site melts and freezes at the
  plate's liquidus and solidifies back to 6; above, to 3. One
  solid_front_vel law for both. Fast-forward and the layer end use the lower
  of the two liquidus/solidus values (`ThermalHost` tl/ts).
- **DED** (`ded`). No powder bed:
  - Outside the box, the cells powder would fill (`layer_thickness`,
    non-`part`) are gas instead. `layer_thickness` and `packing` are errors
    with `ded`.
  - Inside the box, i2 = 0/1 cells are `POWDER | UNBORN` (phase bit 16).
    Until born, a cell is gas: it is not updated, takes no source energy,
    and its neighbours' faces toward it get convection + radiation.
  - A cell is born in the explicit step when three things hold: the laser
    is on, the beam axis is within `radius` of the cell centre, and the
    centre lies less than `depth` below the source plane. It becomes plain
    POWDER at `T_birth` (H = H(T_birth)), so from the next sub-step it
    conducts at `k_powder` and melts to dense at the liquidus. The pool
    pays the added powder's heat and melt, which is blown powder's
    thermal sink.
  - Birth follows the path, sub-step by sub-step. It is not layer-wise:
    switching on a whole layer would put heat capacity and conduction
    ahead of the beam, and leave the free surface in the wrong place.
  - The captured mass rate is set by the bead layout: i2 = 0 inside a box
    that is otherwise i2 = 5 above the substrate.
  - Beam shadowing by the powder cloud and in-flight heating fold into
    `lambda` and `T_birth`.
  - The grid rebuild re-reads phases from i2, so born powder that never
    melted is unborn again at the next layer: powder that didn't stick.
    The default `depth` is the laser plane's rise since the previous build
    (unlimited on the first), so it is not born a second time.
  - Not modelled: forced convection from the nozzle jet varying in space
    (Heigel et al. 2015). A constant `convection h` stands in.
- **Density.** One ρ (from `k`, `alpha`, `cp`) for every phase: give the
  dense value, and make the model layer the consolidated thickness (the path
  z step). Melting t of powder then takes the same energy as the real t/φ of
  powder at φρ. `packing f` instead gives powder cells f·ρ, kept after they
  melt (the bed melts in place without shrinking); it brackets the
  consolidated layer from the other side.
- **Fluxes.** Between two cells, a harmonic mean of k over the centre distance.
  A face next to gas, and the top face, get convection plus radiation. Far
  faces are fixed or adiabatic.
- **Source.** The Gaussian is integrated over each cell (it separates into
  per-axis erf differences, computed once per sub-step), so the energy is
  right on coarse cells too. Power is scaled by the fraction of the
  sub-step the laser is on.
- **Time.** Explicit, sub-cycled at 0.9× the global stable step: the minimum
  over cells of ρ·cp_min·V / Σ conductance at k_max, each bounded over the
  cell's own material (the part's k_max includes its powder); a face between
  materials takes the two k_max in series. The step is the same on every
  rank. T(H) comes from a 1 K table. With `plate_material` the steel-like
  plate usually sets the step; plate cells only below the box are wider in
  z but keep the box's x/y spacing (tensor-product grid), so that buys at
  most 1.5× (6/4 faces), ~1.14× for the first 1.25·dx plate layer.
- **Kernel.** `ConductionStep` runs 32×8-cell teams that march up z. Each
  cell's (w/2)/k(T) per axis is computed once (one k-table lookup, three
  divisions) and shared through team scratch (x/y, plus a halo ring) and
  registers (z); a face is then A/(r_c + r_n). On gfx1151 that took FP64
  from 48.6 to 27.2 ms per call on a 33M-cell grid (FP32: 15.9 → 8.2 ms,
  with fastreal copies of the k tables and widths). Not bit-identical to the
  per-face form under -ffast-math (FP64 ≤ ~1e-12 K per step). Small grids
  (< a few hundred tiles) underfill the GPU and can be slower.
  Stage 1 of the discrete-GPU work (2026-10-28) took out most divisions:
  one 1/k per cell (r = (w/2)·(1/k), the same expression for the cell, its
  scratch copy and the halo ring), precomputed k-table slopes (and a walk
  from the bottom instead of a binary search), 1/width per axis instead of
  /vol, 1/(H(i+1) − H(i)) for T(H), and linear cell indexing. That is 13 → 7
  divisions per cell; sm_90 registers 76 → 70 (double), 50 → 48 (mixed).
  Per-site d6 moved ≤ 6e-11 K (HIP, OpenMP); stats lines unchanged.
  H200 (2 mm deck, 3 ms): 1.22x double, 1.26x mixed.
  **Rank count:** strict (non-fast-math) builds give identical per-site d6
  at np 1/2/4. Under -ffast-math (every Release build) the tiled kernel is
  not np-invariant: tile edges move with the rank boundaries, and fast-math
  rounds the halo-ring copy of a cell's r differently from the in-tile one,
  so np 2 differs from np 1 by ~1e-11 K (7e-12 K already before stage 1).
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

- **`carry yes`** (2026-11-14): a rebuild for a new plane keeps the field.
  The old grid first integrates to the call's t (the next layer's plane is
  set when the previous layer is done, so that interval is the dwell, laser
  off), then `carry_field` copies T into the cells the grids share (x/y
  and the lower z faces are unchanged; it errors otherwise) with H from
  each cell's material; the new top cells start at T0. A new `laser_path`
  still resets. The first cut skipped the integration, so every layer
  started on the field as the previous one solidified (no dwell cooling);
  caught by timing the dwell (`in.rectangle.ded` layer 1: run to 0.26 s vs
  2.25 s gave the same max T). Fixed: layer 1 then cools from 1459 to
  310 K over the 2 s dwell, 96 RKL2 super-steps x 128 stages, 3.1 s wall
  (0.39 M cells, mixed, gfx1151). Decks without `carry`: bit-identical.

## Melt-pool flow (2026-11-14)

`flow lbm` and `composition ... mix flow`. Plan and rationale: flow is
needed to track composition through the pool; a fixed flat surface is
enough for mixing; one rank first (several ranks: see the end of this section).

- `src/KOKKOS/melt_flow_kokkos.h`: `MeltFlow<R>`, header-only, no SPPARKS
  dependencies. D3Q19, TRT (Lambda 3/16) + Guo forcing + Smagorinsky,
  buoyancy, Marangoni force on surface cells, Carman-Kozeny drag, half-way
  bounce-back, specular free-slip top; f stored as f - w. `run(h)` is the
  SI driver: even step count, step adapted to the measured speed (cap
  doubles above 0.15 lattice, halves below 0.03, f rescaled).
  `melt_advect`: three 1-D van Leer sweeps on the flow's own face mass
  fluxes (`flux`, tallied in `step`), see "Conservative transport" below.
- `tools/melt_flow_check.cpp` (target `melt_flow_check`, not in the default
  build): 93 checks, double + single, pass on HIP and OpenMP. Poiseuille
  1e-13; Taylor-Green rate 0.15%, error ratio 4.10; thermocapillary return
  flow 0.19% below the top cell, ratio 3.9; buoyant slot 2e-14;
  hydrostatic, Darcy, mass; uniform scalar 2e-16; step/Gaussian/cube
  conserved and bounded; diffused Gaussian width 0.4%; a 3-D Ti-6Al-4V
  pool through `run` (1.44 m/s, lattice 0.09, same flow from a 4x too
  coarse start).
- What the checks found: (1) bounce-back conserves a staggered momentum,
  so an abrupt force leaves an undamped checkerboard flipping each step;
  `ux/uy/uz` are two-step means and callers take an even number of steps.
  (2) The surface stress is a force on the top cell, which lags the
  profile by ~tau/(8 mu), first order; below it second order. (3) On host
  backends `create_mirror_view(v)` is `v`: keep "before" copies apart.
- Coupling (`ConductionSolverKokkos::flow_step`, before each explicit
  stage): map the box's cells to the flow (fluid = dense above its
  material's solidus; gas/unborn = surface; else wall), window = this and
  the last step's fluid cells + 2, `run(h)`, carry absolute H (H + the
  material's Href), then X under `mix flow`; the stage inverts the carried
  H. `comp_out` writes X to d11/`d_mat` per advance; `remat` follows.
  No RKL2 while liquid. The flow is rebuilt (at rest) with the grid.
- Results (FP64 unless noted): flow off bit-identical on the regression
  decks; zero forcing gives u = 0 and T within 0.32 K. 316L track (340 W,
  1.2 m/s, 5 um): 2.4 m/s outward, pool 29 vs 25 cells wide, peak 2751 vs
  2814 K, mixed within 0.43 K. Graded wall `mix flow`: top layer 0.99 vs
  0.87 well-mixed, central plume, sum of d11 -0.015%. DED pillar's first
  track (Ti-6Al-4V, 10 mm/s, 20 um, mixed): 1.0 m/s, peak 2972 vs 3479 K.
- Full DED pillar with flow (`in.rectangle.ded` + `flow lbm`, 20 layers,
  mixed, gfx1151): completes, 1779 s vs 919 s without flow; sampled peaks
  <= 3012 K (3400 without); 1717 unfused sites, all in the first site
  plane above the substrate (153 without: the flow's pool is shallower on
  the cold plate); grains per section 51/35/21/17/20/36 at z 30..220.
- `composition ... feed surface` (2026-11-14): a new cell takes the pool's
  mean X when it first melts; fed x minus that goes to a reservoir
  (`feed_res`) drained into the liquid cells within the birth depth of the
  source plane under exp(-r^2/r_birth^2), capped at the fed x. Two designs
  failed first: a one-cell skin could not take a 6-cell layer's feed (9.5%
  left in the reservoir); handing over at birth lost the feed of cells
  that never melt (2%). Graded wall: reservoir drains to zero, sum of d11
  51334 vs 51300 fed (+0.07%), layer means 0.13/0.50/0.71/0.77/0.99/1.00
  (slab: 0.20/0.50/0.70/0.77/0.98/0.99): close to slab here, as the feed
  zone is the layer itself; less spread inside a layer.
- Kernel fusion (2026-11-14): the box pass also loads H; the two carried
  fields share the sweeps; one write-back (11 launches per thermal step
  vs 18). It did NOT reduce the memory growth below. Measured 3-layer
  pillar peaks (GB, no flow 0.60): unfenced 1.76; fenced 1.13; fused with
  every flow step a reduction 2.03 (and slower); fused, fenced, plain
  steps, speed measured every 16th call 1.19. Reductions make it worse,
  fences help, launch count does not matter.
- Host memory on HIP (traced 2026-11-15 to glibc's mmap threshold, see below;
  the notes here are the earlier search): it grows with the kernel launches
  (gfx1151). The conduction-only pillar ends at 2.1 GB; with flow (about
  5x the launches per thermal step) it reached 12.6 GB. Not a leak in
  MeltFlow (700k launches standalone: flat 180 MB) nor in any reduction
  pattern or window shape tested alone; seen only inside spparks. A fence
  after each flow step and after `flow_step` cut the 3-layer peak from
  1.76 to 1.13 GB (no-flow: 0.60) at the same speed. Next: fewer launches
  (fuse the flow's setup/advect kernels), and find what the runtime keeps.
- Memory investigation (2026-11-14). Full pillar with flow, fenced + fused:
  7.5 GB peak (12.6 GB before). What is established:
  - the growth is glibc heap ([heap] in smaps), ~5.5 MB/s; the GPU-shared
    mappings only step at grid rebuilds;
  - the allocations are live during the run and freed at shutdown
    (mallinfo2 in use 107 -> 150 MB mid-run, 16 MB at exit);
  - a sampling LD_PRELOAD shim puts them inside libamdhip64 (rocm-hip
    7.1.1): 56-216 byte objects made per kernel launch and per async copy
    (stacks from MeltFlow::run, flow_step, Kokkos DeepCopyAsyncHIP);
  - about 200 bytes per launch inside spparks, 14 standalone with the same
    kernels: the trigger is something in the spparks process.
  Ruled out as the trigger or a cure: MPI (same without mpirun), pinned
  host memory (hipHostMalloc present and used: no change standalone),
  reducer type, window shape, AMD_DIRECT_DISPATCH=0, HIP_FORCE_SYNC_COPY,
  ROC_ACTIVE_WAIT_TIMEOUT, hipDeviceSynchronize / null-stream sync per step
  (-4%); GPU_FLUSH_ON_EXECUTION=1 and event syncs make it worse.
  Not yet tried: a newer ROCm; bisecting which spparks-side call turns the
  retention on (start from the standalone harness and add the app's
  per-step host<->device scalar copies, the team-policy stage kernel, the
  halo exchange); periodic Kokkos finalize is not an option.
- Conservative transport (2026-11-14). The first `melt_advect` used the
  mean u of two cells as the face flux and `q - d(uq) + q d(u)`; the last
  term kept a uniform q uniform but the sum of q drifted wherever the LBM
  velocity was not divergence-free. Now `step` tallies the mass each cell
  passes through each of its six faces (`flux(x, y, z, 6)`, +6 floats per
  cell): an axis population where it arrives; a diagonal one crosses two
  faces by way of the fluid cell(s) between, half each way if both are
  fluid. Summed over a cell's faces that is exactly its change of rho, so
  a sweep `q - (F_hi (qf_hi - q) - F_lo (qf_lo - q)) / m` conserves the sum
  of (1 + rho) q to round-off and keeps a uniform q uniform. q is per unit
  mass; a cell that freezes at rho = 1 + d keeps its q, so the sum of d11
  over sites is conserved to the size of d (not accumulating), not exactly.
  Two LBM defects came out with it: (1) the specular reflection at the
  surface tested GAS above the source's column only; at a step in the
  surface the reflected population was also streamed to the fluid cell
  above, making mass (check 12 without the fix: 0.9% per cell). It now
  needs GAS above both cells. (2) Two fluid cells touching only along an
  edge exchanged populations through the solid corner; that link now
  bounces back (it has no face route). Both change flow-on results.
  DED graded wall (300 W, 10 mm/s, 20 um, mixed; 127500 fed): sum of d11
  off by -0.004% (feed slab) and -0.02% (feed surface); before -1.6% and
  -2.7%. Top-layer mean d11 0.72 / 0.72 (before 0.71 / 0.69; mix pool
  0.67). Unfused sites 504 / 507 (before 426 / 428). 294 s per run.
- Several ranks (2026-11-14). Each rank's flow block is its own cells of
  the box plus two halo layers toward each x/y neighbour (`fh`); the step
  window stays inside the owned cells. `flow_exchange` (one templated
  pack / MPI / unpack, x then y so corners arrive) moves: type (2 layers)
  and T (1) before `run()`; all 19 populations (1 layer) after every flow
  step and rescale, through `MeltFlow::halo`; flux (1) before the sweeps;
  the carried fields (2 layers, along the sweep's axis only) before the x
  and the y sweep, through `melt_advect`'s callback; X (1) before
  `comp_out`, which now writes ghost sites too. Pool sums (mean X, feed
  reservoir), "any fluid" and the measured max speed are MPI-reduced, so
  every rank takes the same flow steps; a rank with no fluid has an empty
  window and only exchanges. Check (OpenMP, FP64, in.thin_wall.ded.graded.flow,
  pool on the rank boundaries): 1 vs 2 (1x2) vs 4 (2x2) ranks give d11,
  sum/max d6 and dump conduction's T and u bit for bit; with `feed
  surface` d11 agrees to 3e-15 (the reduced sums round differently) and
  its total exactly. One rank is bit-identical to the code before.
  HIP: see *HIP check* below.
  ponytail: 19 populations sent where 5 cross a face; an exchange per flow
  step (deeper halos would trade them for wider windows).
- Pool-only storage (2026-11-14). The flow's fields (`FlowWin`: a view
  plus the block index of its first cell) and the solver's carried fields
  (`fq`, `ftmp`, `fXw`, `fX2`) hold only a window of the block, `fW`, kept
  in box cells so it is the same on every rank; each rank holds its part.
  `flow_step` reduces the fluid cells' bounding box over ranks; when this
  step's and the last step's cells (+2) do not fit, the window moves to
  them plus a margin of a quarter of their size (4 cells at least):
  `FlowWin::move` allocates, copies what both windows hold and leaves the
  rest 0, which is the state a cell outside the step window has anyway.
  `FlowFill` then writes type, T, liquid fraction, H and X of the held
  cells (before, a pass over the whole box did). Only `fX` and `fborn`
  (5 bytes per cell, mixed precision) still cover the block; a held cell
  costs 229 bytes. `flow_exchange` sends the held lines and skips layers
  outside the window. Whether a field is exchanged must not depend on a
  rank's own held size (a rank away from the pool holds none): that hung
  the first 4-rank run. Check (OpenMP, FP64, as above): bit for bit the
  dense results on 1 and 4 ranks, with and without `feed surface` (d11,
  i1, i2, d6, and T and u of dump conduction); peak host memory of the
  one-rank run 625 MB against 805 MB. HIP: see *HIP check* below.
  ponytail: a move reallocates every field; keep one larger allocation and
  shift inside it if moves ever show in a profile.
- Properties by composition (2026-11-14): `flow lbm properties cards
  [sigma_table F]`. `flow_props` (temperature_source_conduction.cpp) makes
  tables from the composition cards' `[flow]` (now read by MaterialCard):
  sigma(T, x) (cards: linear mix, 2 T points; or the file), rho = r0(x) +
  r1(x) (T - Tr) with volumes adding, eta(x). `FlowFill` writes three more
  window fields per thermal step, `MeltFlow::sig` (N/m, less the table's
  first value), `dens` (rho/rho0 - 1) and `visc` (nu/nu0), rho0 and nu0 the
  first card's. The kernel (`props`): Marangoni = difference of sig between
  FLUID cells (one-sided at walls: the solid's x is not the liquid's)
  / (1 + dens); buoyancy = -g dens; tau from nu0 visc. So solutal Marangoni,
  solutal buoyancy, and dsigma/dT, expansion, viscosity per composition.
  sig replaces T in the halo exchange. Inertia is still rho0's (Boussinesq).
  melt_flow_check 13 checks the kernel against the plain model.
- HIP check (2026-11-14, gfx1151, build-verify = FP64 Release, the
  host-staged exchange): melt_flow_check passes in both precisions (13
  included). in.thin_wall.ded.graded.flow on 1 vs 4 (2x2) ranks, plain,
  `feed surface` and `properties cards`: d11 within 2e-15, sum/max d6 and
  i2 equal, dump conduction's u equal and T within 5e-13 (a fast-math
  build: not bit for bit, as the strict OpenMP build is). HIP vs OpenMP on
  one rank: d11 within 8e-14. 23 s on one rank, 39 s on four sharing the
  one GPU (OpenMP, 12 threads: about 155 s).
- Per-step cost (2026-11-15). Profile of a Ti-6Al-4V -> Nb DED track on HIP
  (mixed, 939k thermal cells, 30k fluid, kokkos-tools space_time_stack):
  ConductionStep 45%, FlowStep 17%, FlowCells 14%, FlowFill 9%, FlowAdvect
  8%. Four changes (e48f2504): (1) `dt_fd` with mixing is bound by the
  composition bins from the lowest to the highest present in the grid, not
  by all of them, so early layers of a graded build do not take the last
  layers' step; (2) `FlowFill` writes only the cells the step reads (the
  fluid cells of all ranks + 2, `need`), not the held window; (3)
  `MeltFlow::lazy`: a non-fluid cell is cleared only where `FlowFill`
  marked it (fl = -1: fluid at the last call); (4) the stage stamps
  `fgx/fgy/fgz` for cells left above the lowest solidus and `FlowCells`
  scans only that box (`FlowBox`, a small reduction, finds it; `fg_ok`
  false after anything else touched the field: full scan). (2)-(4) are bit
  for bit on OpenMP; (1) changes the step. Same track: 184.5 s -> 96.5 s,
  peak host memory 1.99 -> 1.32 GB. After: ConductionStep 48%, FlowStep
  14%, FlowAdvect 9%, FlowCells 7%, FlowFill 7%, FlowBox 3.5%.
- Limiter overflow (b6456158). `melt_advect`'s van Leer limiter went through
  r = a / b; with b tiny (a well-mixed pool, single precision) r overflowed
  and phi was inf / inf: NaN in X, then everywhere, and d11 no longer
  summed to the feed (+18% in one overheated track). Now 2ab / (a + b). The
  NaN went unnoticed because the builds are -ffast-math (a NaN test is
  compiled away, on host and device): `MeltFlow::bad` reads the bits;
  `run()` returns -1 on a non-finite or > 0.5 cell/step speed and
  `flow_step` stops on a non-finite H or X.
- Host memory on HIP (rocm 7.1.1). The growth is from asynchronous copies
  (each reduction's result is one): with `AMD_SERIALIZE_COPY=3` exported the
  track above peaks at 0.36 GB against 1.32 GB, 3% slower, same result;
  `AMD_SERIALIZE_KERNEL=3` makes it worse (1.82 GB). spparks.cpp sets the
  variable before the first HIP call if the user has not, and that takes
  effect (0.36 GB with nothing exported).
- The larger growth (2026-11-15): glibc's mmap threshold. Two 20-layer
  Ti-6Al-4V -> Nb walls with `AMD_SERIALIZE_COPY=3` still reached 45.8 GB
  (properties cards) and 29.4 GB (constant properties): flat at 0.4-0.8 GB
  for the first 13 / 50 minutes, then 1-2 GB a layer with steps of 2-9 GB;
  one single track of fourteen reached 2.1 GB. Not a leak: malloc_stats at
  exit shows 2 GB in use. glibc raises its mmap threshold (to as much as 32
  MB) whenever a large block is freed (a dump's buffer, a grid rebuild);
  from then on the HIP runtime's short-lived blocks below it come from the
  heap, which fragments and is not given back. The track: 0.35 GB without
  its dumps, 1.06 GB with `dump hdf5` alone, 2.1 GB with both dumps, 0.34
  GB with both and `MALLOC_MMAP_THRESHOLD_=131072` (setting it turns the
  adjustment off), same run time. Six layers of the first wall: 0.93 GB
  against about 8. An OpenMP run of four layers does not grow (0.8 GB).
  spparks.cpp now calls `mallopt(M_MMAP_THRESHOLD, 128 KiB)` in HIP builds
  unless the variable is set. Not yet run: a whole wall with it.
- Mirror symmetry and precision (2026-11-15). A track on the centreline is
  mirror symmetric in y. Double: exactly so, OpenMP (25 ms) and HIP (50 ms,
  to the float dump), 25% Nb on Ti-6Al-4V. Mixed: 1e-6 of the peak speed
  from the first step; with one composition it stays there for the whole
  track, with feed != substrate it jumps to 1e-2..2e-1 after 20-45 ms
  (HIP and OpenMP alike), in single frames first, as when a cell crosses a
  threshold a step before its mirror cell (a composition bin, probably;
  not confirmed). So the flow pattern of a mixed-precision run with a
  composition step is one of many; `properties cards` raises the
  asymmetry but is not its cause. Production runs are double.
- Double against mixed, same track, first 50 ms: HIP 21.0 s against 9.5 s
  (2.2x; whole track 296 s against 140.5 s); OpenMP, 16 threads, 77.2 s
  against 70.7 s (build-strict-omp, build-omp-mixed).
- A DED build as a chain of processes, one per layer (2026-11-15; written
  to bound the host memory before its growth was traced to glibc, above).
  Process k runs a path file holding layers k and k + 1 with `t0` = layer
  k's start, and stops half a recoat after layer k's scan (the next layer
  is in the file only so that the field is carried, not reset, when layer k
  is done; its start time is printed, "layer 2 of 2 ... starts at t=", and
  is the next process's `t0`). It ends with `dump hdf5 ... site i1 i2 d1 d2
  d3 d4 [d11]` and `dump conduction` (`dump_modify last yes` on both, or a
  delta that lands on the stop time). The next process reads them with `set
  site|i2|d1..d4 hdf5`, `reset_time` and `init grid`. Checked on OpenMP
  (in.rectangle.ded, layers 1 and 2 as two processes against one run, at
  1.25 s after layer 2): i2 equal in all 562500 sites; d6 equal at the
  median, within 0.02 K for 99% of the sites, 24 K at the worst. Grain ids
  differ (the keyed random numbers restart with the step count). Not yet
  tried with `flow lbm` (the flow restarts at rest, as at every new layer;
  the feed reservoir is not carried and should be empty by then).
- Is mesh refinement worth it? (2026-11-15, evaluation only.) Where the
  thermal cells are: the grid is a tensor product, so the 18 stretched
  margin cells per side keep the box's resolution along the other two
  axes. Single track (170 x 50 x 35 box cells): 32% of the 939k thermal
  cells are in the lattice box; 20-layer wall at its last layer (206 x 86 x
  253): 45%. Inside the box of a hot single track a grid 2x coarser would
  be within 5 K (by the second difference of T) in about 90% of the cells,
  4x coarser in about 80%. Options, with the gain on the whole run
  estimated from the profile above (conduction 48%):
  (a) `stretch 1.6` or `2.0` instead of 1.25, no code: 939k -> 598k / 495k
      cells, measured 13% / 14% faster; the pool melts 1.9% / 4.1% more
      substrate and peaks 4 / 9 K hotter.
  (b) margins and plate coarse along all three axes (a second, coarse grid
      coupled at the box faces): about half the cells at 20 um, about 1.3x
      on the run; at 5 um the box dominates and the gain falls to 1.1x.
      One to two weeks (stage kernel, halo, `init grid`, dump).
  (c) a fine patch that follows the pool inside a coarse box grid, with
      its own time step: the conduction cost falls to the patch (pool +
      about 0.5 mm), about 1.7x on the run at 20 um, where the flow then
      dominates, but about 10x fewer thermal cells at 5 um (122M box cells
      for the tall wall against about 12M in a patch). A month or two:
      regridding, conservative transfer, sites take T from two levels,
      birth and phases on both, the flow window inside the patch.
  (d) general adaptive refinement (octree or blocks): no more gain than
      (c) for this problem (one pool), much more code.
  Recommendation: not at 20 um (the four changes above and (a) give more
  for less; the flow is the next cost). For 5 um builds (c) is the scheme
  to build, and it is what makes them fit in memory; skip (b) and (d).
- Limits: flat surface; without `properties cards` one set of flow
  constants for every composition; with it the inertia is still one
  density's and eta has no T dependence; melted LOOSE cells carried as
  dense.

## Material (examples: 316L)

The 316L values below are also in the material card `materials/316L.toml`
(docs/MATERIAL_CARDS.md); `examples/am/surge/conduction/in.single_track.card` runs
`in.single_track` from it with `setup_temperature_source conduction card`.

The example decks use 316L, ρ = 7900 kg/m³ (`k` 13.8, `alpha` 3.6166e-6,
`cp` 483 give ρ = k/(α cp)), solidus/liquidus 1675/1708 K, latent 2.9e5
J/kg, emissivity 0.4, k_powder 0.3. Sources: NIST (Pichler et al. 2020,
PMC8193647) for ρ, cp, the melting range and ΔH; Mills (2002) fits as in
MOOSE's ThermalSS316Properties for k(T) (13.8 W/m·K at 300 K, 31.3 at the
solidus, held in the liquid). The earlier decks carried am/finitediff's
values (ρ 5706, a compromise between powder and dense; k 30 constant;
304's melting range; emissivity 1). At the same 150 W the 316L pools are
shallower: deepest melt 70 → 55 µm below the first layer top in
`in.large_rectangle`, 60 → 35 µm in `in.thin_wall` (FP64, 2026-10-17).

**Powder depth test** (`in.single_track`, 316L, 2026-10-17). The model
layer is the consolidated thickness at dense ρ. Against a powder layer
1/φ deeper at `packing φ` (same mass per area, melts in place without
shrinking) and the same deeper layer at dense ρ, mid-track:

| Powder layer | Melt depth into substrate | Remelted substrate (sites per x slice) | Peak T |
|---|---|---|---|
| 25 µm, dense ρ (consolidated) | 47.5 µm | 36 | 2877 K |
| 40 µm, `packing 0.625` | 37.5 µm | 25 | 3632 K |
| 40 µm, dense ρ | 37.5 µm | 18 | 2976 K |

The real case lies between the first two rows (the melt does shrink to the
consolidated height). So the consolidated layer overstates remelt of the
layer below by up to ~10 µm (two sites) and ~30 % in area; a source
calibrated against measured pools on powder absorbs it. `packing` with a
deeper powder layer is for single-layer tests only: the microstructure
model does not shrink melted powder, so a build would grow 1/φ per layer.

## Materials per cell and composition (2026-11-14)

Each cell carries a material byte `M` beside its phase byte `P`; every
material table is a row of a 2D view: H(T) and 1/ΔH on the shared 1 K
grid, the k(T) table (`kN(m)` points, padded) and its slopes, H(T_birth),
the liquidus. `ConductionTemperatureSource::materials(ts, tl)` lists them:
the part (or the composition bins), then the plate. The `PLATE` phase bit
stays (i2 = 6 semantics); `M` alone picks the tables. The halo exchange
packs `P | M << 8` into the phase double. Under FP32 each row is stored
relative to its own H(T0), as the part and plate tables were.

`composition n x1 card1 ... [bins b]` fills the bins from cards at nodes
(linear in x, tables on the union T grid). The kk app maps each site's d11
to a bin at init (`setup_composition`, `d_mat`), passes it to the solver for
the lattice cells, and `PlateRule::tl_at(i, z)` returns the bin's liquidus.
`mix pool` (default; 2026-11-14): after the melt kernel, `mix_pool` sums
the liquid (i2 = 2) sites' x over the active-front list in fixed point
(2^30 over x1..xn, so exact: independent of list order and rank count),
Allreduces, and gives every liquid site the mean and its bin. The solver's
`remat` moves a lattice cell whose bin changed to the new table at the same
T at each `advance` (then a halo exchange), and the stable step bounds
every part cell by all bins. One pool for the whole domain; equal weight
per site. `mix none` keeps x as loaded.

Checks (gfx1151, FP64):
- `in.single_track.plate`, `in.thin_wall.ded`, `in.single_track.card`:
  `dump conduction` T/phase and `dump hdf5` d6 bit-identical to SURGE
  51788775, np 1; `in.single_track.plate` also at np 2.
- Mixed precision (`build-fp32`), same two decks vs 51788775 mixed: max
  |ΔT| 0.003 K (plate) and 0.019 K (DED), phase and spins identical; not
  bitwise (the restructured stencil contracts differently under
  fast-math). Composition x = 0 vs none: bit-identical in mixed too.
- `in.single_track.plate` + `composition 2 0 Ti6Al4V 1 Ti6Al4V`, all sites
  x = 0 and all x = 1: bit-identical to the run without composition.
- `in.thin_wall.ded.graded` (layers x = 0.25..1, 316L → IN625) vs the same
  deck at x = 0: wall width at mid-layer 12/14/16/16 sites vs 12/12/14/14
  (IN625 melts 85 K lower); every solid site keeps its layer's x.

- `mix none`: the graded deck bit-identical to the static version
  (3c9a0e0b); the decks without composition bit-identical to 51788775.
- `mix pool`, graded deck (fed x = 0.25/0.5/0.75/1 on x = 0): mean x of
  the solid at sites z 20/24/28/32/36 = 0.31/0.50/0.64/0.86/0.87, the top
  substrate picks up 0.01; the top layer ends at 0.87, not 1 (about half
  of each pool is remelt). Repeats bit for bit; np 2 and mixed precision
  give the same layer means to 3 digits.

Next: x set at powder birth from a feed schedule, and a composition
database file in place of cards at nodes.

## Validation (2026-10-16, gfx1151 + OpenMP backend)

| Check | Result |
|---|---|
| vs `moser`, constant properties, `k_powder = k`, adiabatic, 1 mm margins | box energy 0.01–0.04 %, peak T ≤ 0.07 % |
| Energy, adiabatic box, latent 0 vs 2.7e5 | stored energy equal to 2.5e-8 (print limit) |
| `stretch 1.0` (12.5 M cells) vs `1.1` (1.2 M), 0.5 mm margins | box energy 1e-4, peak ≤ 0.08 % |
| `stretch 1.25` (default) / `1.5` vs `1.1`: ThinWall, LargerRectangle 2 ms | molten sites +0.16 % / +0.7 %, mean pool \|ΔT\| 0.3 / 1.1 K, ThinWall wall identical; loop 1.66× / 2.1× (ThinWall), 1.34× / 1.67× (LargerRectangle) |
| np 1 / 2 / 4 (4×1×1 and 2×2×1) | per-site d6 identical (strict build; fast-math since the tiled kernel: ~1e-11 K) |
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

After fixing those (am_finitedifference 137d5948, worktree
`/home/tron/spparks-am-finitediff`, `FD_FIXES.md`), am/finitediff at the
same inputs gives ThinWall 60 × 25 µm, as the conduction source does. Both
codes' example decks now use inputs re-tuned to the original ThinWall pool
(150 W absorbed; sigma 42 µm in x/y and 70 µm in z, i.e. s = 59.4 / 99.0 µm
here):

| Deck, 150 W | Solid above substrate | Pool width / melted depth |
|---|---|---|
| ThinWall, am/finitediff (fixed) | 63,662 | 100 / 60 µm |
| ThinWall, conduction, matched (no margins) | 62,448 | 100 / 60 µm |
| ThinWall, conduction, `in.thin_wall` (1 mm margins) | 62,468 | 100 / 65 µm |
| LargerRectangle, am/finitediff (fixed) | 570,300 | depth 75 / 95 µm (median / max) |
| LargerRectangle, conduction, `in.large_rectangle` | 597,200 | depth 110 / 140 µm |

On LargerRectangle the 20 rasters at 50 µm hatch build up heat; the margins
(powder outside the box) keep it where am/finitediff's fixed 300 K box faces,
150 µm from the raster turns, remove it. The conduction rows ran at
`stretch 1.1` (before the 1.25 default).

**Inter-hatch pause.** With the original 2e-7 s jumps, layer 1's melted depth
grows from 65 µm (raster 1) to 110 µm (raster 20) and the peak from 2,356 to
3,781 K. A pause lets each raster start cold (`jump_time`, FP32, layer 1):

| jump_time | depth raster 1 → 20 | peak raster 1 → 20 |
|---|---|---|
| 2e-7 s | 65 → 110 µm | 2,356 → 3,781 K |
| 1 ms | 65 → 100 µm | 2,356 → 2,877 K |
| 2 ms | 65 → 85 µm | 2,356 → 2,627 K |
| 5 ms | 65 → 75 µm | 2,356 → 2,418 K |

`in.large_rectangle` now uses 5 ms with `sts 128`; RKL2 covers the pauses.

**One tall box vs stitched boxes (20 layers, 5 ms pauses).** The two setups:
- **Tall box:** one 360 × 250 × 136-site box takes all 20 layers through
  `laser_path file ... layers 20`.
- **Stitched boxes:** 20 boxes, one per layer. Each spans 36 sites below its
  layer top and is chained through `set ... stitch`/`dump stitch`.

Both use `plate_top 92.5e-6` and `part` (plate ∪ scanned footprint), FP32.
Resolidified = i2 3 and G > 0; grains = 26-connected same-spin
components. Runs: `/home/tron/lr_runs/layers` (`gen.py`, `compare.py`).

| | Tall box | Stitched |
|---|---|---|
| resolidified sites | 8,443,816 | 8,438,831 |
| grains (whole / central crop) | 300,075 / 186,598 | 337,355 / 186,189 |
| vol-wt grain size, crop | 3,984 | 4,240 |
| <001> ∥ z within 15°, crop | 0.188 | 0.185 |
| G quartiles, crop (K/m) | 8.95 / 10.1 / 11.3 e6 | 8.98 / 10.1 / 11.3 e6 |
| V quartiles, crop (m/s) | 0.734 / 0.844 / 0.977 | 0.740 / 0.850 / 0.983 |
| melt depth per layer, median / max | 70/75 → 80/90 µm | 70/75 → 80/90 µm |
| loop time | 5,941 s | 3,161 s (20 × 107–177 s) |

Per-layer melt depth agrees within one 5 µm cell on every layer, and the
central crop agrees in every measure. The 12 % extra grains in the stitched
build are all in the 60-site wall strips (+28 % grains there). They come from
the deck set-up, not from stitching. The stitched decks give powder a random
spin per site (the GP700 fix-ups), the tall deck gives all powder spin 42.
Wall sites that grow epitaxially from powder (32,606 sites in the stitched
build) become many small grains in one case and join one spin-42 network in
the other. Use the same powder initialisation in both before comparing wall
grain statistics.

**Box depth in stitched builds.** Sites below a box cannot melt, so a pool
that reaches the box bottom is silently cut off. Pools change with layer
(here 70 → 80 µm median over 20 layers; more on thin walls or above hot
material), so a fixed depth is a guess. Two guards:

- **`floor_check d`** (additive/texture/kk app command, metres). Each step
  the app takes the lowest liquid site and stops with an error if it is
  within `d` of the box bottom. Because it stops before the box's stitch
  dump, the stitch file is unchanged and the box can be rerun deeper. It also
  prints `floor_check: deepest liquid z = …` whenever the pool reaches a new
  deepest site. Off by default.
- **Adaptive window in the drivers** (`examples/am/surge/conduction/stitch_layers.py`,
  `examples/am/surge/gp700_moser/run_layers.sh`).
  - After each box, the next window is max(min window, ⌈F × deepest melt
    below the top⌉ + floor), with F = 1.5 by default.
  - A box stopped by `floor_check` is rerun 1.5× deeper.

Identity is checked on `%g` text dumps (6 significant digits). GP700 runs and
`compare.py`: `/home/tron/gp700_runs/cond_cmp`.

**Precision.** The stage kernel is FP64-bound on gfx1151 (about 20 FP64
divisions per cell). Built with `-DKOKKOS_PREC=mixed` (`build-fp32`) the
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

Cost: `in.single_track` (0.8 M cells, about 4,000 sub-steps) runs in about 4 s
on one gfx1151. With RKL2 the remaining cost is mostly the explicit steps
while the laser is on.

**DED** (2026-10-26, `in.thin_wall.ded`, FP64; i2 = 3 sites above the
substrate after four layers; median wall width per z slice):

| Deck | Solid above substrate | Remelted substrate | Wall width |
|---|---|---|---|
| `in.thin_wall` (powder bed) | 36,474 | 412 | 12 sites (60 µm) |
| `in.thin_wall.ded` (r = 2 sx, T_birth T0) | 36,364 | 417 | 12 sites |
| `ded T_birth 1700` | 74,324 | 947 | 20 = the whole bead slab |
| `ded radius 59.4e-6` (1 sx) | 27,790 | 276 | 8–10 sites |

- **DED vs powder bed: almost no difference.** The bead's powder heat sink is
  present in both. The LPBF powder at k = 0.3 conducts almost nothing
  sideways, so swapping it for gas surfaces barely changes the pool. DED's
  thermal signature comes from the added mass, which LPBF powder also
  carries. A side-surface difference would need a hotter, taller wall, or
  real convection from the nozzle jet.
- **`T_birth` matters.** Powder born at 1700 K melts the whole slab: the
  bead layout then caps the wall.
- **The birth radius matters.** Beam energy that falls on unborn cells is
  lost, so 1 sx (37 % of the Gaussian outside the radius) shrinks the
  pool. Keep the radius at 2 sx or more.
- **Determinism.** Per-site d6 is identical (all 5 frames of the last
  layer) for HIP np 1 vs 2×2×1, OpenMP np 1 vs 2×2×1, and HIP vs OpenMP.
- **No regression.** The decks without `ded` (`in.single_track`,
  `in.thin_wall`) give stats and dumps bit-identical to the pre-change
  binary.

## Possible next steps
- Keyhole lobes and fluctuations: sample the second lobe and `fluct_*` per sub-step.
- Rebalancing margin cells across ranks.
- Carrying thermal state across stitch boxes. Today each box starts at `T0`.
