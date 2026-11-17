# Free-surface melt-pool flow: theory and implementation

`flow lbm ... free_surface` (2026-11-15 on). This note explains the model
and how the code carries it out. The flow takes three steps per lattice
time step: the **surface**, the **lattice Boltzmann step** and the
**conversion**. Each step has its own section below, giving the theory, the
code, and what it leaves out. The conduction coupling and several-rank
operation come after them.

Code:
- `src/KOKKOS/melt_flow_kokkos.h`: `MeltFlow<R>`, which is standalone and
  free of SPPARKS and MPI. It provides `surface()`, `step()`, `convert()`,
  `inject()` and `melt_advect()`.
- `src/KOKKOS/conduction_solver_kokkos.cpp`: `flow_step()` holds the
  thermal coupling, the powder feed and the write-back. `fs_surface_z()`
  sets where the laser lands.
- `tools/melt_flow_check.cpp`: gates 14-21 cover the free surface.

Developer notes on the flat-lid flow it extends:
`docs/CONDUCTION_SOURCE.md`, "Melt-pool flow". User syntax:
`doc/temperature_source_conduction.txt`.

## 1. The model in one paragraph

The method is a single-phase free-surface lattice Boltzmann method (Körner
et al. 2005; Thürey 2007). The liquid is resolved and the gas is not: gas
enters only as a pressure on the surface. Every cell is one of four types:

| Type | Holds distributions | Mass | Fill ε |
|---|---|---|---|
| `FLUID` | yes | its density ρ | 1 |
| `INTERFACE` | yes | `cmass`, between 0 and ρ | m / ρ |
| `GAS` | no | 0 | 0 |
| `WALL` (solid metal) | no | — | — |

A closed layer of `INTERFACE` cells always separates `FLUID` from `GAS`.
Mass moves across the links between liquid cells. A cell that fills
becomes `FLUID` and a cell that empties becomes `GAS`, with the interface
layer moved to follow. Surface tension and Marangoni stress come from a
**height function**: one metal height per (x, y) column. The thermal
solver decides what is liquid or solid, and the flow decides where the
liquid is.

## 2. Units, the sub-step loop and its step size

Inside `MeltFlow` everything is in lattice units (cell = 1, step = 1,
density = 1). `run(h, lo, hi)` advances by one thermal step h in n flow
sub-steps (n even):

```
run(h):                                    per thermal step (flow_step)
  choose n, rescale f if dt changed
  repeat n times:
    1. surface(lo, hi)        FlowFsColumns, FlowFsSurface         (2-D)
    2. step(lo, hi)           FlowStep                             (3-D)
       post():   melt_advect  FlowAdvect x3 (+ copy)               transport
                 inject       FlowFsInject                         powder
    3.           convert      FlowFsMark, FlowFsRetype, FlowFsExcess
  every 16th call: max |u| (FlowMax), run-away test
```

The caller's `post` hook replaces a bare `convert()`. The carried fields,
specific enthalpy and composition, must move with the step's mass fluxes
before conversion redistributes mass. In `tools/melt_flow_check` there is
no hook and `run` calls `convert()` itself.

**Step size.** n is the larger of two counts:

- **Speed bound.** The fastest cell moves about 0.1 cell per step at
  `u_cap`. `u_cap` starts at `u_ref`, doubles when the measured maximum
  passes 1.5 `u_cap`, and halves below 0.3 `u_cap`.
- **Capillary bound** (fs only):
  $$\Delta t \le c_\sigma \sqrt{\rho\,\Delta x^3/\sigma_{\max}},\qquad c_\sigma = \texttt{fs\_cap} = 0.04,$$
  which keeps lattice σ ≤ c_σ² = 1.6e-3. Without the bound, the shortest
  capillary wave (two cells) travels at about sqrt(πσ), close to the
  lattice speed of sound, and the surface breaks up.
  - Cap values tried: 0.18 grew pits, 0.08 grew a two-cell checkerboard on
    a hot Ti-6Al-4V pool, and 0.04 was calm.
  - σ_max is `sig_rho`: the constant σ, the largest value in the
    properties table, or the largest value in `sigma_table`.

In the 316L DED decks the capillary bound sets n: 26 flow steps per thermal
step. When dt changes, `rescale()` scales f as follows:

- velocity by s;
- pressure deviation by s²;
- non-equilibrium part by s;
- interface mass by the density ratio, so the fill is unchanged.

## 3. Step 1: the surface (`surface()`)

### Theory

**Height function.** For each column (i, j) the metal height is

$$h(i,j) = k_f + 1 + \sum_{\text{interface cells above}} \varepsilon,$$

where k_f is the top full cell (FLUID or WALL) under the column's interface
stack.

**Capillary pressure.** The mean curvature of the graph z = h(x, y) is

$$\kappa = \nabla\cdot\frac{\nabla h}{\sqrt{1+|\nabla h|^2}}
= \frac{h_{xx}(1+h_y^2) + h_{yy}(1+h_x^2) - 2h_{xy}h_xh_y}{(1+h_x^2+h_y^2)^{3/2}},$$

which is negative on a hump. Young–Laplace puts the gas side of the surface
at p_gas − σκ. In LBM pressure is ρ c_s² with c_s² = 1/3, so the gas
density used in step 2's reconstruction is

$$\rho_G - 1 = 3\,(p_\text{gas} - \sigma\kappa)\qquad(\text{lattice units}).$$

`pgas` is a per-column field the caller may set, for example to a recoil
pressure. It is zero today.

**Marangoni stress.** The tangential surface force per unit projected area
is $\sqrt{g}\,\nabla_s\sigma$, with g = 1 + h_x² + h_y². Written in the
tangents (1, 0, h_x) and (0, 1, h_y):

$$\mathbf f_s = \frac{\big[(1+h_y^2)\sigma_x - h_xh_y\sigma_y\big](1,0,h_x) + \big[(1+h_x^2)\sigma_y - h_xh_y\sigma_x\big](0,1,h_y)}{\sqrt{g}}$$

σ_x and σ_y are differences of σ between neighbouring columns. On a flat
surface this reduces to the flat-lid model's force.

### Implementation

**`FlowFsColumns`** is a 2-D kernel over the held columns. It walks down
from the window top past GAS and records, per column:

| Field | Meaning |
|---|---|
| `skt` | the top metal cell. Set to −2 when there is no metal and −1 when solid is on top (no liquid surface) |
| `skb` | the lowest cell that carries the surface force: the FLUID cell under the stack, or the stack's bottom |
| `hcol` | h |
| `scol` | the column's σ source value: T, or σ(T)/ρ from `sigma_table`, or `sig` under `properties cards` |
| `slcol` | the lattice σ of the surface cell |
| `mcol` | the column's mass from `skb` to `skt` (at least 0.5), times 1 + dens |

A column whose stack hangs over gas, such as liquid left on a frozen rim,
gets `skt` = −1 and no surface force: one height cannot describe it.

**`FlowFsSurface`** is a 2-D kernel over the columns the step needs (one
more around the window). It does three things:

1. **Finite differences of h.** Central differences over 3 × 3 columns give
   h_x, h_y, h_xx, h_yy and h_xy, then κ, then `drg` = 3(pgas − σκ).
2. **Neighbours without a liquid surface** (`Hr`). Such a neighbour is
   given an effective height:
   - No metal, or solid standing higher: the column's own height. This
     mirrors the liquid (a right angle), and it also applies at the block
     edge.
   - Dry solid standing lower (the frozen rim under a bead):
     max(h_c − tan θ, h_n) with θ = `fs_angle` = 20°. The liquid meets the
     rim at the contact angle and is not pulled over it.
     - At 0° beads crept over the rim and hung in the air.
     - At 45° the pool humped and broke away, throwing about 600 cells of
       spatter.
     - At 20° the 316L bead was dense, round and continuous.
3. **Surface force.** `sax`, `say` and `saz` are f_s / `mcol`. Step 2
   applies this as one acceleration to every cell from `skb` to `skt`, so
   a nearly empty interface cell is not flung. σ differences are one-sided
   beside a column without σ: with no metal for T, and with no liquid
   under `properties cards`.

**Smoothing.** `fs_raw` (default 1) blends the height with a 1-2-1 × 1-2-1
smoothed height before the forces are computed. It is off by default.
Smoothing allowed a larger step, but it removes the restoring force from
one-cell pits, which then never heal.

**Cost:** 3% of a late-run 316L track (Columns 1.6%, Surface 1.4%).

**Limits** (`ponytail:` in the code). One height per column means no
overhangs, keyholes, balling past 90°, or droplets over gaps. The fix for
those would be local height functions along the normal.

## 4. Step 2: the lattice Boltzmann step (`step()`, kernel `FlowStep`)

### Theory

The lattice is D3Q19 with populations stored as f − w, so that single
precision keeps low-Mach velocities. Collision is TRT (two relaxation
times):

- the symmetric part relaxes at 1/τ, with τ = 3ν + ½;
- the antisymmetric part relaxes at the rate fixed by the magic parameter
  Λ = (τ − ½)(τ_a − ½) = 3/16, which puts a bounce-back wall exactly
  half-way between the cells for any ν.

**Other terms in the collision:**

- **Forcing.** Guo forcing with the half-force velocity
  u = (m/ρ + a/2)/(1 + D/2). The implicit Carman–Kozeny drag is
  D = darcy·(1 − f_l)²/(f_l³ + 1e-3).
- **Eddy viscosity.** Optional Smagorinsky term from the non-equilibrium
  stress.
- **Body force.** Buoyancy is β g (T − T_ref), or −g·dens under
  `properties cards`.

**Free-surface additions:**

1. **Gas links are reconstructed, not reflected** (Körner 2005). A
   population that would stream from a GAS cell into liquid cell x along q
   is replaced by
   $$f_q(\mathbf x) = f^{eq}_q(\rho_G,\mathbf u) + f^{eq}_{\bar q}(\rho_G,\mathbf u) - f_{\bar q}(\mathbf x),$$
   where **u** is the cell's own velocity from the last step and ρ_G comes
   from step 1. In f − w storage the weights cancel, so the term is
   2w_q[δρ_G + ρ_G(4.5(c·u)² − 1.5u²)] − f_q̄, and mixed precision keeps
   working. This is the "only missing populations" variant: every link
   from gas is rebuilt, none from liquid.
2. **Mass exchange.** On each link with a liquid neighbour n, the cell
   gains
   $$\Delta m_q = \phi_q\,\big(f_q^{\text{in}} - f_{\bar q}^{\text{out}}\big),\qquad
     \phi_q = \begin{cases}\tfrac12(\varepsilon_x+\varepsilon_n) & \text{both INTERFACE}\\ 1 & \text{otherwise}\end{cases}$$
   An INTERFACE cell's mass becomes m + Σ Δm_q. A FLUID cell's mass is its
   density.
3. **Partial cells take their share of the force.** Every acceleration in
   an INTERFACE cell is multiplied by clamp(ε, 0, 1). Then the pressure
   under the surface follows the surface height within a cell layer, and
   the column's cells together take the whole surface force.

### Implementation

The kernel is pull-form: one thread per cell gathers what arrives along
each of the 18 links. A link's handling depends on the source cell type
and, for diagonals, on the two cells between:

| Source of the link | Treatment | Mass exchanged |
|---|---|---|
| liquid (FLUID or INTERFACE), axis link | streamed | φ (f_in − f_out) |
| liquid, diagonal, at least one cell between is liquid | streamed; the face flux goes through the liquid cell(s) between, half each way if both are liquid | φ (f_in − f_out) |
| liquid, diagonal, both cells between are not liquid, one is GAS | reconstructed from gas (no tunnelling through a gas corner) | none |
| liquid, diagonal, both cells between are WALL | bounce-back | none |
| GAS | reconstructed from gas | none |
| WALL or the block edge | half-way bounce-back | none |

**Details:**

- **Face fluxes.** `flux(i, j, k, face)` tallies the mass leaving through
  each of the six faces, with the same φ weights. Summed over a cell's
  faces, it equals that cell's change of mass, which is what lets
  `melt_advect` conserve Σ m q through partial cells.
- **Interface viscosity floor.** τ is raised to at least `fs_tau` = 1.0 in
  interface cells. At a metal's τ ≈ 0.51 they are unstable under Marangoni
  forcing: the gate-21 pool broke up at a third of its forcing, was rough
  at 0.6 and held at 1. The cost is a known bias: the surface layer is
  viscous, so it is dragged along with the cell under it and the surface
  speed reads low.
- **Voids close.** An interface cell below its column's `skb` (a bubble
  wall, or the underside of a void) sees ρ_G lowered by 6σ, a one-cell
  bubble's Laplace pressure of 2σ/r with r = 1. Without this, voids
  trapped in a large 316L pool neither closed nor grew; they merged until
  a slot opened through the pool and it collapsed (Sattari deck, 380 ms).
- **Two-step mean.** `ux`, `uy` and `uz` are means of the last two steps.
  Bounce-back conserves a staggered momentum that no collision damps; the
  mean cancels it. `px`, `py` and `pz` are the last step's velocity, used
  for reconstruction and by step 3.
- **No local scratch.** The velocity loops carry `#pragma unroll`, so the
  arrays stay in registers on HIP. Without it the kernel spilled 400-848 B
  per lane (6d5b21fa).

**Cost:** 49% of a late-run 316L track, the largest single kernel. Each
cell reads 19 populations plus types and fills of 18 neighbours and writes
19 populations and 6 face fluxes.

**Limits:**

- φ uses Thürey's plain mean fill; his neighbour-count weights (which
  favour mass moving toward the bulk) are not in.
- Reconstruction covers populations from gas only. The normal-based
  variant (also rebuilding populations whose direction points into the
  liquid) was not needed for the gates.
- Inertia uses one density (Boussinesq), with buoyancy from the real
  densities.

## 5. Between steps 2 and 3: transport and feed (the `post` hook)

**Transport (`melt_advect`).** The carried fields are specific enthalpy q
(J/m³ of the cell's material, absolute) and composition x. Both move with
step 2's face fluxes, so they travel with exactly the mass that moved.

- The scheme is three 1-D van Leer sweeps (x, y, z) in conservative form:
  (mq)' = mq − Δ(F q_f) with m' = m − ΔF. Σ m q is conserved to round-off
  and a uniform q stays uniform.
- The cell mass is `cmass`; step 2 sets it to ρ in FLUID cells.
- **A face beside an INTERFACE cell is first-order upwind**, because a
  partial cell can pass more than half its mass in one step.
- **An INTERFACE cell takes inflow only:**
  q' = q + F_in (q_n − q)/m', clamped between the values being mixed. The
  cell can hold next to nothing.

**Powder (`inject`).** This applies while the laser is on.

- Each column receives `dm(i, j)·dt/h` cells of liquid. `dm` is built once
  per thermal step by `FlowFsPowder` as a Gaussian of 1/e radius
  `powder_radius` about the laser axis.
- **Capture rule:** only where the column's surface cell is fully liquid
  (f_l ≥ 1). Powder on mush or solid is not caught: otherwise the cold rim
  piled up as a mush that never flowed.
- The mass goes into the **lowest** interface cell of the column's stack.
  Feeding the top cell built an empty cell over a partial one: a two-cell
  foam.
- q and x mix mass-weighted with the feed's values (`fqfed`: enthalpy at
  `powder T`; `fXfed`: the composition set on that cell in the deck).
- An overfilled cell passes its excess on in step 3.
- `sgot` tallies what was caught per column, on the device. One reduction
  per thermal step reads it.

**Cost:** FlowAdvect 12%.

## 6. Step 3: conversion (`convert()`)

### Theory

A cell converts when its mass leaves [0, ρ]:

- **Filled** (m > ρ): it becomes FLUID. Its GAS neighbours become INTERFACE
  so that no FLUID cell touches GAS.
- **Emptied** (m < 0): it becomes GAS. Its FLUID neighbours become
  INTERFACE.
- **Excess.** The converted cell's mass beyond its new state goes to the
  surrounding interface cells: m − ρ when filled, m (≤ 0) when emptied.
  Mass is conserved exactly.

A hysteresis κ = `fs_kappa` = 1e-3 (fill above (1 + κ)ρ, empty below −κρ)
stops cells from flipping back and forth on round-off.

### Implementation: three kernels in gather form, so there are no atomics

The kernels run over step 2's window grown by one cell (cells one beyond
can become INTERFACE).

1. **`FlowFsMark`** writes `mark` = FILL / EMPTY / 0 for each interface
   cell.
   - FILL if m > (1 + κ)ρ.
   - EMPTY if m < −κρ, or if the cell is an **isolated droplet**: no FLUID
     or WALL neighbour. Nothing would move its mass, and it would hang
     falling for ever.
   - **Conflict rule:** no cell empties beside a cell that fills (the
     filled cell would then touch gas). Fill wins.
   - **Spatter:** interface cells in the top four layers of the block
     empty unconditionally. The liquid leaves the domain. A filling cell
     there once let a whole droplet through.
2. **`FlowFsRetype`** writes the new types into `type2`; `type` is
   untouched.
   - Marked INTERFACE cells go to FLUID or GAS.
   - A GAS cell next to a FILL becomes INTERFACE. Its f is the equilibrium
     of the mean ρ and u of its liquid, non-emptying neighbours, and it
     starts at mass 0.
   - A FLUID cell next to an EMPTY becomes INTERFACE at full mass.
   - An emptied cell's f, u and fluxes are cleared.
3. **`FlowFsExcess`** writes the new types, masses, fills and scalars.
   - Each cell that stays or becomes INTERFACE gathers from every marked
     neighbour an **equal share**, excess / (number of that neighbour's
     neighbours that are INTERFACE in `type2`).
   - It mixes q and x mass-weighted, clamped to the range of the values
     mixed.
   - It sets ε = m/ρ.
   - Only cells that stay or become interface are written and only
     converted cells are read, so there is no race.
   - A converted cell with no interface neighbour cannot pass its excess
     on. That mass is tallied in `slost` (per column, counted on owned
     cells only) and reported as "droplets lost". The coupling returns it
     through the reservoir (section 7).

**Cost:** 11.6% (Retype 5.4%, Excess 4%, Mark 2.2%). Retype and Excess each
scan 18 neighbours, and Excess re-counts each neighbour's interface
neighbours.

**Limits:**

- Thürey distributes excess weighted by the surface normal (more to the
  liquid side). Equal shares are simpler and conserve mass the same; the
  normal-weighted form could reduce small-scale surface noise.
- Spatter leaves rather than landing again.

## 7. Coupling to the thermal solver (once per thermal step, `flow_step`)

The thermal grid and the flow block share the lattice box's cells. The
order is:

1. **Window.** The flow holds the liquid's bounding box plus 3 cells and a
   margin (`FlowWin::move`), the same box on every rank.
2. **`FlowFsTypes`** sets the new flow type from the previous one and this
   step's T. The thermal side owns the solid/liquid decision:
   - A dense cell above its solidus (at its own composition, interpolated
     between bins) becomes FLUID.
   - A liquid cell that freezes becomes WALL.
   - **An interface stack decides as one** and reads the T and H of its
     host, the full cell below the stack. When the host has frozen:
     - **Capped** (a full cell above the stack): the whole stack becomes
       solid. Left as a film it is gas to the stencil, and it insulated a
       bead from the plate (2,000 K liquid one cell above plate at
       1,100 K).
     - Otherwise it **waits** while any liquid cell beside the stack is
       still above the lowest solidus. Settling at once ate the pool's
       front.
     - Then **settled** (two or more cells deep): the stack's mass fills
       it from the bottom.
     - Or **single** (one cell deep): the cell rounds.
     - In both cases a cell at least half full becomes solid and one less
       than half full becomes gas.
   - **Freeze reservoir.** Each rounding error goes to `fs_res`, reported
     as capped / settled / single. The powder feed pays it back: the next
     thermal steps' feed is scaled by up to ±50%.
3. **`FlowFill`** writes T, liquid fraction, composition and, under
   `properties cards`, `sig`, `dens` and `visc` per cell. A frozen
   interface cell that is still waiting gets f_l = 1e-3, so the mushy drag
   holds it still.
4. **`fs_repair`**: a FLUID cell touching GAS becomes a full INTERFACE
   cell. This covers cells that just melted next to gas.
5. **Powder field** (`FlowFsPowder`; section 5).
6. **`F.run(h)`**: n × (step 1, step 2, transport, feed, step 3).
7. **Tallies.** One reduction gives what was caught and what was lost.
   Droplets lost go to the reservoir. Spatter bursts are logged with the
   time, length, laser position and mass.
8. **Write-back (`FlowFsOut`):**
   - **Interface cells are gas to the conduction stencil.** Their host and
     every interface cell stacked on it share one specific enthalpy, the
     mass-weighted mean.
   - The host carries the stack's heat capacity: it is flagged `PART` with
     `Cap` = 1 + Σ(stack mass), and the stage kernel divides its enthalpy
     change by `Cap`. Before, only the cell right on the host shared;
     powder fed higher up stayed cold and froze as a porous mound under a
     liquid skin.
   - A cell that filled becomes DENSE, taking the material bin of its
     composition.
   - A cell that emptied within the step becomes GAS. Left as metal, the
     next step would make a cell of liquid from nothing.
9. **`fs_surface_z`.** The laser's z profile hangs from each column's top
   full cell (`zsf`), not a fixed plane, so all absorbed power is
   deposited, including on bare plate ahead of the bead. The conduction
   sweep stops two cells above the highest metal (`kz_end`). Metal within
   two cells of the lattice top stops the run.
10. **Sites (`phase_out`).** A site whose cell became metal goes from
    i2 = 5 to 2 (liquid; the app then solidifies it), and one whose cell
    emptied goes to 5.

**Conservation report** (`fs_report`, each time the laser switches off).
It reports:

- powder fed and caught (capture %);
- droplets lost;
- freeze rounding by kind;
- the reservoir;
- metal volume (full cells plus interface masses);
- metal enthalpy.

The balance is metal change − caught + reservoir change. On the 0.06 s
multi-rank test deck it is −1.22 cells on every decomposition. That offset
is not yet traced.

## 8. Several ranks (2026-11-17)

The block is split in x and y; z is not split. The flow block's halo is 2
cells deep and the conduction grid's is 1. Conversion's stencils reach two
cells, so the free surface exchanges more than the flat lid does. Each
flow sub-step:

1. Step 2 runs on owned cells (no f exchange here; `post` does it after
   conversion).
2. Exchange `flux`, ρ and u. Run `melt_advect`; its callback exchanges the
   carried fields before the x and y sweeps.
3. `inject` on owned columns. Exchange `cmass`, q and x.
4. `convert`; `fs_halo` exchanges `mark` after stage 1 and `type2` after
   stage 2.
5. Exchange type, ε, `cmass`, ρ, q, x and f.

**Per thermal step:**

- After `fs_repair` the same set is exchanged.
- The rounding tallies and the caught/lost tallies are summed over ranks.
- `FlowFsPowder` and `fs_surface_z` cover owned columns only. The second
  rule fixed a crash ("metal reached the top"), where the 2-deep flow halo
  was indexed through the 1-deep grid.

**Checks so far:**

- OpenMP, test deck 100 × 60 × 50, 0.06 s, decompositions 1, 1×2, 2×1 and
  2×2:
  - caught within 0.1%;
  - enthalpy within 1e-5 relative;
  - peak T within 1.8 K;
  - no seams in the dumps.
- `melt_flow_check` passes.

The HIP check is running. On CPU the extra ranks are slower (114 s → 189–234 s).

`ponytail:` about 15 exchanges per flow sub-step. Batching them into one
message per neighbour is the next lever if this runs at scale.

## 9. Where the time goes

Kernel shares, 316L DED track, late in the run (0.4 s, HIP, FP64):

| Part | Kernels | Share |
|---|---|---|
| Step 1, surface | FlowFsColumns, FlowFsSurface | 3.0% |
| Step 2, LBM | FlowStep | 49% |
| Transport | FlowAdvect | 12% |
| Step 3, conversion | FlowFsMark, FlowFsRetype, FlowFsExcess | 11.6% |
| Conduction | ConductionStep | 20% |

The flow is 78% of the run, and the capillary bound sets its step (26 per
thermal step).

**Levers, largest first:**

1. **A weaker capillary bound.** Implicit or semi-implicit surface tension,
   or applying σκ only on smoothed long waves while keeping short ones
   stable some other way. n scales as 1/c_σ.
2. **Fusing step 3's three kernels** and the transport sweeps, or launching
   the sub-step as a graph. Late in a run the cost is set by launch
   latency, not memory traffic.
3. **Mixed precision** (`KOKKOS_PREC`): about 2.2× on gfx1151. Production
   runs stay in double.

## 10. Validation

**Self-check gates** (`melt_flow_check`; OpenMP and HIP, double and mixed):

| # | Problem | Measured against |
|---|---|---|
| 14 | Flat pool at rest, surface mid-cell | stays still and flat; mass to 1e-12 |
| 15 | Dam break, many conversions | mass to 1e-12; no FLUID touching GAS |
| 16 | Dam break carrying a scalar | Σ m q to 1e-4; a uniform q stays uniform; in range |
| 17 | Capillary standing wave | period within 5%; decay 2νk² within 20% |
| 18 | Gaussian gas-pressure load | Young–Laplace depression within 5% |
| 19 | Thermocapillary slot with a moving surface | tilt 3τ/(2ρgh) within 5%; velocity profile |
| 20 | Injection into a basin | mass gained = fed; scalar mean as mixed |
| 21 | Ti-6Al-4V pool through `run()`, 20 ms | finite; lattice speed ≤ 0.25 |

Gates 17 and 19 also print their results with the interface τ floor on, to
show its bias.

**Against experiment** (Sattari 2024, 316L DED; LaCie `tinb_fs`):

| Case | Model | Measured |
|---|---|---|
| Round top-hat, bare plate: depth | 560 µm | 552 µm |
| Round top-hat, bare plate: width | 1.14–1.16 mm | 1.189 mm |
| Square beam, powder, absorptivity 0.35, 1.3 s: width | 1.62–1.66 mm | 1.513 mm |
| Same: bead height | 340 µm | 607 µm |
| Same: depth | 420–460 µm | 589 µm |
| Same: capture | 44% | — |

The square-beam beads are flat mesas whose toes never melted. About half
of the height shortfall is capture (about 61% would be needed) and half is
shape.

## 11. Known limits and simplifications

| Limit | Effect | Upgrade path |
|---|---|---|
| One height per column | no overhangs, keyholes or balling past 90° | local height functions along the normal |
| Contact angle at a dry lower rim only (`fs_angle`) | it is a calibration knob; it may depend on the material | wetting from the surface energies |
| Interface τ floor 1.0 | surface speed biased low | a stabilised interface collision |
| Capillary step bound 0.04 | sets the flow cost | implicit surface tension |
| Equal-share excess, plain-mean φ | small-scale surface noise | Thürey's normal and neighbour weights |
| Boussinesq inertia | approximate where Nb and Ti-6Al-4V mix (2× density) | variable-density LBM |
| No recoil pressure | probably sets depth at Nb-on-Ti-6Al-4V temperatures | `pgas` from the vapour pressure (the hook exists) |
| Spatter leaves at the top | mass leaves rather than landing again | a ballistic droplet model, if it matters |
| Powder caught only on fully liquid surface | sets capture with no particle physics | particle size and temperature |
| Unexplained −1.22-cell offset in the mass balance | small, the same on every decomposition | trace it |

## References

- C. Körner, M. Thies, T. Hofmann, N. Thürey, U. Rüde, "Lattice Boltzmann
  model for free surface flow for modeling foaming", J. Stat. Phys. 121
  (2005).
- N. Thürey, "Physically based animation of free surface flows with the
  lattice Boltzmann method", PhD thesis, Erlangen (2007).
- I. Ginzburg, F. Verhaeghe, D. d'Humières, "Two-relaxation-time lattice
  Boltzmann scheme", Commun. Comput. Phys. 3 (2008): Λ = 3/16.
- Z. Guo, C. Zheng, B. Shi, "Discrete lattice effects on the forcing term
  in the lattice Boltzmann method", Phys. Rev. E 65 (2002).
- P. Sahoo, T. DebRoy, M. J. McNallan, "Surface tension of binary metal —
  surface active solute systems", Metall. Trans. B 19 (1988): the
  `tools/surface_tension.py` model.
- waLBerla's free-surface module (GPL-3) was read as a reference only; no
  code was taken from it.
