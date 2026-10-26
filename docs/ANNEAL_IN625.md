# Heat treatment of AM IN625: `potts/anneal`

Status 2026-10-25: v1 implemented (`potts/anneal`, `potts/anneal/kk`): recovery,
recrystallization by nucleation and growth, grain growth, on an additive/texture
state. Not yet: a furnace schedule (time-temperature), annealing twins, calibration.
User manual: `doc/app_potts_anneal.txt`.

## Goal

Simulate stress-relief and full recrystallization (solution) anneals of as-built
IN625 from `additive/texture` output, where each site has its own orientation.
Schedules still to be taken from the literature; common for AM IN625: stress relief
around 870 C / 1 h, solution or recrystallization anneal around 1150 C. A stress
relief should give recovery with little or no recrystallization; a full anneal
recrystallization, then (later) twins and grain growth. Recovery during a stress
relief should delay recrystallization in a later anneal.

## Model

- Grains are labels (i1); same label = no boundary, so in-grain orientation
  gradients survive. Boundary energy: Read-Shockley on the disorientation.
- Stored energy H (d5) from the solidification conditions: PDAS
  lambda = K G^-1/2 V^-1/4 (Hunt, K = 428.44 as spparks-micro-analysis), then
  rho = rho0 + theta_c/(b lambda), H = Hmax rho/rho_ref (`stored_energy pdas`).
- One Metropolis test on dE = d(boundary) + H_j - H_i (as `potts/rx`), optional
  Humphreys mobility. Recovery dH/dt = -r H. Nucleation: seeds as `potts/rx`
  (H >= Hmin, optionally on boundaries), grown `radius` dilation passes, random
  orientation from a 65536-entry table.

## Scaling H: the window on sc/26n (step 0)

Physical stored energy at AM voxel sizes is far above the boundary energy per
voxel (E_v ~ 1 MJ/m^3, gamma ~ 0.6 J/m^2, a = 2 um: H/J ~ 9 E_v a / gamma ~ 30,
past the Eden bound 24). The user chose to scale H into the range where the Potts
front behaves, rather than refine the lattice. Measured with `potts/rx`
(scratchpad `hwin/`), sc/26n:

Flat {100} front, velocity (layers/MCS) and front area / flat area:

| kT | H=1 | 2 | 3 | 4 | 6 | 8 |
|---|---|---|---|---|---|---|
| 0.5 | 0 | 0 | 0 | 0.006 | 0.167 (1.44) | 0.327 (1.84) |
| 2.0 | 0.006 (1.16) | 0.058 (1.33) | 0.111 (1.45) | 0.177 (1.56) | 0.278 (1.77) | 0.359 (1.95) |
| 2.5 | 0.023 (1.35) | 0.081 (1.46) | 0.134 (1.55) | 0.193 (1.65) | 0.287 (1.82) | 0.367 (2.00) |

v/H at kT 2 is 0.037-0.046 for H 3-8, at kT 2.5 0.040-0.048 for H 2-8. Growing
sphere (R0 = 12): at kT 0.5-1 it facets into a cube and stops for H ~ 1.5-4
(lattice pinning); at kT 2-3 dR/dt is close to linear in H with the capillary
zero near H ~ 1.8 (critical radius ~ 22/H). In a polycrystal at kT 2 boundaries
are only thermally rough (boundary area within ~5% of kT 1, ~10 one-site specks
in 64^3).

So: kT ~ 2, H scaled into ~2-8. Consequences:
- Curvature is exaggerated against stored energy by the scale factor: nuclei must
  be larger than physical (`radius`, ~4 for H 3-6), and recrystallization and
  the grain growth after it need separate time calibrations.
- With grains only a few sites across, boundary fluctuations at kT 2 carry H
  between deformed grains (each flip copies H) faster than nuclei grow. AM
  grains are tens of voxels, but check H-field homogenisation on real states.

## Label merging (`merge theta N`)

On the ct15 crop, stress relief then anneal (2026-10-25), the accepted-flip rate
doubled over the last 100 MCS. Five label pairs of nearly one orientation (0.35-1.9
deg, Read-Shockley ~0.1-0.4 per bond against kT 2) intermixed; their interface grew
from 0.1% to 51% of all boundary bonds by t 250. Most of these pairs were not in
contact at load (they meet as the grains between them shrink, and copying
orientations on flips lowers their disorientation), so a one-off merge at load
caught 2 of 5. `merge 2.0 10` (majority of the pair's bonds below theta, union-find
on rank 0, every 10 MCS): no boundaries below 5 deg remain, the boundary area falls
monotonically, flips end at 306k/10 MCS vs 1.75M, run 40 s vs 32 s on HIP.
Analysis: /home/tron/anneal_runs/ct15_keyed_voids/diag/{bonds,pairs}.py.

## Reproducibility

`sweep color/strict`: CPU strict == kk OpenMP, bit for bit, any np. HIP (and
fast-math CPU) differ in the last bits of atan2/log/exp, so trajectories differ;
HIP is np-independent and statistically equivalent (6 seeds, examples/potts_anneal).
-fno-fast-math -ffp-contract=off on the two sources did not close the gap.

## Next

1. Furnace schedule: T(t) piecewise linear; Arrhenius maps MCS to seconds
   (boundary mobility), recovery rate and nucleation rate per second. Separate
   RX and grain-growth time constants (see scaling above).
2. Load a real additive/texture state (set ... stitch i1 i2 d1-d4 d8 d9), check
   the H field and the recrystallized structure; performance at production size.
3. Annealing twins (Sigma3 growth accidents) once recrystallization is calibrated.
4. Calibration: the user's IN625 data at a couple of conditions + literature
   (recrystallized fraction vs time, grain size, hardness for rho).

## In-grain energy (`energy disorientation`)

With labels, a grain's orientation scatter has no energy and no way to recover, and a
flip copies a neighbour's orientation: on the ct15 crop, SR then anneal swept 89% of
sites by deformed-grain growth, leaving clean rims (KAM 0.14 deg) around untouched
noisy cores (0.9-2 deg). The swept rims are physical (a migrating boundary leaves the
consumed dislocations behind); the frozen cores are not. `energy disorientation` puts
Read-Shockley energy on every bond and lets a site copy a same-label neighbour's
orientation (keeping its H), so the scatter coarsens into subgrains everywhere.
At kT 2 a 1 deg bond is nearly free, so `mobility` sets the rate. Median KAM (5 um face
neighbours), start 0.90 deg, r 0.15 for 10 MCS then the 1150 C settings for 250:

| energy, mobility | after SR | after anneal | sites > 1 deg after anneal |
|---|---|---|---|
| labels | 0.83 | 0.20 | 19% (the cores) |
| disorientation, 15 | 0.61 | 0.19 | 4% |
| disorientation, 5 | 0.52 | 0.21 | 2% |
| disorientation, M = 1 | 0.28 | 0.08 | 0% |

Neither mode recrystallizes after this SR (H below Hmin), and the deformed grains
still sweep ~90% of the volume; that is the stage settings, not the energy rule.
