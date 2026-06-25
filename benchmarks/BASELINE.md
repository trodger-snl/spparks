# Rosenthal / Moser MPI scaling benchmarks (kk app)

Baselines captured **before Phase 2** (selective microstructure sync), so the
Phase-2 speedup can be measured later by re-running `run_baseline.sh` and diffing
the loop times.

## What these measure

The per-step **microstructure ghost sync** (`sync_app_data_to_host` →
`comm->all` → `sync_app_data_to_device` for spin / active_flag / quaternions /
output arrays) at the tail of `AppAdditiveTextureKokkos::app_update`. That block
is **common to every temperature source** — Rosenthal, Moser, and HDF5 all incur
it — and is the dominant multi-rank per-step cost (the temperature fills are all
device-resident: positional over `nall`, no T round-trip / no T ghost comm).

Single rank is fast because the sync is gated to output steps (commit "Gate
per-step device→host sync to output steps"); multi-rank must exchange ghosts
every step, so the cost reappears and the runs **anti-scale**.

## Setup

- App `additive/texture/kk`, `sweep color`, lattice `sc/26n`, `dx = 5e-6`.
- Box `0 310 0 316 0 50` = **5,027,937 sites** (matches the HDF5 ct15 study for
  cross-source comparability). `boundary n n n`.
- Linear laser scan across x at mid-width / top plane, `speed 0.5 m/s`,
  `run 2.5e-3` (the full active-scan window; sustained melting throughout).
- Analytic sources — **no external data file**, fully portable.
- Note: Rosenthal is a point source, so on this fine `dx` it samples the 1/r
  singularity and reaches ~70,000 K near the beam. Irrelevant to the *sync-cost*
  timing (cost is per-`nall`, independent of T magnitude) and both sources use
  the identical box; Moser's finite-width Gaussian stays ~4,900 K. Do not read
  these as physically calibrated pools.

## Baseline numbers (this APU: Strix Halo, 1× gfx1151, shared with desktop)

SPPARKS "Loop time" (the run loop) and derived per-step. Moser fast-forwards its
early below-threshold steps (its field ramps from t=0), hence fewer steps —
compare per-step, not loop totals across sources.

| source    | ranks | loop (s) | steps | per-step (ms) |
|-----------|------:|---------:|------:|--------------:|
| Rosenthal |     1 |    5.05  |  500  |     10.1      |
| Rosenthal |     4 |   22.39  |  500  |     44.8      |
| Rosenthal |     8 |   21.11  |  500  |     42.2      |
| Moser     |     1 |   29.24  |  300  |     97.5      |
| Moser     |     4 |   40.75  |  300  |    135.8      |
| Moser     |     8 |   42.23  |  300  |    140.8      |

**Sync overhead (per-step, np4 − np1):** Rosenthal ~35 ms, Moser ~38 ms — both
match the HDF5 study's ~37 ms. This ~35–40 ms/step is what Phase 2 targets.

Moser's np1 per-step (97 ms) is dominated by the device Green's-function
quadrature, not the sync; the sync is the *delta* going multi-rank.

## How to re-run (after Phase 2)

```bash
. /home/tron/spack/share/spack/setup-env.sh && spack env activate spparks-dev
cmake --build build-verify -j 8          # build the post-Phase-2 binary
benchmarks/run_baseline.sh               # writes benchmarks/baseline_rm.csv
```

Compare the new per-step (np4/np8) against the table above. **Success = the
np4/np8 per-step drops toward the np1 value** (the sync overhead shrinks).

GPU-safety: runs are bounded and complete on their own. NEVER `kill -9` a running
spparks/mpirun — a stuck HIP kernel can peg the GPU and force a reboot. Use one
GPU job at a time.
