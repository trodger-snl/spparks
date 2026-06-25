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

## Phase 2 result (selective microstructure sync)

Commit "Phase 2: selective microstructure ghost sync". Same box, same APU.
Loop time, baseline (full sync) → Phase 2 (selective):

| source    | ranks | baseline | Phase 2 | speedup | per-step sync overhead (np4−np1) |
|-----------|------:|---------:|--------:|--------:|---------------------------------:|
| Rosenthal |     1 |   5.05 s |  5.10 s | —       | —                                |
| Rosenthal |     4 |  22.39 s |  5.46 s | **4.1×**| 34.7 ms → **0.7 ms** (~48×)       |
| Rosenthal |     8 |  21.11 s |  5.11 s | **4.1×**|                                  |
| Moser     |     1 |  29.24 s | 29.33 s | —       |                                  |
| Moser     |     4 |  40.75 s | 30.03 s | **1.36×** | 38.4 ms → **2.3 ms** (~17×)    |
| Moser     |     8 |  42.23 s |  (n/a)  | —       |                                  |

The per-step microstructure-sync overhead collapses from ~35–38 ms to ~1–2 ms.
Rosenthal goes from anti-scaling to **near-perfect scaling** (np1≈np4≈np8).
Moser's np1 cost is its Green's-function quadrature (untouched by the sync); the
sync was the multi-rank *delta*, now nearly gone. Correctness: SPK_VERIFY_GHOST
reported 0 mismatches (selective ghost state bit-identical to the full exchange);
d6 unchanged.

**Moser np8 is inconclusive on this box**: 8 expensive-kernel ranks oversubscribe
the single shared GPU, and the runs collided with active desktop GPU use (~94%).
Rosenthal np8 (identical sync code) scales perfectly, so this is a 1-GPU
hardware/contention limit, not the sync change — it should resolve on the 4-GPU
cluster (≤2 ranks/GPU) with proper rank→device binding (the remaining task).

## Backend comparison: OpenMP (16-core CPU) vs HIP (integrated GPU)

Same Kokkos code, same 5M-site box, `build-openmp` (Kokkos OpenMP) vs
`build-verify` (Kokkos HIP, Radeon 8060S APU). CPU configs use all 16 cores
(MPI ranks x OpenMP threads = 16). Loop time:

| source    | ranks | OpenMP CPU | HIP GPU | GPU speedup |
|-----------|------:|-----------:|--------:|------------:|
| Rosenthal | 1 (x16t) |  4.77 s | 5.10 s | 0.94x (CPU faster) |
| Rosenthal | 4 (x4t)  |  5.42 s | 5.46 s | ~1.0x        |
| Rosenthal | 8 (x2t)  |  5.57 s | 5.11 s | 1.09x        |
| Moser     | 1 (x16t) | 45.54 s | 29.33 s | **1.55x**   |
| Moser     | 4 (x4t)  | 47.56 s | 30.03 s | **1.58x**   |

On this **integrated APU** the GPU only beats the 16-core CPU for the
compute-heavy Moser quadrature (~1.55x) and ties for the memory-bound Rosenthal
fill -- because the integrated GPU shares memory bandwidth with the CPU, so its
only edge is raw FLOPs. (Same reason the CPU app beat the GPU on the HDF5 box:
HDF5 interpolation is memory-bound like Rosenthal.)

This **understates** the GPU on real cluster hardware: a discrete H100/MI300X has
far more FLOPs AND dedicated HBM bandwidth (~3 TB/s vs the APU's shared
~256 GB/s), so it would win decisively for both source types. The APU is the
pessimistic CPU-vs-GPU case.

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
