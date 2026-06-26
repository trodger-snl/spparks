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

---

# CPU characterization: kk OpenMP vs native CPU MPI (ct15 box_000, HDF5)

Cross-branch CPU benchmark answering "how does the GPU-line kk app on the Kokkos
**OpenMP** backend compare to the native CPU code under **MPI**, on the same
node?" Run on the **16-core Strix Halo** (AMD Ryzen AI MAX+ 395, ~256 GB/s shared
LPDDR5X), all jobs **sequential** (no core contention), 2026-06-25.

## Setup

- Input: the real study input `box_000/in.init_box_000` (ct15 10-layer HDF5
  source, `ct15_0-9layer.hdf5`), **5,027,937 sites**, full window
  `reset_time 38845.490 → run 38845.644 upto`. Stitch dump stripped; nothing
  written to the LaCie study dir.
- **Job A (kk OpenMP):** app `additive/texture/kk`, `sweep color`,
  `build-openmp` (Kokkos OpenMP backend). `misorientation_target random` dropped
  (unsupported in the kk app; an orientation-sampling detail, negligible cost).
- **Job B (native CPU MPI):** app `additive_temperature_texture` (deprecated CPU
  alias), the SURGE repo `/home/tron/spparks @ SURGE`, plain-MPI build.
- Fairness: microstructure is non-deterministic and the two apps differ, so the
  metric is **wall / loop time for the same simulated window**, not output. All
  configs landed on the **identical final temperature** (sum d6 4.10166e9, max d6
  1606.78 K) — confirming the same thermal field was simulated.

## Headline: native CPU + MPI is ~2× faster than the kk app on CPU

| config                               | wall (s) | loop (s) | cache (s) | RSS/rank |
|--------------------------------------|---------:|---------:|----------:|---------:|
| kk OpenMP, 1 rank × 16 thr           |   292.2  |  272.2   |   2.18    |  4.20 GB |
| kk OpenMP, 1 rank × 32 thr (SMT)     |   286.9  |  266.9   |   2.02    |  4.23 GB |
| kk hybrid, 2 ranks × 8 thr           |   319.0  |  306.5   |   1.88    |  2.52 GB |
| kk hybrid, 4 ranks × 4 thr           |   283.7  |  274.9   |   1.59    |  1.61 GB |
| kk hybrid, **8 ranks × 2 thr**       |   260.8  | **253.5**|   1.71    |  1.10 GB |
| kk MPI, 16 ranks × 1 thr             |   262.1  |  255.5   |   1.66    |  0.82 GB |
| **kk HIP GPU, 1 rank + 16 host thr** |   296.5  |  268.1   |   4.90    |  3.97 GB |
| **native CPU, 15 ranks × 1 (SURGE)** | **136.8**| **130.5**|   2.14    |  0.77 GB |

The **GPU row** is the kk HIP build (Radeon 8060S APU): per-step physics on the
GPU, the per-layer site cache built on **16 host OpenMP threads** (the
"gpu/openmp caching" path). It runs 1 MPI rank bound to the single GPU.

(kk did ~7.12M MC accept events vs CPU ~4.57M — the kk app does ~1.56× more
microstructure churn; normalized per event the gap is ~1.26× rather than 2×.)

## Findings

1. **The cache build is no longer a bottleneck in any config (~1.6–2.2 s).** Both
   strategies parallelize it: kk via the OpenMP layer-cache loops (commit
   "OpenMP-parallelize the per-layer HDF5 site cache..."), MPI implicitly via
   domain decomposition (each rank builds only its subdomain). See
   [reference] the layer-cache OpenMP work.

2. **For the kk app, MPI ranks >> OpenMP threads.** Loop time falls monotonically
   as threads/rank drop: 2×8 (306) → 4×4 (275) → 8×2 (253) ≈ 16×1 (255). Threads
   share the APU's memory pool and contend; ranks get domain-local working sets.
   Sweet spot ≈ 1–2 threads/rank.

3. **SMT (32 threads) buys ~2%** over 16 — within noise. The per-step kernels are
   memory-bandwidth-bound, so extra logical threads add no bandwidth. The only
   gains were in the latency-bound startup loops (cache/intervals). 16 threads
   (= physical cores) is the sweet spot.

4. **The real gap is the app, not the knob.** At equal ~16-way MPI, kk 16×1
   (255 s) is **~2× the native CPU 15×1 (130 s)** — Kokkos/abstraction overhead
   (kernel launches, color-sweep `app_update`, DualView/atomics) plus more MC
   events. No thread/rank mix closes it. **On CPU, use the native app + MPI; the
   kk app earns its keep only on the GPU.**

5. **Memory trade-off (the one thing hybrids buy):** fewer ranks → smaller
   aggregate footprint (2×8 ≈ 5 GB total vs 16×1 ≈ 13 GB) at a speed cost. The
   single-rank kk job holds the whole 4.2 GB domain in one process; useful for
   RAM-constrained / fat single-GPU runs.

6. **The integrated GPU ties the CPU OpenMP run, and stays ~2× behind CPU MPI.**
   kk HIP GPU loop 268 s ≈ kk OpenMP 16-thr 272 s, and ~2× the native CPU 15-rank
   (130 s). No GPU win here because this is an **integrated APU** — the GPU shares
   the same ~256 GB/s LPDDR5X pool as the CPU, so for this bandwidth-bound +
   microstructure-heavy workload it has no bandwidth edge, only raw FLOPs (which
   this workload doesn't lean on). This is the pessimistic GPU case and
   **understates a discrete GPU**: an H100/MI300X has dedicated HBM (~3 TB/s,
   >10×) and would pull ahead decisively. Treat 268 s as an APU floor, not the
   GPU's ceiling.

7. **The host-OpenMP cache path works in the GPU build** — the whole point of the
   "gpu/openmp caching" change. Cache build = 4.9 s on the 1-rank GPU job (vs
   ~24 s serial), so single-rank GPU startup is no longer cache-dominated. It is
   ~2× the pure-OpenMP build's 2.2 s, likely because the HSA/GPU-driver threads
   contend with the host OpenMP pool; still a ~5× win over serial.

## How to re-run

```bash
# scripts used (scratch): run_bench.sh (A vs B), run_hybrid.sh (rank×thread sweep)
# kk:   mpirun -np R --map-by ppr:R:node:pe=T --bind-to core \
#         -x OMP_NUM_THREADS=T build-openmp/spparks < in.kk_full   # additive/texture/kk + sweep color
# gpu:  mpirun -np 1 --bind-to none -x OMP_NUM_THREADS=16 \
#         build-verify/spparks < in.kk_full        # HIP; host OpenMP builds the cache
# cpu:  mpirun -np 15 --bind-to core /home/tron/spparks/build/spparks < in.cpu  # additive_temperature_texture
```

GPU-safety: the GPU run is bounded (`run ... upto`) and completes on its own;
NEVER kill/timeout/SIGTERM it mid-kernel (SDMA-queue wedge -> reboot). One GPU
job at a time; the daily-driver machine must be free of other GPU use.

Note on cross-machine comparison: this APU is ~256 GB/s shared LPDDR5X. An Apple
**M-Pro** is the same bandwidth class (~270 GB/s) so it should not change the
MPI-vs-OpenMP conclusion; an **M-Max** (wider bus, ~1.6–2×) is the chip that
would actually test the bandwidth hypothesis. Kokkos has no Metal backend, so on
Mac the kk app is CPU-only.

---

# Hot-loop optimization: active-front list + OpenMP nodal precompute

Profiling (real Kokkos Tools `space_time_stack`) showed the kk per-step loop was
host-bound and over-scanned the lattice: every microstructure kernel + three
`Check*Candidates` pre-scans ran over the full `nlocal` (up to 5M) even though
the active region is a thin melt/solidification shell, and the host
`precompute_nodal_temperatures` ran serially every step over ~33k mesh nodes.

Two changes (commits "kk active-front list (Steps A–G)" + "Phase 2: OpenMP …
precompute", branch SURGE-kokkos, 2026-06-25):
1. **Active-front list** — one `BuildActiveList` parallel_scan builds the union
   superset of every site any microstructure kernel could touch (liquid OR hot
   OR smoothing-band OR leftover-nucleus); melt/nucleate/particle/solidify and
   the smoothing-list build run over that list (size n_active) instead of
   nlocal; the three `Check*` pre-scans are deleted.
2. **OpenMP** on the per-step nodal-temperature precompute (independent per-node,
   bit-identical).

Both keep the temperature field **bit-identical** (validated single-rank,
1-vs-16 threads, np1==np2 multi-rank, and CPU-app regression). Microstructure is
unchanged statistically; the nucleation always-launch recovery invariant was
validated with a 2-pass re-melt input.

## Results (kk OpenMP, 16 threads, full box_000 window, same APU)

`space_time_stack` profile (38845.51 window, ~2290 steps):

| metric                         | before  | after   |
|--------------------------------|--------:|--------:|
| avg per step                   | 7.13 ms | **2.96 ms (2.4×)** |
| SolidificationKernel           | 2.81 s  | 0.14 s (**20×**) |
| `Check*Candidates` ×3          | 0.44 s  | gone |
| BuildSmoothingList/Nucl/Melt/Particle | ~2.0 s | negligible |
| BuildActiveList (new, 1 scan replaces ~6 passes) | — | 2.65 s |
| host precompute (per call)     | 2.17 ms | 0.24 ms (**9×**) |
| HDF5Fill (temperature; untouched) | 4.69 s | 4.95 s |

End-to-end full-window run (5,027,937 sites), kk OpenMP 16 threads:

| | before | after |
|---|---:|---:|
| **Loop time** | 272.2 s | **171.2 s (1.59×)** |
| Wall | 292.2 s | 191.7 s |
| final temp sum/max (validity) | 4.10166e9 / 1606.78 | 4.10166e9 / 1606.78 (identical) |

This narrows the gap to native CPU 15-rank MPI (130.5 s) from ~2.1× to **~1.3×**.
Remaining per-step costs are the legitimate temperature device fill (HDF5Fill)
and the single O(nlocal) `BuildActiveList` scan (now the 2nd-largest kernel).

### Partial fusion of the active predicate into the temperature fill

The HDF5 fill already visits every site and has T in a register, so it also
evaluates the active predicate and writes a 1-byte flag; `BuildActiveListThin`
compacts that 5 MB flag array instead of re-reading T + active_flag + solid_d
(~100 MB). Clean back-to-back A/B (full window, same machine), final temperature
identical in all four:

| backend | no-fusion | fusion | gain |
|---|---:|---:|---:|
| OpenMP (16t CPU) | 169.5 s | **148.0 s** | **12.7%** |
| HIP (GPU) | 318.1 s | **302.7 s** | **4.8%** |

Bandwidth optimization (eliminates the redundant temperature re-read), so it
helps the bandwidth-starved CPU more. NB the `space_time_stack` per-kernel view
under-counts it (3.6% there vs 12.7% wall) — the saved re-read relieves memory
pressure across the whole loop, not just `BuildActiveList`.

### Backend divergence (important, needs a clean GPU A/B to confirm)

HIP no-fusion (318 s) is **slower than the pre-active-front GPU baseline
(268 s)** — i.e. the active-front list, a large OpenMP win, looks **GPU-negative**:
a GPU parses 5 M vs ~15 k sites per kernel nearly for free (massive parallelism),
so it gains little from "touch fewer sites" but pays for the **extra per-step
`BuildActiveList` kernel launch + fence** (~44 % of the GPU loop is launch/fence/
host overhead). GPUs want fewer/bigger kernels; CPUs want less data touched. The
268 s figure is from an earlier session (not a clean A/B); a pre-active-front-vs-
current GPU A/B is needed to confirm — and if so the active-front path should be
gated to host backends and bypassed on HIP.
