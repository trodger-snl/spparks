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

### Full fusion (atomic-append, no scan) — the GPU follow-up, now shipped

Re-profiling showed `BuildActiveListThin` is the GPU's #2 cost (2.81 s vs 0.39 s
on OpenMP) because a `parallel_scan` over nlocal is a multi-pass prefix sum on a
GPU regardless of how little it reads. **Full fusion** drops the separate scan
entirely: the fill `atomic_fetch_add`-appends active sites straight into
`d_active_sites`. Clean A/B (full window, back-to-back, temps identical):

| backend | partial fusion | full fusion | Δ |
|---|---:|---:|---:|
| OpenMP (16t CPU) | 149.8 s | 153.4 s | **−2.4%** (atomic contention) |
| HIP (GPU) | 305.6 s | **275.0 s** | **+10.0%** (kills the scan) |

Textbook backend split: GPUs want fewer kernels, CPUs want less contention.
Shipped **full fusion everywhere** (one code path) to favor the GPU — the kk
port's target; the small OpenMP regression is on a backend used mainly for
hazard-free validation. (A `if constexpr` backend-conditional could recover the
OpenMP optimum if ever wanted.)

Lesson reinforced: trust **within-session back-to-back A/Bs**, not cross-session
GPU numbers, and not the fence-distorted `space_time_stack` TOTAL on HIP (per-
kernel *relative* costs are still useful for spotting bottlenecks like the scan).

### Device-side time interpolation — the biggest GPU win (win on both backends)

Profiling + a 65%-GPU/50%-CPU observation showed the per-step **host**
`precompute_nodal_temperatures` (time-interp of ~33k active mesh nodes) sat on the
critical path, stalling the GPU each step. Moving it onto the device — upload the
per-node raw time-series CSR + active-node list once per layer, a
`ComputeNodalTemps` device kernel interpolates the ~33k active nodes, the fill
blends as before — removes that host work (precompute + per-step slice copy +
time-interp). Bit-matches the host (lower_bound + linear interp); d6 identical.

Getting the granularity right mattered: **per-site** interp (4×nall searches into
the 752 MB series) was **3.2× slower** on OpenMP; **per-node all-nodes** (232k)
was 1.33× slower; **per-node active-only** (~33k, matching the host subset) is the
win. Clean A/B (full window, temps identical):

| backend | host precompute | device time interp | Δ |
|---|---:|---:|---:|
| OpenMP (16t CPU) | 153.2 s | 151.3 s | **+1.2%** (~neutral) |
| HIP (GPU) | 270.9 s | **171.8 s** | **+36.6%** |

A win on **both** backends → shipped everywhere (no split). This is the single
largest optimization of the hot-loop effort; it closed the HIP↔OpenMP gap from
~2× to **~1.14×** (171.8 vs 151.3 s). Safety: the device path doesn't reproduce
optional nodal mesh-smoothing, so it falls back to the host precompute when
nodal_smooth is ON. Origin: the host-bound diagnosis from the 65%-GPU/50%-CPU
utilization split during a HIP run.

### The active-front list helps the GPU too (clean A/B; earlier "regression" was noise)

A cross-*session* comparison (HIP no-fusion 318 s vs an earlier 268 s pre-active-
front baseline) suggested the active-front list might be GPU-negative. A clean
within-session A/B (full-commit checkouts, both pre-precompute/pre-fusion, run
back-to-back) **refutes that**:

| HIP (full window, same session) | loop |
|---|---:|
| pre-active-front (214f8382) | 268.7 s |
| post-active-front (9c93e951) | **246.8 s** |

The active-front list makes the GPU **~8.2 % faster** (268.7 → 246.8 s), not
slower — and `preAF` (268.7 s) reproduces the original 268 s baseline, so it is
the 318 s no-fusion figure that was anomalous (its session ran ~50 s slow).
**Lesson: cross-session GPU timings on this APU drift ~±20 % (thermal/load); only
within-session back-to-back A/Bs are trustworthy.** Both optimizations help both
backends: active-front ~8.2 % (GPU, this A/B) / large (OpenMP); partial fusion
4.8 % (GPU) / 12.7 % (OpenMP). No backend gating is warranted.

## FP32 vs FP64 device temperature eval (gfx1151, np1, within-session A/B)

The device temperature kernels carry their per-site transcendental math in a
compile-time `fastreal` type (`src/spktype.h`; was `AMReal` in
`src/KOKKOS/am_temp_precision.h` when these numbers were taken). Default is FP64
(`double`) — HPC production precision and the bit-for-bit CPU-validation
reference. `-DSPPARKS_FP32=ON` (formerly `-DSPPARKS_AM_TEMP_FP32=ON`) switches the eval math to `float`. FP32 is
scoped to the transcendental work only: `k_T` and the plan PODs stay double, and
the cancellation-prone site−laser / time-delta differences (and Moser's
quadrature order selection) stay double, so both precisions run the identical
quadrature structure — isolating the knob to the summand.

Same box (5M sites, `region box 0 310 0 316 0 50`), `build-verify` (HIP), the two
`benchmarks/in.{rosenthal,moser}_mpi_bench` inputs run back-to-back:

| source    | temp-update FP64 | temp-update FP32 | kernel speedup | loop FP64 | loop FP32 |
|-----------|-----------------:|-----------------:|:--------------:|----------:|----------:|
| Rosenthal |   1.370 s (500 st) |   0.743 s |  **1.84×** |  4.58 s |  3.74 s |
| Moser     |  27.347 s (300 st) |   6.205 s |  **4.41×** | 29.53 s |  8.26 s |

**Accuracy: `sum(d6)` and `max(d6)` are identical to the diagnostic's 6 s.f.** at
every stats output for both sources (any deviation is below the 6th digit).

Full-precision per-site check (Moser, all 4,898,000 sites dumped `id d6` at
`%.17g`, FP32 vs FP64 final frame, field range 573–4906 K):

| metric | value |
|---|---|
| max relative error | **1.64e-7** (≈ 1 float32 ULP; eps = 1.19e-7) |
| max absolute error | 3.5e-4 K (at a 2578 K site) |
| RMSE | 1.19e-5 K |
| exact bit matches | 34.8 % of sites |

The worst-case error sits right at 1 FP32 ULP — i.e. the "keep differences in
double" split successfully avoids cancellation amplification; FP32 is at its
precision floor, not degraded by the algorithm. Practical impact on the melt
decision is nil: with liquidus 1573 K, a decision only flips for a site whose
FP64 temperature is within ~3.5e-4 K of exactly 1573 K (a measure-zero set).

The Moser win (4.4×) dwarfs Rosenthal's (1.84×) because Moser is 99.5 %
compute-bound in its Green's-function quadrature (many exp/sqrt per site), while
Rosenthal is closer to memory-bound (one exp per site) — consistent with the
OpenMP-vs-HIP characterization above. FP32 breaks the bit-exact CPU match, so on
this hardware it is an opt-in speed knob; FP64 stays the default for HPC/CDNA and
validation.

### FP32 GPU vs the CPU code (within-session, same 5M-site box)

Three CPU/GPU paths run back-to-back this session. "Native MPI" is the
non-Kokkos `additive/texture` app (host Green's-function integrator, pure MPI,
np16 = 16 cores; np32/SMT gives ~nothing on this compute-bound kernel).
"OpenMP" is the same Kokkos source on the OpenMP backend, 16 threads, FP64
(`mpirun --bind-to none` — WITHOUT it, `-np 1` chokes all threads onto one core
and Moser inflates to ~466 s). All temperature fields agree to the diagnostic's
6 s.f. (native and OpenMP are FP64; GPU FP32 differs by ≤1 ULP, see above).

Full loop time (temp-update kernel where separable):

| source    | native MPI np16 | kk OpenMP 16t (FP64) | GPU HIP FP64 | GPU HIP FP32 |
|-----------|----------------:|---------------------:|-------------:|-------------:|
| Moser     | 51.12 s         | 42.01 s (temp 40.34) | 29.53 (27.35)| **8.98 (6.91)** |
| Rosenthal |  **2.77 s**     |  4.17 s (temp 2.07)  |  4.58 (1.37) |  3.78 (0.76) |

Two distinct regimes:

- **Moser (compute-bound): the GPU + FP32 is decisive.** FP32 HIP is **5.7× faster
  than the native CPU MPI app** (51.1 → 8.98 s) and 4.7× faster than kk OpenMP.
  This is the case that justifies the GPU on this APU.
- **Rosenthal (memory-bound, cheap kernel): the CPU wins.** The native MPI app
  (2.77 s) beats even FP32 HIP (3.78 s) — the Kokkos per-step launch/overhead
  dominates this ~1-exp-per-site kernel, and minimal-overhead MPI decomposition
  is simply the better fit. GPU offload (FP32 or not) is not worth it here.

Incidental finding: the **Kokkos OpenMP backend is ~20 % faster than the native
app on Moser** (42.0 vs 51.1 s) — the flattened fixed-order device quadrature
outruns the host adaptive integrator — but ~50 % slower on Rosenthal (4.17 vs
2.77 s) due to that same per-step Kokkos loop overhead on a cheap kernel. So the
Kokkos-CPU number is not leaving Moser performance on the table; if anything the
refactor helped the expensive path on CPU too.

### HDF5 source: host precompute threading is decisive (gfx1151, kk app)

Unlike the analytic sources, the **HDF5 unstructured source has heavy HOST work
per layer** — the per-site element cache build and the site-level thermal-
interval computation (`build_site_element_cache` /
`compute_site_thermal_intervals`, both `#pragma omp parallel for`). This is where
the otherwise-idle CPU cores earn their keep on a GPU run. HIP device (temperature
fills on GPU) + host OpenMP for the precompute, np1, 2.99M-site box, 1 GB
unstructured file (`fixed_reduced_thermal_output.hdf5`), OMP thread sweep:

| threads | element cache | thermal intervals | total precompute | device loop |
|---------|--------------:|------------------:|-----------------:|------------:|
|   1     |   20.43 s     |     2.49 s        |   22.92 s        |   2.68 s    |
|   4     |    5.32 s     |     0.68 s        |    6.00 s (3.8×)  |   2.71 s    |
|  16     |    1.77 s     |     0.23 s        |  **2.00 s (11.5×)** |   2.63 s    |

Serial, the host precompute (22.9 s) is **~90 % of wall-clock** and the GPU sits
idle waiting for it. At 16 threads it drops to 2.0 s — now balanced against the
2.6 s device loop, an **11.5× precompute speedup** (72 % parallel efficiency;
memory-bandwidth-bound falloff from the 96 % seen at 4 threads). The device loop
is flat across thread counts (all on GPU), confirming the win is entirely
host-side. This threading is via the raw `#pragma omp` loops, live in any HIP
build with the `find_package(OpenMP)` exe link (`-fopenmp`);
`SPPARKS_KOKKOS_HOST_OPENMP` adds Kokkos-host-space threading on top but is
orthogonal to this precompute.

**HPC implication (4×H100 + 112 cores):** with 1 rank/GPU × 28 host threads,
this precompute scales the same way — the "idle cores" concern is answered *for
the HDF5 source*: they cut the host bottleneck ~11–15×, keeping the H100s fed.
For the analytic sources there is no host precompute to thread (they stay
GPU-bound; cores idle, which is fine).
