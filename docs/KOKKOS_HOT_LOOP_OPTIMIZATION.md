# Kokkos additive/texture hot-loop optimization

Performance optimization of the `additive/texture/kk` per-step hot loop and the
HDF5 temperature-source per-layer setup, driven by profiling. All changes keep
the **temperature field bit-identical** (the only deterministic output; see
"Validation"). Branch **SURGE-kokkos**.

## TL;DR

End-to-end on the ct15 `box_000` HDF5 input (5,027,937 sites, full window),
kk on the Kokkos **OpenMP** backend, 16 threads, Strix Halo (AMD Ryzen AI
MAX+ 395):

| | before | after |
|---|---:|---:|
| **Loop time** | 272 s | **171 s (1.59×)** |
| per-step (profiled) | 7.13 ms | 2.96 ms (2.4×) |
| per-layer cache build (1 rank) | ~49 s | ~4 s (~12×) |

This narrowed the gap to the native CPU app under 15-rank MPI (130 s) from
~2.1× to ~1.3×. The temperature field (`sum/max d6`) is bit-identical before vs
after every change.

Three optimizations shipped; one (incremental active set) was tried and rejected.

---

## Motivation: what the profiling showed

Tools: **Kokkos Tools `space_time_stack`** (built from the kokkos-tools repo),
the app's built-in per-kernel timers (the "Kokkos Timing Summary"), and the
SPPARKS `Loop time` breakdown. Profiled on the OpenMP backend (hazard-free),
~2290-step window, 5M-site box.

Findings:

1. **The per-step loop was host-bound and over-scanned the lattice.** Summed
   Kokkos *kernel* time was only ~27 % of wall; ~73 % was host code.
2. **Every microstructure kernel — plus three throwaway `Check*Candidates`
   pre-scans — ran over the full `nlocal`** (up to 5 M sites) even though the
   active region (melt pool + solidification shell) is a thin front of
   ~thousands of sites. `SmoothingKernelEfficient` was already ~0 % because it
   alone used a compact list; nothing else did.
3. **The "Temperature update" timer (67 % of the in-loop total) was mostly
   host work**: only ~4.7 s of ~10.5 s was the `HDF5Fill` device kernel; the
   rest was `precompute_nodal_temperatures` running **serially every step** over
   ~33 k active mesh nodes.
4. **Per-layer setup was ~50 % of single-rank runtime**: the HDF5 site-element
   cache build (24 s) and the per-site thermal-interval build (15 s), both
   serial host loops over millions of sites.

The native CPU app is faster partly because it is interval-gated (touches only
active sites) and, under MPI, domain-decomposes the per-layer setup. The kk
app's "positional fill over `nall`" design (chosen to avoid ghost-temperature
comm) does dense work every step.

---

## Optimization 1 — OpenMP-parallelize the per-layer HDF5 setup loops

`src/temperature_source_hdf5_unstructured.cpp` (core, shared with the CPU app).
Commit *"OpenMP-parallelize the per-layer HDF5 site cache + thermal-interval
loops"*.

Two per-layer one-time loops dominated single-rank runtime; both are
embarrassingly parallel (independent per-site/per-node work, read-only against
the immutable-per-layer mesh, distinct output slots):

- **`build_site_element_cache`** — per-site point-in-element location.
  `#pragma omp parallel for schedule(dynamic, 256)` (dynamic because in-mesh
  sites finish in a few tet tests while out-of-mesh sites search a whole grid
  cell and fail). **24.4 s → 2.2 s (11×)** at 16 threads.
- **`compute_site_thermal_intervals`** — per-site hot-interval extraction. Each
  thread accumulates into a private buffer, concatenated under one `critical`;
  the existing `merge_overlapping_intervals` sorts, so accumulation order is
  irrelevant. **25.0 s → 1.8 s (14×)**.

Combined per-layer setup **~49 s → ~4 s**.

Build note: both builds use the `add_executable(spparks …)` path (the
`spparks_lib` CMake branch is dead), and only the OpenMP Kokkos backend brought
`-fopenmp` for free. The **HIP build needed an explicit `find_package(OpenMP)` +
`target_link_libraries(spparks OpenMP::OpenMP_CXX)` in the executable branch**
(plain, keyword-less signature to match the target's other links) — otherwise
the pragmas are silently inert. Verify with `ldd build-verify/spparks | grep
omp`.

---

## Optimization 2 — Active-front index list (the core)

`src/KOKKOS/app_additive_texture_kokkos.{cpp,h}`. Implemented incrementally as
Steps A–G (one commit each), each validated `d6`-bit-identical.

The microstructure kernels were rebuilt to run over a compact **active-front
list** instead of `nlocal`. The list is the **union superset** of every owned
site any microstructure kernel could touch this step, built once per step by a
single `parallel_scan`:

```
site i is active  ⇔  active_flag(i)==2                     (liquid: can nucleate/solidify)
                  ∨  T(i) >= tl                            (hot: can melt)
                  ∨  solid_d(i) ∈ (-nrefine-1, 0)          (smoothing band)
                  ∨  (active_flag(i)==3 ∧ solid_d(i)==-nrefine-2)  (leftover nucleus)
```

Why it is a correct superset (each kernel keeps its own internal guard, so a
looser list is always safe): a site melting this step starts `T>=tl`; a site
nucleating/solidifying starts liquid; particle-flip and smoothing read sites
that were liquid at step start (already listed) plus the explicit band /
leftover-nucleus clauses for cross-step carryover. `n_active==0` ⇔ no liquid,
nothing hot, no band site, no leftover nucleus ⇔ nothing for any kernel to do.

Each kernel takes `(Kokkos::View<int*> d_active_sites, int n_active)`, ranges
over `n_active`, and dereferences `i = d_active_sites(idx)` as its first line —
the body, internal guard, and all neighbor/coordinate reads are unchanged. This
generalizes the pre-existing `BuildSmoothingList` → `d_smoothing_sites` →
`smoothing_kernel_efficient` idiom to the whole phase block.

The three `Check*Candidates` pre-scans were **deleted**: the single
`BuildActiveList` scan subsumes them, and each kernel's own `reduce` returns its
event count (`melting_occurred = naccept>0` preserves the powder-activation
trigger, since melting is deterministic given a candidate). `BuildSmoothingList`
now scans the active list rather than `nlocal`.

**Load-bearing invariant — nucleation always-launch.** Nucleation must never be
gated on an undercooled-liquid *sub-count* (it can momentarily hit 0 while
liquid still exists, permanently stalling recovery). The active list does NOT
reintroduce this: it contains **every** `active_flag==2` site regardless of T,
so nucleation still sees all liquid every active step; only `n_active==0` (no
liquid anywhere) is skipped, where nucleation is impossible by its own guard.
Validated with a 2-pass (`repeats 2`) re-melt input: a fully solidified region
re-melts and shows a second nucleation+solidification wave.

Result (`space_time_stack`, 5M box): `SolidificationKernel` **2.81 s → 0.14 s
(20×)**; the three `Check*` scans gone; melting/nucleation/particle/smoothing-
build dropped to negligible; one new `BuildActiveList` scan (2.65 s) replaces
~6 separate `nlocal` passes.

---

## Optimization 3 — OpenMP-parallelize the nodal-temperature precompute

`src/temperature_source_hdf5_unstructured.cpp`. Commit *"Phase 2: OpenMP-
parallelize precompute_nodal_temperatures"*.

`precompute_nodal_temperatures(time)` runs **every step**, serially interpolating
~33 k active mesh nodes in time. Independent per-node, distinct output slots →
`#pragma omp parallel for` (reduction only on the informational
`nodes_interpolated` counter), **bit-identical** to serial.

**2.17 ms/call → 0.24 ms/call (9×)** at 16 threads. Shared code, so the CPU app
benefits too under hybrid MPI×OpenMP. (`-fopenmp` already wired in by
Optimization 1.)

---

## End-to-end results

`space_time_stack`, same 38845.51 window, 16 threads:

| metric | before | after |
|---|---:|---:|
| avg per step | 7.13 ms | **2.96 ms (2.4×)** |
| SolidificationKernel | 2.81 s | 0.14 s (20×) |
| `Check*Candidates` ×3 | ~0.44 s | gone |
| BuildSmoothingList / Nucl / Melt / Particle | ~2.0 s | negligible |
| BuildActiveList (new — 1 scan replaces ~6 passes) | — | 2.65 s |
| host precompute (per call) | 2.17 ms | 0.24 ms (9×) |
| HDF5Fill (temperature device fill — intentionally untouched) | 4.69 s | 4.95 s |

Full-window run, kk OpenMP 16 threads: **loop 272 → 171 s (1.59×)**, wall
292 → 192 s, final temperature 4.10166e9 / 1606.78 K (identical). Remaining
per-step cost is the legitimate temperature device fill + the single O(`nlocal`)
`BuildActiveList` scan.

---

## Considered and REJECTED — incremental active-set maintenance

The active set overlaps ~100 % step-to-step, suggesting we could maintain it
incrementally (dilate the previous list by one neighbor shell + re-test the
predicate, O(`n_active`)) instead of a full O(`nlocal`) scan, full-scanning only
to reseed an empty set.

**It is not robustly correct for this workload, and was reverted.** An in-code
count-verify (frontier ⊆ full, so equal counts ⇔ no misses) on the real ct15
HDF5 field showed the melt front advances **>1 site/step during pool formation**
— the active set explodes 13 → 998 in ~12 steps, and 1-shell dilation drops up
to ~20 % of the set (leading-edge hot sites), which then melt late → a
systematically biased melt-pool boundary. The misses come entirely from the
long-range **hot** (`T>=tl`) clause; the liquid/band/nucleus clauses *are* local.

(An earlier "0 non-local additions" churn reading was a sampling artifact — the
50-step sampling skipped the fast-growth steps. Small Rosenthal and 2-pass
recovery inputs have gentler ramps and showed 0 misses, masking the issue until
the production field was tested.)

Making it robust by full-scanning every *growth* step erodes the benefit to ~0
(the pool grows for ~half its active lifetime). The only robust angle left is
**fusing** the active-predicate test into the temperature fill (which already
visits every site every step) — modest, bandwidth-bound benefit across 3 fill
paths; low priority while the full scan is correct and fast.

---

## Validation methodology

- **Temperature is the hard gate.** Microstructure is non-deterministic
  (parallel MC); only the temperature field is deterministic. After every change
  we diff the `diag_style array d6 sum d6 max` columns exactly — the active list
  and the precompute never touch `k_T`/the fills/`Tmax`, so any `d6` divergence
  is a bug. Verified single-rank, **1-vs-16 threads**, and **np1==np2** multi-rank.
- **Microstructure: statistical.** `naccept`/`Solid` trends within run-to-run
  noise; a melt-then-recover input confirms the nucleation recovery invariant.
- **Shared-code regression.** Optimization 3 is shared with the CPU app; its
  `d6` is bit-identical 1-vs-16 threads for the CPU `additive_temperature_texture`
  app too.
- **Re-profile** with `space_time_stack` to confirm the O(`nlocal`) scans shrank
  and the `Check*` kernels vanished.
- **HIP**: compile + link only (`ldd` shows `libomp`); never run-and-kill a HIP
  job (gfx1151 SDMA-queue wedge). GPU end-to-end runs are bounded and allowed to
  finish on their own.

## Reproduce / re-profile

```bash
. /home/tron/spack/share/spack/setup-env.sh && spack env activate spparks-dev
cmake -S src -B build-openmp && cmake --build build-openmp -j 8     # OpenMP backend
# profile: build kokkos-tools, then
KOKKOS_TOOLS_LIBS=.../libkp_space_time_stack.so \
  OMP_NUM_THREADS=16 mpirun -np 1 --bind-to none -x KOKKOS_TOOLS_LIBS -x OMP_NUM_THREADS \
  build-openmp/spparks < in.kk_full     # additive/texture/kk + sweep color
```

Full numbers + the CPU/GPU/hybrid comparison are in
`benchmarks/BASELINE.md` (sections "CPU characterization" and "Hot-loop
optimization").

## Files & commits

- `src/KOKKOS/app_additive_texture_kokkos.{cpp,h}` — active-front list (Steps A–G).
- `src/temperature_source_hdf5_unstructured.cpp` — both OpenMP loop sets.
- `src/CMakeLists.txt` — OpenMP wired into the executable target (HIP build).
- `benchmarks/BASELINE.md` — results tables.

git log (SURGE-kokkos): the `9fbc3c71` OpenMP cache commit, the seven
`kk active-front list (Step …)` commits, `ab73c02d` (Phase 2 precompute), and
the `BASELINE.md` records.
