# NewStitchFormat — additive/texture HDF5 examples

These inputs exercise the `additive/texture` app with the **HDF5 unstructured**
temperature source (`setup_temperature_source hdf5_unstructured`).

## Data file (required, not shipped in-repo)

The unstructured thermal HDF5 file is large (~1 GB) and is **not** committed.
Supply your own and point the input at it via the `THERMAL_H5` variable:

```bash
mpirun -np 1 spparks \
  -var THERMAL_H5 /path/to/fixed_reduced_thermal_output.hdf5 \
  < in.init_box_0_kokkos
```

Without an override the input looks for `fixed_reduced_thermal_output.hdf5` in
the run directory.

## Host-precompute threading (recommended)

The HDF5 source does heavy **host** work per layer (per-site element cache +
thermal intervals). Those loops are OpenMP-parallel, so run with threads — and
use `--bind-to none` so a single MPI rank's threads spread across cores (without
it, `-np 1` binds them all to one core):

```bash
OMP_NUM_THREADS=16 mpirun -np 1 --bind-to none spparks -var THERMAL_H5 ... < in.init_box_0_kokkos
```

On gfx1151 this precompute scales ~11.5× from 1→16 threads (see
`benchmarks/BASELINE.md`). It dominates wall-clock when serial, so threading it
matters far more than for the analytic Rosenthal/Moser sources.

## Validating the temperature field

Microstructure output is non-deterministic run-to-run; only the temperature
field is deterministic. Compare the `diag_style array d6 sum d6 max` columns
between the CPU app (`additive/texture`) and the Kokkos app
(`additive/texture/kk`) — they should match. The Kokkos app requires
`sweep color`.

## API migration note (2026-07)

`in.init_box_0_kokkos` was updated to the current API. Older inputs in this
directory (`in.init_box_0*`, `in.gradient_test_*`) may still use the pre-migration
forms and will error until updated the same way:

- `app_style additive_temperature_texture/kk` → `app_style additive/texture/kk`
  (and the CPU `additive_temperature_texture` → `additive/texture`; the temp file
  path is no longer an `app_style` argument — use `setup_temperature_source`).
- `sweep random ...` → `sweep color` for the Kokkos app.
- `texture_parameters c1 c2 c3` → `texture_parameters c2/c1 c3` (2 args; the
  baseline coefficient c1 is fixed at 1.0). Note the resulting c2/c3 must yield
  non-negative mobility over all misorientation angles or the app errors — the
  old three-argument coefficients do not always map to a valid two-argument pair,
  so a known-valid set may be needed (the example uses `2.990000e+00 2`).
- `misorientation_function` likewise takes 2 args.
- `dump ... stitch ...` requires the STITCH package; it is commented out in the
  updated example so it runs without STITCH.
