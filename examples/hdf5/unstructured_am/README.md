# additive/texture with the HDF5 unstructured temperature source

`in.init_box_0.cpu` (`additive/texture`) and `in.init_box_0.kk`
(`additive/texture/kk`) are the same deck except for the app style and the
sweep (the Kokkos app requires `sweep color`). They exercise
`setup_temperature_source hdf5_unstructured`.

## Data file (required, not shipped in-repo)

The unstructured thermal HDF5 file is large (~1 GB) and is **not** committed.
Supply your own and point the input at it via the `THERMAL_H5` variable:

```bash
mpirun -np 1 spparks \
  -var THERMAL_H5 /path/to/fixed_reduced_thermal_output.hdf5 \
  < in.init_box_0.kk
```

Without an override the input looks for `fixed_reduced_thermal_output.hdf5` in
the run directory.

## Host-precompute threading (recommended)

The HDF5 source does heavy **host** work per layer (per-site element cache +
thermal intervals). Those loops are OpenMP-parallel, so run with threads — and
use `--bind-to none` so a single MPI rank's threads spread across cores (without
it, `-np 1` binds them all to one core):

```bash
OMP_NUM_THREADS=16 mpirun -np 1 --bind-to none spparks -var THERMAL_H5 ... < in.init_box_0.kk
```

With several ranks, set `OMP_NUM_THREADS` to cores / ranks: left unset, every
rank starts a thread per core.

On gfx1151 this precompute scales ~11.5× from 1→16 threads (see
`benchmarks/BASELINE.md`). It dominates wall-clock when serial, so threading it
matters far more than for the analytic Rosenthal/Moser sources.

## Validating the temperature field

Microstructure output is non-deterministic run-to-run; only the temperature
field is deterministic. Compare the `diag_style array d6 sum d6 max` columns
of the two decks — they should match.

The commented-out `dump ... stitch` line needs the STITCH package.
