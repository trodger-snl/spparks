# AM temperature-field validation decks

Small decks that exercise every device temperature path of the Kokkos
`additive/texture/kk` app and its host counterpart in the CPU
`additive/texture` app. Only the temperature field is deterministic, so the
pass criterion is a bit-for-bit match of the `d6 sum` / `d6 max` diagnostic
columns (see CLAUDE.md, "Validating the GPU port").

| Deck | Path exercised |
|---|---|
| `in.rosenthal.*` | Rosenthal standard; laser starts 2 mm outside the box so the fast-forward predictor skips ~1.2 ms |
| `in.moser_pause_below.*` | Moser with fluctuations; `pause_below 1500 reset_temperature 600` fires a reset pass mid-run; predictor skip at start |
| `in.moser_pause.*` | Moser static two-pass schedule with a 1 ms `pause`; predictor skips the gap |

The HDF5 device path is covered by `../hdf5/stale_layer`.

```bash
./compare.sh ../../build-verify/spparks                       # CPU vs kk, one binary
./compare.sh ../../build-cpu/spparks ../../build-verify/spparks   # two binaries
```

`fast_forward_search_window 1e-4` is set so the predictors walk finely enough
to fire on these 3-5 ms runs. Each deck takes a few seconds on the GPU; the
CPU Moser decks take about a minute.
