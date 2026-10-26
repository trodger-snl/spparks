# Stale-layer regression test

Reproduces the bug fixed alongside this directory: when a rank's padded
subdomain overlaps **no chunk** of a layer, `load_layer()` used to return
early while leaving the *previous* layer's mesh and time series resident, so
every site kept interpolating the wrong layer's field.

This is reachable in the normal workflow — thin boxes tiled across a part and
stitched back together, where a box can sit far from the layer currently being
deposited.

## Why it needs a synthetic file

The production `ct15_0-9layer.hdf5` cannot reach the path: every layer's chunk
set spans the whole domain, so no box position loses coverage between layers
(checked by scanning candidate positions against the per-layer bounding boxes).
The 100·dx = 0.5 mm chunk-load pad also swamps the 30 µm/layer growth in z.

The fixture therefore builds its own three-layer file where layer 1's mesh sits
0.05 m away — far outside the pad.

## Run it

```bash
cd examples/am/surge/hdf5/stale_layer
python3 make_stale_layer_fixture.py stale_layer_test.hdf5
mpirun -np 1 ../../../../../build-verify/spparks < in.stale_layer   # any KOKKOS build
```

The generated file is ~140 KB and is not committed; regenerate it as above.

## Pass criterion

The box is 16000 sites and `default_temp` is 300 K, so for every output step in
layer 1's window (`t` in `(1e-3, 2e-3]`):

```
sum(d6) = 4.8e+06      (16000 x 300, exactly)
max(d6) = 300
```

and rank 0 prints

```
Rank 0: layer 1 has no chunks overlapping this subdomain; no thermal data
here, sites read default_temp (300.0 K) until a later layer covers them
```

Layers 0 and 2 are hot (`max(d6)` ~1187 K) and must stay hot — a run that
reads 300 K everywhere means the mesh is not being found at all.

Before the fix, layer 1's window instead ramped to `sum(d6)` 1.28e+07 /
`max(d6)` 832 K, reading layer 0's mesh through the cooldown tail the generator
deliberately gives it.

## Note on severity

Real ct15 layers abut without overlapping in time, so a stale layer there falls
foul of the reader's time-bounds check and degrades *silently* to `default_temp`
— the right answer for the wrong reason. Nothing in the format forbids a
cooldown tail, which is why the fixture has one: it turns the latent bug into a
visible one. The device-side hazard (`ComputeNodalTemps` over a stale
active-node count into a zero-extent view) is unconditional once the layer state
is cleared, and is what the `d_n_active_nodes > 0` guard in
`fill_temperature_hdf5_device` exists for.
