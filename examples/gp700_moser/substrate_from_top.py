#!/usr/bin/env python3
"""Cut the top NZ voxel layers of a finished build into a substrate stitch file
(same layout as make_substrate.py: i1, i2 = 1, d1-d4, d8 = d9 = 0, z shifted to
0..NZ-1), so the next build starts on a competed columnar structure instead of
random Voronoi grains.
usage (WORK env): substrate_from_top.py BUILD.st OUT.st [--nz 50]
"""
import argparse, sys, numpy as np
from spparks_micro_analysis.stitch_reader import StitchReader, StitchWriter
ap = argparse.ArgumentParser(); ap.add_argument('src'); ap.add_argument('out'); ap.add_argument('--nz', type=int, default=50)
a = ap.parse_args()
with StitchReader(a.src) as r:
    b = list(r.get_global_bounds()); t = r.times[-1]
    b[4] = b[5] - a.nz + 1
    f = {k: r.read_field_at_time(k, t, bounds=tuple(b))[0] for k in ('i1','i2','d1','d2','d3','d4')}
i2 = f['i2']; bad = i2 != 3
if bad.any():   # stray never-captured liquid voxels (~0.02%): copy the nearest solid voxel
    from scipy.ndimage import distance_transform_edt
    idx = distance_transform_edt(bad, return_distances=False, return_indices=True)
    for k in ('i1','d1','d2','d3','d4'): f[k] = f[k][tuple(idx)]
    print(f'filled {bad.sum()} non-solid voxels from their nearest solid neighbour')
shape = i2.shape; bo = (b[0], b[1], b[2], b[3], 0, a.nz - 1)
w = StitchWriter(a.out, mode='w')
w.write_field_at_time('i1', 0.0, np.ascontiguousarray(f['i1']).astype(np.int32), bo)
w.write_field_at_time('i2', 0.0, np.ones(shape, np.int32), bo)
for k in ('d1','d2','d3','d4'): w.write_field_at_time(k, 0.0, np.ascontiguousarray(f[k], dtype=np.float64), bo)
for k in ('d8','d9','d10'): w.write_field_at_time(k, 0.0, np.zeros(shape), bo)
w.close()
print(f'{a.out}: z {b[4]}..{b[5]} of {a.src} (t={t}) -> 0..{a.nz-1}; {len(np.unique(f["i1"]))} grains, shape {shape}')
