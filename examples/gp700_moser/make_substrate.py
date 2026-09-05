#!/usr/bin/env python3
"""Write a coarse-grained substrate as a stitch file for the GP700 box-0 deck.
Voronoi grains of a given mean size, one random orientation per grain, as
powder (i2 = 1) so growth treats it as solid and the analysis excludes it.
Sites above the substrate are absent from the file; the deck's
`set i1 range ... if i2 < 1` / `set i2 value 1 if i2 < 2` lines seed them.
usage (WORK env): make_substrate.py OUT.st --nx 267 --nz 50 [--grain-um 50] [--dx 3e-6] [--seed 1]
"""
import argparse, sys, numpy as np
from scipy.spatial import cKDTree
sys.path.insert(0, '/home/tron/spparks-micro-analysis/src')
from spparks_micro_analysis.stitch_reader import StitchWriter
ap = argparse.ArgumentParser(); ap.add_argument('out'); ap.add_argument('--nx', type=int, default=267); ap.add_argument('--nz', type=int, default=50)
ap.add_argument('--grain-um', type=float, default=50.0); ap.add_argument('--dx', type=float, default=3e-6); ap.add_argument('--seed', type=int, default=1)
a = ap.parse_args(); rng = np.random.default_rng(a.seed)
dxu = a.dx*1e6; vol = a.nx*a.nx*a.nz*dxu**3; ngr = max(1, int(round(vol / (np.pi/6*a.grain_um**3))))
seeds = rng.random((ngr, 3)) * [a.nx, a.nx, a.nz]
X, Y, Z = np.meshgrid(np.arange(a.nx), np.arange(a.nx), np.arange(a.nz), indexing='ij')
_, lab = cKDTree(seeds).query(np.column_stack([X.ravel(), Y.ravel(), Z.ravel()]))
spins = rng.choice(np.arange(1, 1000001), size=ngr, replace=False)
u = rng.random((ngr, 3)); q = np.column_stack([np.sqrt(1-u[:,0])*np.sin(2*np.pi*u[:,1]), np.sqrt(1-u[:,0])*np.cos(2*np.pi*u[:,1]),
                                               np.sqrt(u[:,0])*np.sin(2*np.pi*u[:,2]), np.sqrt(u[:,0])*np.cos(2*np.pi*u[:,2])])  # uniform unit quaternions
shape = (a.nx, a.nx, a.nz); b = (0, a.nx-1, 0, a.nx-1, 0, a.nz-1)
w = StitchWriter(a.out, mode='w')
w.write_field_at_time('i1', 0.0, spins[lab].reshape(shape).astype(np.int32), b)
w.write_field_at_time('i2', 0.0, np.ones(shape, np.int32), b)
for k, name in enumerate(('d1','d2','d3','d4')): w.write_field_at_time(name, 0.0, np.ascontiguousarray(q[lab, k].reshape(shape)), b)
for name in ('d8','d9'): w.write_field_at_time(name, 0.0, np.zeros(shape), b)
w.close()
print(f'{a.out}: {ngr} grains of ~{a.grain_um:.0f} um in {a.nx*dxu:.0f} x {a.nx*dxu:.0f} x {a.nz*dxu:.0f} um')
