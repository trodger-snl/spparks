#!/usr/bin/env python3
"""Is the grain structure locked to the hatch pattern? Boundary-voxel fraction per x column
and per y column (through the full height), its autocorrelation, the power at the hatch period,
and in-plane grain-width distributions at three heights.
usage (WORK env): hatch_periodicity.py gp700.st --out DIR [--dx 3e-6] [--crop X0 X1 Y0 Y1] [--floor-um 100] [--hatch-um 80]
"""
import argparse, os, sys, json
import numpy as np
from spparks_micro_analysis.stitch_reader import StitchReader
from spparks_micro_analysis.grain_size_analysis import relabel_grains
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
ap = argparse.ArgumentParser(); ap.add_argument('stitch'); ap.add_argument('--out', default='hatch')
ap.add_argument('--dx', type=float, default=3e-6); ap.add_argument('--crop', type=float, nargs=4, default=None)
ap.add_argument('--floor-um', type=float, default=100.0); ap.add_argument('--hatch-um', type=float, default=80.0)
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); dxu = a.dx*1e6
with StitchReader(a.stitch) as r:
    b = r.get_global_bounds(); t = r.times[-1]
    i1 = r.read_field_at_time('i1', t, bounds=b)[0]; i2 = r.read_field_at_time('i2', t, bounds=b)[0]
if a.crop:
    c = [int(round(v/dxu)) for v in a.crop]; i1 = i1[c[0]:c[1], c[2]:c[3], :]; i2 = i2[c[0]:c[1], c[2]:c[3], :]
solid = i2 == 3; lab = relabel_grains(np.where(solid, i1, 0).astype(np.int64))
zf = int(round(a.floor_um/dxu)); lab = lab[:, :, zf:]; solid = solid[:, :, zf:]
nx, ny, nz = lab.shape
# boundary voxels normal to x and to y
bx = (lab[1:,:,:] != lab[:-1,:,:]) & solid[1:,:,:] & solid[:-1,:,:]
by = (lab[:,1:,:] != lab[:,:-1,:]) & solid[:,1:,:] & solid[:,:-1,:]
fx = bx.mean(axis=(1,2)); fy = by.mean(axis=(0,2))          # boundary fraction per x column / y column
def acf(v):
    v = v - v.mean(); n = len(v); c = np.correlate(v, v, 'full')[n-1:]; return c / c[0]
def power_at(v, period_vox):
    v = v - v.mean(); k = np.arange(len(v)); return abs(np.sum(v*np.exp(-2j*np.pi*k/period_vox)))**2 / np.sum(np.abs(np.fft.rfft(v))**2)
ph = a.hatch_um/dxu
out = {'boundary_fraction_x': fx.tolist(), 'boundary_fraction_y': fy.tolist(),
       'acf_x_at_hatch': float(acf(fx)[int(round(ph))]), 'acf_y_at_hatch': float(acf(fy)[int(round(ph))]),
       'power_fraction_x_at_hatch': float(power_at(fx, ph)), 'power_fraction_y_at_hatch': float(power_at(fy, ph))}
# in-plane grain widths (run lengths of constant label along x and along y) at three heights
widths = {}
for frac in (0.25, 0.5, 0.75):
    z = int(nz*frac); W = []
    for axis in (0, 1):
        sl = lab[:, :, z] if axis == 0 else lab[:, :, z].T
        for row in sl:
            ch = np.flatnonzero(np.diff(row)) + 1; runs = np.diff(np.concatenate([[0], ch, [len(row)]])); W.extend(runs[1:-1]*dxu)   # drop the two cut runs
    W = np.array(W); widths[f'{(z+zf)*dxu:.0f}um'] = dict(median=float(np.median(W)), p10=float(np.percentile(W,10)), p90=float(np.percentile(W,90)), n=int(len(W)))
out['width_um'] = widths
json.dump(out, open(f'{a.out}/hatch_periodicity.json', 'w'), indent=1)
print(f"boundary-fraction autocorrelation at a {a.hatch_um:.0f} um shift: x {out['acf_x_at_hatch']:.2f}, y {out['acf_y_at_hatch']:.2f}; "
      f"fraction of spectral power at the hatch period: x {out['power_fraction_x_at_hatch']:.2f}, y {out['power_fraction_y_at_hatch']:.2f}")
for k, v in widths.items(): print(f"in-plane grain width at z={k}: median {v['median']:.0f} um (p10/p90 {v['p10']:.0f}/{v['p90']:.0f}), n={v['n']}")
fig, ax = plt.subplots(1, 3, figsize=(15, 3.8))
xs = (np.arange(len(fx)) + 0.5)*dxu; ys = (np.arange(len(fy)) + 0.5)*dxu
ax[0].plot(xs, fx, color='#2a78d6', lw=1.2, label='boundaries normal to x'); ax[0].plot(ys, fy, color='#c9552a', lw=1.2, label='normal to y')
for k in range(0, int(max(xs.max(), ys.max()))+1, int(a.hatch_um)): ax[0].axvline(k, color='#bbb', lw=0.6, ls=':')
ax[0].set_xlabel('position in crop (um); dotted: hatch period'); ax[0].set_ylabel('boundary-voxel fraction'); ax[0].legend(frameon=False, fontsize=8); ax[0].set_title('grain-boundary density across the crop', loc='left', fontsize=10)
lag = np.arange(len(fx))*dxu; ax[1].plot(lag[:int(200/dxu)], acf(fx)[:int(200/dxu)], color='#2a78d6', label='x'); ax[1].plot(lag[:int(200/dxu)], acf(fy)[:int(200/dxu)], color='#c9552a', label='y')
ax[1].axvline(a.hatch_um, color='#888', lw=0.8, ls='--'); ax[1].set_xlabel('shift (um); dashed: hatch spacing'); ax[1].set_ylabel('autocorrelation'); ax[1].legend(frameon=False); ax[1].set_title('autocorrelation of boundary density', loc='left', fontsize=10)
for (k, v), col in zip(widths.items(), ('#2a78d6', '#7fb069', '#c9552a')):
    pass
z = int(nz*0.5); W = []
for axis in (0, 1):
    sl = lab[:, :, z] if axis == 0 else lab[:, :, z].T
    for row in sl:
        ch = np.flatnonzero(np.diff(row)) + 1; runs = np.diff(np.concatenate([[0], ch, [len(row)]])); W.extend(runs[1:-1]*dxu)
ax[2].hist(W, bins=np.arange(0, 320, 10), color='#2a78d6', alpha=0.8); ax[2].axvline(a.hatch_um, color='#888', lw=0.8, ls='--'); ax[2].set_xlabel('in-plane grain width at mid-height (um)'); ax[2].set_ylabel('count'); ax[2].set_title('grain widths (line intercepts along x and y)', loc='left', fontsize=10)
for a_ in ax: a_.grid(color='#e6e5e1', lw=0.6)
fig.tight_layout(); fig.savefig(f'{a.out}/hatch_periodicity.png', dpi=140); print(f'wrote {a.out}/hatch_periodicity.png')
