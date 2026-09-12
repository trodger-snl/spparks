#!/usr/bin/env python3
"""KAM and GND maps on the same xz section of the GP700 builds (per-voxel model = wide12_v8,
two-term model = wide12_v10) and, at the same 3 um sampling, the IN625 nominal EBSD section.
KAM: first-neighbour kernel average misorientation, 5 deg threshold (spparks_micro_analysis.kam_analysis).
GND: Nye-tensor estimate from the package if available, else the KAM-based 2*theta/(b*d).
usage (WORK env): kam_gnd_maps.py OUT.png"""
import sys, os, numpy as np
sys.path.insert(0, '/home/tron/spparks-micro-analysis/src')
from spparks_micro_analysis.stitch_reader import StitchReader
from spparks_micro_analysis.kam_analysis import calculate_kam_2d
from spparks_micro_analysis.gnd_analysis import calculate_gnd_density, calculate_gnd_advanced
from spparks_micro_analysis.ebsd_reader import read_ang
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
DX = 3.0; FLOOR = 100.0; B = 0.255e-9

def sim_section(st):
    with StitchReader(st) as r:
        b = list(r.get_global_bounds()); t = r.times[-1]
        x0, x1 = 67, 199; ymid = (b[2] + b[3]) // 2; bb = (x0, x1, ymid, ymid, b[4], b[5])
        f = {k: r.read_field_at_time(k, t, bounds=bb)[0] for k in ('i2', 'd1', 'd2', 'd3', 'd4')}
    q = np.stack([f['d1'], f['d2'], f['d3'], f['d4']], -1)[:, 0, :]; sol = (f['i2'][:, 0, :] == 3)
    z0 = int(FLOOR / DX) + 2; q = q[:, z0:]; sol = sol[:, z0:]
    return np.ascontiguousarray(np.transpose(q, (1, 0, 2))), np.ascontiguousarray(sol.T)   # rows = z (build), cols = x

def ebsd_section():
    r = read_ang('/home/tron/HPC-Calibration/ANG-Files/P15/P15-2-side.ang', ci_threshold=0.1)
    q = r['quaternions'][::-1]; v = r['valid_mask'][::-1]                          # vflip: build direction = increasing row
    sr, sc = int(round(DX / 0.433013)), int(round(DX / 0.5))                          # nearest-pixel resampling to 3 um
    q = q[::sr, ::sc]; v = v[::sr, ::sc]; H, W = q.shape[:2]
    n = int(400 / DX); r0 = (H - n) // 2; c0 = (W - n) // 2
    return np.ascontiguousarray(q[r0:r0+n, c0:c0+n]), np.ascontiguousarray(v[r0:r0+n, c0:c0+n])

def gnd_map(q, valid, kam):
    rows, cols = np.nonzero(valid); coords = np.column_stack([cols * DX * 1e-6, rows * DX * 1e-6, np.zeros(len(rows))])
    try:
        g = calculate_gnd_advanced(coords, q[rows, cols], method='nye_tensor', step_size=DX * 1e-6, burgers_vector=B)
        g = np.asarray(g, float); out = np.full(valid.shape, np.nan); out[rows, cols] = g; return out, 'Nye tensor'
    except Exception as e:
        print('  Nye-tensor GND unavailable (%s); using the KAM-based estimate' % type(e).__name__)
        return calculate_gnd_density(kam, burgers_vector=B, step_size=DX * 1e-6), 'from KAM'

panels = [('per-voxel model', *sim_section('/home/tron/gp700_runs/wide12_v8/gp700.st')),
          ('two-term model', *sim_section('/home/tron/gp700_runs/wide12_v10/gp700.st')),
          ('IN625 EBSD, nominal (3 um sampling)', *ebsd_section())]
for extra in sys.argv[2:]:                      # label=path.st -> extra simulation column
    lab_, st = extra.split('='); panels.insert(len(panels) - 1, (lab_, *sim_section(st)))
fig, axes = plt.subplots(2, len(panels), figsize=(4.5 * len(panels), 8.2)); stats = []
for j, (name, q, valid) in enumerate(panels):
    kam = calculate_kam_2d(q, valid, threshold_deg=5.0, crystal_structure='cubic'); kam_deg = np.degrees(kam)
    gnd, gtag = gnd_map(q, valid, kam)
    H, W = valid.shape; ext = [0, W * DX, 0, H * DX]
    im0 = axes[0, j].imshow(np.where(valid, kam_deg, np.nan), origin='lower', extent=ext, cmap='viridis', vmin=0, vmax=4.0)
    im1 = axes[1, j].imshow(np.where(valid, np.log10(np.clip(gnd, 1e11, None)), np.nan), origin='lower', extent=ext, cmap='plasma', vmin=12.5, vmax=14.7)
    axes[0, j].set_title(name, loc='left', fontsize=10); axes[1, j].set_xlabel('um')
    for ax in axes[:, j]: ax.set_aspect('equal')
    kd = kam_deg[valid & np.isfinite(kam_deg)]; gd = gnd[valid & np.isfinite(gnd)]
    kam2 = np.degrees(calculate_kam_2d(q, valid, threshold_deg=2.0, crystal_structure='cubic')); k2 = kam2[valid & np.isfinite(kam2)]
    q10, q25, q50, q75, q90 = np.percentile(kd, [10, 25, 50, 75, 90])
    print(f'{name:36s} KAM(5 deg) mean {np.nanmean(kd):.2f}, p10/25/50/75/90 {q10:.2f}/{q25:.2f}/{q50:.2f}/{q75:.2f}/{q90:.2f} deg, < 0.5 deg {np.mean(kd < 0.5):.2f}, > 2 deg {np.mean(kd > 2):.2f}; '
          f'KAM(2 deg) mean {np.nanmean(k2):.2f} on {np.isfinite(kam2[valid]).mean():.2f} of pixels; GND median {np.nanmedian(gd):.2e} m^-2 ({gtag})')
axes[0, 0].set_ylabel('KAM (deg)\nbuild direction (um)'); axes[1, 0].set_ylabel('log10 GND (m$^{-2}$)\nbuild direction (um)')
fig.colorbar(im0, ax=axes[0, :].tolist(), fraction=0.02, pad=0.01, label='deg'); fig.colorbar(im1, ax=axes[1, :].tolist(), fraction=0.02, pad=0.01, label='log10 m$^{-2}$')
fig.savefig(sys.argv[1], dpi=150, bbox_inches='tight'); print('saved', sys.argv[1])
