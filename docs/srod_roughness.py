#!/usr/bin/env python3
"""How rough is the SROD field inside grains? Variogram of SROD within grains on longitudinal sections at 3 um
sampling: RMS difference of SROD between two pixels of the same grain separated by a lag, laterally (x) and along the
build direction (z). Smooth gradients (Lamb-like) give lateral << vertical; lineage patches give lateral ~ vertical.
Sections: the IN625 nominal EBSD (P15 XZ, nearest-pixel resampled to 3 um) and the GP700 builds (xz through the crop
centre). usage (WORK env): srod_roughness.py OUT.png [extra_label=stitch ...]"""
import sys, os, numpy as np
sys.path.insert(0, '/home/tron/spparks-micro-analysis/src'); sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'scripts'))
from spparks_micro_analysis.stitch_reader import StitchReader
from spparks_micro_analysis.ebsd_reader import read_ang
from spparks_micro_analysis.grain_segmentation import segment_grains_2d
from srod_analysis import disorientation_deg
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
DX = 3.0; FLOOR = 100.0; LAGS = [1, 2, 4, 8, 16]

def sim_section(st):
    with StitchReader(st) as r:
        b = list(r.get_global_bounds()); t = r.times[-1]; ymid = (b[2] + b[3]) // 2; bb = (67, 199, ymid, ymid, b[4], b[5])
        f = {k: r.read_field_at_time(k, t, bounds=bb)[0] for k in ('i1', 'i2', 'd1', 'd2', 'd3', 'd4')}
    q = np.stack([f['d1'], f['d2'], f['d3'], f['d4']], -1)[:, 0, :]; sol = f['i2'][:, 0, :] == 3; spin = np.where(sol, f['i1'][:, 0, :], 0)
    z0 = int(FLOOR / DX) + 2
    return np.ascontiguousarray(np.transpose(q[:, z0:], (1, 0, 2))), np.ascontiguousarray(sol[:, z0:].T), np.ascontiguousarray(spin[:, z0:].T)

def ebsd_section():
    r = read_ang('/home/tron/HPC-Calibration/ANG-Files/P15/P15-2-side.ang', ci_threshold=0.1)
    q = r['quaternions'][::-1]; v = r['valid_mask'][::-1]; sr, sc = int(round(DX / 0.433013)), int(round(DX / 0.5))
    q = q[::sr, ::sc]; v = v[::sr, ::sc]; H, W = q.shape[:2]; n = int(400 / DX); r0 = (H - n) // 2; c0 = (W - n) // 2
    return np.ascontiguousarray(q[r0:r0+n, c0:c0+n]), np.ascontiguousarray(v[r0:r0+n, c0:c0+n]), None

def srod_field(q, valid, lab=None, seg_deg=2.0, min_height_um=60.0):
    """per-grain SROD from the lowest point in the section (rows = build direction); NaN outside analysed grains.
    lab: grain labels (simulation: spin components = the physical grains); None: segment at seg_deg (EBSD)"""
    if lab is None:
        lab, _ = segment_grains_2d(q, valid_mask=valid, threshold_deg=seg_deg, crystal_structure='cubic', min_grain_size=8); lab = np.where(valid, lab, 0)
    else:
        from scipy import ndimage
        out_lab = np.zeros(lab.shape, int); nxt = 1
        for g in np.unique(lab[lab > 0]):
            cc, n = ndimage.label(lab == g)
            for k in range(1, n + 1): out_lab[cc == k] = nxt; nxt += 1
        lab = out_lab
    out = np.full(lab.shape, np.nan); glab = np.zeros(lab.shape, int)
    for g in np.unique(lab[lab > 0]):
        rr, cc = np.nonzero(lab == g)
        if (rr.max() - rr.min() + 1) * DX < min_height_um: continue
        band = rr <= rr.min() + 1; o = np.argmin(np.abs(cc[band] - cc[band].mean())); qo = q[rr[band][o], cc[band][o]]
        out[rr, cc] = disorientation_deg(np.broadcast_to(qo, (len(rr), 4)), q[rr, cc]); glab[rr, cc] = g
    return out, glab

def variogram(s, glab, lags):
    res = {}
    for name, ax in (('lateral', 1), ('vertical', 0)):
        vals = []
        for L in lags:
            a = s[:, :-L] if ax == 1 else s[:-L, :]; b = s[:, L:] if ax == 1 else s[L:, :]
            ga = glab[:, :-L] if ax == 1 else glab[:-L, :]; gb = glab[:, L:] if ax == 1 else glab[L:, :]
            m = (ga > 0) & (ga == gb) & np.isfinite(a) & np.isfinite(b)
            vals.append(np.sqrt(np.mean((a[m] - b[m])**2)) if m.sum() > 100 else np.nan)
        res[name] = np.array(vals)
    return res

eq, ev, _ = ebsd_section()
sections = [('IN625 EBSD, grains at 2 deg', eq, ev, None, 2.0), ('IN625 EBSD, grains at 5 deg', eq, ev, None, 5.0),
            ('per-voxel model, spin grains', *sim_section('/home/tron/gp700_runs/wide12_v8/gp700.st'), None),
            ('two-term model, spin grains', *sim_section('/home/tron/gp700_runs/wide12_v10/gp700.st'), None)]
for extra in sys.argv[2:]:
    lab_, st = extra.split('='); sections.append((lab_ + ', spin grains', *sim_section(st), None))
fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0)); cols = ['k', '#555555', '#7d8ea0', '#0075a9', '#c9552a', '#287968']
print(f'{"section":36s} ' + ' '.join(f'lat{L*DX:.0f}' for L in LAGS) + ' | ' + ' '.join(f'ver{L*DX:.0f}' for L in LAGS) + ' | lat/ver @3um @12um @48um')
for (name, q, valid, lab, seg), c in zip(sections, cols):
    s, glab = srod_field(q, valid, lab=lab, seg_deg=seg or 2.0); v = variogram(s, glab, LAGS); lag_um = np.array(LAGS) * DX
    axes[0].plot(lag_um, v['lateral'], 'o-', color=c, label=name); axes[0].plot(lag_um, v['vertical'], 's--', color=c, alpha=0.6)
    axes[1].plot(lag_um, v['lateral'] / v['lateral'][-1], 'o-', color=c, label=name)
    r = v['lateral'] / v['vertical']
    print(f'{name:36s} ' + ' '.join(f'{x:5.2f}' for x in v['lateral']) + ' | ' + ' '.join(f'{x:5.2f}' for x in v['vertical']) + f' | {r[0]:.2f} {r[2]:.2f} {r[4]:.2f} | 3um/48um {v["lateral"][0]/v["lateral"][-1]:.2f}')
axes[0].set_xscale('log'); axes[0].set_xlabel('lag (um)'); axes[0].set_ylabel('RMS SROD difference within a grain (deg)'); axes[0].set_title('solid: lateral, dashed: along the build direction', loc='left', fontsize=9); axes[0].grid(alpha=0.3); axes[0].legend(fontsize=8)
axes[1].set_xscale('log'); axes[1].set_xlabel('lag (um)'); axes[1].set_ylabel('lateral RMS difference / value at 48 um'); axes[1].set_title('shape: fraction of the 48 um variation already present at short range', loc='left', fontsize=9); axes[1].grid(alpha=0.3); axes[1].set_ylim(0, 1.05)
fig.tight_layout(); fig.savefig(sys.argv[1], dpi=160); print('saved', sys.argv[1])
