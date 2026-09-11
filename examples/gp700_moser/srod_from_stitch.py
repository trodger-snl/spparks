#!/usr/bin/env python3
"""SROD analysis of a 3-D SPPARKS additive build straight from its stitch file (Lamb 2024 conventions).

Reads i1 (grain id) and d1-d4 (quaternion) through StitchReader for a lateral window and z
range (the pattern of spparks-micro-analysis/scripts/extract_stitch_data.py), then:

  3-D, true origins: grains = connected components of the spin field (or --segment-deg
  re-segmentation); a grain's origin is its lowest voxel. Grains whose origin lies above the
  analysis floor are ORIGINATING (SROD referenced to the origin voxel); grains cut by the floor
  are ESTABLISHED (referenced at the centroid of their footprint on the floor). Lamb's filters
  (height >= 60 um, ESD >= 40 um) apply. Outputs mean / 5 / 95 % / RMS against height, the
  D fit (RMS^2 = D h through the origin for originating grains; with an intercept for
  established), and per-grain statistics (max SROD, top-slab mean, rate per 30 um).

  2-D sections: XZ and YZ planes through the window, analysed exactly as the EBSD sections
  are by srod_from_ebsd.py (lowest point IN THE SECTION as reference, height >= 60 um,
  width >= 15 um), so simulation and experiment carry the same section bias.

usage (WORK env, or any env with spparks_micro_analysis on the path):
  srod_from_stitch.py BUILD.st OUT_DIR LABEL --dx 2.5 [--window-um 800] [--floor-um 300]
        [--top-um X] [--z-range Z0 Z1 (voxels)] [--time T] [--segment-deg 2] [--sections 3]
Solid voxels: i2 == 3 if i2 is present (solidified; unremelted substrate excluded), else all.
"""
import sys, os, json, argparse, time, numpy as np
sys.path.insert(0, '/home/tron/spparks-micro-analysis/src')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'scripts'))
from spparks_micro_analysis.stitch_reader import StitchReader
from spparks_micro_analysis.grain_size_analysis import relabel_grains
from srod_analysis import disorientation_deg

ap = argparse.ArgumentParser(); ap.add_argument('stitch'); ap.add_argument('out'); ap.add_argument('label')
ap.add_argument('--dx', type=float, required=True, help='voxel size (um)')
ap.add_argument('--window-um', type=float, default=None, help='central square lateral window (um); default: full extent')
ap.add_argument('--z-range', type=int, nargs=2, default=None, help='inclusive z voxel range to read (default: full)')
ap.add_argument('--floor-um', type=float, default=300.0, help='analysis floor above the bottom of the read volume (um)')
ap.add_argument('--top-um', type=float, default=None, help='ignore everything above this height from the bottom (um)')
ap.add_argument('--time', type=float, default=None); ap.add_argument('--segment-deg', type=float, default=None)
ap.add_argument('--sections', type=int, default=3, help='XZ and YZ planes per direction for the 2-D analysis')
ap.add_argument('--min-height-um', type=float, default=60.0); ap.add_argument('--min-esd-um', type=float, default=40.0)
ap.add_argument('--min-width-um', type=float, default=15.0, help='2-D section width filter (EBSD comparison)')
ap.add_argument('--bin-um', type=float, default=5.0)
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); dxu = a.dx

# ---------------------------------------------------------------- read (StitchReader, inclusive bounds)
t0 = time.time()
with StitchReader(a.stitch) as r:
    b = list(r.get_global_bounds()); t = a.time if a.time is not None else r.times[-1]
    if a.window_um:
        w = int(round(a.window_um / dxu)); cx = (b[0] + b[1]) // 2; cy = (b[2] + b[3]) // 2
        b[0], b[1], b[2], b[3] = max(b[0], cx - w//2), min(b[1], cx - w//2 + w - 1), max(b[2], cy - w//2), min(b[3], cy - w//2 + w - 1)
    if a.z_range: b[4], b[5] = max(b[4], a.z_range[0]), min(b[5], a.z_range[1])
    b = tuple(int(v) for v in b)
    fields = ['i1', 'd1', 'd2', 'd3', 'd4']
    have_i2 = 'i2' in r.get_available_fields()
    if have_i2: fields.append('i2')
    f = {k: r.read_field_at_time(k, t, bounds=b)[0] for k in fields}
spin = f['i1']; q = np.stack([f['d1'], f['d2'], f['d3'], f['d4']], -1)
solid = (f['i2'] == 3) if have_i2 else np.ones(spin.shape, bool)
solid &= np.linalg.norm(q, axis=-1) > 0.5
nx, ny, nz = spin.shape
if a.top_um: solid[:, :, int(a.top_um/dxu):] = False
print(f'{a.label}: {a.stitch} t={t} bounds {b} grid {spin.shape} ({nx*dxu:.0f} x {ny*dxu:.0f} x {nz*dxu:.0f} um), '
      f'solid {solid.mean():.3f}, read {time.time()-t0:.0f} s', flush=True)

# ---------------------------------------------------------------- 3-D grains
if a.segment_deg:
    from spparks_micro_analysis.grain_segmentation import segment_grains
    lab, _ = segment_grains(q, occupied=solid, threshold_deg=a.segment_deg, crystal_structure='cubic', min_grain_size=8)
    lab = np.where(solid, lab, 0)
else:
    lab = relabel_grains(np.where(solid, spin, 0)); lab = np.where(solid, lab, 0)
ids, counts = np.unique(lab[lab > 0], return_counts=True)
order = np.argsort(lab, axis=None, kind='stable'); flat = lab.ravel()[order]
coords = np.column_stack(np.unravel_index(order, lab.shape)); starts = np.searchsorted(flat, ids); ends = starts + counts
zmin = np.array([coords[s:e, 2].min() for s, e in zip(starts, ends)]); zmax = np.array([coords[s:e, 2].max() for s, e in zip(starts, ends)])
height = (zmax - zmin + 1) * dxu; esd = 2 * (3 * counts * dxu**3 / (4 * np.pi))**(1/3)
floor = a.floor_um / dxu; keep = (height >= a.min_height_um) & (esd >= a.min_esd_um) & (zmax >= floor)
print(f'3-D: {len(ids)} grains; {keep.sum()} pass Lamb filters (height >= {a.min_height_um}, ESD >= {a.min_esd_um} um); '
      f'origins above the floor ({a.floor_um} um): {(keep & (zmin >= floor)).sum()}, cut by the floor: {(keep & (zmin < floor)).sum()}', flush=True)

def curve(h, s, bin_um, min_n=50):
    bins = np.arange(0, h.max() + bin_um, bin_um); rows = []
    for lo in bins[:-1]:
        m = (h >= lo) & (h < lo + bin_um) & (h > 0)
        if m.sum() < min_n: continue
        sv = s[m]; p5, p95 = np.percentile(sv, [5, 95])
        rows.append((float(h[m].mean()), float(sv.mean()), float(p5), float(p95), float(np.sqrt((sv**2).mean())), int(m.sum())))
    return np.array(rows)

def fit_D(rows, intercept, hmax=150.0):
    m = rows[:, 0] <= hmax; h = rows[m, 0]; y = rows[m, 4]**2; w = np.sqrt(rows[m, 5])
    if intercept:
        A = np.column_stack([np.ones(len(h)), h]); c = np.linalg.lstsq(A * w[:, None], y * w, rcond=None)[0]; return float(c[1]), float(np.sqrt(max(c[0], 0)))
    return float((h * y * w**2).sum() / (h**2 * w**2).sum()), 0.0

res = dict(label=a.label, stitch=a.stitch, time=float(t), dx_um=dxu, bounds=b, floor_um=a.floor_um, grains_total=int(len(ids)), grains_filtered=int(keep.sum()))
srod = {0: [], 1: []}; hgt = {0: [], 1: []}; per_grain = []
for gi in np.where(keep)[0]:
    c = coords[starts[gi]:ends[gi]]
    if zmin[gi] >= floor:
        o = c[np.argmin(c[:, 2])]; k = 0; zref = o[2]
    else:
        cf = c[c[:, 2] == int(floor)]
        if len(cf) == 0: continue
        cen = cf[:, :2].mean(0); o = cf[np.argmin(((cf[:, :2] - cen)**2).sum(1))]; k = 1; zref = floor
    m = c[:, 2] >= zref; cm = c[m]
    sv = disorientation_deg(np.broadcast_to(q[tuple(o)], (len(cm), 4)), q[cm[:, 0], cm[:, 1], cm[:, 2]]); hv = (cm[:, 2] - zref) * dxu
    srod[k].append(sv); hgt[k].append(hv)
    top = hv >= hv.max() - 30; bot = hv <= 30
    per_grain.append(dict(id=int(ids[gi]), cls=('originating', 'established')[k], height_um=float(height[gi]), esd_um=float(esd[gi]),
                          origin_z_um=float(zmin[gi] * dxu), max_srod=float(sv.max()), mean_srod_top30=float(sv[top].mean()),
                          rate_deg_per_30um=float((sv[top].mean() - sv[bot].mean()) / max(hv.max(), 30.0) * 30.0)))
for k, name in ((0, 'originating'), (1, 'established')):
    if not srod[k]: print(f'3-D {name}: no grains'); continue
    s = np.concatenate(srod[k]); h = np.concatenate(hgt[k]); rows = curve(h, s, a.bin_um)
    if len(rows) < 3: print(f'3-D {name}: too few bins'); continue
    D, rms0 = fit_D(rows, intercept=(k == 1)); band = float(np.median((rows[:, 3] - rows[:, 2]) / rows[:, 1]))
    res[name] = dict(n_grains=int(sum(1 for g in per_grain if g['cls'] == name)), D_deg2_per_um=D, rms0_deg=rms0, band_over_mean=band,
                     h=rows[:, 0].tolist(), mean=rows[:, 1].tolist(), p5=rows[:, 2].tolist(), p95=rows[:, 3].tolist(), rms=rows[:, 4].tolist(), n=rows[:, 5].tolist())
    print(f'3-D {name:11s}: {res[name]["n_grains"]:4d} grains, {len(s):9d} voxels; D = {D:.3f} deg^2/um (fit to 150 um), rms0 {rms0:.2f}, band/mean {band:.2f}; '
          + ' '.join(f'{x:.0f}um:{np.interp(x, rows[:,0], rows[:,1]):.2f}' for x in (50, 100, 150, 200, 300) if x <= rows[-1, 0]), flush=True)
pg = [g for g in per_grain]
if pg:
    mx = np.array([g['max_srod'] for g in pg]); rt = np.array([g['rate_deg_per_30um'] for g in pg])
    res['per_grain_summary'] = dict(max_srod_median=float(np.median(mx)), max_srod_max=float(mx.max()), frac_gt_25=float(np.mean(mx > 25)),
                                    rate_mean=float(rt.mean()), rate_max=float(rt.max()), top30_mean_median=float(np.median([g['mean_srod_top30'] for g in pg])))
    print(f'per grain: max SROD median {np.median(mx):.1f} (max {mx.max():.1f}) deg, > 25 deg: {np.mean(mx > 25):.2f}; rate mean {rt.mean():.2f} / max {rt.max():.2f} deg per 30 um', flush=True)
res['per_grain'] = per_grain

# ---------------------------------------------------------------- 2-D sections, EBSD-style (lowest point in the section)
from spparks_micro_analysis.grain_segmentation import segment_grains_2d
def section_curve(q2, sol2, seg_deg):
    """q2 (H=z rows, W cols, 4); rows increase with the build direction."""
    if seg_deg: lab2, _ = segment_grains_2d(q2, valid_mask=sol2, threshold_deg=seg_deg, crystal_structure='cubic', min_grain_size=8); lab2 = np.where(sol2, lab2, 0)
    else: lab2 = relabel_grains(np.where(sol2, sp2, 0)); lab2 = np.where(sol2, lab2, 0)
    ids2, cnt2 = np.unique(lab2[lab2 > 0], return_counts=True); o2 = np.argsort(lab2, axis=None, kind='stable'); fl2 = lab2.ravel()[o2]
    cc = np.column_stack(np.unravel_index(o2, lab2.shape)); st2 = np.searchsorted(fl2, ids2); en2 = st2 + cnt2
    S, Hh, n = [], [], 0
    for s_, e_ in zip(st2, en2):
        c = cc[s_:e_]; hgt_ = (c[:, 0].max() - c[:, 0].min() + 1) * dxu; wid = (c[:, 1].max() - c[:, 1].min() + 1) * dxu
        if hgt_ < a.min_height_um or wid < a.min_width_um: continue
        zr = c[:, 0].min() + max(1, int(round(1.5 / dxu))); band = c[c[:, 0] == zr]
        if len(band) == 0: continue
        o = band[np.argmin(np.abs(band[:, 1] - band[:, 1].mean()))]; m = c[:, 0] >= zr; cm = c[m]
        S.append(disorientation_deg(np.broadcast_to(q2[o[0], o[1]], (len(cm), 4)), q2[cm[:, 0], cm[:, 1]])); Hh.append((cm[:, 0] - zr) * dxu); n += 1
    if not S: return None
    s = np.concatenate(S); h = np.concatenate(Hh); rows = curve(h, s, a.bin_um, min_n=30)
    if len(rows) < 3: return None
    D, rms0 = fit_D(rows, intercept=True)
    return dict(n_grains=n, D_deg2_per_um=D, rms0_deg=rms0, band_over_mean=float(np.median((rows[:, 3]-rows[:, 2])/rows[:, 1])),
                h=rows[:, 0].tolist(), mean=rows[:, 1].tolist(), p5=rows[:, 2].tolist(), p95=rows[:, 3].tolist(), rms=rows[:, 4].tolist(), n=rows[:, 5].tolist())
zlo = int(floor)                                   # sections start at the floor, like the interior EBSD scans
secs = {}
for plane, npos in (('XZ', a.sections), ('YZ', a.sections)):
    for j in range(npos):
        pos = int((j + 1) * (ny if plane == 'XZ' else nx) / (npos + 1))
        if plane == 'XZ': q2 = np.transpose(q[:, pos, zlo:, :], (1, 0, 2)); sol2 = solid[:, pos, zlo:].T; sp2 = spin[:, pos, zlo:].T
        else:             q2 = np.transpose(q[pos, :, zlo:, :], (1, 0, 2)); sol2 = solid[pos, :, zlo:].T; sp2 = spin[pos, :, zlo:].T
        out = section_curve(np.ascontiguousarray(q2), np.ascontiguousarray(sol2), a.segment_deg or 2.0)
        if out: secs[f'{plane}_{pos}'] = out
if secs:
    res['sections'] = secs
    for k_, v in secs.items(): print(f'2-D {k_:8s}: {v["n_grains"]:4d} grains, D = {v["D_deg2_per_um"]:.3f}, rms0 {v["rms0_deg"]:.2f}, band/mean {v["band_over_mean"]:.2f}, mean@100um {np.interp(100, v["h"], v["mean"]):.2f}')
json.dump(res, open(os.path.join(a.out, f'srod3d_{a.label}.json'), 'w'))

# ---------------------------------------------------------------- figure
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 2, figsize=(11, 4.3)); ax = axes[0]
for name, col in (('originating', '#2a78d6'), ('established', '#c9552a')):
    if name in res:
        c = res[name]; ax.fill_between(c['h'], c['p5'], c['p95'], color=col, alpha=0.15, lw=0); ax.plot(c['h'], c['mean'], color=col, lw=2, label=f'{name} ({c["n_grains"]} grains), D = {c["D_deg2_per_um"]:.2f}')
ax.set_xlabel('height above origin / floor (um)'); ax.set_ylabel('SROD (deg)'); ax.set_title(f'{a.label}: 3-D, true origins', loc='left', fontsize=10); ax.legend(fontsize=8); ax.grid(alpha=0.3)
ax = axes[1]
for k_, v in secs.items(): ax.plot(v['h'], v['mean'], lw=1.5, ls='-' if k_.startswith('XZ') else '--', label=f'{k_} ({v["n_grains"]}), D = {v["D_deg2_per_um"]:.2f}')
ax.set_xlabel('height above the lowest point in the section (um)'); ax.set_title('2-D sections, EBSD-style', loc='left', fontsize=10); ax.legend(fontsize=7); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(os.path.join(a.out, f'srod3d_{a.label}.png'), dpi=150); print('wrote', a.out)
