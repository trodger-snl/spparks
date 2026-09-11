#!/usr/bin/env python3
"""SROD curves from 2-D longitudinal EBSD sections (.ang), Lamb-style.

For every grain (disorientation segmentation at --seg-deg) taller than --min-height-um
along the build direction and wider than --min-width-um, the reference orientation is
the pixel at the grain's lowest point (offset --ref-offset-um up, as Lamb do), and the
SROD of every pixel is its disorientation from that reference, binned by height above
the reference. Because a section cuts the grain, the lowest point in the section is
above the true origin: the curves are of Lamb's 'established' kind (an initial spread
at h = 0), and D is read from the slope of RMS^2 against height.

usage (WORK env):
  srod_from_ebsd.py OUT_DIR LABEL FILE.ang [--vflip] [--rot90] [--ci 0.1] [--seg-deg 2]
The build direction is the row axis of the (transformed) scan, increasing with the row
index (the convention of HPC-Calibration/scripts/prepare_v8_targets.py); --vflip reverses it.
"""
import sys, os, json, argparse, time, numpy as np
sys.path.insert(0, '/home/tron/spparks-micro-analysis/src')
from spparks_micro_analysis.ebsd_reader import read_ang
from spparks_micro_analysis.grain_segmentation import segment_grains_2d
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'scripts'))
from srod_analysis import disorientation_deg

ap = argparse.ArgumentParser(); ap.add_argument('out'); ap.add_argument('label'); ap.add_argument('ang')
ap.add_argument('--vflip', action='store_true'); ap.add_argument('--rot90', action='store_true', help='rot90 CCW (square grids only)')
ap.add_argument('--ci', type=float, default=0.1); ap.add_argument('--seg-deg', type=float, default=2.0)
ap.add_argument('--min-height-um', type=float, default=60.0); ap.add_argument('--min-width-um', type=float, default=15.0)
ap.add_argument('--ref-offset-um', type=float, default=1.5); ap.add_argument('--bin-um', type=float, default=5.0)
ap.add_argument('--min-grain-px', type=int, default=50)
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)

t0 = time.time(); r = read_ang(a.ang, ci_threshold=a.ci)
q, valid, topo = r['quaternions'], r['valid_mask'], r['grid_topology']
md = r['metadata']; hexgrid = md['grid_type'] == 'HexGrid'
dy = md['y_step']; dx = md['x_step']                       # row / column spacing (um)
if a.rot90:
    if hexgrid: raise SystemExit('rot90 needs a square grid: use the rot90_square_native export')
    q = np.rot90(q, 1, axes=(0, 1)).copy(); valid = np.rot90(valid, 1, axes=(0, 1)).copy(); topo = None; dx, dy = dy, dx
print(f'{a.label}: {q.shape} {md["grid_type"]} step {dx} x {dy} um, valid {valid.mean():.3f}, read {time.time()-t0:.0f} s', flush=True)

t0 = time.time()
lab, _n = segment_grains_2d(q, valid_mask=valid, threshold_deg=a.seg_deg, crystal_structure='cubic',
                            min_grain_size=a.min_grain_px, grid_topology=topo)
print(f'segmented at {a.seg_deg} deg: {len(np.unique(lab[lab > 0]))} grains, {time.time()-t0:.0f} s', flush=True)

# height coordinate: row index * dy, increasing with the build direction (or reversed by --vflip)
nrow, ncol = lab.shape
hrow = np.arange(nrow) * dy
if a.vflip: hrow = hrow[::-1]
ids, counts = np.unique(lab[lab > 0], return_counts=True)
order = np.argsort(lab, axis=None, kind='stable'); flat = lab.ravel()[order]
coords = np.column_stack(np.unravel_index(order, lab.shape))
starts = np.searchsorted(flat, ids); ends = starts + counts
srod = []; hgt = []; per_grain = []
for gi, (s_, e_) in enumerate(zip(starts, ends)):
    c = coords[s_:e_]; h = hrow[c[:, 0]]
    height = h.max() - h.min(); width = (c[:, 1].max() - c[:, 1].min() + 1) * dx
    if height < a.min_height_um or width < a.min_width_um: continue
    zref = h.min() + a.ref_offset_um
    band = c[np.abs(h - zref) <= dy]                     # pixels in the reference row(s)
    if len(band) == 0: continue
    cen = band[:, 1].mean(); o = band[np.argmin(np.abs(band[:, 1] - cen))]
    qo = q[o[0], o[1]]; m = h >= zref; cm = c[m]
    sv = disorientation_deg(np.broadcast_to(qo, (len(cm), 4)), q[cm[:, 0], cm[:, 1]])
    hv = h[m] - zref
    srod.append(sv); hgt.append(hv)
    # per-grain: mean SROD per 30 um slab at the top, max, drift of slab-mean orientation approximated by slab-mean SROD
    per_grain.append(dict(id=int(ids[gi]), height_um=float(height), width_um=float(width), area_px=int(counts[gi]),
                          max_srod=float(sv.max()), mean_srod_top30=float(sv[hv >= hv.max() - 30].mean())))
if not srod: raise SystemExit('no grains passed the filters')
srod = np.concatenate(srod); hgt = np.concatenate(hgt)
bins = np.arange(0, hgt.max() + a.bin_um, a.bin_um); rows = []
for lo in bins[:-1]:
    mm = (hgt >= lo) & (hgt < lo + a.bin_um) & (hgt > 0)
    if mm.sum() < 200: continue
    sv = srod[mm]; p5, p95 = np.percentile(sv, [5, 95])
    rows.append((float(hgt[mm].mean()), float(sv.mean()), float(p5), float(p95), float(np.sqrt((sv**2).mean())), int(mm.sum())))
rows = np.array(rows)
# RMS^2 = rms0^2 + D h over 5-150 um, weighted by sqrt(n) (established-type: intercept allowed; same fit as srod_from_stitch.py)
mfit = (rows[:, 0] >= 5) & (rows[:, 0] <= 150); wfit = np.sqrt(rows[mfit, 5])
A = np.column_stack([np.ones(mfit.sum()), rows[mfit, 0]]); coef = np.linalg.lstsq(A * wfit[:, None], (rows[mfit, 4]**2) * wfit, rcond=None)[0]
D, rms0 = coef[1], np.sqrt(max(coef[0], 0.0))
res = dict(label=a.label, ang=a.ang, grains=len(per_grain), seg_deg=a.seg_deg, filters=dict(min_height_um=a.min_height_um, min_width_um=a.min_width_um),
           D_deg2_per_um=float(D), rms0_deg=float(rms0), band_over_mean_median=float(np.median((rows[:, 3]-rows[:, 2])/rows[:, 1])),
           curve=dict(h=rows[:, 0].tolist(), mean=rows[:, 1].tolist(), p5=rows[:, 2].tolist(), p95=rows[:, 3].tolist(), rms=rows[:, 4].tolist(), n=rows[:, 5].tolist()),
           per_grain=per_grain)
json.dump(res, open(os.path.join(a.out, f'srod_{a.label}.json'), 'w'))
pg = np.array([[g['height_um'], g['max_srod'], g['mean_srod_top30']] for g in per_grain])
print(f'{a.label}: {len(per_grain)} grains (height >= {a.min_height_um} um, width >= {a.min_width_um} um); '
      f'D = {D:.3f} deg^2/um, initial spread {rms0:.2f} deg, band/mean {res["band_over_mean_median"]:.2f}; '
      f'per grain: max SROD median {np.median(pg[:,1]):.1f} (max {pg[:,1].max():.1f}) deg, grains > 25 deg: {np.mean(pg[:,1] > 25):.2f}; '
      f'top-30um mean SROD median {np.median(pg[:,2]):.1f} deg', flush=True)
for r_ in rows[::max(1, len(rows)//8)]: print(f'   h={r_[0]:6.1f} n={int(r_[5]):8d} mean {r_[1]:5.2f} p5 {r_[2]:5.2f} p95 {r_[3]:5.2f} rms {r_[4]:5.2f}')
