#!/usr/bin/env python3
"""GROD against SROD on one IN625 longitudinal EBSD section (HPC-Calibration P15, nominal build).
Left: IPF-Z with 2-degree grain boundaries. Middle: GROD, misorientation of every pixel from its grain's
MEAN orientation. Right: SROD, misorientation from the orientation at the grain's lowest point (its
origin in the section). Bottom: both along the build direction inside one tall grain.
Text-free apart from axes so the slide carries the explanation.  usage (WORK env): grod_srod_example.py OUT.png"""
import sys, os, numpy as np
sys.path.insert(0, '/home/tron/spparks-micro-analysis/src'); sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'scripts'))
from spparks_micro_analysis.ebsd_reader import read_ang
from spparks_micro_analysis.grain_segmentation import segment_grains_2d
from spparks_micro_analysis.orientation_analysis import generate_ipf_colors
from srod_analysis import disorientation_deg, qmul, SYM
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

r = read_ang('/home/tron/HPC-Calibration/ANG-Files/P15/P15-2-side.ang', ci_threshold=0.1)
r0, c0, H, W = 1000, 1000, 1400, 1200                       # 600 x 600 um crop (rows 0.433 um, cols 0.5 um)
q = r['quaternions'][r0:r0+H, c0:c0+W]; v = r['valid_mask'][r0:r0+H, c0:c0+W]
q = q[::-1]; v = v[::-1]                                     # vflip (P15 XZ): build direction = increasing row
lab, _ = segment_grains_2d(q, valid_mask=v, threshold_deg=2.0, crystal_structure='cubic', min_grain_size=50); lab = np.where(v, lab, 0)
dy, dx = 0.433013, 0.5
ids, counts = np.unique(lab[lab > 0], return_counts=True)
grod = np.full(lab.shape, np.nan); srod = np.full(lab.shape, np.nan)
SYMQ = np.array([s for s in SYM])
def mean_orientation(qs, qref):
    """average of the symmetry equivalents of qs closest to qref (sign-aligned), normalised"""
    best = np.zeros(len(qs)); bq = qs.copy()
    for s in SYMQ:
        cand = qmul(qs, s[None, :]); d = np.abs((cand * qref).sum(1)); m = d > best; best[m] = d[m]; bq[m] = cand[m]
    bq *= np.sign((bq * qref).sum(1))[:, None]; mq = bq.mean(0); return mq / np.linalg.norm(mq)
tallest = None
for g, n in zip(ids, counts):
    if n < 200: continue
    rr, cc = np.nonzero(lab == g); qs = q[rr, cc]
    ref = qs[np.argmin((rr - rr.mean())**2 + (cc - cc.mean())**2)]; mq = mean_orientation(qs, ref)
    grod[rr, cc] = disorientation_deg(np.broadcast_to(mq, qs.shape), qs)
    band = rr <= rr.min() + 3; o = np.argmin(np.abs(cc[band] - cc[band].mean())); qo = qs[band][o]
    srod[rr, cc] = disorientation_deg(np.broadcast_to(qo, qs.shape), qs)
    height = (rr.max() - rr.min()) * dy
    if tallest is None or height > tallest[0]: tallest = (height, g, rr, cc)
rgb = generate_ipf_colors(q.reshape(-1, 4), 'fcc', 'z').reshape(H, W, 3); rgb[~v] = 0.15
b = np.zeros((H, W), bool); b[1:, :] |= lab[1:, :] != lab[:-1, :]; b[:, 1:] |= lab[:, 1:] != lab[:, :-1]; rgb[b] = 0
ext = [0, W*dx, 0, H*dy]
fig = plt.figure(figsize=(13.5, 6.4)); gs = fig.add_gridspec(2, 3, height_ratios=[3.2, 1.3], hspace=0.28, wspace=0.12)
ax0 = fig.add_subplot(gs[0, 0]); ax0.imshow(rgb, origin='lower', extent=ext); ax0.set_ylabel('build direction (um)')
ax1 = fig.add_subplot(gs[0, 1], sharey=ax0); im1 = ax1.imshow(grod, origin='lower', extent=ext, cmap='viridis', vmin=0, vmax=8)
ax2 = fig.add_subplot(gs[0, 2], sharey=ax0); im2 = ax2.imshow(srod, origin='lower', extent=ext, cmap='viridis', vmin=0, vmax=8)
for ax_ in (ax1, ax2): ax_.tick_params(labelleft=False)
for ax_, t in ((ax0, 'IPF-Z, 2° grain boundaries'), (ax1, 'GROD (deg): from the grain mean'), (ax2, 'SROD (deg): from the grain origin')): ax_.set_title(t, loc='left', fontsize=10); ax_.set_xlabel('um')
fig.colorbar(im2, ax=[ax1, ax2], fraction=0.025, pad=0.02, label='deg')
# highlight the tallest grain and its profiles
height, g, rr, cc = tallest; m = lab == g; edge = np.zeros_like(m); edge[1:] |= m[1:] != m[:-1]; edge[:, 1:] |= m[:, 1:] != m[:, :-1]; edge &= m
ys, xs = np.nonzero(edge)
for ax_ in (ax0, ax1, ax2): ax_.scatter(xs*dx, ys*dy, s=0.3, color='w' if ax_ is not ax0 else 'k')
hrow = (rr - rr.min()) * dy; bins = np.arange(0, hrow.max() + 10, 10)
axp = fig.add_subplot(gs[1, :])
for arr, col, name in ((grod, '#0075a9', 'GROD'), (srod, '#c9552a', 'SROD')):
    vals = arr[rr, cc]; prof = [vals[(hrow >= lo) & (hrow < lo + 10)].mean() for lo in bins[:-1]]
    axp.plot(bins[:-1] + 5, prof, color=col, lw=2, label=name)
axp.set_xlabel('height above the lowest point of the outlined grain (um)'); axp.set_ylabel('deg'); axp.grid(alpha=0.3); axp.legend(fontsize=9, frameon=False)
fig.savefig(sys.argv[1], dpi=150, bbox_inches='tight'); print('saved', sys.argv[1], 'tallest grain', round(height), 'um')
