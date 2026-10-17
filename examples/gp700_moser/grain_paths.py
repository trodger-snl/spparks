#!/usr/bin/env python3
"""Per-grain orientation analysis in the style of Lamb et al. 2023 Fig. 4:
for the N tallest grains passing Lamb's filters, (a) IPF-Z map of the grain on
the xz slice through its centroid, (b) SROD map on the same slice, (c) mean
SROD per height slab, (d) the grain's orientations on the inverse pole figure
of the build direction (standard triangle 001-101-111), coloured by height,
with the per-slab mean path and an arrow from the bottom slab to the top.
usage (WORK env): grain_paths.py gp700.st --out DIR [--dx 3e-6] [--crop X0 X1 Y0 Y1] [--floor-um 100] [--n 4]
"""
import argparse, os, sys
import numpy as np
sys.path.insert(0, '/home/tron/spparks-micro-analysis/src')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'scripts'))
from spparks_micro_analysis.stitch_reader import StitchReader
from spparks_micro_analysis.grain_size_analysis import relabel_grains
from spparks_micro_analysis.orientation_analysis import generate_ipf_colors
from srod_analysis import disorientation_deg
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

ap = argparse.ArgumentParser(); ap.add_argument('stitch'); ap.add_argument('--out', default='grain_paths')
ap.add_argument('--dx', type=float, default=3e-6); ap.add_argument('--floor-um', type=float, default=100.0)
ap.add_argument('--crop', type=float, nargs=4, default=None); ap.add_argument('--n', type=int, default=4)
ap.add_argument('--slab-um', type=float, default=10.0)
ap.add_argument('--segment-deg', type=float, default=None, help='re-segment grains at this disorientation threshold (deg)')
ap.add_argument('--grid', type=int, default=24, help='grains in the compact IPF-path grid (tallest first)')
ap.add_argument('--lamb-pick', default='spread', choices=['tallest', 'spread'], help='grains for the Lamb-style figure: the N tallest, or N tall grains (>= --lamb-min-height-um) whose mean orientations are farthest apart on the IPF triangle')
ap.add_argument('--lamb-min-height-um', type=float, default=300.0)
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); dxu = a.dx*1e6
with StitchReader(a.stitch) as r:
    b = r.get_global_bounds(); t = r.times[-1]
    fields = ('i1','i2','d1','d2','d3','d4') + (('d10',) if 'd10' in r.get_available_fields() else ())
    f = {k: r.read_field_at_time(k, t, bounds=b)[0] for k in fields}
if a.crop:
    c = [int(round(v/dxu)) for v in a.crop]; f = {k: v[c[0]:c[1], c[2]:c[3], :] for k, v in f.items()}
solid = f['i2'] == 3
q = np.stack([f['d1'],f['d2'],f['d3'],f['d4']], -1)
if a.segment_deg:
    from spparks_micro_analysis.grain_segmentation import segment_grains
    lab = np.where(solid, segment_grains(q, occupied=solid, threshold_deg=a.segment_deg, crystal_structure='cubic')[0], 0).astype(np.int64)
    print(f'grains re-segmented at {a.segment_deg} deg')
else:
    lab = relabel_grains(np.where(solid, f['i1'], 0).astype(np.int64))
spath = f.get('d10')
nx, ny, nz = solid.shape
X, Y, Z = np.indices(solid.shape)
ids, inv, vol = np.unique(lab[solid], return_inverse=True, return_counts=True)
zmin = np.full(len(ids), 10**9); np.minimum.at(zmin, inv, Z[solid]); zmax = np.zeros(len(ids), int); np.maximum.at(zmax, inv, Z[solid])
height = (zmax - zmin + 1)*dxu; esd = (6*vol*dxu**3/np.pi)**(1/3)
ok = (ids > 0) & (height >= 60) & (esd >= 40) & (zmin*dxu >= a.floor_um)
order = np.argsort(-(height[ok]*1e6 + vol[ok]))            # tallest first, volume breaks ties
pick = ids[ok][order][:a.n]
print(f'{len(ids)} grains, {ok.sum()} pass filters; analysing {list(pick)}')

# ---- inverse pole figure of the build direction: sample z in the crystal frame, cubic fundamental triangle
def bd_in_crystal(qq):
    w,x,y,z = qq[:,0],qq[:,1],qq[:,2],qq[:,3]
    return np.stack([2*(x*z - w*y), 2*(y*z + w*x), 1-2*(x*x+y*y)], 1)
def triangle_xy(v):
    s = np.sort(np.abs(v), axis=1)[:, ::-1]        # c >= a >= b >= 0 with c the [001] component
    c, aa, bb = s[:,0], s[:,1], s[:,2]
    return aa/(1+c), bb/(1+c)                       # stereographic projection from -[001]; 001->(0,0), 101->(0.414,0), 111->(0.366,0.366)
def draw_triangle(ax):
    ss = np.linspace(0, 1, 60)                        # 101 -> 111 edge: directions (1, s, 1)/norm (x = z, the c = a boundary)
    e = np.stack([np.ones_like(ss), ss, np.ones_like(ss)], 1); e /= np.linalg.norm(e, axis=1)[:,None]
    ex, ey = e[:,0]/(1+e[:,2]), e[:,1]/(1+e[:,2])
    ax.plot([0, ex[0]], [0, 0], color='#444', lw=1); ax.plot(ex, ey, color='#444', lw=1); ax.plot([0, ex[-1]], [0, ey[-1]], color='#444', lw=1)
    for (x_, y_, s_) in ((0, 0, '001'), (ex[0], 0, '101'), (ex[-1], ey[-1], '111')):
        ax.text(x_, y_-0.03 if y_ == 0 else y_+0.015, s_, ha='center', va='top' if y_ == 0 else 'bottom', fontsize=9)
    ax.set_xlim(-0.05, 0.47); ax.set_ylim(-0.07, 0.42); ax.set_aspect('equal'); ax.axis('off')

rgb_all = None
fig, axes = plt.subplots(len(pick), 4, figsize=(17, 4.2*len(pick)), squeeze=False)
for row, g in enumerate(pick):
    m = lab == g; idx = np.argwhere(m); qq = q[m]
    zc = idx[:,2]; o = idx[np.argmin(zc)]                                    # origin = lowest voxel
    srod = disorientation_deg(np.broadcast_to(q[tuple(o)], (len(qq),4)), qq)
    h = (zc - o[2])*dxu
    # (a) IPF map and (b) SROD map on the xz slice through the grain centroid (grain only)
    yc = int(round(idx[:,1].mean()))
    sl = m[:, yc, :]
    rgb = generate_ipf_colors(q[:, yc, :, :].reshape(-1,4), 'fcc', 'z').reshape(nx, nz, 3)
    img = np.ones((nx, nz, 3))*0.92; img[sl] = rgb[sl]
    ax = axes[row,0]; ax.imshow(np.transpose(img, (1,0,2)), origin='lower', extent=[0, nx*dxu, 0, nz*dxu]); ax.set_title(f'grain {g}: IPF-Z, xz slice at y = {yc*dxu:.0f} um', loc='left', fontsize=10)
    sm = np.full((nx, nz), np.nan); sm[sl] = disorientation_deg(np.broadcast_to(q[tuple(o)], (sl.sum(),4)), q[:, yc, :, :][sl])
    ax = axes[row,1]; im = ax.imshow(sm.T, origin='lower', extent=[0, nx*dxu, 0, nz*dxu], cmap='viridis', vmin=0, vmax=max(10, np.nanpercentile(srod, 99)))
    ax.set_title('SROD on the slice (deg)', loc='left', fontsize=10); plt.colorbar(im, ax=ax, fraction=0.04)
    for ax in axes[row,:2]: ax.set_xlabel('x (um)'); ax.set_ylabel('z (um)'); ax.set_xlim(max(0, (idx[:,0].min()-15)*dxu), min(nx, idx[:,0].max()+15)*dxu)
    # (c) mean SROD per height slab
    edges = np.arange(0, h.max()+a.slab_um, a.slab_um); mids, means, p5s, p95s = [], [], [], []
    for lo in edges[:-1]:
        mm = (h >= lo) & (h < lo+a.slab_um)
        if mm.sum() < 20: continue
        mids.append(lo + a.slab_um/2); means.append(srod[mm].mean()); p5s.append(np.percentile(srod[mm], 5)); p95s.append(np.percentile(srod[mm], 95))
    ax = axes[row,2]; ax.fill_between(mids, p5s, p95s, color='#2a78d6', alpha=0.15, lw=0); ax.plot(mids, means, color='#2a78d6', lw=2)
    ax.set_xlabel('height above origin (um)'); ax.set_ylabel('SROD (deg)'); ax.set_title(f'mean SROD per {a.slab_um:.0f} um slab (height {h.max():.0f} um, ESD {esd[ids==g][0]:.0f} um)', loc='left', fontsize=10); ax.grid(axis='y', color='#e6e5e1')
    # (d) IPF projection of every voxel, coloured by height; per-slab mean path; arrow bottom -> top
    ax = axes[row,3]; draw_triangle(ax)
    px, py = triangle_xy(bd_in_crystal(qq))
    sc = ax.scatter(px, py, c=h, s=2, cmap='viridis', alpha=0.5, linewidths=0)
    path = []
    for lo in edges[:-1]:
        mm = (h >= lo) & (h < lo+a.slab_um)
        if mm.sum() < 20: continue
        v = bd_in_crystal(qq[mm]); v = np.sort(np.abs(v), axis=1)[:, ::-1]; vm = v.mean(0); vm /= np.linalg.norm(vm)
        path.append((vm[1]/(1+vm[0]), vm[2]/(1+vm[0])))
    path = np.array(path); ax.plot(path[:,0], path[:,1], color='k', lw=1.2)
    ax.annotate('', xy=path[-1], xytext=path[0], arrowprops=dict(arrowstyle='->', color='#c9552a', lw=2))
    plt.colorbar(sc, ax=ax, fraction=0.04, label='height above origin (um)')
    ax.set_title('IPF of the build direction, bottom -> top', loc='left', fontsize=10)
    def mean_q(qs):                       # sign-aligned mean quaternion
        qs = np.where((qs @ qs[0])[:,None] < 0, -qs, qs); v = qs.mean(0); return v/np.linalg.norm(v)
    qb = mean_q(qq[h < 3*a.slab_um]); qt = mean_q(qq[h >= h.max()-3*a.slab_um])
    drift = disorientation_deg(qb[None,:], qt[None,:])[0]
    top = h >= h.max()-3*a.slab_um; spread_top = disorientation_deg(np.broadcast_to(qt,(top.sum(),4)), qq[top])
    print(f'grain {g}: height {h.max():.0f} um, ESD {esd[ids==g][0]:.0f} um, voxels {m.sum()}, mean SROD top slab {means[-1]:.1f} deg, '
          f'max SROD {srod.max():.1f} deg; mean-orientation drift bottom->top {drift:.1f} deg; spread about the top-slab mean: RMS {np.sqrt((spread_top**2).mean()):.1f} deg')
fig.tight_layout(); fig.savefig(f'{a.out}/grain_paths.png', dpi=140); print(f'wrote {a.out}/grain_paths.png')

# ---- every filtered grain: drift / spread / max SROD / top-slab mean, a compact IPF-path grid, and a summary figure
def mean_q(qs):
    qs = np.where((qs @ qs[0])[:,None] < 0, -qs, qs); v = qs.mean(0); return v/np.linalg.norm(v)
allg = ids[ok][order]; rows = []; paths = {}
for g in allg:
    m = lab == g; idx = np.argwhere(m); qq = q[m]; zc = idx[:,2]; o = idx[np.argmin(zc)]
    h = (zc - o[2])*dxu; srod = disorientation_deg(np.broadcast_to(q[tuple(o)], (len(qq),4)), qq)
    lo_m = h < 3*a.slab_um; hi_m = h >= h.max()-3*a.slab_um
    if lo_m.sum() < 20 or hi_m.sum() < 20: continue
    qb = mean_q(qq[lo_m]); qt = mean_q(qq[hi_m])
    drift = disorientation_deg(qb[None,:], qt[None,:])[0]
    spread = np.sqrt((disorientation_deg(np.broadcast_to(qt,(hi_m.sum(),4)), qq[hi_m])**2).mean())
    edges = np.arange(0, h.max()+a.slab_um, a.slab_um); means = []; pth = []
    for lo in edges[:-1]:
        mm = (h >= lo) & (h < lo+a.slab_um)
        if mm.sum() < 20: continue
        means.append(srod[mm].mean())
        v = np.sort(np.abs(bd_in_crystal(qq[mm])), axis=1)[:, ::-1].mean(0); v /= np.linalg.norm(v); pth.append((v[1]/(1+v[0]), v[2]/(1+v[0])))
    means = np.array(means); mono = float(np.mean(np.diff(means) >= 0)) if len(means) > 1 else np.nan
    # Lamb-style per-layer (30 um) mean SROD at 300 um, and the path factor from d10 (path length / height)
    l300 = (h >= 285) & (h < 315); mean300 = float(srod[l300].mean()) if l300.sum() >= 20 else np.nan
    if spath is not None:
        sp = spath[m] - spath[tuple(o)]; hh = h/dxu; sel = hh >= 10
        pathf = float(np.median(sp[sel]/hh[sel])) if sel.sum() > 20 else np.nan
    else: pathf = np.nan
    rows.append((int(g), float(h.max()), float(esd[ids==g][0]), int(m.sum()), drift, spread, float(srod.max()), float(means[-1]), mono, mean300, pathf))
    paths[int(g)] = np.array(pth)
import csv
with open(f'{a.out}/grain_stats.csv', 'w', newline='') as fcsv:
    w = csv.writer(fcsv); w.writerow(['grain','height_um','esd_um','voxels','drift_deg','spread_top_rms_deg','max_srod_deg','top_slab_mean_deg','monotonic_fraction','mean_srod_at_300um_deg','path_per_height']); w.writerows(rows)
R = np.array([r[4:] for r in rows]); print(f'{len(rows)} filtered grains: drift median {np.median(R[:,0]):.1f} deg (p10/p90 {np.percentile(R[:,0],10):.1f}/{np.percentile(R[:,0],90):.1f}), '
      f'spread median {np.median(R[:,1]):.1f}, max SROD median {np.median(R[:,2]):.1f}, top-slab mean median {np.median(R[:,3]):.1f}, drift/spread median {np.median(R[:,0]/R[:,1]):.2f}, monotonic fraction median {np.nanmedian(R[:,4]):.2f}')
Hh = np.array([r[1] for r in rows]); rate = R[:,0]/Hh*30.0
print(f'Lamb-style: grains with max SROD > 25 deg: {np.mean(R[:,2] > 25):.2f}; max SROD p90/max {np.percentile(R[:,2],90):.1f}/{R[:,2].max():.1f} deg; '
      f'per-layer mean SROD at 300 um (grains taller than 315 um, n={np.sum(~np.isnan(R[:,5]))}): median {np.nanmedian(R[:,5]):.1f}, p90 {np.nanpercentile(R[:,5],90):.1f}, max {np.nanmax(R[:,5]):.1f} deg; '
      f'accumulation rate (drift per 30 um): mean {rate.mean():.2f}, max {rate.max():.2f} deg/30um; path length per height: median {np.nanmedian(R[:,6]):.2f}')
ng = min(a.grid, len(rows)); nc = 6; nr = int(np.ceil(ng/nc))
fig, axes = plt.subplots(nr, nc, figsize=(2.6*nc, 2.5*nr), squeeze=False)
for k, ax in enumerate(axes.ravel()):
    if k >= ng: ax.axis('off'); continue
    g = rows[k][0]; draw_triangle(ax); pth = paths[g]
    ax.scatter(pth[:,0], pth[:,1], c=np.arange(len(pth)), cmap='viridis', s=6, linewidths=0)
    ax.plot(pth[:,0], pth[:,1], color='k', lw=0.6)
    ax.annotate('', xy=pth[-1], xytext=pth[0], arrowprops=dict(arrowstyle='->', color='#c9552a', lw=1.2))
    ax.set_title(f'{g}: h {rows[k][1]:.0f} um, drift {rows[k][4]:.1f}, spread {rows[k][5]:.1f}', fontsize=7.5, loc='left')
fig.suptitle(f'per-slab mean orientation paths on the build-direction IPF, {ng} tallest filtered grains (colour: height)', fontsize=10)
fig.tight_layout(); fig.savefig(f'{a.out}/grain_paths_grid.png', dpi=130)
fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))
ax[0].scatter(R[:,1], R[:,0], s=14, color='#2a78d6', alpha=0.7); lim = max(R[:,0].max(), R[:,1].max())*1.05
ax[0].plot([0,lim],[0,lim], color='#888', lw=0.8, ls='--'); ax[0].set_xlabel('spread about the top-slab mean, RMS (deg)'); ax[0].set_ylabel('drift of the mean, bottom -> top (deg)'); ax[0].set_title(f'{len(rows)} filtered grains', loc='left', fontsize=10)
H = np.array([r[1] for r in rows]); ax[1].scatter(H, R[:,0], s=14, color='#2a78d6', alpha=0.7, label='drift'); ax[1].scatter(H, R[:,2], s=14, color='#c9552a', alpha=0.6, label='max SROD'); ax[1].set_xlabel('grain height (um)'); ax[1].set_ylabel('deg'); ax[1].legend(frameon=False); ax[1].set_title('drift and max SROD vs height', loc='left', fontsize=10)
ax[2].hist(R[:,4][~np.isnan(R[:,4])], bins=np.linspace(0,1,11), color='#2a78d6', alpha=0.8); ax[2].set_xlabel('fraction of slabs where the mean SROD increases'); ax[2].set_ylabel('grains'); ax[2].set_title('monotonicity of SROD(h)', loc='left', fontsize=10)
for a_ in ax: a_.grid(color='#e6e5e1', lw=0.6)
fig.tight_layout(); fig.savefig(f'{a.out}/grain_summary.png', dpi=140); print(f'wrote {a.out}/grain_paths_grid.png, grain_summary.png, grain_stats.csv')

# ---- Lamb 2024 Fig. 4(c,d) style for the N analysed grains: mean SROD per 30 um layer against height, and every voxel on the
# build-direction IPF in its own IPF-Z colour with a black path (per-slab means, arrowhead at the top) -> grain_ipf_lamb.png
picked = [g for g in allg[:a.n]]
if a.lamb_pick == 'spread':      # greedy farthest-point selection on the triangle among tall grains, seeded by the tallest
    cand = [g for g in allg if height[ids == g][0] >= a.lamb_min_height_um] or list(allg[:max(a.n, 12)])
    pos = {}
    for g in cand:
        v = np.sort(np.abs(bd_in_crystal(mean_q(q[lab == g])[None, :])), axis=1)[0, ::-1]; pos[g] = np.array([v[1]/(1+v[0]), v[2]/(1+v[0])])
    picked = [cand[0]]
    while len(picked) < min(a.n, len(cand)):
        picked.append(max((g for g in cand if g not in picked), key=lambda g: min(np.linalg.norm(pos[g]-pos[h]) for h in picked)))
    print('Lamb-style figure grains (spread on the IPF):', [f'{g} h={height[ids==g][0]:.0f}um' for g in picked])
# curve and label colour = IPF-Z colour of the grain's mean orientation, as the clouds and maps (Lamb's labels match their grains)
gcol = {g: tuple(np.clip(generate_ipf_colors(mean_q(q[lab == g])[None, :], 'fcc', 'z')[0], 0, 1)) for g in picked}
fig = plt.figure(figsize=(3.6 + 2.6*len(picked), 3.4)); gs = fig.add_gridspec(1, 1 + len(picked), width_ratios=[1.4] + [1]*len(picked), wspace=0.12)
axc = fig.add_subplot(gs[0, 0]); layer = 30.0
for k, g in enumerate(picked):
    m = lab == g; idx = np.argwhere(m); qq = q[m]; zc = idx[:,2]; o = idx[np.argmin(zc)]
    h = (zc - o[2])*dxu; srod = disorientation_deg(np.broadcast_to(q[tuple(o)], (len(qq),4)), qq)
    edges = np.arange(0, h.max()+layer, layer); mids, means = [], []
    for lo in edges[:-1]:
        mm = (h >= lo) & (h < lo+layer)
        if mm.sum() >= 20: mids.append(lo + layer/2); means.append(srod[mm].mean())
    axc.plot(means, mids, color=gcol[g], lw=2.0, label=f'G{k+1}')
    axd = fig.add_subplot(gs[0, 1+k]); draw_triangle(axd)
    sub = np.random.default_rng(0).choice(len(qq), min(len(qq), 40000), replace=False)
    px, py = triangle_xy(bd_in_crystal(qq[sub])); rgb = generate_ipf_colors(qq[sub], 'fcc', 'z')
    axd.scatter(px, py, c=np.clip(rgb, 0, 1), s=1.5, linewidths=0, alpha=0.6)
    pth = np.array(paths[g]) if g in paths and len(paths[g]) > 1 else None
    if pth is not None:
        axd.plot(pth[:,0], pth[:,1], color='k', lw=1.3); axd.annotate('', xy=pth[-1], xytext=pth[-2], arrowprops=dict(arrowstyle='-|>', color='k', lw=1.3, mutation_scale=10))
    axd.text(0.02, 0.36, f'G{k+1}', color=gcol[g], fontsize=11, fontweight='bold')
    axd.text(0.24, 0.36, f'max {srod.max():.0f}°', color=gcol[g], fontsize=8)
axc.set_xlabel('SROD (°), mean per 30 µm layer'); axc.set_ylabel('height above origin (µm)'); axc.grid(alpha=0.3); axc.legend(fontsize=8, frameon=False, loc='lower right')
fig.savefig(f'{a.out}/grain_ipf_lamb.png', dpi=160, bbox_inches='tight'); print(f'wrote {a.out}/grain_ipf_lamb.png')
