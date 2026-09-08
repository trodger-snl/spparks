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
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); dxu = a.dx*1e6
with StitchReader(a.stitch) as r:
    b = r.get_global_bounds(); t = r.times[-1]
    f = {k: r.read_field_at_time(k, t, bounds=b)[0] for k in ('i1','i2','d1','d2','d3','d4')}
if a.crop:
    c = [int(round(v/dxu)) for v in a.crop]; f = {k: v[c[0]:c[1], c[2]:c[3], :] for k, v in f.items()}
solid = f['i2'] == 3
q = np.stack([f['d1'],f['d2'],f['d3'],f['d4']], -1)
lab = relabel_grains(np.where(solid, f['i1'], 0).astype(np.int64))
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
    tt = np.linspace(0, np.pi/4, 60)                  # 101 -> 111 edge: directions (cos t, sin t, 1)/norm
    e = np.stack([np.cos(tt), np.sin(tt), np.ones_like(tt)], 1); e /= np.linalg.norm(e, axis=1)[:,None]
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
