#!/usr/bin/env python3
"""GP700 build analysis from the stitch file (Plan 2 gates 5.2/5.3).
Grain structure (sizes, columnar heights, texture), IPF cross-sections, pole
figures, and origin-referenced SROD vs distance with Lamb et al.'s grain
filters and originating/established classes.

usage (WORK conda env):
  python gp700_analysis.py gp700.st --out analysis/ [--dx 3e-6] [--floor-um 90] [--D 4.2e5]
"""
import argparse, os, sys, json
import numpy as np
sys.path.insert(0, '/home/tron/spparks-micro-analysis/src')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'scripts'))
from spparks_micro_analysis.stitch_reader import StitchReader
from spparks_micro_analysis.grain_size_analysis import relabel_grains
from spparks_micro_analysis.orientation_analysis import generate_ipf_colors, create_pole_density_plots_from_quaternions
from srod_analysis import disorientation_deg
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument('stitch'); ap.add_argument('--out', default='analysis')
ap.add_argument('--dx', type=float, default=3e-6); ap.add_argument('--floor-um', type=float, default=90.0)
ap.add_argument('--D', type=float, default=None, help='input D, deg^2/m (for the x-input ratio and the walk reference line)'); ap.add_argument('--time', type=float, default=None)
ap.add_argument('--top-um', type=float, default=None, help='ignore everything above this height (um), e.g. the last never-remelted layer')
ap.add_argument('--crop', type=float, nargs=4, metavar=('X0','X1','Y0','Y1'), default=None, help='analyse only this lateral window (um)')
ap.add_argument('--lamb', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'docs'), help='dir with lamb2023_fig5{a,c}_srod.json')
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); dxu = a.dx*1e6
with StitchReader(a.stitch) as r:
    b = r.get_global_bounds(); t = a.time if a.time is not None else r.times[-1]
    f = {k: r.read_field_at_time(k, t, bounds=b)[0] for k in ('i1','i2','d1','d2','d3','d4')}
if a.crop:
    c = [int(round(v/(a.dx*1e6))) for v in a.crop]
    f = {k: v[c[0]:c[1], c[2]:c[3], :] for k, v in f.items()}
    print(f'cropped to x {a.crop[0]:.0f}-{a.crop[1]:.0f} um, y {a.crop[2]:.0f}-{a.crop[3]:.0f} um')
if a.top_um is not None:
    zt = int(round(a.top_um/(a.dx*1e6))); f = {k: v[:, :, :zt] for k, v in f.items()}
    print(f'cropped to z < {a.top_um:.0f} um')
solid = f['i2'] == 3
spin = np.where(solid, f['i1'], 0).astype(np.int64)
q = np.stack([f['d1'],f['d2'],f['d3'],f['d4']], -1)
nx,ny,nz = solid.shape
print(f'{a.stitch} t={t}: grid {solid.shape}, solid {solid.sum()} ({solid.mean():.3f}), liquid {(f["i2"]==2).sum()}')
zs = np.arange(nz); sf = solid.mean(axis=(0,1))
print('solid fraction by z (um): ' + ' '.join(f'{z*dxu:.0f}:{v:.2f}' for z,v in zip(zs[::10], sf[::10])))

# ---- grains: same-spin face-connected regions (O(N) union-find)
lab = relabel_grains(spin)
ids, counts = np.unique(lab[lab>0], return_counts=True)
idx = np.argsort(lab, axis=None, kind='stable'); flat = lab.ravel()[idx]; coords = np.column_stack(np.unravel_index(idx, lab.shape))
starts = np.searchsorted(flat, ids); ends = starts + counts
zmin = np.array([coords[s:e,2].min() for s,e in zip(starts,ends)]); zmax = np.array([coords[s:e,2].max() for s,e in zip(starts,ends)])
height = (zmax - zmin + 1) * dxu; esd = 2*(3*counts*dxu**3/(4*np.pi))**(1/3)
ext = np.array([np.ptp(coords[s:e], axis=0) + 1 for s,e in zip(starts,ends)]) * dxu   # x,y,z extents (um)
big = counts >= 100
print('grain extents (um, grains >= 100 voxels, volume-weighted median): ' +
      ' '.join(f'{ax} {np.median(np.repeat(ext[big,i], counts[big])):.0f}' for i,ax in enumerate('xyz')) +
      f'; grains longer than 300 um along x: {(ext[:,0]>300).sum()}, along y: {(ext[:,1]>300).sum()}')
esd_w = np.repeat(esd, counts); q10,q50,q90 = np.percentile(esd_w,[10,50,90])
print(f'ESD volume-weighted p10/p50/p90: {q10:.1f} / {q50:.1f} / {q90:.1f} um')
# nearest-neighbour same-grain disorientation along z (per-hop step check)
mz = solid[:,:,:-1] & solid[:,:,1:] & (spin[:,:,:-1] == spin[:,:,1:])
nn = disorientation_deg(q[:,:,:-1][mz][:200000], q[:,:,1:][mz][:200000])
print(f'same-grain z-neighbour disorientation: median {np.median(nn):.2f} deg, p90 {np.percentile(nn,90):.2f} deg (n={mz.sum()})')
print(f'grains: {len(ids)}; volume-weighted median ESD {np.median(np.repeat(esd, counts)):.1f} um; '
      f'grains with height>=60um & ESD>=40um: {((height>=60)&(esd>=40)).sum()}; max height {height.max():.0f} um; '
      f'grains spanning >300 um: {(height>300).sum()}')
floor = a.floor_um/dxu

# ---- texture: <001> within 15 deg of the build direction
def axis_angle_to_z(qq):
    w,x,y,z = qq[:,0],qq[:,1],qq[:,2],qq[:,3]
    # rows of R^T = crystal axes in sample frame; take max |component along z| over the 3 <100>
    m = np.stack([2*(x*z - w*y), 2*(y*z + w*x), 1-2*(x*x+y*y)], 1)   # sample z expressed in crystal frame
    return np.degrees(np.arccos(np.clip(np.abs(m).max(1), 0, 1)))
sol_above = solid.copy(); sol_above[:,:,:int(floor)] = False
ang = axis_angle_to_z(q[sol_above])
print(f'texture above floor: <001> within 15 deg of BD: {np.mean(ang<15):.2f} (random ~0.09), within 10 deg: {np.mean(ang<10):.2f}')

# ---- SROD vs distance, Lamb's conventions: distance = height along the build
# direction; originating grains referenced to their origin voxel (lowest z),
# established grains to their cross-section centroid at the analysis floor
# (Lamb: "centroid of the grain on the lowest layer of the dataset").
keep = (height>=60)&(esd>=40)
dist_h=[]; dist_e=[]; srod=[]; cls=[]
for gi in np.where(keep)[0]:
    c = coords[starts[gi]:ends[gi]]
    if zmin[gi] >= floor:                      # originating: origin inside the volume
        o = c[np.argmin(c[:,2])]; k = 0; zref = o[2]
    else:                                      # established: reference at the floor
        cf = c[c[:,2] == int(floor)]
        if len(cf) == 0: continue
        cen = cf[:, :2].mean(0); o = cf[np.argmin(((cf[:, :2]-cen)**2).sum(1))]; k = 1; zref = floor
    m = c[:,2] >= zref; cm = c[m]
    qo = np.broadcast_to(q[tuple(o)], (len(cm),4))
    srod.append(disorientation_deg(qo, q[cm[:,0],cm[:,1],cm[:,2]]))
    dist_h.append((cm[:,2] - zref)*dxu)                       # height along BD (Lamb)
    dist_e.append(np.linalg.norm(cm - o, axis=1)*dxu)          # straight line from the reference voxel
    cls.append(np.full(m.sum(), k))
res = {}
if srod:
    srod=np.concatenate(srod); cls=np.concatenate(cls); dist_h=np.concatenate(dist_h); dist_e=np.concatenate(dist_e)
    for metric, dist, xlabel in (('height', dist_h, 'height above origin / floor (um)'),
                                 ('euclid', dist_e, 'straight-line distance from origin / floor reference (um)')):
        fig,ax = plt.subplots(1,2,figsize=(11,4.4),sharey=True)
        for k,(name,ax_) in enumerate((('originating',ax[0]),('established',ax[1]))):
            m = cls==k
            if m.sum() < 100: print(f'{metric} {name}: too few voxels ({m.sum()})'); continue
            bins = np.arange(0, dist[m].max()+10, 10); rows=[]
            for lo in bins[:-1]:
                mm = m & (dist>=lo) & (dist<lo+10) & (dist>0)
                if mm.sum() < 50: continue
                sv = srod[mm]; p5,p95 = np.percentile(sv,[5,95]); rows.append((dist[mm].mean(), sv.mean(), p5, p95, np.sqrt((sv**2).mean()), mm.sum()))
            rows = np.array(rows); res[f'{metric}_{name}'] = rows.tolist()
            Dfit = (rows[:,0]*rows[:,4]**2).sum()/(rows[:,0]**2).sum()*1e6   # RMS^2 = D d through the origin
            print(f'{metric:6s} {name:11s}: {m.sum():8d} voxels, D_fit = {Dfit:.3g} deg^2/m' + (f' = {Dfit/a.D:.2f} x input' if a.D else '') + '; '
                  f'band/mean ' + ' '.join(f'{(r[3]-r[2])/r[1]:.2f}' for r in rows[::max(1,len(rows)//6)]))
            for r in rows: print(f'   d={r[0]:6.1f} um n={int(r[5]):7d} mean {r[1]:5.2f} p5 {r[2]:5.2f} p95 {r[3]:5.2f} rms {r[4]:5.2f}')
            ax_.fill_between(rows[:,0], rows[:,2], rows[:,3], color='#2a78d6', alpha=0.15, lw=0)
            ax_.plot(rows[:,0], rows[:,1], color='#2a78d6', lw=2, label='simulation mean, 5/95%')
            dd = np.linspace(1, rows[:,0].max(), 200)
            if a.D: ax_.plot(dd, 0.921*np.sqrt(a.D*1e-6*dd), color='#0b0b0b', lw=1.2, ls='--', label='random walk, input D')
            lf = os.path.join(a.lamb, 'lamb2023_fig5%s_srod.json' % ('a' if k==0 else 'c'))
            if os.path.exists(lf):
                L = json.load(open(lf)); ax_.plot(L['x'], L['mean'], color='#eb6834', lw=1.6, label='Lamb 2023 mean (height)')
                ax_.plot(L['x'], L['p5'], color='#eb6834', lw=0.9, ls=(0,(3,2))); ax_.plot(L['x'], L['p95'], color='#eb6834', lw=0.9, ls=(0,(3,2)))
            ax_.set_title(f'{name} ({metric})', loc='left'); ax_.set_xlabel(xlabel); ax_.grid(axis='y', color='#e6e5e1')
        ax[0].set_ylabel('SROD (deg)'); ax[0].legend(frameon=False); fig.tight_layout(); fig.savefig(f'{a.out}/srod_{metric}.png', dpi=150)

# ---- IPF cross-sections (BD = z) and pole figures
rgb = generate_ipf_colors(q.reshape(-1,4), 'fcc', 'z').reshape(nx,ny,nz,3); rgb[~solid] = 0.25
fig,ax = plt.subplots(1,3,figsize=(15,5))
ax[0].imshow(np.transpose(rgb[:,ny//2,:,:],(1,0,2)), origin='lower', extent=[0,nx*dxu,0,nz*dxu]); ax[0].set_title('xz at y = mid (IPF-z)')
ax[1].imshow(np.transpose(rgb[nx//2,:,:,:],(1,0,2)), origin='lower', extent=[0,ny*dxu,0,nz*dxu]); ax[1].set_title('yz at x = mid')
ztop = int(np.max(np.where(sf>0.5)[0])) if (sf>0.5).any() else nz-1
ax[2].imshow(np.transpose(rgb[:,:,ztop-5,:],(1,0,2)), origin='lower', extent=[0,nx*dxu,0,ny*dxu]); ax[2].set_title(f'xy at z = {(ztop-5)*dxu:.0f} um')
for a_ in ax: a_.set_xlabel('um')
fig.tight_layout(); fig.savefig(f'{a.out}/ipf_sections.png', dpi=140)
sub = q[sol_above]; rng = np.random.default_rng(0); sub = sub[rng.choice(len(sub), min(len(sub),200000), replace=False)]
try:
    create_pole_density_plots_from_quaternions(sub, 'fcc', output_path=f'{a.out}/pole')
except Exception as e: print('pole figure skipped:', e)
json.dump(dict(time=float(t), n_grains=int(len(ids)), n_filtered=int(keep.sum()), max_height_um=float(height.max()),
               esd_p10_p50_p90=[float(q10),float(q50),float(q90)], extent_xyz_median_um=[float(np.median(np.repeat(ext[big,i], counts[big]))) for i in range(3)], nn_z_median_deg=float(np.median(nn)), solid_fraction_by_z=sf.tolist(),
               frac_001_15deg=float(np.mean(ang<15)), srod=res), open(f'{a.out}/summary.json','w'), indent=1)
print('wrote', a.out)
