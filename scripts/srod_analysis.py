#!/usr/bin/env python3
"""Origin-referenced misorientation (SROD) vs growth distance from a SPPARKS
text dump (columns id i1 i2 d1 d2 d3 d4 x y z), per Lamb et al. 2023:
grain = 26-connected same-spin solid region, origin = its lowest-z voxel,
SROD = cubic disorientation to the origin, distance = Euclidean from it.

A random-axis rotation walk gives RMS(SROD)^2 = D * distance with a Maxwell
spread, (p95 - p5)/mean = 1.38. Fits D and reports both.

usage: srod_analysis.py DUMP [DUMP ...] --dx 5e-6 [--D 4.2e5] [--min-voxels 100] [--bin 1]
Each DUMP is a file or glob; the last frame of each is analysed and the grains are pooled.
"""
import argparse, glob, sys
import numpy as np
from scipy import ndimage

def cubic_sym():
    s, h = 1/np.sqrt(2), 0.5
    return np.array([(1,0,0,0),(s,s,0,0),(0,1,0,0),(s,-s,0,0),(s,0,s,0),(0,0,1,0),(s,0,-s,0),
        (s,0,0,s),(0,0,0,1),(s,0,0,-s),(0,s,s,0),(0,s,-s,0),(0,s,0,s),(0,s,0,-s),(0,0,s,s),(0,0,s,-s),
        (h,h,h,h),(h,-h,-h,-h),(h,h,-h,h),(h,-h,h,-h),(h,h,h,-h),(h,-h,-h,h),(h,-h,h,h),(h,h,-h,-h)])
SYM = cubic_sym()

def qmul(a, b):
    aw,ax,ay,az = a[...,0],a[...,1],a[...,2],a[...,3]; bw,bx,by,bz = b[...,0],b[...,1],b[...,2],b[...,3]
    return np.stack([aw*bw-ax*bx-ay*by-az*bz, aw*bx+ax*bw+ay*bz-az*by,
                     aw*by-ax*bz+ay*bw+az*bx, aw*bz+ax*by-ay*bx+az*bw], axis=-1)

def disorientation_deg(qa, qb):
    m = qmul(qa * np.array([1,-1,-1,-1.0]), qb)
    best = np.zeros(len(m))
    for s in SYM: np.maximum(best, np.abs(qmul(m, s[None,:])[:,0]), out=best)
    return np.degrees(2*np.arccos(np.clip(best, 0, 1)))

def read_dump(fn):
    with open(fn) as f: lines = f.readlines()
    i = next(k for k,l in enumerate(lines) if l.startswith('ITEM: ATOMS'))
    cols = lines[i].split()[2:]
    data = np.loadtxt(lines[i+1:])
    return fn, {c: data[:,k] for k,c in enumerate(cols)}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('dump', nargs='+'); ap.add_argument('--dx', type=float, required=True)
    ap.add_argument('--D', type=float, default=None, help='expected D, deg^2/m')
    ap.add_argument('--min-voxels', type=int, default=100)
    ap.add_argument('--bin', type=float, default=1.0, help='distance bin, lattice units')
    a = ap.parse_args()
    dist=[]; srod=[]; ngr=0; nsolid=0; ntot=0; files=[]
    for pat in a.dump:
        files += [sorted(glob.glob(pat))[-1]] if glob.glob(pat) else []
    for fn in files:
        fn, d = read_dump(fn)
        x,y,z = (np.rint(d[c]).astype(int) for c in 'xyz')
        x-=x.min(); y-=y.min(); z-=z.min()
        solid = d['i2'] == 3; nsolid += solid.sum(); ntot += len(solid)
        spin = np.where(solid, d['i1'].astype(int), 0)
        grid = np.zeros((x.max()+1, y.max()+1, z.max()+1), int); grid[x,y,z] = spin
        q = np.zeros(grid.shape + (4,)); q[x,y,z] = np.stack([d['d1'],d['d2'],d['d3'],d['d4']],-1)
        # 26-connected same-spin regions
        lab = np.zeros_like(grid); nlab = 0
        for sp in np.unique(grid[grid>0]):
            l, n = ndimage.label(grid == sp, structure=np.ones((3,3,3)))
            lab[l>0] = l[l>0] + nlab; nlab += n
        for g in range(1, nlab+1):
            idx = np.argwhere(lab == g)
            if len(idx) < a.min_voxels: continue
            o = idx[np.argmin(idx[:,2])]                       # lowest-z voxel = origin
            qo = np.broadcast_to(q[tuple(o)], (len(idx),4))
            srod.append(disorientation_deg(qo, q[idx[:,0],idx[:,1],idx[:,2]]))
            dist.append(np.linalg.norm(idx - o, axis=1)); ngr += 1
    if ngr == 0: sys.exit('no grains above --min-voxels')
    dist = np.concatenate(dist); srod = np.concatenate(srod)
    print(f'{len(files)} dump(s): {ngr} grains, {len(dist)} voxels, solid fraction {nsolid/ntot:.3f}')
    bins = np.arange(0, dist.max()+a.bin, a.bin)
    print(' dist(um)  n     mean   p5    p95   rms   (p95-p5)/mean')
    rows=[]
    for lo in bins[:-1]:
        m = (dist>=lo)&(dist<lo+a.bin)&(dist>0)
        if m.sum() < 30: continue
        s = srod[m]; p5,p95 = np.percentile(s,[5,95]); mean=s.mean(); rms=np.sqrt((s**2).mean())
        dm = dist[m].mean()*a.dx
        rows.append((dm, rms)); print(f'{dm*1e6:8.1f} {m.sum():6d} {mean:6.2f} {p5:5.2f} {p95:6.2f} {rms:6.2f}  {(p95-p5)/mean:5.2f}')
    dm = np.array([r[0] for r in rows]); rms = np.array([r[1] for r in rows])
    Dfit = (dm*rms**2).sum()/(dm*dm).sum()                  # RMS^2 = D d, least squares through origin
    print(f'fitted D = {Dfit:.4g} deg^2/m = {Dfit*1e-6:.3f} deg^2/um' + (f'  (expected {a.D:.4g} deg^2/m, ratio {Dfit/a.D:.2f})' if a.D else ''))
if __name__ == '__main__': main()
