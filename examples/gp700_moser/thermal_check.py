#!/usr/bin/env python3
"""Plan 2 5.1 thermal gate for in.gp700_thermal: melt-pool footprint/aspect,
thermal gradient at solidification (d8 = G), and front velocity (d9 = V),
from the text dumps (id i2 d6 d8 d9 x y z). Compare with Met Trans 2026
Fig. 3: G ~3e7 K/m at the fusion boundary, V low over most of the pool and
> 1 m/s only at the tail end.  usage: thermal_check.py 'dump.thermal.*' --dx 3e-6
"""
import argparse, glob
import numpy as np
ap = argparse.ArgumentParser(); ap.add_argument('pattern'); ap.add_argument('--dx', type=float, required=True)
ap.add_argument('--poly', type=float, nargs=4, default=[-3.6057e-7, 3.68655e-4, 7.7686e-3, 0.0], help='solid_front_vel coefficients, high to low')
a = ap.parse_args()
files = sorted(glob.glob(a.pattern), key=lambda f: int(f.rsplit('.',1)[1]))
def read(fn):
    L = open(fn).readlines(); i = next(k for k,l in enumerate(L) if l.startswith('ITEM: ATOMS'))
    t = float(L[1].split()[-1])   # 'ITEM: TIMESTEP' line: '<step> <time>'
    cols = L[i].split()[2:]; d = np.loadtxt(L[i+1:]); return t, {c: d[:,k] for k,c in enumerate(cols)}
dx = a.dx
print('frame  t(ms)   n_liq  pool L(um) W(um) D(um)  W/D   Tmax(K)')
best = None
for fn in files:
    t, d = read(fn)
    liq = d['i2'] == 2
    if liq.sum() == 0: print(f'{fn:22s} {t*1e3:6.3f} {0:6d}'); continue
    x,y,z = d['x'][liq]*dx, d['y'][liq]*dx, d['z'][liq]*dx
    L = np.ptp(x)+dx; W = np.ptp(y)+dx; D = np.ptp(z)+dx
    print(f'{fn:22s} {t*1e3:6.3f} {liq.sum():6d}  {L*1e6:6.0f} {W*1e6:5.0f} {D*1e6:5.0f}  {W/D:4.2f}  {d["d6"].max():8.0f}')
    if best is None or liq.sum() > best[0]: best = (liq.sum(), fn)
# final frame: G and V statistics over solidified sites
t, d = read(files[-1]); sol = d['i2'] == 3
G = d['d8'][sol]; V = d['d9'][sol]; z = d['z'][sol]*dx; y = d['y'][sol]*dx
print(f'\nfinal frame t={t*1e3:.3f} ms: {sol.sum()} solidified sites')
print('G (K/m):  p5 %.2e  median %.2e  p95 %.2e  max %.2e' % (*np.percentile(G,[5,50,95]), G.max()))
print('V (m/s):  p5 %.3f  median %.3f  p95 %.3f  max %.3f' % (*np.percentile(V,[5,50,95]), V.max()))
# G and V vs depth below the surface (fusion boundary = deepest solidified)
ztop = d['z'].max()*dx
for lo,hi in ((0,10e-6),(10e-6,25e-6),(25e-6,45e-6),(45e-6,80e-6),(80e-6,200e-6)):
    m = (ztop - z >= lo) & (ztop - z < hi)
    if m.sum(): print(f'  depth {lo*1e6:3.0f}-{hi*1e6:3.0f} um: n={m.sum():6d}  G med {np.median(G[m]):.2e}  V med {np.median(V[m]):.3f}  V p95 {np.percentile(V[m],95):.3f}')
# undercooling at solidification from V = p(dT) (monotone on [0, ~700 K])
p = np.array(a.poly); dT = np.linspace(0, 600, 6001); Vt = np.polyval(p, dT)
dTs = np.interp(V, Vt, dT)
print('undercooling at solidification (K): p5 %.1f median %.1f p95 %.1f max %.1f ; fraction >= Tl-Ts (54 K): %.2f' % (*np.percentile(dTs,[5,50,95]), dTs.max(), np.mean(dTs>=54)))
