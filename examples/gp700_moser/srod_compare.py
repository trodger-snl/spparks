#!/usr/bin/env python3
"""Side-by-side SROD: simulation (srod3d_<SAMPLE>_<run>.json from srod_from_stitch.py) against the IN625
EBSD sections (srod_<SAMPLE>_<XZ|YZ>.json from srod_from_ebsd.py), per process condition.
usage: srod_compare.py SIM_DIR EBSD_DIR [-o OUT_DIR]"""
import sys, os, glob, json, argparse, numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
ap = argparse.ArgumentParser(); ap.add_argument('sim'); ap.add_argument('ebsd'); ap.add_argument('-o', default=None); a = ap.parse_args()
out = a.o or a.sim; os.makedirs(out, exist_ok=True)
cond = {'P04': '50 % lower power', 'P13': '30 % lower power', 'P15': 'nominal', 'P18': 'defocus at min.', 'P20': '30 % slower speed'}
sims = {}; 
for f in sorted(glob.glob(os.path.join(a.sim, 'srod3d_*.json'))):
    d = json.load(open(f)); sims.setdefault(d['label'][:3], []).append(d)
ebsd = {}
for f in sorted(glob.glob(os.path.join(a.ebsd, 'srod_*.json'))):
    d = json.load(open(f)); ebsd.setdefault(d['label'][:3], []).append(d)
samples = [s for s in cond if s in sims or s in ebsd]
at = lambda c, x: float(np.interp(x, c['h'], c['mean'])) if c and x <= max(c['h']) else float('nan')
print(f'{"sample":6s} {"condition":18s} | {"EBSD sections: D":>16s} {"mean@100":>8s} | {"sim sections: D":>15s} {"mean@100":>8s} | {"sim 3-D orig: D":>15s} {"mean@100":>8s} {"@200":>5s} {"grains":>6s} {"orig:estab":>10s}')
fig, axes = plt.subplots(1, max(1, len(samples)), figsize=(4.2*max(1, len(samples)), 4.0), sharey=True, squeeze=False); axes = axes[0]
for ax, s in zip(axes, samples):
    eD, e100, sD, s100, oD, o100, o200, ng, ratio = [], [], [], [], [], [], [], 0, ''
    for d in ebsd.get(s, []):
        c = d['curve']; eD.append(d['D_deg2_per_um']); e100.append(at(c, 100))
        ax.plot(c['h'], c['mean'], color='k', lw=1.6, ls='-' if d['label'].endswith('XZ') else '--', label=f'EBSD {d["label"][-2:]}')
    for d in sims.get(s, []):
        for k, v in d.get('sections', {}).items():
            sD.append(v['D_deg2_per_um']); s100.append(at(v, 100)); ax.plot(v['h'], v['mean'], color='#0075a9', lw=0.9, alpha=0.6, ls='-' if k.startswith('XZ') else '--')
        if 'originating' in d:
            c = d['originating']; oD.append(c['D_deg2_per_um']); o100.append(at(c, 100)); o200.append(at(c, 200)); ng += c['n_grains']
            ax.plot(c['h'], c['mean'], color='#c9552a', lw=2, label=f'sim 3-D originating ({d["label"][4:]})')
        ratio = f'{d.get("originating", {}).get("n_grains", 0)}:{d.get("established", {}).get("n_grains", 0)}'
    if sims.get(s): ax.plot([], [], color='#0075a9', lw=0.9, label='sim sections (EBSD method)')
    fmt = lambda v: f'{np.nanmean(v):.2f}' if len(v) else '  -  '; rng = lambda v: f'{min(v):.2f}-{max(v):.2f}' if len(v) > 1 else fmt(v)
    print(f'{s:6s} {cond[s]:18s} | {rng(eD):>16s} {fmt(e100):>8s} | {rng(sD):>15s} {fmt(s100):>8s} | {rng(oD):>15s} {fmt(o100):>8s} {fmt(o200):>5s} {ng:6d} {ratio:>10s}')
    ax.set_title(f'{s}: {cond[s]}', loc='left', fontsize=10); ax.set_xlim(0, 250); ax.set_xlabel('height (um)'); ax.grid(alpha=0.3); ax.legend(fontsize=7)
axes[0].set_ylabel('mean SROD (deg)'); fig.tight_layout(); fig.savefig(os.path.join(out, 'srod_compare.png'), dpi=150); print('wrote', os.path.join(out, 'srod_compare.png'))
