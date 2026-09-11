#!/usr/bin/env python3
"""Two-panel SROD density heatmap (0-120 um above the grain origin) from gp700_analysis.py --heatmap-npz files,
with the Lamb 2026 Fig. 7a mode and the 2024 Fig. 5a mean overlaid.
usage: heatmap_plot.py OUT.png LABEL1 FILE1.npz [LABEL2 FILE2.npz ...]
       heatmap_plot.py --lines OUT.png LABEL1 FILE1.npz ...   (two line panels: mean vs Lamb 2024, mode vs Lamb 2026)"""
import sys, os, json, numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
lines = sys.argv[1] == '--lines'; args = sys.argv[2:] if lines else sys.argv[1:]
out = args[0]; runs = list(zip(args[1::2], args[2::2]))
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'docs')
l26 = json.load(open(os.path.join(D, 'lamb2026_fig7a_srod.json'))); l24 = json.load(open(os.path.join(D, 'lamb2023_fig5a_srod.json')))
if lines:
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.4, 4.2), sharey=True)
    cols = ['#2a78d6', '#7d8ea0', '#287968', '#8a2b78']
    for (label, f), c in zip(runs, cols):
        d = np.load(f); H = d['H']; yc = 0.5*(d['ye'][1:] + d['ye'][:-1]); dy = yc[1] - yc[0]
        i = np.clip(H.argmax(1), 1, H.shape[1]-2); rows = np.arange(len(i))
        hm, h0, hp = H[rows, i-1], H[rows, i], H[rows, i+1]
        den = hm - 2*h0 + hp; mode = yc[i] + np.where(den < 0, 0.5*dy*(hm - hp)/np.where(den == 0, 1, den), 0.0)   # parabolic peak between bins
        a1.plot(d['xc'], d['mean'], color=c, lw=2, label=f'{label}: mean'); a2.plot(d['xc'], mode, color=c, lw=2, label=f'{label}: mode')
    x24 = np.array(l24['x']); m = x24 <= 120; a1.plot(x24[m], np.array(l24['mean'])[m], color='#e8632b', lw=2.2, ls='--', label='Lamb 2024 Fig. 5a: mean')
    a2.plot(np.array(l26['x']), l26['mode'], color='#e8632b', lw=2.2, ls='--', label='Lamb 2026 Fig. 7a: mode (digitized)')
    for ax, t in ((a1, 'mean SROD (originating grains) against Lamb 2024'), (a2, 'modal SROD against Lamb 2026')):
        ax.set_xlim(0, 120); ax.set_ylim(0, 8); ax.set_xlabel('height above grain origin (um)'); ax.set_title(t, loc='left', fontsize=10)
        ax.grid(alpha=0.25); ax.legend(loc='upper left', fontsize=9, frameon=False)
    a1.set_ylabel('SROD (deg)'); fig.tight_layout(); fig.savefig(out, dpi=170); print('saved', out); sys.exit(0)
fig, axes = plt.subplots(1, len(runs), figsize=(5.2*len(runs), 4.2), sharey=True, squeeze=False)
for ax, (label, f) in zip(axes[0], runs):
    d = np.load(f); H = d['H'] / np.maximum(d['H'].sum(1, keepdims=True), 1)
    ax.pcolormesh(d['xe'], d['ye'], H.T, cmap='turbo', vmin=0, vmax=np.percentile(H, 99.5))
    ax.plot(d['xc'], d['mean'], color='w', lw=2, label='simulation mean')
    ax.plot(d['xc'], d['mode'], color='w', lw=1.2, ls=':', label='simulation mode')
    x26 = np.array(l26['x']); ax.plot(x26, l26['mode'], color='#ff9a3c', lw=2, ls='--', label='Lamb 2026 Fig. 7a mode (digitized)')
    x24 = np.array(l24['x']); m = x24 <= 120; ax.plot(x24[m], np.array(l24['mean'])[m], color='#dddddd', lw=1.6, ls='-.', label='Lamb 2024 Fig. 5a mean')
    ax.set_xlim(0, 120); ax.set_ylim(0, 25); ax.set_xlabel('height above grain origin (um)'); ax.set_title(label, loc='left', fontsize=10)
    ax.legend(loc='upper left', fontsize=8, facecolor='white', framealpha=0.9, edgecolor='none')
axes[0][0].set_ylabel('SROD (deg)')
fig.tight_layout(); fig.savefig(out, dpi=170); print('saved', out)
