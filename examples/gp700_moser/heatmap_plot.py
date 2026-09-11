#!/usr/bin/env python3
"""Two-panel SROD density heatmap (0-120 um above the grain origin) from gp700_analysis.py --heatmap-npz files,
with the Lamb 2026 Fig. 7a mode and the 2024 Fig. 5a mean overlaid.
usage: heatmap_plot.py OUT.png LABEL1 FILE1.npz [LABEL2 FILE2.npz ...]"""
import sys, os, json, numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
out = sys.argv[1]; runs = list(zip(sys.argv[2::2], sys.argv[3::2]))
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'docs')
l26 = json.load(open(os.path.join(D, 'lamb2026_fig7a_srod.json'))); l24 = json.load(open(os.path.join(D, 'lamb2023_fig5a_srod.json')))
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
