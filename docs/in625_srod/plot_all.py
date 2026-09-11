import json, glob, os, sys, numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
S = sys.argv[1]; files = sorted(glob.glob(os.path.join(S, 'out', 'srod_*.json')))
lamb = json.load(open('/home/tron/spparks-kokkos/docs/lamb2023_fig5c_srod.json'))
cond = {'P04': '50 % lower power', 'P13': '30 % lower power', 'P15': 'nominal', 'P18': 'defocus at min.', 'P20': '30 % slower speed'}
cols = {'P04': '#c41d24', 'P13': '#ec8a00', 'P15': '#0075a9', 'P18': '#287968', 'P20': '#8a2b78'}
fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
rows = []
for f in files:
    d = json.load(open(f)); lab = d['label']; sid = lab[:3]; c = d['curve']; h = np.array(c['h'])
    ls = '-' if lab.endswith('XZ') else '--'
    axes[0].plot(h, c['mean'], color=cols[sid], ls=ls, lw=1.8, label=f'{sid} {cond[sid]} ({lab[-2:]}, {d["grains"]} grains)')
    axes[1].plot(h, np.array(c['rms'])**2, color=cols[sid], ls=ls, lw=1.8)
    # n-weighted fit of RMS^2 = rms0^2 + D h over 5-150 um (where every section still has > 10^4 pixels per bin)
    rms = np.array(c['rms']); n = np.array(c['n']); m = (h >= 5) & (h <= 150)
    A = np.column_stack([np.ones(m.sum()), h[m]]); w = np.sqrt(n[m]); coef = np.linalg.lstsq(A*w[:,None], (rms[m]**2)*w, rcond=None)[0]
    Dw, rms0w = coef[1], np.sqrt(max(coef[0], 0)); mean = np.array(c['mean'])
    at = [float(np.interp(x, h, mean)) for x in (50, 100, 150)]
    d['D_deg2_per_um'], d['rms0_deg'] = Dw, rms0w; d['at'] = at
    rows.append((lab, cond[sid], d['grains'], d['D_deg2_per_um'], d['rms0_deg'], d['band_over_mean_median'],
                 np.median([g['max_srod'] for g in d['per_grain']]), np.mean([g['max_srod'] > 25 for g in d['per_grain']]), np.median([g['height_um'] for g in d['per_grain']]), *at))
lx = np.array(lamb['x']); m = lx <= 300
axes[0].plot(lx[m], np.array(lamb['mean'])[m], color='k', lw=2.2, ls=':', label='GP700, Lamb 2024 Fig. 5c (established grains, 3-D)')
axes[0].set_xlabel('height above the lowest point in the section (um)'); axes[0].set_ylabel('mean SROD (deg)'); axes[0].set_xlim(0, 300); axes[0].grid(alpha=0.3)
axes[0].legend(fontsize=7.5, loc='upper left'); axes[0].set_title('IN625 longitudinal sections: mean SROD', loc='left', fontsize=10)
axes[1].set_xlabel('height (um)'); axes[1].set_ylabel('RMS SROD$^2$ (deg$^2$)'); axes[1].set_xlim(0, 300); axes[1].grid(alpha=0.3)
axes[1].set_title('RMS$^2$ against height (slope = D)', loc='left', fontsize=10)
fig.tight_layout(); fig.savefig(os.path.join(S, 'srod_in625_sections.png'), dpi=160)
print(f'{"section":8s} {"condition":18s} {"grains":>6s} {"D(5-150)":>9s} {"rms0":>5s} {"band/mean":>9s} {"maxSROD med":>11s} {">25deg":>6s} {"h med":>5s} {"mean@50":>7s} {"@100":>5s} {"@150":>5s}')
for r in rows: print(f'{r[0]:8s} {r[1]:18s} {r[2]:6d} {r[3]:9.3f} {r[4]:5.2f} {r[5]:9.2f} {r[6]:11.1f} {r[7]:6.2f} {r[8]:5.0f} {r[9]:7.2f} {r[10]:5.2f} {r[11]:5.2f}')
axes[0].set_xlim(0, 200); axes[1].set_xlim(0, 200); axes[0].set_ylim(0, 12); axes[1].set_ylim(0, 150); fig.savefig(os.path.join(S, 'srod_in625_sections.png'), dpi=160)
import csv
with open(os.path.join(S, 'srod_in625_summary.csv'), 'w', newline='') as fh:
    wr = csv.writer(fh); wr.writerow(['section','condition','grains','D_5_150_deg2_per_um','rms0_deg','band_over_mean','max_srod_median_deg','frac_grains_gt_25deg','grain_height_median_um','mean_srod_50um','mean_srod_100um','mean_srod_150um'])
    for r in rows: wr.writerow([r[0], r[1], r[2]] + [round(float(x), 3) for x in r[3:]])
