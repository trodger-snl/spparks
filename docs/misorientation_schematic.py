#!/usr/bin/env python3
"""Schematic of the misorientation mechanisms in the SPPARKS additive apps
(docs/misorientation_schematic.png): per-grain parent selection, the per-voxel
inheritance walk, the coherent per-grain walk indexed by inheritance path length
with Ornstein-Uhlenbeck persistence, and copy-mode boundary smoothing."""
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, Polygon
rng = np.random.default_rng(3)
INK, MUTED, BLUE, ORANGE, GREEN, GREY = '#1b2230', '#5b6470', '#2a78d6', '#c9552a', '#3a9d5d', '#c9cdd3'
fig, axes = plt.subplots(2, 2, figsize=(15, 12), gridspec_kw=dict(height_ratios=[1, 1.35], width_ratios=[1, 1])); fig.subplots_adjust(hspace=0.22, wspace=0.12, left=0.02, right=0.99, top=0.93, bottom=0.03)

def voxel(ax, x, y, color, txt=None, ec=INK, lw=0.8, alpha=1.0, fs=8):
    ax.add_patch(Rectangle((x, y), 1, 1, facecolor=color, edgecolor=ec, lw=lw, alpha=alpha))
    if txt: ax.text(x+0.5, y+0.5, txt, ha='center', va='center', fontsize=fs, color=INK)

# ---------------------------------------------------------------- (a) per-grain parent selection
ax = axes[0, 0]; ax.set_title('(a) Capture: per-grain parent selection', loc='left', fontsize=11, color=INK)
grainA = {(0,0),(1,0),(2,0),(0,1),(1,1),(0,2)}; grainB = {(3,0),(4,0),(4,1),(3,1)}
for (x,y) in grainA: voxel(ax, x, y, '#bcd3f2')
for (x,y) in grainB: voxel(ax, x, y, '#f4cdb9')
for (x,y) in [(1,2),(2,1),(2,2),(3,2),(4,2),(0,3),(1,3),(2,3),(3,3),(4,3)]:
    voxel(ax, x, y, '#f5f5f2', ec=GREY, alpha=1)
voxel(ax, 2, 1, '#ffffff', txt='i', ec=INK, lw=1.6)
ax.text(0.5, 0.25, 'grain A', fontsize=8, color=BLUE); ax.text(3.5, 0.25, 'grain B', fontsize=8, color=ORANGE)
ax.text(2.5, 3.5, 'liquid / undercooled', ha='center', fontsize=8, color=MUTED)
ax.annotate('', xy=(2.5, 3.1), xytext=(2.5, 4.4), arrowprops=dict(arrowstyle='-|>', color=GREEN, lw=1.5))
ax.text(2.6, 4.45, r'thermal gradient $\nabla T$', fontsize=8, color=GREEN, va='bottom')
ax.text(5.4, 3.9, 'candidates = solid neighbours of i,\ngrouped by grain', fontsize=8.5, color=INK, va='top')
ax.text(5.4, 3.0, r'weight of grain g: $w_g = 1 + c_2\cos(c_3\,\theta_g)$', fontsize=8.5, color=INK, va='top')
ax.text(5.4, 2.55, r'$\theta_g$: angle between the nearest $\langle100\rangle$ of the', fontsize=8, color=MUTED, va='top')
ax.text(5.4, 2.25, r'mean orientation $\bar q_g$ of grain g and $\nabla T$ (clamped $\geq 0$)', fontsize=8, color=MUTED, va='top')
ax.text(5.4, 1.75, '1. draw the grain with probability $\\propto w_g$', fontsize=8.5, color=INK, va='top')
ax.text(5.4, 1.35, '2. draw the parent voxel uniformly among that\n    grain\'s candidates', fontsize=8.5, color=INK, va='top')
ax.text(5.4, 0.6, 'i inherits the parent\'s orientation and\npath length, then steps (b) and (c) apply', fontsize=8.5, color=INK, va='top')
ax.annotate('', xy=(2.15, 1.5), xytext=(1.5, 1.5), arrowprops=dict(arrowstyle='-|>', color=BLUE, lw=1.4))
ax.annotate('', xy=(2.85, 1.5), xytext=(3.5, 1.5), arrowprops=dict(arrowstyle='-|>', color=ORANGE, lw=1.4))
ax.set_xlim(-0.3, 10.2); ax.set_ylim(-0.3, 5.0); ax.set_aspect('equal'); ax.axis('off')

# ---------------------------------------------------------------- (b) per-voxel inheritance walk
ax = axes[0, 1]; ax.set_title('(b) Per-voxel walk: independent rotation per hop (misorientation_diffusion $D_v$)', loc='left', fontsize=11, color=INK)
# lineage tree over 5 levels
pos = {}; layers = [[0], [0,1], [0,1,2], [0,1,2,3], [0,1,2,3,4]]
for L, nodes in enumerate(layers):
    for k, n in enumerate(nodes):
        x = 1.5 + (k - (len(nodes)-1)/2)*1.1; y = 0.4 + L*0.95; pos[(L,k)] = (x, y)
        ax.add_patch(Circle((x, y), 0.22, facecolor='#bcd3f2' if L else BLUE, edgecolor=INK, lw=0.8))
parents = {(1,0):(0,0),(1,1):(0,0),(2,0):(1,0),(2,1):(1,0),(2,2):(1,1),(3,0):(2,0),(3,1):(2,1),(3,2):(2,1),(3,3):(2,2),(4,0):(3,0),(4,1):(3,1),(4,2):(3,2),(4,3):(3,3),(4,4):(3,3)}
for c, p in parents.items():
    (x1,y1),(x0,y0) = pos[c], pos[p]; ax.annotate('', xy=(x1,y1-0.22), xytext=(x0,y0+0.22), arrowprops=dict(arrowstyle='-|>', color=MUTED, lw=0.9))
ax.text(1.5, -0.2, 'grain origin', ha='center', fontsize=8, color=INK)
ax.text(4.3, 4.3, r'each capture: $q_i = R(\theta,\hat n)\,q_{parent}$', fontsize=9, color=INK)
ax.text(4.3, 3.9, r'$\theta = \sqrt{D_v\cdot \mathrm{hop}}$, $\hat n$ uniformly random', fontsize=9, color=INK)
ax.text(4.3, 3.4, 'increments are independent along and\nbetween lineages: cousins diverge', fontsize=8.5, color=MUTED, va='top')
ax.text(4.3, 2.55, r'$\mathrm{RMS}^2 = D_v\cdot$ path length  (Maxwell width)', fontsize=9, color=INK)
ax.text(4.3, 2.1, 'supplies intragranular spread,\nno net drift of the grain mean', fontsize=8.5, color=MUTED, va='top')
# small IPF cloud
tri = Polygon([(6.0,0.2),(7.6,0.2),(7.35,1.55)], closed=True, fill=False, edgecolor=INK, lw=0.8); ax.add_patch(tri)
pts = np.array([6.65,0.75]) + rng.normal(0, 0.18, (60,2)); pts = pts[(pts[:,1]>0.25)&(pts[:,1]<1.3)]
ax.scatter(pts[:,0], pts[:,1], s=6, color=BLUE, alpha=0.6); ax.add_patch(Circle((6.65,0.75), 0.05, color=INK))
ax.text(6.8, -0.15, 'IPF: cloud about a fixed mean', ha='center', fontsize=8, color=MUTED)
ax.set_xlim(-1.2, 8.6); ax.set_ylim(-0.5, 4.8); ax.set_aspect('equal'); ax.axis('off')

# ---------------------------------------------------------------- (c) coherent grain walk by path length
ax = axes[1, 0]; ax.set_title('(c) Coherent grain walk by inheritance path length (misorientation_grain_walk $D_g$ $\\rho$ path)', loc='left', fontsize=11, color=INK)
# a grain column with path levels; two lineages reaching the same level by different routes
for y in range(6):
    for x in range(3):
        voxel(ax, x, y, '#bcd3f2', ec='#8fb0dc', lw=0.6)
for y in range(6): ax.text(-0.35, y+0.5, f'k={y}', ha='right', va='center', fontsize=8, color=MUTED)
ax.text(1.5, -0.55, 'levels k = floor(path length s),\ns carried per voxel (d10)', ha='center', fontsize=8, color=MUTED, va='top')
# lineage 1 straight up, lineage 2 with a lateral step
for y in range(5): ax.annotate('', xy=(0.5, y+1.35), xytext=(0.5, y+0.65), arrowprops=dict(arrowstyle='-|>', color=INK, lw=1.0))
path2 = [(2.5,0.5),(2.5,1.5),(1.5,1.5),(1.5,2.5),(2.5,2.5),(2.5,3.5),(2.5,4.5),(2.5,5.5)]
for (x0,y0),(x1,y1) in zip(path2[:-1], path2[1:]): ax.annotate('', xy=(x1,y1), xytext=(x0,y0), arrowprops=dict(arrowstyle='-|>', color=ORANGE, lw=1.0, shrinkA=6, shrinkB=6))
ax.text(0.5, 6.2, 'lineage 1', ha='center', fontsize=8, color=INK); ax.text(2.5, 6.2, 'lineage 2', ha='center', fontsize=8, color=ORANGE)
# right side: the shared increment sequence
ax.text(4.0, 5.7, 'crossing level k → k+1, every voxel of grain g\nreceives the same rotation vector $v_k(g)$', fontsize=9, color=INK, va='top')
ax.text(4.0, 4.85, r'$v_k = \rho\, v_{k-1} + \sqrt{1-\rho^2}\; v_k^{\prime}$,  $v_k^{\prime} \sim \mathcal{N}(0, D_g\Delta x/3)$ per component', fontsize=8.5, color=INK, va='top')
ax.text(4.0, 4.4, r'$v_k^{\prime}$ hashed from (g, k): a pure function of grain and level,' + '\n' + 'no stored state', fontsize=8.5, color=MUTED, va='top')
ax.text(4.0, 3.75, r'correlation length $\approx \Delta x/(1-\rho)$' + '\n' + r'long-range accumulation $D_g\,(1+\rho)/(1-\rho)$ per unit path', fontsize=8.5, color=MUTED, va='top')
# OU arrows sequence
x0 = 4.3; y0 = 2.55; ang = 0.0; angs = []
for k in range(9):
    ang = 0.8*ang + 0.6*rng.normal(); angs.append(ang)
for k, a in enumerate(angs):
    ax.annotate('', xy=(x0+k*0.55+0.45*np.cos(a*0.6), y0+0.45*np.sin(a*0.6)), xytext=(x0+k*0.55, y0), arrowprops=dict(arrowstyle='-|>', color=ORANGE, lw=1.3))
    ax.text(x0+k*0.55, y0-0.35, f'$v_{k}$', ha='center', fontsize=7.5, color=MUTED)
ax.text(4.0, 1.95, 'the direction persists over ~1/(1−ρ) levels:\nthe grain bends coherently', fontsize=8.5, color=MUTED, va='top')
ax.text(4.0, 1.25, 'lineages 1 and 2 reach k = 5 through the same\nincrements: the coherent part is path independent;\nthe per-voxel part (b) rides on top', fontsize=8.5, color=INK, va='top')
tri = Polygon([(7.9,-0.6),(10.1,-0.6),(9.75,1.25)], closed=True, fill=False, edgecolor=INK, lw=0.8); ax.add_patch(tri)
pth = np.cumsum(rng.normal(0, 0.11, (18,2))*np.array([1.2,0.8]), axis=0) + np.array([8.7,0.1]); pth[:,0]=np.clip(pth[:,0],8.0,9.9); pth[:,1]=np.clip(pth[:,1],-0.5,1.0)
ax.plot(pth[:,0], pth[:,1], color=INK, lw=1); ax.scatter(pth[:,0], pth[:,1], c=np.arange(18), cmap='viridis', s=10, zorder=3)
ax.annotate('', xy=pth[-1], xytext=pth[0], arrowprops=dict(arrowstyle='-|>', color=ORANGE, lw=1.5))
ax.text(9.0, -0.95, 'IPF: directed path of the grain mean', ha='center', fontsize=8, color=MUTED)
ax.set_xlim(-0.9, 10.4); ax.set_ylim(-1.3, 6.6); ax.set_aspect('equal'); ax.axis('off')

# ---------------------------------------------------------------- (d) boundary smoothing: copy vs average
ax = axes[1, 1]; ax.set_title('(d) Boundary smoothing the step after capture (smooth_orientation copy | average)', loc='left', fontsize=11, color=INK)
for x in range(3):
    for y in range(3):
        voxel(ax, x, y, '#bcd3f2' if x < 2 or y == 2 else '#f4cdb9', ec=INK)
voxel(ax, 1, 1, '#f4cdb9', txt='i', lw=1.6)
ax.text(1.5, -0.3, 'i captured into grain B but surrounded by A:\nthe greedy pass flips i to the\nbond-energy minimiser (A)', ha='center', fontsize=8.5, color=INK, va='top')
ax.text(4.0, 2.75, 'copy (used):', fontsize=9.5, color=INK, weight='bold', va='top')
ax.text(4.0, 2.35, 'i takes ONE random A neighbour\'s orientation and path length', fontsize=9, color=INK, va='top')
ax.text(4.0, 1.95, 'the walks\' statistics survive the flip', fontsize=8.5, color=MUTED, va='top')
ax.text(4.0, 1.35, 'average (default, not used):', fontsize=9.5, color=INK, weight='bold', va='top')
ax.text(4.0, 0.95, 'i takes the sign-aligned mean of all A neighbours', fontsize=9, color=INK, va='top')
ax.text(4.0, 0.55, 'cancels the fresh rotations on a voxel-scale mixed front:\nretains 0.3–0.45 of D, widens the population band to 1.8', fontsize=8.5, color=MUTED, va='top')
for (x,y) in [(0.5,1.5),(0.5,2.5),(1.5,2.5),(2.5,2.5),(0.5,0.5)]:
    ax.annotate('', xy=(1.5+0.3*np.sign(x-1.5) if x!=1.5 else 1.5, 1.5+0.3*np.sign(y-1.5) if y!=1.5 else 1.5), xytext=(x, y), arrowprops=dict(arrowstyle='-|>', color=BLUE, lw=0.8, alpha=0.6))
ax.set_xlim(-0.3, 10.2); ax.set_ylim(-1.5, 3.4); ax.set_aspect('equal'); ax.axis('off')

fig.suptitle('Misorientation formation in the SPPARKS additive/texture apps:\n(a) who is inherited from, (b) per-lineage noise, (c) coherent grain bending, (d) what a boundary flip does to the orientation', fontsize=11, color=INK, x=0.02, ha='left')
fig.savefig('/home/tron/spparks-kokkos/docs/misorientation_schematic.png', dpi=150, bbox_inches='tight'); print('saved')
