#!/usr/bin/env python3
"""Schematics of the misorientation walks on one lineage tree, all in the same format.
docs/walk_schematic.png       : (b) per-voxel walk  |  (c) coherent grain walk
docs/walk_schematic_mixed.png : (d) combined model, both walks applied at every hop
Per-voxel: every hop gets its own random rotation, lineages diverge, the grain mean stays.
Coherent: every hop crossing path level k gets the SAME rotation v_k, correlated between levels,
so all voxels of a level share one orientation and the grain bends. Combined: the coherent
path plus a cloud around it."""
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon
from matplotlib.path import Path as MPath
rng = np.random.default_rng(11)
INK, MUTED, ORANGE, GREY = '#1b2230', '#5b6470', '#c9552a', '#c9cdd3'
cmap = plt.get_cmap('viridis')
OUT = '/home/tron/spparks-kokkos/docs/'

# ---------- one lineage tree used by every panel: path levels k = 0..5 with 1,2,3,4,5,6 nodes
layers = [1, 2, 3, 4, 5, 6]; NL = len(layers)
pos = {(L, j): (1.0 + (j - (n-1)/2)*0.95, 0.6 + L*1.05) for L, n in enumerate(layers) for j in range(n)}
parents = {(L, j): (L-1, min(j, layers[L-1]-1)) for L in range(1, NL) for j in range(layers[L])}

# ---------- rotation increments (2-D stand-ins for rotation vectors)
inc_v = {c: rng.normal(0, 1, 2) for c in parents}          # per-voxel: independent per edge
inc_g = []; v = rng.normal(0, 1, 2)                          # coherent: one vector per level, OU rho = 0.9
for L in range(NL-1):
    v = 0.9*v + np.sqrt(1-0.9**2)*rng.normal(0, 1, 2); inc_g.append(v.copy())
inc_g = np.array(inc_g)
tot = inc_g.sum(0); target = np.array([0.5, 0.9]); ang = np.arctan2(target[1], target[0]) - np.arctan2(tot[1], tot[0])
Rm = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]]); inc_g = inc_g @ Rm.T * (2.4/np.linalg.norm(tot))
VOX = 0.55   # per-voxel increment size relative to the coherent one in the combined panel

def increment(mode, c):
    if mode == 'voxel': return inc_v[c]
    if mode == 'coherent': return inc_g[c[0]-1]
    return inc_g[c[0]-1] + VOX*inc_v[c]

def orientation(mode, node):
    o = np.zeros(2); c = node
    while c in parents: o = o + increment(mode, c); c = parents[c]
    return o

TITLES = {
 'voxel': "(b) Per-voxel walk  (misorientation_diffusion $D_v$)\nevery hop: its own random rotation",
 'coherent': "(c) Coherent grain walk  (misorientation_grain_walk $D_g$ $\\rho$ path)\nevery hop across level k: the same rotation $v_k$ for the whole grain",
 'mixed': "(d) Combined model  (both commands, the calibrated deck)\nevery hop: the shared level rotation $v_k$ plus its own random rotation"}
TEXT = {
 'voxel': [(6.2, r'$q_{child} = R(\theta_e, \hat n_e)\; q_{parent}$', 10, INK),
           (5.75, r'per edge e: $\theta_e = \sqrt{D_v\,\mathrm{hop}}$, axis $\hat n_e$ random', 9, INK),
           (5.3, 'orange arrows: one independent increment per edge', 8.5, MUTED),
           (4.75, 'two voxels at the same level differ by the sum of\nall increments back to their common ancestor', 8.5, MUTED),
           (3.95, r'population: $\mathrm{RMS}^2 = D_v \times$ path,  Maxwell width', 9, INK),
           (3.55, 'grain mean: does not move;  intragranular spread: grows', 9, INK)],
 'coherent': [(6.2, r'$q_{child} = R(v_k)\; q_{parent}$  for every edge that crosses level k', 10, INK),
           (5.75, r'$v_k = \rho\, v_{k-1} + \sqrt{1-\rho^2}\; v_k^{\prime}$,  $|v_k|_{\mathrm{rms}} = \sqrt{D_g \Delta x}$', 9, INK),
           (5.3, 'coloured arrows: one increment per level, shared by all edges\nof that level; its direction persists over ~1/(1−ρ) levels', 8.5, MUTED),
           (4.55, 'two voxels at the same level have received the same\nincrements whatever route they took: identical orientation', 8.5, MUTED),
           (3.7, r'population: same $\sqrt{h}$ law (coherent part $D_g\frac{1+\rho}{1-\rho}$ per unit path)', 9, INK),
           (3.3, 'grain mean: bends along a path;  intragranular spread: none', 9, INK)],
 'mixed': [(6.2, r'$q_{child} = R(\theta_e, \hat n_e)\, R(v_k)\; q_{parent}$', 10, INK),
           (5.75, r'coherent $v_k$ as in (c), then the per-voxel $\theta_e$ of (b), at every hop', 9, INK),
           (5.3, 'each edge: the coloured level arrow shared by the grain,\nplus its own orange arrow drawn from the tip', 8.5, MUTED),
           (4.55, 'two voxels at the same level share the coherent part and\ndiffer only by their per-voxel parts', 8.5, MUTED),
           (3.7, r'population: $\mathrm{RMS}^2 = \left(D_v + D_g\frac{1+\rho}{1-\rho}\right) \times$ path (long range)', 9, INK),
           (3.3, 'grain mean: bends along a path;  intragranular spread: grows', 9, INK)]}
IPF = {'voxel': 'IPF: voxels scatter into a cloud that widens with level,\ncentred on the origin (black dot)',
       'coherent': 'IPF: all voxels of a level sit at one point;\nthe levels trace a persistent path away from the origin',
       'mixed': 'IPF: a cloud around each level point;\nthe cloud drifts along the coherent path and widens'}

def panel(ax, mode):
    ax.set_title(TITLES[mode], loc='left', fontsize=11, color=INK)
    for L in range(NL):
        y = pos[(L, 0)][1]
        ax.axhspan(y-0.5, y+0.5, xmin=0.02, xmax=0.62, color='#eef2f7' if L % 2 else '#f7f8fa', zorder=0)
        ax.text(-1.75, y, f'k = {L}', va='center', ha='right', fontsize=9, color=MUTED)
    ax.text(-1.75, pos[(0,0)][1]-0.5, 'path level\n(d10)', va='top', ha='right', fontsize=8.5, color=MUTED)
    for c, p in parents.items():
        (x1, y1), (x0, y0) = pos[c], pos[p]
        ax.annotate('', xy=(x1, y1-0.2), xytext=(x0, y0+0.2), arrowprops=dict(arrowstyle='-', color=GREY, lw=1.0))
        mx, my = (x0+x1)/2, (y0+y1)/2
        lvl = cmap((c[0]-0.5)/(NL-1))
        if mode in ('coherent', 'mixed'):
            d = inc_g[c[0]-1]; d = 0.28*d/np.linalg.norm(d)
            ax.annotate('', xy=(mx+d[0], my+d[1]), xytext=(mx, my), arrowprops=dict(arrowstyle='-|>', color=lvl, lw=1.6, mutation_scale=9))
            mx, my = mx+d[0], my+d[1]
        if mode in ('voxel', 'mixed'):
            d = inc_v[c]; d = (0.28 if mode == 'voxel' else 0.18)*d/np.linalg.norm(d)
            ax.annotate('', xy=(mx+d[0], my+d[1]), xytext=(mx, my), arrowprops=dict(arrowstyle='-|>', color=ORANGE, lw=1.6, mutation_scale=9))
    for n, (x, y) in pos.items():
        ax.add_patch(Circle((x, y), 0.17, facecolor=cmap(n[0]/(NL-1)), edgecolor=INK, lw=0.7, zorder=3))
    ax.text(pos[(0,0)][0], pos[(0,0)][1]-0.45, 'grain origin (reference orientation)', ha='center', fontsize=8.5, color=INK)
    for y, t, fs, col in TEXT[mode]: ax.text(3.9, y, t, fontsize=fs, color=col, va='top' if y < 5.5 else 'baseline')
    # IPF sketch
    tri_pts = [(3.9, -0.45), (7.9, -0.45), (7.25, 2.75)]; ax.add_patch(Polygon(tri_pts, closed=True, fill=False, edgecolor=INK, lw=0.9))
    centre = np.array([5.6, 0.55]) if mode == 'voxel' else np.array([5.6, 0.3])
    ori = {n: orientation(mode, n) for n in pos}
    P = np.array([ori[n] for n in pos]); Lv = np.array([n[0] for n in pos])
    inner = MPath([tuple(centre + 0.88*(np.array(t)-centre)) for t in tri_pts])
    inside = lambda sc: all(inner.contains_point(pt) for pt in centre + sc*P)
    scale = 0.5
    while inside(scale*1.05): scale *= 1.05
    while not inside(scale): scale *= 0.95
    xy = centre + scale*P
    if mode != 'voxel':   # coherent path through the level means
        mean = [centre + scale*orientation('coherent', (L, 0)) for L in range(NL)]
        for L in range(NL-1):
            ax.annotate('', xy=mean[L+1], xytext=mean[L], arrowprops=dict(arrowstyle='-|>', color=cmap((L+0.5)/(NL-1)), lw=2.4, mutation_scale=14, shrinkA=4, shrinkB=4), zorder=5)
        if mode == 'coherent': xy = xy + rng.normal(0, 0.02, xy.shape)
    ax.scatter(xy[:,0], xy[:,1], c=Lv, cmap='viridis', s=34, edgecolor=INK, linewidths=0.4, zorder=3, alpha=0.8 if mode == 'mixed' else 1.0)
    ax.add_patch(Circle(tuple(centre), 0.06, color=INK, zorder=4))
    ax.text(5.9, -0.65, IPF[mode], ha='center', fontsize=8.5, color=MUTED, va='top')
    for (x_, y_, s_, va) in ((3.9, -0.5, '001', 'top'), (7.9, -0.5, '101', 'top'), (7.25, 2.8, '111', 'bottom')):
        ax.text(x_, y_, s_, fontsize=7, color=MUTED, ha='center', va=va)
    ax.set_xlim(-2.9, 9.4); ax.set_ylim(-1.35, 6.5); ax.set_aspect('equal'); ax.axis('off')

SUB = '(node colour = path level; grey lines = parent-child capture; arrows = rotation increment on that hop)'
fig, axes = plt.subplots(1, 2, figsize=(16, 6.3)); fig.subplots_adjust(left=0.01, right=0.99, top=0.86, bottom=0.01, wspace=0.03)
panel(axes[0], 'voxel'); panel(axes[1], 'coherent')
fig.suptitle('Intragranular misorientation in the SPPARKS additive apps: the same lineage tree under the two walks\n' + SUB + '; the combined model applies both at every hop', fontsize=10.5, color=INK, x=0.02, ha='left', y=0.985)
fig.savefig(OUT + 'walk_schematic.png', dpi=150, bbox_inches='tight')

fig, ax = plt.subplots(figsize=(8, 6.3)); fig.subplots_adjust(left=0.01, right=0.99, top=0.86, bottom=0.01)
panel(ax, 'mixed')
fig.suptitle('The combined model on the same lineage tree\n' + SUB, fontsize=10.5, color=INK, x=0.02, ha='left', y=0.985)
fig.savefig(OUT + 'walk_schematic_mixed.png', dpi=150, bbox_inches='tight'); print('saved')
