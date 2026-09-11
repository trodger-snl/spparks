#!/usr/bin/env python3
"""Two-panel schematic contrasting the per-voxel walk and the coherent grain walk on the same
lineage tree (docs/walk_schematic.png). Left: every hop gets its own random rotation, lineages
diverge, the grain mean stays. Right: every hop crossing path level k gets the SAME rotation v_k,
correlated between levels, so all voxels of a level share one orientation and the grain bends."""
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon
from matplotlib.path import Path as MPath
rng = np.random.default_rng(11)
INK, MUTED, BLUE, ORANGE, GREY = '#1b2230', '#5b6470', '#2a78d6', '#c9552a', '#c9cdd3'
cmap = plt.get_cmap('viridis')

# ---------- one lineage tree used by both panels: levels k = 0..5 (path length), 1,2,3,4,5,6 nodes
layers = [1, 2, 3, 4, 5, 6]
pos = {}
for L, n in enumerate(layers):
    for j in range(n):
        pos[(L, j)] = (1.0 + (j - (n-1)/2)*0.95, 0.6 + L*1.05)
parents = {}
for L in range(1, len(layers)):
    for j in range(layers[L]):
        parents[(L, j)] = (L-1, min(j, layers[L-1]-1))

# ---------- rotation increments (2-D stand-ins for rotation vectors), per panel
NL = len(layers)
# per-voxel: independent per edge
inc_v = {c: rng.normal(0, 1, 2) for c in parents}
# coherent: one vector per level, OU-correlated (rho = 0.8)
inc_g = []; v = rng.normal(0, 1, 2)
for L in range(NL-1):
    v = 0.8*v + np.sqrt(1-0.8**2)*rng.normal(0, 1, 2); inc_g.append(v.copy())
inc_g = np.array(inc_g)
# rotate the coherent sequence so its net drift points into the IPF triangle (purely cosmetic)
tot = inc_g.sum(0); target = np.array([0.55, 1.0]); ang = np.arctan2(target[1], target[0]) - np.arctan2(tot[1], tot[0])
Rm = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]]); inc_g = inc_g @ Rm.T
inc_g *= 1.6/np.linalg.norm(tot)
# resulting "orientation" of every node (2-D, for the IPF sketch): cumulative along the lineage
def orientation(node, per_edge):
    o = np.zeros(2); c = node
    while c in parents:
        o = o + per_edge(c); c = parents[c]
    return o
ori_v = {n: orientation(n, lambda c: inc_v[c]) for n in pos}
ori_g = {n: orientation(n, lambda c: inc_g[c[0]-1]) for n in pos}

fig, axes = plt.subplots(1, 2, figsize=(16, 6.3)); fig.subplots_adjust(left=0.01, right=0.99, top=0.86, bottom=0.01, wspace=0.03)
titles = ["(b) Per-voxel walk  (misorientation_diffusion $D_v$)\nevery hop: its own random rotation",
          "(c) Coherent grain walk  (misorientation_grain_walk $D_g$ $\\rho$ path)\nevery hop across level k: the same rotation $v_k$ for the whole grain"]
for ax, title, inc_of, ori, coherent in ((axes[0], titles[0], lambda c: inc_v[c], ori_v, False), (axes[1], titles[1], lambda c: inc_g[c[0]-1], ori_g, True)):
    ax.set_title(title, loc='left', fontsize=11, color=INK)
    # level bands and labels
    for L in range(NL):
        y = pos[(L, 0)][1]
        ax.axhspan(y-0.5, y+0.5, xmin=0.02, xmax=0.62, color='#eef2f7' if L % 2 else '#f7f8fa', zorder=0)
        ax.text(-1.75, y, f'k = {L}', va='center', ha='right', fontsize=9, color=MUTED)
    ax.text(-1.75, pos[(0,0)][1]-0.5, 'path level\n(d10)', va='top', ha='right', fontsize=8.5, color=MUTED)
    # edges with their rotation arrows
    for c, p in parents.items():
        (x1, y1), (x0, y0) = pos[c], pos[p]
        ax.annotate('', xy=(x1, y1-0.2), xytext=(x0, y0+0.2), arrowprops=dict(arrowstyle='-', color=GREY, lw=1.0))
        mx, my = (x0+x1)/2, (y0+y1)/2; d = inc_of(c); d = 0.28*d/np.linalg.norm(d)
        col = cmap((c[0]-0.5)/(NL-1)) if coherent else ORANGE
        ax.annotate('', xy=(mx+d[0], my+d[1]), xytext=(mx, my), arrowprops=dict(arrowstyle='-|>', color=col, lw=1.6, mutation_scale=9))
    for n, (x, y) in pos.items():
        ax.add_patch(Circle((x, y), 0.17, facecolor=cmap(n[0]/(NL-1)) if coherent else '#bcd3f2', edgecolor=INK, lw=0.7, zorder=3))
    ax.text(pos[(0,0)][0], pos[(0,0)][1]-0.45, 'grain origin (reference orientation)', ha='center', fontsize=8.5, color=INK)
    # legend text on the right
    if not coherent:
        ax.text(3.9, 6.2, r'$q_{child} = R(\theta_e, \hat n_e)\; q_{parent}$', fontsize=10, color=INK)
        ax.text(3.9, 5.75, r'per edge e: $\theta_e = \sqrt{D_v\,\mathrm{hop}}$, axis $\hat n_e$ random', fontsize=9, color=INK)
        ax.text(3.9, 5.3, 'orange arrows: one independent increment per edge', fontsize=8.5, color=MUTED)
        ax.text(3.9, 4.75, 'two voxels at the same level differ by the sum of\nall increments back to their common ancestor', fontsize=8.5, color=MUTED, va='top')
        ax.text(3.9, 3.95, r'population: $\mathrm{RMS}^2 = D_v \times$ path,  Maxwell width', fontsize=9, color=INK, va='top')
        ax.text(3.9, 3.55, 'grain mean: does not move;  intragranular spread: grows', fontsize=9, color=INK, va='top')
    else:
        ax.text(3.9, 6.2, r'$q_{child} = R(v_k)\; q_{parent}$  for every edge that crosses level k', fontsize=10, color=INK)
        ax.text(3.9, 5.75, r'$v_k = \rho\, v_{k-1} + \sqrt{1-\rho^2}\; v_k^{\prime}$,  $|v_k|_{\mathrm{rms}} = \sqrt{D_g \Delta x}$', fontsize=9, color=INK)
        ax.text(3.9, 5.3, 'coloured arrows: one increment per level, shared by all edges\nof that level; its direction persists over ~1/(1−ρ) levels', fontsize=8.5, color=MUTED, va='top')
        ax.text(3.9, 4.55, 'two voxels at the same level have received the same\nincrements whatever route they took: identical orientation', fontsize=8.5, color=MUTED, va='top')
        ax.text(3.9, 3.7, r'population: same $\sqrt{h}$ law (coherent part $D_g\frac{1+\rho}{1-\rho}$ per unit path)', fontsize=9, color=INK, va='top')
        ax.text(3.9, 3.3, 'grain mean: bends along a path;  intragranular spread: none', fontsize=9, color=INK, va='top')
    # IPF sketch: triangle with the node orientations
    tri_pts = [(3.9, -0.45), (7.9, -0.45), (7.25, 2.75)]; ax.add_patch(Polygon(tri_pts, closed=True, fill=False, edgecolor=INK, lw=0.9))
    tp = MPath(tri_pts); centre = np.array([5.6, 0.55]); scale = 0.24 if not coherent else 1.0
    P = np.array([ori[n] for n in pos]); Lv = np.array([n[0] for n in pos])
    xy = centre + scale*P
    xy = np.array([pt if tp.contains_point(pt) else centre + 0.5*(pt-centre) for pt in xy])
    if coherent:
        for L in range(NL-1):
            a, b = centre + scale*ori[(L,0)], centre + scale*ori[(L+1,0)]
            ax.annotate('', xy=b, xytext=a, arrowprops=dict(arrowstyle='-|>', color=cmap((L+0.5)/(NL-1)), lw=2.2, mutation_scale=12), zorder=2)
        jitter = rng.normal(0, 0.03, xy.shape)
        ax.scatter(xy[:,0]+jitter[:,0], xy[:,1]+jitter[:,1], c=Lv, cmap='viridis', s=34, edgecolor=INK, linewidths=0.4, zorder=3)
        ax.add_patch(Circle(tuple(centre), 0.06, color=INK, zorder=4))
        ax.text(5.9, -0.65, 'IPF: all voxels of a level sit at one point;\nthe levels trace a persistent path away from the origin', ha='center', fontsize=8.5, color=MUTED, va='top')
    else:
        ax.scatter(xy[:,0], xy[:,1], c=Lv, cmap='viridis', s=34, edgecolor=INK, linewidths=0.4, zorder=3)
        ax.add_patch(Circle(tuple(centre), 0.06, color=INK, zorder=4))
        ax.text(5.9, -0.65, 'IPF: voxels scatter into a cloud that widens with level,\ncentred on the origin (black dot)', ha='center', fontsize=8.5, color=MUTED, va='top')
    for (x_, y_, s_, va) in ((3.9, -0.5, '001', 'top'), (7.9, -0.5, '101', 'top'), (7.25, 2.8, '111', 'bottom')):
        ax.text(x_, y_, s_, fontsize=7, color=MUTED, ha='center', va=va)
    ax.set_xlim(-2.9, 9.4); ax.set_ylim(-1.35, 6.5); ax.set_aspect('equal'); ax.axis('off')
fig.suptitle('Intragranular misorientation in the SPPARKS additive apps: the same lineage tree under the two walks\n(node colour = path level; grey lines = parent-child capture; arrows = rotation increment on that hop; the combined model applies both at every hop)', fontsize=10.5, color=INK, x=0.02, ha='left', y=0.985)
fig.savefig('/home/tron/spparks-kokkos/docs/walk_schematic.png', dpi=150, bbox_inches='tight'); print('saved')
