#!/usr/bin/env python3
"""3-D renders of individual simulated grains in IPF-Z and SROD colouring, like Lamb 2024 Fig. 4(a,b).
usage (WORK env): grain_3d_render.py BUILD.st OUT.png --grains 584 657 142 1062 [--crop 200 600 200 600] [--floor-um 100] [--dx 3e-6]
Grain ids are the relabel_grains labels of the same crop, as printed by grain_paths.py (--lamb-pick spread)."""
import sys, os, argparse, numpy as np
sys.path.insert(0, '/home/tron/spparks-micro-analysis/src'); sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'scripts'))
from spparks_micro_analysis.stitch_reader import StitchReader
from spparks_micro_analysis.grain_size_analysis import relabel_grains
from spparks_micro_analysis.orientation_analysis import generate_ipf_colors
from srod_analysis import disorientation_deg
import pyvista as pv; pv.OFF_SCREEN = True
from PIL import Image, ImageDraw, ImageFont

ap = argparse.ArgumentParser(); ap.add_argument('stitch'); ap.add_argument('out'); ap.add_argument('--grains', type=int, nargs='+', required=True)
ap.add_argument('--crop', type=float, nargs=4, default=[200, 600, 200, 600]); ap.add_argument('--floor-um', type=float, default=100.0); ap.add_argument('--dx', type=float, default=3e-6)
ap.add_argument('--srod-max', type=float, default=25.0); ap.add_argument('--size', type=int, default=420, help='pixels per grain panel')
a = ap.parse_args(); dxu = a.dx * 1e6
with StitchReader(a.stitch) as r:
    b = r.get_global_bounds(); t = r.times[-1]
    f = {k: r.read_field_at_time(k, t, bounds=b)[0] for k in ('i1', 'i2', 'd1', 'd2', 'd3', 'd4')}
c = [int(round(v / dxu)) for v in a.crop]; f = {k: v[c[0]:c[1], c[2]:c[3], :] for k, v in f.items()}
solid = f['i2'] == 3; q = np.stack([f['d1'], f['d2'], f['d3'], f['d4']], -1)
lab = relabel_grains(np.where(solid, f['i1'], 0).astype(np.int64))
zfloor = int(a.floor_um / dxu)

def render(mask, colors=None, scalars=None, clim=None, cmap='turbo'):
    """mask: bool (nx,ny,nz) of the grain inside its bounding box; returns a screenshot array."""
    nx, ny, nz = mask.shape
    g = pv.ImageData(dimensions=(nx + 1, ny + 1, nz + 1), spacing=(dxu, dxu, dxu))
    g.cell_data['m'] = mask.ravel(order='F').astype(np.uint8)
    if colors is not None:
        rgb = np.zeros((mask.size, 3), np.uint8); rgb[mask.ravel(order='F')] = (np.clip(colors, 0, 1) * 255).astype(np.uint8); g.cell_data['rgb'] = rgb
    if scalars is not None:
        sv = np.zeros(mask.size); sv[mask.ravel(order='F')] = scalars; g.cell_data['s'] = sv
    body = g.threshold(0.5, scalars='m')
    p = pv.Plotter(off_screen=True, window_size=(a.size, int(a.size * 1.6)))
    if colors is not None: p.add_mesh(body, scalars='rgb', rgb=True, smooth_shading=False)
    else: p.add_mesh(body, scalars='s', cmap=cmap, clim=clim, show_scalar_bar=False)
    p.add_mesh(pv.Line((0, 0, -20), (100, 0, -20)), color='black', line_width=4)   # 100 um bar
    p.set_background('white'); p.enable_parallel_projection()
    p.camera_position = [(nx*dxu/2 + 900, ny*dxu/2 - 700, nz*dxu/2 + 350), (nx*dxu/2, ny*dxu/2, nz*dxu/2), (0, 0, 1)]
    p.camera.parallel_scale = 230
    return p.screenshot(return_img=True)

tiles_ipf, tiles_srod, info = [], [], []
for k, gid in enumerate(a.grains):
    m = lab == gid
    if not m.any(): print('grain', gid, 'not found'); continue
    idx = np.argwhere(m); lo = idx.min(0); hi = idx.max(0) + 1
    sub = m[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
    qq = q[m]; zc = idx[:, 2]; o = idx[np.argmin(zc)]
    srod = disorientation_deg(np.broadcast_to(q[tuple(o)], (len(qq), 4)), qq)
    rgb = generate_ipf_colors(qq, 'fcc', 'z')
    # order of voxels inside the bounding box (Fortran order for pyvista cell data)
    order = np.ravel_multi_index((idx[:, 0]-lo[0], idx[:, 1]-lo[1], idx[:, 2]-lo[2]), sub.shape, order='F'); srt = np.argsort(order)
    tiles_ipf.append(render(sub, colors=rgb[srt])); tiles_srod.append(render(sub, scalars=srod[srt], clim=(0, a.srod_max)))
    h = (zc.max() - zc.min() + 1) * dxu; info.append((gid, h, srod.max(), srod.mean()))
    print(f'G{k+1} id {gid}: height {h:.0f} um, {m.sum()} voxels, SROD max {srod.max():.1f} mean {srod.mean():.1f} deg', flush=True)
# composite: row 1 IPF, row 2 SROD, colour bar at right
W, H = tiles_ipf[0].shape[1], tiles_ipf[0].shape[0]; n = len(tiles_ipf); pad = 8; barw = 90
out = Image.new('RGB', (n * (W + pad) + barw, 2 * (H + pad) + 30), 'white'); d = ImageDraw.Draw(out)
try: font = ImageFont.truetype('/usr/share/fonts/open-sans/OpenSans-Regular.ttf', 22)
except Exception: font = ImageFont.load_default()
for k, (ti, ts) in enumerate(zip(tiles_ipf, tiles_srod)):
    x = k * (W + pad); out.paste(Image.fromarray(ti), (x, 30)); out.paste(Image.fromarray(ts), (x, H + pad + 30))
    d.text((x + 10, 4), f'G{k+1}', fill='black', font=font); d.text((x + 10, H + pad + 34), f'max {info[k][2]:.0f}°', fill='black', font=font)
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(barw/100, H/100)); fig.subplots_adjust(left=0.05, right=0.35, top=0.95, bottom=0.05)
plt.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(0, a.srod_max), cmap='turbo'), cax=ax, label='SROD (°)'); fig.savefig('/tmp/_cbar.png', dpi=100); plt.close(fig)
cb = Image.open('/tmp/_cbar.png').convert('RGB'); out.paste(cb, (n * (W + pad), H + pad + 30))
out.save(a.out); print('saved', a.out, out.size)
