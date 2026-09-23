#!/usr/bin/env python3
"""Static 3-D render of a whole build from its stitch file: solidified voxels in IPF-Z color, like Lamb 2024 Fig. 3(b).
usage (WORK env): render_domain_3d.py BUILD.st OUT.png [--step 2] [--dx 3e-6]"""
import sys, os, argparse, numpy as np
sys.path.insert(0, '/home/tron/spparks-micro-analysis/src')
from spparks_micro_analysis.stitch_reader import StitchReader
from spparks_micro_analysis.orientation_analysis import generate_ipf_colors
import pyvista as pv; pv.OFF_SCREEN = True
ap = argparse.ArgumentParser(); ap.add_argument('stitch'); ap.add_argument('out')
ap.add_argument('--step', type=int, default=2, help='voxel stride (2 = 6 um cells at 3 um lattice)'); ap.add_argument('--dx', type=float, default=3e-6)
ap.add_argument('--size', type=int, default=1600)
a = ap.parse_args(); dxu = a.dx * 1e6 * a.step
with StitchReader(a.stitch) as r:
    b = r.get_global_bounds(); t = r.times[-1]
    f = {k: r.read_field_at_time(k, t, bounds=b)[0][::a.step, ::a.step, ::a.step] for k in ('i2', 'd1', 'd2', 'd3', 'd4')}
solid = f['i2'] == 3                                    # solidified only: powder, liquid and the unremelted substrate are hidden
q = np.stack([f['d1'], f['d2'], f['d3'], f['d4']], -1)
rgb = np.zeros(solid.shape + (3,), np.uint8); rgb[solid] = (np.clip(generate_ipf_colors(q[solid], 'fcc', 'z'), 0, 1) * 255).astype(np.uint8)
nx, ny, nz = solid.shape; print('grid', solid.shape, 'solid voxels', int(solid.sum()), flush=True)
g = pv.ImageData(dimensions=(nx + 1, ny + 1, nz + 1), spacing=(dxu, dxu, dxu))
g.cell_data['m'] = solid.ravel(order='F').astype(np.uint8); g.cell_data['rgb'] = rgb.transpose(2, 1, 0, 3).reshape(-1, 3)   # Fortran cell order
body = g.threshold(0.5, scalars='m').extract_surface()
p = pv.Plotter(off_screen=True, window_size=(a.size, int(a.size * 0.8))); p.set_background('white'); p.enable_parallel_projection()
p.add_mesh(body, scalars='rgb', rgb=True, smooth_shading=False)
L = nx * dxu; H = nz * dxu
p.camera_position = [(L/2 + 1.6*L, L/2 - 1.3*L, H + 0.9*L), (L/2, L/2, H/2), (0, 0, 1)]; p.camera.parallel_scale = 0.62 * L
z0 = np.argmax(solid.any((0, 1))) * dxu                                   # lowest solidified level (the remelted substrate top)
p.add_mesh(pv.Line((0, -25, z0), (200, -25, z0)), color='black', line_width=6)   # 200 um bar along the front bottom edge
p.add_point_labels([(100, -25, z0 - 90)], ['200 µm'], font_size=int(a.size / 45), text_color='black', shape=None, show_points=False, always_visible=True, justification_horizontal='center')
p.screenshot(a.out); print('saved', a.out)
