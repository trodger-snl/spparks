#!/usr/bin/env python3
"""Part footprint mask from in-situ monitoring data, for `region ID voxel FILE`.

usage: insitu_footprint.py INSITU.h5 LAYER OUT --dx DX --window XLO XHI YLO YHI
           [--margin 5e-3] [--depth 3e-3] [--grid 50e-6] [--close 0.1e-3] [--pmin 50]
       insitu_footprint.py selftest

Each layer's footprint is where the laser (power >= --pmin W) was on: the samples are
binned on a --grid square grid, and a closing of radius --close fills the gaps between
hatches (hatch spacing 0.12 mm needs >= 0.06 mm) without growing the outline. The
footprints of LAYER and the layers below it, down to --depth, stack into one z slice per
layer over the window plus --margin (all lengths in metres, in-situ frame). The file is in
lattice units (metres / --dx); `region voxel` extends the bottom slice below it.
"""
import sys
import numpy as np
from scipy.ndimage import binary_closing


def footprint(x, y, x0, y0, g, nx, ny, close):
    """nx x ny closed footprint of points (x, y) on the grid at (x0, y0), spacing g."""
    r = max(1, int(np.ceil(close / g)))
    p = 2 * r                              # rasterize beyond the crop so closing sees its neighbours
    i = np.floor((x - x0) / g).astype(int) + p
    j = np.floor((y - y0) / g).astype(int) + p
    hit = np.zeros((ny + 2 * p, nx + 2 * p), bool)
    ok = (i >= 0) & (i < nx + 2 * p) & (j >= 0) & (j < ny + 2 * p)
    hit[j[ok], i[ok]] = True
    a = np.arange(-r, r + 1)
    disk = a[:, None] ** 2 + a[None, :] ** 2 <= r * r
    return binary_closing(hit, disk)[p:-p, p:-p]


def write(path, mask, x0, y0, z0, g, dz, dx):
    nz, ny, nx = mask.shape
    with open(path, 'wb') as f:
        f.write(f'voxel {nx} {ny} {nz} {x0 / dx:.10g} {y0 / dx:.10g} {z0 / dx:.10g} '
                f'{g / dx:.10g} {g / dx:.10g} {dz / dx:.10g}\n'.encode())
        f.write(mask.astype(np.uint8).tobytes())


def build(src, layer, out, dx, window, margin=5e-3, depth=3e-3, grid=50e-6, close=0.1e-3, pmin=50.0):
    import h5py
    from laser_path_convert import read_insitu
    with h5py.File(src, 'r') as f:
        have = {int(k[6:]) for k in f if k.startswith('layer_')}
        z = {k: float(f[f'layer_{k}'].attrs['z height (mm)'][0]) * 1e-3 for k in have}
    dz = z[layer] - z[layer - 1] if layer - 1 in z else z[layer]
    layers = [k for k in range(max(min(have), layer - int(np.ceil(depth / dz))), layer + 1) if k in have]
    zs = np.array([z[k] for k in layers])
    if np.abs(np.diff(zs) - dz).max(initial=0) > 0.1 * dz:
        sys.exit(f'layers {layers[0]}-{layer}: z steps are not uniform ({dz} m)')
    x0 = np.floor((window[0] - margin) / grid) * grid
    y0 = np.floor((window[2] - margin) / grid) * grid
    nx = int(np.ceil((window[1] + margin - x0) / grid))
    ny = int(np.ceil((window[3] + margin - y0) / grid))
    mask = np.zeros((len(layers), ny, nx), bool)
    for s, k in enumerate(layers):
        t, x, y, P, _ = read_insitu(src, k)
        on = P >= pmin
        mask[s] = footprint(x[on] * 1e-3, y[on] * 1e-3, x0, y0, grid, nx, ny, close)
    write(out, mask, x0, y0, zs[0] - dz, grid, dz, dx)
    return layers, mask


def selftest():
    # a 2 mm disk hatched at 0.12 mm, sampled every 10 um: closed to the disk, holes filled
    g, R = 50e-6, 2e-3
    ys = np.arange(-R, R, 0.12e-3)
    pts = [(xx, yy) for yy in ys for xx in np.arange(-R, R, 10e-6) if xx * xx + yy * yy <= R * R]
    x, y = np.array(pts).T
    n = int(2 * 2.5e-3 / g)
    m = footprint(x, y, -2.5e-3, -2.5e-3, g, n, n, 0.1e-3)
    area = m.sum() * g * g
    assert abs(area / (np.pi * R * R) - 1) < 0.05, area
    c = n // 2
    assert m[c - 20:c + 20, c - 20:c + 20].all(), 'gaps between hatches'
    assert not m[0].any() and not m[:, 0].any(), 'outline grew to the crop edge'
    # file round trip: header in lattice units, x fastest
    import tempfile, os
    p = os.path.join(tempfile.mkdtemp(), 'v.bin')
    mm = np.zeros((2, 3, 4), bool); mm[1, 2, 3] = True
    write(p, mm, 1e-3, 2e-3, 3e-3, 50e-6, 30e-6, 5e-6)
    b = open(p, 'rb').read()
    head, data = b.split(b'\n', 1)
    assert head == b'voxel 4 3 2 200 400 600 10 10 6', head
    assert len(data) == 24 and data[23] == 1 and sum(data) == 1
    print('selftest ok')


if __name__ == '__main__':
    a = sys.argv[1:]
    if a[:1] == ['selftest']:
        selftest(); sys.exit()
    if len(a) < 3 or '--dx' not in a or '--window' not in a:
        sys.exit(__doc__)
    opt = lambda k, d: float(a[a.index(k) + 1]) if k in a else d
    w = [float(v) for v in a[a.index('--window') + 1:a.index('--window') + 5]]
    layers, mask = build(a[0], int(a[1]), a[2], opt('--dx', None), w, opt('--margin', 5e-3),
                         opt('--depth', 3e-3), opt('--grid', 50e-6), opt('--close', 0.1e-3),
                         opt('--pmin', 50.0))
    print(f'layers {layers[0]}-{layers[-1]}: {mask.shape[2]} x {mask.shape[1]} x {mask.shape[0]} voxels, '
          f'top footprint {mask[-1].sum() * opt("--grid", 50e-6) ** 2 * 1e6:.4g} mm^2 -> {a[2]}')
