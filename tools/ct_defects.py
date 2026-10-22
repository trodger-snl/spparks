#!/usr/bin/env python3
"""Rasterise CT-measured defects (.glb meshes) into an i2 = 5 mask for SPPARKS.

The mask is read by additive/texture and additive/texture/kk with

    set i2 hdf5 mask.h5 skip_novalue

i2 = 5 is the permanent void state: it behaves as air and never melts,
solidifies or becomes powder. The file uses the dump hdf5 layout: an int32
dataset 'i2' shaped [z][y][x], with -1 (no_value) everywhere except the
defects, and root attributes origin = (0,0,0) and dims = (nx,ny,nz) in x,y,z
order. Sites are keyed by absolute lattice index (lattice sc/26n 1.0,
boundary n n n: site (i,j,k) sits at x = i), so a single mask that covers the
whole build serves every sliding box of a staged run.

Modes:
  info   GLB    print part and defect statistics
  crop   GLB    place defects at their measured positions; --origin-mm is the
                CT point that maps to lattice (0,0,0), with axes x->x, y->y, z->z
  sample GLB    Poisson count at --density (default: the CT mean), shapes drawn
                with replacement, uniform random rotation and centre
  selftest      check the rasteriser on meshes with exactly known site counts

Every defect marks at least one site. If no site centre falls inside the mesh,
the site nearest its centroid is marked, and the report counts those cases.

Facts about the Huddle CT15 export (tiff - CT15-2 ..._registered_cleaned.glb):
  - coordinates are in mm, although each mesh's extras say units = "meters".
    The part is a cylinder about 2.8 mm across and 15 mm tall, with z the
    build axis.
  - the xlsx assumes a 4.2 um voxel (vol = ch_vol_vox*0.0042^3). For every mesh,
    convex-hull volume / ch_vol_vox = 80.03 um^3, which implies 4.31 um for the
    registered meshes. Each mesh fills about 53 % of its hull, so mesh-volume
    sizes (median eq. diameter 16 um) are smaller than the hull-based xlsx sizes.
  - the xlsx "eq. diam" column computes (pi/6*V)^(1/3), not (6V/pi)^(1/3), so it
    reads 0.65x low.
  - extras.max_feret_in_microns is ~2x the mesh bounding-box diagonal, so the
    metadata disagrees with the geometry. This tool uses only the geometry.
  - 11 of the 1977 meshes are not closed. The generalized winding number
    (inside when |w| >= 0.5) tolerates them.
  - defects concentrate near the part surface. `sample` places them uniformly,
    so `info` prints the interior density as well as the mean.

Needs numpy and h5py only.
"""
import argparse
import csv
import json
import struct
import sys

import numpy as np

COMP = {5121: np.uint8, 5123: np.uint16, 5125: np.uint32, 5126: np.float32}
NCOMP = {"SCALAR": 1, "VEC3": 3}


def read_glb(path):
    """Return a list of (name, V[n,3] float64, F[m,3] int64) in file units."""
    b = open(path, "rb").read()
    magic, version, _ = struct.unpack("<4sII", b[:12])
    if magic != b"glTF" or version != 2:
        sys.exit(f"{path}: not a glTF 2 binary")
    jlen, jtype = struct.unpack("<II", b[12:20])
    j = json.loads(b[20:20 + jlen])
    bin0 = 20 + jlen + 8
    for n in j.get("nodes", []):
        if any(k in n for k in ("matrix", "translation", "rotation", "scale")):
            sys.exit(f"{path}: node transforms are not supported ({n.get('name')})")

    def acc(i):
        a = j["accessors"][i]
        v = j["bufferViews"][a["bufferView"]]
        if v.get("byteStride", 0) not in (0, NCOMP[a["type"]] * np.dtype(COMP[a["componentType"]]).itemsize):
            sys.exit(f"{path}: interleaved buffer views are not supported")
        off = bin0 + v.get("byteOffset", 0) + a.get("byteOffset", 0)
        x = np.frombuffer(b, COMP[a["componentType"]], a["count"] * NCOMP[a["type"]], off)
        return x.reshape(-1, NCOMP[a["type"]]) if NCOMP[a["type"]] > 1 else x

    out = []
    for m in j["meshes"]:
        V, F = [], []
        for p in m["primitives"]:
            if p.get("mode", 4) != 4:
                sys.exit(f"{path}: mesh {m.get('name')} is not a triangle list")
            F.append(acc(p["indices"]).astype(np.int64).reshape(-1, 3) + sum(len(v) for v in V))
            V.append(acc(p["attributes"]["POSITION"]).astype(np.float64))
        out.append((m.get("name", f"mesh{len(out)}"), np.vstack(V), np.vstack(F)))
    return out


def mesh_volume(V, F):
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    return abs(np.einsum("ij,ij->i", a, np.cross(b, c)).sum()) / 6.0


def inside(P, V, F):
    """Generalized winding number test (Jacobson 2013), orientation-agnostic."""
    T = V[F]                                   # (m,3,3)
    w = np.empty(len(P))
    step = max(1, 2_000_000 // len(F))         # bound the (k,m,3) temporaries
    for s in range(0, len(P), step):
        d = T[None, :, :, :] - P[s:s + step, None, None, :]   # (k,m,3,3)
        a, b, c = d[:, :, 0], d[:, :, 1], d[:, :, 2]
        la, lb, lc = (np.linalg.norm(x, axis=-1) for x in (a, b, c))
        num = np.einsum("kmi,kmi->km", a, np.cross(b, c))
        den = (la * lb * lc + np.einsum("kmi,kmi->km", a, b) * lc
               + np.einsum("kmi,kmi->km", b, c) * la + np.einsum("kmi,kmi->km", c, a) * lb)
        w[s:s + step] = np.arctan2(num, den).sum(1) / (2 * np.pi)
    return np.abs(w) >= 0.5


def rasterise(V, F, dims):
    """V in lattice units. Return (sites[n,3] int, used_minimum) inside dims, or (None, False)."""
    dims = np.asarray(dims)
    lo = np.maximum(np.ceil(V.min(0)).astype(int), 0)
    hi = np.minimum(np.floor(V.max(0)).astype(int), dims - 1)
    if np.all(hi >= lo):
        g = np.stack(np.meshgrid(*[np.arange(l, h + 1) for l, h in zip(lo, hi)],
                                 indexing="ij"), -1).reshape(-1, 3)
        if len(g):
            hit = g[inside(g.astype(float), V, F)]
            if len(hit):
                return hit, False
    c = np.rint(V.mean(0)).astype(int)
    if np.all(c >= 0) and np.all(c < dims):
        return c[None, :], True
    return None, False


def random_rotation(rng):
    u1, u2, u3 = rng.random(3)                 # Shoemake uniform unit quaternion
    q = np.array([np.sqrt(1 - u1) * np.sin(2 * np.pi * u2), np.sqrt(1 - u1) * np.cos(2 * np.pi * u2),
                  np.sqrt(u1) * np.sin(2 * np.pi * u3), np.sqrt(u1) * np.cos(2 * np.pi * u3)])
    w, x, y, z = q[3], q[0], q[1], q[2]
    R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                  [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                  [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
    return np.array([w, x, y, z]), R


def part_stats(defects):
    """Cylinder fit to the CT part (mm): axis centre, radius, height, densities."""
    allv = np.vstack([V for _, V, _ in defects])
    cen = np.array([V.mean(0) for _, V, _ in defects])
    c = (allv.min(0) + allv.max(0)) / 2
    R = np.hypot(allv[:, 0] - c[0], allv[:, 1] - c[1]).max()
    H = allv[:, 2].max() - allv[:, 2].min()
    r = np.hypot(cen[:, 0] - c[0], cen[:, 1] - c[1])
    return dict(center=c, zmin=allv[:, 2].min(), R=R, H=H,
                rho_mean=len(defects) / (np.pi * R * R * H),
                rho_interior=(r < 1.0).sum() / (np.pi * 1.0 * H), r=r)


def info(defects):
    s = part_stats(defects)
    vol = np.array([mesh_volume(V, F) for _, V, F in defects]) * 1e9   # um^3
    deq = (6 * vol / np.pi) ** (1 / 3)
    print(f"defects        {len(defects)}")
    print(f"part (cyl fit) centre x,y = {s['center'][0]:.4f}, {s['center'][1]:.4f} mm, "
          f"z {s['zmin']:.4f} .. {s['zmin'] + s['H']:.4f} mm, R = {s['R']:.4f} mm")
    print(f"density        mean {s['rho_mean']:.2f} /mm^3, interior (r < 1 mm) "
          f"{s['rho_interior']:.2f} /mm^3")
    print(f"porosity       {vol.sum() * 1e-9 / (np.pi * s['R'] ** 2 * s['H']):.3e} (mesh volume / cylinder)")
    print("eq. diameter   um p5/p50/p95/max = " + " ".join(f"{x:.1f}" for x in np.percentile(deq, [5, 50, 95, 100])))
    return s


def place(placements, dims, dx_mm, out, listfile):
    """placements: list of (name, V_lattice, F, quat, mesh_vol_mm3)."""
    mask = np.zeros(dims[::-1], np.int8)       # [z][y][x]
    rows, nmin, nsites, vmesh = [], 0, 0, 0.0
    for name, V, F, q, vol in placements:
        sites, used_min = rasterise(V, F, dims)
        if sites is None:
            continue
        mask[sites[:, 2], sites[:, 1], sites[:, 0]] = 1
        nmin += used_min
        nsites += len(sites)
        vmesh += vol
        c = V.mean(0)
        rows.append([name, *np.round(c, 2), *np.round(q, 6), len(sites),
                     round(vol * 1e9, 2), round(len(sites) * dx_mm ** 3 * 1e9, 2)])
    nvoid = int(mask.sum(dtype=np.int64))
    dvol = np.prod(dims) * dx_mm ** 3
    print(f"placed         {len(rows)} defects, {nmin} at the 1-site minimum")
    print(f"void sites     {nvoid} ({nsites - nvoid} shared by overlapping defects)")
    print(f"porosity       sites {nvoid / np.prod(dims):.3e}, mesh {vmesh / dvol:.3e}"
          " (mesh counts clipped defects whole)")
    write_mask(out, mask, dims)
    if listfile:
        with open(listfile, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["name", "i", "j", "k", "q0", "qx", "qy", "qz",
                        "n_sites", "mesh_vol_um3", "raster_vol_um3"])
            w.writerows(rows)
    return mask


def write_mask(path, mask, dims):
    import h5py
    nz, ny, nx = mask.shape
    with h5py.File(path, "w") as f:
        f.attrs["origin"] = np.zeros(3, np.int64)
        f.attrs["dims"] = np.asarray(dims, np.int64)
        d = f.create_dataset("i2", mask.shape, np.int32, fillvalue=-1,
                             chunks=(min(4, nz), min(128, ny), min(128, nx)),
                             compression="gzip", compression_opts=1)
        for z in range(nz):                    # int8 in memory, int32 on disk
            d[z] = np.where(mask[z], 5, -1).astype(np.int32)
        d.attrs["field"] = "i2"
        d.attrs["type"] = "int32"
        d.attrs["per_site_length"] = np.int32(1)
        d.attrs["no_value"] = np.int32(-1)
    print(f"wrote          {path}  dims {[int(x) for x in dims]}")


def selftest():
    def cube(c, h):
        V = np.array([[x, y, z] for x in (-h, h) for y in (-h, h) for z in (-h, h)], float) + c
        F = np.array([[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1],
                      [2, 3, 7], [2, 7, 6], [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]])
        return V, F

    def octa(c, r):
        V = np.array([[r, 0, 0], [-r, 0, 0], [0, r, 0], [0, -r, 0], [0, 0, r], [0, 0, -r]], float) + c
        F = np.array([[0, 2, 4], [2, 1, 4], [1, 3, 4], [3, 0, 4],
                      [2, 0, 5], [1, 2, 5], [3, 1, 5], [0, 3, 5]])
        return V, F

    dims = (40, 40, 40)
    V, F = cube(np.array([20.0, 20, 20]), 5.2)
    s, m = rasterise(V, F, dims)
    assert len(s) == 11 ** 3 and not m, len(s)
    s, _ = rasterise(V, F[:, ::-1], dims)      # winding orientation must not matter
    assert len(s) == 11 ** 3
    V, F = octa(np.array([20.0, 20, 20]), 4.5)
    g = np.stack(np.meshgrid(*[np.arange(-5, 6)] * 3, indexing="ij"), -1).reshape(-1, 3)
    s, _ = rasterise(V, F, dims)
    assert len(s) == (np.abs(g).sum(1) <= 4.5).sum(), len(s)
    V, F = octa(np.array([3.3, 3.3, 3.3]), 0.2)  # sub-voxel: nearest site only
    s, m = rasterise(V, F, dims)
    assert m and s.tolist() == [[3, 3, 3]], s
    V, F = cube(np.array([-30.0, 20, 20]), 2)    # wholly outside: not placed
    assert rasterise(V, F, dims)[0] is None
    # crop: a cube centred at CT (1.03, 2.06, 3.09) mm, origin (1, 2, 3) mm,
    # dx 3 um -> centred on lattice (10, 20, 30); 7 um half-width -> 5^3 sites
    V, F = cube(np.array([1.03, 2.06, 3.09]), 0.007)
    s, _ = rasterise((V - np.array([1.0, 2.0, 3.0])) / 0.003, F, dims)
    assert len(s) == 125 and s.mean(0).tolist() == [10, 20, 30], (len(s), s.mean(0))
    q, R = random_rotation(np.random.default_rng(1))
    assert abs(np.linalg.norm(q) - 1) < 1e-12 and np.allclose(R @ R.T, np.eye(3))
    print("selftest ok")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="mode", required=True)
    sub.add_parser("selftest")
    p = sub.add_parser("info")
    p.add_argument("glb")
    for name in ("crop", "sample"):
        p = sub.add_parser(name)
        p.add_argument("glb")
        p.add_argument("--dx", type=float, required=True, help="lattice spacing in m (the app_style dx)")
        p.add_argument("--dims", type=int, nargs=3, required=True, metavar=("NX", "NY", "NZ"),
                       help="global lattice extent of the whole build")
        p.add_argument("--out", required=True, help="mask .h5 to write")
        p.add_argument("--list", help="CSV of placed defects")
        if name == "crop":
            p.add_argument("--origin-mm", type=float, nargs=3, required=True, metavar=("X", "Y", "Z"),
                           help="CT point (mm) that maps to lattice site (0,0,0)")
        else:
            p.add_argument("--seed", type=int, required=True)
            p.add_argument("--density", type=float, help="defects per mm^3 (default: CT mean)")
    a = ap.parse_args()

    if a.mode == "selftest":
        return selftest()
    defects = read_glb(a.glb)
    s = info(defects)
    if a.mode == "info":
        return
    dx_mm = a.dx * 1e3
    dims = np.array(a.dims)
    vols = [mesh_volume(V, F) for _, V, F in defects]
    if a.mode == "crop":
        o = np.array(a.origin_mm)
        pl = [(n, (V - o) / dx_mm, F, np.array([1.0, 0, 0, 0]), v)
              for (n, V, F), v in zip(defects, vols)]
    else:
        rng = np.random.default_rng(a.seed)
        rho = a.density if a.density is not None else s["rho_mean"]
        n = rng.poisson(rho * np.prod(dims) * dx_mm ** 3)
        print(f"sampling       {n} defects at {rho:.2f} /mm^3 (seed {a.seed})")
        # ponytail: overlaps allowed; negligible at CT porosity, reject overlaps if sampling densely
        pl = []
        for _ in range(n):
            k = rng.integers(len(defects))
            name, V, F = defects[k]
            q, R = random_rotation(rng)
            c = rng.random(3) * dims - 0.5
            pl.append((name, (V - V.mean(0)) @ R.T / dx_mm + c, F, q, vols[k]))
    place(pl, dims, dx_mm, a.out, a.list)


if __name__ == "__main__":
    main()
