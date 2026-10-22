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
  cluster GLB   patch bootstrap that keeps the CT's spatial clustering (below)
  selftest      check the rasteriser and the cluster mapping

--lo I J K (all placing modes) is the absolute lattice index of the domain's
low corner (the deck's region block lower bounds; default 0 0 0). It is
written as the mask origin.

cluster: the domain is cut into cubic tiles (--tile, default 0.4 mm, about
2x the range of the CT's defect-defect clumping). Each tile copies a CT
sub-volume of the same size, rigidly rotated about the build axis, with the
real relative positions and shapes. Clumping, size mix and depth profile
survive inside a tile; correlations across tile faces do not. The density
is the CT's and is not adjustable.
  - with no footprint source the run is fully interior: each tile comes from
    CT bulk (depth >= --bulk-depth, default 0.35 mm, below which the CT shows
    the near-surface band) at a random angle.
  - with a footprint source the part surface is the outer edge of the scanned
    footprint in each layer. A tile whose centre lies at depth d below that
    edge copies the CT sub-volume at depth d, with the CT's inward normal
    turned onto the footprint's. Tiles deeper than bulk + half a tile diagonal
    come from CT bulk. No defects are placed outside the footprint or above
    the top layer. Lattice z belongs to the first layer plane at or above it;
    sites below the first plane (substrate) use the first layer.
    Footprint sources (repeatable, merged by layer plane z; all metres):
      --laser-path "start X0 Y0 Z0 end X1 Y1 ... [hatch S N]"  deck line form
      --path-file F    laser_path waypoint file (x y z p [v])
      --gcode F        G-code, read with laser_path_convert.py
          (--path-scale / --path-shift apply to both files, as in the deck)
      --thermal-h5 F --liquidus TL   hdf5_unstructured / hdf5_csr file:
          xy of the nodes that reach TL in each layer group
    The footprint is every point within --track-radius of a scan segment (or
    a melted node), with holes filled. Default radius: S/2 for a hatched
    --laser-path, the median melted-node spacing for --thermal-h5; give it for
    files and single lines. --union uses one footprint (all layers OR'd).
    The CT depth is measured from the edge of the CT data, which assumes the
    scan includes the part's as-built surface. If it does not (a reduced field
    of view or a machined core), give --ct-radius (mm, about the fitted axis).
    The scan-edge band is then never copied, and tiles shallower than the data
    reaches come from bulk.
  ponytail: a thin wall or a corner tighter than a tile distorts depth (one
  inward normal per tile); use a smaller --tile there.

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
  - defects concentrate 0.1-0.3 mm below the part surface (25-34 /mm^3 vs ~18 in
    the bulk) and clump weakly: 2.4x the Poisson neighbour count within 50 um,
    +32 % within 100 um, +12 % within 200 um. No layer periodicity in z, no
    angular dependence, and size does not vary with depth. `sample` ignores
    both; `cluster` keeps them.
  - the xy frame matches the CT15 thermal file (ct15_0-9layer.hdf5): the part
    centre is about (40.0, -20.1) mm in both. The thermal file's melted
    footprint is ~8.1 mm across (x 35.9-44.0, y -24.1..-16.0 mm), so the 2.86 mm
    CT cylinder is the part's core, not its surface. Its 0.1-0.3 mm band lies
    ~2.7 mm inside the as-built surface (a radial trend or a scan-edge
    artifact). Use --ct-radius 4.05 with that part. The thermal layers 0-9
    (z ~ 25.0-25.3 mm) are also below the CT's z range (26.0-41.0 mm).

Needs numpy and h5py; cluster mode also uses scipy.
"""
import argparse
import csv
import json
import os
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
    print("depth profile  " + ct_profile(s))
    return s


DEPTH_BINS = [0.0, 0.1, 0.2, 0.3, 0.5, 1.0]   # mm


def ct_profile(s):
    d = s["R"] - s["r"]
    out = []
    for a, b in zip(DEPTH_BINS[:-1], DEPTH_BINS[1:]):
        ra, rb = s["R"] - b, s["R"] - a
        if rb > s.get("R_fit", s["R"]) + 1e-9:     # beyond the CT data (--ct-radius)
            out.append(f"{a:.1f}-{b:.1f} mm -")
            continue
        out.append(f"{a:.1f}-{b:.1f} mm {((d >= a) & (d < b)).sum() / (np.pi * (rb * rb - ra * ra) * s['H']):.1f}")
    return ", ".join(out) + " /mm^3"


def place(placements, dims, lo, dx_mm, out, listfile):
    """placements: list of (name, V in absolute lattice units, F, quat, mesh_vol_mm3)."""
    mask = np.zeros(dims[::-1], np.int8)       # [z][y][x]
    rows, nmin, nsites, vmesh = [], 0, 0, 0.0
    for name, V, F, q, vol in placements:
        sites, used_min = rasterise(V - lo, F, dims)
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
    write_mask(out, mask, dims, lo)
    if listfile:
        with open(listfile, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["name", "i", "j", "k", "q0", "qx", "qy", "qz",
                        "n_sites", "mesh_vol_um3", "raster_vol_um3"])
            w.writerows(rows)
    return mask


def write_mask(path, mask, dims, lo):
    import h5py
    nz, ny, nx = mask.shape
    with h5py.File(path, "w") as f:
        f.attrs["origin"] = np.asarray(lo, np.int64)
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
    print(f"wrote          {path}  origin {[int(x) for x in lo]} dims {[int(x) for x in dims]}")


# ---- cluster mode: part footprint from the scan, then the tile bootstrap ----

def parse_laser_path(line):
    """Deck `laser_path start X0 Y0 Z0 end X1 Y1 ... [hatch S N]` -> (Z0, segments[n,4], radius|None)."""
    t = line.split()
    if t and t[0] == "laser_path":
        t = t[1:]
    if len(t) < 7 or t[0] != "start" or t[4] != "end":
        sys.exit(f"--laser-path: expected 'start X0 Y0 Z0 end X1 Y1 ...' (use --path-file for files): {line}")
    x0, y0, z0, x1, y1 = (float(v) for v in t[1:4] + t[5:7])
    S, N = 0.0, 1
    if "hatch" in t:
        i = t.index("hatch")
        if t[i + 1] == "card":
            sys.exit("--laser-path: 'hatch card' cannot be resolved here; give the spacing")
        S, N = float(t[i + 1]), int(t[i + 2])
    nrm = np.array([-(y1 - y0), x1 - x0]) / np.hypot(x1 - x0, y1 - y0)
    off = np.arange(N)[:, None] * S * nrm
    seg = np.hstack([off + (x0, y0), off + (x1, y1)])
    return z0, seg, (S / 2 if N > 1 else None)


def read_waypoints(path):
    rows = []
    for ln in open(path):
        c = ln.split("#")[0].replace(",", " ").split()
        if len(c) >= 4:
            v = [float(x) for x in c]
            rows.append((v[0], v[1], v[2], v[3], v[4] if len(v) > 4 else None))
    return rows


def waypoint_layers(rows, scale, shift):
    """Waypoint rows (x, y, z, p, v) -> {z (m): segments[n,4] (m)} of the emitting moves."""
    out, prev = {}, None
    for x, y, z, p, _ in rows:
        q = np.array([x, y, z]) * scale + shift
        if prev is not None and p > 0 and q[2] == prev[2]:   # a z change starts a layer, laser off
            out.setdefault(float(q[2]), []).append([prev[0], prev[1], q[0], q[1]])
        prev = q
    return {z: np.array(v) for z, v in out.items()}


def thermal_layers(path, tl, box):
    """{z (m): melted node xy[n,2]} per layer group; box = (xlo, xhi, ylo, yhi) in m."""
    import h5py
    out = {}
    with h5py.File(path, "r") as f:
        for name in sorted((k for k in f if k.isdigit()), key=int):
            g = f[name]
            c = g["nodeCoords"][:]
            keep = (c[:, 0] >= box[0]) & (c[:, 0] <= box[1]) & (c[:, 1] >= box[2]) & (c[:, 1] <= box[3])
            T, tmax = g["temperatures"], np.full(len(c), -np.inf)
            csr = "nodeOffsets" in g
            off = g["nodeOffsets"][:] if csr else None
            B = 1 << 18
            for a in range(0, len(c), B):      # ponytail: reads every series in the box; per-node reads if it gets slow
                b = min(a + B, len(c))
                if not keep[a:b].any():
                    continue
                if csr:
                    cnt = np.diff(off[a:b + 1])
                    nz = cnt > 0
                    if nz.any():
                        v = T[off[a]:off[b]]
                        tmax[a:b][nz] = np.maximum.reduceat(v, (off[a:b] - off[a])[nz])
                else:
                    tmax[a:b] = T[a:b].max(1)
            m = keep & (tmax >= tl)
            if m.any():
                out[float(c[m, 2].max())] = c[m, :2]
            print(f"thermal        {os.path.basename(path)} layer {name}: {int(m.sum())} melted nodes in the box")
    return out


def footprint(layers, dims, lo, dx, pad, union):
    """layers {z (m): [(segments[n,4] m, radius m)]} -> fp: signed depth (mm) maps per layer plane."""
    from scipy import ndimage
    nx, ny = int(dims[0]) + 2 * pad, int(dims[1]) + 2 * pad
    ox, oy = lo[0] - pad, lo[1] - pad          # lattice index of map pixel (0,0)

    def raster(items):
        F = np.zeros((ny, nx), bool)
        for seg, r in items:
            p0 = seg[:, :2] / dx - (ox, oy)
            p1 = seg[:, 2:] / dx - (ox, oy)
            n = np.maximum(1, np.ceil(2 * np.hypot(*(p1 - p0).T)).astype(int))   # a point every <= 0.5 px
            k = np.repeat(np.arange(len(n)), n + 1)
            t = (np.arange(len(k)) - np.repeat(np.cumsum(n + 1) - (n + 1), n + 1)) / np.repeat(n, n + 1)
            ij = np.rint(p0[k] + t[:, None] * (p1 - p0)[k]).astype(int)
            ok = (ij[:, 0] >= 0) & (ij[:, 0] < nx) & (ij[:, 1] >= 0) & (ij[:, 1] < ny)
            lines = np.zeros((ny, nx), bool)
            lines[ij[ok, 1], ij[ok, 0]] = True
            if lines.any():
                F |= ndimage.distance_transform_edt(~lines) <= r / dx
        return ndimage.binary_fill_holes(F)

    def depth(F):
        if F.all():
            return np.full(F.shape, np.inf, np.float32)
        if not F.any():
            return np.full(F.shape, -np.inf, np.float32)
        return ((ndimage.distance_transform_edt(F) - ndimage.distance_transform_edt(~F)) * dx * 1e3).astype(np.float32)

    planes = sorted(layers)
    Fs = [raster(layers[z]) for z in planes]
    if union:
        Fs, planes = [np.logical_or.reduce(Fs)], [planes[-1]]
    maps, which, seen = [], [], {}
    for F in Fs:                               # identical layers share one map
        key = F.tobytes()
        if key not in seen:
            seen[key] = len(maps)
            maps.append(depth(F))
        which.append(seen[key])
    print(f"footprint      {len(planes)} layer plane(s), {len(maps)} distinct footprint(s), "
          f"part area {[round(float((m >= 0).sum()) * (dx * 1e3) ** 2, 3) for m in maps]} mm^2")
    return dict(planes=np.array(planes) / dx, maps=maps, which=np.array(which), ox=ox, oy=oy)


def depth_at(fp, P):
    """Signed depth (mm) below the part edge at absolute lattice points P[n,3]; -inf above the top layer."""
    li = np.searchsorted(fp["planes"], P[:, 2] - 1e-6)
    out = np.full(len(P), -np.inf)
    ny, nx = fp["maps"][0].shape
    ix = np.clip(np.rint(P[:, 0] - fp["ox"]).astype(int), 0, nx - 1)
    iy = np.clip(np.rint(P[:, 1] - fp["oy"]).astype(int), 0, ny - 1)
    for l in np.unique(li[li < len(fp["planes"])]):
        sel = li == l
        out[sel] = fp["maps"][fp["which"][l]][iy[sel], ix[sel]]
    return out


def cluster(defects, vols, s, dims, lo, dx_mm, tile_mm, bulk_mm, rng, fp):
    """Tile bootstrap of the CT population -> placements in absolute lattice units."""
    cen = np.array([V.mean(0) for _, V, _ in defects])
    c0, R, zlo, H = s["center"], s["R"], s["zmin"], s["H"]
    Rfit = s.get("R_fit", R)                   # extent of the CT data; R = the part surface
    Ru = Rfit if R <= Rfit else Rfit - bulk_mm  # CT edge is not the surface: skip its band
    T = max(1, int(round(tile_mm / dx_mm)))
    half = T * dx_mm / 2
    hd = half * np.sqrt(2)                     # tile half-diagonal in xy
    r_int = Rfit - bulk_mm - hd
    if r_int <= 0 or H < 2 * half:
        sys.exit(f"cluster: tile {2 * half:.3f} mm does not fit in the CT bulk (R {R:.3f}, H {H:.3f} mm)")
    lo, hi = np.asarray(lo), np.asarray(lo) + dims
    pl, ntile = [], {"interior": 0, "surface": 0, "outside": 0, "uncovered": 0}
    for tz in range(lo[2], hi[2], T):
        for ty in range(lo[1], hi[1], T):
            for tx in range(lo[0], hi[0], T):
                tlo = np.array([tx, ty, tz])
                thi = np.minimum(tlo + T, hi)
                c = (tlo + thi - 1) / 2.0
                d = np.inf
                if fp is not None:             # depth where the tile meets the part (its floor if the centre is above the top)
                    d = depth_at(fp, c[None])[0]
                    if d == -np.inf:
                        d = depth_at(fp, np.array([[c[0], c[1], tlo[2]]]))[0]
                th = rng.random() * 2 * np.pi
                if -hd < d < bulk_mm + hd and R > Ru and R - d + hd > Ru:
                    ntile["uncovered"] += 1            # shallower than the CT reaches: bulk
                    d = np.inf
                if d >= bulk_mm + hd:
                    rr = r_int * np.sqrt(rng.random())
                    p = c0[:2] + rr * np.array([np.cos(th), np.sin(th)])
                    phi = rng.random() * 2 * np.pi
                    ntile["interior"] += 1
                elif d > -hd:
                    p = c0[:2] + (R - d) * np.array([np.cos(th), np.sin(th)])
                    e = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0]]) + c
                    g = depth_at(fp, e)
                    n = np.array([g[0] - g[1], g[2] - g[3]])
                    a = np.arctan2(n[1], n[0]) if np.all(np.isfinite(n)) and np.hypot(*n) > 0 else rng.random() * 2 * np.pi
                    phi = a - (th + np.pi)             # CT inward normal (-u) -> footprint inward normal
                    ntile["surface"] += 1
                else:
                    ntile["outside"] += 1
                    continue
                pz = zlo + half + rng.random() * (H - 2 * half)
                ref = np.array([p[0], p[1], pz])
                cs, sn = np.cos(phi), np.sin(phi)
                Rz = np.array([[cs, -sn, 0], [sn, cs, 0], [0, 0, 1]])
                m = ((cen - ref) @ Rz.T) / dx_mm + c
                sel = np.all(m >= tlo - 0.5, 1) & np.all(m < thi - 0.5, 1)
                idx = np.nonzero(sel)[0]
                if fp is not None and len(idx):
                    idx = idx[depth_at(fp, m[idx]) >= 0]
                q = np.array([np.cos(phi / 2), 0, 0, np.sin(phi / 2)])
                for k in idx:
                    name, V, F = defects[k]
                    pl.append((name, ((V - ref) @ Rz.T) / dx_mm + c, F, q, vols[k]))
    print(f"tiles          {T}^3 sites ({2 * half:.3f} mm): {ntile['interior']} interior, "
          f"{ntile['surface']} at the surface, {ntile['outside']} outside the part")
    if ntile["uncovered"]:
        print(f"               {ntile['uncovered']} surface tiles were shallower than the CT data "
              f"reaches (--ct-radius {R:.3f} mm vs data {Rfit:.3f} mm): drawn from bulk")
    return pl


def sim_profile(fp, pl, dims, lo, dx_mm):
    """Placed-defect density by depth below the footprint edge, for comparison with the CT."""
    cen = np.array([V.mean(0) for _, V, _, _, _ in pl]) if pl else np.zeros((0, 3))
    d = depth_at(fp, cen)
    ys, xs = np.mgrid[lo[1]:lo[1] + dims[1], lo[0]:lo[0] + dims[0]]
    vol = np.zeros(len(DEPTH_BINS) - 1)
    for z in range(lo[2], lo[2] + dims[2]):
        P = np.stack([xs.ravel(), ys.ravel(), np.full(xs.size, z)], 1)
        vol += np.histogram(depth_at(fp, P), DEPTH_BINS)[0]
    cnt = np.histogram(d, DEPTH_BINS)[0]
    vol = vol * dx_mm ** 3
    return ", ".join(f"{a:.1f}-{b:.1f} mm " + (f"{n / v:.1f}" if v > 0 else "-")
                     for a, b, n, v in zip(DEPTH_BINS[:-1], DEPTH_BINS[1:], cnt, vol)) + " /mm^3"


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

    # cluster: a synthetic CT cylinder (R 1.4 mm, 4 mm tall) whose defects all sit
    # 0.2 mm below its surface. Footprint: 21 hatch lines 50 um apart over
    # x, y in [0, 1] mm -> part [-0.025, 1.025] mm. Every copied defect must land
    # ~0.2 mm inside the footprint edge (curvature + pixels bound the error).
    rng = np.random.default_rng(5)
    th, zz = rng.random(4000) * 2 * np.pi, rng.random(4000) * 4.0
    ct = [(f"d{i}", octa(np.array([1.2 * np.cos(t), 1.2 * np.sin(t), z]), 0.004)[0],
           octa(np.zeros(3), 1)[1]) for i, (t, z) in enumerate(zip(th, zz))]
    st = dict(center=np.zeros(3), R=1.4, zmin=0.0, H=4.0)
    vols = [mesh_volume(V, F) for _, V, F in ct]
    dx, dims, lo = 10e-6, np.array([140, 140, 40]), np.array([-20, -20, 0])
    z0, seg, r = parse_laser_path("laser_path start 0 0 4e-4 end 1e-3 0 speed 1 hatch 50e-6 21")
    assert r == 25e-6 and seg.shape == (21, 4) and np.allclose(seg[-1], [0, 1e-3, 1e-3, 1e-3])
    fp = footprint({z0: [(seg, r)]}, dims, lo, dx, 60, False)
    d = depth_at(fp, np.array([[50.0, 50, 10], [-2.5, 50, 10], [50, 50, 41]]))
    assert abs(d[0] - 0.525) < 0.011 and abs(d[1]) < 0.011 and d[2] == -np.inf, d
    pl = cluster(ct, vols, st, dims, lo, dx * 1e3, 0.2, 0.35, rng, fp)
    P = np.array([V.mean(0) for _, V, _, _, _ in pl])
    dd = depth_at(fp, P)
    # 2nd-nearest edge: corners (two surfaces, one normal per tile) are the known limit
    e2 = np.sort(np.stack([P[:, 0] + 2.5, 102.5 - P[:, 0], P[:, 1] + 2.5, 102.5 - P[:, 1]], 1), 1)[:, 1] * 0.01
    ok = np.abs(dd - 0.2) < 0.06
    assert len(pl) > 100 and ok[e2 > 0.25].all() and ok.mean() > 0.85, (len(pl), ok.mean())
    assert cluster(ct, vols, st, dims, lo, dx * 1e3, 0.2, 0.35, rng, None) == []   # CT bulk is empty
    # true surface (--ct-radius) beyond the CT data: its 0.2 mm band is the scan edge,
    # never copied to the part surface; every tile falls back to (empty) bulk
    assert cluster(ct, vols, dict(st, R=3.0, R_fit=1.4), dims, lo, dx * 1e3, 0.2, 0.35, rng, fp) == []
    print("selftest ok")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="mode", required=True)
    sub.add_parser("selftest")
    p = sub.add_parser("info")
    p.add_argument("glb")
    for name in ("crop", "sample", "cluster"):
        p = sub.add_parser(name)
        p.add_argument("glb")
        p.add_argument("--dx", type=float, required=True, help="lattice spacing in m (the app_style dx)")
        p.add_argument("--dims", type=int, nargs=3, required=True, metavar=("NX", "NY", "NZ"),
                       help="global lattice extent of the whole build")
        p.add_argument("--lo", type=int, nargs=3, default=[0, 0, 0], metavar=("I", "J", "K"),
                       help="absolute lattice index of the low corner (region block lower bounds)")
        p.add_argument("--out", required=True, help="mask .h5 to write")
        p.add_argument("--list", help="CSV of placed defects")
        if name == "crop":
            p.add_argument("--origin-mm", type=float, nargs=3, required=True, metavar=("X", "Y", "Z"),
                           help="CT point (mm) that maps to lattice site (0,0,0)")
        else:
            p.add_argument("--seed", type=int, required=True)
        if name == "sample":
            p.add_argument("--density", type=float, help="defects per mm^3 (default: CT mean)")
        if name == "cluster":
            p.add_argument("--tile", type=float, default=4e-4, help="tile edge (m), default 0.4 mm")
            p.add_argument("--bulk-depth", type=float, default=3.5e-4, help="CT depth (m) taken as bulk")
            p.add_argument("--laser-path", action="append", default=[], help="deck laser_path start/end line")
            p.add_argument("--path-file", action="append", default=[], help="laser_path waypoint file")
            p.add_argument("--gcode", action="append", default=[], help="G-code file")
            p.add_argument("--path-scale", type=float, default=1.0, help="file coordinate scale")
            p.add_argument("--path-shift", type=float, nargs=3, default=[0.0, 0.0, 0.0], help="offset (m) after scale")
            p.add_argument("--thermal-h5", action="append", default=[], help="hdf5_unstructured/csr thermal file")
            p.add_argument("--liquidus", type=float, help="K; a node is part of the footprint if it reaches it")
            p.add_argument("--track-radius", type=float, help="m; footprint = within this of a scan segment")
            p.add_argument("--union", action="store_true", help="one footprint for all layers")
            p.add_argument("--ct-radius", type=float,
                           help="mm; radius of the CT part's as-built surface about the fitted axis, when the "
                                "scan does not reach it (default: the CT data's own extent)")
    a = ap.parse_args()

    if a.mode == "selftest":
        return selftest()
    defects = read_glb(a.glb)
    s = info(defects)
    if a.mode == "info":
        return
    dx_mm = a.dx * 1e3
    dims, lo = np.array(a.dims), np.array(a.lo)
    vols = [mesh_volume(V, F) for _, V, F in defects]
    if a.mode == "crop":
        o = np.array(a.origin_mm)
        pl = [(n, (V - o) / dx_mm, F, np.array([1.0, 0, 0, 0]), v)
              for (n, V, F), v in zip(defects, vols)]
    elif a.mode == "sample":
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
            c = lo + rng.random(3) * dims - 0.5
            pl.append((name, (V - V.mean(0)) @ R.T / dx_mm + c, F, q, vols[k]))
    else:
        if a.ct_radius:
            s["R_fit"], s["R"] = s["R"], a.ct_radius
            print("depth profile  from --ct-radius: " + ct_profile(s))
        fp = cluster_footprint(a, dims, lo)
        pl = cluster(defects, vols, s, dims, lo, dx_mm, a.tile * 1e3, a.bulk_depth * 1e3,
                     np.random.default_rng(a.seed), fp)
        if fp is not None:
            print("depth profile  placed: " + sim_profile(fp, pl, dims, lo, dx_mm))
            print("               CT:     " + ct_profile(s))
    place(pl, dims, lo, dx_mm, a.out, a.list)


def cluster_footprint(a, dims, lo):
    """Merge every footprint source into {z (m): [(segments, radius)]} -> fp, or None (interior)."""
    layers = {}
    def add(z, seg, r):
        if r is None:
            sys.exit("cluster: give --track-radius (half the melt-track width) for this path source")
        layers.setdefault(round(z, 12), []).append((seg, r))
    for line in a.laser_path:
        z, seg, r = parse_laser_path(line)
        add(z, seg, a.track_radius or r)
    rows = [read_waypoints(f) for f in a.path_file]
    if a.gcode:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from laser_path_convert import from_gcode
        rows += [from_gcode(open(f)) for f in a.gcode]
    for r in rows:
        for z, seg in waypoint_layers(r, a.path_scale, np.array(a.path_shift)).items():
            add(z, seg, a.track_radius)
    if a.thermal_h5:
        if a.liquidus is None:
            sys.exit("cluster: --thermal-h5 needs --liquidus")
        from scipy.spatial import cKDTree
        m = (a.tile + a.bulk_depth) * 2
        box = ((lo[0]) * a.dx - m, (lo[0] + dims[0]) * a.dx + m, lo[1] * a.dx - m, (lo[1] + dims[1]) * a.dx + m)
        for f in a.thermal_h5:
            for z, xy in thermal_layers(f, a.liquidus, box).items():
                u = np.unique(xy, axis=0)
                r = a.track_radius or float(np.median(cKDTree(u).query(u, 2)[0][:, 1]))
                add(z, np.hstack([xy, xy]), r)
    if not layers:
        print("footprint      none: fully interior")
        return None
    pad = int(np.ceil((a.tile + a.bulk_depth) * 2 / a.dx))
    return footprint(layers, dims, lo, a.dx, pad, a.union)


if __name__ == "__main__":
    main()
