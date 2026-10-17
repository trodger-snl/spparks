#!/usr/bin/env python3
"""
Build a tiny hdf5_csr thermal file that reaches the zero-chunk path in
HDF5UnstructuredTemperatureSource::load_layer().

Three layers over the same SPPARKS box:
  layer 0 - mesh covers the box            -> chunks load, T is hot
  layer 1 - mesh translated 0.05 m away    -> ZERO overlapping chunks
  layer 2 - mesh covers the box again      -> chunks load, T is hot

Layer 0's time series deliberately carries a COOLDOWN TAIL that extends
through layer 1's window. Real ct15 layers abut without overlapping, which
makes a stale layer degrade silently to default_temp; a tail is physically
ordinary and is what turns the stale mesh into visibly wrong temperatures.

Correct behaviour during layer 1: every site reads default_temp (300 K),
because this box has no thermal data for that layer.
"""
import numpy as np, h5py, sys

OUT = sys.argv[1] if len(sys.argv) > 1 else "stale_layer_test.hdf5"
DX = 5e-6
LAYER_TIMES = np.array([0.0, 1e-3, 2e-3, 3e-3])   # 3 layers -> 4 entries
FAR = 0.05                                         # layer 1 mesh offset (m)

# Structured hex grid -> 6 tets per cell (Freudenthal). Spans the SPPARKS box
# (0..2.05e-4 in x/y, 0..5.5e-5 in z) with margin on every side.
NCX, NCY, NCZ = 6, 6, 4
X0, X1 = -4.0e-4, 6.0e-4
Y0, Y1 = -4.0e-4, 6.0e-4
Z0, Z1 = -2.0e-4, 4.0e-4
NSAMP = 5          # time samples per node

def build_mesh(xshift):
    xs = np.linspace(X0, X1, NCX + 1) + xshift
    ys = np.linspace(Y0, Y1, NCY + 1)
    zs = np.linspace(Z0, Z1, NCZ + 1)
    nx, ny, nz = len(xs), len(ys), len(zs)
    nid = lambda i, j, k: (k * ny + j) * nx + i
    coords = np.empty((nx * ny * nz, 3))
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                coords[nid(i, j, k)] = (xs[i], ys[j], zs[k])
    # 6 tets per hex, all sharing the 0-6 diagonal -> conforming, no gaps
    TETS = [(0,1,3,7),(0,1,7,5),(0,5,7,4),(0,3,2,7),(0,6,4,7),(0,2,6,7)]
    elems = []
    for k in range(nz - 1):
        for j in range(ny - 1):
            for i in range(nx - 1):
                c = [nid(i,j,k),     nid(i+1,j,k),   nid(i,j+1,k),   nid(i+1,j+1,k),
                     nid(i,j,k+1),   nid(i+1,j,k+1), nid(i,j+1,k+1), nid(i+1,j+1,k+1)]
                for t in TETS:
                    elems.append([c[t[0]], c[t[1]], c[t[2]], c[t[3]]])
    return coords, np.array(elems, dtype=np.uint32)

def series(coords, t_lo, t_hi):
    """Per-node (times, temps): a hot band that sweeps in z, 300 K at the ends."""
    n = len(coords)
    ts = np.empty((n, NSAMP)); Ts = np.empty((n, NSAMP))
    tt = np.linspace(t_lo, t_hi, NSAMP)
    for i, (x, y, z) in enumerate(coords):
        ts[i] = tt
        # deterministic, spatially varying, comfortably above 300
        Ts[i] = 300.0 + 900.0 * np.exp(-((z - 2.0e-5) / 1.5e-4) ** 2) \
                      * (0.5 + 0.5 * np.cos(2 * np.pi * (tt - t_lo) / (t_hi - t_lo))) \
                      + 50.0 * (i % 7)
    return ts, Ts

with h5py.File(OUT, "w") as f:
    f.create_dataset("layerTimes", data=LAYER_TIMES, dtype="f8")
    for L in range(3):
        shift = FAR if L == 1 else 0.0
        coords, elems = build_mesh(shift)
        nnode, nelem = len(coords), len(elems)

        # Layer 0 gets a cooldown tail through layer 1's window; others abut.
        t_lo = LAYER_TIMES[L]
        t_hi = LAYER_TIMES[L + 2] if L == 0 else LAYER_TIMES[L + 1]
        ts, Ts = series(coords, t_lo, t_hi)

        g = f.create_group(str(L))
        bbox = np.array([[coords[:,0].min(), coords[:,1].min(), coords[:,2].min(),
                          coords[:,0].max(), coords[:,1].max(), coords[:,2].max()]])
        g.create_dataset("boundingBoxes", data=bbox, dtype="f8")
        g.create_dataset("elemPtrs",  data=np.array([0, nelem], dtype=np.uint32))
        g.create_dataset("nodePtrs",  data=np.array([0, nnode], dtype=np.uint32))
        g.create_dataset("elementToNode", data=elems, dtype=np.uint32)   # chunk-local ids
        g.create_dataset("nodeCoords", data=coords, dtype="f8")
        counts = np.full(nnode, NSAMP, dtype=np.uint32)
        g.create_dataset("dataCounts", data=counts)
        g.create_dataset("nodeOffsets",
                         data=np.concatenate([[0], np.cumsum(counts)]).astype(np.int64))
        g.create_dataset("times",        data=ts.ravel(), dtype="f8")
        g.create_dataset("temperatures", data=Ts.ravel(), dtype="f8")
        print(f"layer {L}: {nnode} nodes, {nelem} tets, x[{bbox[0,0]:.5f},{bbox[0,3]:.5f}] "
              f"t[{t_lo:.4g},{t_hi:.4g}]{'  <-- FAR AWAY' if L == 1 else ''}")
print("wrote", OUT)
