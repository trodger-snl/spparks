#!/usr/bin/env python3
"""Originating-grain census: for each run, count the filtered grains (height >= 60 um, ESD >= 40 um)
whose lowest solid voxel lies more than 60 um above the first remelt floor (Lamb's originating
class in the sense of 'started inside the volume') versus those continuing from the floor.
usage (WORK env): origin_census.py [--crop-frac 0.25] run1/gp700.st run2/gp700.st ...
"""
import argparse, sys, os, re, json
import numpy as np
from spparks_micro_analysis.stitch_reader import StitchReader
from spparks_micro_analysis.grain_size_analysis import relabel_grains
ap = argparse.ArgumentParser(); ap.add_argument('stitch', nargs='+'); ap.add_argument('--dx', type=float, default=3e-6)
ap.add_argument('--crop-frac', type=float, default=0.25, help='crop this fraction off each lateral side')
ap.add_argument('--segment-deg', type=float, default=None, help='re-segment at this disorientation threshold instead of spin labels')
a = ap.parse_args(); dxu = a.dx*1e6
for st in a.stitch:
    with StitchReader(st) as r:
        b = r.get_global_bounds(); t = r.times[-1]
        i1 = r.read_field_at_time('i1', t, bounds=b)[0]; i2 = r.read_field_at_time('i2', t, bounds=b)[0]
        q = np.stack([r.read_field_at_time(k, t, bounds=b)[0] for k in ('d1','d2','d3','d4')], -1) if a.segment_deg else None
    n = i1.shape[0]; c0, c1 = int(n*a.crop_frac), int(n*(1-a.crop_frac)); i1 = i1[c0:c1, c0:c1]; i2 = i2[c0:c1, c0:c1]
    solid = i2 == 3
    if a.segment_deg:
        from spparks_micro_analysis.grain_segmentation import segment_grains
        lab = np.where(solid, segment_grains(q[c0:c1, c0:c1], occupied=solid, threshold_deg=a.segment_deg, crystal_structure='cubic')[0], 0).astype(np.int64)
    else:
        lab = relabel_grains(np.where(solid, i1, 0).astype(np.int64))
    zfloor = np.nonzero(solid.any((0,1)))[0].min()
    ids, inv, vol = np.unique(lab[solid], return_inverse=True, return_counts=True); Z = np.indices(solid.shape)[2][solid]
    zmin = np.full(len(ids), 10**9); np.minimum.at(zmin, inv, Z); zmax = np.zeros(len(ids), int); np.maximum.at(zmax, inv, Z)
    h = (zmax-zmin+1)*dxu; esd = (6*vol*dxu**3/np.pi)**(1/3); ok = (ids > 0) & (h >= 60) & (esd >= 40)
    orig = ok & ((zmin - zfloor)*dxu > 60); cont = ok & ~orig
    deck = os.path.join(os.path.dirname(st), 'box_000', 'in.box'); nucl = tc = None
    if os.path.exists(deck):
        d = open(deck).read(); m = re.search(r'^nucleation_density\s+(\S+)', d, re.M); nucl = float(m.group(1)) if m else None
        m = re.search(r'^critical_undercooling\s+(\S+)', d, re.M); tc = float(m.group(1)) if m else None
    print(f'{st}: nucl {nucl:.2g} tc {tc}: filtered {ok.sum()}, originating >60 um above the floor {orig.sum()} ({100*vol[orig].sum()/max(1,vol[ok].sum()):.1f}% of filtered volume), continuations {cont.sum()}; total labels {len(ids)}')
