#!/usr/bin/env python3
"""Check that a staged build through 'dump hdf5' matches the same build
through 'dump stitch', site for site.

The two files hold the same information in very different shapes: the stitch
file is a history of overlapping blocks that must be unioned newest-last to
recover the final state, while the hdf5 file already *is* the final state.
Reconstructing the stitch side here is what makes the comparison meaningful --
and it is a compact statement of what the hdf5 style makes unnecessary.

Requires h5py.  sqlite3 is in the standard library, so no stitch build is
needed to read the .st file.
"""

import os
import sqlite3
import sys

import h5py
import numpy as np

HDF5_FILE = "hdf5_staged.h5"
STITCH_FILE = "stitch_staged.st"

# stitch fields.type: 1 = int32, 3 = float64
STITCH_TYPES = {1: np.int32, 3: np.float64}
STITCH_FILL = {1: -1, 3: 0.0}


def stitch_final_state(path, field):
    """Union every block at or before the last time, oldest first."""
    db = sqlite3.connect("file:" + path + "?mode=ro", uri=True)
    abs_tol, last_time = db.execute(
        "select absolute_tolerance, last_time from globals").fetchone()
    row = db.execute("select field_id, type from fields where field = ?",
                     (field,)).fetchone()
    if row is None:
        raise SystemExit("field %r not in %s" % (field, path))
    field_id, ftype = row
    dtype = STITCH_TYPES[ftype]

    x1, x2, y1, y2, z1, z2 = db.execute(
        "select min(x_min), max(x_max), min(y_min), max(y_max), "
        "min(z_min), max(z_max) from blocks where field_id = ?",
        (field_id,)).fetchone()
    out = np.full((z2 - z1, y2 - y1, x2 - x1), STITCH_FILL[ftype], dtype=dtype)

    eligible = {r[0] for r in db.execute(
        "select timestamp from times where time < ? + ?", (abs_tol, last_time))}

    for (ts, bx1, by1, bz1, bx2, by2, bz2, blob) in db.execute(
            "select timestamp, x_min, y_min, z_min, x_max, y_max, z_max, state "
            "from blocks where field_id = ? order by timestamp, rowid",
            (field_id,)):
        if ts not in eligible:
            continue
        n = (bx2 - bx1) * (by2 - by1) * (bz2 - bz1)
        block = np.frombuffer(blob, dtype=dtype, count=n).reshape(
            bz2 - bz1, by2 - by1, bx2 - bx1)
        out[bz1 - z1:bz2 - z1, by1 - y1:by2 - y1, bx1 - x1:bx2 - x1] = block

    return out, (x1, x2, y1, y2, z1, z2)


def main():
    for path in (HDF5_FILE, STITCH_FILE):
        if not os.path.exists(path):
            raise SystemExit("%s not found; run run.sh first" % path)

    failures = 0
    with h5py.File(HDF5_FILE, "r") as f:
        origin = list(f.attrs["origin"])
        dims = list(f.attrs["dims"])
        print("hdf5   origin %s dims %s time %g" %
              (origin, dims, float(f.attrs["time"])))

        for name in sorted(f.keys()):
            got = f[name][:]
            want, bounds = stitch_final_state(STITCH_FILE, name)
            same = got.shape == want.shape and np.array_equal(got, want)
            failures += 0 if same else 1
            print("  %-5s %-16s %s" % (
                name, str(got.shape),
                "MATCH" if same else
                "DIFFER (%d cells)" % int((got != want).sum())))

    hs, ss = os.path.getsize(HDF5_FILE), os.path.getsize(STITCH_FILE)
    print("\nfile size: hdf5 %.0f KB, stitch %.0f KB (%.1fx)" %
          (hs / 1024, ss / 1024, ss / float(hs)))

    if failures:
        print("\n%d field(s) DIFFER" % failures)
        return 1
    print("\nall fields match")
    return 0


if __name__ == "__main__":
    sys.exit(main())
