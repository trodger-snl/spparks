#!/bin/bash
# Staged build through 'dump hdf5', with the same build through 'dump stitch'
# for comparison.  Each stage is a separate SPPARKS invocation that seeds
# itself from the previous stage's file, which is the pattern real AM decks
# use (see examples/am/legacy/stitch/am_demo).
#
# Requires a build configured with --highfive (and --package stitch for the
# comparison half).

set -e

SPPARKS=${SPPARKS:-../../../../../build-cpu/spparks}
NP=${NP:-4}
MPIRUN=${MPIRUN:-"mpirun --bind-to none -np $NP"}

rm -f hdf5_staged.h5 stitch_staged.st

echo "=== staged build via dump hdf5 ==="
for s in 1 2 3; do
  echo "--- stage $s"
  $MPIRUN $SPPARKS -in in.stage$s > log.hdf5.stage$s
done

# Same three stages through stitch, for a side-by-side check.  sed swaps the
# backend so the two runs stay identical in every other respect.
echo
echo "=== same build via dump stitch ==="
for s in 1 2 3; do
  sed -e 's/hdf5 1.0 hdf5_staged.h5/stitch 1.0 stitch_staged.st/' \
      -e 's/site hdf5 hdf5_staged.h5/site stitch stitch_staged.st last/' \
      -e 's/reset_time      hdf5 hdf5_staged.h5/reset_time      stitch stitch_staged.st last/' \
      in.stage$s > in.stitch_stage$s
  echo "--- stage $s"
  $MPIRUN $SPPARKS -in in.stitch_stage$s > log.stitch.stage$s
done

echo
ls -l hdf5_staged.h5 stitch_staged.st
echo
echo "Run 'python3 compare.py' to check the two agree site for site."
