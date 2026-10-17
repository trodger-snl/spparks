#!/bin/bash
# Run every deck pair with two binaries and diff the d6 sum/max columns.
# usage: compare.sh <spparks-A> [spparks-B]      (default B = A: CPU vs kk only)
# For each case the .cpu deck runs on A and the .kk deck on B; identical
# columns are the pass criterion (the microstructure is not deterministic).
cd "$(dirname "$0")"
A=$1; B=${2:-$1}; fail=0
d6() { awk '/^ *Time +Naccept/{p=1;next} p&&/^ *[0-9]/{print $1,$(NF-1),$NF} p&&/^Loop/{p=0}' "$1"; }
for case in rosenthal moser_pause_below moser_pause; do
  mpirun -np 1 --bind-to none "$A" < in.$case.cpu > log.$case.cpu 2>&1
  mpirun -np 1 --bind-to none "$B" < in.$case.kk  > log.$case.kk  2>&1
  n=$(d6 log.$case.cpu | wc -l)
  if [ "$n" -gt 3 ] && diff <(d6 log.$case.cpu) <(d6 log.$case.kk) >/dev/null; then
    echo "$case: IDENTICAL d6 sum/max over $n rows"
  else echo "$case: DIFFERS"; diff <(d6 log.$case.cpu) <(d6 log.$case.kk) | head -6; fail=1; fi
done
exit $fail
