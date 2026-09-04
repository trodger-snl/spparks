#!/bin/bash
# Box-per-layer driver for the GP700 Moser build.
# usage: run_layers.sh <spparks-binary> <outdir> [nlayers=12] [seed=12345]
# Detach it (setsid nohup ... &) for a full build; touch <outdir>/STOP to end it
# cleanly after the current box. Completed boxes (completion.marker) are skipped.
# Box n (0-based) = layer n+1: laser plane at z_top = 20 + 10*(n+1) voxels (60 um
# substrate + 30 um layers at dx = 3 um), box spans [max(0, z_top-60), z_top].
# Even layers hatch along x (5 lines, y = 40..360 um), odd along y, serpentine.
# Each box is a fresh Moser history (T0 = preheat, the recoat interval); the
# microstructure continues through the stitch file.
set -eu
BIN=$(readlink -f "$1"); OUT=$(readlink -f "$2"); NL=${3:-12}; SEED=${4:-12345}
HERE=$(cd "$(dirname "$0")" && pwd); TPL=$HERE/in.gp700_layer
mkdir -p "$OUT"; ST=$OUT/gp700.st
DX=3e-6; TBOX=5.0e-3; TGAP=10.0
for ((n=0; n<NL; n++)); do
  D=$OUT/box_$(printf %03d $n); mkdir -p "$D"
  [ -f "$D/completion.marker" ] && { echo "box $n done, skipping"; continue; }
  ztop=$((20 + 10*(n+1))); zlo=$((ztop-60)); [ $zlo -lt 0 ] && zlo=0
  z0=$(python3 -c "print($ztop*$DX)")
  if [ $((n%2)) -eq 0 ]; then
    LASER="laser_path start -100e-6 40e-6 $z0 end 502e-6 40e-6 speed 1.137 hatch 80e-6 5 serpentine pause 1e-4"
  else
    LASER="laser_path start 360e-6 -100e-6 $z0 end 360e-6 502e-6 speed 1.137 hatch 80e-6 5 serpentine pause 1e-4"
  fi
  T0=$(python3 -c "print($n*$TGAP)"); T1=$(python3 -c "print($n*$TGAP + $TBOX)")
  if [ $n -eq 0 ]; then
    INIT="set site range 1 1000000
set i2 value 1
set d6 value 473"
  else
    INIT="set i1 stitch $ST last
set i2 stitch $ST last
set d1 stitch $ST last
set d2 stitch $ST last
set d3 stitch $ST last
set d4 stitch $ST last
set d8 stitch $ST last
set d9 stitch $ST last
set i1 range 1 1000000 if i2 < 1
set i2 value 1 if i2 < 2
set d6 value 473
set d8 value 0.0 if i2 != 3
set d9 value 0.0 if i2 != 3"
  fi
  python3 - "$TPL" "$D/in.box" "$((SEED+n))" "$zlo" "$ztop" "$INIT" "$LASER" "$T0" "$T1" "$ST" <<'PY'
import sys
tpl,out,seed,zlo,zhi,init,laser,t0,t1,st = sys.argv[1:]
s = open(tpl).read()
for k,v in {'__SEED__':seed,'__ZLO__':zlo,'__ZHI__':zhi,'__INIT_OR_STITCH__':init,
            '__LASER__':laser,'__T0__':t0,'__T1__':t1,'__STITCH__':st}.items():
    s = s.replace(k, v)
open(out,'w').write(s)
PY
  [ -f "$OUT/STOP" ] && { echo "STOP file present, stopping before box $n"; exit 0; }
  echo "box $n: z [$zlo,$ztop] t [$T0,$T1]  $(date +%T)"
  # Never kill a running box (HIP kernels mid-flight wedge the GPU): to stop
  # the build, touch $OUT/STOP and let the current box finish.
  ( cd "$D" && mpirun -np 1 --bind-to none "$BIN" < in.box > log.spparks 2>&1 && grep -q 'Loop time' log.spparks && touch completion.marker ) || { echo "box $n FAILED"; exit 1; }
done
echo "all $NL boxes done $(date +%T)"
