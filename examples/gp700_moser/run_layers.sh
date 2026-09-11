#!/bin/bash
# Box-per-layer driver for the GP700 Moser build.
# usage: [DX=3e-6] run_layers.sh <spparks-binary> <outdir> [nlayers=12] [seed=12345] [nx_voxels=267] [nhatch=10] [dt=2e-6] [substrate.st]
# substrate.st (make_substrate.py): copied to <outdir>/gp700.st before box 0, which then
# starts from it like a continuation box instead of single-voxel random grains.
# Lateral box = nx_voxels * 3 um square; hatches at 40 um + k*80 um, lines 100 um past
# both edges. Analyse a central crop (gp700_analysis.py --crop) away from the walls,
# where each pool has solid tracks alongside as in the print.
# Detach it (setsid nohup ... &) for a full build; touch <outdir>/STOP to end it
# cleanly after the current box. Completed boxes (completion.marker) are skipped.
# Box n (0-based) = layer n+1: laser plane at z_top = 50 + 10*(n+1) voxels (150 um
# substrate + 30 um layers at dx = 3 um), box spans [max(0, z_top-60), z_top].
# Even layers hatch along x (5 lines, y = 40..360 um), odd along y, serpentine.
# Each box is a fresh Moser history (T0 = preheat, the recoat interval); the
# microstructure continues through the stitch file. Sites absent from it (the
# new layer) carry the stitch sentinel -1 in d1-d4, which the app would
# normalise to ONE cube-aligned orientation for the whole layer; zeroing them
# makes the app draw a random orientation per voxel at init (powder).
set -eu
BIN=$(readlink -f "$1"); OUT=$(readlink -f "$2"); NL=${3:-12}; SEED=${4:-12345}; NX=${5:-267}; NH=${6:-10}; DT=${7:-2e-6}; SUB=${8:-}
HERE=$(cd "$(dirname "$0")" && pwd); TPL=${TPL:-$HERE/in.gp700_layer}   # TPL=... overrides the deck template
# HATCH_OFFSET=random: seeded per-layer hatch shift in [-40,40) um along the hatch normal (layers n and n+2 no
# longer register); HATCH_OFFSET=<metres>: that shift on every second same-direction layer; default 0.
HOFF=${HATCH_OFFSET:-0}
mkdir -p "$OUT"; ST=$OUT/gp700.st
[ -n "$SUB" ] && [ ! -f "$ST" ] && cp "$SUB" "$ST" && echo "substrate $SUB -> $ST"
DX=${DX:-3e-6}; TGAP=10.0   # voxel size from the environment (DX=1.5e-6 run_layers.sh ...)
# geometry in voxels: 150 um substrate, 30 um layers, 180 um box window
VSUB=$(python3 -c "print(round(150e-6/$DX))"); VLAY=$(python3 -c "print(round(30e-6/$DX))"); VWIN=$(python3 -c "print(round(180e-6/$DX))")
# Inter-hatch pause = the real 10 mm track length / speed (8.8 ms): the previous
# track has solidified (~0.75 ms) before the next hatch overlaps it. Box time
# covers 5 hatches + pauses + 2 ms cooling; the predictor skips the idle gaps.
PAUSE=8.8e-3
# box time: NH passes of (NX*dx+200um)/v, (NH-1) pauses, 2 ms cooling
TBOX=$(python3 -c "print(round($NH*($NX*$DX+200e-6)/1.137 + ($NH-1)*$PAUSE + 2e-3, 6))")
for ((n=0; n<NL; n++)); do
  D=$OUT/box_$(printf %03d $n); mkdir -p "$D"
  [ -f "$D/completion.marker" ] && { echo "box $n done, skipping"; continue; }
  # 150 um substrate (50 voxels): five overlapping hatches remelt 80-100 um below
  # the surface, so anything thinner melts through in box 0.
  ztop=$((VSUB + VLAY*(n+1))); zlo=$((ztop-VWIN)); [ $zlo -lt 0 ] && zlo=0
  z0=$(python3 -c "print($ztop*$DX)")
  LEND=$(python3 -c "print($NX*$DX + 100e-6)"); YLAST=$(python3 -c "print(40e-6 + 80e-6*($NH-1))")
  OFF=$(python3 -c "import random; random.seed($SEED*7919+$n); print(random.uniform(-40e-6,40e-6) if '$HOFF'=='random' else float('$HOFF')*(($n//2)%2))")
  Y0=$(python3 -c "print(40e-6 + $OFF)"); X0=$(python3 -c "print($YLAST + $OFF)")
  if [ $((n%2)) -eq 0 ]; then
    LASER="laser_path start -100e-6 $Y0 $z0 end $LEND $Y0 speed 1.137 hatch 80e-6 $NH serpentine pause $PAUSE"
  else
    LASER="laser_path start $X0 -100e-6 $z0 end $X0 $LEND speed 1.137 hatch 80e-6 $NH serpentine pause $PAUSE"
  fi
  T0=$(python3 -c "print($n*$TGAP)"); T1=$(python3 -c "print($n*$TGAP + $TBOX)")
  if [ $n -eq 0 ] && [ ! -f "$ST" ]; then
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
set d10 stitch $ST last
set d1 value 0.0 if i2 < 1
set d2 value 0.0 if i2 < 1
set d3 value 0.0 if i2 < 1
set d4 value 0.0 if i2 < 1
set i1 range 1 1000000 if i2 < 1
set i2 value 1 if i2 < 2
set d6 value 473
set d8 value 0.0 if i2 != 3
set d9 value 0.0 if i2 != 3
set d10 value 0.0 if i2 != 3"
  fi
  python3 - "$TPL" "$D/in.box" "$((SEED+n))" "$zlo" "$ztop" "$INIT" "$LASER" "$T0" "$T1" "$ST" "$NX" "$DT" "$DX" <<'PY'
import sys
tpl,out,seed,zlo,zhi,init,laser,t0,t1,st,nx,dt,dx = sys.argv[1:]
s = open(tpl).read()
for k,v in {'__SEED__':seed,'__ZLO__':zlo,'__ZHI__':zhi,'__INIT_OR_STITCH__':init,
            '__LASER__':laser,'__T0__':t0,'__T1__':t1,'__STITCH__':st,'__NX__':nx,'__DT__':dt,'__DX__':dx}.items():
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
