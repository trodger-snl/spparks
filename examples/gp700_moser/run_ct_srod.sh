#!/bin/bash
# SROD analysis of the IN625 calibration builds on the machine that holds them (Attaway/tscratch).
# usage: run_ct_srod.sh OUT_DIR [CT_ROOT=/tscratch/trodger]   (needs spparks_micro_analysis importable, e.g. conda env SURGE)
# Each CT run's stitch file is <CT_ROOT>/<run>*/stitch_files/am_data.st (spparks_slurm_driver.py layout), 5 um voxels.
# Sample mapping from HPC-Calibration/"EBSD to Simulation Mapping.txt" (higher-numbered runs preferred).
set -u
OUT=$1; ROOT=${2:-/tscratch/trodger}; HERE=$(cd "$(dirname "$0")" && pwd)
declare -A SAMPLE=( [ct12]=P04 [ct6]=P04 [ct11]=P13 [ct5]=P13 [ct3]=P13 [ct16]=P15 [ct15]=P15 [ct13]=P18 [ct9]=P18 [ct1]=P18 [ct14]=P20 [ct10]=P20 [ct2]=P20 )
mkdir -p "$OUT"
for run in "${!SAMPLE[@]}"; do
  for st in "$ROOT"/${run}-*/stitch_files/am_data.st "$ROOT"/${run}/stitch_files/am_data.st; do
    [ -f "$st" ] || continue
    label=${SAMPLE[$run]}_${run}
    echo "== $label  $st"
    python "$HERE/srod_from_stitch.py" "$st" "$OUT" "$label" --dx 5 --window-um 1200 --floor-um 300 --sections 3 > "$OUT/$label.log" 2>&1 || echo "   FAILED (see $OUT/$label.log)"
    tail -n +2 "$OUT/$label.log" | grep "3-D\|per grain" | head -4
  done
done
echo "then: python $HERE/srod_compare.py $OUT <EBSD_JSON_DIR> -o $OUT/compare"
