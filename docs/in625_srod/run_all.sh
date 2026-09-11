#!/bin/bash
# all ten longitudinal sections; transforms per HPC-Calibration/scripts/prepare_v8_targets.py (rot90 cases use the
# pre-rotated square-grid exports; vflip handled by the height coordinate)
S=/tmp/claude-1000/-home-tron-spparks-kokkos/44fc50d7-1b89-492a-ac82-fa6c4a653ae3/scratchpad/ebsd
P=/home/tron/spparks-kokkos/examples/gp700_moser/srod_from_ebsd.py; A=/home/tron/HPC-Calibration/ANG-Files; T=/home/tron/HPC-Calibration/analysis_outputs/rot90_square_native
PY=/home/tron/miniconda3/envs/WORK/bin/python
run() { $PY $P $S/out "$1" "$2" $3 > $S/out/$1.log 2>&1; }
run P04_XZ $T/P04_side_a_XZ_rot90+vflip_0p8um.ang ""
run P04_YZ $A/P4/P4-3-side.ang --vflip
run P13_XZ $A/P13/P13-2-side.ang ""
run P13_YZ $A/P13/P13-3-side.ang --vflip
run P15_YZ $A/P15/P15-3-side.ang ""
run P18_XZ $A/P18/P18-2-side.ang ""
run P18_YZ $A/P18/P18-3-side.ang --vflip
run P20_XZ $T/P20_side_a_XZ_rot90_0p7um.ang ""
run P20_YZ $A/P20/P20-3-side.ang --vflip
touch $S/out/all.done
