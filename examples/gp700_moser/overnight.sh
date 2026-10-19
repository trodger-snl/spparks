#!/bin/bash
# Unattended chain (2026-09-04 night): analyse v3 and v4 as they finish, score v4
# against Lamb, and if v4 misses run the Sobol study on the proxy geometry.
# Launch detached: setsid nohup ./overnight.sh > overnight.log 2>&1 &
# R (runs dir, default ~/gp700_runs) and PY (the WORK env python) come from the environment.
set -u
R=${R:-$HOME/gp700_runs}; E=$(cd "$(dirname "$0")" && pwd); PY=${PY:-python}
BIN=$R/spparks_ad23c1a5
log(){ echo "[$(date +%H:%M:%S)] $*"; }
analyse(){ # name
  local st=$R/$1/gp700.st; mkdir -p $R/$1/analysis_floor100 $R/$1/analysis_floor240
  $PY $E/gp700_analysis.py $st --out $R/$1/analysis_floor100 --floor-um 100 --crop 200 600 200 600 > $R/$1/analysis_floor100/stdout.txt 2>&1
  $PY $E/gp700_analysis.py $st --out $R/$1/analysis_floor240 --floor-um 240 --crop 200 600 200 600 > $R/$1/analysis_floor240/stdout.txt 2>&1
  log "$1 analysed: $(grep -E '^grains:|^texture|^height originating' $R/$1/analysis_floor100/stdout.txt | tr '\n' ' | ')"
  log "$1 score(floor100): $($PY $E/sobol_study.py --out /tmp/x --binary /bin/true --score-only $R/$1/analysis_floor100/summary.json)"
}
until grep -q 'all 12 boxes done\|FAILED\|STOP file' $R/wide12_v3.log; do sleep 60; done
log "v3 finished: $(tail -1 $R/wide12_v3.log)"; analyse wide12_v3
until grep -q 'all 12 boxes done\|FAILED\|STOP file' $R/wide12_v4.log; do sleep 60; done
log "v4 finished: $(tail -1 $R/wide12_v4.log)"; analyse wide12_v4
SC=$($PY $E/sobol_study.py --out /tmp/x --binary /bin/true --score-only $R/wide12_v4/analysis_floor100/summary.json)
RMS=$($PY -c "import json,sys; s=json.loads(sys.argv[1]); print(s['rms_srod_dev'] if s else 99)" "$SC")
log "v4 rms SROD deviation vs Lamb (25-240 um): $RMS deg"
if $PY -c "import sys; sys.exit(0 if float(sys.argv[1]) > 1.5 else 1)" "$RMS"; then
  log "v4 misses (> 1.5 deg): starting the Sobol study (n=16, proxy 600 um / 7 hatches / 8 layers, deadline 08:30)"
  $PY $E/sobol_study.py --out $R/sobol1 --binary $BIN --n 16 --deadline 08:30
else
  log "v4 within 1.5 deg of Lamb: no Sobol study started"
fi
log "overnight chain finished"
