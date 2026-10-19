#!/bin/bash
# MPI scaling benchmark for the Rosenthal & Moser temperature sources (kk app).
# Self-contained (analytic sources, no external data file). Sequential, bounded
# runs (each completes on its own -- NEVER kill a running GPU job; stuck HIP
# kernels can peg the GPU). Usage: ./run_baseline.sh [spparks_binary]
set -u
SPK="${1:-$(dirname "$0")/../build-verify/spparks}"
cd "$(dirname "$0")"
echo "source,ranks,wall_s,loop_s,steps" > baseline_rm.csv
run() { local src=$1 input=$2 ranks=$3
  local t0=$(date +%s.%N)
  mpirun -np "$ranks" --bind-to core "$SPK" < "$input" > "${src}_np${ranks}.log" 2>&1
  local rc=$?; local wall=$(echo "$(date +%s.%N) - $t0"|bc)
  local loop=$(grep -oE "Loop time of [0-9.]+" "${src}_np${ranks}.log"|grep -oE "[0-9.]+"|tail -1)
  local steps=$(grep -oE "step=[0-9]+" "${src}_np${ranks}.log"|grep -oE "[0-9]+"|tail -1)
  echo "$src,$ranks,$wall,${loop:-NA},${steps:-NA}" >> baseline_rm.csv
  echo "$src np$ranks: rc=$rc wall=${wall}s loop=${loop}s steps=${steps}"
}
for r in 1 4 8; do run rosenthal in.rosenthal_mpi_bench "$r"; done
for r in 1 4 8; do run moser     in.moser_mpi_bench     "$r"; done
cat baseline_rm.csv
