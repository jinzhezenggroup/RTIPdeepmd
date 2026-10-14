#!/bin/bash
#SBATCH -J rtip-synth
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 16
#SBATCH --gres=gpu:1
#SBATCH -o runs/slurm-synthesis-%j.out
# Usage: sbatch run_synthesis.slurm TAG "REACTION [REACTION ...]" "SEEDS" [extra rtipdeepmd options]
#   TAG=site: RTIP on the B-N=N site of 1-R plus the whole small molecule; TAG=whole: whole molecules
set -o pipefail
source /group/software/deepmd-kit-3.1.1/bin/activate /group/software/deepmd-kit-3.1.1
export PYTHONPATH=$PWD OMP_NUM_THREADS=2 DP_INTRA_OP_PARALLELISM_THREADS=2 DP_INTER_OP_PARALLELISM_THREADS=1
M=deepmd_pretrained/DPA-3.2-5M.pt
TAG=$1; REACTIONS=$2; SEEDS=$3; shift 3
for R in $REACTIONS; do
  D=examples/ic5c02384/reactions/$R
  FRAG=()
  [ "$TAG" = site ] && FRAG=(--fragment "$(python3 examples/ic5c02384/reactive_site.py $D/1.xyz)")
  runs=()
  for S in $SEEDS; do
    out=runs/synthesis_$TAG/$R/s$S
    mkdir -p $out
    python3 -m rtipdeepmd --method synthesis --model $M --molecule $D/1.xyz --molecule $D/2.xyz \
      --seed $S --output-dir $out "${FRAG[@]}" "$@" > $out/stdout.log 2>&1 &
    runs+=($out)
  done
  wait
  python3 examples/ic5c02384/analyze_rtip.py --model $M --relax --out runs/synthesis_$TAG/$R/analysis.json "${runs[@]}" > runs/synthesis_$TAG/$R/analysis.log 2>&1
done
