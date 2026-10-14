#!/bin/bash
#SBATCH -J rtip-align
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 16
#SBATCH --gres=gpu:1
#SBATCH -o runs/slurm-aligned-%j.out
# Usage: sbatch run_aligned.slurm "REACTION [REACTION ...]" "GAPS" [extra rtipdeepmd options]
# AttractivePot from the aligned reactant complex (reactant_complex.py) to the mapped SI product,
# then DeepMD relaxation of the product and dimer refinement of the path maximum.
set -o pipefail
source /group/software/deepmd-kit-3.1.1/bin/activate /group/software/deepmd-kit-3.1.1
export PYTHONPATH=$PWD OMP_NUM_THREADS=2 DP_INTRA_OP_PARALLELISM_THREADS=2 DP_INTER_OP_PARALLELISM_THREADS=1
M=deepmd_pretrained/DPA-3.2-5M.pt
REACTIONS=$1; GAPS=$2; shift 2
EXTRA="$*"
jobs=()
for R in $REACTIONS; do
  D=examples/ic5c02384/reactions/$R
  [ -f $D/product_ordered.xyz ] || python3 examples/ic5c02384/map_atoms.py $D
  for G in $GAPS; do
    out=runs/aligned/$R/g$G; mkdir -p $out
    python3 examples/ic5c02384/reactant_complex.py --gap $G $D && cp $D/reactant_complex.xyz $out/complex.xyz
    jobs+=("$R $G")
  done
done
printf '%s\n' "${jobs[@]}" | xargs -P 8 -L 1 bash -c '
  R=$0; G=$1; D=examples/ic5c02384/reactions/$R; out=runs/aligned/$R/g$G
  python3 -m rtipdeepmd --method attractive --model '"$M"' --init $out/complex.xyz --molecule $D/1.xyz --molecule $D/2.xyz \
    --final $D/product_ordered.xyz --seed 0 --output-dir $out '"$EXTRA"' > $out/stdout.log 2>&1'
for R in $REACTIONS; do
  python3 examples/ic5c02384/analyze_rtip.py --model $M --relax --out runs/aligned/$R/analysis.json runs/aligned/$R/g* > runs/aligned/$R/analysis.log 2>&1
done
python3 examples/ic5c02384/ts_barriers.py --model $M --out runs/aligned/ts_from_path.json \
  --from-path $(for R in $REACTIONS; do ls -d runs/aligned/$R/g*; done) > runs/aligned/ts_from_path.log 2>&1
