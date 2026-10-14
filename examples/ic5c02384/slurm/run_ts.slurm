#!/bin/bash
#SBATCH -J rtip-ts
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH -o validation/ts-%j.out
# DeepMD saddle points refined from the SI transition states (writes examples/ic5c02384/ts_geometries/).
set -o pipefail
source /group/software/deepmd-kit-3.1.1/bin/activate /group/software/deepmd-kit-3.1.1
export PYTHONPATH=$PWD OMP_NUM_THREADS=4 DP_INTRA_OP_PARALLELISM_THREADS=4 DP_INTER_OP_PARALLELISM_THREADS=1
python3 examples/ic5c02384/ts_barriers.py --model deepmd_pretrained/DPA-3.2-5M.pt --out validation/ts_barriers.json
