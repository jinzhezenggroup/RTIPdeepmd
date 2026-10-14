#!/bin/bash
#SBATCH -J rtip-thermo
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 16
#SBATCH -o validation/thermo-%j.out
set -o pipefail
source /group/software/deepmd-kit-3.1.1/bin/activate /group/software/deepmd-kit-3.1.1
export OMP_NUM_THREADS=16 DP_INTRA_OP_PARALLELISM_THREADS=16 DP_INTER_OP_PARALLELISM_THREADS=1 CUDA_VISIBLE_DEVICES="" PYTHONPATH=$PWD
python3 examples/ic5c02384/thermo.py --model deepmd_pretrained/DPA-3.2-5M.pt --out validation/thermo.json
