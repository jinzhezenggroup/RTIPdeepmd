#!/bin/bash
#SBATCH -J rtip-deepmd
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 10
#SBATCH --gres=gpu:1
#SBATCH -o slurm-%j.out
#SBATCH -e slurm-%j.err
#
# Usage:  sbatch run.slurm [extra options for `python -m rtipdeepmd`]
#   e.g.  sbatch run.slurm --max-step 20000 --seed 1 --stop-temp 30000
# Environment overrides: DEEPMD_ENV, MODEL_PATH, MODEL_HEAD, OUTPUT_DIR

set -eo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"

DEEPMD_ENV="${DEEPMD_ENV:-/group/software/deepmd-kit-3.1.1}"
source "${DEEPMD_ENV}/bin/activate" "${DEEPMD_ENV}"

MODEL_PATH="${MODEL_PATH:-deepmd_pretrained/DPA-3.2-5M.pt}"
if [[ ! -f "${MODEL_PATH}" ]]; then
	echo "DeepMD model not found: ${MODEL_PATH} (set MODEL_PATH)" >&2
	exit 1
fi

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-1}}"
export DP_INTRA_OP_PARALLELISM_THREADS="${DP_INTRA_OP_PARALLELISM_THREADS:-${OMP_NUM_THREADS}}"
export DP_INTER_OP_PARALLELISM_THREADS="${DP_INTER_OP_PARALLELISM_THREADS:-1}"

OUTPUT_DIR="${OUTPUT_DIR:-runs/${SLURM_JOB_ID:-local}}"

python3 -m rtipdeepmd \
	--model "${MODEL_PATH}" \
	${MODEL_HEAD:+--head "${MODEL_HEAD}"} \
	--output-dir "${OUTPUT_DIR}" \
	"$@"
