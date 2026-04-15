#!/bin/bash
#SBATCH -J jax-rtip
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 10
#SBATCH --gres=gpu:1
#SBATCH -o slurm-%j.out
#SBATCH -e slurm-%j.err

set -eo pipefail

WORKDIR="${SLURM_SUBMIT_DIR:-/home/lhshen/JAXversion}"


cd "${WORKDIR}"
source /group/software/deepmd-kit-3.1.1/bin/activate /group/software/deepmd-kit-3.1.1

if ! command -v python3 >/dev/null 2>&1; then
	echo "python3 not found in current environment"
	exit 1
fi

DEFAULT_MODEL="/home/lhshen/JAXversion/deepmd_pretrained/DPA-3.2-5M.pt"
MODEL_ARG="${MODEL_PATH:-${DEEPMD_MODEL:-${DEFAULT_MODEL}}}"
if [[ -z "${MODEL_ARG}" ]]; then
	echo "DeepMD model is not set. Please export MODEL_PATH or DEEPMD_MODEL."
	exit 1
fi

if [[ -d "${MODEL_ARG}" ]]; then
	echo "DeepMD model path is a directory, not a file: ${MODEL_ARG}"
	exit 1
fi

if [[ ! -f "${MODEL_ARG}" ]]; then
	if [[ "${MODEL_ARG}" == *"/"* || "${MODEL_ARG}" == *.pb || "${MODEL_ARG}" == *.pth || "${MODEL_ARG}" == *.pt ]]; then
		echo "DeepMD model file not found: ${MODEL_ARG}"
		exit 1
	fi

	if command -v dp >/dev/null 2>&1; then
		if ! dp -h 2>/dev/null | grep -q "pretrained"; then
			DP_VERSION="$(python3 - <<'PY'
import deepmd
print(getattr(deepmd, "__version__", "unknown"))
PY
)"
			echo "Current deepmd-kit (${DP_VERSION}) does not support pretrained model aliases: ${MODEL_ARG}"
			echo "Please set MODEL_PATH to an existing .pb/.pth/.pt file."
			exit 1
		fi
	fi
fi

export JAX_ENABLE_X64="${JAX_ENABLE_X64:-1}"


python3 -m src.lib \
	--no-mpi \
	--xyz-dir "${WORKDIR}" \
	--model "${MODEL_ARG}" \
	--output-index "${SLURM_JOB_ID:-0}" \
	--supercell-out "supercell-${SLURM_JOB_ID:-0}.xyz"
