#!/bin/bash
#SBATCH -J jax-rtip
#SBATCH --time=00:15:00
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
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-1}}"
export DP_INTRA_OP_PARALLELISM_THREADS="${DP_INTRA_OP_PARALLELISM_THREADS:-${OMP_NUM_THREADS}}"
export DP_INTER_OP_PARALLELISM_THREADS="${DP_INTER_OP_PARALLELISM_THREADS:-1}"

OUTPUT_INDEX="${OUTPUT_INDEX:-${SLURM_JOB_ID:-0}}"
XYZ_DIR="${XYZ_DIR:-${WORKDIR}}"
SUPERCELL_OUT="${SUPERCELL_OUT:-supercell-${OUTPUT_INDEX}.xyz}"

# Conservative MD defaults for stable DeepMD RTIP startup.
MD_PRINT_STEP="${MD_PRINT_STEP:-10}"
MD_DT="${MD_DT:-0.05}"
MD_TAU="${MD_TAU:-200}"
MD_TEMP_BATH="${MD_TEMP_BATH:-800}"
MD_TEMP_START_SCALE="${MD_TEMP_START_SCALE:-0.02}"
MD_TEMP_RAMP_STEPS="${MD_TEMP_RAMP_STEPS:-1000}"
MD_THERMOSTAT_LAM_MIN="${MD_THERMOSTAT_LAM_MIN:-0.995}"
MD_THERMOSTAT_LAM_MAX="${MD_THERMOSTAT_LAM_MAX:-1.005}"

declare -a MD_OVERRIDES=()
if [[ -n "${MD_MAX_STEP:-}" ]]; then
	MD_OVERRIDES+=(--md-max-step "${MD_MAX_STEP}")
fi
if [[ -n "${MD_PRINT_STEP:-}" ]]; then
	MD_OVERRIDES+=(--md-print-step "${MD_PRINT_STEP}")
fi
if [[ -n "${MD_DT:-}" ]]; then
	MD_OVERRIDES+=(--md-dt "${MD_DT}")
fi
if [[ -n "${MD_TAU:-}" ]]; then
	MD_OVERRIDES+=(--md-tau "${MD_TAU}")
fi
if [[ -n "${MD_TEMP_BATH:-}" ]]; then
	MD_OVERRIDES+=(--md-temp-bath "${MD_TEMP_BATH}")
fi
if [[ -n "${MD_TEMP_START_SCALE:-}" ]]; then
	MD_OVERRIDES+=(--md-temp-start-scale "${MD_TEMP_START_SCALE}")
fi
if [[ -n "${MD_TEMP_RAMP_STEPS:-}" ]]; then
	MD_OVERRIDES+=(--md-temp-ramp-steps "${MD_TEMP_RAMP_STEPS}")
fi
if [[ -n "${MD_THERMOSTAT_LAM_MIN:-}" ]]; then
	MD_OVERRIDES+=(--md-thermostat-lam-min "${MD_THERMOSTAT_LAM_MIN}")
fi
if [[ -n "${MD_THERMOSTAT_LAM_MAX:-}" ]]; then
	MD_OVERRIDES+=(--md-thermostat-lam-max "${MD_THERMOSTAT_LAM_MAX}")
fi
if [[ "${RTIP_DRY_RUN:-0}" == "1" || "${DRY_RUN:-0}" == "1" ]]; then
	MD_OVERRIDES+=(--dry-run)
fi

echo "Running RTIP-DeepMD with model=${MODEL_ARG}, output-index=${OUTPUT_INDEX}"


python3 -m src.lib \
	--no-mpi \
	--xyz-dir "${XYZ_DIR}" \
	--model "${MODEL_ARG}" \
	--output-index "${OUTPUT_INDEX}" \
	--supercell-out "${SUPERCELL_OUT}" \
	"${MD_OVERRIDES[@]}"
