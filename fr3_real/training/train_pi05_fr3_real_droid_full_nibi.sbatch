#!/usr/bin/env bash
# Full pi0.5-DROID fine-tune on Nibi using the pre-converted FR3 dataset.
# Submit from the repository root:
#   sbatch fr3_real/training/train_pi05_fr3_real_droid_full_nibi.sbatch

#SBATCH --job-name=pi05_fr3_nibi
#SBATCH --account=def-mqp2259
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail

NIBI_ROOT="${NIBI_ROOT:-/home/lihuyue/projects/def-mqp2259/fr3_anastasia}"
REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-${NIBI_ROOT}/franka-fr3-robot-learning}}"
OPENPI_DIR="${OPENPI_DIR:-${NIBI_ROOT}/openpi}"
VENV_DIR="${VENV_DIR:-${OPENPI_DIR}/.venv}"
DATASET_REPO_ID="${DATASET_REPO_ID:-local/fr3_real_pick_place_droid}"
DATASET="${DATASET:-${NIBI_ROOT}/${DATASET_REPO_ID}}"

CONFIG="${CONFIG:-pi05_fr3_real_droid_full}"
EXP_NAME="${EXP_NAME:-fr3_real_droid_full_nibi_v1}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_TRAIN_STEPS="${NUM_TRAIN_STEPS:-20000}"
NUM_WORKERS="${NUM_WORKERS:-0}"
SAVE_INTERVAL="${SAVE_INTERVAL:-1000}"
KEEP_PERIOD="${KEEP_PERIOD:-5000}"
CHECKPOINT="${CHECKPOINT:-gs://openpi-assets/checkpoints/pi05_droid/params}"
ASSETS_DIR="${ASSETS_DIR:-gs://openpi-assets/checkpoints/pi05_droid/assets}"
ASSET_ID="${ASSET_ID:-droid}"

CACHE_ROOT="${CACHE_ROOT:-${NIBI_ROOT}/cache}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-${CACHE_ROOT}/uv}"
export TMPDIR="${TMPDIR:-${CACHE_ROOT}/tmp}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${CACHE_ROOT}/xdg}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${CACHE_ROOT}/pip}"
export HF_HOME="${HF_HOME:-${CACHE_ROOT}/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
export TORCH_HOME="${TORCH_HOME:-${CACHE_ROOT}/torch}"
export WANDB_DIR="${WANDB_DIR:-${NIBI_ROOT}/logs/wandb}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-${CACHE_ROOT}/wandb}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${CACHE_ROOT}/matplotlib}"
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-${NIBI_ROOT}/openpi_cache}"
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-${NIBI_ROOT}}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.92}"
export WANDB_MODE="${WANDB_MODE:-offline}"

mkdir -p \
    "${UV_CACHE_DIR}" "${TMPDIR}" "${XDG_CACHE_HOME}" "${PIP_CACHE_DIR}" \
    "${HF_HOME}" "${HUGGINGFACE_HUB_CACHE}" "${TRANSFORMERS_CACHE}" \
    "${TORCH_HOME}" "${WANDB_DIR}" "${WANDB_CACHE_DIR}" "${MPLCONFIGDIR}" \
    "${OPENPI_DATA_HOME}"

if [ "${BATCH_SIZE}" -gt 32 ] || [ "${BATCH_SIZE}" -lt 1 ]; then
    echo "BATCH_SIZE must be between 1 and 32; got ${BATCH_SIZE}." >&2
    exit 2
fi
if [ ! -f "${OPENPI_DIR}/scripts/train.py" ]; then
    echo "Missing OpenPI checkout: ${OPENPI_DIR}" >&2
    echo "Set OPENPI_DIR to the checkout containing scripts/train.py." >&2
    exit 1
fi
if [ ! -f "${VENV_DIR}/bin/activate" ]; then
    echo "Missing OpenPI virtual environment: ${VENV_DIR}" >&2
    echo "Set VENV_DIR if the OpenPI environment is elsewhere." >&2
    exit 1
fi
if [ ! -f "${DATASET}/meta/info.json" ]; then
    echo "Missing dataset: ${DATASET}/meta/info.json" >&2
    exit 1
fi

echo "Repo:       ${REPO_DIR}"
echo "OpenPI:     ${OPENPI_DIR}"
echo "Venv:       ${VENV_DIR}"
echo "Dataset:    ${DATASET}"
echo "Checkpoint: ${CHECKPOINT}"
echo "Assets:     ${ASSETS_DIR} (${ASSET_ID})"
echo "Run:        ${CONFIG}/${EXP_NAME}"
echo "Batch/steps:${BATCH_SIZE}/${NUM_TRAIN_STEPS}"

cd "${OPENPI_DIR}"
source "${VENV_DIR}/bin/activate"

python "${REPO_DIR}/fr3_real/training/prepare_pi05_fr3_real_droid_full.py" \
    --openpi-dir "${OPENPI_DIR}" \
    --config-name "${CONFIG}" \
    --repo-id "${DATASET_REPO_ID}" \
    --num-train-steps "${NUM_TRAIN_STEPS}" \
    --batch-size "${BATCH_SIZE}" \
    --num-workers "${NUM_WORKERS}" \
    --save-interval "${SAVE_INTERVAL}" \
    --keep-period "${KEEP_PERIOD}" \
    --checkpoint "${CHECKPOINT}" \
    --assets-dir "${ASSETS_DIR}" \
    --asset-id "${ASSET_ID}"

python -c "import ast; ast.parse(open('src/openpi/training/config.py').read()); print('config.py parses OK')"
python "${REPO_DIR}/fr3_real/training/inspect_pi05_fr3_lerobot_dataset.py" \
    "${DATASET}" --expected-fps 15 --strict-fps
nvidia-smi

CKPT_DIR="${OPENPI_DIR}/checkpoints/${CONFIG}/${EXP_NAME}"
RESUME_FLAG=()
if [ -d "${CKPT_DIR}" ] && [ -n "$(ls -A "${CKPT_DIR}" 2>/dev/null)" ]; then
    RESUME_FLAG=(--resume)
    echo "Resuming from ${CKPT_DIR}"
fi

uv run --no-sync scripts/train.py "${CONFIG}" --exp-name="${EXP_NAME}" "${RESUME_FLAG[@]}" \
    2>&1 | tee "${NIBI_ROOT}/logs/${EXP_NAME}.log"
