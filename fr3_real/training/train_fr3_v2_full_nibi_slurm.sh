#!/usr/bin/env bash
# Usage: from fr3_real/, run: sbatch training/train_fr3_v2_full_nibi_slurm.sh
# Train one fresh merged-data full fine-tune on a Nibi H100.

#SBATCH --job-name=pi05_fr3_v2_full
#SBATCH --account=def-mqp2259
#SBATCH --gpus=h100:1
#SBATCH --cpus-per-task=12
# Full fine-tuning plus Orbax asynchronous checkpoints reached the 128G limit.
# Leave headroom for the optimizer state and an asynchronous checkpoint save.
#SBATCH --mem=192G
#SBATCH --time=24:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# Slurm may execute a staged copy from /var/spool; submit-dir is the project root.
FR3="${FR3:-${SLURM_SUBMIT_DIR:-${SCRIPT_DIR}}}"
SCRATCH_ROOT="${SCRATCH_ROOT:-/scratch/${USER}/fr3_openpi_runs}"
NORM_MODE="${NORM_MODE:-droid}"
TARGET_STEPS="${TARGET_STEPS:-12000}"
KEEP_PERIOD="${KEEP_PERIOD:-3000}"
REPO_ID="${REPO_ID:-local/fr3_real_pick_place_droid_v2}"
DATASET_VERSION="${REPO_ID##*_}"
EXP_NAME="${EXP_NAME:-}"
NORM_STATS_DIR="${NORM_STATS_DIR:-${FR3}/fr3_custom_assets_v2/droid}"
SAMPLING_MANIFEST="${SAMPLING_MANIFEST:-${FR3}/fr3_phase_sampling_v2.json}"

# Do not inherit cache locations from the login node: Hugging Face otherwise
# materializes the Arrow dataset cache under quota-limited /home.
export HF_LEROBOT_HOME="${FR3}/lerobot_cache"
export HOME="${SCRATCH_ROOT}/home"
export HF_HOME="${SCRATCH_ROOT}/huggingface_cache"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export HF_HUB_CACHE="${HF_HOME}/hub"
export HUGGINGFACE_HUB_CACHE="${HF_HUB_CACHE}"
export UV_CACHE_DIR="${SCRATCH_ROOT}/uv_cache"
export OPENPI_DATA_HOME="${SCRATCH_ROOT}/openpi_cache"
export XDG_CACHE_HOME="${SCRATCH_ROOT}/xdg_cache"
export JAX_COMPILATION_CACHE_DIR="${SCRATCH_ROOT}/jax_cache"
export JAX_COMPILATION_CACHE_MAX_SIZE=21474836480
export WANDB_DIR="${SCRATCH_ROOT}/wandb"
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.92
export WANDB_MODE="${WANDB_MODE:-offline}"

mkdir -p \
  "${HF_HOME}" "${HF_DATASETS_CACHE}" "${HF_HUB_CACHE}" \
  "${UV_CACHE_DIR}" "${OPENPI_DATA_HOME}" \
  "${XDG_CACHE_HOME}" "${JAX_COMPILATION_CACHE_DIR}" "${WANDB_DIR}" \
  "${HOME}/.cache/jax" "${SCRATCH_ROOT}/checkpoints"
test -d "${HF_LEROBOT_HOME}/${REPO_ID}" || {
  echo "Missing merged dataset: ${HF_LEROBOT_HOME}/${REPO_ID}" >&2
  exit 2
}

case "${NORM_MODE}" in
  droid)
    exp_name="${EXP_NAME:-fr3_real_droid_full_${DATASET_VERSION}_droid_norm}"
    norm_args=()
    ;;
  custom)
    exp_name="${EXP_NAME:-fr3_real_droid_full_${DATASET_VERSION}_custom_norm}"
    norm_args=(--norm-stats-dir "${NORM_STATS_DIR}")
    ;;
  *)
    echo "NORM_MODE must be droid or custom" >&2
    exit 2
    ;;
esac

checkpoint_dir="${SCRATCH_ROOT}/checkpoints/pi05_fr3_real_droid_full/${exp_name}"
mode_args=()
if [[ -d "${checkpoint_dir}" ]] && find "${checkpoint_dir}" -mindepth 1 -maxdepth 1 -type d | grep -q .; then
  mode_args+=(--resume)
  echo "[train] resuming ${checkpoint_dir}"
fi

cd "${FR3}/openpi"
source .venv/bin/activate

echo "[train] host=$(hostname) job=${SLURM_JOB_ID:-interactive}"
echo "[train] norm=${NORM_MODE} target_steps=${TARGET_STEPS} keep_period=${KEEP_PERIOD} exp=${exp_name}"
nvidia-smi

uv run --no-sync python "${SCRIPT_DIR}/train_fr3_phase_filtered.py" \
  --openpi-root "${FR3}/openpi" \
  --config pi05_fr3_real_droid_full \
  --dataset-repo "${REPO_ID}" \
  --exp-name "${exp_name}" \
  --checkpoint-base-dir "${SCRATCH_ROOT}/checkpoints" \
  --sample-manifest "${SAMPLING_MANIFEST}" \
  --num-train-steps "${TARGET_STEPS}" \
  --warmup-steps 500 \
  --peak-lr 1e-5 \
  --decay-lr 1e-6 \
  --ema-decay 0.999 \
  --save-interval 1000 \
  --keep-period "${KEEP_PERIOD}" \
  --num-workers 0 \
  "${norm_args[@]}" \
  "${mode_args[@]}"
