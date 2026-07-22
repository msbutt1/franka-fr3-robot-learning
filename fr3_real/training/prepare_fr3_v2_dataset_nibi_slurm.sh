#!/usr/bin/env bash
# Usage: from fr3_real/, run: sbatch training/prepare_fr3_v2_dataset_nibi_slurm.sh
# Build the merged v2 LeRobot dataset on Nibi local disk and publish atomically.

#SBATCH --job-name=prepare_fr3_v2
#SBATCH --account=def-mqp2259
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# Slurm may execute a staged copy from /var/spool; submit-dir is the project root.
FR3="${FR3:-${SLURM_SUBMIT_DIR:-${SCRIPT_DIR}}}"
NEW_RAW_DIR="${NEW_RAW_DIR:?Submit with --export=ALL,NEW_RAW_DIR=/path/to/new/raw/demos}"
REPO_ID="${REPO_ID:-local/fr3_real_pick_place_droid_v2}"
DATASET_NAME="${REPO_ID##*/}"
PROJECT_CACHE="${FR3}/lerobot_cache"
TMP_CACHE="${SLURM_TMPDIR:?}/lerobot_cache"
PUBLISHED_DATASET="${PROJECT_CACHE}/${REPO_ID}"
INCOMING_DATASET="${PROJECT_CACHE}/local/.${DATASET_NAME}.incoming.${SLURM_JOB_ID}"
NORM_STATS_DIR="${NORM_STATS_DIR:-${FR3}/fr3_custom_assets_${DATASET_NAME}/droid}"

if [[ -e "${PUBLISHED_DATASET}" ]]; then
  echo "Refusing to replace published dataset: ${PUBLISHED_DATASET}" >&2
  echo "Use a new versioned REPO_ID for another conversion." >&2
  exit 3
fi

export HF_LEROBOT_HOME="${TMP_CACHE}"
export HF_HOME="${FR3}/huggingface_cache"
export UV_CACHE_DIR="${FR3}/uv_cache"
export OPENPI_DATA_HOME="${FR3}/openpi_cache"
export XDG_CACHE_HOME="${FR3}/xdg_cache"
export JAX_COMPILATION_CACHE_DIR="${FR3}/jax_cache"
export JAX_COMPILATION_CACHE_MAX_SIZE=21474836480
export WANDB_DIR="${FR3}/wandb"
mkdir -p \
  "${TMP_CACHE}" "${HF_HOME}" "${UV_CACHE_DIR}" "${OPENPI_DATA_HOME}" \
  "${XDG_CACHE_HOME}" "${JAX_COMPILATION_CACHE_DIR}" "${WANDB_DIR}"

echo "[prepare] host=$(hostname) job=${SLURM_JOB_ID} tmp=${SLURM_TMPDIR}"
FR3="${FR3}" \
NEW_RAW_DIR="${NEW_RAW_DIR}" \
HF_LEROBOT_HOME="${TMP_CACHE}" \
NORM_STATS_DIR="${NORM_STATS_DIR}" \
  "${SCRIPT_DIR}/prepare_fr3_v2_dataset.sh"

test -d "${TMP_CACHE}/${REPO_ID}" || {
  echo "Prepared dataset is missing: ${TMP_CACHE}/${REPO_ID}" >&2
  exit 4
}
mkdir -p "$(dirname -- "${INCOMING_DATASET}")"
test ! -e "${INCOMING_DATASET}" || {
  echo "Incoming publish path already exists: ${INCOMING_DATASET}" >&2
  exit 4
}
cp -a "${TMP_CACHE}/${REPO_ID}" "${INCOMING_DATASET}"
mv "${INCOMING_DATASET}" "${PUBLISHED_DATASET}"

echo "[prepare] published=${PUBLISHED_DATASET}"
du -sh "${PUBLISHED_DATASET}"
