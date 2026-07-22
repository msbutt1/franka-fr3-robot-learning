#!/usr/bin/env bash
# Usage: from fr3_real/, run: bash training/prepare_fr3_v2_dataset.sh
# Audit, stage, convert, sample, and compute custom stats for the merged FR3 v2 dataset.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
FR3="${FR3:-${SCRIPT_DIR}}"
ORIGINAL_RAW="${ORIGINAL_RAW:-${FR3}/fr3_real_recordings_droid}"
NEW_RAW_DIR="${NEW_RAW_DIR:?Set NEW_RAW_DIR to the directory containing only the new demonstrations}"
MERGED_RAW="${MERGED_RAW:-${FR3}/fr3_real_recordings_droid_v2_staged}"
REPO_ID="${REPO_ID:-local/fr3_real_pick_place_droid_v2}"
NORM_STATS_DIR="${NORM_STATS_DIR:-${FR3}/fr3_custom_assets_v2/droid}"
SAMPLING_MANIFEST="${SAMPLING_MANIFEST:-${FR3}/fr3_phase_sampling_v2.json}"

export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-${FR3}/lerobot_cache}"
export HF_HOME="${HF_HOME:-${FR3}/huggingface_cache}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-${FR3}/uv_cache}"
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-${FR3}/openpi_cache}"

test -d "${FR3}/openpi" || { echo "Missing ${FR3}/openpi" >&2; exit 2; }
test -d "${ORIGINAL_RAW}" || { echo "Missing original data: ${ORIGINAL_RAW}" >&2; exit 2; }
test -d "${NEW_RAW_DIR}" || { echo "Missing new data: ${NEW_RAW_DIR}" >&2; exit 2; }
PYTHON="${FR3}/openpi/.venv/bin/python"
test -x "${PYTHON}" || { echo "Missing OpenPI Python: ${PYTHON}" >&2; exit 2; }
if [[ "${RECONVERT:-0}" == "1" && -f "${NORM_STATS_DIR}/norm_stats.json" ]]; then
  echo "RECONVERT=1 would make existing custom stats ambiguous: ${NORM_STATS_DIR}" >&2
  echo "Set NORM_STATS_DIR to a new versioned path before rebuilding." >&2
  exit 3
fi

"${PYTHON}" "${SCRIPT_DIR}/../model-validation/audit_fr3_new_demos.py" \
  --raw-dir "${NEW_RAW_DIR}" \
  --protected-manifest "${SCRIPT_DIR}/../configs/fr3_spatial_validation_v3.json" \
  --protected-manifest "${SCRIPT_DIR}/../configs/fr3_interstitial_eval_v1.json" \
  --min-distance "${HELD_OUT_RADIUS_M:-0.03}" \
  --min-episodes "${MIN_NEW_DEMOS:-60}" \
  --verify-video-decode \
  --output "${FR3}/fr3_new_demos_audit_v2.json"

stage_flags=()
if [[ -f "${MERGED_RAW}/.fr3_merged_recordings.json" ]]; then
  stage_flags+=(--replace)
fi
"${PYTHON}" "${SCRIPT_DIR}/../data-pipeline/stage_fr3_merged_recordings.py" \
  --source "${ORIGINAL_RAW}" \
  --source "${NEW_RAW_DIR}" \
  --output "${MERGED_RAW}" \
  "${stage_flags[@]}"

dataset_dir="${HF_LEROBOT_HOME}/${REPO_ID}"
convert_flags=()
if [[ -e "${dataset_dir}" ]]; then
  if [[ "${RECONVERT:-0}" == "1" ]]; then
    convert_flags+=(--overwrite)
  else
    echo "[prepare] reusing existing dataset pending exact manifest validation: ${dataset_dir}"
  fi
fi

cd "${FR3}/openpi"
if [[ ! -e "${dataset_dir}" || "${RECONVERT:-0}" == "1" ]]; then
  uv run --no-sync python "${SCRIPT_DIR}/../data-pipeline/convert_raw_recordings_to_lerobot.py" \
    --raw_dir "${MERGED_RAW}" \
    --output_dir "${HF_LEROBOT_HOME}" \
    --repo_id "${REPO_ID}" \
    --schema droid_joint_velocity \
    --fps 15 \
    "${convert_flags[@]}"
fi

uv run --no-sync python "${SCRIPT_DIR}/../data-pipeline/build_fr3_phase_sampling_manifest.py" \
  --raw-dir "${MERGED_RAW}" \
  --output "${SAMPLING_MANIFEST}" \
  --repo-id "${REPO_ID}"

uv run --no-sync python "${SCRIPT_DIR}/train_fr3_phase_filtered.py" \
  --openpi-root "${FR3}/openpi" \
  --config pi05_fr3_real_droid_full \
  --dataset-repo "${REPO_ID}" \
  --exp-name validate_fr3_v2_alignment \
  --sample-manifest "${SAMPLING_MANIFEST}" \
  --num-train-steps 1 \
  --validate-only

if [[ -f "${NORM_STATS_DIR}/norm_stats.json" ]]; then
  echo "[prepare] reusing existing custom stats: ${NORM_STATS_DIR}/norm_stats.json"
else
  uv run --no-sync python "${SCRIPT_DIR}/../data-pipeline/compute_fr3_custom_norm_stats.py" \
    --openpi-root "${FR3}/openpi" \
    --config pi05_fr3_real_droid_full \
    --dataset-repo "${REPO_ID}" \
    --output "${NORM_STATS_DIR}"
fi

echo "[prepare] merged dataset and both normalization branches are ready"
echo "[prepare] dataset=${dataset_dir}"
echo "[prepare] sampling=${SAMPLING_MANIFEST}"
echo "[prepare] custom_stats=${NORM_STATS_DIR}/norm_stats.json"
