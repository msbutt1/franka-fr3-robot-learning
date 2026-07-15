#!/usr/bin/env bash
# Source this before OpenPI install/conversion/training on FIR. It defines the
# complete project layout and keeps all writable caches under scratch.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

export FR3="${FR3:-/home/lihuyue/scratch/anastasia_fr3}"
export REPO_DIR="${REPO_DIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
export OPENPI_DIR="${OPENPI_DIR:-${FR3}/openpi}"
export LEROBOT_CACHE="${LEROBOT_CACHE:-${FR3}/lerobot_cache}"
export RAW_DIR="${RAW_DIR:-${REPO_DIR}/fr3_raw_recordings}"
export DATASET_REPO_ID="${DATASET_REPO_ID:-local/fr3_real_pick_place_droid}"
export DATASET_DIR="${DATASET_DIR:-${LEROBOT_CACHE}/${DATASET_REPO_ID}}"
export DATASET="${DATASET:-${DATASET_DIR}}"
export LOG_DIR="${LOG_DIR:-${FR3}/logs}"

SCRATCH_CACHE_ROOT="${SCRATCH_CACHE_ROOT:-${FR3}}"

export UV_CACHE_DIR="${UV_CACHE_DIR:-${SCRATCH_CACHE_ROOT}/uv_cache}"
export TMPDIR="${TMPDIR:-${SCRATCH_CACHE_ROOT}/tmp}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${SCRATCH_CACHE_ROOT}/xdg_cache}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${SCRATCH_CACHE_ROOT}/pip_cache}"
export HF_HOME="${HF_HOME:-${SCRATCH_CACHE_ROOT}/hf_home}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export TORCH_HOME="${TORCH_HOME:-${SCRATCH_CACHE_ROOT}/torch_home}"
export WANDB_DIR="${WANDB_DIR:-${SCRATCH_CACHE_ROOT}/wandb}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-${SCRATCH_CACHE_ROOT}/wandb_cache}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${SCRATCH_CACHE_ROOT}/mpl_config}"
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-${SCRATCH_CACHE_ROOT}/openpi_cache}"
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-${SCRATCH_CACHE_ROOT}/lerobot_cache}"
export CLOUDSDK_CONFIG="${CLOUDSDK_CONFIG:-${SCRATCH_CACHE_ROOT}/gcloud_config}"
export CUDA_CACHE_PATH="${CUDA_CACHE_PATH:-${SCRATCH_CACHE_ROOT}/cuda_cache}"
export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-${SCRATCH_CACHE_ROOT}/jax_cache}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${SCRATCH_CACHE_ROOT}/triton_cache}"
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-${SCRATCH_CACHE_ROOT}/numba_cache}"
export PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX:-${SCRATCH_CACHE_ROOT}/python_cache}"
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-${SCRATCH_CACHE_ROOT}/uv_python}"
export UV_TOOL_DIR="${UV_TOOL_DIR:-${SCRATCH_CACHE_ROOT}/uv_tools}"
export UV_TOOL_BIN_DIR="${UV_TOOL_BIN_DIR:-${SCRATCH_CACHE_ROOT}/bin}"
export UV_LINK_MODE="${UV_LINK_MODE:-copy}"

export PATH="${UV_TOOL_BIN_DIR}:${PATH}"

mkdir -p \
    "${UV_CACHE_DIR}" \
    "${TMPDIR}" \
    "${XDG_CACHE_HOME}" \
    "${PIP_CACHE_DIR}" \
    "${HF_HOME}" \
    "${HUGGINGFACE_HUB_CACHE}" \
    "${TRANSFORMERS_CACHE}" \
    "${HF_DATASETS_CACHE}" \
    "${TORCH_HOME}" \
    "${WANDB_DIR}" \
    "${WANDB_CACHE_DIR}" \
    "${MPLCONFIGDIR}" \
    "${OPENPI_DATA_HOME}" \
    "${HF_LEROBOT_HOME}" \
    "${CLOUDSDK_CONFIG}" \
    "${CUDA_CACHE_PATH}" \
    "${JAX_COMPILATION_CACHE_DIR}" \
    "${TRITON_CACHE_DIR}" \
    "${NUMBA_CACHE_DIR}" \
    "${PYTHONPYCACHEPREFIX}" \
    "${UV_PYTHON_INSTALL_DIR}" \
    "${UV_TOOL_DIR}" \
    "${UV_TOOL_BIN_DIR}" \
    "${LOG_DIR}" \
    "$(dirname "${DATASET_DIR}")"

echo "FIR scratch environment ready:"
echo "  project:  ${FR3}"
echo "  repo:     ${REPO_DIR}"
echo "  OpenPI:   ${OPENPI_DIR}"
echo "  raw data: ${RAW_DIR}"
echo "  dataset:  ${DATASET_DIR}"
