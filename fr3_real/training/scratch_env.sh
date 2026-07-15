#!/usr/bin/env bash
# Source this before OpenPI install/conversion/training on clusters with small
# home quotas. It keeps tool caches and temporary files under the FR3 scratch dir.

set -euo pipefail

FR3="${FR3:-/home/lihuyue/scratch/anastasia_fr3}"
SCRATCH_CACHE_ROOT="${SCRATCH_CACHE_ROOT:-${FR3}}"

export UV_CACHE_DIR="${UV_CACHE_DIR:-${SCRATCH_CACHE_ROOT}/uv_cache}"
export TMPDIR="${TMPDIR:-${SCRATCH_CACHE_ROOT}/tmp}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${SCRATCH_CACHE_ROOT}/xdg_cache}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${SCRATCH_CACHE_ROOT}/pip_cache}"
export HF_HOME="${HF_HOME:-${SCRATCH_CACHE_ROOT}/hf_home}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
export TORCH_HOME="${TORCH_HOME:-${SCRATCH_CACHE_ROOT}/torch_home}"
export WANDB_DIR="${WANDB_DIR:-${SCRATCH_CACHE_ROOT}/wandb}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-${SCRATCH_CACHE_ROOT}/wandb_cache}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${SCRATCH_CACHE_ROOT}/mpl_config}"
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-${SCRATCH_CACHE_ROOT}/openpi_cache}"
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-${SCRATCH_CACHE_ROOT}/lerobot_cache}"

mkdir -p \
    "${UV_CACHE_DIR}" \
    "${TMPDIR}" \
    "${XDG_CACHE_HOME}" \
    "${PIP_CACHE_DIR}" \
    "${HF_HOME}" \
    "${HUGGINGFACE_HUB_CACHE}" \
    "${TRANSFORMERS_CACHE}" \
    "${TORCH_HOME}" \
    "${WANDB_DIR}" \
    "${WANDB_CACHE_DIR}" \
    "${MPLCONFIGDIR}" \
    "${OPENPI_DATA_HOME}" \
    "${HF_LEROBOT_HOME}"
