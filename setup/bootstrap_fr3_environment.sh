#!/usr/bin/env bash
# Create the FR3 Python environment without installing system packages.

set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${REPO_ROOT}/fr3_real/configs/environment-fr3-recording.yml"
REQUIREMENTS="${REPO_ROOT}/requirements/fr3-recording.txt"
ENV_NAME="${FR3_ENV_NAME:-fr3-recording}"

if command -v conda >/dev/null 2>&1; then
  CONDA_CMD=(conda)
elif command -v mamba >/dev/null 2>&1; then
  CONDA_CMD=(mamba)
else
  echo "Missing Conda or Mamba. Install Miniforge/Conda, then rerun." >&2
  exit 2
fi

if "${CONDA_CMD[@]}" env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  echo "[setup] reusing existing environment: ${ENV_NAME}"
else
  echo "[setup] creating environment: ${ENV_NAME}"
  "${CONDA_CMD[@]}" env create -f "${ENV_FILE}" -n "${ENV_NAME}"
fi

echo "[setup] installing repository Python dependencies"
cd "${REPO_ROOT}"
"${CONDA_CMD[@]}" run -n "${ENV_NAME}" \
  python -m pip install -r "${REQUIREMENTS}"

if "${CONDA_CMD[@]}" run -n "${ENV_NAME}" python -c 'import openpi_client' >/dev/null 2>&1; then
  echo "[setup] openpi-client is already importable"
else
  echo "[setup] installing checked-out openpi-client"
  "${CONDA_CMD[@]}" run -n "${ENV_NAME}" \
    python -m pip install -e "${REPO_ROOT}/openpi-client"
fi

echo "[setup] validating the installed environment"
"${CONDA_CMD[@]}" run -n "${ENV_NAME}" \
  python "${REPO_ROOT}/setup/verify_fr3_setup.py" --python-only

cat <<EOF
[setup] Python setup complete.
[setup] Before live robot use, install/configure the host prerequisites listed by:
  conda run -n ${ENV_NAME} python ${REPO_ROOT}/setup/verify_fr3_setup.py
EOF
