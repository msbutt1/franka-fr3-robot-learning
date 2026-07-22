#!/usr/bin/env bash
# Usage: from fr3_real/, run: bash setup/bootstrap_fr3_environment.sh
# Compatibility launcher: run the repository bootstrap from fr3_real/.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/../../setup/bootstrap_fr3_environment.sh" "$@"
