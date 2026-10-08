#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/data_pipeline/src${PYTHONPATH:+:${PYTHONPATH}}"

if ! python3 -c "import pytest" >/dev/null 2>&1; then
  echo "pytest is required. Install it with: python3 -m pip install pytest" >&2
  exit 1
fi

python3 -m pytest data_pipeline/tests -v
