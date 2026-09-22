#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PYTHON="${SCRIPT_DIR}/.venv/bin/python"

if [ ! -f "${VENV_PYTHON}" ]; then
    echo "Virtual environment not found at ${SCRIPT_DIR}/.venv. Please create it first."
    exit 1
fi

echo "=========================================================="
echo "Starting OmniSearch Agent (RTX A4000 16GB Optimized)"
echo "=========================================================="

PORT=${1:-7860}
"${VENV_PYTHON}" "${SCRIPT_DIR}/app.py" --port "${PORT}"
