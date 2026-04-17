#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${ROOT}/.venv/bin/python"

if [[ ! -x "${PYTHON}" ]]; then
  echo "ERROR: Python executable not found at ${PYTHON}."
  echo "Please create a virtual environment at .venv and install dependencies first."
  exit 1
fi

cd "${ROOT}/SCRIPTS"

echo "Running EDA plots..."
"${PYTHON}" plot_macros.py
"${PYTHON}" ingredient_analysis.py

echo "Training model..."
"${PYTHON}" train.py

echo "Evaluating model..."
"${PYTHON}" evaluate.py

echo "Generating result plot..."
"${PYTHON}" plot_results.py

echo "Done. Output files are under ${ROOT}/OUTPUT"
