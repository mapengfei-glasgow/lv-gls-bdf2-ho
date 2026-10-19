#!/usr/bin/env bash
# Launch on euclid-17/shared checkout. Re-run the same command to resume.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
cd "$ROOT"
: "${TRAIN_PY:?Set TRAIN_PY to the CUDA training Python}"
: "${SOLVER_PY:?Set SOLVER_PY to the DOLFINx Python}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
ulimit -n 8192
exec "$TRAIN_PY" experiments/curriculum/run.py --train-python "$TRAIN_PY" \
    --solver-python "$SOLVER_PY" --gpu "${CUDA_VISIBLE_DEVICES:-0}" "$@"
