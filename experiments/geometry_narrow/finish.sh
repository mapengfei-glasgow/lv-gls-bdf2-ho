#!/usr/bin/env bash
# Data completion and validation only. No model is trained or changed.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
cd "$ROOT"
: "${SOLVER_PY:?Set SOLVER_PY}"
DATA=${DATA:-surrogate_data.geometry_narrow}
CONTROL="$DATA/_campaign"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
while true; do
    ready=0
    for worker in euclid-01 euclid-10 euclid-21; do
        if [ -f "$CONTROL/$worker.failed" ]; then
            echo "Generation failed on $worker; inspect its log"
            exit 1
        fi
        if [ -f "$CONTROL/$worker.done" ]; then ready=$((ready+1)); fi
    done
    [ "$ready" -eq 3 ] && break
    echo "$(date -Is): workers complete $ready/3"
    sleep 60
done
"$SOLVER_PY" experiments/geometry_narrow/validate.py "$DATA"
touch "$CONTROL/validated.done"
echo "Geometry-variation data ready; training was not started"
