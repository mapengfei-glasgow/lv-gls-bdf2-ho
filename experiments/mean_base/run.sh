#!/usr/bin/env bash
# Run on one machine; all paths derive from this checkout.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
cd "$ROOT"
: "${SOLVER_PY:?Set SOLVER_PY to the solver environment's Python}"
: "${TRAIN_PY:?Set TRAIN_PY to the CUDA training environment's Python}"
DATA=${DATA:-surrogate_data.mean_base}
RUN=${RUN:-surrogate_runs/mean_base}
JOBS=${JOBS:-8}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
"$SOLVER_PY" experiments/mean_base/prepare.py --data "$DATA"
"$SOLVER_PY" -m surrogate.generate "$DATA/params.csv" "$DATA" \
    --mesh-root "$DATA/_geometry" --jobs "$JOBS" --steps 200
"$SOLVER_PY" experiments/mean_base/validate.py "$DATA"
bash experiments/mean_base/wait_for_gpu.sh
resume=()
if [ -f "$RUN/last.pt" ]; then resume=(--resume "$RUN/last.pt"); fi
"$TRAIN_PY" -m surrogate.train_base "$DATA" --out "$RUN" \
    --iters 120000 --batch 4 --lr 1e-3 --noise-pos 5e-5 \
    --eval-every 2500 --eval-cases 10 "${resume[@]}"
"$TRAIN_PY" -m surrogate.evaluate "$RUN" --data "$DATA" --split val --ckpt "$RUN/best.pt"
"$TRAIN_PY" -m surrogate.evaluate "$RUN" --data "$DATA" --split test --ckpt "$RUN/best.pt"
