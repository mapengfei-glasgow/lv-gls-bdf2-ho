#!/usr/bin/env bash
# Wait for all workers, validate the full dataset, then train on euclid-17.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
cd "$ROOT"
: "${SOLVER_PY:?Set SOLVER_PY}"
: "${TRAIN_PY:?Set TRAIN_PY}"
DATA=${DATA:-surrogate_data.mean_base}
RUN=${RUN:-surrogate_runs/mean_base}
CONTROL="$DATA/_campaign"
WORKERS=(euclid-17 euclid-01 euclid-10 euclid-21 euclid-23)
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
while true; do
    ready=0
    for worker in "${WORKERS[@]}"; do
        if [ -f "$CONTROL/$worker.failed" ]; then
            echo "Generation failed on $worker; inspect $CONTROL/$worker.log"
            exit 1
        fi
        if [ -f "$CONTROL/$worker.done" ]; then ready=$((ready + 1)); fi
    done
    [ "$ready" -eq "${#WORKERS[@]}" ] && break
    echo "$(date -Is): generation workers complete $ready/${#WORKERS[@]}"
    sleep 60
done
"$SOLVER_PY" experiments/mean_base/validate.py "$DATA"
bash experiments/mean_base/wait_for_gpu.sh
resume=()
if [ -f "$RUN/last.pt" ]; then resume=(--resume "$RUN/last.pt"); fi
"$TRAIN_PY" -m surrogate.train_base "$DATA" --out "$RUN" \
    --iters 120000 --batch 4 --lr 1e-3 --noise-pos 5e-5 \
    --eval-every 2500 --eval-cases 10 "${resume[@]}"
"$TRAIN_PY" -m surrogate.evaluate "$RUN" --data "$DATA" --split val --ckpt "$RUN/best.pt"
"$TRAIN_PY" -m surrogate.evaluate "$RUN" --data "$DATA" --split test --ckpt "$RUN/best.pt"
echo "$(date -Is): mean-base training and evaluation complete"
