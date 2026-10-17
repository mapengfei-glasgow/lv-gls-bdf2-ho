#!/usr/bin/env bash
# Generate one disjoint shard allocation; this script never starts training.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
cd "$ROOT"
WORKER=$1
JOBS=$2
shift 2
[ "$(hostname -s)" = "$WORKER" ] || exit 1
: "${SOLVER_PY:?Set SOLVER_PY}"
DATA=${DATA:-surrogate_data.geometry_narrow}
CONTROL="$DATA/_campaign"
mkdir -p "$CONTROL"
rm -f "$CONTROL/$WORKER.done" "$CONTROL/$WORKER.failed"
trap 'rc=$?; if [ "$rc" -eq 0 ]; then touch "$CONTROL/$WORKER.done"; else echo "$rc" > "$CONTROL/$WORKER.failed"; fi' EXIT
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
"$SOLVER_PY" -m surrogate.generate "$DATA/params.csv" "$DATA" \
    --mesh-root "$DATA/_geometry" --jobs "$JOBS" --steps 200 \
    --shards 5 --shard "$@" --status-file "status.$WORKER.tsv"
