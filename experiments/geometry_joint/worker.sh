#!/usr/bin/env bash
# Data generation only. Each worker owns disjoint shards in a shared checkout.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
cd "$ROOT"
WORKER=$1; JOBS=$2; shift 2
[ "$(hostname -s)" = "$WORKER" ] || exit 1
: "${SOLVER_PY:?Set SOLVER_PY}"
DATA=surrogate_data.geometry_joint
CONTROL="$DATA/_campaign"
mkdir -p "$CONTROL"
rm -f "$CONTROL/$WORKER.done" "$CONTROL/$WORKER.failed"
trap 'rc=$?; if [ "$rc" -eq 0 ]; then touch "$CONTROL/$WORKER.done"; else echo "$rc" > "$CONTROL/$WORKER.failed"; fi' EXIT
while [ ! -f "$CONTROL/prepared.done" ]; do
    [ ! -f "$CONTROL/prepare.failed" ] || { echo 'Geometry preparation failed'; exit 1; }
    sleep 15
done
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
"$SOLVER_PY" -m surrogate.generate "$DATA/params.csv" "$DATA" \
    --mesh-root "$DATA/_geometry" --jobs "$JOBS" --steps 200 \
    --shards 5 --shard "$@" --status-file "status.$WORKER.tsv"
