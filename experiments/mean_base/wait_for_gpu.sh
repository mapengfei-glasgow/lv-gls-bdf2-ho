#!/usr/bin/env bash
# Wait for other users' compute jobs to finish on the selected GPU.
set -euo pipefail
GPU=${CUDA_VISIBLE_DEVICES:-0}
while true; do
    processes=$(nvidia-smi --id="$GPU" --query-compute-apps=pid --format=csv,noheader,nounits)
    [ -z "$processes" ] && break
    echo "$(date -Is): waiting for GPU $GPU; active compute PIDs: $(echo "$processes" | tr '\n' ' ')"
    sleep 60
done
echo "$(date -Is): GPU $GPU is available"
