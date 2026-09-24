#!/bin/bash
# Surrogate smoke test, end to end: 10 Dalton geometries x Sobol parameters -> solver data -> training -> evaluation.
#   double-click, or: bash surrogate/run_smoke.command [JOBS=4] [ITERS=300]
# Needs conda envs lv0d3d (environment.yml) and lvml (environment_ml.yml; created here if missing).
cd "$(dirname "$0")/.."
JOBS="${1:-4}"; ITERS="${2:-300}"
if ! command -v conda >/dev/null; then
  for d in "$HOME/miniconda3" "$HOME/miniforge3" "$HOME/anaconda3" "$HOME/opt/miniconda3" /opt/homebrew/Caskroom/miniconda/base /opt/miniconda3; do
    [ -x "$d/bin/conda" ] && export PATH="$d/bin:$PATH" && break
  done
fi
source "$(conda info --base)/etc/profile.d/conda.sh"
set -e
bash mesh/download_dalton.command 0001 0002 0003 0004 0005 0006 0007 0008 2400 2401
conda activate lv0d3d
[ -f surrogate_data/params.csv ] || python -m surrogate.sample_params --ids 0001-0008 2400 2401 --out surrogate_data/params.csv
python -m surrogate.generate surrogate_data/params.csv surrogate_data --jobs "$JOBS" || echo "some cases failed: see surrogate_data/status.tsv"
conda env list | grep -q "^lvml " || conda env create -f environment_ml.yml
conda activate lvml
python -m surrogate.train surrogate_data --out surrogate_runs/smoke --iters "$ITERS" --latent 64 --mp-steps 6 \
    --batch 2 --eval-every 100 --norm-iters 100
python -m surrogate.evaluate surrogate_runs/smoke --vtk 1
[ -t 0 ] && read -r -p "Done. Enter to close"
