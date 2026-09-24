#!/usr/bin/env bash
# Run P1 / Lagrange / GLS(HO-PDF) full-cycle programs in one conda env.
# Usage: bash run_all.sh [conda_env_name]   (default: lv0d3d; created from environment.yml if missing)
set -uo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
ENV_NAME="${1:-lv0d3d}"
source "$(conda info --base)/etc/profile.d/conda.sh"
if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo ">>> creating env $ENV_NAME"
  conda env create -n "$ENV_NAME" -f "$ROOT/ho_pdf_revision/source/environment.yml" || exit 1
fi
conda activate "$ENV_NAME"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python -c "import sys,dolfinx,ufl,meshio,io4dolfinx; from petsc4py import PETSc; print('ENV', sys.version.split()[0], 'dolfinx', dolfinx.__version__, 'ufl', ufl.__version__, 'scalar', PETSc.ScalarType)" || exit 1
STAMP="$(date +%Y%m%d_%H%M%S)"; RUNS="$ROOT/runs/$STAMP"; mkdir -p "$RUNS"
run() {  # name dir script
  local t0=$SECONDS
  echo ">>> [$1] start $(date +%T)"
  ( cd "$2" && SIMPLE_3D0D_OUTPUT_DIR="$RUNS/$1" bash "$3" ) > "$RUNS/$1.log" 2>&1
  local rc=$?
  echo ">>> [$1] exit=$rc  $((SECONDS-t0)) s" | tee -a "$RUNS/summary.txt"
}
run p1_plain  "$ROOT/p1_plain_full_cycle_program"      run.sh
run lagrange  "$ROOT/lv_lagrange_vtk_and_code/source"  run_lagrange.sh
run gls_hopdf "$ROOT/ho_pdf_revision/source"           run_gls.sh
( cd "$ROOT/ho_pdf_revision/source" && python verify_gls_results.py "$RUNS/gls_hopdf" ) > "$RUNS/gls_hopdf_verify.log" 2>&1
echo ">>> [gls verify] exit=$?" | tee -a "$RUNS/summary.txt"
echo ">>> ALL DONE -> $RUNS"
