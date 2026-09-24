#!/usr/bin/env bash
# Regenerate every result/log of the package in its original location.
# Usage: bash run_all.sh [conda_env_name]   (default lv0d3d; created from environment.yml if missing)
set -uo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"; cd "$ROOT"
ENV_NAME="${1:-lv0d3d}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda env list | awk '{print $1}' | grep -qx "$ENV_NAME" || conda env create -n "$ENV_NAME" -f ho_pdf_revision/source/environment.yml || exit 1
conda activate "$ENV_NAME"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
SUMMARY="$ROOT/run_summary.txt"; : > "$SUMMARY"
python -c "import sys,dolfinx,ufl,basix,numpy,scipy,meshio; from petsc4py import PETSc; print('env', sys.version.split()[0], 'dolfinx', dolfinx.__version__, 'ufl', ufl.__version__, 'petsc', PETSc.Sys.getVersion(), PETSc.ScalarType.__name__)" | tee -a "$SUMMARY" || exit 1

# job NAME LOGFILE WORKDIR [VAR=val ...] -- CMD...
job() {
  local name=$1 log=$2 dir=$3; shift 3; local vars=()
  while [ "$1" != "--" ]; do vars+=("$1"); shift; done; shift
  mkdir -p "$(dirname "$log")"; local t0=$SECONDS
  echo ">>> [$name] start $(date +%T)"
  ( cd "$dir" && env ${vars[@]+"${vars[@]}"} "$@" ) > "$log" 2>&1
  local rc=$?; echo "[$name] exit=$rc $((SECONDS-t0))s" | tee -a "$SUMMARY"
}

# 1) P1 plain (penalty) full cycle
P1=p1_plain_full_cycle_program
job p1_plain "$P1/reference_output/run.log" "$P1" SIMPLE_3D0D_OUTPUT_DIR="$ROOT/$P1/reference_output" -- bash run.sh
for d in ho_pdf_revision lv_gls_bdf2_vtk_and_code lv_lagrange_vtk_and_code; do
  mkdir -p "$d/source/reference_output"; cp "$P1/reference_output/monitor_plain_p1_full_cycle.csv" "$d/source/reference_output/"
done

# 2) Lagrange P2/P1: augmented (beta=1e6, delivered) + pure multiplier (beta=0)
LG=lv_lagrange_vtk_and_code
job lagrange_aug  "$LG/logs/augmented_cycle.log"    "$LG/source" SIMPLE_3D0D_OUTPUT_DIR="$ROOT/$LG" -- python run_lagrange_vtk_cycle.py
job lagrange_pure "$LG/logs/pure_lagrange_cycle.log" "$LG/source" SIMPLE_3D0D_OUTPUT_DIR="$ROOT/$LG/diagnostics/pure_lagrange" SIMPLE_3D0D_LM_AUGMENTATION=0 -- python run_lagrange_vtk_cycle.py
python tools/repro_tools.py lagrange "$LG" "$LG/logs/augmented_cycle.log" "$LG/diagnostics/pure_lagrange" "$LG/logs/pure_lagrange_cycle.log" "$LG"

# 3) Original GLS BDF2: full cycle, 3-step smoke, 3-step no-stabilisation control
GL=lv_gls_bdf2_vtk_and_code
job gls_full   "$GL/logs/gls_full_cycle.log"     "$GL/source" SIMPLE_3D0D_OUTPUT_DIR="$ROOT/$GL" -- bash run_gls.sh
job gls_verify "$GL/logs/gls_full_cycle_validation.log" "$GL/source" -- python verify_gls_results.py "$ROOT/$GL"
job gls_smoke  "$GL/logs/gls_smoke.log"          "$GL/source" SIMPLE_3D0D_STEPS=3 SIMPLE_3D0D_OUTPUT_DIR="$ROOT/$GL/diagnostics/gls_smoke" -- bash run_gls.sh
job gls_nostab "$GL/logs/gls_control_nostab.log" "$GL/source" SIMPLE_3D0D_STEPS=3 LV_GLS_MODE=none_p1p1 SIMPLE_3D0D_OUTPUT_DIR="$ROOT/$GL/diagnostics/no_stabilization_short" -- bash run_gls.sh
job gls_nostab_verify "$GL/logs/gls_control_nostab_validation.log" "$GL/source" -- python verify_gls_results.py "$ROOT/$GL/diagnostics/no_stabilization_short"
python tools/repro_tools.py glsshort "$GL/diagnostics/gls_smoke" "$GL/diagnostics/no_stabilization_short" "$GL/short_control_comparison.json"

# 4) HO-PDF GLS revision: smoke, smoke with fs stiffness, full cycle (+ verification, plot)
HO=ho_pdf_revision; R="$ROOT/$HO/results"
job ho_smoke     "$HO/logs/smoke.log"                "$HO/source" SIMPLE_3D0D_STEPS=3 SIMPLE_3D0D_OUTPUT_DIR="$R/smoke" -- bash run_gls.sh
job ho_smoke_v   "$HO/logs/smoke_validation.log"     "$HO/source" -- python verify_gls_results.py "$R/smoke"
job ho_smoke_fs  "$HO/logs/smoke_fs.log"             "$HO/source" SIMPLE_3D0D_STEPS=3 LV_GLS_INCLUDE_FS=1 SIMPLE_3D0D_OUTPUT_DIR="$R/smoke_fs" -- bash run_gls.sh
job ho_smoke_fs_v "$HO/logs/smoke_fs_validation.log" "$HO/source" -- python verify_gls_results.py "$R/smoke_fs"
job ho_full      "$HO/logs/full_cycle.log"           "$HO/source" SIMPLE_3D0D_OUTPUT_DIR="$R/full_cycle" -- bash run_gls.sh
job ho_full_v    "$HO/logs/full_cycle_validation.log" "$HO/source" -- python verify_gls_results.py "$R/full_cycle"
job ho_plot      "$HO/logs/plot_results.log"         "$HO" MPLBACKEND=Agg -- python plot_results.py
conda list --explicit > "$HO/logs/environment_actual_explicit.txt"
python -c "import dolfinx,ufl,basix,numpy,scipy,meshio; print({'dolfinx':dolfinx.__version__,'ufl':ufl.__version__,'basix':basix.__version__,'numpy':numpy.__version__,'scipy':scipy.__version__,'meshio':meshio.__version__})" > "$HO/logs/versions_actual.txt"
echo ">>> ALL DONE"
