#!/bin/bash
# Mesh-refinement study of gls --tau pdf (C_c = 0.1): generate meshes, run, verify, summarise.
# Double-click in Finder or:  bash convergence.command [conda_env]   (default lv0d3d)
cd "$(dirname "$0")"
if ! command -v conda >/dev/null; then
  for d in "$HOME/miniconda3" "$HOME/miniforge3" "$HOME/anaconda3" "$HOME/opt/miniconda3" /opt/homebrew/Caskroom/miniconda/base /opt/miniconda3; do
    [ -x "$d/bin/conda" ] && export PATH="$d/bin:$PATH" && break
  done
fi
command -v conda >/dev/null || { echo "conda not found"; read -r -p "Enter to close"; exit 1; }
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${1:-lv0d3d}" || exit 1
python -c "import gmsh" 2>/dev/null || conda install -y -c conda-forge gmsh python-gmsh || exit 1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1

SIZES=(1.5 1.0 0.7 0.5 0.35)          # gmsh characteristic length, cm
CASE="gls_pdf"; ARGS="--method gls --tau pdf --cc 0.1"
OUT=results_convergence; mkdir -p "$OUT"
for h in "${SIZES[@]}"; do
  mdir="mesh/lv_plain_h$h"; run="$OUT/${CASE}_h$h"; t0=$SECONDS
  if [ ! -f "$mdir/lv_aha17_v2_plain_mesh.xdmf" ]; then
    mkdir -p "$mdir"
    python tools/generate_mesh.py --mesh-size "$h" --xdmf "$mdir/lv_aha17_v2_plain_mesh.xdmf" \
      --out "${TMPDIR:-/tmp}/lv_bin_h$h" > "$mdir/generate.log" 2>&1 || { echo "[h=$h] mesh generation FAILED (see $mdir/generate.log)"; continue; }
  fi
  rm -rf "$run"; mkdir -p "$run"
  python -m lvsolver.run $ARGS --mesh "$mdir" --out "$run" > "$run/log.txt" 2>&1 \
    && python -m lvsolver.verify "$run" > "$run/verify.txt" 2>&1
  echo "[h=$h] exit=$? $((SECONDS-t0))s  $(grep -m1 '\[Mesh\]' "$run/log.txt")"
done
python tools/convergence.py "$OUT"
[ -t 0 ] && read -r -p "Done. Enter to close"
