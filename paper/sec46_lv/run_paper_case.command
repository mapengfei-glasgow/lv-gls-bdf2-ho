#!/bin/bash
# Sec. 4.6 case: GLS P1/P1 with mu_eff^HO of Eq. (43) incl. the fs term, on the h = 0.35 cm mesh.
# Double-click in Finder or: bash run_paper_case.command [conda_env]   (default lv0d3d)
cd "$(dirname "$0")/../.."
if ! command -v conda >/dev/null; then
  for d in "$HOME/miniconda3" "$HOME/miniforge3" "$HOME/anaconda3" "$HOME/opt/miniconda3" /opt/homebrew/Caskroom/miniconda/base /opt/miniconda3; do
    [ -x "$d/bin/conda" ] && export PATH="$d/bin:$PATH" && break
  done
fi
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${1:-lv0d3d}" || exit 1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
H=0.35; MESH="mesh/lv_plain_h$H"; OUT="results_paper/gls_pdf_fs_h$H"
if [ ! -f "$MESH/lv_aha17_v2_plain_mesh.xdmf" ]; then
  python -c "import gmsh" 2>/dev/null || conda install -y -c conda-forge gmsh python-gmsh || exit 1
  mkdir -p "$MESH"
  python tools/generate_mesh.py --mesh-size $H --xdmf "$MESH/lv_aha17_v2_plain_mesh.xdmf" --out "${TMPDIR:-/tmp}/lv_bin_h$H" > "$MESH/generate.log" 2>&1 || exit 1
fi
rm -rf "$OUT"; mkdir -p "$OUT"; t0=$SECONDS
echo "running $OUT ..."
python -m lvsolver.run --method gls --tau pdf --include-fs --mesh "$MESH" --out "$OUT" > "$OUT/log.txt" 2>&1 \
  && python -m lvsolver.verify "$OUT" > "$OUT/verify.txt" 2>&1 || { echo "FAILED, see $OUT/log.txt"; read -r -p "Enter"; exit 1; }
echo "solve + verify: $((SECONDS-t0)) s"
cd paper/sec46_lv && python extract_data.py "../../$OUT" && python make_figures.py
[ -t 0 ] && read -r -p "Done. Enter to close"
