#!/bin/bash
# Real-LV case on a Dalton geometry: same method/parameters as Sec. 4.6
# (GLS P1/P1, mu_eff incl. fs, c_t=1.5, c_s=3, C_tau=0.03, C_c=0.1, same loads and BCs).
# Double-click, or: bash run_dalton.command [geom id, default 0000] [conda env, default lv0d3d]
cd "$(dirname "$0")"
G="${1:-0000}"; MESH="mesh/dalton_geom_$G"; OUT="results_dalton/geom_$G"
if ! command -v conda >/dev/null; then
  for d in "$HOME/miniconda3" "$HOME/miniforge3" "$HOME/anaconda3" "$HOME/opt/miniconda3" /opt/homebrew/Caskroom/miniconda/base /opt/miniconda3; do
    [ -x "$d/bin/conda" ] && export PATH="$d/bin:$PATH" && break
  done
fi
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${2:-lv0d3d}" || exit 1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
[ -f "$MESH/dalton_lv_mesh.xdmf" ] || bash mesh/download_dalton.command "$G" || exit 1
# Fibre geometry must be plausible (tangent to the wall, helix rotation endo->epi); FORCE=1 overrides.
if ! python tools/check_mesh.py "$MESH" --strict; then
  if [ "${FORCE:-0}" != "1" ]; then
    echo "ABORT: fibre field of $MESH is not consistent with the geometry (see above). Run with FORCE=1 to run anyway."
    [ -t 0 ] && read -r -p "Enter to close"; exit 1
  fi
  echo "FORCE=1: running despite the fibre check"
fi
rm -rf "$OUT"; mkdir -p "$OUT"; t0=$SECONDS
echo "running $OUT (log: $OUT/log.txt) ..."
python -m lvsolver.run --method gls --tau pdf --include-fs --mesh "$MESH" --out "$OUT" > "$OUT/log.txt" 2>&1
rc=$?; echo "solver exit=$rc after $((SECONDS-t0)) s"; tail -3 "$OUT/log.txt"
if [ $rc -eq 0 ]; then
  python -m lvsolver.verify "$OUT" > "$OUT/verify.txt" 2>&1; echo "verify exit=$?"
  python paper/sec46_lv/extract_data.py "$OUT" "$OUT/timeseries.csv" > "$OUT/summary.json" && cat "$OUT/summary.json"
  python paper/sec46_lv/make_figures.py "$OUT/timeseries.csv" "$OUT/figures" > /dev/null
fi
[ -t 0 ] && read -r -p "Done. Enter to close"
