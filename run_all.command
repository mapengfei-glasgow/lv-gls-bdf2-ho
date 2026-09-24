#!/bin/bash
# Run every case into results/, verify each one, write results/summary.md and
# compare with the last committed results.  Double-click in Finder or:
#   bash run_all.command [conda_env]        (default env lv0d3d)
cd "$(dirname "$0")"
if ! command -v conda >/dev/null; then
  for d in "$HOME/miniconda3" "$HOME/miniforge3" "$HOME/anaconda3" "$HOME/opt/miniconda3" /opt/homebrew/Caskroom/miniconda/base /opt/miniconda3; do
    [ -x "$d/bin/conda" ] && export PATH="$d/bin:$PATH" && break
  done
fi
command -v conda >/dev/null || { echo "conda not found"; read -r -p "Enter to close"; exit 1; }
source "$(conda info --base)/etc/profile.d/conda.sh"
ENV_NAME="${1:-lv0d3d}"
conda env list | awk '{print $1}' | grep -qx "$ENV_NAME" || conda env create -n "$ENV_NAME" -f environment.yml || exit 1
conda activate "$ENV_NAME"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1

# case name | arguments of python -m lvsolver.run
CASES=(
  "p1|--method p1"
  "lagrange|--method lagrange --beta 1e6"
  "lagrange_pure|--method lagrange --beta 0"
  "gls_pdf|--method gls --tau pdf"
  "gls_pdf_fs_3steps|--method gls --tau pdf --include-fs --steps 3"
  "gls_reference|--method gls --tau reference"
  "gls_reference_nostab_3steps|--method gls --tau reference --no-stab --steps 3"
)
status=0
for entry in "${CASES[@]}"; do
  name="${entry%%|*}"; opts="${entry#*|}"; out="results/$name"
  rm -rf "$out"; mkdir -p "$out"; t0=$SECONDS
  python -m lvsolver.run $opts --out "$out" > "$out/log.txt" 2>&1 \
    && python -m lvsolver.verify "$out" > "$out/verify.txt" 2>&1
  rc=$?; [ $rc -ne 0 ] && status=1
  echo "[$name] exit=$rc $((SECONDS-t0))s"
done
python -m lvsolver.verify --report results > /dev/null
python tools/compare_runs.py results --ref-git HEAD || status=1
echo "overall status: $status"
[ -t 0 ] && read -r -p "Done. Enter to close"
exit $status
