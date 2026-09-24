#!/usr/bin/env bash
# Batch pipeline for the Dalton LV meshes (huggingface.co/datasets/mapengfei/dalton-lv-meshes).
#
#   bash tools/batch_dalton.sh DATA_DIR OUT_DIR JOBS [ID ...]
#
# DATA_DIR  local copy of the dataset (DATA_DIR/geom_XXXX/dalton_lv_mesh.{xdmf,h5})
# OUT_DIR   results; one folder per geometry, plus OUT_DIR/status.tsv
# JOBS      geometries solved in parallel (each job = 1 MPI rank, 1 thread)
# ID        geometry ids (e.g. 0000 0007 2400); default: every geom_* in DATA_DIR
#
# Per geometry: rule-based fibres -> tools/check_mesh.py --strict -> lvsolver.run (Sec. 4.6
# method: GLS P1/P1, mu_eff incl. fs, C_c = 0.1) -> verify -> extract_data (timeseries.csv,
# summary.json). Finished geometries (summary.json present) are skipped, so the script can be
# re-run after an interruption. Environment variables:
#   KEEP_VTK=1        keep the VTU frames (~1.5 GB per geometry); default: delete after extraction
#   KEEP_MESH=1       keep the rule-based-fibre mesh copy (~8.7 MB); default: delete
#   EXTRA_ARGS="..."  extra options for lvsolver.run (e.g. "--steps 3" for a smoke test)
# Activate the conda environment (environment.yml) before calling this script.
set -uo pipefail
[ $# -ge 3 ] || { sed -n '2,24p' "$0"; exit 2; }
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA="$(cd "$1" && pwd)"; OUT="$2"; JOBS="$3"; shift 3
mkdir -p "$OUT"; OUT="$(cd "$OUT" && pwd)"
export ROOT DATA OUT OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export KEEP_VTK="${KEEP_VTK:-0}" KEEP_MESH="${KEEP_MESH:-0}" EXTRA_ARGS="${EXTRA_ARGS:-}"
[ -f "$OUT/status.tsv" ] || printf "geom\tstatus\tseconds\tnote\n" > "$OUT/status.tsv"
if [ $# -gt 0 ]; then IDS=("$@"); else IDS=($(ls "$DATA" | sed -n 's/^geom_\([0-9]*\)$/\1/p')); fi

one() {
  local g="$1" d="$OUT/geom_$1" t0=$SECONDS note="" st
  if [ -f "$d/summary.json" ]; then return 0; fi
  rm -rf "$d"; mkdir -p "$d"
  local mesh="$d/mesh_rb"
  log() { printf "%s\t%s\t%s\t%s\n" "$g" "$1" "$((SECONDS-t0))" "$2" >> "$OUT/status.tsv"; }
  if ! python "$ROOT/tools/rule_based_fibers.py" "$DATA/geom_$g" "$mesh" > "$d/fibres.log" 2>&1; then
    log fibres_failed "see fibres.log"; return 0; fi
  if ! python "$ROOT/tools/check_mesh.py" "$mesh" --strict > "$d/check.txt" 2>&1; then
    log check_failed "see check.txt"; return 0; fi
  ( cd "$ROOT" && python -m lvsolver.run --method gls --tau pdf --include-fs --mesh "$mesh" \
      --out "$d" $EXTRA_ARGS ) > "$d/log.txt" 2>&1
  st=$?
  if [ $st -ne 0 ]; then
    note="$(grep -m1 -oE 'Step [0-9]+ failed[^,]*|nonpositive[^\n]*|Error[^\n]*' "$d/log.txt" | head -c 120)"
    log solve_failed "${note:-see log.txt}"
  else
    ( cd "$ROOT" && python -m lvsolver.verify "$d" ) > "$d/verify.txt" 2>&1 || note="verify failed"
    if ( cd "$ROOT" && python paper/sec46_lv/extract_data.py "$d" "$d/timeseries.csv" ) > "$d/summary.json.tmp" 2>&1; then
      mv "$d/summary.json.tmp" "$d/summary.json"; log ok "$note"
    else
      mv "$d/summary.json.tmp" "$d/extract.log"; log extract_failed "see extract.log"
    fi
  fi
  [ "$KEEP_VTK" = 1 ] || rm -rf "$d/vtk"
  [ "$KEEP_MESH" = 1 ] || rm -rf "$mesh"
}
export -f one
printf "%s\n" "${IDS[@]}" | xargs -P "$JOBS" -I{} bash -c 'one "$@"' _ {}
echo "done: $(tail -n +2 "$OUT/status.tsv" | cut -f2 | sort | uniq -c | tr '\n' ' ')"
