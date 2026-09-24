#!/bin/bash
# Download geometries of huggingface.co/datasets/mapengfei/dalton-lv-meshes into mesh/dalton_geom_XXXX/
#   double-click           -> the 10 smoke-test geometries of the surrogate (0001-0008, 2400, 2401)
#   bash download_dalton.command 0007 2400 ...
# Geometries already present are skipped. Only mesh/dalton_geom_0000 is tracked by git.
cd "$(dirname "$0")"
IDS=("$@"); [ ${#IDS[@]} -gt 0 ] || IDS=(0001 0002 0003 0004 0005 0006 0007 0008 2400 2401)
BASE="https://huggingface.co/datasets/mapengfei/dalton-lv-meshes/resolve/main"
fail=0
for G in "${IDS[@]}"; do
  D="dalton_geom_$G"
  if [ -f "$D/dalton_lv_mesh.h5" ] && [ -f "$D/dalton_lv_mesh.xdmf" ]; then echo "$D: present"; continue; fi
  mkdir -p "$D"
  for f in dalton_lv_mesh.h5 dalton_lv_mesh.xdmf meta.json; do
    curl -fsSL --retry 3 -o "$D/$f.part" "$BASE/geom_$G/$f" && mv "$D/$f.part" "$D/$f" \
      || { echo "FAILED: geom_$G/$f"; fail=1; }
  done
  echo "$D: $(du -sh "$D" | cut -f1)"
done
[ $fail = 0 ] && echo "all downloaded" || echo "some downloads FAILED (re-run to retry)"
[ -t 0 ] && [ $# -eq 0 ] && read -r -p "Enter to close"
exit $fail
