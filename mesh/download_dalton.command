#!/bin/bash
# Download one geometry of huggingface.co/datasets/mapengfei/dalton-lv-meshes into mesh/dalton_geom_XXXX/
# Double-click (geom_0000) or: bash download_dalton.command 0007
cd "$(dirname "$0")"
G="${1:-0000}"; D="dalton_geom_$G"; URL="https://huggingface.co/datasets/mapengfei/dalton-lv-meshes/resolve/main/geom_$G"
mkdir -p "$D"
for f in dalton_lv_mesh.h5 dalton_lv_mesh.xdmf meta.json; do
  echo "downloading $f"; curl -fL --retry 2 -o "$D/$f" "$URL/$f" || { echo "FAILED: $f"; read -r -p "Enter"; exit 1; }
done
ls -l "$D"
[ -t 0 ] && read -r -p "Done. Enter to close"
