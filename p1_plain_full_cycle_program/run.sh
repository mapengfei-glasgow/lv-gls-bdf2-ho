#!/bin/bash
set -e
cd "$(dirname "$0")"
export SIMPLE_3D0D_MESH_SCALE=0.01
export SIMPLE_3D0D_LONG_AXIS=x
export SIMPLE_3D0D_ENDO_TAG=6
export SIMPLE_3D0D_EPI_TAG=7
export SIMPLE_3D0D_BASE_TAG=5
export SIMPLE_3D0D_OUTPUT_DIR="${SIMPLE_3D0D_OUTPUT_DIR:-$PWD/output}"
if ! python -c "import meshio" 2>/dev/null; then
  echo "meshio is missing. Install it with: pip install -r requirements-extra.txt" >&2
  exit 1
fi
python run_plain_p1_vtk_cycle.py
