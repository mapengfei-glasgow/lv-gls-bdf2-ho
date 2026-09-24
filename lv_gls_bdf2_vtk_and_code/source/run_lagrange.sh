#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
ENV_PREFIX="${CONDA_PREFIX:-$(python -c 'import sys; print(sys.prefix)')}"
export LD_LIBRARY_PATH="$ENV_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
python run_lagrange_vtk_cycle.py
python - <<'PY'
import os
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
root = Path(os.environ.get('SIMPLE_3D0D_OUTPUT_DIR', 'out_lagrange_full_cycle'))
with ZipFile(root.with_suffix('.zip'), 'w', ZIP_DEFLATED) as z:
    for p in sorted(root.rglob('*')):
        if p.is_file():
            z.write(p, p.relative_to(root))
print(root.with_suffix('.zip').resolve())
PY
