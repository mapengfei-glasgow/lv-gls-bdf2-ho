#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
MINICONDA_DIR="${MINICONDA_PREFIX:-$PWD/miniconda3}"
GLS_ENV_DIR="${GLS_ENV_PREFIX:-$PWD/env_gls}"
if [ ! -x "$MINICONDA_DIR/bin/conda" ]; then
    curl -fL --connect-timeout 30 --retry 2 -o Miniconda3.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    bash Miniconda3.sh -b -p "$MINICONDA_DIR"
fi
"$MINICONDA_DIR/bin/conda" create -y -p "$GLS_ENV_DIR" --override-channels -c conda-forge \
    python=3.12 fenics-dolfinx=0.11.0 "petsc=*=real*" io4dolfinx=1.4.0 adios2=2.12.1 \
    mpi4py numpy scipy h5py matplotlib meshio=5.3.5
"$MINICONDA_DIR/bin/conda" run -p "$GLS_ENV_DIR" --no-capture-output python -c 'import dolfinx; print("DOLFINx", dolfinx.__version__)'
"$MINICONDA_DIR/bin/conda" run -p "$GLS_ENV_DIR" --no-capture-output bash run_gls.sh
