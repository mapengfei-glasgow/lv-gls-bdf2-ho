#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
# Linux x86_64; install only inside this project by default.
PREFIX="${MINICONDA_PREFIX:-$PWD/miniconda3}"
if [ ! -x "$PREFIX/bin/conda" ]; then
    curl -fL -o Miniconda3.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    bash Miniconda3.sh -b -p "$PREFIX"
fi
"$PREFIX/bin/conda" create -y -p "$PWD/env_lagrange" --override-channels -c conda-forge \
    python=3.12 fenics-dolfinx=0.11.0 "petsc=*=real*" io4dolfinx=1.4.0 adios2=2.12.1 \
    mpi4py numpy scipy h5py matplotlib meshio=5.3.5
"$PREFIX/bin/conda" run -p "$PWD/env_lagrange" --no-capture-output bash run_lagrange.sh
