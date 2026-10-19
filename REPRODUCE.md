# Reproduce from a clean clone

This branch contains the solver, emulator, PCA basis, mesh topology and rule-based
fibre generator. No other repository or external geometry dataset is required.
Generated meshes, simulation data, checkpoints and report outputs are excluded
from Git; generate them on the destination machine.

## Clone and install

```bash
git clone --branch restructure-repository --single-branch \
    https://github.com/mapengfei-glasgow/lv-gls-bdf2-ho.git
cd lv-gls-bdf2-ho
conda env create -f environment.yml
conda env create -f environment_ml.yml
```

The solver environment is `lv0d3d` (DOLFINx 0.11.0 and real-valued PETSc); the
training environment is `lvml` (PyTorch and numerical/plotting packages). Linux
and macOS have been used. CUDA training requires a compatible NVIDIA driver and
CUDA-enabled PyTorch installation. The base trainer otherwise uses CPU; it does
not select Apple MPS. Installing these dependencies requires package downloads,
but does not require cloning another project.

For geometry generation alone, a Python environment with
`pip install -r geometry/requirements.txt` is sufficient. PyVista/GIF rendering
additionally needs `experiments/mean_base/report/render-requirements.txt` and
working offscreen VTK support.

## Small checks before a large campaign

```bash
conda run -n lvml python -m unittest discover -s tests -v
conda run -n lvml python geometry/generate.py --out geometry/generated/mean
conda run -n lvml python tools/check_mesh.py geometry/generated/mean --strict
conda run -n lv0d3d python -m lvsolver.run \
    --mesh geometry/generated/mean --method gls --tau pdf --include-fs \
    --steps 2 --no-vtk --out results/pca_mean_smoke
```

These checks do not train a useful model or establish full-cycle solver accuracy.
Geometry output directories must be empty/new. By default mesh coordinates are
converted from centimetres to metres by the solver.

## Mean-geometry data and training

Use absolute interpreter paths from your own environments:

```bash
export SOLVER_PY="$(conda run -n lv0d3d python -c 'import sys; print(sys.executable)')"
export TRAIN_PY="$(conda run -n lvml python -c 'import sys; print(sys.executable)')"
mkdir -p surrogate_data.mean_base/_campaign
JOBS=4 CUDA_VISIBLE_DEVICES=0 bash experiments/mean_base/run.sh \
    > surrogate_data.mean_base/_campaign/training.log 2>&1
```

This is a substantial calculation: 1,000 simulations with 200 time steps, followed
by 120,000 training iterations. It creates `surrogate_data.mean_base/` and
`surrogate_runs/mean_base/`. Re-running skips completed simulations and resumes
training with the original settings. See [the campaign guide](experiments/mean_base/README.md).

To inspect data preparation on a smaller design, choose a separate output root:

```bash
conda run -n lvml python experiments/mean_base/prepare.py \
    --data surrogate_data.mean_smoke --cases 10
```

This generates geometry and parameters, not simulation trajectories. To generate
those trajectories, use `python -m surrogate.generate` in the solver environment
with that manifest and its `_geometry/` directory as `--mesh-root`.

## Narrow-geometry curriculum

After the mean campaign completes:

```bash
conda run -n lv0d3d python experiments/geometry_narrow/prepare.py
conda run -n lv0d3d python -m surrogate.generate \
    surrogate_data.geometry_narrow/params.csv surrogate_data.geometry_narrow \
    --mesh-root surrogate_data.geometry_narrow/_geometry --jobs 4
```

The design has 600 cases (384 training / 102 validation / 114 testing). Completed
matching mean trajectories are reused; missing trajectories are generated.
The validator checks every case. The training launcher requires the campaign's
`validated.done` sentinel; on a single machine, create it only after successful
validation:

```bash
mkdir -p surrogate_data.geometry_narrow/_campaign
conda run -n lv0d3d python experiments/geometry_narrow/validate.py surrogate_data.geometry_narrow \
    && touch surrogate_data.geometry_narrow/_campaign/validated.done
```

The mean command above records its log in the location used by the narrow
launcher to rank older checkpoints without an embedded training-loss value.
New checkpoints also store training loss directly.

Then start a separate curriculum run:

```bash
CUDA_VISIBLE_DEVICES=0 bash experiments/geometry_narrow/train.sh
```

The narrow launcher requires CUDA and selects the saved base snapshot with the
lowest recorded training loss. It refuses to overwrite an existing narrow run;
base checkpoints are read only. See [the narrow campaign guide](experiments/geometry_narrow/README.md)
for retries, provenance and resuming. Euclid inventories and worker launchers are
optional deployment helpers; another machine does not need Euclid access.

## Joint wider-geometry data

The next data-only stage samples geometry and physical parameters together,
using 600 distinct 24-dimensional Sobol pairs. Geometry varies over ±0.5σ on
the first 16 PCs; physical ranges are unchanged. It needs no existing data.
See [the joint campaign guide](experiments/geometry_joint/README.md) for commands,
strict geometry checks, CPU workers and independent validation.

## Solver benchmarks and reports

`run_all.command` and `convergence.command` remain supported Bash entry points for
the solver benchmark and mesh-refinement study. They use bundled tools and meshes.
Historical result tables and paper sources are not included. Comparisons against
missing Git reference results print `NO REF`; they do not establish regression
agreement. These scripts replace their own named benchmark outputs when rerun.

The obsolete external-dataset download, Dalton-paper runner and old surrogate
smoke launcher have been removed. Use the PCA workflows above.

Model evaluation, error plots and PyVista/GIF reproduction are described in
[the report guide](experiments/mean_base/report/README.md). Arrays, trained weights
and simulation results must be generated before those reports can run.
