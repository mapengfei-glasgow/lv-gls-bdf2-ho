# Base model on the PCA mean geometry

This campaign replaces geom_0000 as the fixed geometry for the first curriculum
stage. It generates new simulations and trains **from scratch**, with no old
geom_0000 checkpoint or simulation reuse. Existing experiments are left intact.

The geometry is the bundled PCA mean: all 40 scores are zero. This is the mean
in the raw PCA coordinate frame, followed by basal translation and the usual
solver-frame rotation. Centroid interpolation and rule-based fibres use the
self-contained `geometry/` generator.

## Design

- 1,000 scrambled Sobol parameter samples (seed 42), covering the full eight
  parameter ranges in `surrogate/params.py`.
- A shuffled, disjoint split (seed 43): 760 train / 120 validation / 120 test.
  Geometry is fixed; validation and test measure generalisation to parameters.
- GLS P1/P1, PDF stabilisation, fs stiffness included, 200 steps per cardiac
  cycle, with `dt = period_s / 200`.
- Reference training recipe: latent 128, 10 message-passing steps, batch 4,
  Adam, 120,000 iterations, cosine learning rate 1e-3 to 1e-6, position noise
  5e-5 m with corrected targets, no pushforward loss, running normalisers.
- Checkpoint selection uses the first ten validation cases. Final evaluation
  uses all validation and test cases with the selected `best.pt`.

The new trainer, `python -m surrogate.train_base`, saves optimizer/scheduler,
normalisers, graph, raw weights, optional EMA weights, and RNG states. `--resume`
continues the same dataset and schedule. It does not accept an old model as a
warm start. It refuses to overwrite an existing run without `--resume`.

## Run on one machine

Use a DOLFINx 0.11 solver environment (`environment.yml`) and a CUDA-enabled
PyTorch environment (`environment_ml.yml`). Set their Python paths explicitly:

```bash
SOLVER_PY="$HOME/miniconda3/envs/lv0d3d/bin/python" \
TRAIN_PY="$HOME/.venvs/lv-mean-base/bin/python" \
JOBS=8 CUDA_VISIBLE_DEVICES=0 bash experiments/mean_base/run.sh \
    > mean_base.log 2>&1
```

The script prepares the mean geometry and full manifest, generates all solver
cases, validates every case, trains, and evaluates. It stops on any generation
failure so missing cases cannot silently change the training distribution.
Re-running skips completed simulations and resumes training from `last.pt`.
Keep the same training settings when resuming.

Default outputs (ignored by Git): `surrogate_data.mean_base/` and
`surrogate_runs/mean_base/`. `design.json` records parameter ranges, seeds,
splits, and hashes of the PCA asset and parameter manifest. The output names
are separate from historical geom_0000 experiments.

## Distribute generation later

Prepare the same design on each machine, then assign a different shard to each:

```bash
python experiments/mean_base/prepare.py
python -m surrogate.generate surrogate_data.mean_base/params.csv \
    surrogate_data.mean_base --mesh-root surrogate_data.mean_base/_geometry \
    --jobs 8 --shards 2 --shard 0  # use --shard 1 on the other machine
```

Each machine retains the full manifest but solves only its disjoint row subset.
Use separate local output directories, then copy completed case directories
into the training machine's dataset. Do not replace its full `params.csv` with
a subset. Before training, require every planned case to be present:

```bash
python experiments/mean_base/validate.py surrogate_data.mean_base
```

Validation checks completion, solver configuration, parameter agreement,
geometry, topology, trajectory dimensions, and finite displacements. Generated
meshes, simulation fields, and trained checkpoints are not committed.

## Current Euclid deployment

The source lives in the shared home directory at
`~/lv-gls-bdf2-ho-mean-base`. The allocation is recorded in `machines.json`:

| Machine | CPU jobs | Cases | Shards of 10 |
|---|---:|---:|---|
| euclid-17 | 8 | 200 | 0, 1 |
| euclid-01 | 8 | 200 | 2, 3 |
| euclid-10 | 4 | 100 | 4 |
| euclid-21 | 8 | 200 | 5, 6 |
| euclid-23 | 12 | 300 | 7, 8, 9 |

Euclid-17 is explicitly permitted by the user; the other four machines are
public in `experiments/euclids.json`. All write disjoint case directories in
the same shared dataset. Each worker has a separate status TSV and log.
`worker.sh HOST JOBS SHARD...` runs one allocation; `finish.sh` waits for all
five workers, validates the complete data, and trains on euclid-17 GPU 0.
Training waits until the selected GPU has no other compute processes. A full-size
forward/backward preflight passed on the A6000 at batch 4, with about 31 GiB peak
reserved GPU memory.
Both use the same `SOLVER_PY`/`TRAIN_PY` environment variables as `run.sh`.
Worker processes are launched under per-host `flock` locks to prevent duplicates.

Runtime control files are under `surrogate_data.mean_base/_campaign/`:
`euclid-XX.log`, `.pid`, `.done` or `.failed`, and `training.log`/`training.pid`.
Read `training.log` for the global state and each worker's log for generation
failures. If a worker fails, rerun that worker under its lock; completed cases
are retained. Then restart `finish.sh` under `training.lock` if it exited.
Do not run the single-machine `run.sh` concurrently with these workers.

The resumable checkpoint regression test and shard coverage test can be run
with `python -m unittest discover -s tests -v` in the ML environment.
