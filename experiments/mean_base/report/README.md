# Mean-geometry final model report

The report uses `last.pt` from iteration 120,000. Training loss remains the
primary progress measure. Validation/test rollout error is a separate diagnostic.

```bash
# In the CUDA training environment (set OMP_NUM_THREADS=1 and OPENBLAS_NUM_THREADS=1).
python experiments/mean_base/report/evaluate.py --run surrogate_runs/mean_base \
    --data surrogate_data.mean_base --split val --batch 8 --checkpoint last.pt
python experiments/mean_base/report/evaluate.py --run surrogate_runs/mean_base \
    --data surrogate_data.mean_base --split test --batch 8 --checkpoint last.pt
python experiments/mean_base/report/evaluate.py --run surrogate_runs/mean_base \
    --data surrogate_data.mean_base --split train --limit 40 --seed 20261018 --batch 8
python experiments/mean_base/report/export_sample.py
```

`--start`, `--stop`, and `--tag` permit disjoint report parts. The actual report
used euclid-17 GPU 0/1 for the initial validation/test cases, euclid-10 GPU 0/1
for indices 64–119, and euclid-01 GPU 0 for the 40-case training subset. All use
float32 evaluation of the same final checkpoint. `merge.py` checks complete
validation/test ID coverage and consistency of overlapping RMSEs. It keeps a
copy of the original partial result tables. Do not merge while parts are writing.

```bash
python experiments/mean_base/report/merge.py surrogate_runs/mean_base/report \
    --manifest surrogate_data.mean_base/params.csv
cp surrogate_data.mean_base/_campaign/training.log surrogate_runs/mean_base/report/training.log
python experiments/mean_base/report/plots.py surrogate_runs/mean_base/report
python experiments/mean_base/report/architecture.py surrogate_runs/mean_base/report/assets
```

The 40 training cases are selected without replacement with seed 20261018.
The illustration is the first test case in the fixed manifest (`mean_p0012`),
chosen without looking at prediction errors. Full validation and test each have
120 cases. The training diagnostic is explicitly a subset, not a 760-case
training rollout assessment. Test results are evaluated after fitting and were
not used for checkpoint selection.

History plots use cumulative **active** elapsed time from the training log.
Elapsed-time resets at resumed runs are accumulated; downtime is excluded.
The physical-cycle error plot instead uses t/T because period is sampled.
Its bands show per-phase 10th–90th percentiles across cases. All displayed RMSE
curves use component-wise displacement error; vector-norm RMSE is sqrt(3) larger.

## PyVista images and GIFs

Use PyVista 0.48.4, VTK with offscreen rendering, NumPy, and Pillow 10 or newer.
No AI-generated imagery is used. The renderer extracts the tetrahedral boundary,
keeps its original point IDs, deforms it with solver/predicted nodal fields,
and writes PNG frames before GIF encoding. Displacement is shown at true scale,
with a fixed camera and fixed scalar limits over the whole cycle.

```bash
python experiments/mean_base/report/render.py \
    surrogate_runs/mean_base/report/illustrative_sample.npz \
    surrogate_runs/mean_base/report/assets --stride 2 --fps 20
```

Outputs: `simulation.gif`, `evaluation.gif`, PNG frame directories, snapshots,
and `animation.json`. There are 101 frames per GIF. The first prescribed cycle
begins undeformed and need not end in the initial state, so the GIF loop may
have a discontinuity. GIF colours show vector magnitudes in mm; the comparison
error is displayed on the solver geometry.

All generated assets, CSV reports, prediction arrays, and frame directories are
under ignored `surrogate_runs/`; only these reproduction scripts are committed.
