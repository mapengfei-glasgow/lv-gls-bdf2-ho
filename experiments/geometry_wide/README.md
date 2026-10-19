# Wider geometry data: ±0.5 sigma, first 16 PCA modes

**Superseded:** the user requested joint geometry/physics sampling instead of
this crossed design. Its workers have been stopped and outputs preserved. Use
[the joint campaign](../geometry_joint/README.md) for the current workflow.

This is a **data-generation stage**, independent of the running narrow model.
Material/load/cycle parameter vectors and their ranges are unchanged. It expands
geometry coefficient bounds from ±0.25σ to ±0.5σ on the first 16 modes; the other
24 coefficients stay zero. This is an intermediate curriculum stage, not yet
full 40-mode geometry variation.

- Keep all 30 narrow shapes and their 600 completed trajectories through relative
  symlinks; the source dataset is read only.
- Draw 30 additional shapes with seed 20261019. Candidates are uniform in the
  wider coefficient box, outside the old box, conditioned on positive tetrahedral
  volumes. Rejection count and actual coefficients are recorded in `design.json`.
- Cross each new shape with the same 20 physical parameter vectors. Original
  shape/parameter split memberships are preserved. New shapes have 24 train /
  3 validation / 3 test assignments.
- Combined dataset: **1,200 cases**, 60 shapes (48/6/6), split into **768 training /
  204 validation / 228 testing cases**. Only 600 new finite-element simulations
  are needed. Keep test cases out of future training and method selection.
- Solver, constitutive model, pressure/activation waveforms, stabilisation,
  200-step grid, dt=T/200 and tolerances match the previous campaign.

```bash
# Requires the completed narrow dataset. Source paths can be set explicitly.
python experiments/geometry_wide/prepare.py
python experiments/geometry_wide/preflight.py
# In the DOLFINx environment, for a single machine:
python -m surrogate.generate surrogate_data.geometry_wide/params.csv \
    surrogate_data.geometry_wide --mesh-root surrogate_data.geometry_wide/_geometry \
    --jobs 4 --steps 200
python experiments/geometry_narrow/validate.py surrogate_data.geometry_wide
```

Default output: ignored `surrogate_data.geometry_wide/`. Its `params.csv` and
`design.json` are immutable: a changed seed/range/design requires a different
output directory. Existing completed cases and source symlinks are preserved.
Topology is unchanged; each new shape gets regenerated rule-based fibres and
strict mesh/fibre checks. A positive-tetrahedron test alone does not establish
physiological plausibility or guarantee nonlinear solver convergence.

## Distributed run

`machines.json` records the public allocation: euclid-01 (12 jobs, shards 0/1),
euclid-10 (6 jobs, shard 2), euclid-21 (12 jobs, shards 3/4). Run
`worker.sh HOST JOBS SHARD...` with `SOLVER_PY` set to the DOLFINx interpreter,
after `_campaign/preflight.done` exists. Workers generate disjoint shards in the
shared checkout. Locks and detached process launch are supplied by the caller.
No GPU or training process is started.

Launch `finish.py` in the solver environment on one worker machine. It waits for
all workers to finish (success or failure), then tries the existing robust
same-grid nonlinear recovery for failed solver cases, at most two at once.
Failed attempts are archived. Cases without solver metadata are left for
inspection rather than being silently accepted. All cases are independently
validated before `_campaign/validated.done` is written. `retry_summary.json`
records retries. Worker `.failed` markers retain initial-generation failures;
`validated.done` is the final data-readiness signal. Unresolved failures are
listed in `validation.failed`.

No current model, checkpoint or training data root is changed. Using this wider
dataset for a subsequent training stage is a separate step.
