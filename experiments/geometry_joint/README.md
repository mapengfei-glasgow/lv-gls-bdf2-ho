# Joint sampling of wider geometry and physical parameters

This replaces the superseded wider **crossed** design. Each simulation now gets
its own geometry **and** physical-parameter vector, sampled together from a
24-dimensional scrambled Sobol sequence. There is no reuse of a fixed set of
20 physical parameter vectors.

- 600 fresh joint samples; 456 training / 72 validation / 72 testing cases.
- Sobol seed 20261020, shuffled case split seed 20261021.
- Sobol coordinates 0–15 map linearly to the first 16 standardised PCA
  coefficients in **[-0.5, +0.5] sigma**; the other 24 PCA coefficients are zero.
- Coordinates 16–23 map to `surrogate.params.SPACE`: the same four log-uniform
  material constants and four uniform load/cycle parameters used previously.
  Physical ranges have not been widened.
- The sequence is generated in power-of-two blocks, retaining the first 600
  geometrically valid pairs. An invalid tetrahedron rejects the entire joint
  pair; rejected indices and accepted indices are recorded. Truncation/rejection
  does not preserve exact power-of-two Sobol balance.
- Every case has a distinct shape and parameter vector; train/validation/test
  case partitions therefore also separate shapes. This samples a coupled
  high-dimensional space; it is not exhaustive coverage of that space.
- Same solver/PDE, fibre convention, stabilisation, 200 steps and dt=T/200.

The generator requires only the bundled PCA/topology assets and Python numerical
packages; it does not depend on existing datasets or checkpoints.

```bash
# In a numerical Python environment (e.g. lvml):
python experiments/geometry_joint/prepare.py --jobs 6
# Once preparation and strict fibre/mesh checks finish, in the solver environment:
python -m surrogate.generate surrogate_data.geometry_joint/params.csv \
    surrogate_data.geometry_joint --mesh-root surrogate_data.geometry_joint/_geometry \
    --jobs 4 --steps 200
python experiments/geometry_narrow/validate.py surrogate_data.geometry_joint
```

`design.json` records all coefficients, exact joint unit-cube samples, physical
ranges, RNG/split seeds, asset/manifest hashes and rejected Sobol indices.
`joint_unit_samples.npy` provides those exact unit-cube coordinates as an array;
CSV physical values use the same precision as the solver command arguments.
`shape_coeffs.csv` records the 16 geometry inputs separately for future models.
Changed designs are rejected in existing output directories. Generated output
is under ignored `surrogate_data.geometry_joint/`.

Preparation builds each geometry, regenerates rule-based fibres and runs strict
mesh/fibre checks (up to six in parallel). `_campaign/prepared.done` is written
only when **all 600** pass. Any error creates `prepare.failed`, preventing queued
workers from starting. Check logs must be inspected rather than silently skipping
failed shapes or relaxing quality thresholds.

## Distributed deployment and completion

The public-machine allocation in `machines.json` is euclid-01 (12 jobs, shards
0/1), euclid-10 (6 jobs, shard 2), euclid-21 (12 jobs, shards 3/4). Detached
`worker.sh HOST JOBS SHARD...` processes wait for `prepared.done`, then generate
their disjoint simulation shards. Set `SOLVER_PY` to the DOLFINx interpreter.
Run `finish.py` in that environment on one worker machine to wait for all workers,
retry nonlinear solver failures using the existing robust same-grid recovery,
and validate all 600 cases. Failed attempts/logs are preserved. Final readiness
is recorded in `_campaign/validated.done`; unresolved cases are recorded in
`validation.failed`. No training is started by these scripts.

The previously launched `geometry_wide` crossed-design workers were stopped on
user correction. Their outputs, the completed narrow dataset and the running
narrow-model training are preserved. This campaign has its own data root and
locks, and does not modify them. A subsequent training stage is a separate step.
