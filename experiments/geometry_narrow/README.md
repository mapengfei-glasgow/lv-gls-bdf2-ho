# Narrow geometry variation: data preparation only

The first geometry curriculum is centred on the bundled **PCA mean**.
It leaves the running mean-geometry model and checkpoints unchanged.

- 30 shapes: shape 0 is the mean; 29 shapes have independent uniform offsets
  in **[-0.25, 0.25] standard deviations on the first 16 PCs**. Other PCs are zero.
- Fixed topology, 11,779 nodes / 59,940 tetrahedra. Every shape is checked for
  positive tetrahedral volumes, and rule-based fibres are regenerated.
- 20 parameter vectors selected reproducibly from the completed mean campaign:
  16 from its training split, 2 validation, 2 test. Exactly the same parameter
  vectors are crossed with every shape. Seed: 20261017.
- Shape split: 24 train / 3 validation / 3 test. A case is training only when
  both its shape and parameter are training; any test component gives a test
  case; remaining combinations are validation.
- **600 cases: 384 train / 102 validation / 114 test.**
- Existing completed mean-shape trajectories for the selected parameters are
  reused through relative symlinks, when available (20 cases); **580 new solves**.
  The source files are left intact. On a checkout without those outputs, all
  600 cases are generated normally.
- Solver configuration matches the mean campaign: GLS P1/P1, PDF stabilisation,
  fs stiffness included, 200 time steps, dt = period / 200.

```bash
python experiments/geometry_narrow/prepare.py
python -m surrogate.generate surrogate_data.geometry_narrow/params.csv \
    surrogate_data.geometry_narrow \
    --mesh-root surrogate_data.geometry_narrow/_geometry --jobs 8
python experiments/geometry_narrow/validate.py surrogate_data.geometry_narrow
```

Outputs are in the ignored `surrogate_data.geometry_narrow/` directory:
`params.csv`, `design.json` (full coefficients, source parameter IDs, seeds,
asset hashes, split design), `shape_coeffs.csv` (standardised first 16 PCs),
`_geometry/`, and solver case directories. Changing the design in an existing
directory is rejected. Re-running the same design preserves completed outputs.

The public-machine allocation is recorded in `machines.json`: euclid-01 (12
jobs, shards 0/1), euclid-10 (6 jobs, shard 2), euclid-21 (12 jobs, shards 3/4),
with five shards in total. `worker.sh HOST JOBS SHARD...` generates disjoint
case directories in the shared Euclid checkout. Per-worker logs and sentinels
are stored under `_campaign/`. `finish.sh` waits and validates every simulation
against its own reference shape. It creates `validated.done` when the data is
ready. **None of these scripts starts training.**

## Nonlinear retry provenance

`retry.py` archives failed case directories under `_failed_attempts/` and never
overwrites completed cases. The corrected `robust` strategy explicitly sets the
effective SNES iteration cap after GLS initialisation (which otherwise overrides
the global cap), then tries backtracking Newton and trust-region solves with
previous/extrapolated initial guesses at the **same time step**. The PDE, material
and load parameters, geometry, stabilisation, tolerances, and 200-step time grid
are retained. Requested/effective settings and fallback events are recorded in
the case metadata. This is numerical recovery, not a change to the sampled design.

```bash
python experiments/geometry_narrow/retry.py --data surrogate_data.geometry_narrow \
    --case narrow_g01_p13 --strategy robust
```
