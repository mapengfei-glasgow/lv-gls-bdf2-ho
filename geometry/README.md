# Self-contained LV geometry generator

This directory contains everything needed to regenerate Dalton-family LV meshes.
No dataset download, external project, DOLFINx, or pretrained emulator is needed.
Install the numerical dependencies (or use the repository's solver/ML environment):

```bash
python -m pip install -r geometry/requirements.txt
python geometry/generate.py --out geometry/generated/mean
python geometry/generate.py --center anchor --out geometry/generated/anchor
python geometry/generate.py --center anchor --modes 16 --scale 0.25 --seed 42 \
    --out geometry/generated/narrow
python geometry/generate.py --modes 40 --scale 1 --seed 42 \
    --out geometry/generated/full
```

Run from any working directory by supplying the path to `generate.py`. Assets are
resolved relative to the script. Outputs under `geometry/generated/` are ignored
by Git. Do not commit generated meshes. Custom output locations should likewise
be kept out of version control.

## PCA convention and provenance

`assets/pca_basis.npz` was extracted from the existing
`_work/pca_dalton/pca_dalton.npz`, fitted by `fit_pca_dalton.py` to all 2,967 Dalton
geometries (including the original training, validation, and test partitions).
It contains the raw-coordinate mean (6,784 x 3, flattened), the first 40 unit
PCA modes, their explained variance ratios, score standard deviations computed
with `ddof=0`, and the scores of geom_0000 (the curriculum anchor).
Individual source geometries and the full per-geometry score table are omitted.
This preserves the existing fitted shape space; it is not a PCA fit on training
geometries alone.

For 40 standardised offsets `z`, reconstruction is:

```
scores = z * sigma                         # center=mean
scores = anchor_scores + z * sigma         # center=anchor
raw_coordinates = mean + scores @ modes
```

`--scale` samples each of the first `--modes` offsets uniformly in
`[-scale, scale]`; remaining offsets are zero. This supports a fixed shape
(scale 0), a narrow curriculum, and wider geometry variation. There is no
universal valid maximum scale: inverted/nonfinite tetrahedra are rejected.
For exact coefficients, supply `--coefficients offsets.npy` (shape `(40,)`).
These offsets are saved as `coefficients.npy`; physical PCA scores are saved as
`scores.npy`. Historical experimental files named `scores_u.npy` have inconsistent
conventions and should not be passed here without checking their units.

`assets/topology.npz` contains the fixed tetrahedral connectivity, cell and
boundary tags, raw-corner permutation, centroid interpolation, and basal node
indices. These were extracted from the converted geom_0000 template, using
coordinate matching to recover the raw-node ordering. No individual geometry's
coordinates or reference fibre fields are stored in this topology asset.
Each original hexahedron contributes a centroid and 12 tetrahedra, producing
11,779 nodes and 59,940 tetrahedra. The mapping was verified against geom_0000.

## Output and solver use

Each output directory contains `dalton_lv_mesh.xdmf` and `dalton_lv_mesh.h5`,
`coefficients.npy`, `scores.npy`, `meta.json`, and `fibres.json`.
Coordinates are in **cm**; the long axis is x, the base is at x=0 (to numerical
precision), and the apex points toward negative x. The transformation from raw
coordinates is `(raw_z - mean_base_z, raw_y, -raw_x)`. Boundary tags are
5=base, 6=endocardium, 7=epicardium.

`fibers.py` is the shared rule-based fibre generator: it solves a P1
Laplace transmural problem and rebuilds fibres (+60 degrees endocardium,
-60 degrees epicardium), sheets, and normals for each generated shape.
The problematic original Dalton fibre arrays are not reused.

```bash
python tools/check_mesh.py geometry/generated/narrow --strict
python -m lvsolver.run --method gls --tau pdf --include-fs \
    --mesh geometry/generated/narrow --out results/generated_narrow
```

To use `surrogate.generate`, name output directories
`geometry/generated/dalton_geom_<id>` and pass `--mesh-root geometry/generated`.
Standard Python packages still need installation; all geometry-specific assets
are bundled here. Re-fitting PCA from source meshes is outside this directory's
scope.
