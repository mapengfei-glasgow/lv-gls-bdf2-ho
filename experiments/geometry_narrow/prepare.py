"""Prepare the first geometry curriculum around the PCA mean; data only."""
import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from geometry.generate import generate, reconstruct
from surrogate.params import NAMES


def combined_split(shape_split, parameter_split):
    if shape_split == parameter_split == 'train':
        return 'train'
    if 'test' in (shape_split, parameter_split):
        return 'test'
    return 'val'


def prepare(data, base_data, seed=20261017, scale=.25):
    data, base_data = Path(data).resolve(), Path(base_data).resolve()
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError('scale must be positive and finite')
    with (base_data / 'params.csv').open() as f:
        base_rows = list(csv.DictReader(f))
    rng = np.random.default_rng(seed)
    parameters = []
    for split, count in [('train', 16), ('val', 2), ('test', 2)]:
        group = [r for r in base_rows if r['split'] == split]
        if len(group) < count:
            raise ValueError(f'base manifest needs {count} {split} parameters')
        parameters.extend(group[i] for i in rng.choice(len(group), count, replace=False))
    coefficients = np.zeros((30, 40))
    coefficients[1:, :16] = rng.uniform(-scale, scale, (29, 16))
    shape_splits = ['train'] * 24 + ['val'] * 3 + ['test'] * 3
    rows = []
    for shape in range(30):
        for param, source in enumerate(parameters):
            rows.append(dict(case=f'narrow_g{shape:02d}_p{param:02d}',
                             geom=f'narrow{shape:03d}',
                             split=combined_split(shape_splits[shape], source['split']),
                             **{k: source[k] for k in NAMES}))
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=['case', 'geom', 'split', *NAMES])
    writer.writeheader(); writer.writerows(rows)
    manifest = stream.getvalue().encode()
    provenance = dict(seed=seed, scale_sigma=scale, varied_modes=16, center='mean',
                      shape_splits=shape_splits, coefficients=coefficients.tolist(),
                      parameter_source_cases=[r['case'] for r in parameters],
                      parameter_splits=[r['split'] for r in parameters],
                      splits={s: sum(r['split'] == s for r in rows) for s in ('train','val','test')},
                      steps=200, dt='period_s / 200',
                      pca_sha256=hashlib.sha256((ROOT/'geometry/assets/pca_basis.npz').read_bytes()).hexdigest(),
                      base_manifest_sha256=hashlib.sha256((base_data/'params.csv').read_bytes()).hexdigest(),
                      params_sha256=hashlib.sha256(manifest).hexdigest())
    data.mkdir(parents=True, exist_ok=True)
    if (data/'design.json').exists() and json.loads((data/'design.json').read_text()) != provenance:
        raise ValueError('existing design differs; use a new data directory')
    if (data/'params.csv').exists() and (data/'params.csv').read_bytes() != manifest:
        raise ValueError('existing parameter manifest differs; use a new data directory')
    # Check every shape before writing meshes or scheduling simulations.
    with np.load(ROOT/'geometry/assets/topology.npz') as z:
        cells = z['cells']
    for shape, coeff in enumerate(coefficients):
        _, xyz, _ = reconstruct(coeff)
        p = xyz[cells]
        vol = np.einsum('ij,ij->i', np.cross(p[:,1]-p[:,0], p[:,2]-p[:,0]),p[:,3]-p[:,0])/6
        if not np.isfinite(vol).all() or (vol <= 0).any():
            raise ValueError(f'shape {shape} has invalid tetrahedra')
    (data/'design.json').write_text(json.dumps(provenance, indent=2)+'\n')
    (data/'params.csv').write_bytes(manifest)
    with (data/'shape_coeffs.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['geom', *[f'pc{i:02d}' for i in range(16)]])
        for shape, coeff in enumerate(coefficients):
            writer.writerow([f'narrow{shape:03d}', *coeff[:16]])
    reused = 0
    for shape, coeff in enumerate(coefficients):
        mesh = data/'_geometry'/f'dalton_geom_narrow{shape:03d}'
        if mesh.exists():
            meta = json.loads((mesh/'meta.json').read_text())
            if meta['center'] != 'mean' or not np.array_equal(meta['coefficients'], coeff):
                raise ValueError(f'existing geometry differs: {mesh}')
        else:
            generate(mesh, coeff, center='mean')
        if shape == 0:
            # Identical geometry and parameters: borrow immutable completed mean cases.
            for param, source in enumerate(parameters):
                src = base_data/source['case']
                dst = data/f'narrow_g00_p{param:02d}'
                try:
                    complete = json.loads((src/'metadata.json').read_text())['status']=='completed'
                except (OSError, ValueError, KeyError):
                    complete = False
                if complete and (src/'fields/u.npy').exists():
                    if not dst.exists() and not dst.is_symlink():
                        os.symlink(os.path.relpath(src, data), dst)
                    reused += int(dst.is_symlink())
    print(f'Prepared {len(rows)} cases, {reused} reused mean cases; splits={provenance["splits"]}',flush=True)
    return provenance


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data', type=Path, default=ROOT/'surrogate_data.geometry_narrow')
    ap.add_argument('--base-data', type=Path, default=ROOT/'surrogate_data.mean_base')
    ap.add_argument('--seed', type=int, default=20261017)
    ap.add_argument('--scale', type=float, default=.25)
    a=ap.parse_args()
    prepare(a.data, a.base_data, a.seed, a.scale)
