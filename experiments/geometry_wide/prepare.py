"""Expand the narrow geometry design to ±0.5 sigma; preserve all source cases."""
import argparse
from collections import Counter
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
from experiments.geometry_narrow.prepare import combined_split
from geometry.generate import generate, reconstruct
from surrogate.params import NAMES


def link(source, destination):
    if destination.is_symlink():
        if destination.resolve() != source.resolve():
            raise ValueError(f'conflicting symlink: {destination}')
    elif not destination.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.symlink_to(os.path.relpath(source, destination.parent), target_is_directory=True)
    else:
        raise ValueError(f'refusing to replace existing source-case copy: {destination}')


def prepare(data, source, seed=20261019, scale=.5):
    data, source = Path(data).resolve(), Path(source).resolve()
    if data == source:
        raise ValueError('expanded output must differ from source')
    old = json.loads((source/'design.json').read_text())
    if not np.isfinite(scale) or scale <= old['scale_sigma']:
        raise ValueError('new coefficient bound must exceed the narrow bound')
    if old['varied_modes'] != 16:
        raise ValueError('expected 16-mode narrow curriculum')
    with (source/'params.csv').open() as f:
        prior = list(csv.DictReader(f))
    if len(prior) != 600 or len(old['coefficients']) != 30:
        raise ValueError('expected complete 30-shape, 600-case narrow design')
    for row in prior:
        d = source/row['case']
        if json.loads((d/'metadata.json').read_text()).get('status') != 'completed' or not (d/'fields/u.npy').exists():
            raise ValueError(f'source case incomplete: {row["case"]}')
    source_hash = hashlib.sha256((source/'params.csv').read_bytes()).hexdigest()
    if old['params_sha256'] != source_hash:
        raise ValueError('source manifest hash mismatch')
    parameters = [r for r in prior if r['geom'] == prior[0]['geom']]
    if len(parameters) != 20:
        raise ValueError('expected 20 matched parameter vectors')
    rng = np.random.default_rng(seed)
    extra = np.zeros((30, 40))
    rejected = 0
    with np.load(ROOT/'geometry/assets/topology.npz') as z:
        cells = z['cells']
    # Draw from the wider box outside the old box. Invalid candidates are
    # rejected, not repaired/clipped, and their count is recorded.
    for shape in range(30):
        for attempt in range(10000):
            coeff = np.zeros(40)
            coeff[:16] = rng.uniform(-scale, scale, 16)
            if np.abs(coeff[:16]).max() <= old['scale_sigma']:
                rejected += 1
                continue
            _, xyz, _ = reconstruct(coeff)
            p = xyz[cells]
            vol = np.einsum('ij,ij->i', np.cross(p[:,1]-p[:,0],p[:,2]-p[:,0]),p[:,3]-p[:,0])/6
            if np.isfinite(vol).all() and (vol > 0).all():
                extra[shape] = coeff
                break
            rejected += 1
        else:
            raise ValueError('unable to sample valid wider geometry')
    coefficients = np.vstack([old['coefficients'], extra])
    shape_splits = old['shape_splits'] + ['train']*24 + ['val']*3 + ['test']*3
    old_geoms = list(dict.fromkeys(r['geom'] for r in prior))
    geoms = old_geoms + [f'wide{i:03d}' for i in range(30)]
    rows = list(prior)
    for shape in range(30):
        for param, row in enumerate(parameters):
            rows.append(dict(case=f'wide_g{shape:02d}_p{param:02d}', geom=geoms[30+shape],
                split=combined_split(shape_splits[30+shape],old['parameter_splits'][param]),
                **{k:row[k] for k in NAMES}))
    stream = io.StringIO(newline='')
    w = csv.DictWriter(stream, fieldnames=['case','geom','split',*NAMES])
    w.writeheader();w.writerows(rows)
    manifest = stream.getvalue().encode()
    design = dict(seed=seed, scale_sigma=scale, varied_modes=16, center='mean',
        prior_scale_sigma=old['scale_sigma'], existing_cases=600, new_cases=600,
        source_manifest_sha256=source_hash, source_design_sha256=hashlib.sha256((source/'design.json').read_bytes()).hexdigest(),
        coefficients=coefficients.tolist(), geometry_ids=geoms, shape_splits=shape_splits,
        parameter_source_cases=old['parameter_source_cases'], parameter_splits=old['parameter_splits'],
        splits=dict(Counter(r['split'] for r in rows)), steps=200, dt='period_s / 200',
        sampling='Keep all narrow shapes; sample new shapes uniformly in wider box outside old box, conditioned on positive tetrahedra',
        rejected_candidates=rejected,
        pca_sha256=hashlib.sha256((ROOT/'geometry/assets/pca_basis.npz').read_bytes()).hexdigest(),
        params_sha256=hashlib.sha256(manifest).hexdigest())
    data.mkdir(parents=True, exist_ok=True)
    if (data/'design.json').exists() and json.loads((data/'design.json').read_text()) != design:
        raise ValueError('existing design differs; choose a new data directory')
    if (data/'params.csv').exists() and (data/'params.csv').read_bytes() != manifest:
        raise ValueError('existing manifest differs; choose a new data directory')
    (data/'design.json').write_text(json.dumps(design,indent=2)+'\n')
    (data/'params.csv').write_bytes(manifest)
    with (data/'shape_coeffs.csv').open('w',newline='') as f:
        w = csv.writer(f);w.writerow(['geom',*[f'pc{i:02d}' for i in range(16)]])
        w.writerows([g,*c[:16]] for g,c in zip(geoms,coefficients))
    for row in prior:
        link(source/row['case'], data/row['case'])
    for i, geom in enumerate(geoms):
        dest=data/'_geometry'/f'dalton_geom_{geom}'
        if i<30:
            link(source/'_geometry'/f'dalton_geom_{geom}',dest)
        elif dest.exists():
            meta=json.loads((dest/'meta.json').read_text())
            if meta['center'] != 'mean' or not np.array_equal(meta['coefficients'],coefficients[i]):
                raise ValueError(f'geometry differs: {dest}')
        else:
            generate(dest,coefficients[i],center='mean')
    (data/'_campaign').mkdir(exist_ok=True)
    print(f'Prepared {len(rows)} cases: 600 reused + 600 new; splits={design["splits"]}',flush=True)
    return design


if __name__ == '__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data',type=Path,default=ROOT/'surrogate_data.geometry_wide')
    ap.add_argument('--source',type=Path,default=ROOT/'surrogate_data.geometry_narrow')
    ap.add_argument('--seed',type=int,default=20261019)
    ap.add_argument('--scale',type=float,default=.5)
    a=ap.parse_args();prepare(a.data,a.source,a.seed,a.scale)
