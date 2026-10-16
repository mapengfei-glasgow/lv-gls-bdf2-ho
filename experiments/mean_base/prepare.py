"""Prepare a fixed PCA-mean geometry and reproducible parameter design."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.stats import qmc

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from geometry.generate import generate
from surrogate.params import NAMES, SPACE, from_unit


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data', type=Path, default=ROOT / 'surrogate_data.mean_base')
    ap.add_argument('--cases', type=int, default=1000)
    ap.add_argument('--seed', type=int, default=42)
    a = ap.parse_args()
    if a.cases < 10:
        ap.error('at least 10 cases required for train/val/test splits')
    data = a.data.resolve()
    # Power-of-two Sobol design, truncated to the requested count.
    unit = qmc.Sobol(len(NAMES), scramble=True, seed=a.seed).random_base2(
        int(np.ceil(np.log2(a.cases))))[:a.cases]
    params = from_unit(unit)
    order = np.random.default_rng(a.seed + 1).permutation(a.cases)
    ntrain, nval = int(.76 * a.cases), int(.12 * a.cases)
    splits = np.full(a.cases, 'test', dtype='<U5')
    splits[order[:ntrain]] = 'train'
    splits[order[ntrain:ntrain + nval]] = 'val'
    import io
    stream = io.StringIO(newline='')
    w = csv.writer(stream)
    w.writerow(['case', 'geom', 'split', *NAMES])
    for i in range(a.cases):
        w.writerow([f'mean_p{i:04d}', 'mean', splits[i], *[f'{v:.8g}' for v in params[i]]])
    manifest = stream.getvalue()
    if (data / 'params.csv').exists() and (data / 'params.csv').read_bytes() != manifest.encode():
        raise SystemExit('existing parameter design differs; choose a new --data directory')
    data.mkdir(parents=True, exist_ok=True)
    mesh = data / '_geometry/dalton_geom_mean'
    if not mesh.exists():
        generate(mesh, np.zeros(40), center='mean')
    else:
        meta = json.loads((mesh / 'meta.json').read_text())
        if meta['center'] != 'mean' or np.any(meta['coefficients']):
            raise SystemExit('existing geometry is not the PCA mean')
    (data / 'params.csv').write_bytes(manifest.encode())
    asset = ROOT / 'geometry/assets/pca_basis.npz'
    (data / 'design.json').write_text(json.dumps(dict(
        geometry='PCA mean; all 40 scores zero', seed=a.seed, cases=a.cases,
        splits={s: int((splits == s).sum()) for s in ('train', 'val', 'test')},
        parameter_space=SPACE, steps=200, dt='period_s / 200',
        pca_sha256=hashlib.sha256(asset.read_bytes()).hexdigest(),
        params_sha256=hashlib.sha256(manifest.encode()).hexdigest(),
    ), indent=2) + '\n')
    print(f'Prepared {a.cases} cases in {data}')


if __name__ == '__main__':
    main()
