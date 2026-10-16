"""Require every planned case to match the mean geometry and solver design."""
import argparse
import csv
import json
from pathlib import Path

import h5py
import numpy as np


def validate(data, steps=200):
    data = Path(data)
    rows = list(csv.DictReader((data / 'params.csv').open()))
    with h5py.File(data / '_geometry/dalton_geom_mean/dalton_lv_mesh.h5') as f:
        coords = f['Mesh/mesh/geometry'][:] * .01
        cells = f['Mesh/mesh/topology'][:]
    bad = []
    for row in rows:
        d = data / row['case']
        try:
            m = json.loads((d / 'metadata.json').read_text())
            assert m['status'] == 'completed' and m['completed_steps'] == steps
            assert m['method'] == 'gls' and m['tau'] == 'pdf' and m['include_fs_stiffness']
            for key in ('a', 'b', 'a_f', 'b_f'):
                assert np.isclose(m['material_parameters'][key], float(row[key]), rtol=1e-7)
            for key in ('ta_amp_kPa', 'p_ed_kPa', 'p_amp_kPa', 'period_s'):
                assert np.isclose(m['loads'][key], float(row[key]), rtol=1e-7)
            assert np.isclose(m['dt_s'], float(row['period_s']) / steps, rtol=1e-8)
            fields = d / 'fields'
            np.testing.assert_allclose(np.load(fields / 'coords.npy'), coords, atol=1e-12, rtol=0)
            np.testing.assert_array_equal(np.load(fields / 'cells.npy'), cells)
            u = np.load(fields / 'u.npy', mmap_mode='r')
            assert u.shape == (steps + 1, len(coords), 3)
            assert np.isfinite(u).all()
        except (OSError, ValueError, KeyError, AssertionError) as e:
            bad.append(f"{row['case']}: {type(e).__name__} {str(e)[:160]}")
    if bad:
        raise SystemExit(f'{len(bad)}/{len(rows)} cases missing or invalid:\n' + '\n'.join(bad[:20]))
    print(f'Validated all {len(rows)} cases against PCA mean, parameters, topology, and {steps} steps')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('data', type=Path)
    ap.add_argument('--steps', type=int, default=200)
    a = ap.parse_args()
    validate(a.data, a.steps)
