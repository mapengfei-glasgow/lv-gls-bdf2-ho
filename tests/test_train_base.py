import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import torch

from surrogate.params import NAMES, SPACE


class BaseResumeTest(unittest.TestCase):
    def test_resume_matches_uninterrupted_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root / 'data'
            data.mkdir()
            rows = []
            for i, split in enumerate(('train', 'val', 'test')):
                case = f'c{i}'
                rows.append([case, 'mean', split, *[v[0] for v in SPACE.values()]])
                d = data / case
                fields = d / 'fields'
                fields.mkdir(parents=True)
                (d / 'metadata.json').write_text(json.dumps({'status': 'completed'}))
                coords = np.array([[0, 0, 0], [-.01, 0, 0], [0, .01, 0], [0, 0, .01]])
                u = np.arange(4)[:, None, None] ** 2 * np.ones((4, 4, 3)) * 1e-6
                arrays = dict(coords=coords, cells=np.array([[0, 2, 1, 3]]),
                              node_tags=np.array([4, 1, 2, 4], np.uint8),
                              endo_facets=np.array([[0, 1, 3]]), u=u.astype('float32'),
                              loads=np.zeros((4, 2)), time=np.arange(4) * .005)
                for key, value in arrays.items():
                    np.save(fields / f'{key}.npy', value)
            with (data / 'params.csv').open('w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['case', 'geom', 'split', *NAMES])
                writer.writerows(rows)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1',
                       OPENBLAS_NUM_THREADS='1')
            args = [str(data), '--iters', '4', '--batch', '1', '--latent', '8',
                    '--mp-steps', '1', '--eval-every', '2', '--eval-cases', '1', '--ema']
            snapshot = root / 'snapshot.pt'
            # Capture iteration 2 without changing the four-step cosine schedule.
            code = '''
import torch, runpy, sys
original = torch.save
snapshot = sys.argv.pop(1)
def save(obj, path):
    if obj.get('iter') == 2:
        original(obj, snapshot)
    original(obj, path)
torch.save = save
runpy.run_module('surrogate.train_base', run_name='__main__')
'''
            subprocess.run([sys.executable, '-c', code, str(snapshot), *args,
                            '--out', str(root / 'full')], env=env, check=True)
            subprocess.run([sys.executable, '-m', 'surrogate.train_base', *args,
                            '--resume', str(snapshot), '--out', str(root / 'resumed')],
                           env=env, check=True)
            full = torch.load(root / 'full/last.pt', weights_only=False)
            resumed = torch.load(root / 'resumed/last.pt', weights_only=False)
            for key in ('model', 'raw_model', 'ema'):
                for name in full[key]:
                    torch.testing.assert_close(full[key][name], resumed[key][name],
                                               rtol=0, atol=0)
            for a, b in zip(full['norms'], resumed['norms']):
                self.assertEqual(a['count'], b['count'])
                for key in ('sum', 'sumsq'):
                    torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
