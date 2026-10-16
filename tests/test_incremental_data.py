import csv
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from surrogate.data import LVDataset
from surrogate.params import NAMES, SPACE
from surrogate.train_base import add_completed_cases


class IncrementalDataTest(unittest.TestCase):
    def test_only_completed_training_cases_are_admitted_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            specs = [('base', 'train', 'completed'), ('new', 'train', 'running'),
                     ('val', 'val', 'completed'), ('test', 'test', 'completed')]
            with (root / 'params.csv').open('w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['case', 'geom', 'split', *NAMES])
                for name, split, status in specs:
                    writer.writerow([name, 'mean', split, *[v[0] for v in SPACE.values()]])
                    d = root / name
                    fields = d / 'fields'
                    fields.mkdir(parents=True)
                    (d / 'metadata.json').write_text(json.dumps({'status': status}))
                    arrays = dict(coords=np.eye(4, 3) * .01,
                                  cells=np.array([[0, 1, 2, 3]]),
                                  node_tags=np.array([1, 2, 4, 4], np.uint8),
                                  endo_facets=np.array([[0, 1, 2]]),
                                  u=np.zeros((4, 4, 3), np.float32),
                                  loads=np.zeros((4, 2)), time=np.arange(4) * .005)
                    for key, array in arrays.items():
                        np.save(fields / f'{key}.npy', array)
            ds = LVDataset(root)
            train, added = add_completed_cases(ds, ds.split('train'))
            self.assertEqual([c.name for c in train], ['base'])
            self.assertFalse(added)
            (root / 'new/metadata.json').write_text(json.dumps({'status': 'completed'}))
            train, added = add_completed_cases(ds, train)
            self.assertEqual([c.name for c in train], ['base', 'new'])
            self.assertEqual([c.name for c in added], ['new'])
            train, added = add_completed_cases(ds, train)
            self.assertFalse(added)
            self.assertEqual(len(train), 2)


if __name__ == '__main__':
    unittest.main()
