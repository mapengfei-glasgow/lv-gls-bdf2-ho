import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from experiments.geometry_narrow.prepare import prepare
from surrogate.params import NAMES, SPACE


class GeometryNarrowTest(unittest.TestCase):
    def test_crossed_design_coefficients_reuse_and_idempotence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = root / 'base'
            base.mkdir()
            splits = ['train'] * 16 + ['val'] * 2 + ['test'] * 2
            with (base / 'params.csv').open('w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['case', 'geom', 'split', *NAMES])
                for i, split in enumerate(splits):
                    case = f'base_{i}'
                    writer.writerow([case, 'mean', split, *[v[0] for v in SPACE.values()]])
                    d = base / case
                    (d / 'fields').mkdir(parents=True)
                    (d / 'metadata.json').write_text(json.dumps({'status': 'completed'}))
                    np.save(d / 'fields/u.npy', np.zeros((1, 4, 3)))
            data = root / 'narrow'

            def fake_generate(out, coeff, center):
                out.mkdir(parents=True)
                (out / 'meta.json').write_text(json.dumps(dict(center=center, coefficients=coeff.tolist())))

            with patch('experiments.geometry_narrow.prepare.generate', side_effect=fake_generate) as gen:
                design = prepare(data, base)
                self.assertEqual(gen.call_count, 30)
                prepare(data, base)
                self.assertEqual(gen.call_count, 30)
                with self.assertRaises(ValueError):
                    prepare(data, base, scale=.5)
            self.assertEqual(design['splits'], {'train': 384, 'val': 102, 'test': 114})
            coefficients = np.asarray(design['coefficients'])
            self.assertEqual(coefficients.shape, (30, 40))
            self.assertFalse(coefficients[0].any())
            self.assertFalse(coefficients[:, 16:].any())
            self.assertLessEqual(np.abs(coefficients).max(), .25)
            with (data / 'params.csv').open() as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len({r['case'] for r in rows}), 600)
            self.assertEqual(sum((data / r['case']).is_symlink() for r in rows), 20)
            self.assertTrue(all((base / f'base_{i}/fields/u.npy').exists() for i in range(20)))


if __name__ == '__main__':
    unittest.main()
