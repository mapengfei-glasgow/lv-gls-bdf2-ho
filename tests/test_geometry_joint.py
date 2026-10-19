import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from experiments.geometry_joint.prepare import sample,prepare
from surrogate.params import NAMES,from_unit


class GeometryJointTest(unittest.TestCase):
    def test_joint_mapping_split_disjointness_and_reproducibility(self):
        rows,coeff,design=sample(cases=50)
        rows2,coeff2,design2=sample(cases=50)
        self.assertEqual(rows,rows2);self.assertEqual(design,design2)
        np.testing.assert_array_equal(coeff,coeff2)
        self.assertEqual(design['joint_dimensions'],24)
        self.assertEqual(design['splits'],{'train':38,'val':6,'test':6})
        unit=np.asarray(design['joint_unit_samples'])
        np.testing.assert_allclose(coeff[:,:16],unit[:,:16]-.5)
        self.assertFalse(coeff[:,16:].any())
        physical=np.array([[float(r[k]) for k in NAMES] for r in rows])
        np.testing.assert_allclose(physical,from_unit(unit[:,16:]),rtol=1e-7)
        self.assertEqual(len({tuple(p) for p in physical}),50)
        self.assertEqual(len({r['geom'] for r in rows}),50)
        self.assertEqual(len({r['case'] for r in rows}),50)
        for a in ['train','val','test']:
            for b in ['train','val','test']:
                if a!=b:self.assertTrue({r['geom'] for r in rows if r['split']==a}.isdisjoint(
                    {r['geom'] for r in rows if r['split']==b}))

    def test_preparation_idempotence_and_changed_design_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'joint'
            def fake_generate(out,coeff,center):
                out.mkdir(parents=True)
                (out/'meta.json').write_text(json.dumps(dict(center=center,coefficients=coeff.tolist())))
            with patch('experiments.geometry_joint.prepare.generate',side_effect=fake_generate) as gen, \
                 patch('experiments.geometry_joint.prepare.subprocess.run'):
                prepare(root,cases=10,jobs=2)
                self.assertTrue((root/'_campaign/prepared.done').exists())
                self.assertEqual(gen.call_count,10)
                prepare(root,cases=10,jobs=2)
                self.assertEqual(gen.call_count,10)
                with self.assertRaises(ValueError):prepare(root,cases=10,scale=.75)
            with (root/'params.csv').open() as f:self.assertEqual(len(list(csv.DictReader(f))),10)


if __name__=='__main__':unittest.main()
