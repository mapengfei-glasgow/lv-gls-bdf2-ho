import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from experiments.geometry_narrow.prepare import prepare as prepare_narrow
from experiments.geometry_wide.prepare import prepare
from surrogate.params import NAMES, SPACE


class GeometryWideTest(unittest.TestCase):
    def test_expansion_reuses_source_and_preserves_splits_and_parameters(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);base=root/'base';base.mkdir()
            with (base/'params.csv').open('w',newline='') as f:
                w=csv.writer(f);w.writerow(['case','geom','split',*NAMES])
                for i,split in enumerate(['train']*16+['val']*2+['test']*2):
                    w.writerow([f'b{i}','mean',split,*[v[0] for v in SPACE.values()]])
                    d=base/f'b{i}';(d/'fields').mkdir(parents=True)
                    (d/'metadata.json').write_text(json.dumps({'status':'completed'}))
                    np.save(d/'fields/u.npy',np.zeros((1,4,3)))
            def fake_generate(out,coeff,center):
                out.mkdir(parents=True)
                (out/'meta.json').write_text(json.dumps(dict(center=center,coefficients=coeff.tolist())))
            narrow=root/'narrow';wide=root/'wide'
            with patch('experiments.geometry_narrow.prepare.generate',side_effect=fake_generate):
                prepare_narrow(narrow,base)
            prior=list(csv.DictReader((narrow/'params.csv').open()))
            for row in prior:
                d=narrow/row['case']
                if not d.exists():
                    (d/'fields').mkdir(parents=True)
                    (d/'metadata.json').write_text(json.dumps({'status':'completed'}))
                    np.save(d/'fields/u.npy',np.zeros((1,4,3)))
            digest=hashlib.sha256((narrow/'params.csv').read_bytes()).hexdigest()
            with patch('experiments.geometry_wide.prepare.generate',side_effect=fake_generate) as gen:
                design=prepare(wide,narrow)
                self.assertEqual(gen.call_count,30)
                prepare(wide,narrow)
                self.assertEqual(gen.call_count,30)
                with self.assertRaises(ValueError):prepare(wide,narrow,scale=.75)
            self.assertEqual(design['splits'],{'train':768,'val':204,'test':228})
            rows=list(csv.DictReader((wide/'params.csv').open()))
            self.assertEqual(rows[:600],prior)
            self.assertEqual(len(rows),1200)
            self.assertEqual(sum((wide/r['case']).is_symlink() for r in rows),600)
            for i,r in enumerate(rows[600:]):
                self.assertEqual([r[k] for k in NAMES],[prior[i%20][k] for k in NAMES])
            coeff=np.asarray(design['coefficients'])[30:]
            self.assertTrue((np.abs(coeff[:,:16]).max(axis=1)>.25).all())
            self.assertLessEqual(np.abs(coeff).max(),.5)
            self.assertFalse(coeff[:,16:].any())
            self.assertEqual(digest,hashlib.sha256((narrow/'params.csv').read_bytes()).hexdigest())
            with self.assertRaises(ValueError):prepare(narrow,narrow)


if __name__=='__main__':unittest.main()
