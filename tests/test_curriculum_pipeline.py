import argparse
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from experiments.curriculum.run import Pipeline,best_source,freeze_source,is_complete,train_command


class CurriculumTest(unittest.TestCase):
    def test_minimum_loss_snapshot_and_resume_preserve_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'source';source.mkdir()
            for name,it,loss in [('last',1000,.009),('best',900,.008),('best_loss',800,.007)]:
                torch.save(dict(iter=it,training_loss=loss,args=dict(iters=1000)),source/f'{name}.pt')
            hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in source.glob('*.pt')}
            self.assertTrue(is_complete(source,1000))
            self.assertEqual(best_source(source).name,'best_loss.pt')
            control=root/'controller';snapshot=freeze_source(source,control)
            self.assertEqual(torch.load(snapshot,weights_only=False)['iter'],800)
            self.assertEqual(freeze_source(source,control),snapshot)
            out=root/'new_model'
            cmd=train_command('python',root/'data',out,snapshot,1000)
            self.assertIn('--init-from',cmd);self.assertNotIn('--resume',cmd)
            out.mkdir();torch.save(dict(iter=500,args=dict(iters=1000)),out/'last.pt')
            self.assertFalse(is_complete(out,1000))
            cmd=train_command('python',root/'data',out,snapshot,1000)
            self.assertIn('--resume',cmd);self.assertNotIn('--init-from',cmd)
            with self.assertRaises(ValueError):is_complete(source,2000)
            self.assertEqual(hashes,{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in source.glob('*.pt')})

    def pipeline(self,root):
        machines=root/'machines.json';machines.write_text(json.dumps({'workers':[]}))
        a=argparse.Namespace(control=root/'control',machines=machines,gpu='0',
            train_python='python',solver_python='solver',cases=600,iters=120000,poll=.01,
            source_run=root/'narrow',half_data=root/'half_data',half_run=root/'half_run',
            full_data=root/'full_data',full_run=root/'full_run')
        return Pipeline(a)

    def test_wait_then_half_data_training_then_full_data_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            pipeline=self.pipeline(Path(tmp));events=[]
            pipeline.dataset=lambda data,scale,seed:events.append(('data',scale))
            pipeline.train=lambda source,data,out,label:events.append(('train',label,source))
            with patch('experiments.curriculum.run.is_complete',side_effect=[False,True]), \
                 patch('experiments.curriculum.run.time.sleep') as sleep:
                pipeline.run()
            sleep.assert_called_once()
            self.assertEqual(events,[('data',.5),('train','sigma050',pipeline.a.source_run),
                                     ('data',1.),('train','sigma100',pipeline.a.half_run)])
            self.assertTrue((pipeline.control/'complete.done').exists())

    def test_validation_failure_prevents_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            pipeline=self.pipeline(Path(tmp));calls=[]
            def bad_data(*args):raise ValueError('failed simulation')
            pipeline.dataset=bad_data;pipeline.train=lambda *args:calls.append(args)
            with patch('experiments.curriculum.run.is_complete',return_value=True):
                with self.assertRaises(ValueError):pipeline.run()
            self.assertEqual(calls,[])
            self.assertFalse((pipeline.control/'complete.done').exists())


if __name__=='__main__':unittest.main()
