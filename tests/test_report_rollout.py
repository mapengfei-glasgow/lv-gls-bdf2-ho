from types import SimpleNamespace
import unittest
import numpy as np
import torch
from surrogate.data import Graph, RunningNormalizer
from surrogate.graph import build_augmented_graph
from surrogate.meshgraphnet import MeshGraphNet
from surrogate.train import Surrogate
from experiments.mean_base.report.evaluate import rollout_batch
from experiments.mean_base.report.plots import history


class ReportTest(unittest.TestCase):
    def test_batched_rollout_matches_serial_with_different_dt(self):
        coords=np.eye(4,3).astype('float32')*.01
        edges=np.array([[i,j] for i in range(4) for j in range(4) if i!=j])
        g=Graph(build_augmented_graph(coords,edges),np.zeros((4,3),np.float32),'cpu')
        norms=[RunningNormalizer(d,'cpu') for d in [18,8,3]]
        for n in norms:
            n.count=1.;n.sumsq=torch.ones(n.dim,dtype=torch.float64)
        sur=Surrogate(MeshGraphNet(18,8,latent=8,mlp_hidden=8,processor_steps=1),g,norms)
        cases=[SimpleNamespace(name=f'c{i}',coords=coords,dt=.005+i*.001,steps=3,
                               loads=np.zeros((4,2),np.float32),punit=np.zeros(8,np.float32)) for i in range(2)]
        batch=rollout_batch(sur,cases)
        for i,c in enumerate(cases):
            np.testing.assert_allclose(batch[i],sur.rollout(c),rtol=1e-5,atol=1e-6)

    def test_elapsed_time_accounts_for_resume_segments(self):
        s='it 1000 loss 0.5 rollout RMSE 10 mm (100s)\nit 12000 loss 0.1 rollout RMSE 1 mm (300s)\nit 13000 loss 0.09 rollout RMSE 0.9 mm (50s)'
        rows=history(s)
        self.assertAlmostEqual(rows[-1]['cumulative_active_hours'],350/3600)


if __name__=='__main__':unittest.main()
