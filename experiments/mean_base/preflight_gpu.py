"""Check one full-size training step on the selected GPU before a long campaign."""
import argparse
from pathlib import Path
import sys
from types import SimpleNamespace

import h5py
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from surrogate.data import Graph, NODE_IN, EDGE_IN
from surrogate.graph import build_augmented_graph
from surrogate.meshgraphnet import MeshGraphNet


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data', type=Path, default=ROOT / 'surrogate_data.mean_base')
    ap.add_argument('--batch', type=int, default=4)
    a = ap.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit('CUDA is required for this preflight')
    with h5py.File(a.data / '_geometry/dalton_geom_mean/dalton_lv_mesh.h5') as h:
        coords = h['Mesh/mesh/geometry'][:] * .01
        cells = h['Mesh/mesh/topology'][:]
    edges = cells[:, [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]].reshape(-1, 2)
    edges = np.unique(np.sort(edges, axis=1), axis=0)
    edges = np.vstack([edges, edges[:, ::-1]])
    aug = build_augmented_graph(coords, edges)
    graph = Graph(aug, np.zeros((len(coords), 3), np.float32), 'cuda')
    case = SimpleNamespace(name='mean', coords=coords, punit=np.zeros(8, np.float32))
    u = torch.zeros(a.batch, len(coords), 3, device='cuda')
    features = graph.inputs([case] * a.batch, u, u,
                            torch.zeros(a.batch, 2, device='cuda'),
                            torch.full((a.batch,), .005, device='cuda'))
    model = MeshGraphNet(NODE_IN, EDGE_IN, latent=128, processor_steps=10).cuda()
    opt = torch.optim.Adam(model.parameters())
    torch.cuda.reset_peak_memory_stats()
    model(*features).square().mean().backward()
    opt.step()
    torch.cuda.synchronize()
    print(f'PASS: batch={a.batch}, nodes={aug["n_total"]}, edges={aug["n_edges"]}, '
          f'GPU={torch.cuda.get_device_name()}, '
          f'peak reserved={torch.cuda.max_memory_reserved() / 2**30:.2f} GiB', flush=True)


if __name__ == '__main__':
    main()
