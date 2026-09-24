"""Data set of solver runs (surrogate.generate) and the graph features of one time step.

State of a trajectory: nodal displacement u_n (m); v_n = (u_n - u_{n-1})/dt_n.
Network target (per real node): Delta_n = u_{n+1} - u_n - dt v_n = u_{n+1} - 2 u_n + u_{n-1}.
Node features (NODE_IN = 18):  v_n (3) | tags endo/epi/base (3) | virtual flag (1) |
    8 parameters in [-1, 1] (surrogate/params.py) | p_endo, Ta at t_{n+1} in kPa (2) | dt (1)
Edge features (EDGE_IN = 8):   X_j - X_i, |X_j - X_i|, x_j - x_i, |x_j - x_i|   (x = X + u;
    virtual nodes do not move).
"""
import csv
import json
from pathlib import Path

import numpy as np
import torch

from .graph import aug_coords
from .params import NAMES, to_unit

NODE_IN = 18
EDGE_IN = 8


class Case:
    def __init__(self, d, row):
        f = d / "fields"
        self.name, self.geom, self.split = row["case"], row["geom"], row["split"]
        self.params = np.array([float(row[k]) for k in NAMES])
        self.punit = to_unit(self.params)[0].astype(np.float32)
        self.coords = np.load(f / "coords.npy")
        self.u = np.load(f / "u.npy", mmap_mode="r")          # (T+1, N, 3)
        self.loads = np.load(f / "loads.npy").astype(np.float32)
        self.time = np.load(f / "time.npy")
        self.steps = len(self.time) - 1
        self.dt = float(self.time[1] - self.time[0])
        self.dir = d


class LVDataset:
    def __init__(self, root):
        self.root = Path(root).resolve()
        rows = list(csv.DictReader((self.root / "params.csv").open()))
        self.cases = []
        for r in rows:
            d = self.root / r["case"]
            try:
                ok = json.loads((d / "metadata.json").read_text())["status"] == "completed"
            except (OSError, ValueError, KeyError):
                ok = False
            if ok and (d / "fields" / "u.npy").exists():
                self.cases.append(Case(d, r))
        if not self.cases:
            raise RuntimeError(f"no completed cases in {self.root}")
        f = self.cases[0].dir / "fields"
        self.cells = np.load(f / "cells.npy")
        self.node_tags = np.load(f / "node_tags.npy")
        self.endo_facets = np.load(f / "endo_facets.npy")
        for c in self.cases[1:]:
            if not np.array_equal(np.load(c.dir / "fields" / "cells.npy"), self.cells):
                raise RuntimeError(f"{c.name}: topology differs from {self.cases[0].name}")
        e = self.cells[:, [0, 0, 0, 1, 1, 2, 1, 2, 3, 2, 3, 3]].reshape(-1, 2, 6).transpose(0, 2, 1).reshape(-1, 2)
        e = np.unique(np.sort(e, axis=1), axis=0)
        self.edges = np.vstack([e, e[:, ::-1]]).astype(np.int64)
        self.tags = np.stack([(self.node_tags >> k) & 1 for k in range(3)], axis=1).astype(np.float32)
        self.n_nodes = len(self.node_tags)
        self._volume = None

    def split(self, name):
        return [c for c in self.cases if c.split == name]

    # --- cavity volume -------------------------------------------------------
    def volume_setup(self, coords):
        """Orient the endocardial triangles (normal into the solid) once, on a reference shape."""
        tri = self.endo_facets.copy()
        opp = {}
        for tet in self.cells:
            for k in range(4):
                opp[tuple(sorted(np.delete(tet, k)))] = tet[k]
        o = np.array([opp[tuple(sorted(t))] for t in tri])
        P = coords
        a, b, c = P[tri[:, 0]], P[tri[:, 1]], P[tri[:, 2]]
        flip = np.einsum("ij,ij->i", np.cross(b - a, c - a), P[o] - a) > 0
        tri[flip] = tri[flip][:, [0, 2, 1]]
        self._volume = tri
        return tri

    def cavity_volume(self, coords, u):
        """V(t) in mL for u of shape (T, N, 3): -1/3 int_endo (x - x0).n da, x0 on the base centre
        at the endocardial rim plane (same definition as tools/convergence.py; long axis x)."""
        tri = self._volume if self._volume is not None else self.volume_setup(coords)
        base = (self.node_tags & 4) > 0
        x0 = coords[base].mean(axis=0)
        x0[0] = coords[np.unique(tri), 0].max()
        x = coords[None] + np.asarray(u, dtype=np.float64) - x0
        a, b, c = x[:, tri[:, 0]], x[:, tri[:, 1]], x[:, tri[:, 2]]
        return -np.einsum("tij,tij->t", a, np.cross(b, c)) / 6 * 1e6


class Graph:
    """Augmented graph on the device, batched feature assembly."""

    def __init__(self, aug, tags, device):
        self.aug, self.device = aug, device
        self.n_real, self.n_total, self.n_edges = aug["n_real"], aug["n_total"], aug["n_edges"]
        topo = torch.as_tensor(aug["topology"], dtype=torch.long, device=device)
        self.snd, self.rcv = topo[:, 0], topo[:, 1]
        self.tags = torch.as_tensor(tags, device=device)
        self._X = {}

    def coords(self, case):
        if case.name not in self._X:
            self._X[case.name] = torch.as_tensor(aug_coords(case.coords, self.aug), dtype=torch.float32,
                                                 device=self.device)
        return self._X[case.name]

    def inputs(self, cases, u, v, load, dt):
        """cases: list of B Case; u, v: (B, n_real, 3); load: (B, 2); dt: (B,) ->
        node features (B*n_total, NODE_IN), edge features (B*n_edges, EDGE_IN), senders, receivers."""
        B, nr, nt, dev = len(cases), self.n_real, self.n_total, self.device
        x = torch.zeros(B, nt, NODE_IN, device=dev)
        x[:, :nr, 0:3] = v
        x[:, :nr, 3:6] = self.tags
        x[:, nr:, 6] = 1.0
        x[:, :, 7:15] = torch.stack([torch.as_tensor(c.punit, device=dev) for c in cases])[:, None, :]
        x[:, :, 15:17] = load[:, None, :]
        x[:, :, 17] = dt[:, None]
        X = torch.stack([self.coords(c) for c in cases])                  # (B, nt, 3)
        ua = torch.zeros(B, nt, 3, device=dev)
        ua[:, :nr] = u
        ref = X[:, self.rcv] - X[:, self.snd]
        cur = ref + ua[:, self.rcv] - ua[:, self.snd]
        e = torch.cat([ref, ref.norm(dim=-1, keepdim=True), cur, cur.norm(dim=-1, keepdim=True)], dim=-1)
        off = (torch.arange(B, device=dev) * nt).repeat_interleave(self.n_edges)
        return (x.reshape(-1, NODE_IN), e.reshape(-1, EDGE_IN),
                self.snd.repeat(B) + off, self.rcv.repeat(B) + off)


class RunningNormalizer:
    """Zero-mean unit-variance normalizer; statistics accumulated until frozen."""

    def __init__(self, dim, device="cpu"):
        self.dim, self.device = dim, device
        self.sum = torch.zeros(dim, dtype=torch.float64, device="cpu")
        self.sumsq = torch.zeros(dim, dtype=torch.float64, device="cpu")
        self.count = 0.0
        self.frozen = False
        self._cache = None

    def update(self, x):
        if self.frozen:
            return
        x = x.detach().to("cpu", torch.float64).reshape(-1, self.dim)
        self.sum += x.sum(0)
        self.sumsq += (x * x).sum(0)
        self.count += x.shape[0]
        self._cache = None

    def mean_std(self):
        if self._cache is None:
            mean = self.sum / max(self.count, 1.0)
            var = (self.sumsq / max(self.count, 1.0) - mean * mean).clamp(min=0.0)
            self._cache = (mean.float().to(self.device), (var.sqrt() + 1e-8).float().to(self.device))
        return self._cache

    def normalize(self, x):
        m, s = self.mean_std()
        return (x - m) / s

    def unnormalize(self, x):
        m, s = self.mean_std()
        return x * s + m

    def state(self):
        return dict(sum=self.sum, sumsq=self.sumsq, count=self.count)

    def load(self, st):
        self.sum, self.sumsq, self.count = st["sum"], st["sumsq"], st["count"]
        self.frozen, self._cache = True, None
