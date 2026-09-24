"""Nodal time series for surrogate training data (``lvsolver.run --save-fields``).

Everything is stored in the node/cell order of the input HDF5 mesh, so all Dalton
geometries (same topology) share one node numbering.  ``<out>/fields/``:

  coords.npy     (N, 3)      float64  reference coordinates, m
  cells.npy      (M, 4)      int32    tetrahedra
  node_tags.npy  (N,)        uint8    bit 1 endo, bit 2 epi, bit 4 base
  endo_facets.npy (F, 3)     int32    endocardial boundary triangles (cavity volume)
  fibers.npy     (M, 3)      float32  cell fibre f0 (sheets.npy: s0)
  time.npy       (T+1,)      float64  t_0 = 0 (reference state) ... t_T
  u.npy          (T+1, N, 3) float32  displacement, m
  p.npy          (T+1, N)    float32  pressure, Pa
  loads.npy      (T+1, 2)    float64  [p_endo, Ta] in kPa
  params.json                         material / load parameters of the run
"""
import json
from pathlib import Path

import h5py
import numpy as np
from scipy.spatial import cKDTree

TAG_BITS = {"endo": 1, "epi": 2, "base": 4}


def _h5_file(mesh_dir):
    mesh_dir = Path(mesh_dir)
    xdmf = sorted(mesh_dir.glob("*.xdmf"))
    if xdmf and xdmf[0].with_suffix(".h5").exists():
        return xdmf[0].with_suffix(".h5")
    return sorted(mesh_dir.glob("*.h5"))[0]


class FieldRecorder:
    def __init__(self, mesh_dir, scale, points, markers):
        """points: coordinates of the exported P1 dofs (the VTU points), m;
        markers: {"endo": tag, "epi": tag, "base": tag}."""
        with h5py.File(_h5_file(mesh_dir), "r") as f:
            self.coords = f["Mesh/mesh/geometry"][:].astype(np.float64) * scale
            self.cells = f["Mesh/mesh/topology"][:].astype(np.int32)
            self.fibers = f["Mesh/fibers"][:].astype(np.float32)
            self.sheets = f["Mesh/sheets"][:].astype(np.float32)
            ftop = f["MeshTags/facet_tags/topology"][:].astype(np.int64)
            fval = f["MeshTags/facet_tags/Values"][:].ravel()
        dist, self.idx = cKDTree(points).query(self.coords)
        if dist.max() > 1e-9 or len(np.unique(self.idx)) != len(self.coords):
            raise RuntimeError("P1 dofs do not match the mesh nodes one-to-one")
        self.node_tags = np.zeros(len(self.coords), np.uint8)
        for name, bit in TAG_BITS.items():
            self.node_tags[np.unique(ftop[fval == markers[name]])] |= bit
        self.endo_facets = ftop[fval == markers["endo"]].astype(np.int32)
        n = len(self.coords)
        self.time, self.u, self.p, self.loads = [0.0], [np.zeros((n, 3), np.float32)], [np.zeros(n, np.float32)], [None]

    def record(self, t, u_dofs, p_dofs, p_endo_kpa, ta_kpa):
        self.time.append(float(t))
        self.u.append(np.asarray(u_dofs)[self.idx].astype(np.float32))
        self.p.append(np.asarray(p_dofs)[self.idx].astype(np.float32))
        self.loads.append((float(p_endo_kpa), float(ta_kpa)))

    def write(self, out, params, load0=(0.0, 0.0)):
        d = Path(out) / "fields"
        d.mkdir(parents=True, exist_ok=True)
        self.loads[0] = load0
        for name, arr in dict(coords=self.coords, cells=self.cells, node_tags=self.node_tags,
                              endo_facets=self.endo_facets,
                              fibers=self.fibers, sheets=self.sheets, time=np.array(self.time),
                              u=np.stack(self.u), p=np.stack(self.p),
                              loads=np.array(self.loads, dtype=np.float64)).items():
            np.save(d / f"{name}.npy", arr)
        (d / "params.json").write_text(json.dumps(params, indent=2) + "\n")
