"""Rule-based myocardial fibres for an LV mesh in the lvsolver XDMF/HDF5 layout.

  python geometry/fibers.py <src_mesh_dir> <dst_mesh_dir> [--endo 60 --epi -60]

Copies the mesh and replaces Mesh/fibers, Mesh/sheets, Mesh/normals (per cell):
  1. transmural coordinate phi: P1 Laplace problem, phi = 0 on the endocardium (tag 6),
     phi = 1 on the epicardium (tag 7), homogeneous Neumann on the base (tag 5);
  2. per cell: e_t = grad(phi)/|grad(phi)| (transmural), e_l = long axis projected on
     the wall tangent plane (long axis = normal of the planar base, apex -> base),
     e_c = e_t x e_l (circumferential);
  3. helix angle alpha = alpha_endo + (alpha_epi - alpha_endo) * phi_cell,
     f0 = cos(alpha) e_c + sin(alpha) e_l,  s0 = e_t,  n0 = f0 x s0  (det[f0,s0,n0] = +1).
Same conventions as tools/generate_mesh.py (+60 deg endo -> -60 deg epi, sheet transmural).
Near the apex, where e_t is parallel to the long axis, e_l falls back to a fixed
direction orthogonal to e_t. Only numpy/scipy/h5py are needed.
"""
import argparse
import json
import shutil
from pathlib import Path

import h5py
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def unit(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def p1_gradients(X, T):
    """Barycentric gradients (nc, 4, 3) and volumes of P1 tetrahedra."""
    D = X[T[:, 1:]] - X[T[:, :1]]                      # (nc, 3, 3), rows = edges
    Dinv = np.linalg.inv(D)                            # grad(lambda_i) = Dinv[:, :, i-1], i = 1..3
    G = np.empty((len(T), 4, 3))
    G[:, 1:, :] = np.transpose(Dinv, (0, 2, 1))
    G[:, 0, :] = -G[:, 1:, :].sum(axis=1)
    return G, np.abs(np.linalg.det(D)) / 6


def laplace(X, T, G, vol, dirichlet):
    """Solve -div grad u = 0 with u = value on the given node sets (dict node_ids -> value)."""
    Ke = vol[:, None, None] * np.einsum("cik,cjk->cij", G, G)
    rows = np.repeat(T, 4, axis=1).ravel(); cols = np.tile(T, (1, 4)).ravel()
    K = sp.csr_matrix((Ke.ravel(), (rows, cols)), shape=(len(X),) * 2)
    u = np.zeros(len(X)); fixed = np.zeros(len(X), bool)
    for nodes, value in dirichlet:
        u[nodes] = value; fixed[nodes] = True
    free = ~fixed
    rhs = -K[free][:, fixed] @ u[fixed]
    u[free] = spla.spsolve(K[free][:, free].tocsc(), rhs)
    return u


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", type=Path); ap.add_argument("dst", type=Path)
    ap.add_argument("--endo", type=float, default=60.0); ap.add_argument("--epi", type=float, default=-60.0)
    a = ap.parse_args()
    a.dst.mkdir(parents=True, exist_ok=True)
    for p in a.src.iterdir():
        if p.suffix in (".h5", ".xdmf"):
            shutil.copy2(p, a.dst / p.name)
    h5 = next(a.dst.glob("*_mesh.h5"))
    with h5py.File(h5, "r+") as f:
        X, T = f["Mesh/mesh/geometry"][:], f["Mesh/mesh/topology"][:].astype(np.int64)
        ft, fv = f["MeshTags/facet_tags/topology"][:].astype(np.int64), f["MeshTags/facet_tags/Values"][:].ravel()
        endo, epi, base = (np.unique(ft[fv == k]) for k in (6, 7, 5))
        both = np.intersect1d(endo, epi)
        if len(both):
            raise RuntimeError(f"{len(both)} nodes are on both endo and epi")
        G, vol = p1_gradients(X, T)
        phi = laplace(X, T, G, vol, [(endo, 0.0), (epi, 1.0)])
        # long axis: normal of the planar base, oriented apex -> base
        B = X[base]; bc = B.mean(0)
        axis = np.linalg.svd(B - bc)[2][-1]; axis *= np.sign((bc - X.mean(0)) @ axis)
        et = unit(np.einsum("ci,cik->ck", phi[T], G))
        el = axis - (et @ axis)[:, None] * et
        weak = np.linalg.norm(el, axis=1) < 1e-3              # apex: e_t parallel to the axis
        if weak.any():
            ref = np.eye(3)[np.argmin(np.abs(axis))]
            el[weak] = ref - (et[weak] @ ref)[:, None] * et[weak]
        el = unit(el); ec = np.cross(et, el)
        phic = phi[T].mean(1)
        alpha = np.radians(a.endo + (a.epi - a.endo) * phic)
        f0 = unit(np.cos(alpha)[:, None] * ec + np.sin(alpha)[:, None] * el)
        s0 = et; n0 = unit(np.cross(f0, s0))
        for name, arr in (("fibers", f0), ("sheets", s0), ("normals", n0)):
            del f["Mesh/" + name]
            f.create_dataset("Mesh/" + name, data=arr.astype(np.float64), compression="gzip", compression_opts=4)
    info = dict(source=str(a.src), method="rule-based: P1 Laplace transmural phi (endo 0, epi 1); "
                "e_l = base-normal axis projected on wall; f0 = cos(a) e_c + sin(a) e_l; s0 = grad(phi); n0 = f0 x s0",
                helix_endo_deg=a.endo, helix_epi_deg=a.epi, long_axis=axis.round(6).tolist(),
                apex_fallback_cells=int(weak.sum()), phi_range=[float(phi.min()), float(phi.max())])
    (a.dst / "fibres.json").write_text(json.dumps(info, indent=2) + "\n")
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
