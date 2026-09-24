"""Sanity checks for an LV mesh in the lvsolver XDMF/HDF5 layout (numpy/h5py/scipy only).

  python tools/check_mesh.py mesh/<dir> [--strict]

Checks: required datasets; positive tet orientation; every boundary face tagged
exactly once with 5 (base) / 6 (endo) / 7 (epi); endo closer to the long axis
than epi; per-cell fibre/sheet/normal frame unit and orthonormal (-> stress-free
reference state); and fibre *geometry*: fraction of mid-ventricular cells whose
fibre lies within 30 deg of the local wall tangent plane, the transmural change
of the helix angle, and the face-neighbour coherence |f_i . f_j| (a fibre array
stored in a different cell order gives a coherence below that of random pairs).
The long axis is the normal of the (planar) base. Exit code 1 with --strict if
the fibre geometry fails (tangent fraction < 0.8 or helix change < 60 deg).
"""
import sys
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
from scipy.spatial import cKDTree


def main(mesh_dir, strict=False):
    f = h5py.File(next(Path(mesh_dir).glob("*_mesh.h5")), "r")
    X, T = f["Mesh/mesh/geometry"][:], f["Mesh/mesh/topology"][:]
    ft, fv = f["MeshTags/facet_tags/topology"][:], f["MeshTags/facet_tags/Values"][:].ravel()
    fb, sh, nm = (f["Mesh/" + k][:] for k in ("fibers", "sheets", "normals"))
    ok = True
    vol = np.linalg.det(X[T[:, 1:]] - X[T[:, :1]]) / 6
    print(f"{mesh_dir}: {len(X)} nodes, {len(T)} tets, wall volume {abs(vol).sum():.4g} (mesh units^3), "
          f"negative tets {(vol < 0).sum()}")
    face, adj = defaultdict(list), []
    for i, t in enumerate(T):
        for k in range(4):
            face[tuple(sorted(np.delete(t, k)))].append(i)
    bnd = {k for k, v in face.items() if len(v) == 1}
    adj = np.array([v for v in face.values() if len(v) == 2])
    tagged = [tuple(sorted(x)) for x in ft]
    print(f"  tags {dict(zip(*[a.tolist() for a in np.unique(fv, return_counts=True)]))}; boundary faces {len(bnd)}, "
          f"tagged boundary {sum(t in bnd for t in tagged)}, untagged boundary {len(bnd - set(tagged))}, "
          f"tagged non-boundary {sum(t not in bnd for t in tagged)}")
    B = X[np.unique(ft[fv == 5])]; bc = B.mean(0)
    axis = np.linalg.svd(B - bc)[2][-1]
    axis *= np.sign((bc - X.mean(0)) @ axis)            # points from apex to base
    rad = lambda P: np.linalg.norm((P - bc) - np.outer((P - bc) @ axis, axis), axis=1).mean()
    print(f"  long axis {axis.round(3)} (base plane RMS offset {np.sqrt((((B - bc) @ axis) ** 2).mean()):.2e}); "
          f"mean radius endo(6) {rad(X[np.unique(ft[fv == 6])]):.3f} < epi(7) {rad(X[np.unique(ft[fv == 7])]):.3f}")
    nrm = [np.abs(np.linalg.norm(a, axis=1) - 1).max() for a in (fb, sh, nm)]
    orth = max(np.abs(np.einsum("ij,ij->i", a, b)).max() for a, b in ((fb, sh), (fb, nm), (sh, nm)))
    print(f"  frame: max | |v|-1 | {max(nrm):.1e}, max |v_i.v_j| {orth:.1e} -> reference state stress-free: {max(nrm) < 1e-8 and orth < 1e-8}")
    c = X[T].mean(1); Ce, Cp = X[ft[fv == 6]].mean(1), X[ft[fv == 7]].mean(1)
    t = Cp[cKDTree(Cp).query(c)[1]] - Ce[cKDTree(Ce).query(c)[1]]; t /= np.linalg.norm(t, axis=1)[:, None]
    s = (c - bc) @ axis; L = -s.min(); mid = (s < -0.2 * L) & (s > -0.8 * L)
    de, dp = cKDTree(Ce).query(c)[0], cKDTree(Cp).query(c)[0]; phi = de / (de + dp)
    ep = axis - (t @ axis)[:, None] * t; ep /= np.linalg.norm(ep, axis=1)[:, None]; ec = np.cross(t, ep)
    fp = fb - np.einsum("ij,ij->i", fb, t)[:, None] * t
    ha = np.degrees(np.arctan2(np.einsum("ij,ij->i", fp, ep), np.einsum("ij,ij->i", fp, ec))); ha = (ha + 90) % 180 - 90
    q = np.quantile(phi[mid], [0, .2, .4, .6, .8, 1])
    hq = [float(np.median(ha[mid & (phi >= a) & (phi <= b)])) for a, b in zip(q[:-1], q[1:])]
    tang = (np.abs(np.einsum("ij,ij->i", fb, t))[mid] < 0.5).mean()
    coh = np.median(np.abs(np.einsum("ij,ij->i", fb[adj[:, 0]], fb[adj[:, 1]])))
    rp = np.random.default_rng(0).integers(0, len(T), size=(len(adj), 2))
    cr = np.median(np.abs(np.einsum("ij,ij->i", fb[rp[:, 0]], fb[rp[:, 1]])))
    print(f"  fibre geometry (mid-ventricle, {mid.sum()} cells): within 30 deg of wall tangent plane {tang:.2f}; "
          f"helix angle endo->epi quintiles {[round(h, 1) for h in hq]}; "
          f"neighbour coherence {coh:.2f} (random pairs {cr:.2f})")
    good = tang >= 0.8 and abs(hq[0] - hq[-1]) >= 60 and coh > cr
    print(f"  fibre geometry plausible: {good}")
    return 0 if (good or not strict) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], "--strict" in sys.argv))
