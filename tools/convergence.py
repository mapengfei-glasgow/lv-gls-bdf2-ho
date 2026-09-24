"""Summarise a mesh-refinement study: results_convergence/<case>_h<size>/ -> convergence.md

For every level: cells, nodes, h = (|Omega|/N_cells)^(1/3), J statistics,
cavity volume V(t) (exact surface integral over the endocardial facets of the
VTU frames, not the Delaunay estimate V_mL of monitor.csv) and max|u|(t).  Differences are taken
against the finest level; the observed order is the least-squares slope of
log|f_h - f_finest| vs log h over the non-finest levels.
Usage: python tools/convergence.py results_convergence
"""
import csv
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import h5py
import meshio
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

KEY_T = (0.2, 0.5, 0.65, 0.8, 1.0)


def cavity_volume(d):
    """Exact polyhedral cavity volume from the endocardial facets (tag 6) of every VTU frame:
    V = -1/3 int_endo (x - x0).n da with n the solid's outward normal; x0 on the endo base-rim plane."""
    d = Path(d)
    meta = json.loads((d / "metadata.json").read_text())
    ax = "xyz".index(meta["long_axis"])
    mp = meta.get("mesh_path")
    mdir = Path(mp) if mp and Path(mp).is_dir() else ROOT / "mesh" / meta["mesh"]
    h5 = next(mdir.glob("*_mesh.h5"))
    with h5py.File(h5, "r") as f:
        xyz = f["Mesh/mesh/geometry"][:] * meta["mesh_scale"]
        ft = f["MeshTags/facet_tags/topology"][:].astype(np.int64)
        fv = f["MeshTags/facet_tags/Values"][:].ravel()
    frames = ET.parse(next((d / "vtk").glob("*.pvd"))).findall(".//DataSet")
    fr0 = meshio.read(d / "vtk" / frames[0].attrib["file"])
    P, cells = fr0.points, fr0.cells[0].data
    if fr0.cells[0].type != "tetra":
        raise ValueError("P1 output expected")
    key = lambda a: [tuple(r) for r in np.round(a / 1e-9).astype(np.int64)]
    pos = {k: i for i, k in enumerate(key(P))}
    h2v = np.array([pos[k] for k in key(xyz)])
    tri = h2v[ft[fv == 6]]
    opp = {}
    for tet in cells:
        for k in range(4):
            opp[tuple(sorted(np.delete(tet, k)))] = tet[k]
    o = np.array([opp[tuple(sorted(t))] for t in tri])
    a, b, c = P[tri[:, 0]], P[tri[:, 1]], P[tri[:, 2]]
    flip = np.einsum("ij,ij->i", np.cross(b - a, c - a), P[o] - a) > 0   # normal points into the solid
    tri[flip] = tri[flip][:, [0, 2, 1]]
    x0 = np.array(meta["base_center_m"], dtype=float)
    x0[ax] = P[np.unique(tri), ax].max()                                   # endo rim plane
    V = []
    for ds in frames:
        x = P + meshio.read(d / "vtk" / ds.attrib["file"]).point_data["displacement"] - x0
        V.append(-np.einsum("ij,ij->i", x[tri[:, 0]], np.cross(x[tri[:, 1]], x[tri[:, 2]])).sum() / 6)
    X = P - x0
    V0 = -np.einsum("ij,ij->i", X[tri[:, 0]], np.cross(X[tri[:, 1]], X[tri[:, 2]])).sum() / 6
    return V0 * 1e6, np.array(V) * 1e6


def load(d):
    rows = list(csv.DictReader((d / "monitor.csv").open()))
    col = lambda k: np.array([float(r[k]) for r in rows])
    frame = meshio.read(next((d / "vtk").glob("*_00000.vtu")))
    X = frame.points[frame.cells[0].data]
    vol = np.abs(np.linalg.det(X[:, 1:] - X[:, :1])) / 6
    val = json.loads((d / "validation.json").read_text())
    V0, V = cavity_volume(d)
    return dict(name=d.name, cells=len(vol), nodes=len(frame.points), h=(vol.sum() / len(vol)) ** (1 / 3),
                t=col("time_s"), V0=V0, V=V, u=col("max_u_m"), rms=col("J_minus_1_RMS"),
                jmean=col("J_mean"), pmax=col("p_max_Pa"), val=val)


def order(hs, errs):
    m = [(h, e) for h, e in zip(hs, errs) if e > 0]
    if len(m) < 2:
        return float("nan")
    return float(np.polyfit(np.log([h for h, _ in m]), np.log([e for _, e in m]), 1)[0])


def main(root):
    root = Path(root)
    lv = sorted((load(d) for d in root.iterdir() if (d / "validation.json").exists()), key=lambda r: -r["h"])
    if not lv:
        sys.exit("no completed levels")
    f = lv[-1]
    idx = [int(np.argmin(abs(f["t"] - t))) for t in KEY_T]
    L = [f"# Mesh refinement: {lv[0]['name'].rsplit('_h', 1)[0]}", "",
         "Same dt = 0.005 s and stabilisation parameters on every level; reference = finest level.",
         "V = exact cavity volume -1/3 int_endo (x-x0).n da from the VTU frames (V ref: reference configuration).", "",
         "| level | cells | nodes | h (mm) | max Newton | J min | J max | max RMS(J-1) | max abs(J_mean-1) | V ref mL | "
         + " | ".join(f"V({t}) mL" for t in KEY_T) + " | max abs(u) (mm) |",
         "|---" * (11 + len(KEY_T)) + "|"]
    for r in lv:
        v = r["val"]
        L.append(f"| {r['name']} | {r['cells']} | {r['nodes']} | {1e3 * r['h']:.2f} | {v['max_newton_iterations']} | "
                 f"{v['J_quad_min']:.3f} | {v['J_quad_max']:.3f} | {v['max_J_minus_1_RMS']:.4f} | "
                 f"{v['max_J_mean_drift']:.2e} | {r['V0']:.2f} | " + " | ".join(f"{r['V'][i]:.2f}" for i in idx)
                 + f" | {1e3 * r['u'].max():.3f} |")
    L += ["", "## Difference to the finest level (max over the cycle, relative)", "",
          "| level | h (mm) | V(t) | max abs(u)(t) | RMS(J-1)(t), absolute |", "|---|---|---|---|---|"]
    eV, eU, eJ = [], [], []
    for r in lv[:-1]:
        eV.append(np.max(np.abs(r["V"] - f["V"])) / np.max(f["V"]))
        eU.append(np.max(np.abs(r["u"] - f["u"])) / np.max(f["u"]))
        eJ.append(np.max(r["rms"]))
        L.append(f"| {r['name']} | {1e3 * r['h']:.2f} | {eV[-1]:.2e} | {eU[-1]:.2e} | {eJ[-1]:.4f} |")
    hs = [r["h"] for r in lv[:-1]]
    L += ["", f"Observed order (slope of log error vs log h): V {order(hs, eV):.2f}, max abs(u) {order(hs, eU):.2f}; "
          f"max RMS(J-1) itself vs h over all levels: {order([r['h'] for r in lv], [np.max(r['rms']) for r in lv]):.2f} "
          "(a consistent incompressible scheme should give a positive slope, i.e. RMS(J-1) -> 0 as h -> 0)."]
    (root / "convergence.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results_convergence")
