"""Extract the Sec. 4.6 time series from one LV result directory.

  python extract_data.py <result_dir> [out.csv]     (default data/lv_timeseries.csv)

Columns: loading, exact cavity volume (surface integral over the endocardial
facets, tools/convergence.py:cavity_volume), long-axis length (base centre to the
most apical epicardial vertex), equatorial wall thickness (mean radial distance
epi - endo of the facet vertices within +-6 mm of X1 = -3 cm), J_mean, Newton
iterations, max|u|, pressure extrema, kinetic energy.  Row 0 is the reference state.
"""
import csv
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import h5py
import meshio
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from convergence import cavity_volume  # noqa: E402


def main(d, out):
    d = Path(d)
    meta = json.loads((d / "metadata.json").read_text())
    ax = "xyz".index(meta["long_axis"])
    V0, V = cavity_volume(d)
    rows = list(csv.DictReader((d / "monitor.csv").open()))
    g = lambda k: np.array([float(r[k]) for r in rows])
    frames = ET.parse(next((d / "vtk").glob("*.pvd"))).findall(".//DataSet")
    P = meshio.read(d / "vtk" / frames[0].attrib["file"]).points
    mp = meta.get("mesh_path")
    mdir = Path(mp) if mp and Path(mp).is_dir() else ROOT / "mesh" / meta["mesh"]
    with h5py.File(next(mdir.glob("*_mesh.h5")), "r") as f:
        xyz = f["Mesh/mesh/geometry"][:] * meta["mesh_scale"]
        ft = f["MeshTags/facet_tags/topology"][:]
        fv = f["MeshTags/facet_tags/Values"][:].ravel()
    key = lambda a: [tuple(r) for r in np.round(a / 1e-9).astype(np.int64)]
    pos = {k: i for i, k in enumerate(key(P))}
    h2v = np.array([pos[k] for k in key(xyz)])
    endo, epi = np.unique(h2v[ft[fv == 6]]), np.unique(h2v[ft[fv == 7]])
    apex = epi[np.argmin(P[epi, ax])]
    cross = [i for i in range(3) if i != ax]
    band = lambda n: n[np.abs(P[n, ax] + 0.03) < 0.006]
    eb, pb = band(endo), band(epi)
    radius = lambda x, n: np.sqrt((x[n][:, cross] ** 2).sum(axis=1)).mean()
    base = meta["base_center_m"][ax]
    L, T = [base - P[apex, ax]], [radius(P, pb) - radius(P, eb)]
    for ds in frames:
        x = P + meshio.read(d / "vtk" / ds.attrib["file"]).point_data["displacement"]
        L.append(base - x[apex, ax])
        T.append(radius(x, pb) - radius(x, eb))
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["time_s", "p_endo_kPa", "Ta_kPa", "V_cavity_mL", "long_axis_length_mm",
                    "equatorial_wall_thickness_mm", "J_mean", "newton_iterations", "max_u_mm",
                    "p_min_kPa", "p_max_kPa", "kinetic_energy_mJ"])
        w.writerow([0, 0, 0, V0, 1e3 * L[0], 1e3 * T[0], 1, 0, 0, 0, 0, 0])
        for i, r in enumerate(rows):
            w.writerow([g("time_s")[i], g("p_endo_kPa")[i], g("Ta_kPa")[i], V[i], 1e3 * L[i + 1], 1e3 * T[i + 1],
                        g("J_mean")[i], int(g("newton_iterations")[i]), 1e3 * g("max_u_m")[i],
                        1e-3 * g("p_min_Pa")[i], 1e-3 * g("p_max_Pa")[i], 1e3 * g("kinetic_energy_J")[i]])
    t, Vv, Ll, Tt = g("time_s"), np.array(V), np.array(L[1:]), np.array(T[1:])
    ed, es = int(np.argmin(abs(t - 0.5))), int(np.argmin(np.where(t < 0.8, Vv, np.inf)))
    print(json.dumps(dict(
        mesh=meta["mesh"], cells_band=[len(eb), len(pb)], include_fs=meta.get("include_fs_stiffness"),
        steps=len(rows), newton_max=int(g("newton_iterations").max()),
        newton_counts={k: int((g("newton_iterations") == k).sum()) for k in range(1, int(g("newton_iterations").max()) + 1)
                       if (g("newton_iterations") == k).any()},
        max_residual=float(g("total_residual_norm").max()), V_ref=V0, V_ED=Vv[ed], t_ES=t[es], V_ES=Vv[es],
        EF=100 * (Vv[ed] - Vv[es]) / Vv[ed], L0=1e3 * L[0], L_ED=1e3 * Ll[ed], L_ES=1e3 * Ll[es],
        shortening=100 * (Ll[ed] - Ll[es]) / Ll[ed], T0=1e3 * T[0], T_ED=1e3 * Tt[ed], T_ES=1e3 * Tt[es],
        thickening=100 * (Tt[es] - Tt[ed]) / Tt[ed], max_u_mm=1e3 * g("max_u_m").max(),
        t_max_u=float(t[g("max_u_m").argmax()]), p_min_kPa=1e-3 * g("p_min_Pa").min(),
        p_max_kPa=1e-3 * g("p_max_Pa").max(), J_mean_drift=float(np.abs(g("J_mean") - 1).max()),
        h_range_mm=[1e3 * meta["h_min_m"], 1e3 * meta["h_max_m"]]), indent=1))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else HERE / "data" / "lv_timeseries.csv")
