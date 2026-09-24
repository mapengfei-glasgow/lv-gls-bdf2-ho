"""Independent check of one result directory (numpy + meshio only, no DOLFINx).

Always: completed status, frame/row/time consistency, finite data, SNES converged.
P1 velocity/displacement output (p1, gls P1/P1) additionally:
  - BE/BDF2 kinematic recurrence between exported u and v,
  - J, J_mean, RMS(J-1) and kinetic energy rebuilt from the nodal fields,
  - stabilisation scales (mu_eff, tau_m, tau_c or tau_reference) rebuilt from F.
Writes validation.json.   Usage: python -m lvsolver.verify results/<case>
"""
import argparse
import csv
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import meshio
import numpy as np


def f(rows, key):
    return [float(r[key]) for r in rows if r.get(key, "") != ""]


def verify(root):
    root = Path(root)
    meta = json.loads((root / "metadata.json").read_text())
    rows = list(csv.DictReader((root / "monitor.csv").open()))
    datasets = ET.parse(next((root / "vtk").glob("*.pvd"))).findall(".//DataSet")
    assert meta["status"] == "completed", meta
    assert len(rows) == len(datasets) == meta["completed_steps"] == meta["requested_steps"]
    dt, scheme = meta["dt_s"], meta["time_scheme"]
    err = dict(kinematic=0.0, J=0.0, kinetic=0.0, scales=0.0)
    p1_checks = False
    u_n = u_nn = None
    for i, (ds, row) in enumerate(zip(datasets, rows)):
        assert all(np.isfinite(float(v)) for v in row.values() if v != "")
        assert int(row["snes_reason"]) > 0
        assert abs(float(row["time_s"]) - (i + 1) * dt) < 1e-12
        assert abs(float(ds.attrib["timestep"]) - float(row["time_s"])) < 1e-12
        frame = meshio.read(root / "vtk" / ds.attrib["file"])
        for value in frame.point_data.values():
            assert np.isfinite(value).all()
        for blocks in frame.cell_data.values():
            assert all(np.isfinite(b).all() for b in blocks)
        if frame.cells[0].type != "tetra":
            continue                      # P2 output: generic checks only
        p1_checks = True
        u, v = frame.point_data["displacement"], frame.point_data["velocity"]
        if u_n is None:
            u_n, u_nn = np.zeros_like(u), np.zeros_like(u)
        if i == 0 or scheme == "be":
            disc = u - u_n - dt * v
        else:
            disc = 3 * u - 4 * u_n + u_nn - 2 * dt * v
        err["kinematic"] = max(err["kinematic"], float(np.abs(disc).max()))
        cells = frame.cells[0].data
        X = frame.points[cells]
        D0 = X[:, 1:, :] - X[:, :1, :]
        D1 = (frame.points + u)[cells][:, 1:, :] - (frame.points + u)[cells][:, :1, :]
        J = np.linalg.det(D1) / np.linalg.det(D0)
        assert J.min() > 0
        err["J"] = max(err["J"], float(np.abs(J - frame.cell_data["J"][0]).max()))
        vol = np.abs(np.linalg.det(D0)) / 6
        assert np.isclose(np.dot(vol, J) / vol.sum(), float(row["J_mean"]), rtol=1e-9, atol=1e-11)
        assert np.isclose(np.sqrt(np.dot(vol, (J - 1)**2) / vol.sum()), float(row["J_minus_1_RMS"]),
                          rtol=1e-9, atol=1e-11)
        vv = v[cells]
        kin = 0.5 * meta["rho_kg_m3"] * np.sum(vol / 20 * (np.sum(vv.sum(axis=1)**2, axis=1)
                                                          + np.sum(vv**2, axis=(1, 2))))
        err["kinetic"] = max(err["kinetic"], abs(kin - float(row["kinetic_energy_J"])))
        h = np.max(np.linalg.norm(X[:, :, None, :] - X[:, None, :, :], axis=-1), axis=(1, 2))
        rho = meta["rho_kg_m3"]
        expected = {}
        if "tau_reference" in frame.cell_data:
            expected["tau_reference"] = 1 / (meta["c1"] * meta["mu_ref_Pa"] / h**2 + meta["c2"] * rho / dt**2)
        if "mu_eff_Pa" in frame.cell_data:
            F = np.swapaxes(np.linalg.solve(D0, D1), 1, 2)
            C = np.swapaxes(F, 1, 2) @ F
            f0, s0 = frame.cell_data["f0"][0], frame.cell_data["s0"][0]
            I1 = J**(-2 / 3) * np.trace(C, axis1=1, axis2=2)
            I4f = np.einsum("ni,nij,nj->n", f0, C, f0)
            I4s = np.einsum("ni,nij,nj->n", s0, C, s0)
            I8 = np.einsum("ni,nij,nj->n", f0, C, s0)
            eps = meta["positive_part_epsilon"]
            xf = .5 * (I4f - 1 + np.sqrt((I4f - 1)**2 + eps))
            xs = .5 * (I4s - 1 + np.sqrt((I4s - 1)**2 + eps))
            m = meta["material_parameters"]
            mu = (m["a"] * np.exp(m["b"] * (I1 - 3)) + 2 * m["a_f"] * xf * np.exp(m["b_f"] * xf**2)
                  + 2 * m["a_s"] * xs * np.exp(m["b_s"] * xs**2))
            if meta["include_fs_stiffness"]:
                mu += 2 * m["a_fs"] * np.sqrt(I8**2 + eps) * np.exp(m["b_fs"] * I8**2)
            omega = meta["c_t"] * rho / dt + meta["c_s"] * mu * dt / h**2
            expected.update(mu_eff_Pa=mu, tau_m=meta["C_tau"] / omega, tau_c=meta["C_c"] * h**2 * omega)
        for name, ref in expected.items():
            np.testing.assert_allclose(frame.cell_data[name][0], ref, rtol=1e-8, atol=1e-12)
            err["scales"] = max(err["scales"], float(np.max(np.abs(frame.cell_data[name][0] / ref - 1))))
        u_nn, u_n = u_n, u.copy()
    if p1_checks:
        assert err["kinematic"] < 1e-11, err
        assert err["J"] < 1e-9, err
        assert err["kinetic"] < 1e-9, err
    summary = dict(
        method=meta["method"], frames=len(rows), end_time_s=float(rows[-1]["time_s"]),
        all_steps_converged=True, independent_P1_checks=p1_checks,
        max_newton_iterations=max(int(r["newton_iterations"]) for r in rows),
        J_quad_min=min(f(rows, "J_quad_min")), J_quad_max=max(f(rows, "J_quad_max")),
        max_J_minus_1_RMS=max(f(rows, "J_minus_1_RMS")),
        final_J_mean=f(rows, "J_mean")[-1], max_J_mean_drift=max(abs(x - 1) for x in f(rows, "J_mean")),
        V_min_mL=min(f(rows, "V_mL")), V_max_mL=max(f(rows, "V_mL")), V_final_mL=f(rows, "V_mL")[-1],
        p_min_Pa=min(f(rows, "p_min_Pa"), default=None), p_max_Pa=max(f(rows, "p_max_Pa"), default=None),
        max_displacement_m=max(f(rows, "max_u_m")),
        max_total_residual=max(f(rows, "total_residual_norm"), default=None),
        max_kinematic_recurrence_error_m=err["kinematic"] if p1_checks else None,
        max_independent_J_error=err["J"] if p1_checks else None,
        max_independent_kinetic_energy_error_J=err["kinetic"] if p1_checks else None,
        max_stabilisation_scale_rel_error=err["scales"] if p1_checks else None)
    (root / "validation.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def report(results):
    """Write results/summary.md: one row per case from validation.json."""
    results = Path(results)
    cols = [("method", "method"), ("frames", "steps"), ("max_newton_iterations", "max Newton"),
            ("J_quad_min", "J min"), ("J_quad_max", "J max"), ("max_J_minus_1_RMS", "max RMS(J-1)"),
            ("max_J_mean_drift", "max |J_mean-1|"), ("V_final_mL", "V end (mL)"),
            ("max_displacement_m", "max |u| (m)"), ("independent_P1_checks", "P1 checks")]
    lines = ["# Results summary", "", "Generated by `python -m lvsolver.verify --report results`.", "",
             "| case | " + " | ".join(c[1] for c in cols) + " |", "|---" * (len(cols) + 1) + "|"]
    for vfile in sorted(results.glob("*/validation.json")):
        s = json.loads(vfile.read_text())
        cells = [f"{s[k]:.6g}" if isinstance(s[k], float) else str(s[k]) for k, _ in cols]
        lines.append(f"| {vfile.parent.name} | " + " | ".join(cells) + " |")
    (results / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("path", type=Path)
    ap.add_argument("--report", action="store_true", help="path is the results root; write summary.md")
    a = ap.parse_args()
    if a.report:
        report(a.path)
    else:
        print(json.dumps(verify(a.path), indent=2))
