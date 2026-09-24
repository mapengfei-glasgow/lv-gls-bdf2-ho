"""Run one LV case and write monitor.csv, metadata.json and VTU frames.

  python -m lvsolver.run --method p1 --out results/p1
  python -m lvsolver.run --method lagrange --beta 0 --out results/lagrange_pure
  python -m lvsolver.run --method gls --tau pdf --out results/gls_pdf
  python -m lvsolver.run --method gls --tau reference --no-stab --steps 3 --out results/x
  python -m lvsolver.run --method gls --include-fs --set a=300 --ta-amp 70 --period 0.8 \
      --dt 0.004 --no-vtk --save-fields --out results/y      # surrogate training data
"""
import argparse
import functools
import csv
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import basix
import meshio
import numpy as np
import ufl
from dolfinx import __version__ as dolfinx_version, fem, plot
from mpi4py import MPI
from scipy.spatial import cKDTree

from .material import MATERIALS
from .mesh import LVGeometry, LeftVentricleCapacity, compute_base_center
from .export import FieldRecorder
from .model import CONSTITUTIVE_PARAMS, KPA_TO_PA, P_AMP_KPA, P_ED_KPA, TA_AMP_KPA, Ta_kPa, p_endo_kPa
from .problems import METHODS

ROOT = Path(__file__).resolve().parents[1]
FIELDS = ["time_s", "p_endo_kPa", "Ta_kPa", "V_mL",
          "J_center_min", "J_center_max", "J_quad_min", "J_quad_max", "J_minus_1_RMS", "J_mean",
          "max_u_m", "max_v_m_per_s", "p_min_Pa", "p_max_Pa", "kinetic_energy_J",
          "newton_iterations", "snes_reason",
          "galerkin_p_norm", "pspg_norm", "lsic_norm", "galerkin_v_norm", "total_residual_norm",
          "constraint_rate_RMS_per_s"]


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--method", choices=sorted(METHODS), required=True)
    ap.add_argument("--out", type=Path, required=True, help="new or empty output directory")
    g = ap.add_argument_group("time")
    g.add_argument("--dt", type=float, default=0.005)
    g.add_argument("--t-end", type=float, default=None, help="default: one cycle (--period)")
    g.add_argument("--steps", type=int, default=None, help="default round(t_end/dt)")
    g = ap.add_argument_group("mesh")
    g.add_argument("--mesh", type=Path, default=ROOT / "mesh" / "lv_aha17_v2_plain")
    g.add_argument("--scale", type=float, default=0.01, help="mesh units -> m")
    g.add_argument("--long-axis", default="x")
    g.add_argument("--tags", type=int, nargs=3, default=(6, 7, 5), metavar=("ENDO", "EPI", "BASE"))
    g = ap.add_argument_group("material")
    g.add_argument("--material", choices=sorted(MATERIALS), default=None,
                   help="default: ho_pdf for gls --tau pdf, ho otherwise")
    g.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                   help="override a constitutive constant, e.g. --set a=300 --set b_f=2.0 (Pa or -)")
    g = ap.add_argument_group("loads (Eqs. 79/80)")
    g.add_argument("--p-ed", type=float, default=P_ED_KPA, help="diastolic pressure plateau, kPa")
    g.add_argument("--p-amp", type=float, default=P_AMP_KPA, help="systolic pressure amplitude, kPa")
    g.add_argument("--ta-amp", type=float, default=TA_AMP_KPA, help="active tension amplitude, kPa")
    g.add_argument("--period", type=float, default=1.0, help="cycle length T, s; curves evaluated at t/T")
    g = ap.add_argument_group("output")
    g.add_argument("--no-vtk", action="store_true", help="do not write VTU frames")
    g.add_argument("--save-fields", action="store_true",
                   help="write nodal u, p time series to <out>/fields/ (P1 only; see lvsolver/export.py)")
    g = ap.add_argument_group("lagrange")
    g.add_argument("--beta", type=float, default=1e6, help="augmentation beta/2 (J-1)^2, Pa; 0 = pure multiplier")
    g = ap.add_argument_group("gls")
    g.add_argument("--tau", choices=("pdf", "reference"), default="pdf")
    g.add_argument("--no-stab", action="store_true", help="Galerkin control; stabilisation only diagnosed")
    g.add_argument("--velocity-degree", type=int, choices=(1, 2), default=1)
    g.add_argument("--time-scheme", choices=("bdf2", "be"), default="bdf2")
    g.add_argument("--ct", type=float, default=1.5)
    g.add_argument("--cs", type=float, default=3.0)
    g.add_argument("--ctau", type=float, default=0.03)
    g.add_argument("--cc", type=float, default=0.1)
    g.add_argument("--include-fs", action="store_true", help="add the fs coupling to mu_eff")
    g.add_argument("--c1", type=float, default=1.0)
    g.add_argument("--c2", type=float, default=2.0)
    g.add_argument("--mu-ref", type=float, default=None, help="default a+a_f+a_s+a_fs")
    args = ap.parse_args(argv)
    args.params = dict(CONSTITUTIVE_PARAMS)
    for item in args.set:
        key, _, value = item.partition("=")
        if key not in CONSTITUTIVE_PARAMS or not value:
            ap.error(f"--set {item}: expected KEY=VALUE with KEY in {sorted(CONSTITUTIVE_PARAMS)}")
        args.params[key] = float(value)
    args.p_curve = functools.partial(p_endo_kPa, p_ed=args.p_ed, p_amp=args.p_amp, period=args.period)
    args.ta_curve = functools.partial(Ta_kPa, ta_amp=args.ta_amp, period=args.period)
    if args.t_end is None:
        args.t_end = args.period
    return args


def build(args, geo, base_center):
    material_name = args.material or ("ho_pdf" if args.method == "gls" and args.tau == "pdf" else "ho")
    material = MATERIALS[material_name](geo.f0, geo.s0, geo.msh, parameters=args.params)
    loads = (lambda t: args.p_curve(t) * KPA_TO_PA, lambda t: args.ta_curve(t) * KPA_TO_PA)
    common = dict(dt=args.dt, loads=loads, base_center=base_center, long_axis=args.long_axis)
    if args.method == "lagrange":
        extra = dict(beta=args.beta)
    elif args.method == "gls":
        extra = dict(tau=args.tau, stabilized=not args.no_stab, velocity_degree=args.velocity_degree,
                     time_scheme=args.time_scheme, c_t=args.ct, c_s=args.cs, C_tau=args.ctau, C_c=args.cc,
                     include_fs=args.include_fs, c1=args.c1, c2=args.c2, mu_ref=args.mu_ref)
    else:
        extra = {}
    return material_name, METHODS[args.method](geo, material, **common, **extra)


def write_pvd(path, stem, times):
    root = ET.Element("VTKFile", type="Collection", version="0.1", byte_order="LittleEndian")
    col = ET.SubElement(root, "Collection")
    for i, t in enumerate(times):
        ET.SubElement(col, "DataSet", timestep=f"{t:.12g}", group="", part="0", file=f"{stem}_{i:05d}.vtu")
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)


def main(argv=None):
    args = parse_args(argv)
    if MPI.COMM_WORLD.size != 1:
        raise RuntimeError("The meshio VTU exporter requires a single MPI rank")
    nsteps = args.steps if args.steps is not None else round(args.t_end / args.dt)
    if args.dt <= 0 or nsteps < 1:
        raise ValueError("positive dt and steps required")
    out, vtk_dir = args.out, args.out / "vtk"
    out.mkdir(parents=True, exist_ok=True)
    if not args.no_vtk:
        vtk_dir.mkdir(exist_ok=True)
    if any(vtk_dir.glob("*.vtu")) or (out / "fields").exists():
        raise RuntimeError(f"{vtk_dir} already has frames: choose a new --out")

    endo, epi, base = args.tags
    geo = LVGeometry(args.mesh, scale=args.scale, markers={"epi": epi, "base": base, "endo": endo})
    mesh = geo.msh
    center = compute_base_center(mesh, geo.bdry, base, MPI.COMM_WORLD)
    material_name, problem = build(args, geo, center)
    capacity = LeftVentricleCapacity(mesh, geo.bdry, geo.bdry_markers, ("CG", problem.degree, (3,)))

    metadata = dict(problem.configuration, material=material_name, dt_s=args.dt, requested_steps=nsteps,
                    material_parameters=args.params,
                    loads=dict(p_ed_kPa=args.p_ed, p_amp_kPa=args.p_amp, ta_amp_kPa=args.ta_amp,
                               period_s=args.period), rho_kg_m3=problem.parameters["rho"],
                    mesh=Path(args.mesh).name, mesh_path=str(Path(args.mesh).resolve()),
                    mesh_scale=args.scale, long_axis=args.long_axis,
                    base_center_m=[float(c) for c in center],
                    base_constraint="axial/tangential penalty gamma_top; radial free",
                    dolfinx=dolfinx_version, status="running", completed_steps=0)
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print("configuration: " + json.dumps(metadata), flush=True)

    # --- output mappings (VTU points = dofs of the displacement space) ------
    topology, cell_types, points = plot.vtk_mesh(problem.u_space)
    if problem.degree == 1:
        assert np.all(cell_types == 10)
        cells = [("tetra", topology.reshape(-1, 5)[:, 1:])]
    else:
        assert np.all(cell_types == 71)
        cells = [("VTK_LAGRANGE_TETRAHEDRON", topology.reshape(-1, 11)[:, 1:])]
    recorder = None
    if args.save_fields:
        if problem.degree != 1 or problem.p is None:
            raise ValueError("--save-fields needs the P1/P1 velocity-pressure output (gls, velocity degree 1)")
        recorder = FieldRecorder(args.mesh, args.scale, points, {"epi": epi, "base": base, "endo": endo})
    if problem.p is not None:
        Q = fem.functionspace(mesh, ("Lagrange", problem.degree))
        p_out = fem.Function(Q)
        dist, pmap = cKDTree(Q.tabulate_dof_coordinates()).query(points)
        assert dist.max() < 1e-12
    V0 = fem.functionspace(mesh, ("DG", 0))
    local_cells = np.arange(mesh.topology.index_map(3).size_local, dtype=np.int32)
    cmap = np.array([V0.dofmap.cell_dofs(i)[0] for i in local_cells])
    fmap = np.array([geo.f0.function_space.dofmap.cell_dofs(i)[0] for i in local_cells])
    f0 = geo.f0.x.array.reshape(-1, 3)[fmap].copy()
    s0 = geo.s0.x.array.reshape(-1, 3)[fmap].copy()

    # --- diagnostics ----------------------------------------------------------
    j = ufl.det(ufl.Identity(3) + ufl.grad(problem.u))
    Jfun = fem.Function(V0)
    jcenter = fem.Expression(j, V0.element.interpolation_points)
    jquad = fem.Expression(j, basix.make_quadrature(mesh.basix_cell(), 5)[0])
    ref_volume = fem.assemble_scalar(fem.form(1.0 * geo.dx))
    jerr = fem.form((j - 1)**2 * geo.dx)
    jmean = fem.form(j * geo.dx)
    kinetic = fem.form(0.5 * problem.parameters["rho"] * ufl.inner(problem.velocity, problem.velocity) * geo.dx)
    scales = {name: (fem.Function(V0), fem.Expression(expr, V0.element.interpolation_points))
              for name, expr in problem.stab_scales.items()}

    times = []
    try:
        with (out / "monitor.csv").open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=FIELDS, restval="")
            writer.writeheader()
            for step in range(nsteps):
                t = (step + 1) * args.dt
                problem.update_time_dependent_values(t)
                problem.solve(step)
                # everything below uses the converged step BEFORE the history shift
                Jfun.interpolate(jcenter)
                jq = jquad.eval(mesh, local_cells)
                if not np.isfinite(jq).all() or jq.min() <= 0:
                    raise RuntimeError(f"nonpositive or nonfinite J at step {step}")
                for fun, expr in scales.values():
                    fun.interpolate(expr)
                    if not np.isfinite(fun.x.array).all() or fun.x.array.min() < 0:
                        raise RuntimeError(f"invalid stabilisation scale at step {step}")
                u = problem.u.x.array.reshape(-1, 3).copy()
                v = problem.velocity.x.array.reshape(-1, 3).copy()
                jc = Jfun.x.array[cmap].copy()
                row = dict(time_s=t, p_endo_kPa=args.p_curve(t), Ta_kPa=args.ta_curve(t),
                           V_mL=capacity.calculate_volume(problem.u) * 1e6,
                           J_center_min=float(jc.min()), J_center_max=float(jc.max()),
                           J_quad_min=float(jq.min()), J_quad_max=float(jq.max()),
                           J_minus_1_RMS=float(np.sqrt(fem.assemble_scalar(jerr) / ref_volume)),
                           J_mean=float(fem.assemble_scalar(jmean) / ref_volume),
                           max_u_m=float(np.linalg.norm(u, axis=1).max()),
                           max_v_m_per_s=float(np.linalg.norm(v, axis=1).max()),
                           kinetic_energy_J=float(fem.assemble_scalar(kinetic)), **problem.last_diagnostics)
                point_data = {"displacement": u, "velocity": v}
                if problem.p is not None:
                    row.update(p_min_Pa=float(problem.p.x.array.min()), p_max_Pa=float(problem.p.x.array.max()))
                    p_out.interpolate(problem.p)
                    point_data["pressure_Pa"] = p_out.x.array[pmap].copy()
                if not all(np.isfinite(float(x)) for x in row.values()):
                    raise RuntimeError(f"nonfinite output at step {step}")
                cell_data = {"J": [jc], "f0": [f0], "s0": [s0],
                             **{name: [fun.x.array[cmap].copy()] for name, (fun, _) in scales.items()}}
                if not args.no_vtk:
                    meshio.write(vtk_dir / f"{args.method}_{step:05d}.vtu",
                                 meshio.Mesh(points, cells, point_data, cell_data))
                if recorder is not None:
                    recorder.record(t, u, point_data["pressure_Pa"], row["p_endo_kPa"], row["Ta_kPa"])
                writer.writerow(row)
                fh.flush()
                times.append(t)
                if not args.no_vtk:
                    write_pvd(vtk_dir / f"{args.method}.pvd", args.method, times)
                problem.advance()
                if step % 10 == 0 or step == nsteps - 1:
                    print(f"[{args.method}] t={t:.4f} V={row['V_mL']:.4f} mL "
                          f"J=[{row['J_quad_min']:.5f},{row['J_quad_max']:.5f}] "
                          f"J_RMS={row['J_minus_1_RMS']:.5g} |u|max={row['max_u_m']:.5g}", flush=True)
        if recorder is not None:
            recorder.write(out, dict(args.params, **metadata["loads"], dt_s=args.dt, steps=nsteps),
                           load0=(args.p_curve(0.0), args.ta_curve(0.0)))
        metadata.update(status="completed", completed_steps=len(times))
    except Exception as exc:
        metadata.update(status="failed", completed_steps=len(times), error=str(exc))
        raise
    finally:
        (out / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
