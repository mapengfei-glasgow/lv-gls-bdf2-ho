"""Run the incompressible P2/P1 Lagrange multiplier solver on the plain mesh for a full cycle,
writing one VTU frame per time step (displacement, velocity, J, f0, s0)."""
import csv
import json
import os
from pathlib import Path

import basix
import meshio
import numpy as np
import ufl
from dolfinx import fem, plot
from mpi4py import MPI

HERE = Path(__file__).resolve().parent
os.environ.setdefault("SIMPLE_3D0D_MESH_DIR", str(HERE / "mesh_plain"))
os.environ.setdefault("SIMPLE_3D0D_MESH_SCALE", "0.01")
os.environ.setdefault("SIMPLE_3D0D_LONG_AXIS", "x")
os.environ.setdefault("SIMPLE_3D0D_ENDO_TAG", "6")
os.environ.setdefault("SIMPLE_3D0D_EPI_TAG", "7")
os.environ.setdefault("SIMPLE_3D0D_BASE_TAG", "5")

import main_3d_only as base  # noqa: E402  (reads SIMPLE_3D0D_* at import)
from material import HolzapfelOgden  # noqa: E402
from mesh_io import LeftVentricleCapacity  # noqa: E402
from solidmechanics_lagrange import SolidMechanicsProblem_Lagrange  # noqa: E402


def main():
    if MPI.COMM_WORLD.size != 1:
        raise RuntimeError("This meshio exporter requires a single MPI rank.")
    dt = base.TIME_PARAMS["dt"]
    nsteps = base.NUM_STEPS
    mesh_dir = os.environ["SIMPLE_3D0D_MESH_DIR"]
    scale = float(os.environ.get("SIMPLE_3D0D_MESH_SCALE", "0.01"))
    long_axis = os.environ.get("SIMPLE_3D0D_LONG_AXIS", "x").lower()
    endo_id = int(os.environ.get("SIMPLE_3D0D_ENDO_TAG", "6"))
    epi_id = int(os.environ.get("SIMPLE_3D0D_EPI_TAG", "7"))
    base_id = int(os.environ.get("SIMPLE_3D0D_BASE_TAG", "5"))

    out_dir = Path(os.environ.get("SIMPLE_3D0D_OUTPUT_DIR", str(HERE / "out_lagrange_full_cycle")))
    vtk_dir = out_dir / "vtk"
    vtk_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "run_metadata.json").write_text(json.dumps({
        "method": "P2/P1 augmented Lagrange incompressibility",
        "augmentation_Pa": float(os.environ.get("SIMPLE_3D0D_LM_AUGMENTATION", "1e6")),
        "dt_s": dt, "steps": nsteps, "mesh_scale": scale,
    }, indent=2) + "\n")

    pressure_pa = lambda t: base.p_endo_kPa(t) * base.KPA_TO_PA
    active_pa = lambda t: base.Ta_kPa(t) * base.KPA_TO_PA
    time_curves = {"4": pressure_pa, "5": active_pa}

    print(f"[lagrange-p2p1] Reading XDMF LV mesh from {mesh_dir} ...", flush=True)
    io_solid = base.IO_Solid_LV_XDMF(
        mesh_dir, scale=scale,
        markers={"epi": epi_id, "base": base_id, "endo": endo_id},
    )
    base_center = base.compute_base_center(io_solid.msh, io_solid.bdry, base_id, MPI.COMM_WORLD)
    unflat = {
        "SOLVER_PARAMS": base.SOLVER_PARAMS,
        "TIME_PARAMS": base.TIME_PARAMS,
        "MARKERS": {"epi": epi_id, "base": base_id, "endo": endo_id},
        "GEOMETRY": {"long_axis": long_axis, "base_center": tuple(base_center)},
    }
    material = HolzapfelOgden(
        f0=io_solid.f0, s0=io_solid.s0, mesh=io_solid.msh,
        parameters=base.CONSTITUTIVE_PARAMS,
    )
    problem = SolidMechanicsProblem_Lagrange(io_solid, material, unflat, time_curves)
    capacity = LeftVentricleCapacity(
        mesh=io_solid.msh, bdry=io_solid.bdry,
        bdry_markers=io_solid.bdry_markers,
        function_space_descriptor=("CG", 2, (3,)),
    )

    msh = io_solid.msh
    topology, cell_types, points = plot.vtk_mesh(problem.u_space)
    cells = [("VTK_LAGRANGE_TETRAHEDRON", topology.reshape(-1, 11)[:, 1:])]
    if not np.all(cell_types == 71):
        raise RuntimeError("Expected VTK_LAGRANGE_TETRAHEDRON")
    scalar_p2 = fem.functionspace(msh, ("Lagrange", 2))
    p_out = fem.Function(scalar_p2)
    from scipy.spatial import cKDTree
    distance, scalar_order = cKDTree(scalar_p2.tabulate_dof_coordinates()).query(points)
    if distance.max() > 1e-12:
        raise RuntimeError("Cannot map pressure output to displacement nodes")

    V0 = fem.functionspace(msh, ("DG", 0))
    j_fun = fem.Function(V0, name="J")
    j_expr = fem.Expression(
        ufl.det(ufl.Identity(3) + ufl.grad(problem.u)),
        V0.element.interpolation_points,
    )
    cell_order = np.array([V0.dofmap.cell_dofs(i)[0] for i in range(len(cells[0][1]))])
    fiber_order = np.array([io_solid.f0.function_space.dofmap.cell_dofs(i)[0] for i in range(len(cell_order))])
    f0 = io_solid.f0.x.array.reshape(-1, 3)[fiber_order].copy()
    s0 = io_solid.s0.x.array.reshape(-1, 3)[fiber_order].copy()

    print(f"[lagrange-p2p1] base_center={tuple(round(float(v),5) for v in base_center)}, "
          f"steps={nsteps}, dt={dt}, output={out_dir}", flush=True)

    j_ufl = ufl.det(ufl.Identity(3) + ufl.grad(problem.u))
    quadrature_points, _ = basix.make_quadrature(msh.basix_cell(), 5)
    j_quad = fem.Expression(j_ufl, quadrature_points)
    owned_cells = np.arange(msh.topology.index_map(3).size_local, dtype=np.int32)
    error_form = fem.form((j_ufl - 1)**2 * io_solid.dx)
    volume_ref = fem.assemble_scalar(fem.form(1.0 * io_solid.dx))
    history = []
    times = []
    for step in range(nsteps):
        t = (step + 1) * dt
        problem.update_time_dependent_values(t)
        problem.solve(step)
        v_ml = capacity.calculate_volume(problem.u) * 1e6
        problem.update_variables()

        u_arr = problem.u.x.array.reshape(-1, 3).copy()
        v_arr = problem.v_old.x.array.reshape(-1, 3).copy()
        j_fun.interpolate(j_expr)
        j_arr = j_fun.x.array[cell_order].copy()
        p_out.interpolate(problem.p)
        j_rms = np.sqrt(fem.assemble_scalar(error_form) / volume_ref)
        j_samples = j_quad.eval(msh, owned_cells)
        if not np.isfinite(j_samples).all() or np.min(j_samples) <= 0:
            raise RuntimeError(f"Invalid sampled Jacobian at step {step}")

        frame = meshio.Mesh(
            points, cells,
            point_data={"displacement": u_arr, "velocity": v_arr, "lagrange_pressure_Pa": p_out.x.array[scalar_order].copy()},
            cell_data={"J_cell_center": [j_arr], "f0": [f0], "s0": [s0]},
        )
        meshio.write(vtk_dir / f"lagrange_{step:05d}.vtu", frame, file_format="vtu")
        times.append(t)
        history.append([t, base.p_endo_kPa(t), base.Ta_kPa(t), v_ml,
                        float(np.min(j_arr)), float(np.max(j_arr)), float(np.max(np.linalg.norm(u_arr, axis=1))), float(j_rms), float(problem.p.x.array.min()), float(problem.p.x.array.max()), float(j_samples.min()), float(j_samples.max())])

        if step % 10 == 0 or step == nsteps - 1:
            print(f"[lagrange-p2p1] t={t:7.4f} s | p_endo={base.p_endo_kPa(t):8.4f} kPa | "
                  f"Ta={base.Ta_kPa(t):8.4f} kPa | V={v_ml:9.4f} mL | "
                  f"J=[{np.min(j_arr):.4f},{np.max(j_arr):.4f}]", flush=True)

    with open(vtk_dir / "lagrange.pvd", "w") as fh:
        fh.write('<?xml version="1.0"?>\n')
        fh.write('<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">\n')
        fh.write("  <Collection>\n")
        for k, t in enumerate(times):
            fh.write(f'    <DataSet timestep="{t:.10g}" group="" part="0" '
                     f'file="lagrange_{k:05d}.vtu"/>\n')
        fh.write("  </Collection>\n</VTKFile>\n")

    if MPI.COMM_WORLD.rank == 0:
        with open(out_dir / "monitor_lagrange_full_cycle.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["time_s", "p_endo_kPa", "Ta_kPa", "V_mL", "J_center_min", "J_center_max", "max_u_m", "J_minus_1_RMS", "p_lagrange_min_Pa", "p_lagrange_max_Pa", "J_quadrature_min", "J_quadrature_max"])
            w.writerows(history)
        print(f"[lagrange-p2p1] Saved {out_dir/'monitor_lagrange_full_cycle.csv'}", flush=True)
    print("[lagrange-p2p1] Done.", flush=True)


if __name__ == "__main__":
    main()
