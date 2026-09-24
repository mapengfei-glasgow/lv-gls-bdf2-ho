"""Run the original P1 penalty solver on the plain mesh for a full cycle,
writing one VTU frame per time step (displacement, velocity, J, f0, s0)."""
import csv
import os
from pathlib import Path

import meshio
import numpy as np
import ufl
from dolfinx import fem
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
from solidmechanicsproblem import SolidMechanicsProblem_Bar  # noqa: E402


def main():
    dt = base.TIME_PARAMS["dt"]
    nsteps = base.NUM_STEPS
    mesh_dir = os.environ["SIMPLE_3D0D_MESH_DIR"]
    scale = float(os.environ.get("SIMPLE_3D0D_MESH_SCALE", "0.01"))
    long_axis = os.environ.get("SIMPLE_3D0D_LONG_AXIS", "x").lower()
    endo_id = int(os.environ.get("SIMPLE_3D0D_ENDO_TAG", "6"))
    epi_id = int(os.environ.get("SIMPLE_3D0D_EPI_TAG", "7"))
    base_id = int(os.environ.get("SIMPLE_3D0D_BASE_TAG", "5"))

    out_dir = Path(os.environ.get("SIMPLE_3D0D_OUTPUT_DIR", str(HERE / "out_plain_p1_full_cycle")))
    vtk_dir = out_dir / "vtk"
    vtk_dir.mkdir(parents=True, exist_ok=True)

    pressure_pa = lambda t: base.p_endo_kPa(t) * base.KPA_TO_PA
    active_pa = lambda t: base.Ta_kPa(t) * base.KPA_TO_PA
    time_curves = {"4": pressure_pa, "5": active_pa}

    print(f"[plain-p1] Reading XDMF LV mesh from {mesh_dir} ...", flush=True)
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
    problem = SolidMechanicsProblem_Bar(io_solid, material, unflat, time_curves)
    capacity = LeftVentricleCapacity(
        mesh=io_solid.msh, bdry=io_solid.bdry,
        bdry_markers=io_solid.bdry_markers,
        function_space_descriptor=("CG", 1, (3,)),
    )

    msh = io_solid.msh
    points = msh.geometry.x.copy()
    try:
        gdm = msh.geometry.dofmaps[0]
    except (AttributeError, IndexError):
        gdm = msh.geometry.dofmap
    cells = [("tetra", np.asarray(gdm, dtype=np.int64))]
    if problem.u.x.array.size != points.shape[0] * 3:
        raise RuntimeError(
            f"P1 dof/geometry mismatch: u dofs {problem.u.x.array.size}, points {points.shape[0]}"
        )

    V0 = fem.functionspace(msh, ("DG", 0))
    j_fun = fem.Function(V0, name="J")
    j_expr = fem.Expression(
        ufl.det(ufl.Identity(3) + ufl.grad(problem.u)),
        V0.element.interpolation_points,
    )
    f0 = io_solid.f0.x.array.reshape(-1, 3).copy()
    s0 = io_solid.s0.x.array.reshape(-1, 3).copy()

    print(f"[plain-p1] base_center={tuple(round(float(v),5) for v in base_center)}, "
          f"steps={nsteps}, dt={dt}, output={out_dir}", flush=True)

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
        j_arr = j_fun.x.array.copy()

        frame = meshio.Mesh(
            points, cells,
            point_data={"displacement": u_arr, "velocity": v_arr},
            cell_data={"J": [j_arr], "f0": [f0], "s0": [s0]},
        )
        meshio.write(vtk_dir / f"plain_{step:05d}.vtu", frame, file_format="vtu")
        times.append(t)
        history.append([t, base.p_endo_kPa(t), base.Ta_kPa(t), v_ml,
                        float(np.min(j_arr)), float(np.max(j_arr)), float(np.max(np.linalg.norm(u_arr, axis=1)))])

        if step % 10 == 0 or step == nsteps - 1:
            print(f"[plain-p1] t={t:7.4f} s | p_endo={base.p_endo_kPa(t):8.4f} kPa | "
                  f"Ta={base.Ta_kPa(t):8.4f} kPa | V={v_ml:9.4f} mL | "
                  f"J=[{np.min(j_arr):.4f},{np.max(j_arr):.4f}]", flush=True)

    with open(vtk_dir / "plain.pvd", "w") as fh:
        fh.write('<?xml version="1.0"?>\n')
        fh.write('<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">\n')
        fh.write("  <Collection>\n")
        for k, t in enumerate(times):
            fh.write(f'    <DataSet timestep="{t:.10g}" group="" part="0" '
                     f'file="plain_{k:05d}.vtu"/>\n')
        fh.write("  </Collection>\n</VTKFile>\n")

    if MPI.COMM_WORLD.rank == 0:
        with open(out_dir / "monitor_plain_p1_full_cycle.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["time_s", "p_endo_kPa", "Ta_kPa", "V_mL", "J_min", "J_max", "max_u_m"])
            w.writerows(history)
        print(f"[plain-p1] Saved {out_dir/'monitor_plain_p1_full_cycle.csv'}", flush=True)
    print("[plain-p1] Done.", flush=True)


if __name__ == "__main__":
    main()
