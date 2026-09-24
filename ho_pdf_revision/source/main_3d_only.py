"""3D-only LV contraction model with directly prescribed loading.

No 0D lumped circulation is used.  The endocardial pressure and active fibre
tension are prescribed directly as functions of time.

The supplied equations are in kPa; the existing DOLFINx formulation in this
folder uses Pa (consistent with the Holzapfel-Ogden parameters), so the two
time functions below return values converted to Pa.

Active fibre tension is already implemented in ``material._PK1_active`` as

    T(t, I4f) = Ta(t) * (1 + 4.9 * (sqrt(I4f) - 1)) * F * (f0 ⊗ f0)

where the scalar ``Ta(t)`` is supplied through the time curve ``"5"``.
"""

import csv
import os
from pathlib import Path

import numpy as np
import ufl
from mpi4py import MPI

from material import HolzapfelOgden
from mesh_io import IO_Solid_LV2, LeftVentricleCapacity
from solidmechanicsproblem import SolidMechanicsProblem_Bar

# ---------------------------------------------------------------------------
# Directly prescribed loading (kPa)
# ---------------------------------------------------------------------------

def p_endo_kPa(t):
    """Endocardial pressure, Eq. 79, in kPa."""
    if t < 0.2:
        return 1.067 * t / 0.2
    if t < 0.5:
        return 1.067
    if t < 0.65:
        return 1.067 + 13.46 * (1.0 - np.exp(-((t - 0.5) ** 2) / 0.004))
    if t < 0.8:
        return 1.067 + 13.46 * (1.0 - np.exp(-((0.8 - t) ** 2) / 0.004))
    return 1.067


def Ta_kPa(t):
    """Active fibre tension amplitude, Eq. 80, in kPa."""
    if t < 0.5:
        return 0.0
    if t < 0.65:
        return 84.26 * (1.0 - np.exp(-((t - 0.5) ** 2) / 0.005))
    if t < 0.8:
        return 84.26 * (1.0 - np.exp(-((0.8 - t) ** 2) / 0.005))
    return 0.0


# ---------------------------------------------------------------------------
# Numerical parameters (same style as main.py)
# ---------------------------------------------------------------------------
mmHg_TO_PA = 133.322
PA_TO_MMHG = 1.0 / mmHg_TO_PA
KPA_TO_PA = 1000.0

SOLVER_PARAMS = {
    "snes_type": "newtonls",
    "snes_linesearch_type": "basic",
    "snes_max_it": 20,
    "snes_stol": 1e-8,
    "snes_atol": 1e-8,
    "snes_rtol": 1e-8,
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
    "snes_monitor": None,
}

TIME_PARAMS = {"dt": float(os.environ.get("SIMPLE_3D0D_DT", "0.005"))}
T_END = float(os.environ.get("SIMPLE_3D0D_T_END", "1.0"))
NUM_STEPS = int(os.environ.get("SIMPLE_3D0D_STEPS", str(round(T_END / TIME_PARAMS["dt"]))))

CONSTITUTIVE_PARAMS = {
    "a": 2244.87 / 10.0,
    "b": 1.6215,
    "a_f": 24267.0 / 10.0,
    "b_f": 1.8268,
    "a_s": 5562.38 / 10.0,
    "b_s": 0.7746,
    "a_fs": 3905.16 / 10.0,
    "b_fs": 1.695,
    "kappa": 1e6,
    "eta": 1e2,
    "k": 100.0,
}

MARKERS = {"epi": 4096, "base": 4098, "endo": 4097}

UNFLAT = {
    "SOLVER_PARAMS": SOLVER_PARAMS,
    "TIME_PARAMS": TIME_PARAMS,
    "MARKERS": MARKERS,
}


class IO_Solid_LV_XDMF:
    """Minimal IO adapter for the same XDMF/H5 meshes used by
    ``left_ventricle/left_ventricle_mesh.py``."""

    def __init__(self, mesh_dir, scale=0.01, markers=None):
        from left_ventricle.left_ventricle_mesh import (
            load_left_ventricle_mesh,
        )

        if markers is None:
            markers = {"epi": 7, "base": 5, "endo": 6}
        self.bdry_markers = dict(markers)
        self.msh, self.bdry, self.f0, self.s0 = load_left_ventricle_mesh(
            MPI.COMM_WORLD, scale=scale, mesh_dir=mesh_dir
        )
        self.fdim = self.msh.topology.dim - 1
        metadata = {"quadrature_degree": 5}
        self.ds = ufl.Measure(
            "ds", domain=self.msh,
            subdomain_data=self.bdry, metadata=metadata,
        )
        self.dx = ufl.Measure("dx", domain=self.msh, metadata=metadata)

    def scale_meter(self):
        # Scaling is already applied by load_left_ventricle_mesh(scale=...).
        pass


def compute_base_center(msh, boundaries, base_id, comm=MPI.COMM_WORLD):
    """Global bounding-box center of the base facet vertices."""
    fdim = msh.topology.dim - 1
    msh.topology.create_connectivity(fdim, 0)
    c_to_v = msh.topology.connectivity(fdim, 0)
    facets = boundaries.find(base_id)
    if comm.allreduce(len(facets), op=MPI.SUM) == 0:
        raise RuntimeError(f"No base facets with tag {base_id} on any rank")
    conn = c_to_v.array.reshape(-1, 3)
    nodes = np.unique(conn[facets].flatten())
    local = msh.geometry.x[nodes]
    if local.size == 0:
        lmin = np.full(3, np.inf)
        lmax = np.full(3, -np.inf)
    else:
        lmin = local.min(axis=0)
        lmax = local.max(axis=0)
    gmin = np.array([comm.allreduce(lmin[i], op=MPI.MIN) for i in range(3)])
    gmax = np.array([comm.allreduce(lmax[i], op=MPI.MAX) for i in range(3)])
    return 0.5 * (gmin + gmax)


def main():
    history = []

    # Convert prescribed kPa loads to Pa for the existing solid mechanics code.
    def pressure_pa(t):
        return p_endo_kPa(t) * KPA_TO_PA

    def active_pa(t):
        return Ta_kPa(t) * KPA_TO_PA

    time_curves = {
        "4": pressure_pa,   # endocardial pressure on the endo surface
        "5": active_pa,     # active fibre tension amplitude Ta(t)
    }

    mesh_dir = os.environ.get("SIMPLE_3D0D_MESH_DIR")
    if mesh_dir:
        scale = float(os.environ.get("SIMPLE_3D0D_MESH_SCALE", "0.01"))
        long_axis = os.environ.get("SIMPLE_3D0D_LONG_AXIS", "x").lower()
        endo_id = int(os.environ.get("SIMPLE_3D0D_ENDO_TAG", "6"))
        epi_id = int(os.environ.get("SIMPLE_3D0D_EPI_TAG", "7"))
        base_id = int(os.environ.get("SIMPLE_3D0D_BASE_TAG", "5"))
        print(f"Reading XDMF LV mesh from {mesh_dir} ...")
        io_solid = IO_Solid_LV_XDMF(
            mesh_dir, scale=scale,
            markers={"epi": epi_id, "base": base_id, "endo": endo_id},
        )
        base_center = compute_base_center(
            io_solid.msh, io_solid.bdry, base_id, MPI.COMM_WORLD
        )
        unflat = dict(UNFLAT)
        unflat["MARKERS"] = {
            "epi": epi_id, "base": base_id, "endo": endo_id,
        }
        unflat["GEOMETRY"] = {
            "long_axis": long_axis, "base_center": tuple(base_center),
        }
    else:
        print("Reading LV_2 mesh ...")
        io_solid = IO_Solid_LV2()
        io_solid.scale_meter()
        unflat = dict(UNFLAT)
        unflat["GEOMETRY"] = {
            "long_axis": "z", "base_center": (0.075, 0.075, 0.0),
        }

    material = HolzapfelOgden(
        f0=io_solid.f0,
        s0=io_solid.s0,
        mesh=io_solid.msh,
        parameters=CONSTITUTIVE_PARAMS,
    )
    problem = SolidMechanicsProblem_Bar(io_solid, material, unflat, time_curves)
    calculate_capacity = LeftVentricleCapacity(
        mesh=io_solid.msh,
        bdry=io_solid.bdry,
        bdry_markers=io_solid.bdry_markers,
        function_space_descriptor=("CG", 1, (3,)),
    )

    print(f"Running {NUM_STEPS} time steps with dt={TIME_PARAMS['dt']} ...")

    for step in range(NUM_STEPS):
        t = (step + 1) * TIME_PARAMS["dt"]
        problem.update_time_dependent_values(t)
        problem.solve(step)

        v3d_ml = calculate_capacity.calculate_volume(problem.u) * 1e6
        problem.update_variables()

        history.append(
            [
                t,
                p_endo_kPa(t),
                Ta_kPa(t),
                v3d_ml,
            ]
        )

        if step % 10 == 0 or step == NUM_STEPS - 1:
            print(
                f"t={t:7.4f} s | p_endo={p_endo_kPa(t):8.4f} kPa | "
                f"Ta={Ta_kPa(t):8.4f} kPa | V_3D={v3d_ml:8.4f} mL",
                flush=True,
            )

    if MPI.COMM_WORLD.rank == 0:
        _save_results(history)

    print("Done.")


def _save_results(history):
    out_dir = Path(
        os.environ.get(
            "SIMPLE_3D0D_OUTPUT_DIR",
            str(Path(__file__).resolve().parent),
        )
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "results_3d_only.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time_s", "p_endo_kPa", "Ta_kPa", "V_3D_mL"])
        writer.writerows(history)
    print(f"Saved time series to {csv_path}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Matplotlib not available, skip figure: {exc}")
        return

    time_s = [row[0] for row in history]
    p_endo = [row[1] for row in history]
    ta = [row[2] for row in history]
    v3d = [row[3] for row in history]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].plot(time_s, p_endo, label="p_endo", linewidth=2)
    axes[0].plot(time_s, ta, label="Ta", linewidth=2)
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("kPa")
    axes[0].set_title("Prescribed loading")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(time_s, v3d, linewidth=2, color="C2")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("LV volume (mL)")
    axes[1].set_title("3D LV volume")
    axes[1].grid(True)

    axes[2].plot(v3d, p_endo, linewidth=2, color="C3")
    axes[2].set_xlabel("LV volume (mL)")
    axes[2].set_ylabel("Endocardial pressure (kPa)")
    axes[2].set_title("3D-only PV-like loop")
    axes[2].grid(True)

    fig.tight_layout()
    fig_path = out_dir / "pv_loop_3d_only.png"
    fig.savefig(fig_path, dpi=150)
    print(f"Saved figure to {fig_path}")


if __name__ == "__main__":
    main()
