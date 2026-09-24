"""Minimal mesh I/O for the LV mesh used by the 3D-0D demo.

This is a self-contained copy of ``src/softmech/mesh/lv2.py`` that reads the
mesh from this directory instead of the old ``$PATH_DATA`` location.
"""

from pathlib import Path

import dolfinx
import io4dolfinx
import numpy as np
import ufl
from mpi4py import MPI
from scipy.spatial import Delaunay


class LeftVentricleCapacity:
    """Compute the LV cavity volume from an endocardial surface."""

    def __init__(self, mesh, bdry, bdry_markers, function_space_descriptor):
        self.mesh = mesh
        self.bdry = bdry
        self.bdry_markers = bdry_markers
        V = dolfinx.fem.functionspace(self.mesh, function_space_descriptor)
        endo_dofs = dolfinx.fem.locate_dofs_topological(
            V, self.bdry.dim, self.bdry.find(self.bdry_markers["endo"])
        )
        coordinate = dolfinx.fem.Function(V, name="refrence")
        coordinate.interpolate(lambda x: np.array([x[0], x[1], x[2]]))
        endo_points = coordinate.x.array.reshape(-1, self.mesh.geometry.dim)[endo_dofs]
        self.endo_points_local = endo_points.astype(np.float64)

        local_n = np.array([self.endo_points_local.shape[0]], dtype=np.int32)
        all_n = None
        if MPI.COMM_WORLD.rank == 0:
            all_n = np.empty(MPI.COMM_WORLD.size, dtype=np.int32)
        MPI.COMM_WORLD.Gather(local_n, all_n, root=0)

        if MPI.COMM_WORLD.rank == 0:
            total_n = np.sum(all_n)
            endo_points_global = np.empty((total_n, self.mesh.geometry.dim), dtype=np.float64)
        else:
            endo_points_global = None

        counts = all_n * self.mesh.geometry.dim if MPI.COMM_WORLD.rank == 0 else None
        displs = (
            np.insert(np.cumsum(all_n), 0, 0)[:-1] * self.mesh.geometry.dim
            if MPI.COMM_WORLD.rank == 0
            else None
        )

        MPI.COMM_WORLD.Gatherv(
            sendbuf=self.endo_points_local.ravel(),
            recvbuf=(endo_points_global, counts, displs, MPI.DOUBLE)
            if MPI.COMM_WORLD.rank == 0
            else None,
            root=0,
        )

        self.V = V
        self.endo_dofs = endo_dofs
        self.endo_points_global = endo_points_global
        if MPI.COMM_WORLD.rank == 0:
            self.tri = Delaunay(self.endo_points_global)
        else:
            self.tri = None

    def calculate_volume(self, displacement):
        endo_disp = displacement.x.array.reshape(-1, self.mesh.geometry.dim)[self.endo_dofs]
        self.endo_disp_local = endo_disp.astype(np.float64)
        local_n = np.array([self.endo_disp_local.shape[0]], dtype=np.int32)
        all_n = None
        if MPI.COMM_WORLD.rank == 0:
            all_n = np.empty(MPI.COMM_WORLD.size, dtype=np.int32)
        MPI.COMM_WORLD.Gather(local_n, all_n, root=0)

        if MPI.COMM_WORLD.rank == 0:
            total_n = np.sum(all_n)
            endo_disp_global = np.empty((total_n, self.mesh.geometry.dim), dtype=np.float64)
        else:
            endo_disp_global = None

        counts = all_n * self.mesh.geometry.dim if MPI.COMM_WORLD.rank == 0 else None
        displs = (
            np.insert(np.cumsum(all_n), 0, 0)[:-1] * self.mesh.geometry.dim
            if MPI.COMM_WORLD.rank == 0
            else None
        )

        MPI.COMM_WORLD.Gatherv(
            sendbuf=self.endo_disp_local.ravel(),
            recvbuf=(endo_disp_global, counts, displs, MPI.DOUBLE)
            if MPI.COMM_WORLD.rank == 0
            else None,
            root=0,
        )

        if MPI.COMM_WORLD.rank == 0:
            endo_points_current = self.endo_points_global + endo_disp_global
            tetra = endo_points_current[self.tri.simplices]
            volumes = np.abs(np.linalg.det(tetra[:, :3] - tetra[:, 3][:, None, :])) / 6
            sum_volume = np.sum(volumes)
        else:
            sum_volume = None

        return MPI.COMM_WORLD.bcast(sum_volume, root=0)


class IO_Solid_LV2:
    """Read the LV_2 mesh/fibres from the local ``mesh_LV_2`` directory."""

    def __init__(self, mesh_dir=None):
        if mesh_dir is None:
            mesh_dir = Path(__file__).resolve().parent / "mesh_LV_2"
        mesh_dir = Path(mesh_dir)
        checkpoint_file = str(mesh_dir / "mesh.bp")
        mesh_tags_file = str(mesh_dir / "meshtags.bp")

        _metadata = {"quadrature_degree": 5}
        self.bdry_markers = {"epi": 4096, "base": 4098, "endo": 4097}
        self.center = {"X": 7.5, "Y": 7.5}
        self.msh = io4dolfinx.read_mesh(checkpoint_file, MPI.COMM_WORLD)
        self.bdry = io4dolfinx.read_meshtags(
            mesh_tags_file, self.msh, meshtag_name="meshtags_2"
        )
        self.area = io4dolfinx.read_meshtags(
            mesh_tags_file, self.msh, meshtag_name="meshtags_3"
        )
        self.fdim = self.msh.topology.dim - 1
        self.ds = ufl.Measure(
            "ds", domain=self.msh, subdomain_data=self.bdry, metadata=_metadata
        )
        self.dx = ufl.Measure(
            "dx", domain=self.msh, subdomain_data=self.area, metadata=_metadata
        )

        W_fiber = dolfinx.fem.functionspace(
            self.msh, ("DG", 0, (self.msh.geometry.dim,))
        )
        self.f0 = dolfinx.fem.Function(W_fiber, name="f0")
        self.s0 = dolfinx.fem.Function(W_fiber, name="s0")
        io4dolfinx.read_function(checkpoint_file, self.f0, time=0.0, name="f0")
        io4dolfinx.read_function(checkpoint_file, self.s0, time=0.0, name="s0")

    def scale_meter(self):
        # Original mesh is in centimetres; SI is used by the solid solver.
        self.msh.geometry.x[:] *= 0.01

    def set_output_files(self, output_dir):
        self.output_dir = output_dir
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        io4dolfinx.write_mesh(f"{output_dir}/checkpoint.bp", self.msh)
        io4dolfinx.write_mesh(f"{output_dir}/fibers.bp", self.msh)
        io4dolfinx.write_function(f"{output_dir}/fibers.bp", self.f0, time=0.0)
        io4dolfinx.write_function(f"{output_dir}/fibers.bp", self.s0, time=0.0)

    def close(self):
        pass

    def output(self, state, t, step):
        if step % 10 == 0:
            io4dolfinx.write_function(
                f"{self.output_dir}/checkpoint.bp", state, time=t
            )
