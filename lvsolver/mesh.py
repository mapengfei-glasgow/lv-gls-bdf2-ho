"""LV mesh/fibre loader (legacy FEniCS XDMF + HDF5), base centre and cavity volume."""
from pathlib import Path

import dolfinx
import h5py
import numpy as np
import ufl
from dolfinx.io import XDMFFile
from dolfinx.mesh import meshtags
from mpi4py import MPI
from scipy.spatial import Delaunay, cKDTree


def _find_xdmf(mesh_dir: Path) -> Path:
    candidates = sorted(mesh_dir.glob("*.xdmf"))
    if not candidates:
        raise FileNotFoundError(f"No *.xdmf file found in {mesh_dir}")
    for c in candidates:
        if c.stem.endswith("_mesh"):
            return c
    return candidates[0]


def load_xdmf_mesh(comm, mesh_dir, scale):
    """Read mesh, facet tags and DG0 fibre/sheet fields; scale coordinates (cm -> m).

    Facet tags are rebuilt from the HDF5 facet topology because DOLFINx's
    legacy-XDMF facet-tag reader can mis-map this file.
    """
    mesh_dir = Path(mesh_dir)
    xdmf_file = _find_xdmf(mesh_dir)
    h5_file = xdmf_file.with_suffix(".h5")
    if not h5_file.exists():
        maybe = list(mesh_dir.glob("*.h5"))
        if not maybe:
            raise FileNotFoundError(f"No matching .h5 file for {xdmf_file}")
        h5_file = maybe[0]

    with XDMFFile(comm, str(xdmf_file), "r") as xdmf:
        mesh = xdmf.read_mesh(name="mesh")

    # Map local topology vertices -> input-global vertex ids through the
    # cell connectivity and geometry dofmap (robust in parallel), then tag
    # only owned facets.
    fdim = mesh.topology.dim - 1
    mesh.topology.create_entities(fdim)
    mesh.topology.create_connectivity(fdim, 0)
    mesh.topology.create_connectivity(mesh.topology.dim, 0)
    c2v = mesh.topology.connectivity(fdim, 0).array.reshape(-1, 3)
    c3v = mesh.topology.connectivity(mesh.topology.dim, 0).array.reshape(-1, 4)
    cell_input_globals = mesh.geometry.input_global_indices[mesh.geometry.dofmaps[0]]

    num_vertices = mesh.topology.index_map(0).size_local + mesh.topology.index_map(0).num_ghosts
    vertex_to_input_global = np.full(num_vertices, -1, dtype=np.int64)
    for cell in range(c3v.shape[0]):
        for local_vertex in range(c3v.shape[1]):
            v = int(c3v[cell, local_vertex])
            g = int(cell_input_globals[cell, local_vertex])
            if vertex_to_input_global[v] == -1:
                vertex_to_input_global[v] = g
            elif vertex_to_input_global[v] != g:
                raise RuntimeError(f"Inconsistent input-global vertex IDs while reading {h5_file}")
    if np.any(vertex_to_input_global < 0):
        raise RuntimeError(f"Could not map all topology vertices for {h5_file}")

    with h5py.File(h5_file, "r") as f:
        h5_top = f["Mesh/mesh/topology"][:]
        h5_xyz = f["Mesh/mesh/geometry"][:]
        h5_fibers = f["Mesh/fibers"][:]
        h5_sheets = f["Mesh/sheets"][:]
        h5_ftopo = f["MeshTags/facet_tags/topology"][:].astype(np.int64)
        h5_fvals = f["MeshTags/facet_tags/Values"][:].ravel().astype(np.int32)

    h5_facet_tag = {tuple(sorted(int(x) for x in tri)): int(val) for tri, val in zip(h5_ftopo, h5_fvals)}
    indices, values = [], []
    for i in range(min(mesh.topology.index_map(fdim).size_local, c2v.shape[0])):
        val = h5_facet_tag.get(tuple(sorted(int(vertex_to_input_global[v]) for v in c2v[i])))
        if val is not None:
            indices.append(i)
            values.append(val)
    boundaries = meshtags(mesh, fdim, np.asarray(indices, dtype=np.int32), np.asarray(values, dtype=np.int32))

    # DG0 fibres/sheets matched by cell centroid (cell order may be permuted).
    W = dolfinx.fem.functionspace(mesh, ("DG", 0, (mesh.geometry.dim,)))
    f0 = dolfinx.fem.Function(W, name="f0")
    s0 = dolfinx.fem.Function(W, name="s0")
    tree = cKDTree(h5_xyz[h5_top].mean(axis=1))

    def _interpolator(data):
        return lambda x: data[tree.query(np.column_stack((x[0], x[1], x[2])))[1]].T

    f0.interpolate(_interpolator(h5_fibers))
    s0.interpolate(_interpolator(h5_sheets))
    f0.x.scatter_forward()
    s0.x.scatter_forward()

    if scale != 1.0:
        mesh.geometry.x[:, :] *= scale

    if comm.rank == 0:
        x = mesh.geometry.x
        print(f"[Mesh] {mesh.topology.index_map(3).size_global} cells, "
              f"{mesh.topology.index_map(0).size_global} nodes | "
              f"x∈[{x[:, 0].min():.4f},{x[:, 0].max():.4f}] "
              f"y∈[{x[:, 1].min():.4f},{x[:, 1].max():.4f}] "
              f"z∈[{x[:, 2].min():.4f},{x[:, 2].max():.4f}]", flush=True)
    return mesh, boundaries, f0, s0


class LVGeometry:
    """Mesh, facet tags, fibres and degree-5 integration measures."""

    def __init__(self, mesh_dir, scale=0.01, markers=None):
        self.bdry_markers = dict(markers or {"epi": 7, "base": 5, "endo": 6})
        self.msh, self.bdry, self.f0, self.s0 = load_xdmf_mesh(MPI.COMM_WORLD, mesh_dir, scale)
        metadata = {"quadrature_degree": 5}
        self.ds = ufl.Measure("ds", domain=self.msh, subdomain_data=self.bdry, metadata=metadata)
        self.dx = ufl.Measure("dx", domain=self.msh, metadata=metadata)


def compute_base_center(msh, boundaries, base_id, comm=MPI.COMM_WORLD):
    """Global bounding-box centre of the base facet vertices."""
    fdim = msh.topology.dim - 1
    msh.topology.create_connectivity(fdim, 0)
    facets = boundaries.find(base_id)
    if comm.allreduce(len(facets), op=MPI.SUM) == 0:
        raise RuntimeError(f"No base facets with tag {base_id} on any rank")
    conn = msh.topology.connectivity(fdim, 0).array.reshape(-1, 3)
    local = msh.geometry.x[np.unique(conn[facets].flatten())]
    lmin = local.min(axis=0) if local.size else np.full(3, np.inf)
    lmax = local.max(axis=0) if local.size else np.full(3, -np.inf)
    gmin = np.array([comm.allreduce(lmin[i], op=MPI.MIN) for i in range(3)])
    gmax = np.array([comm.allreduce(lmax[i], op=MPI.MAX) for i in range(3)])
    return 0.5 * (gmin + gmax)


class LeftVentricleCapacity:
    """Cavity volume estimate: Delaunay tetrahedralisation of the reference
    endocardial dof points, re-evaluated on the deformed points."""

    def __init__(self, mesh, bdry, bdry_markers, function_space_descriptor):
        comm = MPI.COMM_WORLD
        self.mesh = mesh
        V = dolfinx.fem.functionspace(mesh, function_space_descriptor)
        self.endo_dofs = dolfinx.fem.locate_dofs_topological(V, bdry.dim, bdry.find(bdry_markers["endo"]))
        coordinate = dolfinx.fem.Function(V)
        coordinate.interpolate(lambda x: np.array([x[0], x[1], x[2]]))
        points = coordinate.x.array.reshape(-1, mesh.geometry.dim)[self.endo_dofs].astype(np.float64)
        self.endo_points_global = self._gather(points)
        self.tri = Delaunay(self.endo_points_global) if comm.rank == 0 else None

    def _gather(self, local):
        comm = MPI.COMM_WORLD
        gdim = self.mesh.geometry.dim
        all_n = np.empty(comm.size, dtype=np.int32) if comm.rank == 0 else None
        comm.Gather(np.array([local.shape[0]], dtype=np.int32), all_n, root=0)
        if comm.rank == 0:
            out = np.empty((np.sum(all_n), gdim), dtype=np.float64)
            counts = all_n * gdim
            displs = np.insert(np.cumsum(all_n), 0, 0)[:-1] * gdim
            comm.Gatherv(sendbuf=local.ravel(), recvbuf=(out, counts, displs, MPI.DOUBLE), root=0)
            return out
        comm.Gatherv(sendbuf=local.ravel(), recvbuf=None, root=0)
        return None

    def calculate_volume(self, displacement):
        disp = displacement.x.array.reshape(-1, self.mesh.geometry.dim)[self.endo_dofs].astype(np.float64)
        disp = self._gather(disp)
        volume = None
        if MPI.COMM_WORLD.rank == 0:
            tetra = (self.endo_points_global + disp)[self.tri.simplices]
            volume = np.sum(np.abs(np.linalg.det(tetra[:, :3] - tetra[:, 3][:, None, :])) / 6)
        return MPI.COMM_WORLD.bcast(volume, root=0)
