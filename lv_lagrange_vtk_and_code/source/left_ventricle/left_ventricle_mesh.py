"""DOLFINx mesh/fibre loader for the P1/P1 left-ventricle experiment.

Two input layouts are supported:

* an ADIOS2/io4dolfinx checkpoint (``mesh.bp`` + ``meshtags.bp``), as used by
  the repository's original ``mesh_LV_2``;
* a legacy FEniCS XDMF/HDF5 mesh directory (``*_mesh.xdmf`` + ``*_mesh.h5``),
  as used by ``lv_aha17_v2``.

For the XDMF layout the facet tags are rebuilt from the HDF5 facet topology,
because DOLFINx's legacy-XDMF facet-tag reader can mis-map this file.
"""

from pathlib import Path

import h5py
import numpy as np
import dolfinx
import io4dolfinx
from dolfinx.io import XDMFFile
from dolfinx.mesh import meshtags
from mpi4py import MPI
from scipy.spatial import cKDTree


def _default_mesh_dir() -> Path:
    # left_ventricle/ -> repository root/mesh_LV_2
    return Path(__file__).resolve().parent.parent / "mesh_LV_2"


def _dg0_vector_space(mesh):
    gdim = mesh.geometry.dim
    return dolfinx.fem.functionspace(mesh, ("DG", 0, (gdim,)))


def _find_xdmf(mesh_dir: Path) -> Path:
    candidates = sorted(mesh_dir.glob("*.xdmf"))
    if not candidates:
        raise FileNotFoundError(f"No *.xdmf file found in {mesh_dir}")
    for c in candidates:
        if c.stem.endswith("_mesh"):
            return c
    return candidates[0]


def _load_xdmf(comm, mesh_dir: Path, scale: float):
    xdmf_file = _find_xdmf(mesh_dir)
    h5_file = xdmf_file.with_suffix(".h5")
    if not h5_file.exists():
        maybe = list(mesh_dir.glob("*.h5"))
        if not maybe:
            raise FileNotFoundError(f"No matching .h5 file for {xdmf_file}")
        h5_file = maybe[0]

    with XDMFFile(comm, str(xdmf_file), "r") as xdmf:
        mesh = xdmf.read_mesh(name="mesh")

    gdim = mesh.geometry.dim

    # Rebuild facet tags from the HDF5 facet topology.  The XDMF reader
    # distributes the mesh, so topology vertices, geometry dofs and input
    # global vertex numbers may all use different local orderings.  Build a
    # local topology-vertex -> input-global vertex map from cell connectivity
    # and the geometry dofmap, then tag only owned facets.  This is robust in
    # parallel; the old direct ``input_global[c2v]`` lookup missed many facets
    # because c2v contains topology indices, not geometry-dof indices.
    fdim = mesh.topology.dim - 1
    mesh.topology.create_entities(fdim)
    mesh.topology.create_connectivity(fdim, 0)
    mesh.topology.create_connectivity(mesh.topology.dim, 0)
    c2v = mesh.topology.connectivity(fdim, 0).array.reshape(-1, 3)
    c3v = mesh.topology.connectivity(mesh.topology.dim, 0).array.reshape(-1, 4)

    try:
        geom_dofmap = mesh.geometry.dofmaps[0]
    except (AttributeError, IndexError):  # dolfinx < 0.9 compatibility
        geom_dofmap = mesh.geometry.dofmap
    cell_input_globals = mesh.geometry.input_global_indices[geom_dofmap]

    num_vertices = (
        mesh.topology.index_map(0).size_local
        + mesh.topology.index_map(0).num_ghosts
    )
    vertex_to_input_global = np.full(num_vertices, -1, dtype=np.int64)
    for cell in range(c3v.shape[0]):
        for local_vertex in range(c3v.shape[1]):
            v = int(c3v[cell, local_vertex])
            g = int(cell_input_globals[cell, local_vertex])
            if vertex_to_input_global[v] == -1:
                vertex_to_input_global[v] = g
            elif vertex_to_input_global[v] != g:
                raise RuntimeError(
                    "Inconsistent input-global vertex IDs while reading "
                    f"{h5_file}"
                )
    if np.any(vertex_to_input_global < 0):
        raise RuntimeError(
            f"Could not map all topology vertices for {h5_file}"
        )

    with h5py.File(h5_file, "r") as f:
        h5_top = f["Mesh/mesh/topology"][:]
        h5_xyz = f["Mesh/mesh/geometry"][:]
        h5_fibers = f["Mesh/fibers"][:]
        h5_sheets = f["Mesh/sheets"][:]
        h5_ftopo = f["MeshTags/facet_tags/topology"][:].astype(np.int64)
        h5_fvals = f["MeshTags/facet_tags/Values"][:].ravel().astype(np.int32)

    h5_facet_tag = {}
    for tri, val in zip(h5_ftopo, h5_fvals):
        key = tuple(sorted(int(x) for x in tri))
        h5_facet_tag[key] = int(val)

    n_owned_facets = mesh.topology.index_map(fdim).size_local
    indices = []
    values = []
    for i in range(min(n_owned_facets, c2v.shape[0])):
        key = tuple(sorted(int(vertex_to_input_global[v]) for v in c2v[i]))
        val = h5_facet_tag.get(key)
        if val is not None:
            indices.append(i)
            values.append(val)

    boundaries = meshtags(
        mesh, fdim,
        np.asarray(indices, dtype=np.int32),
        np.asarray(values, dtype=np.int32),
    )

    # Assign DG0 fibres/sheets by matching cell centroids.  This avoids
    # assuming that the XDMF reader preserves the original cell ordering.
    W_vec_dg0 = _dg0_vector_space(mesh)
    f0 = dolfinx.fem.Function(W_vec_dg0, name="f0")
    s0 = dolfinx.fem.Function(W_vec_dg0, name="s0")

    h5_centroids = h5_xyz[h5_top].mean(axis=1)
    tree = cKDTree(h5_centroids)

    def _interpolator(data):
        def callback(x):
            query = np.column_stack((x[0], x[1], x[2]))
            return data[tree.query(query)[1]].T
        return callback

    f0.interpolate(_interpolator(h5_fibers))
    s0.interpolate(_interpolator(h5_sheets))
    f0.x.scatter_forward()
    s0.x.scatter_forward()

    if scale != 1.0:
        mesh.geometry.x[:, :] *= scale

    return mesh, boundaries, f0, s0


def _load_bp(comm, mesh_dir: Path, scale: float):
    checkpoint = str(mesh_dir / "mesh.bp")
    tags_file = str(mesh_dir / "meshtags.bp")

    mesh = io4dolfinx.read_mesh(checkpoint, comm)
    boundaries = io4dolfinx.read_meshtags(
        tags_file, mesh, meshtag_name="meshtags_2"
    )

    W_vec_dg0 = _dg0_vector_space(mesh)
    f0 = dolfinx.fem.Function(W_vec_dg0, name="f0")
    s0 = dolfinx.fem.Function(W_vec_dg0, name="s0")
    io4dolfinx.read_function(checkpoint, f0, time=0.0, name="f0")
    io4dolfinx.read_function(checkpoint, s0, time=0.0, name="s0")

    if scale != 1.0:
        mesh.geometry.x[:, :] *= scale
    return mesh, boundaries, f0, s0


def load_left_ventricle_mesh(comm=MPI.COMM_WORLD, scale: float = 0.01,
                             mesh_dir: str | Path | None = None):
    """Load an LV mesh, boundary tags, fibre and sheet fields.

    Parameters
    ----------
    comm:
        MPI communicator.
    scale:
        Coordinate scaling.  Source meshes are in centimetres; the solver uses
        SI, so the default is ``0.01`` (cm -> m).
    mesh_dir:
        Directory containing either ``mesh.bp``/``meshtags.bp`` or an XDMF
        mesh (``*_mesh.xdmf`` + ``*_mesh.h5``).

    Returns
    -------
    mesh, boundaries, f0, s0
    """
    mesh_dir = Path(mesh_dir) if mesh_dir is not None else _default_mesh_dir()
    if (mesh_dir / "mesh.bp").exists():
        mesh, boundaries, f0, s0 = _load_bp(comm, mesh_dir, scale)
    else:
        mesh, boundaries, f0, s0 = _load_xdmf(comm, mesh_dir, scale)

    if comm.rank == 0:
        coords = mesh.geometry.x
        print(
            f"[Mesh] {mesh.topology.index_map(3).size_global} cells, "
            f"{mesh.topology.index_map(0).size_global} nodes | "
            f"x∈[{coords[:, 0].min():.4f},{coords[:, 0].max():.4f}] "
            f"y∈[{coords[:, 1].min():.4f},{coords[:, 1].max():.4f}] "
            f"z∈[{coords[:, 2].min():.4f},{coords[:, 2].max():.4f}]"
        )
    return mesh, boundaries, f0, s0
