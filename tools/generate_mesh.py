#!/usr/bin/env python3
"""Generate a standalone coarse LV v2 mesh WITHOUT AHA17 partition.

单文件版本: 只依赖 dolfinx 环境自带的 numpy / mpi4py / h5py 和 gmsh.
不再 import 仓库内其他 .py 文件; 把这个文件复制到任意目录, 在有
dolfinx + gmsh 的 Python 环境里即可运行.

几何: 与 lv_aha17_v2 相同 (r_long 9.0/9.7, r_short 2.5/3.5, 锥形基底).
分区: 无 AHA17; 整块心肌 cell tag = 0; 面 tag 5=BASE, 6=ENDO, 7=EPI.
纤维: cardiac-geometriesx 0.5.2 create_microstructure 的 Laplace + prolate
      管线移植, 逐单元 (P1 形心采样 + Gram-Schmidt).

==================== 如何使用 ====================

默认生成 mesh size = 1.5 cm 的无分区粗网格:

    python gen_lv_aha17_v2_plain.py

修改 --mesh-size (单位 cm, gmsh characteristic length):

    python gen_lv_aha17_v2_plain.py --mesh-size 1.0   # ~1958 cells
    python gen_lv_aha17_v2_plain.py --mesh-size 1.2   # ~1520 cells
    python gen_lv_aha17_v2_plain.py --mesh-size 1.5   # ~972 cells  (默认)
    python gen_lv_aha17_v2_plain.py --mesh-size 2.0   # ~699 cells  (最粗测试值)

规则: --mesh-size 数值越小 -> 网格越细; 数值越大 -> 网格越粗.
      粗于 2.0 也可以试, 但需自行检查网格质量/结果收敛性.

同时指定 XDMF 和 .bin 输出目录:

    python gen_lv_aha17_v2_plain.py \
        --mesh-size 2.0 \
        --xdmf data/lv_aha17_v2_plain/lv_aha17_v2_plain_mesh.xdmf \
        --out src/ibm/data/lv_aha17_v2_plain

只生成到当前目录 (不依赖仓库目录结构):

    python gen_lv_aha17_v2_plain.py \
        --mesh-size 1.5 \
        --xdmf ./lv_aha17_v2_plain_mesh.xdmf \
        --out ./lv_aha17_v2_plain_bin

运行环境:
    dolfinx (包含 numpy / mpi4py / h5py) + gmsh.
    若缺少 gmsh, 请先 `conda install -c conda-forge gmsh` 或 `pip install gmsh`.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re

import gmsh
import h5py  # noqa: F401  # 先于 petsc4py/PETSc 导入, 避免 HDF5 冲突
import numpy as np


# ---------------------------------------------------------------------------
# Geometry constants (cm) — same values as the repository v2 generator
# ---------------------------------------------------------------------------
R_LONG_ENDO = 9.0
R_SHORT_ENDO = 2.5
R_LONG_EPI = 9.7
R_SHORT_EPI = 3.5
EPI = dict(cx=0.0, cy=0.0, cz=0.0, rx=R_LONG_EPI, ry=R_SHORT_EPI, rz=R_SHORT_EPI)
ENDO = dict(cx=0.0, cy=0.0, cz=0.0, rx=R_LONG_ENDO, ry=R_SHORT_ENDO, rz=R_SHORT_ENDO)

X_BASE_ENDO = R_LONG_ENDO * 5.0 / 17.0
X_BASE_EPI = R_LONG_EPI * 5.0 / 20.0
RHO_BASE_ENDO = R_SHORT_ENDO * math.sin(math.acos(5.0 / 17.0))
RHO_BASE_EPI = R_SHORT_EPI * math.sin(math.acos(5.0 / 20.0))
SLOPE_BASE = (RHO_BASE_EPI - RHO_BASE_ENDO) / (X_BASE_EPI - X_BASE_ENDO)

X_BASE = X_BASE_ENDO
X_ENDO_APEX = -R_LONG_ENDO
X_EPI_APEX = -R_LONG_EPI
L_CAVITY = X_BASE_ENDO - X_ENDO_APEX
X_BASAL_MID = X_BASE_ENDO - L_CAVITY / 3.0
X_MID_APICAL = X_BASE_ENDO - 2.0 * L_CAVITY / 3.0

FIBER_ANGLE_ENDO = 60.0
FIBER_ANGLE_EPI = -60.0
CUT_R = 25.0
LOCAL_FACES = [(1, 2, 3), (0, 2, 3), (0, 1, 3), (0, 1, 2)]

# --- helper copied from repository generator: ell_value ---
def ell_value(p, c):
    """Value of the ellipsoid equation sum((x-c)/r)^2."""
    return ((p[0] - c["cx"]) / c["rx"]) ** 2 + \
           ((p[1] - c["cy"]) / c["ry"]) ** 2 + \
           ((p[2] - c["cz"]) / c["rz"]) ** 2

# --- helper copied from repository generator: ellipsoid ---
def ellipsoid(cx, cy, cz, rx, ry, rz):
    s = gmsh.model.occ.addSphere(cx, cy, cz, 1.0)
    gmsh.model.occ.dilate([(3, s)], cx, cy, cz, rx, ry, rz)
    return s

# --- helper copied from repository generator: extract_mesh ---
def extract_mesh(piece_seg):
    """Extract (coords, tets, cell_marker) from the live gmsh model (gmsh 节点序).

    tets 按体积片 (AHA 段) 顺序拼接; cell_marker[e] = 该单元所在段的 AHA 段号
    (与 XDMF cell_tags 同源: 都来自同一 physical groups).
    """
    node_tags, coords, _ = gmsh.model.mesh.getNodes()
    tag_to_idx = {t: i for i, t in enumerate(node_tags)}
    coords = np.asarray(coords, dtype=np.float64).reshape(-1, 3)

    tet_list, marker_list = [], []
    for _, tag in gmsh.model.getEntities(3):
        types, etags, ntags = gmsh.model.mesh.getElements(3, tag)
        if 4 not in types:
            continue
        i = list(types).index(4)
        nt = np.asarray(ntags[i], dtype=np.int64).reshape(-1, 4)
        if nt.size == 0:
            continue
        tet_list.append(nt)
        marker_list.append(np.full(nt.shape[0], piece_seg[tag], dtype=np.int64))
    tets = np.concatenate(tet_list)
    cell_marker = np.concatenate(marker_list)
    tets = np.vectorize(tag_to_idx.__getitem__, otypes=[np.int64])(tets)
    return coords, tets, cell_marker

# --- helper copied from repository generator: _norm_cols ---
def _norm_cols(v):
    """Normalize each column of a (3,n) array; zero columns stay zero (无 NaN/告警)."""
    n = np.linalg.norm(v, axis=0, keepdims=True)
    out = np.zeros_like(v)
    np.divide(v, n, out=out, where=(n > 0.0))
    return out

# --- helper copied from repository generator: compute_fibers_cgx ---
def compute_fibers_cgx(mesh, facet_tags):
    """cardiac-geometriesx 0.5.2 纤维管线移植 (返回 nn×9, dolfinx 节点序).

    与 cgx 0.5.2 `create_microstructure` + `utils.laplace` + `compute_system`
    (src/cardiac_geometries/fibers/lv_ellipsoid.py, fibers/utils.py) 公式逐行对应:
      1. Laplace: Δψ=0, ψ=0 on ENDO(6), ψ=1 on EPI(7), P1, LU 求解 (透壁坐标);
      2. prolate 球坐标解析系 (long_axis=0, x 为长轴, cgx 原生):
         mu = arctan2(sqrt(y²+z²)/rs, x/rl), theta = π - arctan2(z, -y);
         e_mu / e_theta 为子午向/周向切向, rs/rl 按 ψ 在 endo/epi 半径间线性插值;
         alpha = (60° + (-60°-60°)·ψ)·π/180  (心内膜 +60° → 心外膜 -60°);
         f0 = sinα·e_mu + cosα·e_theta, n0 = e_mu×e_theta (透壁径向), s0 = f0×n0.
    输出列布局 [f0, sheet, normal] = cgx 的 [f0, n0, s0] (sheet=径向, normal=f0×sheet,
    与仓库 ideal_biv 右手基约定一致, det=+1; cgx 原 (f0,s0,n0) 为左手系):
    f0 与 cgx 完全一致, sheet/normal 方向集一致, 仅标签互换 (Guccione 对 s/n 对称).

    与 cgx 原实现的差异 (均有注释):
      * dolfinx 0.10 LinearProblem 需 petsc_options_prefix (cgx 0.5.2 面向 0.9/0.10-dev);
      * apex 节点 (sin μ→0, cgx 原实现会除零得 NaN): θ=0 + n0 取径向极限 (-x 方向) 正则化,
        只影响轴上单点, 逐单元形心采样不受影响.
    """
    import ufl
    from dolfinx import fem, default_scalar_type
    from dolfinx.fem.petsc import LinearProblem

    V = fem.functionspace(mesh, ("P", 1))
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a = ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = v * fem.Constant(mesh, default_scalar_type(0.0)) * ufl.dx

    endo_facets = facet_tags.find(6)
    endo_dofs = fem.locate_dofs_topological(V, 2, endo_facets)
    epi_facets = facet_tags.find(7)
    epi_dofs = fem.locate_dofs_topological(V, 2, epi_facets)
    bcs = [fem.dirichletbc(fem.Constant(mesh, default_scalar_type(0.0)), endo_dofs, V),
           fem.dirichletbc(fem.Constant(mesh, default_scalar_type(1.0)), epi_dofs, V)]

    problem = LinearProblem(a, L, bcs=bcs,
                            petsc_options_prefix="lv_aha17_laplace",
                            petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
    t = problem.solve()
    t = t.x.array  # (cgx: t = t_func.x.array) PETSc 向量 → numpy, 与 dof 坐标同序

    dof_coordinates = V.tabulate_dof_coordinates()
    x = dof_coordinates[:, 0]
    y = dof_coordinates[:, 1]
    z = dof_coordinates[:, 2]

    # --- cgx compute_system (long_axis=0, verbatim formulas) ---
    alpha = (FIBER_ANGLE_ENDO +
             (FIBER_ANGLE_EPI - FIBER_ANGLE_ENDO) * t) * (np.pi / 180)
    rl = R_LONG_ENDO + (R_LONG_EPI - R_LONG_ENDO) * t
    rs = R_SHORT_ENDO + (R_SHORT_EPI - R_SHORT_ENDO) * t

    # cgx mu_theta (long_axis=0)
    mu = np.arctan2(np.sqrt(y ** 2 + z ** 2) / rs, x / rl)
    theta = np.pi - np.arctan2(z, -y)
    theta[mu < 1e-7] = 0.0

    # apex 正则化 (cgx 原实现此处会 NaN): sinμ→0 的轴上节点取 θ=0
    apex = np.sin(mu) < 1e-12
    theta[apex] = 0.0

    e_mu = np.array([-rl * np.sin(mu),
                     rs * np.cos(mu) * np.cos(theta),
                     rs * np.cos(mu) * np.sin(theta)])
    e_theta = np.array([np.zeros_like(t),
                        -rs * np.sin(mu) * np.sin(theta),
                        rs * np.sin(mu) * np.cos(theta)])
    e_mu = _norm_cols(e_mu)
    e_theta = _norm_cols(e_theta)
    e_theta[:, apex] = 0.0  # 轴上无周向, 置零 (f0 退化为子午向)

    f0 = np.sin(alpha) * e_mu + np.cos(alpha) * e_theta
    f0 = _norm_cols(f0)

    n0 = np.cross(e_mu, e_theta, axis=0)
    n0 = _norm_cols(n0)
    n0[:, apex] = np.array([-1.0, 0.0, 0.0])[:, None]  # 径向极限 (-x 方向)

    s0 = np.cross(f0, n0, axis=0)
    s0 = _norm_cols(s0)

    # 列布局 [f0, sheet=径向(cgx n0), normal=f0×sheet(cgx s0)] → det=+1 (仓库约定)
    fiber = np.empty((dof_coordinates.shape[0], 9), dtype=np.float64)
    fiber[:, 0:3] = f0.T
    fiber[:, 3:6] = n0.T
    fiber[:, 6:9] = s0.T
    return fiber

# --- helper copied from repository generator: map_fibers_to_gmsh ---
def map_fibers_to_gmsh(fiber_dx, dx_coords, gm_coords):
    """把 dolfinx 节点序的纤维 (nn,9) 按坐标双射映射到 gmsh 节点序."""
    if len(dx_coords) != len(gm_coords):
        raise RuntimeError(f"node count mismatch: dolfinx {len(dx_coords)} vs "
                           f"gmsh {len(gm_coords)}")
    scale = max(1.0, float(np.abs(gm_coords).max()))

    def key(a):
        return (np.asarray(a, dtype=np.float64) / scale * 1e9).round().astype(np.int64)

    V3 = np.dtype([("x", np.int64), ("y", np.int64), ("z", np.int64)])
    pos = {}
    for i, row in enumerate(key(dx_coords).view(V3)):
        pos[row.tobytes()] = i
    out = np.empty((len(gm_coords), 9), dtype=np.float64)
    miss = 0
    for j, row in enumerate(key(gm_coords).view(V3)):
        i = pos.get(row.tobytes())
        if i is None:
            miss += 1
            continue
        out[j] = fiber_dx[i]
    if miss:
        raise RuntimeError(f"fiber node mapping: {miss}/{len(gm_coords)} unmatched")
    return out

# --- helper copied from repository generator: write_xdmf ---
def write_xdmf(xdmf_path, mesh, cell_tags, facet_tags, fiber_dx):
    """Write unified dolfinx XDMF (+ .h5): mesh + cell/facet tags + per-cell fiber attrs."""
    from mpi4py import MPI
    from dolfinx.io import XDMFFile

    cell_tags.name = "cell_tags"
    facet_tags.name = "facet_tags"
    tdim = mesh.topology.dim
    mesh.topology.create_connectivity(tdim - 1, tdim)
    mesh.topology.create_connectivity(tdim, tdim)
    out_dir = os.path.dirname(os.path.abspath(xdmf_path))
    os.makedirs(out_dir, exist_ok=True)
    with XDMFFile(MPI.COMM_WORLD, str(xdmf_path), "w") as xdmf:
        xdmf.write_mesh(mesh)
        xdmf.write_meshtags(cell_tags, mesh.geometry)
        xdmf.write_meshtags(facet_tags, mesh.geometry)

    mesh.topology.create_connectivity(tdim, 0)
    cells = np.asarray(mesh.topology.connectivity(tdim, 0).array,
                       dtype=np.int64).reshape(-1, 4)
    f0c, s0c, n0c = p1_to_cell_fibers(fiber_dx, cells)
    attach_cell_vector_attrs(xdmf_path, os.path.splitext(str(xdmf_path))[0] + ".h5",
                             {"fibers": f0c, "sheets": s0c, "normals": n0c},
                             len(cells))
    print(f"  wrote dolfinx mesh -> {xdmf_path} (+ .h5, per-cell fiber attrs)")

# --- helper copied from repository generator: write_bin ---
def write_bin(path, arr, dtype=None):
    np.asarray(arr, dtype=dtype).tofile(path)

# --- helper copied from repository generator: p1_to_cell_fibers ---
def p1_to_cell_fibers(fiber, cells):
    """P1 逐节点纤维场 (nn,9) [f0,s0,n0] -> 分片常数逐单元基 (nc,3)x3.

    在单元形心 (P1 单积分点, 形函数值 1/4) 采样 P1 场, 再 Gram-Schmidt
    重新正交化 (节点基逐点正交, 单元平均后长度/正交性会略有偏差)。
    返回 (f0_cell, s0_cell, n0_cell), 每个 (nc,3), 行 e 与 cells 第 e 行对齐。
    """
    v = fiber[cells]  # (nc, 4, 9)
    f0 = v[:, :, 0:3].mean(axis=1)
    s0 = v[:, :, 3:6].mean(axis=1)
    f0 = f0 / np.linalg.norm(f0, axis=1, keepdims=True)
    s0 = s0 - (s0 * f0).sum(axis=1)[:, None] * f0
    s0 = s0 / np.linalg.norm(s0, axis=1, keepdims=True)
    n0 = np.cross(f0, s0)
    n0 = n0 / np.linalg.norm(n0, axis=1, keepdims=True)
    return f0, s0, n0

# --- helper copied from repository generator: attach_cell_vector_attrs ---
def attach_cell_vector_attrs(xdmf_path, h5_path, attrs, nc):
    """给统一 XDMF 追加逐单元 Vector attribute (Center="Cell"), 布局同 BV_mesh.xdmf.

    attrs: {name: (nc,3) float64}; h5 数据集写到 Mesh/<name>, XDMF 的
    mesh grid 内插入对应 <Attribute> (与 scripts/cardiology 导入脚本同格式)。
    重复调用幂等 (覆盖同名数据集/attribute)。
    """
    import h5py
    import re

    with h5py.File(str(h5_path), "a") as h:
        grp = h.require_group("Mesh")
        for name, arr in attrs.items():
            dset_path = f"Mesh/{name}"
            if dset_path in h:
                del h[dset_path]
            grp.create_dataset(name, data=np.asarray(arr, dtype=np.float64),
                               compression="gzip", compression_opts=4)
    text = open(str(xdmf_path)).read()
    h5_name = os.path.basename(str(h5_path))
    for n in attrs:  # 幂等: 先删同名旧 attribute
        text = re.sub(r'      <Attribute Name="%s" AttributeType="Vector" Center="Cell">.*?</Attribute>\n'
                      % re.escape(n), "", text, flags=re.S)
    blocks = "\n".join(
        f'      <Attribute Name="{n}" AttributeType="Vector" Center="Cell">\n'
        f'        <DataItem Dimensions="{nc} 3" Format="HDF">{h5_name}:/Mesh/{n}</DataItem>\n'
        f'      </Attribute>'
        for n in attrs)
    m = re.search(r'(<Grid Name="mesh" GridType="Uniform">.*?)(    </Grid>)', text, flags=re.S)
    if m is None:
        raise RuntimeError(f"mesh grid not found in {xdmf_path}")
    text = text[:m.end(1)] + "\n" + blocks + "\n" + text[m.end(1):]
    open(str(xdmf_path), "w").write(text)

# --- helper copied from repository generator: write_p1_solid_data ---
def write_p1_solid_data(out_dir, coords, tets, cell_marker,
                        face_cell, face_local, face_marker, face_normal, face_area,
                        fiber, info):
    """Write the P1-tetrahedron IBM solid .bin dataset (shared with gen_pulse_biv_data.py).

    coords: (nn,3) float64, tets: (nc,4) int64, fiber: (nn,9) float64.
    """
    os.makedirs(out_dir, exist_ok=True)
    nc, nn = tets.shape[0], coords.shape[0]
    print(f"  mesh: {nc} cells, {nn} nodes")

    write_bin(os.path.join(out_dir, "num_cells.bin"), [nc], np.int64)
    write_bin(os.path.join(out_dir, "num_dofs.bin"), [nn], np.int64)
    write_bin(os.path.join(out_dir, "num_qp.bin"), [1], np.int64)
    write_bin(os.path.join(out_dir, "num_basis.bin"), [4], np.int64)

    dofmaps = np.concatenate([tets, tets, tets], axis=1).astype(np.int64)
    write_bin(os.path.join(out_dir, "dofmaps.bin"), dofmaps.ravel(), np.int64)

    write_bin(os.path.join(out_dir, "cell_marker.bin"), np.asarray(cell_marker, dtype=np.int64))

    nq, nb = 1, 4
    basis_val = np.full((nq, nb), 0.25, dtype=np.float64)
    basis_deriv = np.zeros((nq, nb, 3), dtype=np.float64)
    basis_deriv[0, 0] = (-1.0, -1.0, -1.0)
    basis_deriv[0, 1] = (1.0, 0.0, 0.0)
    basis_deriv[0, 2] = (0.0, 1.0, 0.0)
    basis_deriv[0, 3] = (0.0, 0.0, 1.0)

    basis_vals = np.zeros((nq, nb * 3), dtype=np.float64)
    for a in range(nb):
        basis_vals[0, a * 3:a * 3 + 3] = basis_val[0, a]
    write_bin(os.path.join(out_dir, "basis_values_ref.bin"), basis_vals.ravel(), np.float64)

    basis_derivs = np.zeros((nq, nb * 3), dtype=np.float64)
    for a in range(nb):
        basis_derivs[0, a * 3:a * 3 + 3] = basis_deriv[0, a]
    write_bin(os.path.join(out_dir, "basis_derivatives_ref.bin"), basis_derivs.ravel(), np.float64)

    write_bin(os.path.join(out_dir, "weights.bin"), [1.0 / 6.0], np.float64)
    write_bin(os.path.join(out_dir, "ref_nodes.bin"), coords.ravel(), np.float64)

    v0 = coords[tets[:, 0]]
    v1 = coords[tets[:, 1]]
    v2 = coords[tets[:, 2]]
    v3 = coords[tets[:, 3]]
    H = np.stack([v1 - v0, v2 - v0, v3 - v0], axis=2)
    detH = np.linalg.det(H)
    J = np.abs(detH).astype(np.float64)
    invH = np.linalg.inv(H).astype(np.float64)
    write_bin(os.path.join(out_dir, "J.bin"), J, np.float64)
    write_bin(os.path.join(out_dir, "inv_H.bin"), invH.reshape(-1, 9).ravel(), np.float64)

    face_marker = np.asarray(face_marker, dtype=np.int64)
    nf = len(face_cell)
    nq_face = 1
    print(f"  faces: {nf} (BASE={np.sum(face_marker == 5)}, "
          f"ENDO_LV={np.sum(face_marker == 6)}, ENDO_RV={np.sum(face_marker == 7)}, "
          f"EPI={np.sum(face_marker == 8)})")

    write_bin(os.path.join(out_dir, "num_faces.bin"), [nf], np.int64)
    write_bin(os.path.join(out_dir, "num_qp_face.bin"), [nq_face], np.int64)
    write_bin(os.path.join(out_dir, "face_cell.bin"), face_cell, np.int64)
    write_bin(os.path.join(out_dir, "face_local_idx.bin"), face_local, np.int64)
    write_bin(os.path.join(out_dir, "face_marker.bin"), face_marker, np.int64)
    write_bin(os.path.join(out_dir, "face_normal.bin"), face_normal, np.float64)
    write_bin(os.path.join(out_dir, "face_area.bin"), face_area, np.float64)

    face_wts = np.array([0.5], dtype=np.float64)
    face_qp = np.array([1.0 / 3.0, 1.0 / 3.0], dtype=np.float64)
    write_bin(os.path.join(out_dir, "face_weights.bin"), face_wts, np.float64)
    write_bin(os.path.join(out_dir, "face_qp_coords.bin"), face_qp, np.float64)

    face_nodes = [
        (0.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0),
        (1.0 / 3.0, 0.0, 1.0 / 3.0, 1.0 / 3.0),
        (1.0 / 3.0, 1.0 / 3.0, 0.0, 1.0 / 3.0),
        (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0, 0.0),
    ]
    grad_ref = np.array([[-1.0, -1.0, -1.0],
                         [1.0, 0.0, 0.0],
                         [0.0, 1.0, 0.0],
                         [0.0, 0.0, 1.0]], dtype=np.float64)
    for lf in range(4):
        vals = np.zeros((nq_face, nb * 3), dtype=np.float64)
        grads = np.zeros((nq_face, nb * 3), dtype=np.float64)
        for a in range(nb):
            vals[0, a * 3:a * 3 + 3] = face_nodes[lf][a]
            grads[0, a * 3:a * 3 + 3] = grad_ref[a]
        write_bin(os.path.join(out_dir, f"face_basis_{lf}.bin"), vals.ravel(), np.float64)
        write_bin(os.path.join(out_dir, f"face_basis_grad_{lf}.bin"), grads.ravel(), np.float64)

    # 纤维: P1 逐节点场 (nn,9) -> 分片常数逐单元基 (形心采样 + Gram-Schmidt),
    # 行 e 与 dofmaps 第 e 单元对齐 (求解器 Guccione/RealLV 通道的 fiber_dir 约定)
    stale = os.path.join(out_dir, "fiber.bin")  # 旧的逐节点格式, 若存在则清除
    if os.path.exists(stale):
        os.remove(stale)
    fiber = np.asarray(fiber, dtype=np.float64)
    f0c, s0c, n0c = p1_to_cell_fibers(fiber, tets)
    write_bin(os.path.join(out_dir, "fiber_dir.bin"), f0c.ravel(), np.float64)
    write_bin(os.path.join(out_dir, "sheet_dir.bin"), s0c.ravel(), np.float64)
    write_bin(os.path.join(out_dir, "normal_dir.bin"), n0c.ravel(), np.float64)
    print(f"  fiber_dir/sheet_dir/normal_dir.bin: {nc} cells x 3 (P1 centroid + Gram-Schmidt)")

    with open(os.path.join(out_dir, "info.json"), "w") as f:
        json.dump(info, f, indent=2)

    print(f"Wrote P1 solid data -> {out_dir}")



# ---------------------------------------------------------------------------
# OCC construction: same geometry as v2, but one single volume
# ---------------------------------------------------------------------------
def build_plain_volume():
    """Build the conical-base LV myocardium as one OCC volume (cell tag 0)."""
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("lv_aha17_v2_plain")

    epi = ellipsoid(EPI["cx"], EPI["cy"], EPI["cz"],
                    EPI["rx"], EPI["ry"], EPI["rz"])
    endo = ellipsoid(ENDO["cx"], ENDO["cy"], ENDO["cz"],
                     ENDO["rx"], ENDO["ry"], ENDO["rz"])
    myo0, _ = gmsh.model.occ.cut([(3, epi)], [(3, endo)])

    # 平截: 去掉 x > X_BASE_ENDO (与 AHA 版本完全相同)
    box = gmsh.model.occ.addBox(X_BASE_ENDO, -CUT_R, -CUT_R,
                                CUT_R * 2, 2 * CUT_R, 2 * CUT_R)
    myo_flat, _ = gmsh.model.occ.cut(myo0, [(3, box)], removeTool=True)

    # 锥形基底: 与 AHA 版本相同的大锥体求交, 保留锥面内侧
    BIG = 50.0
    r_big = RHO_BASE_ENDO + SLOPE_BASE * (-BIG)
    cone = gmsh.model.occ.addCone(X_BASE_ENDO, 0.0, 0.0, -BIG, 0.0, 0.0,
                                  RHO_BASE_ENDO, r_big)
    myo, _ = gmsh.model.occ.intersect(myo_flat, [(3, cone)], removeTool=True)
    gmsh.model.occ.synchronize()

    vols = gmsh.model.getEntities(3)
    if len(vols) != 1:
        raise RuntimeError(f"Expected a single myocardium volume, got {len(vols)}")
    tag = vols[0][1]
    gmsh.model.addPhysicalGroup(3, [tag], 0)
    return {tag: 0}


# ---------------------------------------------------------------------------
# Robust boundary tagging
# ---------------------------------------------------------------------------
def _candidate_residuals(pt):
    """Residuals (cone, endo, epi) for one point; zero means exactly on surface."""
    x, y, z = (float(v) for v in pt)
    rho = math.hypot(y, z)
    cone = abs(rho - (RHO_BASE_ENDO + SLOPE_BASE * (x - X_BASE_ENDO)))
    endo = abs(ell_value(pt, ENDO) - 1.0)
    epi = abs(ell_value(pt, EPI) - 1.0)
    return cone, endo, epi


_MARKERS = (5, 6, 7)  # BASE, ENDO, EPI


def marker_at_point(pt):
    """Classify a point lying on a surface.  Ties resolve to BASE by design."""
    res = _candidate_residuals(pt)
    return _MARKERS[int(np.argmin(res))], res


def marker_for_triangle(tri):
    """Classify a mesh triangle by the worst vertex residual for each surface.

    For triangles on the thin conical base band a centroid test is unreliable
    at coarse mesh sizes; requiring all three vertices to match one of the
    analytic surfaces is stable up to very coarse meshes.
    """
    cone = max(abs(math.hypot(p[1], p[2]) -
                   (RHO_BASE_ENDO + SLOPE_BASE * (p[0] - X_BASE_ENDO)))
               for p in tri)
    endo = max(abs(ell_value(p, ENDO) - 1.0) for p in tri)
    epi = max(abs(ell_value(p, EPI) - 1.0) for p in tri)
    return _MARKERS[int(np.argmin((cone, endo, epi)))], (cone, endo, epi)


def _surface_sample(tag):
    """Return a physical point actually on OCC surface `tag`, or None."""
    try:
        mn, mx = gmsh.model.getParametrizationBounds(2, tag)
        uv = [0.5 * (float(a) + float(b))
              for a, b in zip(np.asarray(mn).ravel(), np.asarray(mx).ravel())]
        xyz = np.asarray(gmsh.model.getValue(2, tag, uv), dtype=np.float64).ravel()
        if xyz.size != 3 or not np.all(np.isfinite(xyz)):
            return None
        return xyz
    except Exception:
        return None


def _first_entity_triangle(tag):
    """Return the first 3D triangle of surface `tag` in gmsh node coordinates."""
    node_tags, coords, _ = gmsh.model.mesh.getNodes()
    tag_to_idx = {t: i for i, t in enumerate(node_tags)}
    coords = np.asarray(coords, dtype=np.float64).reshape(-1, 3)
    types, _, ntags = gmsh.model.mesh.getElements(2, tag)
    if 2 not in types:
        return None
    i = list(types).index(2)
    tri_tags = np.asarray(ntags[i], dtype=np.int64).reshape(-1, 3)
    if tri_tags.size == 0:
        return None
    return coords[[tag_to_idx[t] for t in tri_tags[0]]]


def attach_faces_robust():
    """Add 5/6/7 physical surface groups using exact surface samples.

    Returns a {surface_tag: marker} map for the later .bin face mapping.
    """
    surface_marker = {}
    face_groups = {}
    for _, tag in gmsh.model.getEntities(2):
        up, _ = gmsh.model.getAdjacencies(2, tag)
        if len(up) != 1:
            continue

        marker = None
        pt = _surface_sample(tag)
        if pt is not None:
            marker, res = marker_at_point(pt)
            # 参数中点若碰到曲面交线 (例如 apex / 基底接缝), 用三角形兜底
            srt = sorted(res)
            if srt[1] > 1e-8:
                marker = None
        if marker is None:
            tri = _first_entity_triangle(tag)
            if tri is None:
                raise RuntimeError(f"cannot sample boundary surface tag={tag}")
            marker, _ = marker_for_triangle(tri)

        surface_marker[tag] = marker
        face_groups.setdefault(marker, []).append(tag)

    for marker, tags in sorted(face_groups.items()):
        gmsh.model.addPhysicalGroup(2, tags, marker)
    return surface_marker


def build_face_marker_map(surface_marker):
    """Map sorted vertex-index tuples of boundary triangles to face markers."""
    node_tags, _, _ = gmsh.model.mesh.getNodes()
    tag_to_idx = {t: i for i, t in enumerate(node_tags)}
    mapping = {}
    for _, tag in gmsh.model.getEntities(2):
        if tag not in surface_marker:
            continue
        marker = surface_marker[tag]
        types, _, ntags = gmsh.model.mesh.getElements(2, tag)
        if 2 not in types:
            continue
        i = list(types).index(2)
        tris = np.asarray(ntags[i], dtype=np.int64).reshape(-1, 3)
        for tri in tris:
            key = tuple(sorted(tag_to_idx[t] for t in tri))
            mapping[key] = marker
    return mapping


def classify_faces_robust(coords, tets, face_marker_map):
    """Boundary faces with robust markers from the gmsh surface map."""
    nc = tets.shape[0]
    cell_centroid = coords[tets].mean(axis=1)

    face_map = {}
    for c in range(nc):
        tet = tets[c]
        for lf, (a, b, d) in enumerate(LOCAL_FACES):
            key = tuple(sorted((tet[a], tet[b], tet[d])))
            if key in face_map:
                face_map[key] = None
            else:
                face_map[key] = [c, lf, key]

    face_cell, face_local, face_marker = [], [], []
    face_normal, face_area = [], []
    for key, v in face_map.items():
        if v is None:
            continue
        c, lf, _ = v
        if key not in face_marker_map:
            raise RuntimeError(f"boundary face {key} missing from surface marker map")
        tri = coords[list(key)]
        cen = tri.mean(axis=0)
        n = np.cross(tri[1] - tri[0], tri[2] - tri[0])
        area = 0.5 * np.linalg.norm(n)
        if area == 0.0:
            raise RuntimeError(f"zero-area boundary face {key}")
        n = n / (2.0 * area)
        if np.dot(n, cen - cell_centroid[c]) < 0.0:
            n = -n
        face_cell.append(c)
        face_local.append(lf)
        face_marker.append(face_marker_map[key])
        face_normal.extend(n)
        face_area.append(area)

    assert len(face_cell) == len(face_marker_map), (
        f"boundary face count mismatch: extracted {len(face_cell)} vs surface map "
        f"{len(face_marker_map)}")
    return (np.asarray(face_cell, dtype=np.int64),
            np.asarray(face_local, dtype=np.int64),
            np.asarray(face_marker, dtype=np.int64),
            np.asarray(face_normal, dtype=np.float64),
            np.asarray(face_area, dtype=np.float64))


def write_info_json(xdmf_path, mesh_size, nc, nn, facet_counts):
    """Write data/<case>/info.json."""
    info = {
        "case": "lv_aha17_v2_plain",
        "description": ("Ideal LV from document Section 3.1/3.2 (conical base), "
                        "WITHOUT AHA17 partition; whole myocardium cell tag 0. "
                        "Boundary tagging is robust for coarse meshes."),
        "mesh_type": "P1 tetrahedra",
        "units": "cm",
        "markers": {
            "cell_tags": {"0": "MYOCARDIUM (no AHA partition)"},
            "facet_tags": {"5": "BASE", "6": "ENDO", "7": "EPI"},
        },
        "counts": {"nodes": int(nn), "cells": int(nc),
                   "facets": {str(k): int(v) for k, v in sorted(facet_counts.items())}},
        "geometry_cm": {
            "epi_ellipsoid": EPI, "endo_ellipsoid": ENDO,
            "x_base": X_BASE, "x_endo_apex": X_ENDO_APEX, "x_epi_apex": X_EPI_APEX,
            "base_cone": {"x_endo": X_BASE_ENDO, "rho_endo": RHO_BASE_ENDO,
                          "x_epi": R_LONG_EPI * 5.0 / 20.0,
                          "rho_epi": R_SHORT_EPI * math.sin(math.acos(5.0 / 20.0))},
            "long_axis": "x (cardiac-geometriesx 0.5.2 native, long_axis=0)",
        },
        "extents_cm": {
            "x": [float(X_EPI_APEX), float(X_BASE)],
            "y": [-float(R_SHORT_EPI), float(R_SHORT_EPI)],
            "z": [-float(R_SHORT_EPI), float(R_SHORT_EPI)],
        },
        "fiber": {
            "generator": "cardiac-geometriesx 0.5.2 create_microstructure "
                         "(laplace transmural + prolate spheroidal analytic system)",
            "params": {
                "r_short_endo": R_SHORT_ENDO, "r_short_epi": R_SHORT_EPI,
                "r_long_endo": R_LONG_ENDO, "r_long_epi": R_LONG_EPI,
                "fiber_angle_endo": FIBER_ANGLE_ENDO,
                "fiber_angle_epi": FIBER_ANGLE_EPI,
                "fiber_space": "P_1",
            },
            "layout": "per-cell piecewise constant; P1 field at cell centroid "
                      "+ Gram-Schmidt (repo convention)",
            "attributes": ["fibers", "sheets", "normals"],
        },
        "generator": "scripts/gen_lv_aha17_v2_plain.py",
        "reproduce": f"python scripts/gen_lv_aha17_v2_plain.py --mesh-size {mesh_size:g}",
        "source_geometry": "same as data/lv_aha17_v2 (no AHA partition)",
    }
    out_dir = os.path.dirname(os.path.abspath(xdmf_path))
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "info.json"), "w") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    # 修改网格粗细: --mesh-size <cm>, 数值越大网格越粗.
    # 常用值: 1.0 -> ~1958 cells; 1.2 -> ~1520 cells;
    #         1.5 -> ~972 cells (default); 2.0 -> ~699 cells.
    parser.add_argument("--mesh-size", type=float, default=1.5,
                        help="gmsh characteristic mesh size (cm); default 1.5 "
                             "(coarser than lv_aha17_v2's 0.7)")
    # 默认输出相对“当前工作目录”, 这样单文件复制到任何目录都能直接运行;
    # 在仓库根目录下运行时会落到标准位置 src/ibm/data/... 和 data/...
    default_out = os.path.join(os.getcwd(), "src", "ibm", "data", "lv_aha17_v2_plain")
    parser.add_argument("--out", default=default_out,
                        help="solver .bin output directory")
    default_xdmf = os.path.join(os.getcwd(), "data", "lv_aha17_v2_plain",
                                "lv_aha17_v2_plain_mesh.xdmf")
    parser.add_argument("--xdmf", default=default_xdmf,
                        help="unified dolfinx XDMF output")
    args = parser.parse_args()

    print(f"Building LV v2 plain mesh (no AHA partition, mesh size = {args.mesh_size:g}) ...")
    piece_seg = build_plain_volume()

    gmsh.option.setNumber("Mesh.MeshSizeMax", args.mesh_size)
    gmsh.option.setNumber("Mesh.MeshSizeMin", 0.5 * args.mesh_size)
    gmsh.model.mesh.generate(3)
    try:
        gmsh.model.mesh.optimize("Netgen")
        print("  Netgen mesh optimization applied")
    except Exception as e:  # noqa: BLE001
        print(f"  Netgen optimize unavailable: {e}")

    surface_marker = attach_faces_robust()
    print(f"  robust boundary physical groups 5/6/7 attached "
          f"(surfaces: {surface_marker})")

    # dolfinx mesh (XDMF + fibers use the same model_to_mesh)
    from mpi4py import MPI
    try:
        from dolfinx.io.gmshio import model_to_mesh
    except ImportError:
        from dolfinx.io.gmsh import model_to_mesh
    md = model_to_mesh(gmsh.model, MPI.COMM_WORLD, 0, gdim=3)
    if hasattr(md, "mesh"):
        mesh, cell_tags, facet_tags = md.mesh, md.cell_tags, md.facet_tags
    else:
        mesh, cell_tags, facet_tags = md
    if cell_tags is None or facet_tags is None:
        raise RuntimeError("model_to_mesh 未返回 tags —— 检查 physical groups")

    coords, tets, cell_marker = extract_mesh(piece_seg)
    face_marker_map = build_face_marker_map(surface_marker)
    face_cell, face_local, face_marker, face_normal, face_area = \
        classify_faces_robust(coords, tets, face_marker_map)

    facet_values, facet_counts = np.unique(facet_tags.values, return_counts=True)
    bin_values, bin_counts = np.unique(face_marker, return_counts=True)
    if dict(zip(facet_values.tolist(), facet_counts.tolist())) != \
       dict(zip(bin_values.tolist(), bin_counts.tolist())):
        raise RuntimeError(
            "XDMF facet tags and .bin face markers disagree: "
            f"xdmf={dict(zip(facet_values.tolist(), facet_counts.tolist()))} "
            f"bin={dict(zip(bin_values.tolist(), bin_counts.tolist()))}")

    fiber_dx = compute_fibers_cgx(mesh, facet_tags)
    print("  fibers: cgx create_microstructure (Laplace + prolate)")
    write_xdmf(args.xdmf, mesh, cell_tags, facet_tags, fiber_dx)

    nc, nn = tets.shape[0], coords.shape[0]
    print(f"  mesh: {nc} cells, {nn} nodes")
    print("  face markers (cells): " +
          ", ".join(f"{int(k)}={int(v)}" for k, v in zip(bin_values, bin_counts)))
    print(f"  cell markers: {np.unique(cell_marker).tolist()}")
    write_info_json(args.xdmf, args.mesh_size, nc, nn,
                    {int(k): int(v) for k, v in zip(bin_values, bin_counts)})

    # 纤维 dolfinx 节点序 -> gmsh 节点序
    dx_coords = np.asarray(mesh.geometry.x.reshape(-1, 3), dtype=np.float64)
    fiber = map_fibers_to_gmsh(fiber_dx, dx_coords, coords)

    os.makedirs(args.out, exist_ok=True)
    info = {
        "case": "lv_aha17_v2_plain",
        "mesh_type": "P1 tetrahedra",
        "mesh_files": [args.xdmf, os.path.splitext(args.xdmf)[0] + ".h5"],
        "markers": {
            "cell": "0 (no AHA partition)",
            "BASE": 5, "ENDO": 6, "EPI": 7,
        },
        "fiber": {
            "generator": "cardiac-geometriesx 0.5.2 create_microstructure (port)",
            "files": ["fiber_dir.bin", "sheet_dir.bin", "normal_dir.bin"],
            "layout": "per-cell [f0(3)], [s0(3)], [n0(3)]",
            "xdmf_attributes": ["fibers", "sheets", "normals"],
            "alpha_endo_deg": FIBER_ANGLE_ENDO,
            "alpha_epi_deg": FIBER_ANGLE_EPI,
        },
        "geometry_cm": {
            "epi_ellipsoid": EPI, "endo_ellipsoid": ENDO,
            "x_base": X_BASE, "x_endo_apex": X_ENDO_APEX,
            "x_epi_apex": X_EPI_APEX,
            "base_cone": {"x_endo": X_BASE_ENDO, "rho_endo": RHO_BASE_ENDO,
                          "x_epi": R_LONG_EPI * 5.0 / 20.0,
                          "rho_epi": R_SHORT_EPI * math.sin(math.acos(5.0 / 20.0))},
        },
        "source_geometry": "same as data/lv_aha17_v2 (no AHA partition)",
    }
    write_p1_solid_data(args.out, coords, tets, cell_marker,
                        face_cell, face_local, face_marker, face_normal, face_area,
                        fiber, info)
    gmsh.finalize()
    print(f"Done: XDMF -> {args.xdmf}, solver .bin -> {args.out}")


if __name__ == "__main__":
    main()
