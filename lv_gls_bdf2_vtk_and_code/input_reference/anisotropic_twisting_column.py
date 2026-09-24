# sym: PhysicalUnits
"""
3D 扭转柱问题 — 各向异性纤维增强 Standard Reinforced Model，BDF2 (v,p) 动态求解器

支持四种稳定化/单元模式:
  - "none"      → P2/P1 + 无稳定化 (Taylor-Hood, inf-sup 稳定)
  - "none_p1p1" → P1/P1 + 无稳定化 (仅用于对比验证)
  - "gls"       → P1/P1 + GLS 稳定化
  - "oss"       → P1/P1 + GLS+OSS 稳定化

根据论文 [27, 28] 描述:
  - 几何: 1×1×6 m 平行六面体
  - 底面固定 (v=0), 其余面零牵引力
  - 初速度: 扭转模式 v₀ = 100*sin(πy/12)*(z, 0, -x) m/s
  - 初位移: u₀ = 0
  - 材料: Standard Reinforced Model (各向异性纤维增强), G=Gf=5.67×10⁶ Pa, ρ₀=1.1×10³ kg/m³
  - 求解器: DynamicConsistentGLSBDF2Solver (BDF2)
"""
from dolfin import *
from utilities import PhysicalUnits
from twisting_beam_mesh import create_three_dimensional_beam_mesh
from stabilized_hyperelasticity_twisting_colum import StandardReinforcedModel, DynamicConsistentGLSBDF2Solver
from mpi4py import MPI as pyMPI
import os, time, csv, glob, logging, sys, datetime
import numpy as np
import matplotlib.pyplot as plt

# ---- 将 workspace root 加入 sys.path, 使 logger_utils 等可导入 ----
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from logger_utils import setup_logger, log_section, log_params, log_timing

# ====================================================================
# 单位系统: SI (kg, m, s)
# ====================================================================
units = PhysicalUnits()  # kg=1, m=1, s=1 → Pa = kg/(m·s²) = 1

# ---- 网格/几何参数 ----
long_axis = 'y'
angle_deg = 50.2
mesh_size = 1/4  # 网格尺寸 (m)
c1 = 1; c2 = 2  # 稳定化参数 (GLS)

# ====================================================================
# 稳定化模式 (同时控制单元阶次):
#   "none"      → P2/P1 + 无稳定化 (Taylor-Hood, inf-sup 稳定)
#   "none_p1p1" → P1/P1 + 无稳定化 (仅用于对比验证)
#   "gls"       → P1/P1 + GLS 稳定化
#   "oss"       → P1/P1 + GLS+OSS 稳定化
# ====================================================================
stab_mode = "none_p1p1"  # 可选: "none", "none_p1p1", "gls", "oss"
stab_label = "BDF2(noStab)" 

# 根据 stab_mode 自动确定单元阶次 (无需手动改 v_order/p_order)
if stab_mode == "none":
    v_order, p_order = 2, 1   # P2/P1 Taylor-Hood
else:
    v_order, p_order = 1, 1   # P1/P1 (gls / oss / none_p1p1)

# ============================================================
# 材料参数 (各向异性纤维增强 Standard Reinforced Model)
# ============================================================
G = 5.67e6 * units.Pa           # 基体剪切模量 (Pa)
Gf = 5.67e6 * units.Pa          # 纤维刚度 (Pa)
f0 = [0.0, 0.5, 0.866]          # 纤维方向 (绕 y 轴长轴)
rho = 1.1e3 * units.kg/(units.m**3)   # 密度 (kg/m³)
mu0 = G + Gf                    # 初始等效剪切模量 (Pa)
c_mu = np.sqrt(mu0 / rho)       # 剪切波速 (m/s)

dt = 0.001                   # 时间步长 (s)
T_end = 0.1                     # 总模拟时间 (s)
num_steps = int(T_end / dt)

# ====================================================================
# 输出目录 + 运行日志
# ====================================================================
RESULT_DIR = f"results/dynamic_twisting_column_{stab_label}_v{v_order}_p{p_order}_mesh{mesh_size}_dt{dt:.4f}_{stab_mode}"
os.makedirs(RESULT_DIR, exist_ok=True)

comm = MPI.comm_world
rank = comm.rank

# ---- 日志配置 (使用 logger_utils) ----
log = setup_logger(RESULT_DIR, log_name_prefix="run")
log_section(log, "3D 扭转柱动态求解器 — BDF2 (v,p)")
log.info(f"运行时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
log.info(f"结果目录: {RESULT_DIR}")
log.info(f"稳定化模式: {stab_mode}  (V={v_order}, P={p_order})")
log.info(f"  进程数: {comm.size}")

# ---- 清理旧文件 ----
if rank == 0:
    for pattern in [f"{RESULT_DIR}/*.h5", f"{RESULT_DIR}/*.xdmf"]:
        for f in glob.glob(pattern):
            try:
                os.remove(f)
                log.debug(f"  已清理旧文件: {f}")
            except OSError:
                pass
comm.barrier()

# ====================================================================
# 计时
# ====================================================================
timings = {
    '网格生成': 0.0,
    '求解器初始化': 0.0,
    '求解': 0.0,
}
t_total_start = time.time()

log.info(f"材料: Standard Reinforced Model, G={G:.2e} Pa, Gf={Gf:.2e} Pa")
log.info(f"      μ₀=G+Gf={mu0:.2e} Pa, ρ={rho:.1f} kg/m³, c_μ={c_mu:.1f} m/s")
log.info(f"      f0={f0}")
log.info(f"      dt={dt}s, T_end={T_end}s, num_steps={num_steps}")
log.info(f"单元: P{v_order}/P{p_order} (速度/压力)")

# ====================================================================
# 1. 网格生成 (1×1×6 m)
# ====================================================================
t0 = time.time()
mesh, boundaries, point_D = create_three_dimensional_beam_mesh(
    mesh_size=mesh_size, long_axis=long_axis, angle_deg=angle_deg)
timings['网格生成'] = time.time() - t0
log.info(f"网格生成完成: {timings['网格生成']:.2f}s, "
         f"单元数={mesh.num_cells()}, 节点数={mesh.num_vertices()}")
log.info(f"  参考点 D (旋转后): ({point_D[0]:.4f}, {point_D[1]:.4f}, {point_D[2]:.4f})")

# 导出网格和边界 (仅用于调试检查)
xdmf_mesh = XDMFFile(comm, f"{RESULT_DIR}/mesh.xdmf")
xdmf_mesh.write(mesh)
xdmf_mesh.close()

xdmf_bnd = XDMFFile(comm, f"{RESULT_DIR}/boundaries.xdmf")
xdmf_bnd.write(boundaries)
xdmf_bnd.close()
log.debug(f"网格/边界已导出到 {RESULT_DIR}/")

# ====================================================================
# 2. 求解器初始化
# ====================================================================
t0 = time.time()
material = StandardReinforcedModel(G=G, Gf=Gf, f0=f0)

# 根据 stab_mode 设置求解器参数
_force_stab    = (stab_mode in ("gls", "oss"))
_use_oss       = (stab_mode == "oss")
_force_no_stab = (stab_mode == "none_p1p1")

solver = DynamicConsistentGLSBDF2Solver(
    mesh, boundaries, material, dt=dt, rho=rho,
    v_order=v_order, p_order=p_order, c1=c1, c2=c2,
    force_stab=_force_stab, use_oss=_use_oss,
    force_no_stab=_force_no_stab,
)
timings['求解器初始化'] = time.time() - t0
log.info(f"求解器初始化: {timings['求解器初始化']:.2f}s")

# ---- 输出求解器 GLS 诊断信息 ----
if rank == 0:
    d = solver._diag
    log.info(f"\nGLS 稳定参数 (Newton-then-Stabilize):")
    log.info(f"  dt={d['dt']:.6f}, dt_mu ∈ [{d['dt_mu_min']:.6f}, {d['dt_mu_max']:.6f}]")
    log.info(f"  h ∈ [{d['h_min']:.4f}, {d['h_max']:.4f}], "
             f"c_mu = {d['c_mu']:.4f}  (mu={d['mu']:.2f}, rho={d['rho']})")
    log.info(f"  dt/dt_mu_min = {d['ratio']:.1f}x  |  "
             f"c1={d['c1']}, c2={d['c2']}")
    log.info(f"  τ_GLS ∈ [{d['tau_min']:.2e}, {d['tau_max']:.2e}]")
    if d['ratio'] > 10.0:
        log.warning(f"  ⚠ dt 超出 dt_mu_min {d['ratio']:.1f}x，建议减小 dt 或增大 c1/c2")

# ====================================================================
# 初始条件: 扭转模式 v₀ = 100*sin(πy/12)*(z, 0, -x)  (m/s)
#   绕 y 轴 (长轴) 扭转:
#     v_x = 100*sin(π*y/12)*z  → y=6 时 max v_x=±50 (z=±0.5)
#     v_y = 0
#     v_z = -100*sin(π*y/12)*x → y=6 时 max v_z=∓50 (x=±0.5)
# ====================================================================
initial_velocity = Expression(
    ("100.0*sin(pi*x[1]/12.0)*x[2]",     # v_x = 100*sin(π*y/12)*z
     "0.0",                               # v_y = 0
     "-100.0*sin(pi*x[1]/12.0)*x[0]"),   # v_z = -100*sin(π*y/12)*x
    degree=2
)

# 设置初始速度到 w_n 和 w_nn
v_space = solver.W.sub(0).collapse()
v_init = interpolate(initial_velocity, v_space)
assign(solver.w_n.sub(0), v_init)
solver.w_nn.assign(solver.w_n)

log.info(f"初始条件: u₀=0, v₀=100*sin(πy/12)*(z, 0, -x) m/s")
log.info(f"  v₀_max = {100.0*0.5:.1f} m/s (y=6, |x|=|z|=0.5 处)")
log.debug(f"  u_n 范数: {solver.u_n.vector().norm('l2'):.2e}")
log.debug(f"  u_nn 范数: {solver.u_nn.vector().norm('l2'):.2e}")

# ====================================================================
# 边界条件: 长轴起始端 y=0 固定 (标记 6 = MinEnd)
# ====================================================================
bc_bottom = DirichletBC(
    solver.W.sub(0), Constant((0.0, 0.0, 0.0)), boundaries, 6)
bcs = [bc_bottom]
log.info("边界条件: 长轴起始端 (标记 6) 完全固定 v=(0,0,0)")
# ====================================================================
# 3. 时间循环
# ====================================================================
log.info(f"开始时间积分: {num_steps} 步, dt={dt}s, T={T_end}s")

# 共享 XDMF 输出 (形变+压力在同一文件)
xdmf = XDMFFile(comm, f"{RESULT_DIR}/results.xdmf")
xdmf.parameters["flush_output"] = True
xdmf.parameters["functions_share_mesh"] = True

# 监测点: 自由端顶角 D (旋转后坐标)
point = Point(*point_D)
time_points = []
ux_values = []
uy_values = []
uz_values = []
csv_filename = f"{RESULT_DIR}/ux_v{v_order}_p{p_order}_{stab_label}_dt{dt:.4f}_{stab_mode}.csv"

t_loop_start = time.time()
step_timings = []

try:
    for step in range(num_steps):
        t_step_start = time.time()
        t = (step + 1) * dt

        use_pred = (step > 0)
        is_first = (step == 0)

        log.debug(f"  Step {step+1}/{num_steps}: t={t:.4f}, "
                  f"BDF={'BE' if is_first else 'BDF2'}, predictor={use_pred}")

        solver.update_oss_projection()
        u_sol, v_sol, p_sol = solver.solve(
            bcs, tol=1e-8, use_predictor=use_pred, first_step=is_first
        )

        # 诊断: 稳定项 vs Galerkin 范数比
        if solver._use_stab and (step <= 1 or step % 20 == 0):
            G_p    = assemble(solver.galerkin_pressure_form)
            S_pspg = assemble(solver.stab_pspg_form)
            log.info(f"  [诊断 step={step}] ||G_p||={G_p.norm('l2'):.3e}  "
                     f"||PSPG||={S_pspg.norm('l2'):.3e}  "
                     f"ratio_p={S_pspg.norm('l2')/max(G_p.norm('l2'),1e-30):.3e}")

        # BDF2 历史更新
        if step > 0:
            solver.w_nn.assign(solver.w_n)
        solver.w_n.assign(solver.w)

        # 命名并保存
        u_sol.rename("displacement", "displacement")
        p_sol.rename("pressure", "pressure")

        # 每 5 步保存 XDMF
        if step % 5 == 0 or step == num_steps - 1:
            xdmf.write(u_sol, t)
            xdmf.write(p_sol, t)

        # 监测点位移
        ux_local = 0.0; uy_local = 0.0; uz_local = 0.0
        count_local = 0
        try:
            ux_local = u_sol(point)[0]
            uy_local = u_sol(point)[1]
            uz_local = u_sol(point)[2]
            count_local = 1
        except RuntimeError:
            pass
        ux_sum = MPI.sum(comm, ux_local)
        uy_sum = MPI.sum(comm, uy_local)
        uz_sum = MPI.sum(comm, uz_local)
        count_sum = MPI.sum(comm, count_local)
        ux = ux_sum / count_sum if count_sum > 0 else 0.0
        uy = uy_sum / count_sum if count_sum > 0 else 0.0
        uz = uz_sum / count_sum if count_sum > 0 else 0.0
        time_points.append(t)
        ux_values.append(ux)
        uy_values.append(uy)
        uz_values.append(uz)

        # 每步计时
        t_step = time.time() - t_step_start
        step_timings.append(t_step)
        if rank == 0 and (step < 5 or step % 20 == 0 or step == num_steps - 1):
            log.info(f"  Step {step+1:>4d}/{num_steps} | t={t:.4f} | "
                     f"ux={ux:.6e} m | 耗时={t_step:.3f}s")

    xdmf.close()

    t_loop = time.time() - t_loop_start
    timings['求解'] = t_loop

    if rank == 0:
        log.info(f"求解完成! 最终 ux={ux_values[-1]:.6e}, "
                 f"uy={uy_values[-1]:.6e}, uz={uz_values[-1]:.6e} m")
        log.info(f"总求解时间: {t_loop:.2f}s, "
                 f"平均每步: {t_loop/num_steps:.3f}s")

        # 保存 CSV
        with open(csv_filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['time', 'ux', 'uy', 'uz'])
            for ti, uxi, uyi, uzi in zip(time_points, ux_values, uy_values, uz_values):
                writer.writerow([ti, uxi, uyi, uzi])
        log.info(f"CSV 已保存: {csv_filename}")

except Exception as e:
    log.error(f"求解失败: {e}")
    if rank == 0:
        import traceback
        traceback.print_exc()
    try:
        xdmf.close()
    except:
        pass
    comm.barrier()
    exit(1)

# ====================================================================
# 计时统计
# ====================================================================
t_total = time.time() - t_total_start
if rank == 0:
    log_timing(log, timings, t_total, title="计时统计")

    if len(step_timings) > 0:
        log.info(f"每步耗时: min={min(step_timings):.3f}s, "
                 f"max={max(step_timings):.3f}s, "
                 f"mean={sum(step_timings)/len(step_timings):.3f}s")

    # 材料参数摘要
    log_section(log, "材料参数")
    log.info(f"  G   = {G:.2e} Pa (基体剪切模量)")
    log.info(f"  Gf  = {Gf:.2e} Pa (纤维刚度)")
    log.info(f"  μ₀  = G+Gf = {mu0:.2e} Pa ({mu0/1e6:.4f} MPa)")
    log.info(f"  f0  = {f0} (纤维方向)")
    log.info(f"  ρ   = {rho:.1f} kg/m³")
    log.info(f"  cμ  = {c_mu:.2f} m/s (剪切波速)")
    log.info(f"  dt  = {dt:.4f} s")
    log.info(f"  求解器: DynamicConsistentGLSBDF2Solver (BDF2)")
    log.info(f"  稳定化: {stab_mode}  (V={v_order}, P={p_order})")
    log.info(f"  输出:    {RESULT_DIR}/")
    if len(time_points) > 0:
        log.info(f"\n最终位移 (监测点): ux={ux_values[-1]:.6e}, "
                 f"uy={uy_values[-1]:.6e}, uz={uz_values[-1]:.6e} m")

        # --- ux ---
        plt.figure(figsize=(10, 5))
        plt.plot(time_points, ux_values, 'b-', linewidth=2)
        plt.xlabel('Time (s)'); plt.ylabel('Displacement ux (m)')
        plt.title(f'3D Twisting Column — {stab_mode} BDF2 (V{v_order}P{p_order})')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.savefig(f'{RESULT_DIR}/ux_vs_time_{stab_mode}.png', dpi=300)
        plt.close()

        # --- uy ---
        plt.figure(figsize=(10, 5))
        plt.plot(time_points, uy_values, 'r-', linewidth=2)
        plt.xlabel('Time (s)'); plt.ylabel('Displacement uy (m)')
        plt.title(f'3D Twisting Column — {stab_mode} BDF2 (V{v_order}P{p_order})')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.savefig(f'{RESULT_DIR}/uy_vs_time_{stab_mode}.png', dpi=300)
        plt.close()

        # --- uz ---
        plt.figure(figsize=(10, 5))
        plt.plot(time_points, uz_values, 'g-', linewidth=2)
        plt.xlabel('Time (s)'); plt.ylabel('Displacement uz (m)')
        plt.title(f'3D Twisting Column — {stab_mode} BDF2 (V{v_order}P{p_order})')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.savefig(f'{RESULT_DIR}/uz_vs_time_{stab_mode}.png', dpi=300)
        plt.close()

        log.info(f"图片已保存: ux/uy/uz_vs_time_{stab_mode}.png")
    else:
        log.warning("无数据可绘图.")
