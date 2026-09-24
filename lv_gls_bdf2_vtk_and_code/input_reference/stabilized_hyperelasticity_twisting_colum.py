from dolfin import *
from ufl import Identity, tr, det, grad, inv, derivative, variable, ln, conditional, as_vector, sqrt, Max, Min
import numpy as np
from petsc4py import PETSc

# Material models
class NeoHookean:
    def __init__(self, mu):
        self.mu = mu

    def strain_energy(self, F):
        C = F.T*F
        J = det(F)
        return self.mu/2*(J**(-2/3)*tr(C) - 3)

# Material models
class StandardReinforcedModel:
    def __init__(self, G, Gf, f0):
        self.G = G
        self.Gf = Gf
        self.f0 = f0
        self.mu = G + Gf  # 初始等效剪切模量 (求解器诊断用)
        
    def strain_energy(self, F):
        C = F.T*F
        J = det(F)
        # 2D 等容第一不变量: I1_bar = J^(-2/3)*tr(C)
        I1_bar = J**(-2.0/3.0) * tr(C)
        f0_vec = as_vector(self.f0)
        I4f = inner(C * f0_vec, f0_vec)
        # 光滑化 max(I4f, 1.0): Newton 二次收敛
        I4f_smooth = 0.5 * (I4f + 1.0 + sqrt((I4f - 1.0)**2 + 1e-12))
        return self.G/2 * (I1_bar - 3) + self.Gf/2 * (I4f_smooth - 1.0)**2

class DynamicConsistentGLSBDF2Solver:
    """BDF2 时间积分的一致 GLS 稳定化求解器 (v, p 混合形式).

    统一的系数形式:
      加速度: a = (α1·v - α2·v_n + α3·v_nn) / (α0·Δt)
      位移:   u = (β1·u_n - β2·u_nn + β3·Δt·v) / β0

    第一步用 Backward Euler:       α=[1,1,0,1], β=[1,0,1,1]
    第二步及以后用 BDF2:          α=[3,4,1,2], β=[4,1,2,3]
    """
    def __init__(self, mesh, boundaries, material, dt, rho=1.0,
                 v_order=2, p_order=1, c1=4.0, c2=2.0, force_stab=False,
                 force_no_stab=False, use_oss=True):
        self.mesh = mesh; self.boundaries = boundaries
        self.material = material; self.h = CellDiameter(mesh)
        self.dt_c = Constant(dt); self.rho_c = Constant(rho)
        self._dt = dt; self.v_order = v_order; self.p_order = p_order
        # 稳定化开关优先级: force_no_stab > force_stab > 自动判定
        if force_no_stab:
            self._use_stab = False
        elif force_stab:
            self._use_stab = True
        else:
            self._use_stab = not (v_order >= 2 and p_order == v_order - 1)
        self._use_oss = use_oss and self._use_stab  # 仅稳定化开启时 OSS 有效

        # ---- 统一的符号系数（运行时可通过 assign 切换 BE / BDF2） ----
        self.alpha_0 = Constant(1.0)   # Δt 分母
        self.alpha_1 = Constant(1.0)   # v 系数
        self.alpha_2 = Constant(1.0)   # v_n 系数
        self.alpha_3 = Constant(0.0)   # v_nn 系数
        self.beta_0  = Constant(1.0)   # u 分母
        self.beta_1  = Constant(1.0)   # u_n 系数
        self.beta_2  = Constant(0.0)   # u_nn 系数
        self.beta_3  = Constant(1.0)   # Δt·v 系数

        Vv = VectorElement("CG", mesh.ufl_cell(), v_order)
        Q  = FiniteElement("CG", mesh.ufl_cell(), p_order)
        self.W = FunctionSpace(mesh, MixedElement([Vv, Q]))
        self.w = Function(self.W)
        self.w_n = Function(self.W)      # t^n
        self.w_nn = Function(self.W)     # t^{n-1}
        v, p = split(self.w)
        v_n, p_n = split(self.w_n)
        v_nn, p_nn = split(self.w_nn)
        self.w_v, q = TestFunctions(self.W)
        self.w_trial = TrialFunction(self.W)

        # 统一加速度: (α1·v - α2·v_n + α3·v_nn) / (α0·Δt)
        a_gen = (self.alpha_1 * v - self.alpha_2 * v_n
                 + self.alpha_3 * v_nn) / (self.alpha_0 * self.dt_c)

        Vu = VectorFunctionSpace(mesh, "CG", v_order)
        self.u_n = Function(Vu)
        self.u_nn = Function(Vu)
        # 统一位移: u = (β1·u_n - β2·u_nn + β3·Δt·v) / β0
        u = (self.beta_1 * self.u_n - self.beta_2 * self.u_nn
             + self.beta_3 * self.dt_c * v) / self.beta_0

        dim = mesh.geometry().dim(); I = Identity(dim)
        F = I + grad(u); F_v = variable(F)
        J = det(F_v); H = J * inv(F_v).T
        P = diff(self.material.strain_energy(F_v), F_v) - p * H

        R_mom = self.rho_c * a_gen - div(P)

        # ---- OSS 投影场 (仅 use_oss=True 时使用) ----
        if self._use_oss:
            V_xi = VectorFunctionSpace(mesh, "CG", 1)
            self.xi = Function(V_xi)
            xi_test = TestFunction(V_xi)
            xi_trial = TrialFunction(V_xi)
            self._xi_a_form = inner(xi_trial, xi_test) * dx
            self._xi_L_form = inner(R_mom, xi_test) * dx
            self._xi_A = assemble(self._xi_a_form)

            Q_eta = FunctionSpace(mesh, "CG", 1)
            self.eta = Function(Q_eta)
            eta_test = TestFunction(Q_eta)
            eta_trial = TrialFunction(Q_eta)
            self._eta_a_form = inner(eta_trial, eta_test) * dx
            self._eta_A = assemble(self._eta_a_form)
        else:
            self.xi = None
            self.eta = None
        
        # ============================================================
        # GLS 稳定参数 (Newton-then-Stabilize 框架)
        # ------------------------------------------------------------
        # τ⁻¹ = c1 · μ_0/h² + c2 · ρ_0/Δt²
        #   准静态 (Δt 大): τ ≈ h²/(c1·μ_0)
        #   动态 (Δt 小):    τ ≈ Δt²/(c2·ρ_0)
        # 默认 c1=4.0, c2=2.0 (Hughes-Franca-Hulbert, 线元 Fourier 分析)
        # ============================================================
        mu = material.mu
        self.c1 = c1; self.c2 = c2
        tau_gls = 1.0 / (c1 * mu / self.h**2 + c2 * self.rho_c / self.dt_c**2)

        # === 网格特征时间诊断 (静默模式, 仅写入日志) ===
        c_mu_val = float(sqrt(mu / self.rho_c))
        h_min = MPI.min(MPI.comm_world, mesh.hmin())
        h_max = MPI.max(MPI.comm_world, mesh.hmax())
        dt_mu_min = h_min / c_mu_val
        dt_mu_max = h_max / c_mu_val

        # 存储诊断值供外部日志使用 (不在此处 print)
        rho_val = float(self.rho_c)
        dt_val = float(self._dt)
        tau_min = 1.0 / (c1 * mu / h_min**2 + c2 * rho_val / dt_val**2)
        tau_max = 1.0 / (c1 * mu / h_max**2 + c2 * rho_val / dt_val**2)
        ratio = self._dt / dt_mu_min if dt_mu_min > 0 else float('inf')
        self._diag = {
            'dt': dt_val,
            'h_min': h_min, 'h_max': h_max,
            'c_mu': c_mu_val, 'mu': float(mu),
            'rho': rho_val,
            'dt_mu_min': dt_mu_min, 'dt_mu_max': dt_mu_max,
            'tau_min': tau_min, 'tau_max': tau_max,
            'ratio': ratio,
            'c1': c1, 'c2': c2,
        }

        if hasattr(self.material, 'pressure_residual'):
            R_p = self.material.pressure_residual(J, p)
        elif hasattr(self.material, 'kappa'):
            R_p = J - 1 + p / self.material.kappa
        else:
            R_p = inner(H, grad(v))

        # ---- 组装 R_p 的 OSS 投影线性形式 (仅 OSS 模式) ----
        if self._use_oss:
            self._eta_L_form = inner(R_p, eta_test) * dx

        Res_v = (inner(self.rho_c * a_gen, self.w_v) * dx
                 + inner(P, grad(self.w_v)) * dx)
        Res_p = inner(R_p, q) * dx

        # ---- 暴露子残差供诊断 ----
        self.galerkin_form = Res_v + Res_p
        self.galerkin_pressure_form = Res_p  # 仅压力方程的 Galerkin 项
        # 无 OSS 的纯 GLS 形式 (用于诊断对比)
        self.stab_lsic_noss = tau_gls * inner(H, grad(self.w_v)) * R_p * dx
        self.stab_pspg_noss = tau_gls * inner(R_mom, dot(H, grad(q))) * dx

        if self._use_stab:
            if self._use_oss:
                # OSS-GLS: 用 R_p - Π_h[R_p] 和 R_mom - Π_h[R_mom]
                Res_v += tau_gls * inner(H, grad(self.w_v)) * (R_p - self.eta) * dx
                Res_p += tau_gls * inner(R_mom - self.xi, dot(H, grad(q))) * dx
                # 暴露 OSS 稳定子残差
                self.stab_lsic_form = tau_gls * inner(H, grad(self.w_v)) * (R_p - self.eta) * dx
                self.stab_pspg_form = tau_gls * inner(R_mom - self.xi, dot(H, grad(q))) * dx
            else:
                # 纯 GLS (无 OSS 投影)
                Res_v += self.stab_lsic_noss
                Res_p += self.stab_pspg_noss
                self.stab_lsic_form = self.stab_lsic_noss
                self.stab_pspg_form = self.stab_pspg_noss
        else:
            self.stab_lsic_form = 0 * Res_v  # 占位, 不会用到
            self.stab_pspg_form = 0 * Res_p
        self.Res = Res_v + Res_p
        self.Jacobian = derivative(self.Res, self.w, self.w_trial)

        # 初始默认 BE
        self._set_backward_euler()

        # ---- 全局 PETSc 选项: 回溯线搜索 (仅在 __init__ 设置一次) ----
        PETScOptions.set("snes_linesearch_type", "bt")
        parameters.parse()

    # ---------- 系数切换 ----------
    def _set_backward_euler(self):
        """第一步：一阶 Backward Euler"""
        self.alpha_0.assign(1.0); self.alpha_1.assign(1.0)
        self.alpha_2.assign(1.0); self.alpha_3.assign(0.0)
        self.beta_0.assign(1.0);  self.beta_1.assign(1.0)
        self.beta_2.assign(0.0);  self.beta_3.assign(1.0)
        self._is_bdf2 = False

    def _set_bdf2(self):
        """第二步及以后：二阶 BDF2"""
        self.alpha_0.assign(2.0); self.alpha_1.assign(3.0)
        self.alpha_2.assign(4.0); self.alpha_3.assign(1.0)
        self.beta_0.assign(3.0);  self.beta_1.assign(4.0)
        self.beta_2.assign(1.0);  self.beta_3.assign(2.0)
        self._is_bdf2 = True

    # ---------- 位移更新（统一公式） ----------
    def _update_displacement(self, v_sol_vec):
        """u^{n+1} = (β1·u^n - β2·u^{n-1} + β3·Δt·v) / β0"""
        b0 = self.beta_0.values()[0]
        b1 = self.beta_1.values()[0]
        b2 = self.beta_2.values()[0]
        b3 = self.beta_3.values()[0]

        # u_tmp = β1 * u_n
        u_new = self.u_n.vector().copy()
        u_new *= b1
        # u_tmp -= β2 * u_nn
        u_new.axpy(-b2, self.u_nn.vector())
        # u_tmp += β3 * Δt * v_sol
        u_new.axpy(b3 * self._dt, v_sol_vec)
        # u_tmp /= β0
        u_new /= b0

        # 更新历史: u_nn ← u_n, u_n ← u_new
        self.u_nn.assign(self.u_n)
        self.u_n.vector().zero()
        self.u_n.vector().axpy(1.0, u_new)

    # ---------- OSS 投影更新 ----------
    def update_oss_projection(self):
        """滞后更新 xi = Π_h[R_mom] 和 eta = Π_h[R_p] (仅 OSS 模式)."""
        if not self._use_oss:
            return
        b_xi = assemble(self._xi_L_form)
        solve(self._xi_A, self.xi.vector(), b_xi)
        b_eta = assemble(self._eta_L_form)
        solve(self._eta_A, self.eta.vector(), b_eta)

    # ---------- 求解 ----------
    def solve(self, bcs, tol=1e-8, use_predictor=True, first_step=False,
              predictor_type="linear"):
        """predictor_type: "linear" (2*w_n - w_nn, 激进) 或 "constant" (w_n, 保守)"""
        # 1. 设置时间积分系数
        if first_step:
            self._set_backward_euler()
        else:
            self._set_bdf2()

        # 2. 预测器
        if use_predictor:
            self.w.vector().zero()
            if self._is_bdf2 and predictor_type == "linear":
                self.w.vector().axpy(2.0, self.w_n.vector())
                self.w.vector().axpy(-1.0, self.w_nn.vector())
            else:
                self.w.vector().axpy(1.0, self.w_n.vector())
            for bc in bcs:
                bc.apply(self.w.vector())

        # 3. 求解非线性问题 (回溯线搜索已在 __init__ 中全局配置)
        problem = NonlinearVariationalProblem(self.Res, self.w, bcs, self.Jacobian)
        solver = NonlinearVariationalSolver(problem)
        solver.parameters["nonlinear_solver"] = "newton"
        solver.parameters["newton_solver"]["linear_solver"] = "mumps"
        solver.parameters["newton_solver"]["absolute_tolerance"] = tol
        solver.parameters["newton_solver"]["maximum_iterations"] = 50
        solver.parameters["newton_solver"]["relaxation_parameter"] = 1.0
        solver.parameters["newton_solver"]["error_on_nonconvergence"] = True
        solver.solve()
        v_sol, p_sol = self.w.split(deepcopy=True)

        # 4. OSS 投影: 用收敛值更新 xi=Π_h[R_mom], eta=Π_h[R_p] (仅 OSS 模式)
        if self._use_oss:
            self.update_oss_projection()

        # 5. 位移更新 (BDF2 或 BE 统一公式)
        self._update_displacement(v_sol.vector())
        return Function(self.u_n.function_space(), self.u_n.vector()), v_sol, p_sol
