"""The three LV discretisations sharing loads, boundary terms and inertia.

Common model: rho a + Div(P) with Holzapfel-Ogden passive stress, viscosity
eta, active fibre stress Ta(t), endocardial follower pressure p_endo(t),
epicardial normal damping beta_epi and a base penalty gamma_top (axial
component fixed, in-plane radial motion free).

  P1Penalty  u in P1, compressible (W_compress, kappa), a = (u-u_n-dt v_n)/dt^2
  Lagrange   (u, p) in P2 x P1, W_compress replaced by -p cof(F) + beta/2 (J-1)^2
  GLS        (v, p) in P1 x P1, BE first step then BDF2, rate constraint
             cof(F):grad(v) = 0 with PSPG + LSIC stabilisation
"""
import numpy as np
import ufl
import basix.ufl
from dolfinx import default_real_type, default_scalar_type, fem
from dolfinx.fem.petsc import NonlinearProblem, assemble_vector
from mpi4py import MPI
from petsc4py import PETSc

from .model import SOLVER_PARAMS


class P1Penalty:
    method = "p1"
    degree = 1
    time_scheme = "be"
    prefix = "lv_p1_"

    def __init__(self, geo, material, dt, loads, base_center, long_axis="x"):
        self.io_solid = geo
        self.material = material
        self.ds, self.dx = geo.ds, geo.dx
        self.epi, self.top, self.endo = (geo.bdry_markers[k] for k in ("epi", "base", "endo"))

        self.long_axis = str(long_axis).lower()
        if self.long_axis not in ("x", "y", "z"):
            raise ValueError(f"long_axis must be x/y/z, got {self.long_axis!r}")
        self.axial_index = {"x": 0, "y": 1, "z": 2}[self.long_axis]
        self.cross_indices = [i for i in range(3) if i != self.axial_index]
        self.base_center = tuple(float(v) for v in base_center)

        self.parameters = {"rho": 1e3, "dt": dt, "alpha_top": 0.0, "alpha_epi": 0.0,
                           "beta_top": 0.0, "beta_epi": 5e3, "gamma_top": 1e6}
        if dt <= 0:
            raise ValueError("dt must be positive")
        self.N = ufl.FacetNormal(geo.msh)

        # (Constant, curve(t)) pairs: endocardial pressure and active tension, in Pa.
        pressure = fem.Constant(geo.msh, default_real_type(0.0))
        active = fem.Constant(geo.msh, default_real_type(0.0))
        self.parameters["p"] = pressure
        material.tau = active
        self.time_dependent_values = [(pressure, loads[0]), (active, loads[1])]

        self.p = None               # pressure field (mixed methods only)
        self.last_diagnostics = {}
        self.stab_scales = {}       # name -> UFL expression, exported per cell
        self._init_spaces()
        self._init_forms()

    # --- shared model -------------------------------------------------------
    def _acceleration_form(self, a, w):
        return ufl.inner(self.parameters["rho"] * a, w) * self.dx

    def _first_piola(self, F, v):
        F_dot = ufl.grad(v)
        l = F_dot * ufl.inv(F)
        d = 0.5 * (l + l.T)
        E_dot = ufl.variable(F.T * d * F)
        return ufl.diff(self.material.strain_energy(F), F) + F * ufl.diff(
            self.material.W_visco(E_dot), E_dot
        )

    def _form(self, u, v, w):
        """Stress, active stress and boundary terms of the momentum residual."""
        F = ufl.variable(ufl.grad(u) + ufl.Identity(3))
        P = self._first_piola(F, v)
        P_active = self.material.active_pk1(F)

        epi = ufl.dot(self.parameters["alpha_epi"] * u, self.N) + ufl.dot(
            self.parameters["beta_epi"] * v, self.N
        )

        # Base penalty: axial displacement penalised, in-plane motion only
        # allowed along the reference radial direction from the base centre.
        c1, c2 = self.cross_indices
        axial = self.axial_index
        X0 = ufl.SpatialCoordinate(self.io_solid.msh)
        R = (X0[c1] - self.base_center[c1], X0[c2] - self.base_center[c2])
        r = (X0[c1] + u[c1] - self.base_center[c1], X0[c2] + u[c2] - self.base_center[c2])
        direction_length = (R[0] * r[0] + R[1] * r[1]) / (R[0] * R[0] + R[1] * R[1])
        top_comp = [0.0, 0.0, 0.0]
        top_comp[c1] = -self.parameters["gamma_top"] * (direction_length * R[0] - r[0])
        top_comp[c2] = -self.parameters["gamma_top"] * (direction_length * R[1] - r[1])
        top_comp[axial] = self.parameters["gamma_top"] * u[axial]
        top = ufl.as_vector(top_comp)

        form = ufl.inner(P, ufl.grad(w)) * self.dx
        form += ufl.inner(P_active, ufl.grad(w)) * self.dx
        form += ufl.inner(self.parameters["p"] * ufl.det(F) * ufl.inv(F).T * self.N, w) * self.ds(self.endo)
        form += ufl.inner(epi * w, self.N) * self.ds(self.epi)
        form += ufl.inner(top, w) * self.ds(self.top)
        return form

    def update_time_dependent_values(self, t):
        for constant, curve in self.time_dependent_values:
            constant.value = curve(t)

    def _check(self, step, state):
        reason = self.problem.solver.getConvergedReason()
        if reason <= 0 or not np.isfinite(state.x.array).all():
            raise RuntimeError(f"Step {step} failed: SNES reason={reason}")
        its = self.problem.solver.getIterationNumber()
        self.last_diagnostics = {"newton_iterations": its, "snes_reason": int(reason)}
        print(f"Step {step}: iterations={its}, reason={reason}", flush=True)

    @property
    def configuration(self):
        return dict(method=self.method, velocity_degree=self.degree, time_scheme=self.time_scheme,
                    **self._configuration())

    def _configuration(self):
        return dict(element="P1 displacement", volumetric="penalty W_compress, kappa from material")

    # --- P1 penalty specifics -----------------------------------------------
    def _init_spaces(self):
        self.u_space = fem.functionspace(self.io_solid.msh, ("CG", 1, (3,)))
        self.u = fem.Function(self.u_space, name="displacement")
        self.u_test = ufl.TestFunction(self.u_space)
        self.u_old = fem.Function(self.u_space)
        self.v_old = fem.Function(self.u_space)
        self.velocity = fem.Function(self.u_space, name="velocity")

    def _init_forms(self):
        dt = self.parameters["dt"]
        a_new = (self.u - self.u_old - dt * self.v_old) / (dt**2)
        v_new = self.v_old + dt * a_new
        form = self._acceleration_form(a_new, self.u_test) + self._form(self.u, v_new, self.u_test)
        self.problem = NonlinearProblem(form, self.u, bcs=[], petsc_options_prefix=self.prefix,
                                        petsc_options=SOLVER_PARAMS)

    def solve(self, step):
        self.problem.solve()
        self._check(step, self.u)
        self.velocity.x.array[:] = (self.u.x.array - self.u_old.x.array) / self.parameters["dt"]

    def advance(self):
        """Shift time history after output of the converged step."""
        self.v_old.x.array[:] = self.velocity.x.array
        self.u_old.x.array[:] = self.u.x.array


class Lagrange(P1Penalty):
    method = "lagrange"
    degree = 2
    prefix = "lv_lagrange_"

    def __init__(self, *args, beta=1e6, **kwargs):
        if beta < 0:
            raise ValueError("augmentation beta must be nonnegative")
        self.beta = float(beta)
        super().__init__(*args, **kwargs)

    def _configuration(self):
        return dict(element="P2/P1 Taylor-Hood (u, p)", augmentation_Pa=self.beta,
                    volumetric="pressure multiplier -p cof(F) + beta/2 (J-1)^2; test eq. q (J-1)")

    def _init_spaces(self):
        mesh = self.io_solid.msh
        cell = mesh.basix_cell()
        element = basix.ufl.mixed_element([
            basix.ufl.element("Lagrange", cell, 2, shape=(3,)),
            basix.ufl.element("Lagrange", cell, 1),
        ])
        self.W = fem.functionspace(mesh, element)
        self.state = fem.Function(self.W, name="mixed_state")
        self.u_expr, self.p_expr = ufl.split(self.state)
        self.u_test, self.p_test = ufl.TestFunctions(self.W)
        self.u_space, self.u_map = self.W.sub(0).collapse()
        self.p_space, self.p_map = self.W.sub(1).collapse()
        self.u = fem.Function(self.u_space, name="displacement")
        self.p = fem.Function(self.p_space, name="pressure")
        self.u_old = fem.Function(self.u_space)
        self.v_old = fem.Function(self.u_space)
        self.velocity = fem.Function(self.u_space, name="velocity")

    def _first_piola(self, F, v):
        # Bulk energy removed; p enforces the weak constraint, beta augments it.
        energy = self.material.strain_energy(F) - self.material.W_compress(ufl.det(F))
        energy += 0.5 * self.beta * (ufl.det(F) - 1)**2
        F_dot = ufl.grad(v)
        l = F_dot * ufl.inv(F)
        E_dot = ufl.variable(F.T * (0.5 * (l + l.T)) * F)
        return (ufl.diff(energy, F)
                - self.p_expr * ufl.cofac(F)
                + F * ufl.diff(self.material.W_visco(E_dot), E_dot))

    def _init_forms(self):
        dt = self.parameters["dt"]
        a = (self.u_expr - self.u_old - dt * self.v_old) / dt**2
        v = (self.u_expr - self.u_old) / dt
        residual = self._acceleration_form(a, self.u_test)
        residual += self._form(self.u_expr, v, self.u_test)
        residual -= self.p_test * (ufl.det(ufl.Identity(3) + ufl.grad(self.u_expr)) - 1) * self.dx
        opts = dict(SOLVER_PARAMS)
        opts.update(snes_linesearch_type="bt", snes_max_it=50,
                    snes_error_if_not_converged=True, ksp_error_if_not_converged=True)
        self.problem = NonlinearProblem(residual, self.state, bcs=[],
                                        petsc_options_prefix=self.prefix, petsc_options=opts)

    def solve(self, step):
        self.problem.solve()
        self._check(step, self.state)
        self.state.x.scatter_forward()
        self.u.x.array[:] = self.state.x.array[self.u_map]
        self.p.x.array[:] = self.state.x.array[self.p_map]
        self.u.x.scatter_forward()
        self.p.x.scatter_forward()
        self.velocity.x.array[:] = (self.u.x.array - self.u_old.x.array) / self.parameters["dt"]


class GLS(P1Penalty):
    """Velocity-pressure BE/BDF2 with PSPG (pressure eq.) and LSIC (velocity eq.).

    tau="pdf"        omega = c_t rho/dt + c_s mu_eff dt/h^2, tau_m = C_tau/omega,
                     tau_c = C_c h^2 omega, mu_eff from HolzapfelOgdenPDF (default)
    tau="reference"  tau = 1/(c1 mu_ref/h^2 + c2 rho/dt^2) for both terms, as in
                     the uploaded DynamicConsistentGLSBDF2Solver
    stabilized=False keeps the terms only as diagnostics (Galerkin control);
                     velocity_degree=2 is then allowed (P2/P1 control).
    """
    method = "gls"
    prefix = "lv_gls_"

    def __init__(self, *args, tau="pdf", stabilized=True, velocity_degree=1, time_scheme="bdf2",
                 c_t=1.5, c_s=3.0, C_tau=0.03, C_c=0.1, include_fs=False,
                 c1=1.0, c2=2.0, mu_ref=None, **kwargs):
        if tau not in ("pdf", "reference"):
            raise ValueError("tau must be pdf or reference")
        if time_scheme not in ("bdf2", "be"):
            raise ValueError("time_scheme must be bdf2 or be")
        if velocity_degree not in (1, 2) or (stabilized and velocity_degree != 1):
            raise ValueError("stabilised GLS is P1/P1; velocity_degree=2 only with stabilized=False")
        if not all(np.isfinite(x) and x > 0 for x in (c_t, c_s, c1, c2)):
            raise ValueError("c_t, c_s, c1, c2 must be finite and positive")
        if not all(np.isfinite(x) and x >= 0 for x in (C_tau, C_c)):
            raise ValueError("C_tau and C_c must be finite and nonnegative")
        self.tau_kind, self.stabilized, self.degree, self.time_scheme = tau, stabilized, velocity_degree, time_scheme
        self.c_t, self.c_s, self.C_tau, self.C_c, self.include_fs = c_t, c_s, C_tau, C_c, include_fs
        self.c1, self.c2, self.mu_ref = c1, c2, mu_ref
        super().__init__(*args, **kwargs)

    def _init_spaces(self):
        mesh = self.io_solid.msh
        cell = mesh.basix_cell()
        mixed = basix.ufl.mixed_element([
            basix.ufl.element("Lagrange", cell, self.degree, shape=(3,)),
            basix.ufl.element("Lagrange", cell, 1),
        ])
        self.W = fem.functionspace(mesh, mixed)
        self.state = fem.Function(self.W, name="velocity_pressure")
        self.state_n = fem.Function(self.W)
        self.state_nn = fem.Function(self.W)
        self.v_expr, self.p_expr = ufl.split(self.state)
        self.v_n_expr, _ = ufl.split(self.state_n)
        self.v_nn_expr, _ = ufl.split(self.state_nn)
        self.u_test, self.p_test = ufl.TestFunctions(self.W)
        self.u_space, self.v_map = self.W.sub(0).collapse()
        self.p_space, self.p_map = self.W.sub(1).collapse()
        self.u = fem.Function(self.u_space, name="displacement")
        self.velocity = fem.Function(self.u_space, name="velocity")
        self.p = fem.Function(self.p_space, name="pressure")
        self.u_old = fem.Function(self.u_space)
        self.u_older = fem.Function(self.u_space)
        self.v_old = fem.Function(self.u_space)
        # a = (a1 v - a2 v_n + a3 v_nn)/(a0 dt),  u = (b1 u_n - b2 u_nn + b3 dt v)/b0
        self.alpha = [fem.Constant(mesh, default_scalar_type(v)) for v in (1, 1, 1, 0)]
        self.beta = [fem.Constant(mesh, default_scalar_type(v)) for v in (1, 1, 0, 1)]
        self._set_coefficients(False)
        dt = self.parameters["dt"]
        b0, b1, b2, b3 = self.beta
        a0, a1, a2, a3 = self.alpha
        self.u_expr = (b1*self.u_old-b2*self.u_older+b3*dt*self.v_expr)/b0
        self.a_expr = (a1*self.v_expr-a2*self.v_n_expr+a3*self.v_nn_expr)/(a0*dt)

    def _set_coefficients(self, bdf2):
        avals = (2, 3, 4, 1) if bdf2 else (1, 1, 1, 0)
        bvals = (3, 4, 1, 2) if bdf2 else (1, 1, 0, 1)
        for constant, value in zip(self.alpha, avals):
            constant.value = default_scalar_type(value)
        for constant, value in zip(self.beta, bvals):
            constant.value = default_scalar_type(value)
        self._bvals = bvals

    def _first_piola(self, F, v):
        # No p/kappa: the bulk penalty is removed, p is the rate-constraint multiplier.
        W = self.material.strain_energy(F)-self.material.W_compress(ufl.det(F))
        l = ufl.grad(v)*ufl.inv(F)
        E_dot = ufl.variable(F.T*(0.5*(l+l.T))*F)
        return (ufl.diff(W, F) - self.p_expr*ufl.cofac(F)
                + F*ufl.diff(self.material.W_visco(E_dot), E_dot))

    def _init_forms(self):
        mesh = self.io_solid.msh
        dt = self.parameters["dt"]
        rho = self.parameters["rho"]
        F = ufl.variable(ufl.Identity(3)+ufl.grad(self.u_expr))
        H = ufl.cofac(F)
        # Strong momentum residual includes every volume stress of the Galerkin form.
        P = self._first_piola(F, self.v_expr) + self.material.active_pk1(F)
        self.R_mom = rho*self.a_expr-ufl.div(P)
        self.R_p = ufl.inner(H, ufl.grad(self.v_expr))
        gal_v = self._acceleration_form(self.a_expr, self.u_test)
        gal_v += self._form(self.u_expr, self.v_expr, self.u_test)
        gal_p = self.R_p*self.p_test*self.dx
        h = ufl.CellDiameter(mesh)
        if self.tau_kind == "pdf":
            if not hasattr(self.material, "effective_stiffness"):
                raise ValueError("tau='pdf' needs the ho_pdf material")
            self.mu_eff = self.material.effective_stiffness(F, include_fs=self.include_fs)
            self.omega = self.c_t*rho/dt + self.c_s*self.mu_eff*dt/h**2
            self.tau_m = self.C_tau/self.omega
            self.tau_c = self.C_c*h**2*self.omega
            self.stab_lsic = self.tau_c*ufl.inner(H, ufl.grad(self.u_test))*self.R_p*self.dx
            self.stab_pspg = self.tau_m*ufl.inner(H*ufl.grad(self.p_test), self.R_mom)*self.dx
            self.stab_scales = {"mu_eff_Pa": self.mu_eff, "tau_m": self.tau_m, "tau_c": self.tau_c}
        else:
            # Reference-scale proxy a+a_f+a_s+a_fs, NOT an anisotropic tangent modulus.
            proxy = sum(float(getattr(self.material, name).value) for name in ("a", "a_f", "a_s", "a_fs"))
            self.mu_ref = float(self.mu_ref if self.mu_ref is not None else proxy)
            if self.mu_ref <= 0:
                raise ValueError("mu_ref must be positive")
            self.tau = 1/(self.c1*self.mu_ref/h**2 + self.c2*rho/dt**2)
            self.stab_lsic = self.tau*ufl.inner(H, ufl.grad(self.u_test))*self.R_p*self.dx
            self.stab_pspg = self.tau*ufl.inner(self.R_mom, H*ufl.grad(self.p_test))*self.dx
            self.stab_scales = {"tau_reference": self.tau}
        residual = gal_v+gal_p
        if self.stabilized:
            residual += self.stab_lsic+self.stab_pspg
        opts = dict(SOLVER_PARAMS)
        opts.update(snes_linesearch_type="bt", snes_max_it=60,
                    snes_atol=1e-10, snes_rtol=1e-9, snes_stol=1e-12,
                    snes_error_if_not_converged=True, ksp_error_if_not_converged=True)
        self.problem = NonlinearProblem(residual, self.state, bcs=[],
                                        petsc_options_prefix=self.prefix, petsc_options=opts)
        self._diag_forms = {name: fem.form(form) for name, form in {
            "galerkin_p_norm": gal_p, "pspg_norm": self.stab_pspg,
            "lsic_norm": self.stab_lsic, "galerkin_v_norm": gal_v,
            "total_residual_norm": residual,
        }.items()}
        self._rate_squared = fem.form(self.R_p**2*self.dx)
        self._reference_volume = fem.assemble_scalar(fem.form(1.0*self.dx))
        V0 = fem.functionspace(mesh, ("DG", 0))
        hfun = fem.Function(V0)
        hfun.interpolate(fem.Expression(h, V0.element.interpolation_points))
        self.h_range = (float(mesh.comm.allreduce(hfun.x.array.min(), op=MPI.MIN)),
                        float(mesh.comm.allreduce(hfun.x.array.max(), op=MPI.MAX)))

    def _configuration(self):
        cfg = dict(element=f"P{self.degree}/P1 (v, p)", stabilized=self.stabilized, tau=self.tau_kind,
                   pressure_constraint="cofac(F):grad(v)", h_min_m=self.h_range[0], h_max_m=self.h_range[1])
        if self.tau_kind == "pdf":
            cfg.update(c_t=self.c_t, c_s=self.c_s, C_tau=self.C_tau, C_c=self.C_c,
                       include_fs_stiffness=self.include_fs,
                       positive_part_epsilon=self.material.epsilon)
        else:
            cfg.update(c1=self.c1, c2=self.c2, mu_ref_Pa=self.mu_ref)
        return cfg

    def solve(self, step):
        self._set_coefficients(step > 0 and self.time_scheme == "bdf2")
        # Conservative predictor (previous state); BDF2 history is unchanged.
        self.state.x.array[:] = self.state_n.x.array
        self.state.x.scatter_forward()
        self.problem.solve()
        self._check(step, self.state)
        self.state.x.scatter_forward()
        self.velocity.x.array[:] = self.state.x.array[self.v_map]
        self.p.x.array[:] = self.state.x.array[self.p_map]
        b0, b1, b2, b3 = self._bvals
        self.u.x.array[:] = (b1*self.u_old.x.array-b2*self.u_older.x.array
                             + b3*self.parameters["dt"]*self.velocity.x.array)/b0
        for fun in (self.u, self.velocity, self.p):
            fun.x.scatter_forward()
        for name, form in self._diag_forms.items():
            vec = assemble_vector(form)
            vec.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
            self.last_diagnostics[name] = float(vec.norm())
            vec.destroy()
        self.last_diagnostics["constraint_rate_RMS_per_s"] = float(np.sqrt(
            fem.assemble_scalar(self._rate_squared)/self._reference_volume))

    def advance(self):
        # Stabilisation scales depend on u_n, u_nn: evaluate them before this shift.
        self.u_older.x.array[:] = self.u_old.x.array
        self.u_old.x.array[:] = self.u.x.array
        self.v_old.x.array[:] = self.velocity.x.array
        self.state_nn.x.array[:] = self.state_n.x.array
        self.state_n.x.array[:] = self.state.x.array
        for fun in (self.u_older, self.u_old, self.v_old, self.state_nn, self.state_n):
            fun.x.scatter_forward()


METHODS = {"p1": P1Penalty, "lagrange": Lagrange, "gls": GLS}
