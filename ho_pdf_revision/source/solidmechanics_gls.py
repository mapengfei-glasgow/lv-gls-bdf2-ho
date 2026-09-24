"""Velocity-pressure BE/BDF2 with PDF HO stiffness and separate PSPG/LSIC scales."""
import os
import basix.ufl
import numpy as np
import ufl
from dolfinx import fem, default_scalar_type
from dolfinx.fem.petsc import NonlinearProblem, assemble_vector
from mpi4py import MPI
from solidmechanicsproblem import SolidMechanicsProblem_Bar


class SolidMechanicsProblem_GLS(SolidMechanicsProblem_Bar):
    def _init_spaces(self):
        mesh = self.io_solid.msh
        self.mode = os.environ.get("LV_GLS_MODE", "gls")
        if self.mode not in ("gls", "none_p1p1", "none"):
            raise ValueError("LV_GLS_MODE must be gls, none_p1p1, or none")
        self.degree = 2 if self.mode == "none" else 1
        self.time_scheme = os.environ.get("LV_GLS_TIME_SCHEME", "bdf2")
        if self.time_scheme not in ("bdf2", "be"):
            raise ValueError("LV_GLS_TIME_SCHEME must be bdf2 or be")
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
        self.p = fem.Function(self.p_space, name="lagrange_pressure_Pa")
        self.u_old = fem.Function(self.u_space)
        self.u_older = fem.Function(self.u_space)
        self.v_old = fem.Function(self.u_space)
        self.alpha = [fem.Constant(mesh, default_scalar_type(v)) for v in (1, 1, 1, 0)]
        self.beta = [fem.Constant(mesh, default_scalar_type(v)) for v in (1, 1, 0, 1)]
        self._set_coefficients(False)
        dt = self.parameters["dt"]
        if dt <= 0:
            raise ValueError("dt must be positive")
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
        # Do not branch on material.kappa: that would silently introduce p/kappa.
        W = self.material.strain_energy_1(F)-self.material.W_compress(ufl.det(F))
        l = ufl.grad(v)*ufl.inv(F)
        E_dot = ufl.variable(F.T*(0.5*(l+l.T))*F)
        return (ufl.diff(W, F) - self.p_expr*ufl.cofac(F)
                + F*ufl.diff(self.material.W_visco(E_dot), E_dot))

    def _init_forms(self):
        mesh = self.io_solid.msh
        dt = self.parameters["dt"]
        F = ufl.variable(ufl.Identity(3)+ufl.grad(self.u_expr))
        H = ufl.cofac(F)
        # Include every volume stress used by the Galerkin momentum equation.
        P = self._first_piola(F, self.v_expr) + self.material._PK1_active(F)
        self.R_mom = self.parameters["rho"]*self.a_expr-ufl.div(P)
        self.R_p = ufl.inner(H, ufl.grad(self.v_expr))
        gal_v = self._acceleration_form(self.a_expr, self.u_test)
        gal_v += self._form(self.u_expr, self.v_expr, self.u_test)
        gal_p = self.R_p*self.p_test*self.dx
        self.c_t = float(os.environ.get("LV_GLS_CT", "1.5"))
        self.c_s = float(os.environ.get("LV_GLS_CS", "3.0"))
        self.C_tau = float(os.environ.get("LV_GLS_CTAU", "0.03"))
        self.C_c = float(os.environ.get("LV_GLS_CC", "0.1"))
        include_fs = os.environ.get("LV_GLS_INCLUDE_FS", "0")
        if include_fs not in ("0", "1"):
            raise ValueError("LV_GLS_INCLUDE_FS must be 0 or 1")
        self.include_fs = include_fs == "1"
        if not all(np.isfinite(x) and x > 0 for x in (self.c_t, self.c_s)):
            raise ValueError("c_t and c_s must be finite and positive")
        if not all(np.isfinite(x) and x >= 0 for x in (self.C_tau, self.C_c)):
            raise ValueError("C_tau and C_c must be finite and nonnegative")
        if any(key in os.environ for key in ("LV_GLS_C1", "LV_GLS_C2", "LV_GLS_MU_REF")):
            raise ValueError("Old GLS scale options removed; use CT, CS, CTAU, CC, INCLUDE_FS")
        h = ufl.CellDiameter(mesh)
        self.mu_eff = self.material.effective_stiffness(F, include_fs=self.include_fs)
        self.omega = self.c_t*self.parameters["rho"]/dt + self.c_s*self.mu_eff*dt/h**2
        self.tau_m = self.C_tau/self.omega
        self.tau_c = self.C_c*h**2*self.omega
        self.stab_lsic = self.tau_c*ufl.inner(H, ufl.grad(self.u_test))*self.R_p*self.dx
        self.stab_pspg = self.tau_m*ufl.inner(H*ufl.grad(self.p_test), self.R_mom)*self.dx
        residual = gal_v+gal_p
        if self.mode == "gls":
            residual += self.stab_lsic+self.stab_pspg
        opts = dict(self.unflat["SOLVER_PARAMS"])
        opts.update(snes_linesearch_type="bt", snes_max_it=60,
                    snes_atol=1e-10, snes_rtol=1e-9, snes_stol=1e-12,
                    snes_error_if_not_converged=True, ksp_error_if_not_converged=True)
        self.problem = NonlinearProblem(residual, self.state, bcs=[],
            petsc_options_prefix="lv_gls_bdf2_", petsc_options=opts)
        self._diag_forms = {name:fem.form(form) for name,form in {
            "galerkin_p_norm":gal_p, "pspg_norm":self.stab_pspg,
            "lsic_norm":self.stab_lsic, "galerkin_v_norm":gal_v,
            "total_residual_norm":residual,
        }.items()}
        self._rate_squared = fem.form(self.R_p**2*self.dx)
        self._reference_volume = fem.assemble_scalar(fem.form(1.0*self.dx))
        V0 = fem.functionspace(mesh, ("DG", 0))
        hfun = fem.Function(V0)
        hfun.interpolate(fem.Expression(h,V0.element.interpolation_points))
        self.configuration = dict(mode=self.mode, velocity_degree=self.degree, pressure_degree=1,
            time_scheme=self.time_scheme, c_t=self.c_t, c_s=self.c_s,
            C_tau=self.C_tau, C_c=self.C_c, include_fs_stiffness=self.include_fs,
            mu_eff_definition="PDF HO deformation-dependent scalar scale (not full tangent tensor)",
            h_min_m=float(mesh.comm.allreduce(hfun.x.array.min(), op=MPI.MIN)),
            h_max_m=float(mesh.comm.allreduce(hfun.x.array.max(), op=MPI.MAX)),
            pressure_constraint="cofac(F):grad(v)", augmentation_Pa=0.0,
            material_invariants="I1 is isochoric; I4f/I4s/I8fs use raw C",
            positive_part_epsilon=self.material.epsilon,
            formulation="PDF separate PSPG/LSIC scales with retained rate constraint", OSS=False)

    def solve(self, step):
        self._set_coefficients(step>0 and self.time_scheme=="bdf2")
        # Conservative predictor; BDF2 history itself is unchanged.
        self.state.x.array[:] = self.state_n.x.array
        self.state.x.scatter_forward()
        self.problem.solve()
        reason = self.problem.solver.getConvergedReason()
        if reason <= 0 or not np.isfinite(self.state.x.array).all():
            raise RuntimeError(f"Step {step} failed: SNES reason={reason}")
        self.state.x.scatter_forward()
        self.velocity.x.array[:] = self.state.x.array[self.v_map]
        self.p.x.array[:] = self.state.x.array[self.p_map]
        b0,b1,b2,b3 = self._bvals
        self.u.x.array[:] = (b1*self.u_old.x.array-b2*self.u_older.x.array
                             +b3*self.parameters["dt"]*self.velocity.x.array)/b0
        for fun in (self.u,self.velocity,self.p):
            fun.x.scatter_forward()
        self.last_diagnostics = {}
        for name,form in self._diag_forms.items():
            vec=assemble_vector(form)
            from petsc4py import PETSc
            vec.ghostUpdate(addv=PETSc.InsertMode.ADD,mode=PETSc.ScatterMode.REVERSE)
            self.last_diagnostics[name]=float(vec.norm())
            vec.destroy()
        self.last_diagnostics["constraint_rate_RMS_per_s"] = float(np.sqrt(
            fem.assemble_scalar(self._rate_squared)/self._reference_volume))
        self.last_diagnostics["newton_iterations"] = self.problem.solver.getIterationNumber()
        self.last_diagnostics["snes_reason"] = int(reason)
        print(f"Step {step}: iterations={self.problem.solver.getIterationNumber()}, reason={reason}",flush=True)

    def update_variables(self):
        # Called only after output/diagnostics for the current converged state.
        self.u_older.x.array[:] = self.u_old.x.array
        self.u_old.x.array[:] = self.u.x.array
        self.v_old.x.array[:] = self.velocity.x.array
        self.state_nn.x.array[:] = self.state_n.x.array
        self.state_n.x.array[:] = self.state.x.array
        for fun in (self.u_older,self.u_old,self.v_old,self.state_nn,self.state_n):
            fun.x.scatter_forward()
