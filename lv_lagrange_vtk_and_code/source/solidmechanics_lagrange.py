"""Incompressible Taylor-Hood P2/P1 extension; p>0 denotes compression."""
import os
import basix.ufl
import numpy as np
import ufl
from dolfinx import fem
from dolfinx.fem.petsc import NonlinearProblem
from solidmechanicsproblem import SolidMechanicsProblem_Bar


class SolidMechanicsProblem_Lagrange(SolidMechanicsProblem_Bar):
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
        self.p = fem.Function(self.p_space, name="lagrange_pressure_Pa")
        self.u_old = fem.Function(self.u_space)
        self.v_old = fem.Function(self.u_space)

    def _first_piola(self, F, v):
        # Remove the original bulk energy. p enforces the weak constraint.
        # Consistent augmentation controls unresolved volume modes on coarse meshes.
        energy = self.material.strain_energy_1(F) - self.material.W_compress(ufl.det(F))
        beta = float(os.environ.get("SIMPLE_3D0D_LM_AUGMENTATION", "1e6"))
        if beta < 0:
            raise ValueError("LM augmentation must be nonnegative")
        energy += 0.5 * beta * (ufl.det(F) - 1)**2
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
        opts = dict(self.unflat["SOLVER_PARAMS"])
        opts.update(snes_linesearch_type="bt", snes_max_it=50,
                    snes_error_if_not_converged=True, ksp_error_if_not_converged=True)
        self.problem = NonlinearProblem(residual, self.state, bcs=[],
            petsc_options_prefix="lv_lagrange_", petsc_options=opts)

    def solve(self, step):
        self.problem.solve()
        reason = self.problem.solver.getConvergedReason()
        if reason <= 0 or not np.isfinite(self.state.x.array).all():
            raise RuntimeError(f"Step {step} failed: SNES reason {reason}")
        self.state.x.scatter_forward()
        self.u.x.array[:] = self.state.x.array[self.u_map]
        self.p.x.array[:] = self.state.x.array[self.p_map]
        self.u.x.scatter_forward()
        self.p.x.scatter_forward()
        print(f"Step {step}: iterations={self.problem.solver.getIterationNumber()}, reason={reason}", flush=True)
