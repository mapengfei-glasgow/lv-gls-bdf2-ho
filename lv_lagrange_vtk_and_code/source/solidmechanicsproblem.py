"""Simplified 3D LV solid mechanics problem for the 3D-0D coupling demo.

Compared with the original wangxuan version, this file does not depend on the
coronary tree.  The active-stress term, if any, is applied over the whole
myocardium instead of coronary perfusion regions.
"""

import dolfinx
import ufl
from dolfinx import default_real_type, default_scalar_type, fem
from dolfinx.fem.petsc import NonlinearProblem


class SolidMechanicsProblem_Bar:
    def __init__(self, io_solid, material, unflat, time_curves):
        self.io_solid = io_solid
        self.material = material
        self.unflat = unflat
        self.ds = io_solid.ds
        self.dx = io_solid.dx

        if "MARKERS" in unflat:
            self.epi = unflat["MARKERS"]["epi"]
            self.top = unflat["MARKERS"]["base"]
            self.endo = unflat["MARKERS"]["endo"]
        else:
            # Defaults matching the LV_2 mesh.
            self.epi = 4096
            self.top = 4098
            self.endo = 4097

        # Base-constraint geometry.  Defaults preserve the original LV_2
        # behaviour (long axis z, base centre at 7.5 cm, 7.5 cm in the
        # scaled metre coordinates).  For other meshes, pass
        # unflat["GEOMETRY"] = {"long_axis": "x", "base_center": (...)}.
        geometry = unflat.get("GEOMETRY", {})
        self.long_axis = str(geometry.get("long_axis", "z")).lower()
        if self.long_axis not in ("x", "y", "z"):
            raise ValueError(
                f"long_axis must be x/y/z, got {self.long_axis!r}"
            )
        self.axial_index = {"x": 0, "y": 1, "z": 2}[self.long_axis]
        self.cross_indices = [
            i for i in range(3) if i != self.axial_index
        ]
        self.base_center = tuple(
            float(v) for v in geometry.get(
                "base_center", (0.075, 0.075, 0.0)
            )
        )

        self.parameters = {}
        self.parameters["rho"] = 1e3
        self.parameters["dt"] = unflat["TIME_PARAMS"]["dt"]
        self.parameters["alpha_top"] = 0.0
        self.parameters["alpha_epi"] = 0.0
        self.parameters["beta_top"] = 0.0
        self.parameters["beta_epi"] = 5e3
        self.parameters["gamma_top"] = 1e6
        self.N = ufl.FacetNormal(io_solid.msh)

        self.time_dependent_values = []

        pressure_magnitude = fem.Constant(io_solid.msh, default_real_type(0.0))
        self.parameters["p"] = pressure_magnitude
        self.time_dependent_values.append((pressure_magnitude, time_curves["4"]))

        active_stress = fem.Constant(io_solid.msh, default_real_type(0.0))
        material.tau = active_stress
        self.time_dependent_values.append((active_stress, time_curves["5"]))

        self._init_spaces()
        self._init_forms()

    def _init_spaces(self):
        self.u_space = fem.functionspace(
            self.io_solid.msh,
            ("CG", 1, (self.io_solid.msh.geometry.dim,)),
        )
        self.u = fem.Function(self.u_space, name="u")
        self.u_test = ufl.TestFunction(self.u_space)

        self.u_old = fem.Function(self.u_space)
        self.v_old = fem.Function(self.u_space)

    def _acceleration_form(self, a, w):
        return ufl.inner(self.parameters["rho"] * a, w) * self.dx

    def _first_piola(self, F, v):
        F_dot = ufl.grad(v)
        l = F_dot * ufl.inv(F)
        d = 0.5 * (l + l.T)
        E_dot = ufl.variable(F.T * d * F)
        return ufl.diff(self.material.strain_energy_1(F), F) + F * ufl.diff(
            self.material.W_visco(E_dot), E_dot
        )

    def _form(self, u, v, w):
        F = ufl.variable(ufl.grad(u) + ufl.Identity(3))
        P = self._first_piola(F, v)
        P_active = self.material._PK1_active(F)

        epi = ufl.dot(self.parameters["alpha_epi"] * u, self.N) + ufl.dot(
            self.parameters["beta_epi"] * v, self.N
        )

        # Base constraint that keeps the ventricle from moving as a rigid body.
        # Generalised to match left_ventricle: the axial component is fixed
        # (up to the weak penalty), while the circumferential component is
        # allowed to expand radially.
        c1, c2 = self.cross_indices
        axial = self.axial_index
        X0 = ufl.SpatialCoordinate(self.io_solid.msh)
        R = (
            X0[c1] - self.base_center[c1],
            X0[c2] - self.base_center[c2],
        )
        r = (
            X0[c1] + u[c1] - self.base_center[c1],
            X0[c2] + u[c2] - self.base_center[c2],
        )
        direction_length = (R[0] * r[0] + R[1] * r[1]) / (
            R[0] * R[0] + R[1] * R[1]
        )
        top_comp = [0.0, 0.0, 0.0]
        top_comp[c1] = -self.parameters["gamma_top"] * (
            direction_length * R[0] - r[0]
        )
        top_comp[c2] = -self.parameters["gamma_top"] * (
            direction_length * R[1] - r[1]
        )
        top_comp[axial] = self.parameters["gamma_top"] * u[axial]
        top = ufl.as_vector(top_comp)

        # Passive + (optional) active stress over the whole myocardium.
        form = ufl.inner(P, ufl.grad(w)) * self.dx
        form += ufl.inner(P_active, ufl.grad(w)) * self.dx
        form += self._pressure_term(F, w)
        form += ufl.inner(epi * w, self.N) * self.ds(self.epi)
        form += ufl.inner(top, w) * self.ds(self.top)
        return form

    def _init_forms(self):
        dt = self.parameters["dt"]
        a_new = (self.u - self.u_old - dt * self.v_old) / (dt**2)
        v_new = self.v_old + dt * a_new

        form = self._acceleration_form(a_new, self.u_test) + self._form(
            self.u, v_new, self.u_test
        )
        self.problem = NonlinearProblem(
            form,
            self.u,
            bcs=[],
            petsc_options_prefix="elastodynamics_3D_pressure_",
            petsc_options=self.unflat["SOLVER_PARAMS"],
        )

    def _pressure_term(self, F, w):
        return ufl.inner(
            self.parameters["p"] * ufl.det(F) * ufl.inv(F).T * self.N,
            w,
        ) * self.ds(self.endo)

    def update_time_dependent_values(self, t):
        for constant, curve in self.time_dependent_values:
            constant.value = curve(t)

    def solve(self, step):
        try:
            self.problem.solve()
            converged_reason = self.problem.solver.getConvergedReason()
            num_iterations = self.problem.solver.getIterationNumber()
            print(
                f"Step {step} converged in {num_iterations} iterations, "
                f"reason={converged_reason}"
            )
        except Exception as e:
            converged = False
            print(f"Step {step} FAILED to converge: {e}")
            return
        if converged_reason is None or converged_reason <= 0:
            print(f"Step {step}: convergence failed, reason={converged_reason}")

    def update_variables(self):
        dt = self.parameters["dt"]
        self.v_old.x.array[:] = (self.u.x.array - self.u_old.x.array) / dt
        self.u_old.x.array[:] = self.u.x.array
