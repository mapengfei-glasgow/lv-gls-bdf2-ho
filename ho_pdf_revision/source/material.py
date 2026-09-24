"""Minimal Holzapfel-Ogden passive/active material for the 3D LV solid."""

import ufl
from dolfinx import default_scalar_type, fem


def heaviside(x, k, use_exp: bool = True):
    """Smooth Heaviside approximation using UFL expressions."""
    if use_exp:
        return 1 / (1 + ufl.exp(-k * (x - 1)))
    return ufl.conditional(ufl.ge(x, 0.0), 1.0, 0.0)


class HolzapfelOgden:
    def __init__(self, f0, s0, mesh, parameters=None) -> None:
        parameters = parameters or {}
        defaults = HolzapfelOgden.default_parameters()
        defaults.update(parameters)
        self.f0 = f0
        self.s0 = s0

        self.a = fem.Constant(mesh, default_scalar_type(defaults["a"]))
        self.b = fem.Constant(mesh, default_scalar_type(defaults["b"]))
        self.a_f = fem.Constant(mesh, default_scalar_type(defaults["a_f"]))
        self.b_f = fem.Constant(mesh, default_scalar_type(defaults["b_f"]))
        self.a_s = fem.Constant(mesh, default_scalar_type(defaults["a_s"]))
        self.b_s = fem.Constant(mesh, default_scalar_type(defaults["b_s"]))
        self.a_fs = fem.Constant(mesh, default_scalar_type(defaults["a_fs"]))
        self.b_fs = fem.Constant(mesh, default_scalar_type(defaults["b_fs"]))
        self.kappa = fem.Constant(mesh, default_scalar_type(defaults["kappa"]))
        self.eta = fem.Constant(mesh, default_scalar_type(defaults["eta"]))
        self.k = fem.Constant(mesh, default_scalar_type(defaults["k"]))
        self.tau = fem.Constant(mesh, default_scalar_type(0.0))

    @staticmethod
    def default_parameters():
        return {
            "a": 59.0,
            "b": 8.023,
            "a_f": 18472.0,
            "b_f": 16.026,
            "a_s": 2481.0,
            "b_s": 11.120,
            "a_fs": 216.0,
            "b_fs": 11.436,
            "kappa": 1e6,
            "eta": 1e2,
            "k": 100.0,
        }

    def W_1(self, I1):
        return self.a / (2.0 * self.b) * (ufl.exp(self.b * (I1 - 3)) - 1.0)

    def W_4(self, I4, direction):
        assert direction in ["f", "s"]
        a = self.a_f if direction == "f" else self.a_s
        b = self.b_f if direction == "f" else self.b_s
        return (
            a / (2.0 * b)
            * heaviside(I4, k=self.k)
            * (ufl.exp(b * (I4 - 1) ** 2) - 1.0)
        )

    def W_8(self, I8):
        return self.a_fs / (2.0 * self.b_fs) * (ufl.exp(self.b_fs * I8**2) - 1.0)

    def W_compress(self, J):
        return 0.25 * self.kappa * (J**2 - 1 - 2 * ufl.ln(J))

    def W_visco(self, E_dot):
        return 0.5 * self.eta * ufl.tr(E_dot * E_dot)

    def Wactive(self, I4f):
        return 0.5 * self.tau * (I4f - 1)

    def strain_energy(self, F):
        C = F.T * F
        J = ufl.det(F)
        f0 = self.f0 / ufl.sqrt(ufl.dot(self.f0, self.f0))
        s0 = self.s0 / ufl.sqrt(ufl.dot(self.s0, self.s0))
        I1 = pow(J, -2 / 3) * ufl.tr(C)
        I4f = ufl.inner(f0, C * f0)
        I4s = ufl.inner(s0, C * s0)
        I8fs = ufl.inner(f0, C * s0)
        return (
            self.W_1(I1)
            + self.W_4(I4f, "f")
            + self.W_4(I4s, "s")
            + self.W_8(I8fs)
            + self.Wactive(I4f)
            + self.W_compress(J)
        )

    def _PK1_active(self, F):
        C = F.T * F
        J = ufl.det(F)
        bar_C = C * ufl.elem_pow(J, -2.0 / 3.0)
        f0 = self.f0
        I_4f = ufl.inner(f0, bar_C * f0)
        P = self.tau * (1.0 + 4.9 * (ufl.sqrt(I_4f) - 1.0)) * F * ufl.outer(f0, f0)
        return P

    def strain_energy_1(self, F):
        C = F.T * F
        J = ufl.det(F)
        bar_C = C * ufl.elem_pow(J, -2.0 / 3.0)

        f0 = self.f0
        s0 = self.s0

        I1 = pow(J, -2 / 3) * ufl.tr(C)
        I4f = ufl.inner(f0, bar_C * f0)
        I4s = ufl.inner(s0, bar_C * s0)
        I8fs = ufl.inner(f0, bar_C * s0)

        Wcompress = self.W_compress(J)

        W1 = self.W_1(I1)
        W4f = self.W_4(I4f, "f")
        W4s = self.W_4(I4s, "s")
        W8fs = self.W_8(I8fs)

        return W1 + W4f + W4s + W8fs + Wcompress
