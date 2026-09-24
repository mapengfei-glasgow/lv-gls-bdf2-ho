"""Holzapfel-Ogden material: original LV form and the HO-PDF (GLS) variant."""
import ufl
from dolfinx import default_scalar_type, fem


def heaviside(x, k, use_exp: bool = True):
    """Smooth Heaviside approximation."""
    if use_exp:
        return 1 / (1 + ufl.exp(-k * (x - 1)))
    return ufl.conditional(ufl.ge(x, 0.0), 1.0, 0.0)


class HolzapfelOgden:
    """Original LV material: isochoric invariants, smooth-Heaviside fibre switch,
    volumetric penalty W_compress, viscosity eta and active fibre stress."""

    def __init__(self, f0, s0, mesh, parameters) -> None:
        self.f0 = f0
        self.s0 = s0
        for name in ("a", "b", "a_f", "b_f", "a_s", "b_s", "a_fs", "b_fs", "kappa", "eta", "k"):
            setattr(self, name, fem.Constant(mesh, default_scalar_type(parameters[name])))
        self.tau = fem.Constant(mesh, default_scalar_type(0.0))  # active tension Ta(t), set by the problem

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

    def active_pk1(self, F):
        """T = Ta (1 + 4.9 (sqrt(I4f_bar) - 1)) F f0 x f0."""
        C = F.T * F
        J = ufl.det(F)
        bar_C = C * ufl.elem_pow(J, -2.0 / 3.0)
        f0 = self.f0
        I_4f = ufl.inner(f0, bar_C * f0)
        return self.tau * (1.0 + 4.9 * (ufl.sqrt(I_4f) - 1.0)) * F * ufl.outer(f0, f0)

    def strain_energy(self, F):
        """Passive energy + volumetric penalty (all invariants isochoric)."""
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


class HolzapfelOgdenPDF(HolzapfelOgden):
    """HO form of the GLS PDF: only I1 is isochoric, I4f/I4s/I8fs use raw C,
    the fibre switch is the smooth positive part xi = (x + sqrt(x^2 + eps))/2."""

    epsilon = 1e-12

    def invariants_pdf(self, F):
        C = F.T * F
        return (ufl.det(F)**(-2.0/3.0) * ufl.tr(C),
                ufl.inner(self.f0, C*self.f0),
                ufl.inner(self.s0, C*self.s0),
                ufl.inner(self.f0, C*self.s0))

    def positive_part(self, x):
        return 0.5 * (x + ufl.sqrt(x*x + self.epsilon))

    def W_4(self, I4, direction):
        if direction not in ("f", "s"):
            raise ValueError("direction must be f or s")
        a = self.a_f if direction == "f" else self.a_s
        b = self.b_f if direction == "f" else self.b_s
        xi = self.positive_part(I4 - 1)
        return a/(2*b) * (ufl.exp(b*xi**2) - 1)

    def passive_energy_pdf(self, F):
        I1, I4f, I4s, I8fs = self.invariants_pdf(F)
        # W_1 differs from the printed energy only by a constant.
        return self.W_1(I1) + self.W_4(I4f, "f") + self.W_4(I4s, "s") + self.W_8(I8fs)

    def strain_energy(self, F):
        return self.passive_energy_pdf(F) + self.W_compress(ufl.det(F))

    def effective_stiffness(self, F, include_fs=False):
        """Scalar stiffness scale mu_eff used by the PDF stabilisation parameters."""
        I1, I4f, I4s, I8fs = self.invariants_pdf(F)
        xf, xs = self.positive_part(I4f-1), self.positive_part(I4s-1)
        mu = (self.a*ufl.exp(self.b*(I1-3))
              + 2*self.a_f*xf*ufl.exp(self.b_f*xf**2)
              + 2*self.a_s*xs*ufl.exp(self.b_s*xs**2))
        if include_fs:
            mu += 2*self.a_fs*ufl.sqrt(I8fs**2+self.epsilon)*ufl.exp(self.b_fs*I8fs**2)
        return mu


MATERIALS = {"ho": HolzapfelOgden, "ho_pdf": HolzapfelOgdenPDF}
