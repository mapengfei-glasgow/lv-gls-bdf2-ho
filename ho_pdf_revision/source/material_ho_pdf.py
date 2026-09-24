"""Passive HO energy and GLS scalar stiffness from the supplied PDF, pp. 1-2."""
import ufl
from material import HolzapfelOgden


class HolzapfelOgdenPDF(HolzapfelOgden):
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

    def strain_energy_1(self, F):
        return self.passive_energy_pdf(F) + self.W_compress(ufl.det(F))

    def effective_stiffness(self, F, include_fs=False):
        I1, I4f, I4s, I8fs = self.invariants_pdf(F)
        xf, xs = self.positive_part(I4f-1), self.positive_part(I4s-1)
        mu = (self.a*ufl.exp(self.b*(I1-3))
              + 2*self.a_f*xf*ufl.exp(self.b_f*xf**2)
              + 2*self.a_s*xs*ufl.exp(self.b_s*xs**2))
        if include_fs:
            mu += 2*self.a_fs*ufl.sqrt(I8fs**2+self.epsilon)*ufl.exp(self.b_fs*I8fs**2)
        return mu
