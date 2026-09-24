"""Prescribed loading, material constants and default solver options.

The loads are given in kPa (Eq. 79/80) and converted to Pa for the solver.
"""
import numpy as np

KPA_TO_PA = 1000.0


# Nominal load constants (Sec. 4.6); the data generator of the surrogate varies them.
P_ED_KPA = 1.067      # diastolic plateau of Eq. 79
P_AMP_KPA = 13.46     # systolic amplitude of Eq. 79 (peak p_endo = P_ED + 0.9964 P_AMP)
TA_AMP_KPA = 84.26    # amplitude of Eq. 80 (peak Ta = 0.9889 TA_AMP)


def p_endo_kPa(t, p_ed=P_ED_KPA, p_amp=P_AMP_KPA, period=1.0):
    """Endocardial pressure, Eq. 79, in kPa; the curve is evaluated at t/period."""
    t = t / period
    if t < 0.2:
        return p_ed * t / 0.2
    if t < 0.5:
        return p_ed
    if t < 0.65:
        return p_ed + p_amp * (1.0 - np.exp(-((t - 0.5) ** 2) / 0.004))
    if t < 0.8:
        return p_ed + p_amp * (1.0 - np.exp(-((0.8 - t) ** 2) / 0.004))
    return p_ed


def Ta_kPa(t, ta_amp=TA_AMP_KPA, period=1.0):
    """Active fibre tension amplitude, Eq. 80, in kPa; evaluated at t/period."""
    t = t / period
    if t < 0.5:
        return 0.0
    if t < 0.65:
        return ta_amp * (1.0 - np.exp(-((t - 0.5) ** 2) / 0.005))
    if t < 0.8:
        return ta_amp * (1.0 - np.exp(-((0.8 - t) ** 2) / 0.005))
    return 0.0


# Holzapfel-Ogden constants (Pa); kappa is only used by the P1 penalty method.
CONSTITUTIVE_PARAMS = {
    "a": 2244.87 / 10.0,
    "b": 1.6215,
    "a_f": 24267.0 / 10.0,
    "b_f": 1.8268,
    "a_s": 5562.38 / 10.0,
    "b_s": 0.7746,
    "a_fs": 3905.16 / 10.0,
    "b_fs": 1.695,
    "kappa": 1e6,
    "eta": 1e2,
    "k": 100.0,
}

SOLVER_PARAMS = {
    "snes_type": "newtonls",
    "snes_linesearch_type": "basic",
    "snes_max_it": 20,
    "snes_stol": 1e-8,
    "snes_atol": 1e-8,
    "snes_rtol": 1e-8,
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
    "snes_monitor": None,
}
