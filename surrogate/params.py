"""Parameters varied by the surrogate data set (besides the geometry).

Nominal values are those of Sec. 4.6 (lvsolver.model). Material constants are sampled
log-uniformly as multiples of the nominal value, loads and the cycle length uniformly.
The network sees every parameter mapped to [-1, 1] (``to_unit``).
"""
import numpy as np

from lvsolver.model import CONSTITUTIVE_PARAMS as C, P_AMP_KPA, P_ED_KPA, TA_AMP_KPA

# name: (nominal, low, high, "log" | "lin"); low/high are absolute values
SPACE = {
    "a":          (C["a"],     C["a"] / 1.5,   C["a"] * 1.5,   "log"),   # Pa
    "b":          (C["b"],     C["b"] / 1.5,   C["b"] * 1.5,   "log"),   # -
    "a_f":        (C["a_f"],   C["a_f"] / 1.5, C["a_f"] * 1.5, "log"),   # Pa
    "b_f":        (C["b_f"],   C["b_f"] / 1.5, C["b_f"] * 1.5, "log"),   # -
    "ta_amp_kPa": (TA_AMP_KPA, 60.0,  110.0, "lin"),
    "p_ed_kPa":   (P_ED_KPA,   0.8,   1.6,   "lin"),                     # 6-12 mmHg
    "p_amp_kPa":  (P_AMP_KPA,  10.0,  16.0,  "lin"),
    "period_s":   (1.0,        0.75,  1.2,   "lin"),                     # 50-80 bpm
}
NAMES = list(SPACE)


def from_unit(u):
    """(n, 8) samples in [0, 1] -> (n, 8) physical values."""
    u = np.atleast_2d(u)
    out = np.empty_like(u, dtype=float)
    for j, (_, lo, hi, kind) in enumerate(SPACE.values()):
        out[:, j] = lo * (hi / lo) ** u[:, j] if kind == "log" else lo + (hi - lo) * u[:, j]
    return out


def to_unit(x):
    """(n, 8) physical values -> (n, 8) features in [-1, 1] (nominal is not necessarily 0)."""
    x = np.atleast_2d(np.asarray(x, dtype=float))
    out = np.empty_like(x)
    for j, (_, lo, hi, kind) in enumerate(SPACE.values()):
        out[:, j] = np.log(x[:, j] / lo) / np.log(hi / lo) if kind == "log" else (x[:, j] - lo) / (hi - lo)
    return 2.0 * out - 1.0


def solver_args(row):
    """Command-line options of lvsolver.run for one parameter row (dict, values str or float)."""
    a, b, a_f, b_f = (float(row[k]) for k in ("a", "b", "a_f", "b_f"))
    ta_amp, p_ed, p_amp, period = (float(row[k]) for k in ("ta_amp_kPa", "p_ed_kPa", "p_amp_kPa", "period_s"))
    return ["--set", f"a={a:.8g}", "--set", f"b={b:.8g}",
            "--set", f"a_f={a_f:.8g}", "--set", f"b_f={b_f:.8g}",
            "--ta-amp", f"{ta_amp:.8g}", "--p-ed", f"{p_ed:.8g}",
            "--p-amp", f"{p_amp:.8g}", "--period", f"{period:.8g}"]
