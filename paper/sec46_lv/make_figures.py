"""Figures for Sec. 4.6 (dynamic LV problem) from data/lv_timeseries.csv.

The CSV is extracted from results/gls_pdf_fs (GLS P1/P1, HO with mu_eff incl. fs,
c_t=1.5, c_s=3, C_tau=0.03, C_c=0.1). Cavity volume = exact surface integral over
the endocardial facets (tools/convergence.py:cavity_volume).
Run: python make_figures.py   -> figures/lv_*.pdf (+ .png previews)
"""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"     # validated categorical slots 1-3
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({
    "font.family": "serif", "mathtext.fontset": "cm", "font.size": 9,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "axes.linewidth": 0.6,
    "xtick.color": INK2, "ytick.color": INK2, "xtick.labelcolor": INK, "ytick.labelcolor": INK,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 3, "ytick.major.size": 3,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "lines.linewidth": 1.6, "legend.frameon": False, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})

rows = list(csv.DictReader(open(HERE / "data" / "lv_timeseries.csv")))
D = {k: np.array([float(r[k]) for r in rows]) for k in rows[0]}
t = D["time_s"]
ED = int(np.argmin(abs(t - 0.5)))
ES = int(np.argmin(np.where(t < 0.8, D["V_cavity_mL"], np.inf)))
PHASES = [(0.0, 0.2, "filling"), (0.2, 0.5, "diastasis"), (0.5, 0.8, "contraction\n& relaxation"), (0.8, 1.0, "diastole")]


def phases(ax, label=False):
    for i, (a, b, name) in enumerate(PHASES):
        if i % 2 == 0:
            ax.axvspan(a, b, color="#f1f0ec", lw=0, zorder=0)
        if label:
            ax.text((a + b) / 2, 1.02, name, transform=ax.get_xaxis_transform(), ha="center", va="bottom",
                    fontsize=7.5, color=INK2)
    ax.set_xlim(0, 1)


def label(ax, x, y, s, color=INK, **kw):
    ax.annotate(s, (x, y), fontsize=8.5, color=color, **kw)


def save(fig, name):
    fig.savefig(HERE / "figures" / f"{name}.pdf")
    fig.savefig(HERE / "figures" / f"{name}.png", dpi=200)
    plt.close(fig)


# --- Fig. A: prescribed loading ------------------------------------------------
fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.3), constrained_layout=True)
ax = axs[0]; phases(ax, True)
ax.plot(t, D["p_endo_kPa"], color=BLUE)
ax.set_xlabel("$t$ (s)"); ax.set_ylabel(r"$p_{\mathrm{endo}}$ (kPa)")
ax.annotate(f"{D['p_endo_kPa'].max():.1f} kPa", (0.65, D["p_endo_kPa"].max()), (0.83, 12.5), fontsize=8.5,
            arrowprops=dict(arrowstyle="-", color=INK2, lw=0.6))
ax.set_title("(a)", loc="left", fontsize=9, fontweight="bold", pad=16)
ax = axs[1]; phases(ax, True)
ax.plot(t, D["Ta_kPa"], color=BLUE)
ax.set_xlabel("$t$ (s)"); ax.set_ylabel(r"$T_a$ (kPa)")
ax.annotate(f"{D['Ta_kPa'].max():.1f} kPa", (0.65, D["Ta_kPa"].max()), (0.82, 72), fontsize=8.5,
            arrowprops=dict(arrowstyle="-", color=INK2, lw=0.6))
ax.set_title("(b)", loc="left", fontsize=9, fontweight="bold", pad=16)
save(fig, "lv_loading")

# --- Fig. B: global response --------------------------------------------------
fig, axs = plt.subplots(1, 3, figsize=(6.8, 2.35), constrained_layout=True)
V, p = D["V_cavity_mL"], D["p_endo_kPa"]
ax = axs[0]; phases(ax)
ax.plot(t, V, color=BLUE)
for i, s, xy in ((ED, "ED", (0.14, 382)), (ES, "ES", (0.70, 150))):
    ax.plot(t[i], V[i], "o", ms=4.5, color=BLUE, mec="white", mew=1.0, zorder=3)
    ax.annotate(f"{s}: {V[i]:.0f} mL", (t[i], V[i]), xy, fontsize=8.5,
                arrowprops=dict(arrowstyle="-", color=INK2, lw=0.6) if s == "ED" else None)
ax.set_ylim(120, 400)
ax.set_xlabel("$t$ (s)"); ax.set_ylabel("cavity volume $V$ (mL)")
ax.set_title("(a)", loc="left", fontsize=9, fontweight="bold", pad=3)

ax = axs[1]
ax.plot(V, p, color=BLUE)
for i in (60, 115, 140, 185):
    j = i + 2
    ax.annotate("", (V[j], p[j]), (V[i], p[i]), arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=0, mutation_scale=9))
for i, s, off in ((ED, "ED", (6, -12)), (ES, "ES", (6, 2))):
    ax.plot(V[i], p[i], "o", ms=4.5, color=BLUE, mec="white", mew=1.0, zorder=3)
    ax.annotate(s, (V[i], p[i]), textcoords="offset points", xytext=off, fontsize=8.5)
ax.set_xlabel("cavity volume $V$ (mL)"); ax.set_ylabel(r"$p_{\mathrm{endo}}$ (kPa)")
ax.set_title("(b)", loc="left", fontsize=9, fontweight="bold", pad=3)

ax = axs[2]; phases(ax)
L, T = D["long_axis_length_mm"], D["equatorial_wall_thickness_mm"]
ax.axhline(1, color=INK2, lw=0.6, ls=(0, (2, 2)))
ax.plot(t, L / L[0], color=BLUE, ls="-")
ax.plot(t, T / T[0], color=ORANGE, ls=(0, (5, 1.5)))
ax.set_ylim(0.6, 1.31)
ax.annotate("long-axis length", (0.44, 1.252), (0.60, 1.255), fontsize=8.5, va="bottom",
            arrowprops=dict(arrowstyle="-", color=INK2, lw=0.6))
ax.annotate("wall thickness", (0.42, 0.637), (0.20, 0.705), fontsize=8.5,
            arrowprops=dict(arrowstyle="-", color=INK2, lw=0.6))
ax.set_xlabel("$t$ (s)"); ax.set_ylabel("ratio to reference")
ax.set_title("(c)", loc="left", fontsize=9, fontweight="bold", pad=3)
save(fig, "lv_global_response")

# --- Fig. C: numerical behaviour ----------------------------------------------
fig, axs = plt.subplots(1, 3, figsize=(6.8, 2.3), constrained_layout=True)
tt = t[1:]
ax = axs[0]; phases(ax)
ax.plot(tt, D["newton_iterations"][1:], ls="none", marker="o", ms=2.6, color=BLUE)
ax.set_ylim(0, 4.5); ax.set_yticks([0, 1, 2, 3, 4])
ax.set_xlabel("$t$ (s)"); ax.set_ylabel("Newton iterations per step")
ax.set_title("(a)", loc="left", fontsize=9, fontweight="bold", pad=3)
ax = axs[1]; phases(ax)
ax.plot(tt, 100 * (D["J_mean"][1:] - 1), color=BLUE)
ax.axhline(0, color=INK2, lw=0.6)
ax.set_xlabel("$t$ (s)"); ax.set_ylabel(r"$\bar J - 1$ (%)")
ax.set_title("(b)", loc="left", fontsize=9, fontweight="bold", pad=3)
ax = axs[2]; phases(ax)
ax.plot(tt, D["p_max_kPa"][1:], color=BLUE, label=r"$\max\,p_h$")
ax.plot(tt, D["p_min_kPa"][1:], color=ORANGE, ls=(0, (5, 1.5)), label=r"$\min\,p_h$")
ax.plot(tt, D["p_endo_kPa"][1:], color=AQUA, ls=(0, (1, 1.2)), lw=1.4, label=r"$p_{\mathrm{endo}}$")
ax.legend(loc="upper left", fontsize=8, handlelength=2.2, borderaxespad=0.2)
ax.set_ylim(-4, 15.5)
ax.set_xlabel("$t$ (s)"); ax.set_ylabel("pressure (kPa)")
ax.set_title("(c)", loc="left", fontsize=9, fontweight="bold", pad=3)
save(fig, "lv_numerics")
print("ED", t[ED], V[ED], "ES", t[ES], V[ES])
