"""Plot a training run's history.csv (written by surrogate.train) as a loss / rollout-RMSE curve.

  python tools/plot_history.py surrogate_runs/firstpass
  python tools/plot_history.py surrogate_runs/full --out /tmp/full_history.png

Top panel:    training losses (total = 1-step + pf-weight * pushforward, both parts separately)
              on a log scale.
Bottom panel: validation rollout RMSE (mm) at the --eval-every checkpoints.
Writes <run>/history.png by default.
"""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", type=Path, help="training run directory containing history.csv")
    ap.add_argument("--out", type=Path, default=None, help="output png (default: RUN/history.png)")
    a = ap.parse_args(argv)
    rows = list(csv.DictReader((a.run / "history.csv").open()))
    out = a.out or (a.run / "history.png")

    def col(name):
        return np.array([float(r[name]) if r.get(name) not in ("", None) else np.nan for r in rows])

    it = np.array([int(r["iter"]) for r in rows])
    loss, l1, lpf = col("loss"), col("loss_1step"), col("loss_pf")
    rmse = col("val_rollout_rmse_m") * 1e3
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(7.5, 6.5), sharex=True)
    a1.semilogy(it, loss, "-", lw=1.0, label="total (1-step + pf)", color="C0")
    a1.semilogy(it, l1, "-", lw=1.0, label="1-step", color="C1")
    a1.semilogy(it, lpf, "-", lw=1.0, label="pushforward", color="C2")
    a1.set_ylabel("training loss (normalised)")
    a1.set_title(a.run.name)
    a1.legend(fontsize=8)
    a1.grid(alpha=0.3, which="both")
    a2.plot(it, rmse, "o-", ms=4)
    a2.set_ylabel("rollout RMSE (mm)")
    a2.set_xlabel("iteration")
    a2.grid(alpha=0.3)
    best = np.nanargmin(rmse)
    a2.annotate(f"best {rmse[best]:.2f} mm @ {it[best]}", (it[best], rmse[best]),
                textcoords="offset points", xytext=(8, 8), fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"{len(rows)} rows -> {out}")


if __name__ == "__main__":
    main()
