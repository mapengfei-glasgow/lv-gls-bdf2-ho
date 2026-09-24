"""Sobol design: one row per (geometry, sample) with the 8 parameters of surrogate/params.py.

  python -m surrogate.sample_params --ids 0001-0008 2400 2401 --out surrogate_data/params.csv
  python -m surrogate.sample_params --ids 0000-2966 --per-geom 1 --out params_all.csv   # full set

Split by geometry id (dataset split): < 2250 train, 2250-2399 val, >= 2400 test.
--nominal writes the Sec. 4.6 values for every row (solver / pipeline check).
"""
import argparse
import csv
from pathlib import Path

import numpy as np
import warnings

from scipy.stats import qmc

from .params import NAMES, SPACE, from_unit


def parse_ids(items):
    ids = []
    for it in items:
        if "-" in it:
            a, b = it.split("-")
            ids += [f"{i:04d}" for i in range(int(a), int(b) + 1)]
        else:
            ids.append(f"{int(it):04d}")
    return ids


def split_of(gid):
    g = int(gid)
    return "train" if g < 2250 else ("val" if g < 2400 else "test")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ids", nargs="+", required=True, help="geometry ids or ranges, e.g. 0001-0008 2400")
    ap.add_argument("--per-geom", type=int, default=1, help="parameter samples per geometry")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--nominal", action="store_true")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    ids = parse_ids(a.ids)
    n = len(ids) * a.per_geom
    if a.nominal:
        X = np.tile([v[0] for v in SPACE.values()], (n, 1))
    else:
        warnings.filterwarnings("ignore", message=".*balance properties.*")
        X = from_unit(qmc.Sobol(d=len(NAMES), scramble=True, seed=a.seed).random(n))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["case", "geom", "split"] + NAMES)
        k = 0
        for gid in ids:
            for j in range(a.per_geom):
                case = f"geom_{gid}" if a.per_geom == 1 else f"geom_{gid}_s{j}"
                w.writerow([case, gid, split_of(gid)] + [f"{x:.8g}" for x in X[k]])
                k += 1
    print(f"{n} rows -> {a.out}")


if __name__ == "__main__":
    main()
