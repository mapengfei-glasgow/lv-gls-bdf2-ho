"""Batch data generation with the paper's solver (GLS P1/P1, mu_eff incl. fs, C_c = 0.1).

  python -m surrogate.generate surrogate_data/params.csv surrogate_data --jobs 4 [--steps 200]

Per row of params.csv (surrogate.sample_params): rule-based fibres (+60/-60 deg, NOT the
dataset's fibres) -> tools/check_mesh.py --strict -> lvsolver.run --no-vtk --save-fields with
dt = period/steps. Results: OUT/<case>/{fields/, monitor.csv, metadata.json, log.txt};
OUT/status.tsv logs every attempt. Finished cases are skipped (re-run to resume).
Meshes are looked up as MESH_ROOT/dalton_geom_<id>/ or MESH_ROOT/geom_<id>/ (HF layout).
Run inside the solver environment (environment.yml).
"""
import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .params import NAMES, solver_args

ROOT = Path(__file__).resolve().parents[1]


def find_mesh(root, gid):
    for d in (root / f"dalton_geom_{gid}", root / f"geom_{gid}"):
        if (d / "dalton_lv_mesh.xdmf").exists():
            return d
    raise FileNotFoundError(f"geometry {gid} not found under {root} (mesh/download_dalton.command {gid})")


def done(d):
    try:
        return json.loads((d / "metadata.json").read_text())["status"] == "completed" and (d / "fields" / "u.npy").exists()
    except (OSError, ValueError, KeyError):
        return False


def one(row, out, mesh_root, steps, extra, status, loose=False):
    d = out / row["case"]
    if done(d):
        return "skip"
    t0 = time.time()
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)
    mesh = out / "_meshes" / row["case"]
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1")

    def run(cmd, log):
        with (d / log).open("w") as fh:
            return subprocess.run(cmd, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT, env=env).returncode

    period = float(row["period_s"])
    steps_ = [
        ("fibres", [sys.executable, "tools/rule_based_fibers.py", str(find_mesh(mesh_root, row["geom"])), str(mesh)], "fibres.log"),
        ("check", [sys.executable, "tools/check_mesh.py", str(mesh)] + ([] if loose else ["--strict"]), "check.txt"),
        ("solve", [sys.executable, "-m", "lvsolver.run", "--method", "gls", "--tau", "pdf", "--include-fs",
                   "--mesh", str(mesh), "--out", str(d), "--no-vtk", "--save-fields",
                   "--dt", f"{period / steps:.10g}", "--steps", str(steps), *solver_args(row), *extra], "log.txt"),
    ]
    result = "ok"
    for name, cmd, log in steps_:
        if run(cmd, log) != 0:
            result = f"{name}_failed"
            break
    shutil.rmtree(mesh, ignore_errors=True)
    with status.open("a") as fh:
        fh.write(f"{row['case']}\t{result}\t{time.time() - t0:.0f}\n")
    print(f"{row['case']}: {result} ({time.time() - t0:.0f} s)", flush=True)
    return result


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("params", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--jobs", type=int, default=1, help="cases in parallel (1 MPI rank, 1 thread each)")
    ap.add_argument("--steps", type=int, default=200, help="time steps per cycle")
    ap.add_argument("--mesh-root", type=Path, default=ROOT / "mesh")
    ap.add_argument("--extra", default="", help="extra lvsolver.run options (quoted)")
    ap.add_argument("--loose-check", action="store_true",
                    help="run check_mesh without --strict (log only; do not abort on the fibre heuristic)")
    a = ap.parse_args(argv)
    rows = list(csv.DictReader(a.params.open()))
    missing = [k for k in ["case", "geom"] + NAMES if k not in rows[0]]
    if missing:
        raise SystemExit(f"{a.params}: missing columns {missing}")
    out = a.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if a.params.resolve() != (out / "params.csv"):
        shutil.copy(a.params, out / "params.csv")
    status = out / "status.tsv"
    if not status.exists():
        status.write_text("case\tresult\tseconds\n")
    with ThreadPoolExecutor(a.jobs) as pool:
        res = list(pool.map(lambda r: one(r, out, a.mesh_root.resolve(), a.steps, a.extra.split(), status,
                                          loose=a.loose_check), rows))
    counts = {k: res.count(k) for k in sorted(set(res))}
    print(f"done: {counts}")
    return 0 if all(r in ("ok", "skip") for r in res) else 1


if __name__ == "__main__":
    sys.exit(main())
