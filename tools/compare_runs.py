"""Compare results/<case>/monitor.csv with a reference run (another directory or a git revision).

  python tools/compare_runs.py results --ref-git HEAD      # vs. last committed results
  python tools/compare_runs.py results --ref-dir other_results

Per column: max |new-ref| / max |ref| over time. Residual/norm columns sit at
solver-tolerance level and are reported separately. Exit code 1 if any main
column exceeds --tol (default 1e-10).
"""
import argparse
import csv
import io
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIAG = re.compile(r"residual|_norm$|rate_RMS")


def read_ref(args, rel):
    if args.ref_dir:
        p = Path(args.ref_dir) / rel
        return p.read_text() if p.exists() else None
    r = subprocess.run(["git", "show", f"{args.ref_git}:{rel.as_posix()}"], cwd=ROOT, capture_output=True, text=True)
    return r.stdout if r.returncode == 0 else None


def compare(new_txt, ref_txt):
    A, B = list(csv.DictReader(io.StringIO(new_txt))), list(csv.DictReader(io.StringIO(ref_txt)))
    if len(A) != len(B):
        return None, f"{len(A)} vs {len(B)} rows", (0.0, "")
    main, diag = (0.0, "-"), (0.0, "-")
    for c in B[0]:
        pairs = [(a.get(c, ""), b[c]) for a, b in zip(A, B)]
        if all(x == "" and y == "" for x, y in pairs):
            continue
        if any((x == "") != (y == "") for x, y in pairs):
            return None, f"column {c} presence differs", (0.0, "")
        new, ref = [float(x) for x, _ in pairs], [float(y) for _, y in pairs]
        d = max(abs(x - y) for x, y in zip(new, ref)) / (max(abs(y) for y in ref) or 1.0)
        if DIAG.search(c):
            diag = max(diag, (d, c))
        else:
            main = max(main, (d, c))
    return main, "", diag


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results", type=Path)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--ref-git", default="HEAD")
    g.add_argument("--ref-dir")
    ap.add_argument("--tol", type=float, default=1e-10)
    args = ap.parse_args()
    results = args.results.resolve()
    ok = True
    print(f"reference: {args.ref_dir or 'git ' + args.ref_git}   tol={args.tol:g}")
    print(f"{'case':28s} {'verdict':9s} {'max rel diff (column)':34s} residual-type")
    for mon in sorted(results.glob("*/monitor.csv")):
        rel = mon.relative_to(ROOT) if not args.ref_dir else mon.relative_to(results)
        ref = read_ref(args, rel)
        case = mon.parent.name
        if ref is None:
            print(f"{case:28s} {'NO REF':9s}")
            continue
        main_d, msg, diag = compare(mon.read_text(), ref)
        if main_d is None:
            ok = False
            print(f"{case:28s} {'FAIL':9s} {msg}")
            continue
        verdict = "bitwise" if main_d[0] == 0 else ("PASS" if main_d[0] <= args.tol else "FAIL")
        ok &= verdict != "FAIL"
        print(f"{case:28s} {verdict:9s} {main_d[0]:.2e} ({main_d[1]}){'':6s} {diag[0]:.2e} ({diag[1]})")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
