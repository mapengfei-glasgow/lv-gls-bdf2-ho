"""Helpers for run_all.sh: rebuild summary JSONs and compare regenerated files with git HEAD.

  python tools/repro_tools.py lagrange  AUG_DIR AUG_LOG PURE_DIR PURE_LOG OUT_DIR
  python tools/repro_tools.py glsshort  GLS_DIR NOSTAB_DIR OUT_JSON
  python tools/repro_tools.py compare   REPORT_MD RUN_SUMMARY
"""
import csv, io, json, re, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIAG = re.compile(r"residual|_norm|error", re.I)   # solver-tolerance-level quantities


def rows(p):
    return list(csv.DictReader(open(p)))


def max_newton(log):
    its = [int(m) for m in re.findall(r"iterations=(\d+)", Path(log).read_text())]
    return max(its) if its else None


def lagrange_block(d, log):
    r = rows(Path(d) / "monitor_lagrange_full_cycle.csv")
    f = lambda k: [float(x[k]) for x in r]
    return dict(steps=len(r), end_time_s=float(r[-1]["time_s"]), max_Newton_iterations=max_newton(log),
                J_quadrature_min=min(f("J_quadrature_min")), J_quadrature_max=max(f("J_quadrature_max")),
                max_J_minus_1_RMS=max(f("J_minus_1_RMS")), V_min_mL=min(f("V_mL")), V_max_mL=max(f("V_mL")),
                V_final_mL=f("V_mL")[-1], max_displacement_m=max(f("max_u_m")),
                VTK_frames_read_back=len(list((Path(d) / "vtk").glob("*.vtu"))))


def cmd_lagrange(aug, aug_log, pure, pure_log, out):
    a, p = lagrange_block(aug, aug_log), lagrange_block(pure, pure_log)
    out = Path(out)
    (out / "validation_summary.json").write_text(json.dumps(a, indent=2) + "\n")
    (out / "pure_vs_augmented.json").write_text(json.dumps({"pure_lagrange": p, "augmented_lagrange": a}, indent=2) + "\n")


def cmd_glsshort(gls, nostab, out):
    keys = ["p_min_Pa", "p_max_Pa", "J_minus_1_RMS", "galerkin_p_norm", "pspg_norm", "total_residual_norm"]
    g, n = rows(Path(gls) / "monitor_gls.csv")[-1], rows(Path(nostab) / "monitor_gls.csv")[-1]
    Path(out).write_text(json.dumps({
        "description": "Same P1/P1 space, BE/BDF2, LV loads and dt; only GLS inclusion differs. Short diagnostic, not stability proof.",
        "time_s": float(g["time_s"]),
        "gls": {k: float(g[k]) for k in keys}, "none_p1p1": {k: float(n[k]) for k in keys},
        "note": "PSPG norm in none_p1p1 is a candidate diagnostic only; that term is not added to the unstabilized equation."},
        indent=1) + "\n")


def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def cmp_csv(new_txt, ref_txt):
    A, B = list(csv.DictReader(io.StringIO(new_txt))), list(csv.DictReader(io.StringIO(ref_txt)))
    if len(A) != len(B):
        return None, f"rows {len(A)} vs {len(B)}", None
    worst, worst_diag = (0.0, ""), (0.0, "")
    for c in B[0]:
        ref = [num(r[c]) for r in B]; new = [num(r.get(c)) for r in A]
        if any(v is None for v in ref + new):
            continue
        scale = max(abs(v) for v in ref) or 1.0
        d = max(abs(x - y) for x, y in zip(new, ref)) / scale
        tgt = "diag" if DIAG.search(c) else "main"
        if tgt == "main" and d > worst[0]: worst = (d, c)
        if tgt == "diag" and d > worst_diag[0]: worst_diag = (d, c)
    return worst[0], worst[1], worst_diag


def flat(d, p=""):
    if isinstance(d, dict):
        for k, v in d.items():
            yield from flat(v, f"{p}{k}.")
    elif isinstance(d, list):
        for i, v in enumerate(d):
            yield from flat(v, f"{p}{i}.")
    else:
        yield p[:-1], d


def cmp_json(new_txt, ref_txt):
    A, B = dict(flat(json.loads(new_txt))), dict(flat(json.loads(ref_txt)))
    worst, worst_diag, other = (0.0, ""), (0.0, ""), []
    for k, rv in B.items():
        nv = A.get(k)
        if num(rv) is None or isinstance(rv, bool):
            if nv != rv: other.append(k)
            continue
        if num(nv) is None:
            other.append(k); continue
        d = abs(float(nv) - float(rv)) / (abs(float(rv)) or 1.0)
        if DIAG.search(k):
            if d > worst_diag[0]: worst_diag = (d, k)
        elif d > worst[0]:
            worst = (d, k)
    return worst[0], worst[1] + (f"; changed: {', '.join(other[:4])}" if other else ""), worst_diag


def verdict(d):
    return "bitwise" if d == 0 else "round-off" if d <= 1e-8 else "tolerance" if d <= 1e-4 else "DIFFERENT"


def cmd_compare(report, run_summary):
    tracked = subprocess.run(["git", "ls-tree", "-r", "--name-only", "HEAD"], cwd=ROOT,
                             capture_output=True, text=True, check=True).stdout.split()
    targets = [p for p in tracked if p.endswith((".csv", ".json")) and not p.startswith("runs/")
               and not p.endswith("mesh_plain/info.json")]
    prev = {  # previous macOS run (commit 43e2fc4) -> regenerated location
        "p1_plain_full_cycle_program/reference_output/monitor_plain_p1_full_cycle.csv":
            "runs/20260924_012845/p1_plain/monitor_plain_p1_full_cycle.csv",
        "lv_lagrange_vtk_and_code/monitor_lagrange_full_cycle.csv":
            "runs/20260924_012845/lagrange/monitor_lagrange_full_cycle.csv",
        "ho_pdf_revision/results/full_cycle/monitor_gls.csv": "runs/20260924_012845/gls_hopdf/monitor_gls.csv",
        "ho_pdf_revision/results/full_cycle/validation_summary.json":
            "runs/20260924_012845/gls_hopdf/validation_summary.json",
    }
    out = ["# Reproducibility check", "",
           "Regenerated files vs. the versions in the previous commit. Relative difference = max |new-ref| / max |ref| per column",
           "(CSV) or |new-ref|/|ref| per key (JSON). Residual / norm / error quantities are reported separately",
           "because they sit at solver-tolerance level. Verdict: bitwise (0), round-off (<=1e-8), tolerance (<=1e-4), DIFFERENT.", "",
           "## Run status", "", "```", Path(run_summary).read_text().strip(), "```", "",
           "## vs. original package results (previous commit)", "",
           "| file | verdict | max rel. diff (field) | residual-type max rel. diff |", "|---|---|---|---|"]

    def line(path, ref_spec):
        f = ROOT / path
        if not f.exists():
            return f"| `{path}` | NOT REGENERATED | | |"
        ref = subprocess.run(["git", "show", f"HEAD:{ref_spec}"], cwd=ROOT, capture_output=True, text=True).stdout
        d, where, diag = (cmp_csv if path.endswith(".csv") else cmp_json)(f.read_text(), ref)
        if d is None:
            return f"| `{path}` | DIFFERENT | {where} | |"
        dg = f"{diag[0]:.1e} ({diag[1]})" if diag and diag[1] else ""
        return f"| `{path}` | {verdict(d)} | {d:.1e} ({where}) | {dg} |"

    for p in targets:
        out.append(line(p, p))
    out += ["", "## vs. previous macOS run (commit 43e2fc4, runs/20260924_012845)", "",
            "| file | verdict | max rel. diff (field) | residual-type max rel. diff |", "|---|---|---|---|"]
    for p, ref in prev.items():
        out.append(line(p, ref))
    Path(report).write_text("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    {"lagrange": cmd_lagrange, "glsshort": cmd_glsshort, "compare": cmd_compare}[sys.argv[1]](*sys.argv[2:])
