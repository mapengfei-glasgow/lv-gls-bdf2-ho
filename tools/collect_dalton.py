"""Collect OUT_DIR/geom_*/summary.json (from tools/batch_dalton.sh) into one CSV.

  python tools/collect_dalton.py OUT_DIR [DATA_DIR] [-o OUT_DIR/summary_all.csv]

One row per finished geometry: the scalar entries of summary.json (V_ref, V_ED, V_ES, EF,
shortening, thickening, max_u_mm, p range, Newton counts, ...) plus split / wall_volume_cm3
from DATA_DIR/geom_*/meta.json when DATA_DIR is given, and the status from status.tsv.
"""
import argparse
import csv
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path); ap.add_argument("data", type=Path, nargs="?")
    ap.add_argument("-o", type=Path, default=None)
    a = ap.parse_args()
    status = {}
    if (a.out / "status.tsv").exists():
        for r in csv.DictReader(open(a.out / "status.tsv"), delimiter="\t"):
            status[r["geom"]] = r                       # last line per geometry wins
    rows = []
    for s in sorted(a.out.glob("geom_*/summary.json")):
        g = s.parent.name.split("_", 1)[1]
        d = json.loads(s.read_text())
        row = {"geom": g, "status": status.get(g, {}).get("status", ""),
               "seconds": status.get(g, {}).get("seconds", "")}
        row.update({k: v for k, v in d.items() if isinstance(v, (int, float, str, bool))})
        row["newton_counts"] = json.dumps(d.get("newton_counts", {}))
        if a.data and (a.data / f"geom_{g}" / "meta.json").exists():
            m = json.loads((a.data / f"geom_{g}" / "meta.json").read_text())
            row.update(split=m.get("split"), wall_volume_cm3=m.get("wall_volume_cm3"))
        rows.append(row)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    out = a.o or a.out / "summary_all.csv"
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(rows)
    failed = {g: r["status"] for g, r in status.items() if r["status"] != "ok"}
    print(f"{len(rows)} finished geometries -> {out}; not ok: {len(failed)} {dict(list(failed.items())[:10])}")


if __name__ == "__main__":
    main()
