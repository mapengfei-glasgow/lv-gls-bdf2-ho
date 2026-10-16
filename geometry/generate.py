"""Generate a Dalton-family LV mesh using only the bundled PCA assets."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import h5py
import numpy as np

HERE = Path(__file__).resolve().parent


def reconstruct(coefficients, center="mean"):
    """Standardised PCA offsets -> raw corners and solver-frame coordinates (cm)."""
    with np.load(HERE / "assets/pca_basis.npz") as pca:
        z = np.asarray(coefficients, dtype=np.float64)
        if z.shape != (40,) or not np.isfinite(z).all():
            raise ValueError("coefficients must contain 40 finite standardised PCA offsets")
        scores = z * pca["sigma"]
        if center == "anchor":
            scores += pca["anchor_scores"]
        elif center != "mean":
            raise ValueError("center must be mean or anchor")
        raw = (pca["mean"] + scores @ pca["modes"]).reshape(-1, 3)
    with np.load(HERE / "assets/topology.npz") as topology:
        base = raw[topology["base_raw_indices"], 2].mean()
        rotate = lambda p: np.column_stack((p[:, 2] - base, p[:, 1], -p[:, 0]))
        xyz = np.empty((len(topology["corner_ids"]) + len(topology["centroid_ids"]), 3))
        xyz[topology["corner_ids"]] = rotate(raw[topology["raw_indices"]])
        xyz[topology["centroid_ids"]] = rotate(raw[topology["centroid_corners"]].mean(1))
    return raw, xyz, scores


def generate(out, coefficients, center="mean"):
    raw, xyz, scores = reconstruct(coefficients, center)
    with np.load(HERE / "assets/topology.npz") as topology:
        cells = topology["cells"]
        p = xyz[cells]
        volumes = np.einsum("ij,ij->i", np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]),
                            p[:, 3] - p[:, 0]) / 6
        if not np.isfinite(xyz).all() or not np.isfinite(volumes).all() or (volumes <= 0).any():
            raise ValueError("PCA coefficients produce nonfinite or inverted tetrahedra")
        out = Path(out)
        if out.exists() and any(out.iterdir()):
            raise FileExistsError(f"output directory is not empty: {out}")
        with tempfile.TemporaryDirectory(prefix="lv-pca-") as tmp:
            src = Path(tmp)
            with h5py.File(src / "dalton_lv_mesh.h5", "w") as h:
                h["Mesh/mesh/geometry"] = xyz
                h["Mesh/mesh/topology"] = cells
                # Placeholders replaced by the bundled Laplace fibre generator.
                for name, vector in zip(("fibers", "sheets", "normals"), np.eye(3)):
                    h[f"Mesh/{name}"] = np.tile(vector, (len(cells), 1))
                for kind in ("cell", "facet"):
                    h[f"MeshTags/{kind}_tags/topology"] = topology[f"{kind}_topology"]
                    h[f"MeshTags/{kind}_tags/Values"] = topology[f"{kind}_values"]
            (src / "dalton_lv_mesh.xdmf").write_text((HERE / "mesh.xdmf.in").read_text().format(
                nn=len(xyz), nt=len(cells), nft=len(topology["facet_topology"]), h5="dalton_lv_mesh.h5"))
            # Generate into temporary storage so failures leave no partial output.
            dst = src / "with_fibers"
            subprocess.run([sys.executable, str(HERE / "fibers.py"), str(src), str(dst)], check=True)
            shutil.copytree(dst, out, dirs_exist_ok=True)
    fibre_info = json.loads((out / "fibres.json").read_text())
    fibre_info["source"] = "bundled PCA reconstruction"
    (out / "fibres.json").write_text(json.dumps(fibre_info, indent=2) + "\n")
    np.save(out / "coefficients.npy", np.asarray(coefficients, dtype=np.float64))
    np.save(out / "scores.npy", scores)
    (out / "meta.json").write_text(json.dumps(dict(
        center=center, coefficients=np.asarray(coefficients).tolist(), units="cm",
        nodes=len(xyz), tetrahedra=len(cells), wall_volume_cm3=float(volumes.sum()),
        min_tet_volume_cm3=float(volumes.min()),
        facet_tags={"base": 5, "endo": 6, "epi": 7},
        fibers="Laplace rule-based: endo +60 degrees, epi -60 degrees",
    ), indent=2) + "\n")
    return xyz


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=HERE / "generated/geom_0000")
    ap.add_argument("--center", choices=("mean", "anchor"), default="mean")
    ap.add_argument("--coefficients", type=Path, help=".npy array of 40 standardised PCA offsets")
    ap.add_argument("--scale", type=float, default=0, help="sample uniform offsets in [-scale, scale]")
    ap.add_argument("--modes", type=int, default=40, help="number of PCs varied when sampling (1-40)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if not 1 <= args.modes <= 40 or not np.isfinite(args.scale) or args.scale < 0:
        ap.error("require 1 <= modes <= 40 and a finite nonnegative scale")
    if args.coefficients and args.scale != 0:
        ap.error("choose --coefficients or --scale")
    z = np.zeros(40)
    if args.coefficients:
        z = np.load(args.coefficients, allow_pickle=False)
    else:
        z[:args.modes] = np.random.default_rng(args.seed).uniform(-args.scale, args.scale, args.modes)
    generate(args.out, z, args.center)
    print(f"Generated {args.out}")


if __name__ == "__main__":
    main()
