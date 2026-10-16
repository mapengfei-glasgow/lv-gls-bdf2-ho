"""Rollout of a trained surrogate on held-out cases: displacement error, cavity volume, EF.

  python -m surrogate.evaluate surrogate_runs/smoke [--split test] [--data DIR] [--vtk 1] [--ckpt FILE]

Writes RUN/eval_<split>/: metrics.csv (per case), volume.png (V(t) solver vs surrogate),
pv.png (p_endo-V loops) and, with --vtk K, ParaView frames (u_solver, u_surrogate, error)
for the first K cases. EF = (V(t=T/2) - min V over systole)/V(t=T/2) (end of the diastolic
plateau of Eq. 79 -> end systole).

Checkpoint choice: RUN/last.pt by default (the true final state; RUN/best.pt only when
last.pt is missing); pass --ckpt FILE to override. Metric note: rmse_mm is the per-node
vector-norm RMS sqrt(mean ||du||^2); the component-wise RMS used by the training logs and
by select_meshgen_model.py equals rmse_mm / sqrt(3) and is also written as rmse_comp_mm.
"""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import meshio
import numpy as np
import torch

from .data import EDGE_IN, NODE_IN, Graph, LVDataset, RunningNormalizer
from .meshgraphnet import MeshGraphNet
from .train import Surrogate, pick_device


def load(run, data, device, ckpt=None):
    path = Path(ckpt) if ckpt else next(p for p in (run / "last.pt", run / "best.pt") if p.exists())
    ck = torch.load(path, map_location="cpu", weights_only=False)
    ds = LVDataset(data or ck["data"])
    coeff_dim = int(ck["hparams"].get("coeff_dim", 0))
    if coeff_dim and ds.coeff_dim != coeff_dim:
        raise SystemExit(f"{ds.root}: checkpoint expects coeff_dim={coeff_dim}, "
                         f"dataset has {ds.coeff_dim} (shape_coeffs.csv)")
    node_dim = NODE_IN + coeff_dim
    model = MeshGraphNet(node_dim, EDGE_IN, 3, latent=ck["hparams"]["latent"], mlp_hidden=ck["hparams"]["latent"],
                         processor_steps=ck["hparams"]["mp_steps"]).to(device)
    model.load_state_dict(ck["model"])
    norms = []
    for dim, st in zip((node_dim, EDGE_IN, 3), ck["norms"]):
        n = RunningNormalizer(dim, device); n.load(st); norms.append(n)
    ck["ckpt_path"] = str(path)
    return Surrogate(model, Graph(ck["aug"], ds.tags, device, coeff_dim=coeff_dim), norms), ds, ck


def ef(t, V, period):
    ed = int(np.argmin(np.abs(t - 0.5 * period)))
    sys_ = (t >= 0.5 * period) & (t <= 0.8 * period)
    es = np.flatnonzero(sys_)[np.argmin(V[sys_])]
    return V[ed], V[es], 100 * (V[ed] - V[es]) / V[ed]


def write_vtk(d, ds, case, U, P, stride=0):
    d.mkdir(parents=True, exist_ok=True)
    step = max(1, stride) if stride > 0 else max(1, len(U) // 100)
    items = []
    for n in range(0, len(U), step):
        f = d / f"frame_{n:05d}.vtu"
        meshio.write(f, meshio.Mesh(case.coords, [("tetra", ds.cells)],
                                    point_data={"u_solver": U[n], "u_surrogate": P[n],
                                                "error_m": np.linalg.norm(P[n] - U[n], axis=1)}))
        items.append(f'<DataSet timestep="{case.time[n]:.8g}" file="{f.name}"/>')
    (d / "rollout.pvd").write_text('<?xml version="1.0"?>\n<VTKFile type="Collection" version="0.1">'
                                   "<Collection>" + "".join(items) + "</Collection></VTKFile>\n")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", type=Path)
    ap.add_argument("--data", type=Path, default=None, help="default: data set used for training")
    ap.add_argument("--ckpt", type=Path, default=None, help="checkpoint file (default: RUN/last.pt, else RUN/best.pt)")
    ap.add_argument("--split", default="test")
    ap.add_argument("--vtk", type=int, default=0, help="export ParaView frames for the first K cases")
    ap.add_argument("--vtk-stride", type=int, default=0, help="write every Nth frame for --vtk (0 = ~100 frames)")
    ap.add_argument("--device", default="auto")
    a = ap.parse_args(argv)
    sur, ds, ck = load(a.run, a.data, pick_device(a.device), a.ckpt)
    print(f"[evaluate] ckpt={ck.get('ckpt_path')}  it={ck.get('iter')}  stored val RMSE "
          f"{(ck.get('val_rollout_rmse') or float('nan')) * 1e3:.2f} mm (training comp-RMS)", flush=True)
    cases = ds.split(a.split)
    if not cases:
        raise SystemExit(f"no cases in split {a.split}")
    out = a.run / f"eval_{a.split}"
    out.mkdir(exist_ok=True)
    rows, curves = [], []
    for k, c in enumerate(cases):
        U, P = np.asarray(c.u, dtype=np.float64), sur.rollout(c).astype(np.float64)
        err = np.linalg.norm(P - U, axis=2)                                   # (T+1, N)
        Vt, Vp = ds.cavity_volume(c.coords, U), ds.cavity_volume(c.coords, P)
        et, ep = ef(c.time, Vt, c.params[-1]), ef(c.time, Vp, c.params[-1])
        rows.append(dict(case=c.name, geom=c.geom,
                         rmse_mm=1e3 * np.sqrt((err ** 2).mean()),
                         rmse_comp_mm=1e3 * np.sqrt((err ** 2).mean() / 3),
                         rel_l2=np.sqrt((err ** 2).sum() / (U ** 2).sum()),
                         max_err_mm=1e3 * err.max(), max_u_mm=1e3 * np.linalg.norm(U, axis=2).max(),
                         V0_mL=Vt[0], V_ED_mL=et[0], V_ED_pred_mL=ep[0], V_ES_mL=et[1], V_ES_pred_mL=ep[1],
                         EF=et[2], EF_pred=ep[2], V_max_abs_err_mL=np.abs(Vp - Vt).max()))
        curves.append((c, Vt, Vp))
        if k < a.vtk:
            write_vtk(out / f"vtk_{c.name}", ds, c, U, P, a.vtk_stride)
        print(" ".join(f"{k_}={v:.4g}" if isinstance(v, float) else f"{k_}={v}" for k_, v in rows[-1].items()),
              flush=True)
    with (out / "metrics.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    ncol = min(4, len(curves)); nrow = -(-len(curves) // ncol)
    for name, xy in (("volume", lambda c, V: (c.time / c.params[-1], V)), ("pv", lambda c, V: (V, c.loads[:, 0]))):
        fig, axes = plt.subplots(nrow, ncol, figsize=(3.2 * ncol, 2.6 * nrow), squeeze=False)
        for ax, (c, Vt, Vp) in zip(axes.flat, curves):
            ax.plot(*xy(c, Vt), "-", color="C0", label="solver")
            ax.plot(*xy(c, Vp), "--", color="C1", label="surrogate")
            ax.set_title(c.name, fontsize=9); ax.grid(alpha=0.3); ax.tick_params(labelsize=7)
            ax.set_xlabel("t / T" if name == "volume" else "V (mL)", fontsize=8)
            ax.set_ylabel("V (mL)" if name == "volume" else "p_endo (kPa)", fontsize=8)
        for ax in axes.flat[len(curves):]:
            ax.axis("off")
        axes.flat[0].legend(fontsize=7)
        fig.tight_layout(); fig.savefig(out / f"{name}.png", dpi=150); plt.close(fig)
    m = {k: np.mean([r[k] for r in rows]) for k in ("rmse_mm", "rmse_comp_mm", "rel_l2", "max_err_mm")}
    print(f"{a.split}: {len(rows)} cases, mean RMSE {m['rmse_mm']:.3f} mm (vector) = "
          f"{m['rmse_comp_mm']:.3f} mm (component), rel L2 {m['rel_l2']:.3f}, "
          f"max err {m['max_err_mm']:.2f} mm, |EF err| {np.mean([abs(r['EF'] - r['EF_pred']) for r in rows]):.2f} % "
          f"-> {out}")


if __name__ == "__main__":
    main()
