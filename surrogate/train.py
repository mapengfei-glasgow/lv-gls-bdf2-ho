"""Train the augmented-graph MeshGraphNet on solver runs (surrogate.generate).

  python -m surrogate.train surrogate_data --out surrogate_runs/smoke --iters 300 --latent 64 --mp-steps 6

Samples (case, n) uniformly over the training split; loss = one-step MSE of the normalised
Delta_n (surrogate/data.py) + pf_weight * pushforward MSE (the detached predicted u_{n+1} is
advanced once more and compared with u_{n+2}). Every --eval-every iterations: full
autoregressive rollout on the validation split (test split if there is none); best.pt keeps
the lowest rollout RMSE. Output: OUT/{best.pt, last.pt, history.csv, train.log}.
"""
import argparse
import csv
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .data import EDGE_IN, NODE_IN, Graph, LVDataset, RunningNormalizer
from .graph import build_augmented_graph
from .meshgraphnet import MeshGraphNet, init_decoder_zero

AUG_KW = dict(root_leaf_factor=4, min_root_nodes=6, n_nearest_nbrs=3)


def pick_device(name):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class Surrogate:
    """Model + normalisers + graph: one explicit step u_n, v_n -> u_{n+1}."""

    def __init__(self, model, graph, norms):
        self.model, self.g = model, graph
        self.xn, self.en, self.yn = norms

    def delta(self, cases, u, v, load, dt, update=False):
        x, e, s, r = self.g.inputs(cases, u, v, load, dt)
        if update:
            self.xn.update(x); self.en.update(e)
        out = self.model(self.xn.normalize(x), self.en.normalize(e), s, r)
        return out.view(len(cases), self.g.n_total, 3)[:, :self.g.n_real]      # normalised Delta

    @torch.no_grad()
    def rollout(self, case, steps=None):
        """Autoregressive prediction from rest; returns (T+1, N, 3) numpy (u_0 = 0)."""
        self.model.eval()
        dev, T = self.g.device, steps or case.steps
        u = torch.zeros(1, self.g.n_real, 3, device=dev)
        v = torch.zeros_like(u)
        dt = torch.tensor([case.dt], device=dev, dtype=torch.float32)
        out = [u[0].cpu().numpy()]
        for n in range(T):
            load = torch.as_tensor(case.loads[n + 1][None], device=dev)
            un = u + dt * v + self.yn.unnormalize(self.delta([case], u, v, load, dt))
            v, u = (un - u) / dt, un
            out.append(u[0].cpu().numpy())
        return np.stack(out)


def build(ds, train_cases, args, device):
    ref = np.mean([c.coords for c in train_cases], axis=0)
    aug = build_augmented_graph(ref, ds.edges, **AUG_KW)
    g = Graph(aug, ds.tags, device)
    model = MeshGraphNet(NODE_IN, EDGE_IN, 3, latent=args.latent, mlp_hidden=args.latent,
                         processor_steps=args.mp_steps).to(device)
    norms = (RunningNormalizer(NODE_IN, device), RunningNormalizer(EDGE_IN, device), RunningNormalizer(3, device))
    return Surrogate(model, g, norms), aug


def load_batch(cases, n, dev):
    """u_{n-1}, u_n, u_{n+1}, u_{n+2} (zero-padded), loads at n+1, n+2, dt."""
    U = {k: [] for k in (-1, 0, 1, 2)}
    L1, L2, dt = [], [], []
    for c, m in zip(cases, n):
        for k in U:
            j = m + k
            U[k].append(np.asarray(c.u[j]) if 0 <= j <= c.steps else np.zeros(c.u.shape[1:], np.float32))
        L1.append(c.loads[m + 1]); L2.append(c.loads[min(m + 2, c.steps)]); dt.append(c.dt)
    t = lambda a: torch.as_tensor(np.stack(a), dtype=torch.float32, device=dev)
    return {k: t(v) for k, v in U.items()}, t(L1), t(L2), torch.tensor(dt, dtype=torch.float32, device=dev)


def rollout_rmse(sur, cases):
    r = [float(np.sqrt(((sur.rollout(c) - np.asarray(c.u)) ** 2).mean())) for c in cases]
    return float(np.mean(r)), r


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--iters", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--latent", type=int, default=128)
    ap.add_argument("--mp-steps", type=int, default=10)
    ap.add_argument("--pf-weight", type=float, default=0.5, help="pushforward loss weight (0: off)")
    ap.add_argument("--noise", type=float, default=0.0, help="std (m) of noise added to u_n during training")
    ap.add_argument("--norm-iters", type=int, default=1000, help="iterations accumulating normaliser stats")
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--eval-cases", type=int, default=4)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    torch.manual_seed(args.seed); rng = np.random.default_rng(args.seed)
    dev = pick_device(args.device)
    args.out.mkdir(parents=True, exist_ok=True)
    logf = (args.out / "train.log").open("a")

    def log(msg):
        print(msg, flush=True); logf.write(msg + "\n"); logf.flush()

    ds = LVDataset(args.data)
    train = ds.split("train")
    val = ds.split("val") or ds.split("test")
    if not train:
        raise SystemExit("no training cases")
    if not ds.split("val"):
        log("note: no validation split, model selection uses the test split")
    val = val[:args.eval_cases]
    sur, aug = build(ds, train, args, dev)
    log(f"device={dev}  cases: train={len(train)} val/test={len(ds.split('val'))}/{len(ds.split('test'))}  "
        f"graph: {aug['n_real']} real + {aug['n_total'] - aug['n_real']} virtual nodes, {aug['n_edges']} edges  "
        f"params={sum(p.numel() for p in sur.model.parameters()) / 1e6:.2f}M")
    init_decoder_zero(sur.model)
    opt = torch.optim.Adam(sur.model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.iters, eta_min=1e-6)
    samples = [(i, n) for i, c in enumerate(train) for n in range(c.steps)]
    hist = (args.out / "history.csv").open("w", newline="")
    hw = csv.writer(hist); hw.writerow(["iter", "loss", "loss_1step", "loss_pf", "val_rollout_rmse_m", "seconds"])
    best, t0, run1, runpf = math.inf, time.time(), 0.0, 0.0

    def save(path, it, rmse):
        torch.save(dict(model=sur.model.state_dict(), iter=it, val_rollout_rmse=rmse,
                        norms=[n.state() for n in (sur.xn, sur.en, sur.yn)],
                        aug={k: aug[k] for k in ("aug_coords", "topology", "parents", "counts", "n_real",
                                                 "n_total", "n_edges")},
                        hparams=dict(latent=args.latent, mp_steps=args.mp_steps), data=str(ds.root),
                        args={k: str(v) for k, v in vars(args).items()}), path)

    for it in range(1, args.iters + 1):
        sur.model.train()
        if it == args.norm_iters + 1:
            for nrm in (sur.xn, sur.en, sur.yn):
                nrm.frozen = True
        pick = rng.integers(len(samples), size=args.batch)
        cases = [train[samples[k][0]] for k in pick]
        n = [samples[k][1] for k in pick]
        U, L1, L2, dt = load_batch(cases, n, dev)
        dtb = dt[:, None, None]
        un, up = U[0], U[-1]
        if args.noise > 0:
            un = un + args.noise * torch.randn_like(un)
        v = (un - up) / dtb
        y = U[1] - 2 * un + up
        sur.yn.update(y)
        d1 = sur.delta(cases, un, v, L1, dt, update=True)
        loss1 = ((d1 - sur.yn.normalize(y)) ** 2).mean()
        loss = loss1
        lpf = torch.zeros(())
        valid = torch.tensor([m + 2 <= c.steps for c, m in zip(cases, n)], device=dev)
        if args.pf_weight > 0 and valid.any():
            u1 = (un + dtb * v + sur.yn.unnormalize(d1)).detach()
            v1 = (u1 - un) / dtb
            d2 = sur.delta(cases, u1, v1, L2, dt)
            y2 = U[2] - 2 * u1 + un
            lpf = (((d2 - sur.yn.normalize(y2)) ** 2).mean(dim=(1, 2)) * valid).sum() / valid.sum()
            loss = loss + args.pf_weight * lpf
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(sur.model.parameters(), 1.0)
        opt.step(); sched.step()
        run1 += loss1.item(); runpf += lpf.item()
        rmse = ""
        if it % args.eval_every == 0 or it == args.iters:
            rmse, per = rollout_rmse(sur, val)
            if rmse < best:
                best = rmse
                save(args.out / "best.pt", it, rmse)
            log(f"[eval] it {it}: rollout RMSE {rmse * 1e3:.3f} mm "
                f"({', '.join(f'{c.name} {r * 1e3:.3f}' for c, r in zip(val, per))})"
                + ("  <- best" if rmse <= best else ""))
        if it % 10 == 0 or rmse != "":
            k = 10 if it % 10 == 0 else 1
            hw.writerow([it, (run1 + args.pf_weight * runpf) / k, run1 / k, runpf / k, rmse, round(time.time() - t0, 1)])
            hist.flush()
            if it % 50 == 0:
                log(f"it {it:6d}  loss {run1 / k:.4e} + pf {runpf / k:.4e}  lr {sched.get_last_lr()[0]:.2e}  "
                    f"{time.time() - t0:.0f} s")
            run1 = runpf = 0.0
    save(args.out / "last.pt", args.iters, best)
    (args.out / "summary.json").write_text(json.dumps(dict(best_val_rollout_rmse_m=best, iters=args.iters,
                                                           seconds=round(time.time() - t0, 1)), indent=2) + "\n")
    log(f"done: best val rollout RMSE {best * 1e3:.3f} mm -> {args.out / 'best.pt'}")


if __name__ == "__main__":
    main()
