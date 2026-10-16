"""Train the fixed-geometry base model from scratch, or resume a complete checkpoint.

Uses the existing reference recipe: position noise with corrected targets, one-step
loss, and running normalisers. Validation is required; test cases are never used
for checkpoint selection. Checkpoints retain raw and optional EMA weights, training
state, normalisers, graph, and RNG states for continuation.
"""
import argparse
import hashlib
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from surrogate.data import EDGE_IN, NODE_IN, Graph, LVDataset, RunningNormalizer
from surrogate.graph import build_augmented_graph
from surrogate.meshgraphnet import MeshGraphNet, init_decoder_zero

NOISE_GAMMA = 0.1
AUG_KW = dict(root_leaf_factor=4, min_root_nodes=6, n_nearest_nbrs=3)


def add_completed_cases(ds, train, validate_mean=False):
    """Append completed training cases; retain the existing sample order and splits."""
    fresh = LVDataset(ds.root)
    np.testing.assert_array_equal(fresh.cells, ds.cells)
    np.testing.assert_array_equal(fresh.node_tags, ds.node_tags)
    known = {c.name for c in train}
    added = [c for c in fresh.split("train") if c.name not in known]
    if added and validate_mean:
        from experiments.mean_base.validate import validate
        validate(ds.root, names=[c.name for c in added])
    for c in added:
        np.testing.assert_allclose(c.coords, train[0].coords, atol=1e-12, rtol=0)
        if not np.isfinite(c.u).all():
            raise ValueError(f"{c.name}: nonfinite displacements")
    return train + added, added


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--iters", type=int, default=12000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--latent", type=int, default=128)
    ap.add_argument("--mp-steps", type=int, default=10)
    ap.add_argument("--noise-pos", type=float, default=5e-5)
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--eval-cases", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--refresh-every", type=int, default=0,
                    help="admit newly completed training cases every N iterations (0: fixed dataset)")
    ap.add_argument("--validate-mean", action="store_true", help="validate admitted simulations against the mean campaign design")
    ap.add_argument("--resume", type=Path, default=None, help="resume this training run, including optimiser and normalisers")
    ap.add_argument("--ema", action="store_true", help="track + evaluate/save EMA weights")
    ap.add_argument("--ema-decay", type=float, default=0.999)
    ap.add_argument("--accum", type=int, default=1,
                    help="gradient-accumulation micro-steps per optimizer step (effective batch = batch*accum)")
    a = ap.parse_args()
    if min(a.iters, a.batch, a.eval_every, a.eval_cases, a.accum) < 1:
        ap.error("iteration, batch, evaluation, and accumulation counts must be positive")
    if a.eval_every % a.accum or a.iters % a.accum:
        ap.error("eval-every and iters must be divisible by accum")
    if a.refresh_every < 0:
        ap.error("refresh-every must be nonnegative")
    if a.out.exists() and any(a.out.iterdir()) and not a.resume:
        ap.error("output directory is not empty; use --resume or a new --out")
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = LVDataset(a.data)
    train = ds.split("train")
    val = ds.split("val")[:a.eval_cases]
    if not train or not val:
        raise SystemExit("both train and validation cases are required")
    resume = torch.load(a.resume, map_location="cpu", weights_only=False) if a.resume else None
    if resume:
        for key in ("iters", "batch", "lr", "latent", "mp_steps", "noise_pos", "accum", "ema", "ema_decay", "seed", "eval_cases", "eval_every", "refresh_every", "validate_mean"):
            if resume["args"].get(key, ap.get_default(key)) != getattr(a, key):
                raise SystemExit(f"resume requires the same --{key.replace('_', '-')}")
        digest = hashlib.sha256((ds.root / "params.csv").read_bytes()).hexdigest()
        if resume.get("params_sha256") != digest or Path(resume["data"]).resolve() != ds.root:
            raise SystemExit("resume requires the original dataset and parameter manifest")
        by_name = {c.name: c for c in ds.cases}
        val = [by_name[name] for name in resume.get("val_cases", [c.name for c in val])]
        train = [by_name[name] for name in resume.get("train_cases", [c.name for c in train])]
    if a.validate_mean:
        from experiments.mean_base.validate import validate
        validate(ds.root, names=[c.name for c in train + val])
    node_dim = NODE_IN
    ref = np.mean([c.coords for c in train], axis=0)
    aug = resume["aug"] if resume else build_augmented_graph(ref, ds.edges, **AUG_KW)
    g = Graph(aug, ds.tags, dev)
    model = MeshGraphNet(node_dim, EDGE_IN, 3, latent=a.latent, mlp_hidden=a.latent,
                         processor_steps=a.mp_steps).to(dev)
    init_decoder_zero(model)
    if resume:
        model.load_state_dict(resume["raw_model"])
    print(f"device={dev}  train={len(train)} val/test={len(val)}  graph: {aug['n_real']} real + "
          f"{aug['n_total'] - aug['n_real']} virtual nodes, {aug['n_edges']} edges, "
          f"params={sum(p.numel() for p in model.parameters()) / 1e6:.2f}M  "
          f"batch={a.batch} accum={a.accum} noise_pos={a.noise_pos} gamma={NOISE_GAMMA} "
          f"geometry=PCA-mean", flush=True)
    a.out.mkdir(parents=True, exist_ok=True)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    n_opt = max(1, a.iters // a.accum)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_opt, eta_min=1e-6)
    ema = None
    if a.ema:
        ema = {k: v.detach().clone() for k, v in model.state_dict().items() if v.is_floating_point()}
        print(f"EMA enabled (decay {a.ema_decay})", flush=True)
    xn, en, yn = (RunningNormalizer(node_dim, dev), RunningNormalizer(EDGE_IN, dev),
                  RunningNormalizer(3, dev))
    if resume:
        opt.load_state_dict(resume["optimizer"])
        sched.load_state_dict(resume["scheduler"])
        for norm, state in zip((xn, en, yn), resume["norms"]):
            norm.load(state)
            norm.frozen = False
        if ema is not None:
            ema = {k: v.to(dev) for k, v in resume["ema"].items()}
    samples = [(i, n) for i, c in enumerate(train) for n in range(c.steps)]

    def load(cases, ns):
        U = {k: [] for k in (-1, 0, 1)}
        L1, dt = [], []
        for c, m in zip(cases, ns):
            for k in U:
                j = m + k
                U[k].append(np.asarray(c.u[j]) if 0 <= j <= c.steps
                            else np.zeros(c.u.shape[1:], np.float32))
            L1.append(c.loads[m + 1])
            dt.append(c.dt)
        t = lambda x: torch.as_tensor(np.stack(x), dtype=torch.float32, device=dev)
        return ({k: t(v) for k, v in U.items()}, t(L1),
                torch.tensor(dt, dtype=torch.float32, device=dev))

    def delta(cases, u, v, load_, dt, update=False):
        x, e, s, r = g.inputs(cases, u, v, load_, dt)
        if update:
            xn.update(x)
            en.update(e)
        return model(xn.normalize(x), en.normalize(e), s, r).view(len(cases), g.n_total, 3)[:, :g.n_real]

    @torch.no_grad()
    def rollout(case):
        model.eval()
        u = torch.zeros(1, g.n_real, 3, device=dev)
        v = torch.zeros_like(u)
        dt = torch.tensor([case.dt], device=dev, dtype=torch.float32)
        out = [u[0].cpu().numpy()]
        for n in range(case.steps):
            load_ = torch.as_tensor(case.loads[n + 1][None], device=dev)
            un = u + dt * v + yn.unnormalize(delta([case], u, v, load_, dt))
            v, u = (un - u) / dt, un
            out.append(u[0].cpu().numpy())
        return np.stack(out)

    upd_pairs = []
    for i in range(30):
        c = val[i % len(val)]
        upd_pairs.append((c, int(rng.integers(1, c.steps))))

    @torch.no_grad()
    def upd_um():
        """Clean-input one-step update RMS in um (absolute; independent of normaliser scale)."""
        model.eval()
        tot, n = 0.0, 0
        for c, m in upd_pairs:
            u0 = torch.tensor(np.asarray(c.u[m][None]), dtype=torch.float32, device=dev)
            um1 = torch.tensor(np.asarray(c.u[m - 1][None]), dtype=torch.float32, device=dev)
            up1 = torch.tensor(np.asarray(c.u[m + 1][None]), dtype=torch.float32, device=dev)
            dtb1 = torch.tensor([c.dt], dtype=torch.float32, device=dev)
            load_ = torch.as_tensor(c.loads[m + 1][None], dtype=torch.float32, device=dev)
            d1 = delta([c], u0, (u0 - um1) / dtb1, load_, dtb1)
            e = yn.unnormalize(d1) - (up1 - 2 * u0 + um1)
            tot += (e ** 2).sum().item()
            n += e.numel()
        return 1e6 * float(np.sqrt(tot / n))

    best, t0, run_loss = resume["best"] if resume else float("inf"), time.time(), 0.0
    if resume:
        rng.bit_generator.state = resume["numpy_rng"]
        torch.set_rng_state(resume["torch_rng"])
        if torch.cuda.is_available() and resume["cuda_rng"] is not None:
            torch.cuda.set_rng_state_all(resume["cuda_rng"])
    start = resume["iter"] + 1 if resume else 1
    window = 0
    run_kin = 0.0
    for it in range(start, a.iters + 1):
        if a.refresh_every and (it == start or (it - 1) % a.refresh_every == 0):
            train, added = add_completed_cases(ds, train, a.validate_mean)
            if added:
                samples = [(i, n) for i, c in enumerate(train) for n in range(c.steps)]
                print(f"[data] it {it}: added {len(added)} training cases; total={len(train)}; "
                      f"validation remains {len(val)} fixed cases", flush=True)
        window += 1
        model.train()
        pick = rng.integers(len(samples), size=a.batch)
        cases = [train[samples[k][0]] for k in pick]
        ns = [samples[k][1] for k in pick]
        U, L1, dt = load(cases, ns)
        dtb = dt[:, None, None]
        un, up = U[0], U[-1]
        y_true = U[1] - 2 * un + up
        if a.noise_pos > 0:
            eps_p = a.noise_pos * torch.randn_like(up)
            eps_n = a.noise_pos * torch.randn_like(un)
            x_prev, x_n = up + eps_p, un + eps_n
            v = (x_n - x_prev) / dtb
            y = y_true + ((2 * un - up) - (2 * x_n - x_prev)) + (1 - NOISE_GAMMA) * eps_n
            u_in = x_n
        else:
            v = (un - up) / dtb
            y = y_true
            u_in = un
        yn.update(y)
        d1 = delta(cases, u_in, v, L1, dt, update=True)
        kin = ((d1 - yn.normalize(y)) ** 2).mean()
        loss = kin
        run_kin += kin.item()
        (loss / a.accum).backward()
        if it % a.accum == 0 or it == a.iters:
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            opt.zero_grad()
            sched.step()
            if ema is not None:
                with torch.no_grad():
                    for k, v in model.state_dict().items():
                        if k in ema:
                            ema[k].mul_(a.ema_decay).add_(v, alpha=1.0 - a.ema_decay)
        run_loss += loss.item()
        if it % 100 == 0:
            print(f"[train] it {it} loss={run_loss / window:.4f} train_cases={len(train)}", flush=True)
        if it % a.eval_every == 0 or it == a.iters:
            raw_sd = None
            if ema is not None:
                raw_sd = {k: v.detach().clone() for k, v in model.state_dict().items()}
                sd = dict(raw_sd)
                sd.update({k: ema[k].to(sd[k].dtype) for k in ema})
                model.load_state_dict(sd)
            rm = [float(np.sqrt(((rollout(c) - np.asarray(c.u)) ** 2).mean())) for c in val]
            rmse = float(np.mean(rm))
            upd = upd_um()
            print(f"it {it:6d}  loss {run_loss / window:.4f}  rollout RMSE {rmse * 1e3:.2f} mm"
                  f"  lr {sched.get_last_lr()[0]:.2e}  ({time.time() - t0:.0f}s)"
                  f"  kin {run_kin / window:.4f}"
                  f"  upd {upd:.2f} um", flush=True)
            run_loss = run_kin = 0.0
            window = 0
            improved = rmse < best
            best = min(best, rmse)
            ck = dict(model=model.state_dict(), iter=it, val_rollout_rmse=rmse,
                      raw_model=raw_sd if raw_sd is not None else model.state_dict(),
                      optimizer=opt.state_dict(), scheduler=sched.state_dict(), ema=ema,
                      best=best, numpy_rng=rng.bit_generator.state,
                      torch_rng=torch.get_rng_state(),
                      cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                      params_sha256=hashlib.sha256((ds.root / "params.csv").read_bytes()).hexdigest(),
                      train_cases=[c.name for c in train], val_cases=[c.name for c in val],
                      norms=[n.state() for n in (xn, en, yn)],
                      aug={k: aug[k] for k in ("aug_coords", "topology", "parents", "counts",
                                               "n_real", "n_total", "n_edges")},
                      hparams=dict(latent=a.latent, mp_steps=a.mp_steps, coeff_dim=0),
                      data=str(ds.root), args=vars(a))
            torch.save(ck, a.out / "last.pt.tmp")
            (a.out / "last.pt.tmp").replace(a.out / "last.pt")
            if improved:
                torch.save(ck, a.out / "best.pt.tmp")
                (a.out / "best.pt.tmp").replace(a.out / "best.pt")
            if raw_sd is not None:
                model.load_state_dict(raw_sd)   # restore raw weights, keep training
    print("done, best rollout RMSE", best)


if __name__ == "__main__":
    main()
