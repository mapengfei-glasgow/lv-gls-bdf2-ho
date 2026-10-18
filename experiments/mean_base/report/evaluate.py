"""Batched final-checkpoint evaluation with per-phase errors for report figures."""
import argparse
import csv
import json
from pathlib import Path
import resource
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from surrogate.evaluate import load, ef
from surrogate.train import pick_device


@torch.no_grad()
def rollout_batch(sur, cases):
    sur.model.eval()
    if len({c.steps for c in cases}) != 1:
        raise ValueError('batch requires equal step counts')
    device = sur.g.device
    u = torch.zeros(len(cases), sur.g.n_real, 3, device=device)
    v = torch.zeros_like(u)
    dt = torch.tensor([c.dt for c in cases], dtype=torch.float32, device=device)
    states = [u.cpu().numpy()]
    for n in range(cases[0].steps):
        loads = torch.as_tensor(np.stack([c.loads[n+1] for c in cases]), device=device)
        un = u + dt[:, None, None] * v + sur.yn.unnormalize(sur.delta(cases, u, v, loads, dt))
        v, u = (un-u) / dt[:, None, None], un
        states.append(u.cpu().numpy())
    return np.stack(states, axis=1)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run',type=Path,required=True)
    ap.add_argument('--data',type=Path,required=True)
    ap.add_argument('--split',choices=['train','val','test'],required=True)
    ap.add_argument('--batch',type=int,default=8)
    ap.add_argument('--limit',type=int,default=0)
    ap.add_argument('--seed',type=int,default=20261018)
    ap.add_argument('--checkpoint',default='last.pt')
    ap.add_argument('--start',type=int,default=0)
    ap.add_argument('--stop',type=int,default=0)
    ap.add_argument('--tag',default='')
    a=ap.parse_args()
    soft,hard=resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE,(min(hard,8192),hard))
    sur,ds,ck=load(a.run,a.data,pick_device('cuda'),a.run/a.checkpoint,split=a.split)
    cases=ds.split(a.split)
    if a.limit and a.limit<len(cases):
        selection=sorted(np.random.default_rng(a.seed).choice(len(cases),a.limit,replace=False))
        cases=[cases[i] for i in selection]
    cases=cases[a.start:a.stop or None]
    out=a.run/'report'/a.split
    if a.tag:
        out=out/a.tag
    out.mkdir(parents=True,exist_ok=True)
    rows=[]; time_rows=[]; start=time.time()
    for offset in range(0,len(cases),a.batch):
        chunk=cases[offset:offset+a.batch]
        predictions=rollout_batch(sur,chunk)
        for c,p in zip(chunk,predictions):
            target=np.asarray(c.u,dtype=np.float64)
            sq=(p.astype(np.float64)-target)**2
            phase_rmse=np.sqrt(sq.mean(axis=(1,2)))*1e3
            Vt=ds.cavity_volume(c.coords,target)
            Vp=ds.cavity_volume(c.coords,p)
            et,ep=ef(c.time,Vt,c.params[-1]),ef(c.time,Vp,c.params[-1])
            rows.append(dict(case=c.name,geom=c.geom,split=c.split,
                             rmse_comp_mm=float(np.sqrt(sq.mean())*1e3),
                             rmse_vector_mm=float(np.sqrt(sq.mean()*3)*1e3),
                             relative_l2=float(np.sqrt(sq.sum()/np.square(target).sum())),
                             max_error_mm=float(np.sqrt(sq.sum(axis=2)).max()*1e3),
                             EF_solver=float(et[2]),EF_pred=float(ep[2]),
                             EF_abs_error_pp=float(abs(ep[2]-et[2])),
                             max_volume_error_ml=float(np.abs(Vp-Vt).max())))
            for n in range(c.steps+1):
                time_rows.append(dict(case=c.name,split=c.split,time_s=float(c.time[n]),
                                      phase=float(c.time[n]/c.params[-1]),
                                      rmse_comp_mm=float(phase_rmse[n]),
                                      volume_solver_ml=float(Vt[n]),volume_pred_ml=float(Vp[n]),
                                      p_endo_kpa=float(c.loads[n,0])))
        for name,values in [('metrics.csv',rows),('error_vs_time.csv',time_rows)]:
            with (out/name).open('w',newline='') as f:
                w=csv.DictWriter(f,fieldnames=list(values[0]));w.writeheader();w.writerows(values)
        print(f'{a.split}: {len(rows)}/{len(cases)} cases, mean component RMSE '
              f'{np.mean([r["rmse_comp_mm"] for r in rows]):.4f} mm, {time.time()-start:.1f}s',flush=True)
    summary=dict(split=a.split,cases=len(rows),checkpoint=a.checkpoint,checkpoint_iter=ck['iter'],
                 selection_seed=a.seed,requested_limit=a.limit,
                 mean_component_rmse_mm=float(np.mean([r['rmse_comp_mm'] for r in rows])),
                 median_component_rmse_mm=float(np.median([r['rmse_comp_mm'] for r in rows])),
                 mean_vector_rmse_mm=float(np.mean([r['rmse_vector_mm'] for r in rows])),
                 mean_relative_l2=float(np.mean([r['relative_l2'] for r in rows])),
                 mean_EF_abs_error_pp=float(np.mean([r['EF_abs_error_pp'] for r in rows])),
                 seconds=time.time()-start,
                 model_parameters=sum(p.numel() for p in sur.model.parameters()),
                 graph={k:sur.g.aug[k] for k in ['n_real','n_total','n_edges']},
                 hparams=ck['hparams'])
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    if a.split=='test' and not a.tag:
        ordered=sorted(rows,key=lambda r:r['rmse_comp_mm'])
        selected=ordered[len(ordered)//2]['case']
        c=next(c for c in cases if c.name==selected)
        p=sur.rollout(c)
        np.savez_compressed(out/'representative.npz',case=c.name,coords=c.coords,cells=ds.cells,
                            node_tags=ds.node_tags,time=c.time,loads=c.loads,
                            u=np.asarray(c.u),prediction=p)
        (out/'representative.json').write_text(json.dumps(dict(case=c.name,
            selection='upper median of test cases sorted by component rollout RMSE',
            metrics=next(r for r in rows if r['case']==selected)),indent=2)+'\n')
    print(json.dumps(summary),flush=True)


if __name__=='__main__':
    main()
