"""Plot loss/history and final split errors; outputs are generated report assets."""
import argparse
import csv
import json
from pathlib import Path
import re
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from surrogate.params import SPACE


def history(log):
    pattern=r'^it\s+(\d+)\s+loss\s+([\d.eE+-]+)\s+rollout RMSE\s+([\d.eE+-]+) mm.*?\(([\d.]+)s\)'
    rows=[];offset=0;previous=0
    for i,l,r,t in re.findall(pattern,log,re.M):
        t=float(t)
        if t<previous:offset+=previous
        rows.append(dict(iteration=int(i),training_loss=float(l),validation_rollout_comp_rmse_mm=float(r),
                         cumulative_active_hours=(offset+t)/3600))
        previous=t
    return rows


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('report',type=Path)
    a=ap.parse_args(); root=a.report
    assets=root/'assets';assets.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                         'axes.grid':True,'grid.alpha':.18,'savefig.dpi':180})
    rows=history((root/'training.log').read_text())
    with (assets/'training_history.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    hours=np.array([r['cumulative_active_hours'] for r in rows])
    losses=np.array([r['training_loss'] for r in rows])
    rollout=np.array([r['validation_rollout_comp_rmse_mm'] for r in rows])
    for filename,values,color,title,ylabel in [
        ('training_loss.png',losses,'#16776f','Training loss over the mean-geometry run','Normalised one-step MSE'),
        ('rollout_history.png',rollout,'#7656a2','Validation rollout error during training','Component RMSE (mm), 10 fixed cases')]:
        fig,ax=plt.subplots(figsize=(8.8,4.3),layout='constrained')
        ax.semilogy(hours,values,color=color,lw=1.8)
        ax.set(xlabel='Cumulative active training/evaluation time (hours)',ylabel=ylabel,title=title)
        ax.scatter(hours[-1],values[-1],color=color,zorder=3)
        ax.annotate(f'{values[-1]:.4f}' if filename.startswith('training') else f'{values[-1]:.2f} mm',
                    (hours[-1],values[-1]),xytext=(-55,18),textcoords='offset points',color=color)
        fig.savefig(assets/filename);plt.close(fig)
    summary={}
    if all((root/s/'summary.json').exists() for s in ['train','val','test']):
        fig,ax=plt.subplots(figsize=(8.8,4.6),layout='constrained')
        for split,label,color in [('train','Training subset','#16776f'),('val','Validation','#417db9'),('test','Test','#bd7138')]:
            summary[split]=json.loads((root/split/'summary.json').read_text())
            with (root/split/'error_vs_time.csv').open() as f:data=list(csv.DictReader(f))
            case_names=list(dict.fromkeys(r['case'] for r in data))
            curves=np.array([[float(r['rmse_comp_mm']) for r in data if r['case']==name] for name in case_names])
            phase=np.linspace(0,1,curves.shape[1])
            mean=curves.mean(0)
            ax.plot(phase,mean,label=f'{label} (n={len(case_names)})',color=color,lw=2)
            low,high=np.percentile(curves,[10,90],axis=0)
            ax.fill_between(phase,low,high,color=color,alpha=.10,lw=0)
        ax.set(xlabel='Normalised cardiac time t / T',ylabel='Mean per-case component RMSE (mm)',
               title='Final-checkpoint rollout error through the cardiac cycle',xlim=(0,1))
        ax.legend(frameon=False)
        fig.savefig(assets/'split_errors_vs_time.png');plt.close(fig)
    representative=root/'illustrative_sample.npz'
    if representative.exists():
        with np.load(representative) as z:
            U=z['u'].astype('float64');P=z['prediction'].astype('float64');t=z['time'];case=str(z['case'])
        with (root/'test/error_vs_time.csv').open() as f:
            data=[r for r in csv.DictReader(f) if r['case']==case]
        fig,axes=plt.subplots(1,2,figsize=(10.2,4),layout='constrained')
        axes[0].plot(t,np.sqrt(((P-U)**2).mean(axis=(1,2)))*1000,color='#a55a36',lw=2)
        axes[0].set(xlabel='Time (s)',ylabel='Component RMSE (mm)',title='Representative test-case rollout error')
        for key,label,color,style in [('volume_solver_ml','Solver','#16776f','-'),('volume_pred_ml','Emulator','#7656a2','--')]:
            axes[1].plot(t,[float(r[key]) for r in data],style,color=color,lw=2,label=label)
        axes[1].set(xlabel='Time (s)',ylabel='Cavity volume (mL)',title=case)
        axes[1].legend(frameon=False)
        fig.savefig(assets/'representative_error_volume.png');plt.close(fig)
    # Architecture is exported directly from the final checkpoint evaluation.
    config=dict(parameter_space={k:dict(nominal=v[0],low=v[1],high=v[2],distribution=v[3]) for k,v in SPACE.items()},
                final_history=rows[-1],best_monitored_rollout=min(rows,key=lambda r:r['validation_rollout_comp_rmse_mm']),
                splits=summary,
                time_definition='Sum of active elapsed times from log segments; excludes downtime between resumes',
                error_definition='Component RMSE: sqrt(mean((u_pred-u_solver)^2)) over xyz, nodes and time',
                test_evaluation='Final checkpoint only; no test history used for model selection')
    (assets/'report_summary.json').write_text(json.dumps(config,indent=2)+'\n')
    print(json.dumps(config,indent=2))


if __name__=='__main__':
    main()
