"""Merge disjoint public-GPU report parts after complete split coverage is available."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np


def read(path):
    with path.open() as f:return list(csv.DictReader(f))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('report',type=Path)
    ap.add_argument('--manifest',type=Path,required=True)
    a=ap.parse_args()
    expected={s:{r['case'] for r in read(a.manifest) if r['split']==s} for s in ('val','test')}
    for split in ['val','test']:
        root=a.report/split
        public=root/'public_gpu'
        if not (public/'summary.json').exists():raise SystemExit(f'{split}: public GPU part still running')
        files=[root/'metrics.csv',public/'metrics.csv']
        metrics={}
        for f in files:
            for row in read(f):
                if row['case'] in metrics:
                    delta=abs(float(row['rmse_comp_mm'])-float(metrics[row['case']]['rmse_comp_mm']))
                    if delta>.005:raise ValueError(f'overlapping evaluation differs by {delta} mm')
                else:metrics[row['case']]=row
        if set(metrics)!=expected[split]:raise SystemExit(f'{split}: {len(metrics)}/{len(expected[split])} covered')
        phases={}
        for f in [root/'error_vs_time.csv',public/'error_vs_time.csv']:
            for row in read(f):phases.setdefault((row['case'],row['time_s']),row)
        from collections import Counter
        phase_counts=Counter(r['case'] for r in phases.values())
        if set(phase_counts)!=expected[split] or any(n!=201 for n in phase_counts.values()):
            raise ValueError(f'{split}: incomplete per-cycle error data')
        archive=root/'original_gpu_part';archive.mkdir(exist_ok=True)
        for name in ['metrics.csv','error_vs_time.csv']:
            shutil.copy2(root/name,archive/name)
        for name,rows in [('metrics.csv',[metrics[k] for k in sorted(metrics)]),
                          ('error_vs_time.csv',sorted(phases.values(),key=lambda r:(r['case'],float(r['time_s']))) )]:
            with (root/name).open('w',newline='') as f:
                w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
        summary=json.loads((public/'summary.json').read_text())
        values=list(metrics.values())
        summary.update(cases=len(values),requested_limit=0,
            checkpoint_sha256=hashlib.sha256((a.report.parent/'last.pt').read_bytes()).hexdigest(),
            mean_component_rmse_mm=float(np.mean([float(r['rmse_comp_mm']) for r in values])),
            median_component_rmse_mm=float(np.median([float(r['rmse_comp_mm']) for r in values])),
            mean_vector_rmse_mm=float(np.mean([float(r['rmse_vector_mm']) for r in values])),
            mean_relative_l2=float(np.mean([float(r['relative_l2']) for r in values])),
            mean_EF_abs_error_pp=float(np.mean([float(r['EF_abs_error_pp']) for r in values])),
            provenance='Merged float32 evaluations of final last.pt from euclid-17 and euclid-10',
            time_note='Evaluation was distributed; per-part seconds are not a total wall-clock duration')
        (root/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
        print(split,len(values),summary['mean_component_rmse_mm'],flush=True)


if __name__=='__main__':main()
