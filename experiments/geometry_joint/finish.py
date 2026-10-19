"""Wait for generation, retry nonlinear failures, and validate; never train."""
from concurrent.futures import ThreadPoolExecutor
import csv
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from surrogate.generate import done
DATA=ROOT/'surrogate_data.geometry_joint'
control=DATA/'_campaign'
workers=json.loads((ROOT/'experiments/geometry_joint/machines.json').read_text())['workers']
while not all(any((control/f'{w["host"]}.{state}').exists() for state in ['done','failed']) for w in workers):
    if (control/"prepare.failed").exists():raise SystemExit("Geometry preparation failed")
    time.sleep(60)
with (DATA/'params.csv').open() as f:rows=list(csv.DictReader(f))
failed=[r['case'] for r in rows if not done(DATA/r['case'])]

def retry(case):
    # Only recover solver failures. Missing files or fibre-check failures need inspection.
    d=DATA/case
    if not (d/'metadata.json').exists():return dict(case=case,returncode=None,reason='missing solver metadata')
    with (control/f'retry_{case}.log').open('w') as log:
        rc=subprocess.run([sys.executable,str(ROOT/'experiments/geometry_narrow/retry.py'),
            '--data',str(DATA),'--case',case,'--strategy','robust'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT).returncode
    return dict(case=case,returncode=rc)

with ThreadPoolExecutor(max_workers=2) as pool:
    retries=list(pool.map(retry,failed))
(control/'retry_summary.json').write_text(json.dumps(retries,indent=2)+'\n')
remaining=[r['case'] for r in rows if not done(DATA/r['case'])]
if remaining:
    (control/'validation.failed').write_text(json.dumps(remaining,indent=2)+'\n')
    raise SystemExit(f'{len(remaining)} cases still incomplete; inspect logs')
subprocess.run([sys.executable,str(ROOT/'experiments/geometry_narrow/validate.py'),str(DATA)],cwd=ROOT,check=True)
(control/'validated.done').touch()
print('All jointly sampled geometry/parameter cases validated; no training started',flush=True)
