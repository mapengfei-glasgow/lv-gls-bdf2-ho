"""Strict fibre/mesh checks on every newly sampled shape before simulation."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
DATA=ROOT/'surrogate_data.geometry_wide'
control=DATA/'_campaign'
control.mkdir(parents=True,exist_ok=True)
design=json.loads((DATA/'design.json').read_text())

def check(geom):
    with (control/f'check_{geom}.log').open('w') as log:
        rc=subprocess.run([sys.executable,str(ROOT/'tools/check_mesh.py'),
            str(DATA/'_geometry'/f'dalton_geom_{geom}'),'--strict'],stdout=log,stderr=subprocess.STDOUT,cwd=ROOT).returncode
    print(geom,rc,flush=True)
    return rc

with ThreadPoolExecutor(max_workers=6) as pool:
    codes=list(pool.map(check,design['geometry_ids'][30:]))
if any(codes):raise SystemExit('New mesh failed strict checks; no simulations launched')
(control/'preflight.done').touch()
print('All 30 new shapes passed strict checks',flush=True)
