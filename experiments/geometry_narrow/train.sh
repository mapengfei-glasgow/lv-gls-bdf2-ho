#!/usr/bin/env bash
# New curriculum stage. Source checkpoints are read only; no single-geometry outputs are written.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
cd "$ROOT"
: "${TRAIN_PY:?Set TRAIN_PY to the CUDA training Python}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/lv-narrow-mpl}"
ulimit -n 8192
DATA=surrogate_data.geometry_narrow
RUN=surrogate_runs/geometry_narrow_from_mean_loss
[ -f "$DATA/_campaign/validated.done" ] || { echo 'Full narrow-data validation required'; exit 1; }
[ ! -e "$RUN" ] || { echo "Refusing to overwrite existing run: $RUN"; exit 1; }
SOURCE=$("$TRAIN_PY" - <<'PY'
import hashlib,json,re
from collections import Counter
from pathlib import Path
import numpy as np
import torch
root=Path('surrogate_runs/mean_base')
log=Path('surrogate_data.mean_base/_campaign/training.log').read_text()
losses={int(i):float(v) for i,v in re.findall(r'^it\s+(\d+)\s+loss\s+([\d.eE+-]+)',log,re.M)}
candidates=[]
for p in root.glob('*.pt'):
 ck=torch.load(p,map_location='cpu',weights_only=False)
 loss=ck.get('training_loss',losses.get(ck['iter']))
 if loss is not None:candidates.append((loss,ck['iter'],str(p),ck))
assert candidates,'No checkpoint with recorded training loss'
loss,it,path,ck=min(candidates,key=lambda x:(x[0],-x[1]))
import csv
rows=list(csv.DictReader(Path('surrogate_data.geometry_narrow/params.csv').open()))
assert len(rows)==600 and Counter(r['split'] for r in rows)=={'train':384,'val':102,'test':114}
base=Path(ck['data'])/ck['train_cases'][0]/'fields'
for r in rows:
 d=Path('surrogate_data.geometry_narrow')/r['case']
 assert json.loads((d/'metadata.json').read_text())['status']=='completed'
 for key in ['cells','node_tags']:
  np.testing.assert_array_equal(np.load(d/'fields'/f'{key}.npy'),np.load(base/f'{key}.npy'))
assert torch.cuda.is_available(),'CUDA required'
(torch.ones(1,device='cuda')+1).cpu()
provenance=dict(source_checkpoint=path,source_iteration=it,source_logged_training_loss=loss,
 sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),
 selection='Minimum recorded window-averaged training loss among available saved base checkpoints',
 data_counts=dict(Counter(r['split'] for r in rows)),geometry_training='Reference and deformed edge features; no added PCA input columns',
 optimiser='New Adam, cosine 1e-4 to 1e-6; pretrained weights, graph and running normalisers retained')
Path('surrogate_data.geometry_narrow/_campaign/training_initialization.json').write_text(json.dumps(provenance,indent=2)+'\n')
print(path)
PY
)
cat "$DATA/_campaign/training_initialization.json"
exec "$TRAIN_PY" -m surrogate.train_base "$DATA" --out "$RUN" \
    --init-from "$SOURCE" --iters 120000 --batch 4 --lr 1e-4 \
    --noise-pos 5e-5 --eval-every 1000 --eval-cases 10 --seed 0
