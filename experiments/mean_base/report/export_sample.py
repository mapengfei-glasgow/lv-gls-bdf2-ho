"""Export a predeclared illustrative held-out case, without selecting by its error."""
import json
from pathlib import Path
import sys
import resource
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from surrogate.evaluate import load
from surrogate.train import pick_device

if __name__=='__main__':
    soft,hard=resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE,(min(hard,8192),hard))
    run=ROOT/'surrogate_runs/mean_base'
    sur,ds,ck=load(run,ROOT/'surrogate_data.mean_base',pick_device('cuda'),run/'last.pt',split='test')
    c=ds.split('test')[0]
    print('Illustrative sample selected before error evaluation:',c.name,flush=True)
    prediction=sur.rollout(c)
    out=run/'report';out.mkdir(exist_ok=True)
    np.savez_compressed(out/'illustrative_sample.npz',case=c.name,coords=c.coords,cells=ds.cells,
                        node_tags=ds.node_tags,time=c.time,loads=c.loads,u=np.asarray(c.u),prediction=prediction)
    (out/'illustrative_sample.json').write_text(json.dumps(dict(case=c.name,
        selection='first test case in the fixed manifest, chosen without inspecting prediction errors',
        checkpoint='last.pt',iteration=ck['iter']),indent=2)+'\n')
    print('Illustrative sample exported',flush=True)
