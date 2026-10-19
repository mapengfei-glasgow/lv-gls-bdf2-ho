"""Joint Sobol sampling of geometry coefficients and physical parameters; data only."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from scipy.stats import qmc

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from geometry.generate import generate, reconstruct
from surrogate.params import NAMES, SPACE, from_unit


def sample(cases=600,seed=20261020,scale=.5):
    if cases<10 or not np.isfinite(scale) or scale<=0:
        raise ValueError('at least 10 cases and a positive finite scale are required')
    sampler=qmc.Sobol(d=16+len(NAMES),scramble=True,seed=seed)
    block_size=2**int(np.ceil(np.log2(cases)))
    accepted=[];coeffs=[];indices=[];rejected=[];offset=0
    with np.load(ROOT/'geometry/assets/topology.npz') as z:cells=z['cells']
    while len(accepted)<cases:
        block=sampler.random_base2(int(np.log2(block_size))) if offset==0 else sampler.random(block_size)
        for j,unit in enumerate(block):
            coeff=np.zeros(40);coeff[:16]=(2*unit[:16]-1)*scale
            _,xyz,_=reconstruct(coeff);p=xyz[cells]
            vol=np.einsum('ij,ij->i',np.cross(p[:,1]-p[:,0],p[:,2]-p[:,0]),p[:,3]-p[:,0])/6
            if not np.isfinite(vol).all() or (vol<=0).any():
                rejected.append(offset+j);continue
            accepted.append(unit);coeffs.append(coeff);indices.append(offset+j)
            if len(accepted)==cases:break
        offset+=len(block)
        if offset>100*block_size:raise ValueError('too many invalid mesh candidates')
    unit=np.asarray(accepted);coeffs=np.asarray(coeffs)
    physical=from_unit(unit[:,16:])
    perm=np.random.default_rng(seed+1).permutation(cases)
    splits=np.full(cases,'test',dtype='<U5')
    ntrain,nval=int(.76*cases),int(.12*cases)
    splits[perm[:ntrain]]='train';splits[perm[ntrain:ntrain+nval]]='val'
    rows=[dict(case=f'joint{i:04d}',geom=f'joint{i:04d}',split=splits[i],
               **{k:f'{v:.8g}' for k,v in zip(NAMES,physical[i])}) for i in range(cases)]
    design=dict(seed=seed,split_seed=seed+1,cases=cases,scale_sigma=scale,varied_modes=16,
        center='mean',joint_dimensions=24,coefficients=coeffs.tolist(),
        accepted_sobol_indices=indices,rejected_sobol_indices=rejected,generated_sobol_points=offset,
        joint_unit_samples=unit.tolist(),
        sampling='One scrambled 24-dimensional Sobol sequence: 16 geometry coefficients + 8 physical parameters; reject entire pairs with inverted/nonfinite tetrahedra',
        splits=dict(Counter(splits)),parameter_space=SPACE,
        steps=200,dt='period_s / 200',
        pca_sha256=hashlib.sha256((ROOT/'geometry/assets/pca_basis.npz').read_bytes()).hexdigest(),
        topology_sha256=hashlib.sha256((ROOT/'geometry/assets/topology.npz').read_bytes()).hexdigest())
    return rows,coeffs,design


def prepare(data,cases=600,seed=20261020,scale=.5,jobs=6):
    if jobs<1:raise ValueError('jobs must be positive')
    data=Path(data).resolve();rows,coeffs,design=sample(cases,seed,scale)
    stream=io.StringIO(newline='');w=csv.DictWriter(stream,fieldnames=['case','geom','split',*NAMES])
    w.writeheader();w.writerows(rows);manifest=stream.getvalue().encode()
    design['params_sha256']=hashlib.sha256(manifest).hexdigest()
    data.mkdir(parents=True,exist_ok=True);control=data/'_campaign';control.mkdir(exist_ok=True)
    if (data/'design.json').exists() and json.loads((data/'design.json').read_text())!=json.loads(json.dumps(design)):
        raise ValueError('existing design differs; choose a new output directory')
    if (data/'params.csv').exists() and (data/'params.csv').read_bytes()!=manifest:
        raise ValueError('existing manifest differs; choose a new output directory')
    (data/'design.json').write_text(json.dumps(design,indent=2)+'\n');(data/'params.csv').write_bytes(manifest)
    np.save(data/'joint_unit_samples.npy',np.asarray(design['joint_unit_samples']))
    with (data/'shape_coeffs.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['geom',*[f'pc{i:02d}' for i in range(16)]])
        w.writerows([r['geom'],*c[:16]] for r,c in zip(rows,coeffs))
    def mesh(pair):
        row,coeff=pair;out=data/'_geometry'/f'dalton_geom_{row["geom"]}'
        if out.exists():
            meta=json.loads((out/'meta.json').read_text())
            if meta['center']!='mean' or not np.array_equal(meta['coefficients'],coeff):raise ValueError(f'mesh differs: {out}')
        else:generate(out,coeff,center='mean')
        with (control/f'check_{row["geom"]}.log').open('w') as log:
            subprocess.run([sys.executable,str(ROOT/'tools/check_mesh.py'),str(out),'--strict'],
                cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        print(f'Geometry ready: {row["geom"]}',flush=True)
    with ThreadPoolExecutor(max_workers=jobs) as pool:list(pool.map(mesh,zip(rows,coeffs)))
    (control/'prepared.done').touch()
    print(f'Prepared and checked {cases} jointly sampled geometry/parameter pairs; splits={design["splits"]}',flush=True)
    return design


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data',type=Path,default=ROOT/'surrogate_data.geometry_joint')
    ap.add_argument('--cases',type=int,default=600);ap.add_argument('--seed',type=int,default=20261020)
    ap.add_argument('--scale',type=float,default=.5);ap.add_argument('--jobs',type=int,default=6)
    a=ap.parse_args()
    try:prepare(a.data,a.cases,a.seed,a.scale,a.jobs)
    except Exception:
        control=a.data/'_campaign';control.mkdir(parents=True,exist_ok=True)
        (control/'prepare.failed').touch()
        raise
