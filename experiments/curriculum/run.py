"""Resume-safe 0.25 -> 0.5 -> 1.0 sigma curriculum. Run in the shared checkout."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import socket
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))


def stamp():return datetime.now(timezone.utc).isoformat()


def write_json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(path)


def read_checkpoint(path):
    import torch
    return torch.load(path,map_location='cpu',weights_only=False)


def is_complete(run,iters=None):
    path=run/'last.pt'
    if not path.exists():return False
    ck=read_checkpoint(path)
    if iters is None:iters=ck['args']['iters']
    if ck['args']['iters']!=iters:raise ValueError(f'iteration schedule differs: {run}')
    return ck['iter']==iters


def best_source(run):
    candidates=[]
    for path in run.glob('*.pt'):
        ck=read_checkpoint(path);loss=ck.get('training_loss')
        if loss is not None and math.isfinite(loss):candidates.append((loss,-ck['iter'],path))
    if not candidates:raise ValueError(f'no saved checkpoint with finite training loss: {run}')
    return min(candidates)[2]


def freeze_source(source_run,control):
    """Copy into controller storage, never into a source/output model directory."""
    snapshot=control/'initial.pt';meta=control/'initialization.json'
    if snapshot.exists():
        expected=json.loads(meta.read_text())
        if expected['source_run']!=str(source_run.resolve()):raise ValueError('initial source run differs')
        if hashlib.sha256(snapshot.read_bytes()).hexdigest()!=expected['sha256']:raise ValueError('initial snapshot hash differs')
        return snapshot
    chosen=best_source(source_run);control.mkdir(parents=True,exist_ok=True)
    tmp=control/'initial.pt.tmp';shutil.copy2(chosen,tmp);tmp.replace(snapshot)
    ck=read_checkpoint(snapshot)
    write_json(meta,dict(source_run=str(source_run.resolve()),source_checkpoint=str(chosen),
        iteration=ck['iter'],training_loss=ck['training_loss'],
        sha256=hashlib.sha256(snapshot.read_bytes()).hexdigest(),selection='minimum saved training-window loss'))
    return snapshot


def train_command(python,data,out,snapshot,iters):
    if out.resolve() in (snapshot.resolve().parent,snapshot.resolve()):raise ValueError('unsafe training output')
    command=[python,'-m','surrogate.train_base',str(data),'--out',str(out),
        '--iters',str(iters),'--batch','4','--lr','1e-4','--noise-pos','5e-5',
        '--eval-every','1000','--eval-cases','10','--seed','0']
    if (out/'last.pt').exists():command+=['--resume',str(out/'last.pt')]
    else:
        if out.exists() and any(out.iterdir()):raise ValueError(f'nonempty output without checkpoint: {out}')
        command+=['--init-from',str(snapshot)]
    return command


def verify_data(data,scale,cases):
    import csv
    from surrogate.generate import done
    from surrogate.params import SPACE
    design=json.loads((data/'design.json').read_text())
    if design['scale_sigma']!=scale or design['varied_modes']!=16 or design['cases']!=cases or design['joint_dimensions']!=24:
        raise ValueError(f'dataset design differs: {data}')
    if design['steps']!=200 or design['center']!='mean' or design['parameter_space']!=json.loads(json.dumps(SPACE)):
        raise ValueError('PDE/parameter design differs')
    if hashlib.sha256((data/'params.csv').read_bytes()).hexdigest()!=design['params_sha256']:
        raise ValueError('parameter manifest hash differs')
    with (data/'params.csv').open() as f:rows=list(csv.DictReader(f))
    if len(rows)!=cases or any(not done(data/r['case']) for r in rows):raise ValueError('dataset has incomplete cases')
    if not (data/'_campaign/validated.done').exists():raise ValueError('dataset not independently validated')


def cuda_probe():
    try:
        import torch
        if not torch.cuda.is_available():return dict(ready=False,error='CUDA unavailable')
        torch.ones(1,device='cuda').sum().item()
        free,total=torch.cuda.mem_get_info()
        # Also works when NVML/nvidia-smi is broken but CUDA itself is usable.
        return dict(ready=total-free<1024**3,free_GiB=free/1024**3,total_GiB=total/1024**3)
    except Exception as exc:return dict(ready=False,error=str(exc))


def worker_task(a):
    data=a.data.resolve();control=data/'_campaign';control.mkdir(parents=True,exist_ok=True)
    if a.task=='prepare':
        from experiments.geometry_joint.prepare import prepare
        prepare(data,a.cases,a.seed,a.scale,jobs=6)
    elif a.task=='generate':
        import csv
        from surrogate.generate import done
        # A restarted controller waits for any surviving worker for these shards.
        with (control/f'curriculum_{socket.gethostname().split(".")[0]}.lock').open('w') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            with (data/'params.csv').open() as f:rows=list(csv.DictReader(f))
            for i,row in enumerate(rows):
                if i%5 not in a.shard:continue
                case=data/row['case']
                if case.exists() and not done(case):
                    if case.is_symlink():raise ValueError(f'refusing to replace source link: {case}')
                    archived=data/'_failed_attempts'/row['case']/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
                    archived.parent.mkdir(parents=True,exist_ok=True);shutil.move(str(case),str(archived))
            command=[sys.executable,'-m','surrogate.generate',str(data/'params.csv'),str(data),
                '--mesh-root',str(data/'_geometry'),'--jobs',str(a.jobs),'--steps','200',
                '--shards','5','--shard',*[str(i) for i in a.shard],
                '--status-file',f'curriculum_status.{socket.gethostname().split(".")[0]}.tsv']
            return subprocess.run(command,cwd=ROOT).returncode
    elif a.task=='validate':
        import csv
        from surrogate.generate import done
        with (data/'params.csv').open() as f:rows=list(csv.DictReader(f))
        failed=[r['case'] for r in rows if not done(data/r['case'])]
        def retry(case):
            if not (data/case/'metadata.json').exists():return dict(case=case,returncode=None)
            with (control/f'curriculum_retry_{case}.log').open('a') as log:
                rc=subprocess.run([sys.executable,str(ROOT/'experiments/geometry_narrow/retry.py'),
                    '--data',str(data),'--case',case,'--strategy','robust'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT).returncode
            return dict(case=case,returncode=rc)
        with ThreadPoolExecutor(max_workers=2) as pool:retries=list(pool.map(retry,failed))
        write_json(control/'curriculum_retries.json',retries)
        remaining=[r['case'] for r in rows if not done(data/r['case'])]
        if remaining:
            write_json(control/'validation.failed',remaining)
            raise ValueError(f'{len(remaining)} simulations incomplete: {data}')
        subprocess.run([sys.executable,str(ROOT/'experiments/geometry_narrow/validate.py'),str(data)],cwd=ROOT,check=True)
        (control/'validated.done').touch()
        (control/'validation.failed').unlink(missing_ok=True)
    return 0


class Pipeline:
    def __init__(self,a):
        self.a=a;self.control=a.control.resolve();self.control.mkdir(parents=True,exist_ok=True)
        self.workers=json.loads(a.machines.read_text())['workers']
        self.env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
            PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES=a.gpu)
        self.env.setdefault('MPLCONFIGDIR',str(self.control/'mpl-cache'))

    def status(self,**state):
        state.update(updated_at_utc=stamp());write_json(self.control/'status.json',state)
        print(json.dumps(state),flush=True)

    def remote(self,worker,task,data,**kw):
        host=worker['host']+'.maths.gla.ac.uk'
        command=[str(Path(self.a.solver_python).expanduser()),str(Path(__file__).resolve()),
                 '--task',task,'--data',str(data), '--cases',str(self.a.cases)]
        if task=='generate':command+=['--jobs',str(worker['jobs']),'--shard',*[str(s) for s in worker['shards']]]
        for key,value in kw.items():command+=['--'+key,str(value)]
        remote='cd '+shlex.quote(str(ROOT))+' && '+shlex.join(
            ['env','OMP_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1','PYTHONUNBUFFERED=1','PYTHONDONTWRITEBYTECODE=1',*command])
        ssh=['ssh','-o','BatchMode=yes','-o','StrictHostKeyChecking=accept-new','-o','ConnectTimeout=10',host,remote]
        with (data/'_campaign'/f'curriculum_{task}_{worker["host"]}.log').open('a') as log:
            log.write('\nSTART '+stamp()+'\n');log.flush()
            return subprocess.run(ssh,env=self.env,stdout=log,stderr=subprocess.STDOUT).returncode

    def dataset(self,data,scale,seed):
        data.mkdir(parents=True,exist_ok=True);(data/'_campaign').mkdir(exist_ok=True)
        if (data/'_campaign/validated.done').exists():
            verify_data(data,scale,self.a.cases);return
        self.status(step='prepare_data',data=str(data),scale=scale)
        if not (data/'_campaign/prepared.done').exists():
            if self.remote(self.workers[0],'prepare',data,scale=scale,seed=seed):raise RuntimeError('geometry preparation failed; inspect campaign log')
        self.status(step='simulate',data=str(data),scale=scale)
        with ThreadPoolExecutor(max_workers=len(self.workers)) as pool:
            codes=list(pool.map(lambda w:self.remote(w,'generate',data),self.workers))
        self.status(step='validate_and_recover',data=str(data),worker_exit_codes=codes)
        if self.remote(self.workers[0],'validate',data):raise RuntimeError('simulation recovery/validation failed; training not started')
        verify_data(data,scale,self.a.cases)

    def wait_gpu(self):
        while True:
            try:
                p=subprocess.run([self.a.train_python,str(Path(__file__).resolve()),'--task','cuda'],
                    env=self.env,capture_output=True,text=True,timeout=45)
                info=json.loads(p.stdout.strip().splitlines()[-1])
            except Exception as exc:info=dict(ready=False,error=str(exc))
            if info.get('ready'):return
            self.status(step='waiting_for_idle_CUDA',gpu=self.a.gpu,probe=info)
            time.sleep(self.a.poll)

    def train(self,source,data,out,label):
        if (out/'last.pt').exists() and Path(read_checkpoint(out/'last.pt')['data']).resolve()!=data.resolve():
            raise ValueError('existing output belongs to a different dataset')
        stage_control=self.control/label
        if is_complete(out,self.a.iters):return
        snapshot=freeze_source(source,stage_control)
        if source.resolve()==out.resolve():raise ValueError('source and output must differ')
        self.wait_gpu()
        command=train_command(self.a.train_python,data,out,snapshot,self.a.iters)
        self.status(step='training',stage=label,data=str(data),out=str(out),initialization=json.loads((stage_control/'initialization.json').read_text()))
        with (stage_control/'training.log').open('a') as log:
            log.write('\nSTART '+stamp()+'\n');log.flush()
            subprocess.run(command,cwd=ROOT,env=self.env,stdout=log,stderr=subprocess.STDOUT,check=True)
        if not is_complete(out,self.a.iters):raise RuntimeError('training returned without final checkpoint')
        (stage_control/'training.done').touch()

    def run(self):
        with (self.control/'pipeline.lock').open('w') as lock:
            try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:raise RuntimeError('curriculum controller already running')
            plan={k:str(getattr(self.a,k)) for k in ['source_run','half_data','half_run','full_data','full_run','cases','iters']}
            plan.update(recipe='one-step-noise-5e-5-lr1e-4-batch4-eval1000-v1',
                machines_sha256=hashlib.sha256(self.a.machines.read_bytes()).hexdigest())
            plan_path=self.control/'plan.json'
            if plan_path.exists() and json.loads(plan_path.read_text())!=plan:
                raise ValueError('pipeline plan differs; choose a new control directory')
            write_json(plan_path,plan)
            while not is_complete(self.a.source_run):
                self.status(step='waiting_for_current_narrow_training',source=str(self.a.source_run))
                time.sleep(self.a.poll)
            for label,data,out,scale,seed in [
                ('sigma050',self.a.half_data,self.a.half_run,.5,20261020),
                ('sigma100',self.a.full_data,self.a.full_run,1.,20261022)]:
                self.dataset(data,scale,seed)
                self.train(self.a.source_run if scale==.5 else self.a.half_run,data,out,label)
            self.status(step='complete',half_run=str(self.a.half_run),full_run=str(self.a.full_run))
            (self.control/'complete.done').touch()


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--task',choices=['pipeline','prepare','generate','validate','cuda'],default='pipeline')
    ap.add_argument('--data',type=Path);ap.add_argument('--scale',type=float,default=.5)
    ap.add_argument('--seed',type=int,default=20261020);ap.add_argument('--jobs',type=int,default=6)
    ap.add_argument('--shard',type=int,nargs='+',default=[0])
    ap.add_argument('--cases',type=int,default=600);ap.add_argument('--iters',type=int,default=120000)
    ap.add_argument('--poll',type=float,default=60);ap.add_argument('--gpu',default='0')
    ap.add_argument('--solver-python',default=str(Path('~/miniconda3/envs/lv0d3d/bin/python').expanduser()))
    ap.add_argument('--train-python',default=sys.executable)
    ap.add_argument('--machines',type=Path,default=ROOT/'experiments/geometry_joint/machines.json')
    ap.add_argument('--control',type=Path,default=ROOT/'surrogate_runs/geometry_curriculum_control')
    ap.add_argument('--source-run',type=Path,default=ROOT/'surrogate_runs/geometry_narrow_from_mean_loss')
    ap.add_argument('--half-data',type=Path,default=ROOT/'surrogate_data.geometry_joint')
    ap.add_argument('--half-run',type=Path,default=ROOT/'surrogate_runs/geometry_joint_sigma050')
    ap.add_argument('--full-data',type=Path,default=ROOT/'surrogate_data.geometry_joint_sigma100')
    ap.add_argument('--full-run',type=Path,default=ROOT/'surrogate_runs/geometry_joint_sigma100')
    a=ap.parse_args()
    if a.poll<=0 or a.iters<1000 or a.iters%1000:ap.error('poll must be positive; iterations must be a positive multiple of 1000')
    for key in ['source_run','half_data','half_run','full_data','full_run','control']:
        setattr(a,key,getattr(a,key).resolve())
    if len({a.source_run,a.half_run,a.full_run})!=3 or a.half_data==a.full_data:
        ap.error('each stage needs distinct data/model paths')
    if a.task=='cuda':print(json.dumps(cuda_probe()));return
    if a.task!='pipeline':raise SystemExit(worker_task(a))
    try:Pipeline(a).run()
    except Exception as exc:
        # Do not overwrite status owned by a different live controller.
        if 'already running' not in str(exc):write_json(a.control/'status.json',dict(step='failed',error=str(exc),updated_at_utc=stamp()))
        raise


if __name__=='__main__':main()
