"""Retry a failed case with robust nonlinear options, preserving its PDE and time grid."""
import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from surrogate.params import solver_args
from lvsolver import model


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data',type=Path,required=True)
    ap.add_argument('--case',required=True)
    ap.add_argument('--strategy',choices=['bt','trust','robust'],required=True)
    a=ap.parse_args()
    with (a.data/'params.csv').open() as f:
        rows={r['case']:r for r in csv.DictReader(f)}
    row=rows[a.case]
    out=a.data/a.case
    if out.exists():
        meta=json.loads((out/'metadata.json').read_text())
        if meta.get('status')=='completed':
            print('Case already complete; preserving it',flush=True);return
        archive=a.data/'_failed_attempts'/a.case/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        archive.parent.mkdir(parents=True,exist_ok=True)
        shutil.move(str(out),str(archive))
    options=dict(snes_type='newtontr' if a.strategy=='trust' else 'newtonls',
                 snes_max_it=200,snes_max_funcs=10000)
    if a.strategy!='trust':
        options.update(snes_linesearch_type='bt',snes_linesearch_max_it=40)
    else:
        model.SOLVER_PARAMS.pop('snes_linesearch_type',None)
    model.SOLVER_PARAMS.update(options)
    from lvsolver.problems import GLS
    original_init = GLS._init_forms
    effective = []
    events = []
    def configure(self):
        original_init(self)
        self.problem.solver.setTolerances(max_it=200)
        effective.append(dict(type=self.problem.solver.getType(),
                              tolerances=list(self.problem.solver.getTolerances())))
        if a.strategy=='robust':
            from petsc4py import PETSc
            raw_solve = self.problem.solve
            def robust_solve():
                previous = self.state_n.x.array.copy()
                difference = previous-self.state_nn.x.array
                attempts=[('newtonls',previous,'previous'),
                          ('newtontr',previous+.5*difference,'half extrapolation'),
                          ('newtonls',previous+difference,'full extrapolation'),
                          ('newtontr',previous,'previous')]
                last_error=None
                for index,(kind,guess,label) in enumerate(attempts):
                    solver=self.problem.solver
                    solver.setType(kind);solver.setTolerances(max_it=200)
                    if kind=='newtonls':solver.getLineSearch().setType('bt')
                    self.state.x.array[:]=guess;self.state.x.scatter_forward()
                    try:
                        result=raw_solve()
                        if index:
                            events.append(dict(time_s=getattr(self,'_retry_time',None),
                                solver=kind,guess=label,success=True,iterations=solver.getIterationNumber()))
                        return result
                    except PETSc.Error as exc:
                        last_error=exc
                        events.append(dict(time_s=getattr(self,'_retry_time',None),solver=kind,
                            guess=label,success=False,iterations=solver.getIterationNumber()))
                        print('Nonlinear fallback:',events[-1],flush=True)
                raise last_error
            self.problem.solve=robust_solve
    GLS._init_forms = configure
    original_update = GLS.update_time_dependent_values
    def update(self,t):
        self._retry_time=t
        return original_update(self,t)
    GLS.update_time_dependent_values=update
    from lvsolver.run import main as solve
    argv=['--method','gls','--tau','pdf','--include-fs',
          '--mesh',str(a.data/'_geometry'/f'dalton_geom_{row["geom"]}'),
          '--out',str(out),'--steps','200','--dt',f'{float(row["period_s"])/200:.10g}',
          '--no-vtk','--save-fields',*solver_args(row)]
    try:
        solve(argv)
    finally:
        p=out/'metadata.json'
        if p.exists():
            m=json.loads(p.read_text());m['nonlinear_retry']=dict(strategy=a.strategy,requested_options=options,effective_snes=effective,fallback_events=events,
               note='Constitutive law, loads, geometry, stabilisation and 200-step time grid unchanged')
            p.write_text(json.dumps(m,indent=2)+'\n')


if __name__=='__main__':
    main()
