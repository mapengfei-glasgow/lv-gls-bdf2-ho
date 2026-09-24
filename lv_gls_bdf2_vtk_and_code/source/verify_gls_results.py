"""Verify saved GLS files, time recurrence and independently reconstructed J.

Usage: python verify_gls_results.py /absolute/path/to/result_directory
Only numpy and meshio are required; no DOLFINx or MPI initialization.
"""
import argparse
import csv
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import meshio
import numpy as np


def verify(root):
    metadata=json.loads((root/'run_metadata.json').read_text())
    rows=list(csv.DictReader((root/'monitor_gls.csv').open()))
    datasets=ET.parse(root/'vtk/gls.pvd').findall('.//DataSet')
    assert metadata['status']=='completed',metadata
    assert len(rows)==len(datasets)==metadata['completed_steps']==metadata['requested_steps']
    dt=metadata['dt_s']
    max_kinematic_error=max_j_error=max_energy_error=0.0
    u_n=u_nn=None
    for i,(dataset,row) in enumerate(zip(datasets,rows)):
        assert all(np.isfinite(float(v)) for v in row.values())
        assert int(row['snes_reason'])>0
        assert abs(float(row['time_s'])-(i+1)*dt)<1e-12
        assert abs(float(dataset.attrib['timestep'])-float(row['time_s']))<1e-12
        frame=meshio.read(root/'vtk'/dataset.attrib['file'])
        assert frame.cells[0].type=='tetra', 'This independent J check is for P1 only'
        assert frame.cells[0].data.shape==(972,4)
        assert frame.points.shape==(324,3)
        for value in frame.point_data.values():
            assert np.isfinite(value).all()
        for blocks in frame.cell_data.values():
            assert all(np.isfinite(v).all() for v in blocks)
        u,v=frame.point_data['displacement'],frame.point_data['velocity']
        if u_n is None:
            u_n,u_nn=np.zeros_like(u),np.zeros_like(u)
        if i==0 or metadata['time_scheme']=='be':
            discrepancy=u-u_n-dt*v
        else:
            discrepancy=3*u-4*u_n+u_nn-2*dt*v
        max_kinematic_error=max(max_kinematic_error,float(np.abs(discrepancy).max()))
        cells=frame.cells[0].data
        reference=frame.points[cells]
        deformed=(frame.points+u)[cells]
        D0=reference[:,1:,:]-reference[:,:1,:]
        D1=deformed[:,1:,:]-deformed[:,:1,:]
        j_reconstructed=np.linalg.det(D1)/np.linalg.det(D0)
        max_j_error=max(max_j_error,float(np.abs(j_reconstructed-frame.cell_data['J'][0]).max()))
        assert j_reconstructed.min()>0
        vol=np.abs(np.linalg.det(D0))/6
        j_mean=np.dot(vol,j_reconstructed)/vol.sum()
        j_rms=np.sqrt(np.dot(vol,(j_reconstructed-1)**2)/vol.sum())
        assert np.isclose(j_mean,float(row['J_mean']),rtol=1e-9,atol=1e-11)
        assert np.isclose(j_rms,float(row['J_minus_1_RMS']),rtol=1e-9,atol=1e-11)
        vertices_v=v[cells]
        kinetic=0.5*1000*np.sum(vol/20*(np.sum(vertices_v.sum(axis=1)**2,axis=1)
                                      +np.sum(vertices_v**2,axis=(1,2))))
        max_energy_error=max(max_energy_error,abs(kinetic-float(row['kinetic_energy_J'])))
        u_nn,u_n=u_n,u.copy()
    assert max_kinematic_error<1e-11,max_kinematic_error
    assert max_j_error<1e-9,max_j_error
    assert max_energy_error<1e-9,max_energy_error
    summary=dict(frames=len(rows),end_time_s=float(rows[-1]['time_s']),
        all_steps_converged=True,max_newton_iterations=max(int(r['newton_iterations']) for r in rows),
        J_min=min(float(r['J_min']) for r in rows),J_max=max(float(r['J_max']) for r in rows),
        maximum_J_RMS=max(float(r['J_minus_1_RMS']) for r in rows),
        final_J_mean=float(rows[-1]['J_mean']),
        max_J_mean_drift=max(abs(float(r['J_mean'])-1) for r in rows),
        minimum_V_mL=min(float(r['V_mL']) for r in rows),maximum_V_mL=max(float(r['V_mL']) for r in rows),
        final_V_mL=float(rows[-1]['V_mL']),
        minimum_pressure_Pa=min(float(r['p_min_Pa']) for r in rows),
        maximum_pressure_Pa=max(float(r['p_max_Pa']) for r in rows),
        max_displacement_m=max(float(r['max_u_m']) for r in rows),
        max_total_algebraic_residual=max(float(r['total_residual_norm']) for r in rows),
        max_kinematic_recurrence_error_m=max_kinematic_error,
        max_independent_J_error=max_j_error,max_independent_kinetic_energy_error_J=max_energy_error)
    (root/'validation_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('result_directory',type=Path)
    verify(parser.parse_args().result_directory)
