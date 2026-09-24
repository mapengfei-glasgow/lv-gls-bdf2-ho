"""LV BDF2 P1/P1 PDF HO GLS experiment; incremental VTK and diagnostics."""
import csv
import json
import os
from pathlib import Path
import xml.etree.ElementTree as ET

import basix
import meshio
import numpy as np
import ufl
from dolfinx import fem, plot
from mpi4py import MPI
from scipy.spatial import cKDTree

# Reuse the existing mesh defaults and loading definitions without running it.
from run_lagrange_vtk_cycle import HERE, base, HolzapfelOgden, LeftVentricleCapacity
from solidmechanics_gls import SolidMechanicsProblem_GLS
from material_ho_pdf import HolzapfelOgdenPDF


def write_pvd(vtk_dir, times):
    root=ET.Element("VTKFile",type="Collection",version="0.1",byte_order="LittleEndian")
    collection=ET.SubElement(root,"Collection")
    for i,t in enumerate(times):
        ET.SubElement(collection,"DataSet",timestep=f"{t:.12g}",group="",part="0",file=f"gls_{i:05d}.vtu")
    ET.ElementTree(root).write(vtk_dir/"gls.pvd",encoding="utf-8",xml_declaration=True)


def main():
    if MPI.COMM_WORLD.size != 1:
        raise RuntimeError("The meshio exporter requires a single MPI rank")
    out=Path(os.environ.get("SIMPLE_3D0D_OUTPUT_DIR",str(HERE/"out_gls_bdf2")))
    vtk_dir=out/"vtk"
    vtk_dir.mkdir(parents=True,exist_ok=True)
    if list(vtk_dir.glob("*.vtu")):
        raise RuntimeError("Output already has frames: choose a new SIMPLE_3D0D_OUTPUT_DIR")
    dt,nsteps=base.TIME_PARAMS["dt"],base.NUM_STEPS
    if dt<=0 or nsteps<1:
        raise ValueError("Positive dt and steps required")
    markers={name:int(os.environ[f"SIMPLE_3D0D_{name.upper()}_TAG"]) for name in ("epi","base","endo")}
    io=base.IO_Solid_LV_XDMF(os.environ["SIMPLE_3D0D_MESH_DIR"],
        scale=float(os.environ["SIMPLE_3D0D_MESH_SCALE"]),markers=markers)
    mesh=io.msh
    center=base.compute_base_center(mesh,io.bdry,markers["base"],MPI.COMM_WORLD)
    config={"SOLVER_PARAMS":base.SOLVER_PARAMS,"TIME_PARAMS":base.TIME_PARAMS,"MARKERS":markers,
            "GEOMETRY":{"long_axis":os.environ["SIMPLE_3D0D_LONG_AXIS"],"base_center":tuple(center)}}
    material=HolzapfelOgdenPDF(io.f0,io.s0,mesh,parameters=base.CONSTITUTIVE_PARAMS)
    problem=SolidMechanicsProblem_GLS(io,material,config,
        {"4":lambda t:base.p_endo_kPa(t)*1000,"5":lambda t:base.Ta_kPa(t)*1000})
    metadata=dict(problem.configuration,dt_s=dt,requested_steps=nsteps,
                  material_parameters=base.CONSTITUTIVE_PARAMS, rho_kg_m3=problem.parameters["rho"],
                  long_axis=os.environ["SIMPLE_3D0D_LONG_AXIS"],base_center_m=list(center),
                  base_constraint="original axial/tangential penalty; radial free",
                  mesh_scale=float(os.environ["SIMPLE_3D0D_MESH_SCALE"]),status="running",completed_steps=0)
    (out/"run_metadata.json").write_text(json.dumps(metadata,indent=2)+"\n")
    print("GLS configuration: "+json.dumps(metadata),flush=True)
    capacity=LeftVentricleCapacity(mesh,io.bdry,io.bdry_markers,("CG",problem.degree,(3,)))
    topology,cell_types,points=plot.vtk_mesh(problem.u_space)
    if problem.degree==1:
        assert np.all(cell_types==10)
        cells=[("tetra",topology.reshape(-1,5)[:,1:])]
    else:
        assert np.all(cell_types==71)
        cells=[("VTK_LAGRANGE_TETRAHEDRON",topology.reshape(-1,11)[:,1:])]
    Qout=fem.functionspace(mesh,("Lagrange",problem.degree))
    p_out=fem.Function(Qout)
    distances,pmap=cKDTree(Qout.tabulate_dof_coordinates()).query(points)
    assert np.max(distances)<1e-12
    V0=fem.functionspace(mesh,("DG",0))
    Jfun=fem.Function(V0)
    j=ufl.det(ufl.Identity(3)+ufl.grad(problem.u))
    jexpr=fem.Expression(j,V0.element.interpolation_points)
    qpoints,_=basix.make_quadrature(mesh.basix_cell(),5)
    jquad=fem.Expression(j,qpoints)
    local_cells=np.arange(mesh.topology.index_map(3).size_local,dtype=np.int32)
    jmap=np.array([V0.dofmap.cell_dofs(i)[0] for i in local_cells])
    fmap=np.array([io.f0.function_space.dofmap.cell_dofs(i)[0] for i in local_cells])
    ref_volume=fem.assemble_scalar(fem.form(1.0*io.dx))
    jerr=fem.form((j-1)**2*io.dx)
    jmean=fem.form(j*io.dx)
    kinetic=fem.form(0.5*problem.parameters["rho"]*ufl.inner(problem.velocity,problem.velocity)*io.dx)
    scales = {name: (fem.Function(V0), fem.Expression(expr, V0.element.interpolation_points))
              for name, expr in {"mu_eff_Pa": problem.mu_eff,
                                 "tau_m": problem.tau_m, "tau_c": problem.tau_c}.items()}
    fields=["time_s","p_endo_kPa","Ta_kPa","V_mL","J_min","J_max","J_minus_1_RMS","J_mean",
            "max_u_m","max_v_m_per_s","p_min_Pa","p_max_Pa","kinetic_energy_J",
            "newton_iterations","snes_reason","galerkin_p_norm","pspg_norm","lsic_norm",
            "galerkin_v_norm","total_residual_norm","constraint_rate_RMS_per_s"]
    times=[]
    try:
        with (out/"monitor_gls.csv").open("w",newline="") as file:
            writer=csv.DictWriter(file,fieldnames=fields)
            writer.writeheader();file.flush()
            for step in range(nsteps):
                t=(step+1)*dt
                problem.update_time_dependent_values(t)
                problem.solve(step)
                Jfun.interpolate(jexpr)
                # Evaluate with the converged state BEFORE shifting BDF2 history.
                for scale_fun, scale_expr in scales.values():
                    scale_fun.interpolate(scale_expr)
                    if not np.isfinite(scale_fun.x.array).all() or scale_fun.x.array.min() < 0:
                        raise RuntimeError(f"Invalid stabilization scale at step {step}")
                jq=jquad.eval(mesh,local_cells)
                if not np.isfinite(jq).all() or jq.min()<=0:
                    raise RuntimeError(f"Nonpositive or nonfinite J at step {step}")
                p_out.interpolate(problem.p)
                u=problem.u.x.array.reshape(-1,3).copy()
                v=problem.velocity.x.array.reshape(-1,3).copy()
                row=dict(time_s=t,p_endo_kPa=base.p_endo_kPa(t),Ta_kPa=base.Ta_kPa(t),
                    V_mL=capacity.calculate_volume(problem.u)*1e6,
                    J_min=float(jq.min()),J_max=float(jq.max()),
                    J_minus_1_RMS=float(np.sqrt(fem.assemble_scalar(jerr)/ref_volume)),
                    J_mean=float(fem.assemble_scalar(jmean)/ref_volume),
                    max_u_m=float(np.linalg.norm(u,axis=1).max()),
                    max_v_m_per_s=float(np.linalg.norm(v,axis=1).max()),
                    p_min_Pa=float(problem.p.x.array.min()),p_max_Pa=float(problem.p.x.array.max()),
                    kinetic_energy_J=float(fem.assemble_scalar(kinetic)),**problem.last_diagnostics)
                if not all(np.isfinite(float(value)) for value in row.values()):
                    raise RuntimeError(f"Nonfinite output at step {step}")
                frame=meshio.Mesh(points,cells,
                    point_data={"displacement":u,"velocity":v,"lagrange_pressure_Pa":p_out.x.array[pmap].copy()},
                    cell_data={"J":[Jfun.x.array[jmap].copy()],**{name: [fun.x.array[jmap].copy()] for name, (fun, _) in scales.items()},
                               "f0":[io.f0.x.array.reshape(-1,3)[fmap].copy()],
                               "s0":[io.s0.x.array.reshape(-1,3)[fmap].copy()]})
                meshio.write(vtk_dir/f"gls_{step:05d}.vtu",frame)
                writer.writerow(row);file.flush()
                times.append(t);write_pvd(vtk_dir,times)
                problem.update_variables()
                if step%10==0 or step==nsteps-1:
                    print(f"[GLS] t={t:.4f} V={row['V_mL']:.4f} J=[{row['J_min']:.5f},{row['J_max']:.5f}] "
                          f"J_RMS={row['J_minus_1_RMS']:.5g} |Gp|={row['galerkin_p_norm']:.3g} "
                          f"|Spspg|={row['pspg_norm']:.3g}",flush=True)
        metadata.update(status="completed",completed_steps=len(times))
    except Exception as exc:
        metadata.update(status="failed",completed_steps=len(times),error=str(exc))
        raise
    finally:
        (out/"run_metadata.json").write_text(json.dumps(metadata,indent=2)+"\n")


if __name__=="__main__":
    main()
