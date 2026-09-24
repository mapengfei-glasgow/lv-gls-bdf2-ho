# Original P1 LV solver — full-cycle plain-mesh package

Self-contained package for the successful original P1 (displacement + kappa penalty)
full-cycle run on the 972-cell `lv_aha17_v2_plain` mesh.

## Contents

```
run_plain_p1_vtk_cycle.py   driver: full 200-step cycle + one VTU per step
run.sh                      convenience launcher
main_3d_only.py             root model/parameters/loading (imported by the driver)
solidmechanicsproblem.py    P1 penalty solid solver
material.py                 Holzapfel–Ogden material
mesh_io.py                  mesh/volume adapters
left_ventricle/             XDMF mesh/fibre reader used by main_3d_only.py
mesh_plain/                 lv_aha17_v2_plain XDMF + H5 mesh (cm units) + info.json
reference_output/           monitor CSV and summary figure from the successful run
environment.yml             conda environment sketch
requirements-extra.txt      additional pip package (meshio)
versions.txt                exact versions used for the uploaded run
```

## Environment

The uploaded run used:

- python 3.12.13
- dolfinx / basix / ffcx 0.11.0, ufl 2026.1.0, io4dolfinx 1.4.0, ADIOS2 2.12.1
- PETSc 3.25.5 / MUMPS, mpi4py 4.1.2
- numpy 2.5.3, scipy 1.18.0, h5py 3.16.0, meshio 5.3.5

Create an environment with `conda env create -f environment.yml` (or use an existing
DOLFINx 0.11 environment), then:

```bash
conda activate lv0d3d
pip install -r requirements-extra.txt
```

## Run

```bash
bash run.sh                       # full cycle: dt=0.005 s, 200 steps, t=0.005..1.0 s
# or
SIMPLE_3D0D_STEPS=20 bash run.sh  # quick test
# output goes to ./output by default; override with SIMPLE_3D0D_OUTPUT_DIR=/path
```

The driver writes:

- `output/vtk/plain.pvd` and `output/vtk/plain_00000.vtu` ... `plain_00199.vtu`
  (open `plain.pvd` in ParaView)
- `output/monitor_plain_p1_full_cycle.csv`

Each VTU contains point data `displacement` [m], `velocity` [m/s] and cell data
`J`, `f0`, `s0`. The source mesh is in centimetres; the solver scales it by 0.01
(`SIMPLE_3D0D_MESH_SCALE=0.01`) so all exported lengths/displacements are in metres.

## Reference result (uploaded run)

- V_max = 252.5951 mL @ t=0.500 s
- V_min = 141.8193 mL @ t=0.650 s
- V(1.0) = 252.1898 mL
- peak p_endo = 14.4785 kPa, peak Ta = 83.324 kPa @ t=0.650 s
- J range over the cycle = [0.8918, 1.1192]
- max |u| = 1.8438e-02 m

The packaged VTK time series of this run is available at:

https://bucket.r2.mapengfei.cn/LV0D3D/plain_p1_full_cycle_vtk.zip
