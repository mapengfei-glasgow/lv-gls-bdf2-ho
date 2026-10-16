"""Convert solver field dumps (fields/*.npy written by lvsolver --save-fields / surrogate.generate)
to ParaView-readable VTU frames plus a PVD time series.

  python tools/fields_to_vtk.py check/geom_0000                 # every 10th frame
  python tools/fields_to_vtk.py surrogate_data/geom_0001 --stride 20
  python tools/fields_to_vtk.py surrogate_data/geom_0001 --stride 0   # all frames

Reads fields/{coords,cells,u,time}.npy; writes OUT/frame_XXXXX.vtu with the deformed
coordinates and the displacement u (m and mm) as point data, and OUT/rollout.pvd.
Open the .pvd file in ParaView and press play to see the whole cycle.
Run in an environment with numpy + meshio (lv0d3d or lvml).
"""
import argparse
from pathlib import Path

import meshio
import numpy as np


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("case", type=Path, help="case directory containing fields/*.npy")
    ap.add_argument("--stride", type=int, default=10, help="write every Nth frame (0 = all frames)")
    ap.add_argument("--out", type=Path, default=None, help="output directory (default: CASE/vtk)")
    a = ap.parse_args(argv)
    f = a.case / "fields"
    coords = np.asarray(np.load(f / "coords.npy"), dtype=np.float64)
    cells = np.load(f / "cells.npy")
    u = np.load(f / "u.npy")
    time = np.load(f / "time.npy")
    out = a.out or (a.case / "vtk")
    out.mkdir(parents=True, exist_ok=True)
    stride = max(1, a.stride) if a.stride != 0 else 1
    idx = list(range(0, len(time), stride))
    if idx[-1] != len(time) - 1:
        idx.append(len(time) - 1)
    items = []
    for n in idx:
        fp = out / f"frame_{n:05d}.vtu"
        meshio.write(fp, meshio.Mesh(coords + u[n], [("tetra", cells)],
                                     point_data={"u_m": u[n], "u_mm": 1e3 * u[n]}))
        items.append(f'<DataSet timestep="{time[n]:.8g}" file="{fp.name}"/>')
    (out / "rollout.pvd").write_text('<?xml version="1.0"?>\n<VTKFile type="Collection" version="0.1">'
                                     "<Collection>" + "".join(items) + "</Collection></VTKFile>\n")
    print(f"{len(idx)} frames -> {out}  (open {out / 'rollout.pvd'} in ParaView)")


if __name__ == "__main__":
    main()
