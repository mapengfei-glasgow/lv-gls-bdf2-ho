# Run 2026-09-24 (macOS arm64, conda env lv0d3d: Python 3.12.13, DOLFINx 0.11.0, UFL 2026.1.0, PETSc real)

| program | exit | wall time | vs. reference monitor CSV (200 steps) |
|---|---|---|---|
| P1 plain (`p1_plain_full_cycle_program`) | 0 | 47 s | all columns rel. diff <= 3e-14 |
| Lagrange P2/P1 augmented (`lv_lagrange_vtk_and_code/source`) | 0 | 101 s | u, J, p rel. diff <= 2e-13; V_mL up to 4% (final 276.58 vs 267.53 mL) |
| GLS BDF2 HO-PDF (`ho_pdf_revision/source`) | 0 | 31 s | rel. diff <= 2e-11 (residual norm 4e-3); verify_gls_results.py identical to reference |

Lagrange V_mL: the solution fields match to round-off, so the gap comes from the Delaunay
cavity-volume post-processing (mesh_io.LeftVentricleCapacity) on P2 endocardial nodes: the
Delaunay tetrahedralisation of this degenerate point set is not unique and differs across
platform / qhull builds (already 0.015 mL at t=0.005 s with |du| ~ 1e-18 m).
VTK frames and lagrange.zip are in this folder but not tracked by git.
