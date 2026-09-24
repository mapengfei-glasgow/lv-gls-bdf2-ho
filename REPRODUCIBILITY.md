# Reproducibility check

Regenerated files vs. the versions in the previous commit. Relative difference = max |new-ref| / max |ref| per column
(CSV) or |new-ref|/|ref| per key (JSON). Residual / norm / error quantities are reported separately
because they sit at solver-tolerance level. Verdict: bitwise (0), round-off (<=1e-8), tolerance (<=1e-4), DIFFERENT.

## Run status

```
env 3.12.13 dolfinx 0.11.0 ufl 2026.1.0 petsc (3, 25, 5) float64
[p1_plain] exit=0 9s
[lagrange_aug] exit=0 89s
[lagrange_pure] exit=0 94s
[gls_full] exit=0 29s
[gls_verify] exit=0 0s
[gls_smoke] exit=0 2s
[gls_nostab] exit=0 9s
[gls_nostab_verify] exit=0 0s
[ho_smoke] exit=0 2s
[ho_smoke_v] exit=0 0s
[ho_smoke_fs] exit=0 14s
[ho_smoke_fs_v] exit=0 0s
[ho_full] exit=0 14s
[ho_full_v] exit=0 1s
[ho_plot] exit=0 9s
```

## vs. original package results (previous commit)

| file | verdict | max rel. diff (field) | residual-type max rel. diff |
|---|---|---|---|
| `ho_pdf_revision/results/full_cycle/monitor_gls.csv` | round-off | 1.6e-14 (p_min_Pa) | 1.2e-05 (total_residual_norm) |
| `ho_pdf_revision/results/full_cycle/run_metadata.json` | bitwise | 0.0e+00 () |  |
| `ho_pdf_revision/results/full_cycle/validation_summary.json` | round-off | 4.6e-14 (max_J_mean_drift) | 5.9e-02 (max_kinematic_recurrence_error_m) |
| `ho_pdf_revision/results/smoke/monitor_gls.csv` | round-off | 1.2e-14 (max_v_m_per_s) | 5.8e-06 (total_residual_norm) |
| `ho_pdf_revision/results/smoke/run_metadata.json` | bitwise | 0.0e+00 () |  |
| `ho_pdf_revision/results/smoke/validation_summary.json` | round-off | 1.2e-10 (max_J_mean_drift) | 2.2e-01 (max_independent_kinetic_energy_error_J) |
| `ho_pdf_revision/results/smoke_fs/monitor_gls.csv` | round-off | 1.1e-14 (max_v_m_per_s) | 1.8e-06 (total_residual_norm) |
| `ho_pdf_revision/results/smoke_fs/run_metadata.json` | bitwise | 0.0e+00 () |  |
| `ho_pdf_revision/results/smoke_fs/validation_summary.json` | round-off | 8.9e-15 (max_displacement_m) | 4.0e-01 (max_independent_kinetic_energy_error_J) |
| `ho_pdf_revision/source/reference_output/monitor_plain_p1_full_cycle.csv` | round-off | 9.4e-16 (max_u_m) |  |
| `lv_gls_bdf2_vtk_and_code/diagnostics/no_stabilization_short/monitor_gls.csv` | round-off | 7.4e-15 (max_v_m_per_s) | 2.8e-07 (galerkin_v_norm) |
| `lv_gls_bdf2_vtk_and_code/diagnostics/no_stabilization_short/run_metadata.json` | bitwise | 0.0e+00 () |  |
| `lv_gls_bdf2_vtk_and_code/diagnostics/no_stabilization_short/validation_summary.json` | round-off | 3.1e-15 (max_displacement_m) | 1.0e+00 (max_independent_kinetic_energy_error_J) |
| `lv_gls_bdf2_vtk_and_code/monitor_gls.csv` | round-off | 2.0e-14 (p_min_Pa) | 2.2e-05 (total_residual_norm) |
| `lv_gls_bdf2_vtk_and_code/run_metadata.json` | bitwise | 0.0e+00 () |  |
| `lv_gls_bdf2_vtk_and_code/short_control_comparison.json` | round-off | 1.6e-15 (none_p1p1.J_minus_1_RMS) | 1.1e-06 (gls.total_residual_norm) |
| `lv_gls_bdf2_vtk_and_code/source/reference_output/monitor_plain_p1_full_cycle.csv` | round-off | 9.4e-16 (max_u_m) |  |
| `lv_gls_bdf2_vtk_and_code/validation_summary.json` | round-off | 1.2e-13 (max_J_mean_drift) | 4.5e-02 (max_independent_J_error) |
| `lv_lagrange_vtk_and_code/monitor_lagrange_full_cycle.csv` | DIFFERENT | 3.8e-02 (V_mL) |  |
| `lv_lagrange_vtk_and_code/pure_vs_augmented.json` | DIFFERENT | 3.6e-02 (augmented_lagrange.V_max_mL) |  |
| `lv_lagrange_vtk_and_code/run_metadata.json` | bitwise | 0.0e+00 () |  |
| `lv_lagrange_vtk_and_code/source/reference_output/monitor_plain_p1_full_cycle.csv` | round-off | 9.4e-16 (max_u_m) |  |
| `lv_lagrange_vtk_and_code/validation_summary.json` | DIFFERENT | 3.6e-02 (V_max_mL) |  |
| `p1_plain_full_cycle_program/reference_output/monitor_plain_p1_full_cycle.csv` | round-off | 9.4e-16 (max_u_m) |  |

## vs. previous macOS run (commit 43e2fc4, runs/20260924_012845)

| file | verdict | max rel. diff (field) | residual-type max rel. diff |
|---|---|---|---|
| `p1_plain_full_cycle_program/reference_output/monitor_plain_p1_full_cycle.csv` | bitwise | 0.0e+00 () |  |
| `lv_lagrange_vtk_and_code/monitor_lagrange_full_cycle.csv` | bitwise | 0.0e+00 () |  |
| `ho_pdf_revision/results/full_cycle/monitor_gls.csv` | bitwise | 0.0e+00 () |  |
| `ho_pdf_revision/results/full_cycle/validation_summary.json` | bitwise | 0.0e+00 () |  |
