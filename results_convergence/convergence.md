# Mesh refinement: gls_pdf

Same dt = 0.005 s and stabilisation parameters on every level; reference = finest level.
V = exact cavity volume -1/3 int_endo (x-x0).n da from the VTU frames (V ref: reference configuration).

| level | cells | nodes | h (mm) | max Newton | J min | J max | max RMS(J-1) | max abs(J_mean-1) | V ref mL | V(0.2) mL | V(0.5) mL | V(0.65) mL | V(0.8) mL | V(1.0) mL | max abs(u) (mm) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gls_pdf_h1.5 | 899 | 297 | 5.83 | 3 | 0.155 | 1.848 | 0.1966 | 2.30e-03 | 158.78 | 318.81 | 358.61 | 132.34 | 231.10 | 354.58 | 33.860 |
| gls_pdf_h1.0 | 1921 | 635 | 4.51 | 4 | 0.048 | 2.219 | 0.1874 | 2.82e-03 | 164.46 | 333.83 | 376.51 | 144.74 | 251.15 | 371.65 | 34.646 |
| gls_pdf_h0.7 | 3786 | 1240 | 3.61 | 4 | 0.101 | 2.377 | 0.1662 | 3.12e-03 | 166.17 | 336.84 | 379.90 | 146.41 | 253.73 | 375.82 | 35.124 |
| gls_pdf_h0.5 | 8954 | 2636 | 2.71 | 5 | 0.140 | 2.464 | 0.3610 | 2.10e-03 | 167.25 | 347.71 | 390.97 | 144.25 | 246.13 | 387.66 | 35.586 |
| gls_pdf_h0.35 | 21707 | 5840 | 2.02 | 6 | 0.185 | 3.031 | 0.3706 | 2.09e-03 | 167.74 | 351.54 | 394.92 | 145.70 | 248.95 | 389.87 | 34.525 |

## Difference to the finest level (max over the cycle, relative)

| level | h (mm) | V(t) | max abs(u)(t) | RMS(J-1)(t), absolute |
|---|---|---|---|---|
| gls_pdf_h1.5 | 5.83 | 9.37e-02 | 8.16e-02 | 0.1966 |
| gls_pdf_h1.0 | 4.51 | 5.81e-02 | 1.48e-01 | 0.1874 |
| gls_pdf_h0.7 | 3.61 | 5.49e-02 | 1.52e-01 | 0.1662 |
| gls_pdf_h0.5 | 2.71 | 1.34e-02 | 5.69e-02 | 0.3610 |

Observed order (slope of log error vs log h): V 2.37, max abs(u) 0.46; max RMS(J-1) itself vs h over all levels: -0.75 (a consistent incompressible scheme should give a positive slope, i.e. RMS(J-1) -> 0 as h -> 0).
