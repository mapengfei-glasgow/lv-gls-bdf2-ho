"""Dynamic 3D left-ventricle solid mechanics with prescribed loading.

Methods (see README.md):
  p1        P1 displacement, compressible Holzapfel-Ogden with volumetric penalty
  lagrange  P2/P1 Taylor-Hood displacement-pressure, (augmented) Lagrange multiplier
  gls       P1/P1 velocity-pressure, BE/BDF2, PSPG + LSIC stabilisation
"""
