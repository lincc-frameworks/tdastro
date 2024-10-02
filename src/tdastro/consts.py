import numpy as np
from astropy import units as u

PARSEC_TO_CM = (1 * u.pc).to_value(u.cm)

GAUSS_EFF_AREA2FWHM_SQ = np.pi / (2 * np.log(2))  # ~2.266
"""Effective area of symmetric Gaussian to FWHM squared conversion factor.

It is roughly 2.266, see
https://smtn-002.lsst.io/v/OPSIM-1171/index.html

Notes
-----
This is derived from two facts for a symmetric 2D Gaussian:
1. FWHM² = 8 * ln(2) * sigma², where sigma is the standard deviation.
2. The convolution of a Gaussian with itself at position of (x,y)=μ̄ is
   1 / sigma² / (4 π).
   E.g.
   ∫ g²(x,y) dx dy = 1 / sigma² / (4π),
   where
   g(x,y) = 1 / (2π sigma²) exp(-(x²+y²)/2/sigma²).
"""

ANGSTROM_TO_CM = (1.0 * u.AA).to_value(u.cm)
"""Conversion factor from Angstrom to cm, 1e-8"""


CGS_FNU_UNIT_TO_NJY = (1.0 * u.erg / u.second / u.cm**2 / u.Hz).to_value(u.nJy)
"""Conversion factor from erg/s/cm²/Hz to nJy, 1e32"""
