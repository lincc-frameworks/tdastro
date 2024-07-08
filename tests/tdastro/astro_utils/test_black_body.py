import astropy.constants as const
import numpy as np
from scipy.integrate import trapezoid
from tdastro.astro_utils.black_body import black_body_luminosity_density


def test_bolometric_luminosity():
    """Test that integral of black-body luminosity density is equal to the bolomertric luminosity"""
    speed_of_light = const.c.cgs.value
    bolometric_luminosity = const.L_sun.cgs.value
    radius = const.R_sun.cgs.value
    teff = (bolometric_luminosity / (4 * np.pi * radius**2 * const.sigma_sb.cgs.value)) ** 0.25

    # From 10 to 100,000 Angstroms
    wavelengths = np.logspace(5, 1, 1001) * 1e-8

    luminosity_density = black_body_luminosity_density(teff, radius, wavelengths)
    total_luminosity = trapezoid(y=luminosity_density, x=speed_of_light / wavelengths)

    np.testing.assert_allclose(total_luminosity, bolometric_luminosity, rtol=1e-3)
