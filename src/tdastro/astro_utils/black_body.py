import numpy as np
from astropy import units as u
from astropy.modeling.physical_models import BlackBody


def black_body_luminosity_density_per_solid(temperature, radius, wavelengths):
    """Calculate the black-body luminosity density per solid angle.

    It is L_nu / (4 pi) for a spherical isotropic source.

    Parameters
    ----------
    temperature : `float`
        The effective temperature of the star, in kelvins.
    radius : `float`
        The radius of the star, in solar radii.
    wavelengths : `numpy.ndarray`
        A length N array of wavelengths.

    Returns
    -------
    luminosity_density : `numpy.ndarray`
        A length N array of luminosity density values.
    """
    black_body = BlackBody(temperature * u.K)
    intensity_per_freq = black_body(wavelengths * u.cm).to_value(
        u.erg * u.cm**-2 * u.s**-1 * u.steradian**-1 * u.Hz**-1
    )
    surface_flux = intensity_per_freq * np.pi
    # 4pi r^2 over 4pi
    surface_area_per_solid_angle = radius**2
    return surface_flux * surface_area_per_solid_angle
