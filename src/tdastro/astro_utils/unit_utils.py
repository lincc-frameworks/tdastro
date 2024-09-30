import numpy as np
from astropy import constants as const


def flam_to_fnu(flux_flam, wavelengths, wave_unit=None, flam_unit=None, fnu_unit=None):
    """
    Covert flux from f_lambda unit to f_nu unit

    Parameters
    ----------
    flux_flam : `list` or `numpy.ndarray`
        The flux values in flam units. This can be a single N-length array
        or an M x N matrix.
    wavelengths: `list` or `numpy.ndarray`
        The wavelength values associated with the input flux values.
        This can be a single N-length array or an M x N matrix. If it is an
        N-length array, the same wavelength values are used for each flux_flam.
    wave_unit: `astropy.units.Unit`
        The unit for the wavelength values.
    flam_unit: `astropy.units.Unit`
        The unit for the input flux_flam values.
    fnu_unit: `astropy.units.Unit`
        The unit for the output flux_fnu values.

    Returns
    -------
    flux_fnu : `list` or `np.array`
        The flux values in fnu units.
    """
    flux_flam = np.array(flux_flam) * flam_unit
    wavelengths = np.array(wavelengths) * wave_unit

    # Check if we need to repeat wavelengths to match the number
    # of rows in flux_flam.
    if len(flux_flam.shape) > 1 and len(wavelengths.shape) == 1:
        num_rows = flux_flam.shape[0]
        wavelengths = np.tile(wavelengths, (num_rows, 1))

    # Check that the shapes match.
    if flux_flam.shape != wavelengths.shape:
        raise ValueError(
            f"Mismatched sizes for flux_flam={flux_flam.shape} " f"and wavelengths={wavelengths.shape}."
        )

    # convert flux in flam_unit (e.g. ergs/s/cm^2/A) to fnu_unit (e.g. nJy or ergs/s/cm^2/Hz)
    flux_fnu = (flux_flam * (wavelengths**2) / const.c).to_value(fnu_unit)
    return flux_fnu
