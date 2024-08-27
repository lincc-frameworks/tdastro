from astropy import constants as const


def flam_to_fnu(flux_flam, wavelengths, wave_unit=None, flam_unit=None, fnu_unit=None):
    """
    Covert flux from f_lambda unit to f_nu unit

    Parameters
    ----------
    flux_flam : `list` or `numpy.ndarray`
        The flux values in flam units.
    wavelengths: `list` or `numpy.ndarray`
        The wavelength values associated with the input flux values.
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

    flux_flam = flux_flam * flam_unit

    # convert flux in flam_unit (e.g. ergs/s/cm^2/A) to fnu_unit (e.g. nJy or ergs/s/cm^2/Hz)
    flux_fnu = (flux_flam * (wavelengths * wave_unit) ** 2 / const.c).to_value(fnu_unit)

    return flux_fnu
