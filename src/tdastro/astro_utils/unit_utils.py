from astropy import constants as const


def flam_to_fnu(flux_flam, wavelengths, wave_unit=None, flam_unit=None, fnu_unit=None):
    """
    Covert flux from f_lambda unit to f_nu unit
    """

    flux_flam = flux_flam * flam_unit

    # convert flux in flam_unit (e.g. ergs/s/cm^2/A) to fnu_unit (e.g. nJy or ergs/s/cm^2/Hz)
    flux_fnu = (flux_flam * (wavelengths * wave_unit) ** 2 / const.c).to(fnu_unit).value

    return flux_fnu
