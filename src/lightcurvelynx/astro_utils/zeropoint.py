from __future__ import annotations  # "type1 | type2" syntax in Python <3.9

import numpy as np
import numpy.typing as npt

from lightcurvelynx.astro_utils.mag_flux import mag2flux

_lsstcam_extinction_coeff = {
    "u": -0.458,
    "g": -0.208,
    "r": -0.122,
    "i": -0.074,
    "z": -0.057,
    "y": -0.095,
}
"""The extinction coefficients for the LSST filters.

Values are from
https://community.lsst.org/t/release-of-v3-4-simulations/8548/12
Calculated with syseng_throughputs v1.9
"""

_lsstcam_zeropoint_per_sec_zenith = {
    "u": 26.524,
    "g": 28.508,
    "r": 28.361,
    "i": 28.171,
    "z": 27.782,
    "y": 26.818,
}
"""The zeropoints for the LSST filters at zenith

This is magnitude that produces 1 electron in a 1 second exposure,
see _assign_zero_points() docs for more details.

Values are from
https://community.lsst.org/t/release-of-v3-4-simulations/8548/12
Calculated with syseng_throughputs v1.9
"""


# Suppress "no docstring", because we define it via an attribute.
def magnitude_electron_zeropoint(  # noqa: D103
    *,
    band: npt.ArrayLike,
    airmass: npt.ArrayLike,
    exptime: npt.ArrayLike,
    instr_zp: dict[str, float] | None,
    ext_coeff: dict[str, float] | None,
) -> npt.ArrayLike:
    instr_zp = _lsstcam_zeropoint_per_sec_zenith if instr_zp is None else instr_zp
    ext_coeff = _lsstcam_extinction_coeff if ext_coeff is None else ext_coeff

    instr_zp_getter = np.vectorize(instr_zp.get)
    ext_coeff_getter = np.vectorize(ext_coeff.get)

    return instr_zp_getter(band) + ext_coeff_getter(band) * (airmass - 1) + 2.5 * np.log10(exptime)


magnitude_electron_zeropoint.__doc__ = f"""Photometric zeropoint (magnitude that produces 1 electron) for
    LSST bandpasses (v1.9), using a standard atmosphere scaled
    for different airmasses and scaled for exposure times.

    Parameters
    ----------
    band : ndarray of str
        The filter for which to return the photometric zeropoint.
    airmass : ndarray of float
        The airmass at which to return the photometric zeropoint.
    exptime : ndarray of float
        The exposure time for which to return the photometric zeropoint.
    instr_zp : dict[str, float] or None
        The instrumental zeropoint for each bandpass,
        i.e. AB-magnitude that produces 1 electron in a 1-second exposure.
        Keys are the bandpass names, values are the zeropoints.
        If None, the LSST zeropoints are used:
        {_lsstcam_zeropoint_per_sec_zenith}
    ext_coeff : dict[str, float]
        Atmospheric extinction coefficient for each bandpass.
        Keys are the bandpass names, values are the coefficients.
        If None, the LSST coefficients are used:
        {_lsstcam_extinction_coeff}

    Returns
    -------
    ndarray of float
        AB mags that produces 1 electron.

    Notes
    -----
    Typically, zeropoints are defined as the magnitude of a source
    which would produce 1 count in a 1 second exposure -
    here we use *electron* counts, not ADU counts.

    References
    ----------
    Lynne Jones - https://community.lsst.org/t/release-of-v3-4-simulations/8548/12
    """


# Suppress "no docstring", because we define it via an attribute.
def flux_electron_zeropoint(  # noqa: D103
    *,
    instr_zp_mag: dict[str, float] | None,
    ext_coeff: dict[str, float] | None,
    band: npt.ArrayLike,
    airmass: npt.ArrayLike,
    exptime: npt.ArrayLike,
) -> npt.ArrayLike:
    mag_zp_electron = magnitude_electron_zeropoint(
        instr_zp=instr_zp_mag, ext_coeff=ext_coeff, band=band, airmass=airmass, exptime=exptime
    )
    return mag2flux(mag_zp_electron)


flux_electron_zeropoint.__doc__ = f"""Flux (nJy) producing 1 electron.

    Parameters
    ----------
    band : nparray of str
        The filter for which to return the photometric zeropoint.
    airmass : ndarray of float
        The airmass at which to return the photometric zeropoint.
    exptime : ndarray of float
        The exposure time for which to return the photometric zeropoint.
    instr_zp_mag : dict[str, float]
        The instrumental zeropoint for each bandpass in AB magnitudes,
        i.e. the magnitude that produces 1 electron in a 1-second exposure.
        Keys are the bandpass names, values are the zeropoints.
        If None, the LSST zeropoints are used:
        {_lsstcam_zeropoint_per_sec_zenith}
    ext_coeff : dict[str, float]
        Atmospheric extinction coefficient for each bandpass.
        Keys are the bandpass names, values are the coefficients.
        If None, the LSST coefficients are used:
        {_lsstcam_extinction_coeff}

    Returns
    -------
    ndarray of float
        Flux (nJy) per electron.
    """


def calculate_zp_from_maglim(
    maglim=None,
    sky=None,
    fwhm=None,
    gain=None,
    readnoise=None,
    darkcurrent=None,
    exptime=None,
    nexposure=1,
):
    """Calculate zero points based on the 5-sigma mag limit.

    snr = flux/fluxerr
    fluxerr = sqrt(flux + sky*npix*gain + readnoise**2*nexposure*npix + darkcurrent*npix*exptime*nexposure)
    5 = flux/fluxerr
    25 = flux**2/(flux + sky*npix*Gain + readnoise**2*nexposure*npix + darkcurrent*npix*exptime*nexposure)
    flux**2 - 25*flux -25*( sky*npix*Gain
                            + readnoise**2*nexposure*npix
                            + darkcurrent*npix*exptime*nexposure)
                        = 0
    flux = 12.5 + 0.5*sqrt(625
                            + 100( sky*npix*Gain
                            + readnoise**2*nexposure*npix
                            + darkcurrent*npix*exptime*nexposure) )
    zp = 2.5log(flux) + maglim

    Parameters
    ----------
    maglim : float or ndarray
        Five-sigma magnitude limit.
    sky : float or ndarry
        Sky background in ADU/pixel.
    fwhm : float or ndarray
        PSF in pixels.
    gain : float or ndarray; default is _ztfcam_ccd_gain
        CCD gain.
    readnoise : float or ndarray; default is _ztfcam_readout_noise
        Read noise (in e-/pixel).
    darkcurrent : float or ndarray; default is _ztfcam_dark_current
        Dark current (in e-/pixel/second).
    exptime : float or ndarray
        Exposure time (in seconds).
    nexposure : int or ndarray
        Number of exposure.

    Returns
    -------
    zp: float or ndarray
        Instrument zero point (that converts 1 e- to magnitude).
    """
    npix = 2.266 * fwhm**2  # = 4 * pi * sigma**2 = pi/2/ln2 * FWHM**2
    flux_at_5sigma_limit = 12.5 + 2.5 * np.sqrt(
        25.0
        + 4.0
        * (sky * npix * gain + readnoise**2 * nexposure * npix + darkcurrent * npix * exptime * nexposure)
    )
    zp = 2.5 * np.log10(flux_at_5sigma_limit) + maglim

    return zp
