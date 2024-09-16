"""Flux-magnitude conversion utilities."""

import numpy as np
import numpy.typing as npt

# AB definition is zp=8.9 for 1 Jy
MAG_AB_ZP_NJY = 8.9 + 2.5 * 9


def mag2flux(mag: npt.ArrayLike) -> npt.ArrayLike:
    """Convert AB magnitude to flux in nJy

    Parameters
    ----------
    mag : ndarray of `float`
        The magnitude to convert to flux.

    Returns
    -------
    flux : ndarray of `float`
        The flux corresponding to the input magnitude.
    """
    return np.power(10.0, -0.4 * (mag - MAG_AB_ZP_NJY))


def flux2mag(flux_njy: npt.ArrayLike) -> npt.ArrayLike:
    """Convert flux in nJy to AB magnitude

    Parameters
    ----------
    flux_njy : ndarray of `float`
        The flux to convert to magnitude.

    Returns
    -------
    mag : ndarray of `float`
        The magnitude corresponding to the input flux.
    """
    return MAG_AB_ZP_NJY - 2.5 * np.log10(flux_njy)
