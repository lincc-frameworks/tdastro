import numpy as np
import numpy.typing as npt

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


def poisson_flux_std(
    flux: npt.ArrayLike,
    *,
    pixel_scale: npt.ArrayLike,
    total_exposure_time: npt.ArrayLike,
    exposure_count: npt.ArrayLike,
    footprint: npt.ArrayLike,
    sky: npt.ArrayLike,
    zp: npt.ArrayLike,
    readout_noise: npt.ArrayLike,
    dark_current: npt.ArrayLike,
) -> npt.ArrayLike:
    """Simulate photon noise for flux measurements.

    Parameters
    ----------
    flux : array_like of float
        Source flux in energy units, e.g. nJy.
    pixel_scale : array_like of float
        Pixel scale of the detector,
        in angular units (e.g. arcsec) per pixel.
    total_exposure_time : array_like of float
        Total exposure time of all observation, in time units
        (e.g. seconds).
    exposure_count : array_like of int
        Number of exposures in the observation.
    sky : array_like of float
        Sky background per unit angular area,
        in the units of flux / pixel_scale * pixel, e.g.
        nJy / arcsec^2.
    footprint : array_like of float
        Point spread function effective area,
        in squared angular units, e.g. arcsec^2.
    zp : array_like of float
        Zero point flux for the observation, e.g. flux
        giving a single electron during the total exposure time.
    readout_noise : array_like of float
        Standard deviation of the readout electrons per pixel per exposure.
    dark_current : array_like of float
        Mean dark current electrons per pixel per unit time.

    Returns
    -------
    array_like
        Simulated flux noise, in the same units as the input flux.

    Notes
    -----

    1. We do not specify units for the input parameters, but they
    should be consistent with each other.

    2. Here we assume that the sky and source photon noises follow
    Poisson statistics in the limit of large number of photons,
    e.g. they are both considered to be normal distributed with
    variance equal to the number of photons. Readout noise is
    assumed to be Poisson distributed with variance (squared mean)
    equal to the square of the given value. Dark current is assumed
    to be Poisson distributed with variance (squared mean) equal
    to the product of the given value and the exposure time.
    The output is Poisson standard deviation of the sum of all
    these noises converted to the flux units.
    """
    area_px = footprint / pixel_scale**2

    # Get variances, in electrons^2
    source_variance = flux / zp
    sky_variance = sky * footprint / zp
    readout_variance = readout_noise**2 * area_px * exposure_count
    dark_variance = dark_current * total_exposure_time * area_px

    total_variance = source_variance + sky_variance + readout_variance + dark_variance

    return np.sqrt(total_variance) * zp
