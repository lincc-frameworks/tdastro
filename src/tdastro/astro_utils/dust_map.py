"""A wrapper for querying dust maps and then applying the corresponding
extinction functions.

This module is a wrapper for the following libraries:
  * dustmaps:
        Green 2018, JOSS, 3(26), 695.
        https://github.com/gregreen/dustmaps
  * dust_extinction:
        Gordon 2024, JOSS, 9(100), 7023.
        https://github.com/karllark/dust_extinction

"""

import astropy.units as u
from astropy.coordinates import SkyCoord


class DustExtinctionEffect:
    """A general dust extinction model.

    Attributes
    ----------
    dust_map : `dustmaps.DustMap`
        The dust map.
    extinction_model : `function`
        The extinction function to use.
    """

    def __init__(self, dust_map, extinction_model):
        self.dust_map = dust_map
        self.extinction_model = extinction_model

    def apply(self, flux_density, wavelengths, ra, dec, dist=1.0):
        """Apply the effect to observations (flux_density values)

        Parameters
        ----------
        flux_density : `numpy.ndarray`
            An array of flux density values (in nJy).
        wavelengths : `numpy.ndarray`, optional
            An array of wavelengths (in angstroms).
        ra : `float`
            The object's right ascension (in degrees).
        dec : `float`
            The object's declination (in degrees).
        dist : `float`
            The object's distance (in ?).
            Default = 1.0

        Returns
        -------
        flux_density : `numpy.ndarray`
            The results (in nJy).
        """
        # Get the extinction value at the object's location.
        coord = SkyCoord(ra, dec, dist, frame="icrs", unit="deg")
        ebv = self.dust_map.query(coord)

        # Do we need to convert ebv by a factor from this table:
        # https://iopscience.iop.org/article/10.1088/0004-637X/737/2/103#apj398709t6

        return flux_density * self.extinction_model.extinguish(wavelengths * u.angstrom, Ebv=ebv)
