"""A wrapper for querying dust maps and returning E(B-V) values.

This module supports a variety of external dustmaps libaries, but
was primarily designed to work with the dustmaps package:
https://github.com/gregreen/dustmaps
"""

import numpy as np
from astropy.coordinates import SkyCoord

from tdastro.base_models import FunctionNode


class DustEBV(FunctionNode):
    """A wrapper that queries a dust map and returns the ebv for each location.

    Attributes
    ----------
    query_fn : function
        The function to query the dust map ebv value given (RA, dec).

    Parameters
    ----------
    query_fn : function
        The function to query the dust map ebv value given (RA, dec).
    **kwargs : `dict`, optional
        Any additional keyword arguments.
    """

    def __init__(self, query_fn, **kwargs):
        super().__init__(query_fn, outputs=["ebv"], **kwargs)


class DustmapWrapper(DustEBV):
    """A convenience wrapper for the dustmap package
    https://github.com/gregreen/dustmaps

    Citation: Green 2018, JOSS, 3(26), 695.

    Parameters
    ----------
    dust_map : DustMap object
        The dust map. Since different dustmap's query function may produce different
        outputs, you should include the corresponding ebv_func to transform the result
        into ebv if needed.
    ebv_func : function, optional
        A function to translate the result of the dustmap query into an ebv.
    **kwargs : `dict`, optional
        Any additional keyword arguments.
    """

    def __init__(self, dust_map, ebv_func=None, **kwargs):
        super().__init__(self.compute_ebv, **kwargs)
        self._ebv_func = ebv_func
        self._dust_map = dust_map

    def compute_ebv(self, ra, dec):
        """Compute the E(B-V) value for a given location.

        Parameters
        ----------
        ra : float or np.array
            The object's right ascension (in degrees).
        dec : float or np.array
            The object's declination (in degrees).

        Returns
        -------
        ebv : float or np.array
            The E(B-V) value or array of values.
        """
        if self._dust_map is None:
            raise ValueError("A dust map must be provided to compute E(B-V) values.")

        # Wrap the RA and dec in a SkyCoord object
        coord = SkyCoord(ra, dec, frame="icrs", unit="deg")
        dustmap_value = self._dust_map.query(coord)

        if self._ebv_func is not None:
            return self._ebv_func(dustmap_value)
        return dustmap_value


class ConstantHemisphereDustMap(DustEBV):
    """A DustMap with a constant value in each hemisphere.

    Attributes
    ----------
    north_ebv : `float`
        The DustMap's ebv value at all points in the Northern Hemisphere.
    south_ebv : `float`
        The DustMap's ebv value at all points in the Southern Hemisphere.
    """

    def __init__(self, north_ebv, south_ebv, **kwargs):
        super().__init__(self.compute_ebv, **kwargs)
        self.north_ebv = north_ebv
        self.south_ebv = south_ebv

    def compute_ebv(self, ra, dec):
        """Compute the E(B-V) value for a given location.

        Parameters
        ----------
        ra : float or np.array
            The object's right ascension (in degrees).
        dec : float or np.array
            The object's declination (in degrees).

        Returns
        -------
        ebv : float or np.array
            The E(B-V) value or array of values.
        """
        if np.isscalar(ra):
            return self.north_ebv if dec >= 0 else self.south_ebv

        ebv = np.where(np.asarray(dec) < 0, self.south_ebv, self.north_ebv)
        return ebv

    def query(self, coords):
        """A query function to match the DustMap interface so that
        we can pass ConstantHemisphereDustMap into DustmapWrapper
        for testing purposes.

        Note
        ----
        This shouldn't be used in production code. Use the compute_ebv()
        function directly.

        Parameters
        ----------
        coords : SkyCoord
            The object's coordinates.

        Returns
        -------
        ebv : float or np.array
            The E(B-V) value or array of values.
        """
        return self.compute_ebv(coords.ra.deg, coords.dec.deg)
