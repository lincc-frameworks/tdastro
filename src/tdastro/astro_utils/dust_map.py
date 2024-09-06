"""A wrapper for querying dust maps and then applying the corresponding
extinction functions.

This module is a wrapper for the following libraries:
  * dustmaps - https://github.com/gregreen/dustmaps
  * dust_extinction - https://github.com/karllark/dust_extinction
"""


from astropy.coordinates import SkyCoord
from dustmaps.config import config
from pathlib import Path

import dustmaps

class DustExtinctionEffect():
    f"""A general dust extinction model.

    Attributes
    ----------
    dust_map : ``dustmaps.DustMap``
        The dust map.
    extinction_func : `function`
        The extinction function to use.
    r_v : `float`, optional
        The ratio of total extinction to selective extinction to pass to the
        extinction function. See: https://extinction.readthedocs.io/
        Set to ``None`` if the parameter is not used.

    Parameters
    ----------
    dust_map_name : `str`
        The name of the dustmap to use. Valid options include

    """

    def __init__(self, dust_map, extinction_type, r_v=None, **kwargs):
        """Create a dust extinction model.

        Parameters
        ----------
        dust_map : `dustmaps.DustMap`
            The dust map.
        extinction_type : `str`
            The extinction function to use. Must be one of:
            "ccm89", "odonnell94", "calzetti00", "fitzpatrick99", or "fm07"
        r_v : `float`, optional
            The ratio of total extinction to selective extinction to pass to the
            extinction function. See: https://extinction.readthedocs.io/
            Can be set to ``None`` when using "fitzpatrick99" or "fm07"
        **kwargs : `dict`, optional
            Any additional keyword arguments.
        """
        super().__init__(**kwargs)
        self.dust_map = dust_map
        self.r_v = r_v

        if extinction_type == "ccm89":
            self.extinction_func = extinction.ccm89
            if r_v is None:
                raise ValueError("r_v must be set for ccm89")
        elif extinction_type == "odonnell94":
            self.extinction_func = extinction.odonnell94
            if r_v is None:
                raise ValueError("r_v must be set for odonnell94")
        elif extinction_type == "calzetti00":
            self.extinction_func = extinction.calzetti00
            if r_v is None:
                raise ValueError("r_v must be set for calzetti00")
        elif extinction_type == "fitzpatrick99":
            self.extinction_func = extinction.fitzpatrick99
        elif extinction_type == "fm07":
            self.extinction_func = extinction.fm07


    @staticmethod
    def download_map(map_name, cache_path="../data_cache/dust_maps"):
        """Download a dust map given it's name and using the dustmaps package.

        Parameters
        ----------
        map_name : `str`
            The name of the dust map. Should be one of: std, csfd, planck
            planck_GNILC, bayestar, iphas, marshall, chen2014, lenz2017,
            pg2010, leike_ensslin_2019, 

        """
        cache_dir = Path(cache_path)
        cache_dir.mkdir(exist_ok=True, parents=True)

        if map_name == "sfd":
            import dustmaps.sfd
            dustmaps.sfd.fetch()
        elif map_name

import dustmaps.csfd
dustmaps.csfd.fetch()

import dustmaps.planck
dustmaps.planck.fetch()

import dustmaps.planck
dustmaps.planck.fetch(which='GNILC')

import dustmaps.bayestar
dustmaps.bayestar.fetch()

import dustmaps.iphas
dustmaps.iphas.fetch()

import dustmaps.marshall
dustmaps.marshall.fetch()

import dustmaps.chen2014
dustmaps.chen2014.fetch()

import dustmaps.lenz2017
dustmaps.lenz2017.fetch()

import dustmaps.pg2010
dustmaps.pg2010.fetch()

import dustmaps.leike_ensslin_2019
dustmaps.leike_ensslin_2019.fetch()

import dustmaps.leike2020
dustmaps.leike2020.fetch()

import dustmaps.edenhofer2023
dustmaps.edenhofer2023.fetch()

import dustmaps.gaia_tge
dustmaps.gaia_tge.fetch()

Path(__file__).parent / "passbands" / self.survey




from dustmaps.config import config
config['data_dir'] = '/path/to/store/maps/in'



    def apply(self, flux_density, wavelengths=None, graph_state=None, **kwargs):
        """Apply the effect to observations (flux_density values)

        Parameters
        ----------
        flux_density : `numpy.ndarray`
            An array of flux density values.
        wavelengths : `numpy.ndarray`, optional
            An array of wavelengths.
        graph_state : `GraphState`
            An object mapping graph parameters to their values.
        **kwargs : `dict`, optional
           Any additional keyword arguments.

        Returns
        -------
        flux_density : `numpy.ndarray`
            The results.
        """
        # Get the extinction value at the object's location.
        if physical_model is None:
            raise ValueError("physical_model cannot be None")
        if physical_model.distance is None:
            dist = 1.0
        else:
            dist = physical_model.distance
        coord = SkyCoord(physical_model.ra, physical_model.dec, dist, frame="icrs", unit="deg")
        ebv = self.dust_map.query(coord)

        # Apply the extinction.
        if wavelengths is None:
            raise ValueError("wavelengths cannot be None")

        if self.r_v is None:
            ext = self.extinction_func(wavelengths, ebv)
        else:
            ext = self.extinction_func(wavelengths, ebv, self.r_v)
        return extinction.apply(ext, flux_density)
