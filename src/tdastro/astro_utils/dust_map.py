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

import importlib
from pkgutil import iter_modules

import astropy.units as u
import dust_extinction
import dustmaps
from astropy.coordinates import SkyCoord
from dustmaps.config import config as dm_config

from tdastro import _TDASTRO_CACHE_DATA_DIR


class DustExtinctionEffect:
    """A general dust extinction model.

    Attributes
    ----------
    dust_map : `dustmaps.DustMap` or `str`
        The dust map or its name.
    ext_model : `function` or `str`
        The extinction function to use or its name.
    """

    def __init__(self, dust_map, ext_model, **kwargs):
        if isinstance(dust_map, str):
            dust_map = DustExtinctionEffect.load_dustmap(dust_map)
        self.dust_map = dust_map

        if isinstance(ext_model, str):
            ext_model = DustExtinctionEffect.load_extinction_model(ext_model, **kwargs)
        self.extinction_model = ext_model

    @staticmethod
    def load_dustmap(name):
        """Load a dustmap from files, downloading it if needed.

        Parameters
        ----------
        name : str
            The name of the dustmap.
            Must be one of: bayestar, chen2014, csfd, edenhofer2023, iphas,
            leike_ensslin_2019, leike2020, lenz2017, marshall, pg2010, planck,
            or sfd.

        Returns
        -------
        dust_map : `dustmaps.DustMap`
            A "query" object for the requested dustmap.
        """
        # Find the correct submodule within dustmaps and load it.
        dm_module = None
        for submodule in iter_modules(dustmaps.__path__):
            if name == submodule.name:
                dm_module = importlib.import_module(f"dustmaps.{name}")
        if dm_module is None:
            raise KeyError(f"Invalid dustmap '{name}'")

        # Fetch the data to TDAstro's cache directory.
        dm_config["data_dir"] = str(_TDASTRO_CACHE_DATA_DIR / "dustmaps")
        dm_module.fetch()

        # Get the query object by searching for a class using the {Module}Query
        # naming convention.
        target_name = f"{name}query"
        query_class_name = None
        for attr in dir(dm_module):
            if attr.lower() == target_name:
                query_class_name = attr
        if query_class_name is None:
            raise ValueError(f"Unable to find query class within module dustmaps.{name}")

        # Get the class, create a query object, and return that object.
        dm_class = getattr(dm_module, query_class_name)
        return dm_class()

    @staticmethod
    def load_extinction_model(name, **kwargs):
        """Load the extinction model.

        Parameters
        ----------
        name : str
            The name of the extinction model to use.
        **kwargs : dict
            Any additional keyword arguments needed to create that argument.

        Returns
        -------
        ext_obj
            A extinction object.
        """
        for submodule in iter_modules(dust_extinction.__path__):
            ext_module = importlib.import_module(f"dust_extinction.{submodule.name}")
            if ext_module is not None and name in dir(ext_module):
                ext_class = getattr(ext_module, name)
                return ext_class(**kwargs)
        raise KeyError(f"Invalid dust extinction model '{name}'")

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
