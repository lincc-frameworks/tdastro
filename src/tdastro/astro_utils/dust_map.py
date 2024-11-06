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
from astropy.coordinates import SkyCoord
from dustmaps.config import config as dm_config

from tdastro import _TDASTRO_CACHE_DATA_DIR


class DustExtinctionEffect:
    """A general dust extinction model.

    Attributes
    ----------
    dust_map : dustmaps.DustMap or str
        The dust map or its name. Since different dustmap's query function
        may produce different outputs, you should include the corresponding
        ebv_func to transform the result into ebv if needed.
    extinction_model : function or str
        The extinction object from the dust_extinction library or its name.
        If a string is provided, the code will find a matching extinction
        function in the dust_extinction package and use that.
    ebv_func : function
        A function to translate the result of the dustmap query into an ebv.
    **kwargs : `dict`, optional
        Any additional keyword arguments.
    """

    def __init__(self, dust_map, extinction_model, ebv_func=None, **kwargs):
        self.ebv_func = ebv_func

        if isinstance(dust_map, str):
            # Initially we only support loading the SFD dustmap by string.
            # But we can expand this as needed.
            dust_map = dust_map.lower()
            if dust_map == "sfd":
                self._load_sfd_dustmap()
            else:
                raise ValueError("Unsupported load from dustmap {dust_map}")
        else:
            # If given a dustmap, use that directly.
            self.dust_map = dust_map

        if isinstance(extinction_model, str):
            extinction_model = DustExtinctionEffect.load_extinction_model(extinction_model, **kwargs)
        self.extinction_model = extinction_model

    def _load_sfd_dustmap(self):
        """Load the SFD dustmap, downloading it if needed.

        Uses data from:
        1.  Schlegel, Finkbeiner, and Davis
            The Astrophysical Journal, Volume 500, Issue 2, pp. 525-553.
            https://ui.adsabs.harvard.edu/abs/1998ApJ...500..525S/abstract

        2.  Schlegel and Finkbeiner
            The Astrophysical Journal, Volume 737, Issue 2, article id. 103, 13 pp. (2011).
            https://ui.adsabs.harvard.edu/abs/2011ApJ...737..103S/abstract

        Returns
        -------
        sfd_query : SFDQuery
            The "query" object for the requested dustmap.
        """
        import dustmaps.sfd

        # Download the dustmap if needed.
        dm_config["data_dir"] = str(_TDASTRO_CACHE_DATA_DIR / "dustmaps")
        dustmaps.sfd.fetch()

        # Load the dustmap.
        from dustmaps.sfd import SFDQuery

        self.dust_map = SFDQuery()

        # Add the correction function.
        def _sfd_scale_ebv(input, **kwargs):
            """Scale the result of the SFD query."""

        self.ebv_func = _sfd_scale_ebv

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

    def apply(self, flux_density, wavelengths, ebv=None, ra=None, dec=None, dist=None, **kwargs):
        """Apply the effect to observations (flux_density values). The user can either
        provide a ebv value directly or (RA, dec) and distance information that can
        be used in the dustmap query.

        Parameters
        ----------
        flux_density : numpy.ndarray
            An array of flux density values (in nJy).
        wavelengths : numpy.ndarray, optional
            An array of wavelengths (in angstroms).
        ebv : float or np.array
            A given ebv value or array of values. If present then this is used
            instead of looking it up in the dust map.
        ra : float, optional
            The object's right ascension (in degrees).
        dec : float, optional
            The object's declination (in degrees).
        dist : float, optional
            The object's distance (in parsecs).
        **kwargs : `dict`, optional
            Any additional keyword arguments.

        Returns
        -------
        flux_density : numpy.ndarray
            The results (in nJy).
        """
        if ebv is None:
            if self.dust_map is None:
                raise ValueError("If ebv=None then a dust map must be provided.")
            if ra is None or dec is None:
                raise ValueError("If ebv=None then ra, dec must be provided for a lookup.")

            # Get the extinction value at the object's location.
            if dist is not None:
                coord = SkyCoord(ra, dec, dist, frame="icrs", unit="deg")
            else:
                coord = SkyCoord(ra, dec, frame="icrs", unit="deg")
            dustmap_value = self.dust_map.query(coord)

            # Perform any corrections needed for this dust map.
            if self.ebv_func is not None:
                ebv = self.ebv_func(dustmap_value, **kwargs)
            else:
                ebv = dustmap_value

        print(f"Using ebv={ebv}")

        return flux_density * self.extinction_model.extinguish(wavelengths * u.angstrom, Ebv=ebv)
