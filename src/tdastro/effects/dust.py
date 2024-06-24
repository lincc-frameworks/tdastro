"""Functions for loading and querying dust maps and then applying the corresponding
extinction functions.

This module is a wrapper for the following libraries:
  * sfdmap - https://github.com/kbarbary/sfdmap
  * extinction - https://github.com/kbarbary/extinction
"""

import extinction

from tdastro.base_models import EffectModel


class DustExtinction(EffectModel):
    """A general dust extinction model.

    Attributes
    ----------
    dust_map : `sfdmap.SFDMap`
        The dust map.
    extinction_func : `function`
        The extinction function to use.
    r_v : `float`, optional
        The ratio of total extinction to selective extinction to pass to the
        extinction function. See: https://extinction.readthedocs.io/
        Set to ``None`` if the parameter is not used.
    """

    def __init__(self, dust_map, extinction_type, r_v=None, **kwargs):
        """Create a dust extinction model.

        Parameters
        ----------
        dust_map : `sfdmap.SFDMap`
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
        else:
            raise ValueError(f"Unrecognized extinction function {extinction_type}")

    def required_parameters(self):
        """Returns a list of the parameters of a PhysicalModel
        that this effect needs to access.

        Returns
        -------
        parameters : `list` of `str`
            A list of every required parameter the effect needs.
        """
        return ['ra', 'dec']

    def apply(self, flux_density, wavelengths, physical_model, **kwargs):
        """Apply the effect to observations (flux_density values)

        Parameters
        ----------
        flux_density : `numpy.ndarray`
            An array of flux density values.
        wavelengths : `numpy.ndarray`
            An array of wavelengths.
        physical_model : `PhysicalModel`
            A PhysicalModel from which the effect may query parameters
            such as redshift, position, or distance.
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
        ebv = self.dust_map.ebv(physical_model.ra, physical_model.dec)

        # Apply the extinction.
        if wavelengths is None:
            raise ValueError("wavelengths cannot be None")
        
        if r_v is None:
            ext = self.extinction_func(wavelengths, ebv)
        else:
            ext = self.extinction_func(wavelengths, ebv, self.r_v)
        return extinction.apply(ext, flux_density)
