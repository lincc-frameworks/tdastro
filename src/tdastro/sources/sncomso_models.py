"""Wrappers for the models defined in sncosmo.

https://github.com/sncosmo/sncosmo/blob/v2.10.1/sncosmo/models.py
https://sncosmo.readthedocs.io/en/stable/models.html
"""

import numpy as np
from astropy import units as u
from citation_compass import CiteClass
from sncosmo.models import get_source

from tdastro.astro_utils.unit_utils import flam_to_fnu
from tdastro.sources.physical_model import PhysicalModel


class SncosmoWrapperModel(PhysicalModel, CiteClass):
    """A wrapper for sncosmo models.

    Parameterized values include:
      * dec - The object's declination in degrees. [from PhysicalModel]
      * distance - The object's luminosity distance in pc. [from PhysicalModel]
      * ra - The object's right ascension in degrees. [from PhysicalModel]
      * redshift - The object's redshift. [from PhysicalModel]
      * t0 - The t0 of the zero phase, date. [from PhysicalModel]
    Additional parameterized values are used for specific sncosmo models.

    References
    ----------
    * sncosmo - https://zenodo.org/records/14714968
    * Individual models might require citation. See references in the sncosmo documentation.

    Attributes
    ----------
    source : sncosmo.Source
        The underlying source model.
    source_name : str
        The name used to set the source.
    source_param_names : list
        A list of the source model's parameters that we need to set.
    wave_extrapolation : WaveExtrapolationModel, optional
        The extrapolation model to use for wavelengths that fall outside
        the sncosmo model's bounds.  If None then the model will use all zeros.

    Parameters
    ----------
    source_name : str
        The name used to set the source.
    node_label : str, optional
        An identifier (or name) for the current node.
    wave_extrapolation : WaveExtrapolationModel, optional
        The extrapolation model to use for wavelengths that fall outside
        the sncosmo model's bounds.  If None then the model will use all zeros
    **kwargs : dict, optional
        Any additional keyword arguments.
    """

    # A class variable for the units so we are not computing them each time.
    _FLAM_UNIT = u.erg / u.second / u.cm**2 / u.AA

    def __init__(
        self,
        source_name,
        node_label=None,
        wave_extrapolation=None,
        **kwargs,
    ):
        super().__init__(node_label=node_label, **kwargs)
        self.source_name = source_name
        self.source = get_source(source_name)

        # Set the extrapolation for values outside the sncosmo bounds.
        self.wave_extrapolation = wave_extrapolation

        # Use the kwargs to initialize the sncosmo model's parameters.
        self.source_param_names = []
        for key, value in kwargs.items():
            if key not in self.setters:
                self.add_parameter(key, value)
            if key in self.source.param_names:
                self.source_param_names.append(key)

    @property
    def param_names(self):
        """Return a list of the model's parameter names."""
        return self.source.param_names

    @property
    def parameter_values(self):
        """Return a list of the model's parameter values."""
        return self.source.parameters

    def _update_sncosmo_model_parameters(self, graph_state):
        """Update the parameters for the wrapped sncosmo model."""
        local_params = graph_state.get_node_state(self.node_string, 0)
        sn_params = {}
        for name in self.source_param_names:
            sn_params[name] = local_params[name]
        self.source.set(**sn_params)

    def get(self, name):
        """Get the value of a specific parameter.

        Parameters
        ----------
        name : str
            The name of the parameter.

        Returns
        -------
        The parameter value.
        """
        return self.source.get(name)

    def set(self, **kwargs):
        """Set the parameters of the model.

        These must all be constants to be compatible with sncosmo.

        Parameters
        ----------
        **kwargs : dict
            The parameters to set and their values.
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                self.set_parameter(key, value)
            else:
                self.add_parameter(key, value)
            if key not in self.source_param_names:
                self.source_param_names.append(key)
        self.source.set(**kwargs)

    def _sample_helper(self, graph_state, seen_nodes, num_samples=1, rng_info=None):
        """Internal recursive function to sample the model's underlying parameters
        if they are provided by a function or ParameterizedNode.

        Calls ParameterNode's _sample_helper() then updates the parameters
        for the sncosmo model.

        Parameters
        ----------
        graph_state : GraphState
            An object mapping graph parameters to their values. This object is modified
            in place as it is sampled.
        seen_nodes : dict
            A dictionary mapping nodes seen during this sampling run to their ID.
            Used to avoid sampling nodes multiple times and to validity check the graph.
        num_samples : int
            A count of the number of samples to compute.
            Default: 1
        rng_info : numpy.random._generator.Generator, optional
            A given numpy random number generator to use for this computation. If not
            provided, the function uses the node's random number generator.

        Raises
        ------
        Raise a ValueError the sampling encounters a problem with the order of dependencies.
        """
        super()._sample_helper(graph_state, seen_nodes, rng_info)
        self._update_sncosmo_model_parameters(graph_state)

    def mask_by_time(self, times, graph_state=None):
        """Compute a mask for whether a given time is of interest for a given object.
        For example, a user can use this function to generate a mask to include
        only the observations of interest for a window around the supernova.

        Parameters
        ----------
        times : numpy.ndarray
            A length T array of observer frame timestamps in MJD.
        graph_state : GraphState, optional
            An object mapping graph parameters to their values.

        Returns
        -------
        time_mask : numpy.ndarray
            A length T array of Booleans indicating whether the time is of interest.
        """
        if graph_state is None:
            raise ValueError("graph_state needed to compute mask_by_time")

        z = self.get_param(graph_state, "redshift", 0.0)
        if z is None:
            z = 0.0

        t0 = self.get_param(graph_state, "t0", 0.0)
        if t0 is None:
            t0 = 0.0

        # Compute the mask.
        good_times = (times > t0 + self.source.minphase() * (1.0 + z)) & (
            times < t0 + self.source.maxphase() * (1.0 + z)
        )
        return good_times

    def compute_flux(self, times, wavelengths, graph_state=None, **kwargs):
        """Draw effect-free observations for this object.

        Parameters
        ----------
        times : numpy.ndarray
            A length T array of rest frame timestamps.
        wavelengths : numpy.ndarray, optional
            A length N array of wavelengths (in angstroms).
        graph_state : GraphState
            An object mapping graph parameters to their values.
        **kwargs : dict, optional
           Any additional keyword arguments.

        Returns
        -------
        flux_density : numpy.ndarray
            A length T x N matrix of SED values (in nJy).
        """
        params = self.get_local_params(graph_state)
        self._update_sncosmo_model_parameters(graph_state)

        # sncosmo gives an error if the wavelengths are out of bounds, so we truncate for the actual call.
        before_mask = wavelengths < self.source.minwave()
        after_mask = wavelengths > self.source.maxwave()
        in_range = ~before_mask & ~after_mask

        # Pad the wavelengths with the min and max values and compute the flux (flam) at those points.
        query_waves = np.concatenate(
            ([self.source.minwave()], wavelengths[in_range], [self.source.maxwave()])
        )
        model_flam = self.source.flux(times - params["t0"], query_waves)

        # Convert the flux to the desired units. We only bother converting the
        # wavelengths that are in range.
        model_fnu = flam_to_fnu(
            model_flam,
            query_waves,
            wave_unit=u.AA,
            flam_unit=self._FLAM_UNIT,
            fnu_unit=u.nJy,
        )

        # Initially zero pad the fnu array until we fill in the values with extrapolation.
        flux_fnu = np.zeros((len(times), len(wavelengths)))
        flux_fnu[:, in_range] = model_fnu[:, 1:-1]

        # Do extrapolation for wavelengths that fell outside the model's bounds
        # using the fnu values. The endpoints of model_fnu are the first and last
        # valid values for the model.
        if self.wave_extrapolation is not None:
            # Compute the flux values before the model's first valid wavelength.
            flux_fnu[:, before_mask] = self.wave_extrapolation(
                self.source.minwave(),
                model_fnu[:, 0],
                wavelengths[before_mask],
            )

            # Compute the flux values after the model's last valid wavelength.
            flux_fnu[:, after_mask] = self.wave_extrapolation(
                self.source.maxwave(),
                model_fnu[:, -1],
                wavelengths[after_mask],
            )

        return flux_fnu
