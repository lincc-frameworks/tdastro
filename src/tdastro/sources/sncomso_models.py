"""Wrappers for the models defined in sncosmo.

https://github.com/sncosmo/sncosmo/blob/v2.10.1/sncosmo/models.py
https://sncosmo.readthedocs.io/en/stable/models.html
"""

from astropy import units as u
from sncosmo.models import get_source

from tdastro.astro_utils.unit_utils import flam_to_fnu
from tdastro.sources.physical_model import PhysicalModel


class SncosmoWrapperModel(PhysicalModel):
    """A wrapper for sncosmo models.

    Attributes
    ----------
    source : `sncosmo.Source`
        The underlying source model.
    source_name : `str`
        The name used to set the source.
    source_param_names : `list`
        A list of the source model's parameters that we need to set.

    Parameters
    ----------
    source_name : `str`
        The name used to set the source.
    **kwargs : `dict`, optional
        Any additional keyword arguments.
    """

    def __init__(self, source_name, t0=0.0, **kwargs):
        super().__init__(**kwargs)
        self.source_name = source_name
        self.source = get_source(source_name)

        # Set the object's t0 as a parameter for the PhysicalModel, but
        # not the sncosmo model.
        self.add_parameter("t0", t0)

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
        name : `str`
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
        **kwargs : `dict`
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
        graph_state : `GraphState`
            An object mapping graph parameters to their values. This object is modified
            in place as it is sampled.
        seen_nodes : `dict`
            A dictionary mapping nodes seen during this sampling run to their ID.
            Used to avoid sampling nodes multiple times and to validity check the graph.
        num_samples : `int`
            A count of the number of samples to compute.
            Default: 1
        rng_info : numpy.random._generator.Generator, optional
            A given numpy random number generator to use for this computation. If not
            provided, the function uses the node's random number generator.

        Raises
        ------
        Raise a ``ValueError`` the sampling encounters a problem with the order of dependencies.
        """
        super()._sample_helper(graph_state, seen_nodes, rng_info)
        self._update_sncosmo_model_parameters(graph_state)

    def compute_flux(self, times, wavelengths, graph_state=None, **kwargs):
        """Draw effect-free observations for this object.

        Parameters
        ----------
        times : `numpy.ndarray`
            A length T array of rest frame timestamps.
        wavelengths : `numpy.ndarray`, optional
            A length N array of wavelengths (in angstroms).
        graph_state : `GraphState`
            An object mapping graph parameters to their values.
        **kwargs : `dict`, optional
           Any additional keyword arguments.

        Returns
        -------
        flux_density : `numpy.ndarray`
            A length T x N matrix of SED values (in nJy).
        """
        params = self.get_local_params(graph_state)
        self._update_sncosmo_model_parameters(graph_state)

        flux_flam = self.source.flux(times - params["t0"], wavelengths)
        flux_fnu = flam_to_fnu(
            flux_flam,
            wavelengths,
            wave_unit=u.AA,
            flam_unit=u.erg / u.second / u.cm**2 / u.AA,
            fnu_unit=u.nJy,
        )

        return flux_fnu
