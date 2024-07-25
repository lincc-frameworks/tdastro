"""Wrappers for the models defined in sncosmo.

https://github.com/sncosmo/sncosmo/blob/v2.10.1/sncosmo/models.py
https://sncosmo.readthedocs.io/en/stable/models.html
"""

from sncosmo.models import get_source

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

    def __init__(self, source_name, **kwargs):
        super().__init__(**kwargs)
        self.source_name = source_name
        self.source = get_source(source_name)

        # Use the kwargs to initialize the sncosmo model's parameters.
        self.source_param_names = []
        for key, value in kwargs.items():
            if not hasattr(self, key):
                self.add_parameter(key, value)
            if key in self.source.param_names:
                self.source_param_names.append(key)
        self._update_sncosmo_model_parameters()

    @property
    def param_names(self):
        """Return a list of the model's parameter names."""
        return self.source.param_names

    @property
    def parameter_values(self):
        """Return a list of the model's parameter values."""
        return self.source.parameters

    def _update_sncosmo_model_parameters(self):
        """Update the parameters for the wrapped sncosmo model."""
        params = {}
        for name in self.source_param_names:
            params[name] = getattr(self, name)
        self.source.set(**params)

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
        self._update_sncosmo_model_parameters()

    def _sample_helper(self, depth, seen_nodes, **kwargs):
        """Internal recursive function to sample the model's underlying parameters
        if they are provided by a function or ParameterizedNode.

        Calls ParameterNode's _sample_helper() then updates the parameters
        for the sncosmo model.

        Parameters
        ----------
        depth : `int`
            The recursive depth remaining. Used to prevent infinite loops.
            Users should not need to set this manually.
        seen_nodes : `dict`
            A dictionary mapping nodes seen during this sampling run to their ID.
            Used to avoid sampling nodes multiple times and to validity check the graph.
        **kwargs : `dict`, optional
            All the keyword arguments, including the values needed to sample
            parameters.

        Raises
        ------
        Raise a ``ValueError`` the depth of the sampling encounters a problem
        with the order of dependencies.
        """
        super()._sample_helper(depth, seen_nodes, **kwargs)
        self._update_sncosmo_model_parameters()

    def _evaluate(self, times, wavelengths, **kwargs):
        """Draw effect-free observations for this object.

        Parameters
        ----------
        times : `numpy.ndarray`
            A length T array of timestamps.
        wavelengths : `numpy.ndarray`, optional
            A length N array of wavelengths.
        **kwargs : `dict`, optional
           Any additional keyword arguments.

        Returns
        -------
        flux_density : `numpy.ndarray`
            A length T x N matrix of SED values.
        """
        return self.source.flux(times, wavelengths)
