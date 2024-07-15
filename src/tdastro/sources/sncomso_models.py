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

    @property
    def param_names(self):
        """Return a list of the model's parameter names."""
        return self.source.param_names

    @property
    def parameters(self):
        """Return a list of the model's parameter values."""
        return self.source.parameters

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
        self.source.set(**kwargs)
        for key, value in kwargs.items():
            if hasattr(self, key):
                self.set_parameter(key, value)
            else:
                self.add_parameter(key, value)

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
