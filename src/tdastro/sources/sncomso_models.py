"""Wrappers for the models defined in sncosmo.

https://github.com/sncosmo/sncosmo/blob/v2.10.1/sncosmo/models.py
https://sncosmo.readthedocs.io/en/stable/models.html
"""

from sncosmo.models import Model

from tdastro.base_models import PhysicalModel


class SncosmoModel(PhysicalModel):
    """A wrapper for sncosmo models.

    Attributes
    ----------
    model : `sncosmo.Model`
        The underlying model.
    model_name : `str`
        The name used to set the model.

    Parameters
    ----------
    model_name : `str`
        The name used to set the model.
    **kwargs : `dict`, optional
        Any additional keyword arguments.
    """

    def __init__(self, model_name, **kwargs):
        super().__init__(**kwargs)
        self.model_name = model_name
        self.model = Model(source=model_name)

    def __str__(self):
        """Return the string representation of the model."""
        return f"SncosmoModel({self.model_name})"

    @property
    def param_names(self):
        """Return a list of the model's parameter names."""
        return self.model.param_names

    @property
    def parameters(self):
        """Return a list of the model's parameter values."""
        return self.model.parameters

    @property
    def source(self):
        """Return the model's sncosmo source instance."""
        return self.model.source

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
        return self.model.get(name)

    def set(self, **kwargs):
        """Set the parameters of the model.

        These must all be constants to be compatible with sncosmo.

        Parameters
        ----------
        **kwargs : `dict`
            The parameters to set and their values.
        """
        self.model.set(**kwargs)
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
        return self.model.flux(times, wavelengths)
