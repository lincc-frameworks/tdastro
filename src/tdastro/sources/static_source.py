import numpy as np

from tdastro.base_models import PhysicalModel


class StaticSource(PhysicalModel):
    """A static source.

    Attributes
    ----------
    brightness : `float`
        The inherent brightness
    """

    def __init__(self, brightness, **kwargs):
        """Create a StaticSource object.

        Parameters
        ----------
        brightness : `float`, `function`, `ParameterizedModel`, or `None`
            The inherent brightness
        **kwargs : `dict`, optional
           Any additional keyword arguments.
        """
        super().__init__(**kwargs)
        self.add_parameter("brightness", brightness, required=True, **kwargs)

    def __str__(self):
        """Return the string representation of the model."""
        return "StaticSource(self.brightness)"

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
        return np.full((len(times), len(wavelengths)), self.brightness)
