import types

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
        brightness : `float`, `function`, or `None`
            The inherent brightness
        **kwargs : `dict`, optional
           Any additional keyword arguments.
        """
        super().__init__(**kwargs)

        if brightness is None:
            # If we were not given the parameter, use a default sampling function.
            self.brightness = np.random.rand(10.0, 20.0)
        elif isinstance(brightness, types.FunctionType):
            # If we were given a sampling function, use it.
            self.brightness = brightness(**kwargs)
        else:
            # Otherwise assume we were given the parameter itself.
            self.brightness = brightness

    def _evaluate(self, times, wavelengths=None, **kwargs):
        """Draw effect-free observations for this object.

        Parameters
        ----------
        times : `numpy.ndarray`
            A length N array of timestamps.
        wavelengths : `numpy.ndarray`, optional
            A length N array of wavelengths.
        **kwargs : `dict`, optional
           Any additional keyword arguments.

        Returns
        -------
        flux_density : `numpy.ndarray`
            A length N-array of flux densities.
        """
        return np.full_like(times, self.brightness)
