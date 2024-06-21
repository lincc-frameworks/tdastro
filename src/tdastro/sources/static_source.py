import numpy as np
import types

from tdastro.base_models import PhysicalModel


class StaticSource(PhysicalModel):
    def __init__(self, brightness, **kwargs):
        super().__init__(**kwargs)

        if brightness is None:
            # If we were not given the parameter, use a default sampling function.
            self.brightness = np.random.rand(10.0, 20.0)
        elif type(brightness) is types.FunctionType:
            # If we were given a sampling function, use it.
            self.brightness = brightness(**kwargs)
        else:
            # Otherwise assume we were given the parameter itself.
            self.brightness = brightness

    def _evaluate(self, times, bands=None, **kwargs):
        """Draw effect-free observations for this object.

        Parameters
        ----------
        times : `numpy.ndarray`
           An array of timestamps.
        bands : `numpy.ndarray`, optional
           An array of bands. If ``None`` then does something.

        Returns
        -------
        flux : `numpy.ndarray`
           The results.
        """
        return np.full_like(times, self.brightness)
