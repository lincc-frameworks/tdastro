import numpy as np

from tdastro.sources.static_source import StaticSource


class StepSource(StaticSource):
    """A static source that is on for a fixed amount of time

    Attributes
    ----------
    brightness : `float`
        The inherent brightness
    t0 : `float`
        The time the step function starts
    t1 : `float`
        The time the step function ends
    """

    def __init__(self, brightness, t0, t1, **kwargs):
        super().__init__(brightness, **kwargs)
        self.add_parameter("t0", t0, required=True, **kwargs)
        self.add_parameter("t1", t1, required=True, **kwargs)

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
        flux_density = np.zeros((len(times), len(wavelengths)))

        time_mask = (times >= self.t0) & (times <= self.t1)
        flux_density[time_mask] = self.brightness
        return flux_density
