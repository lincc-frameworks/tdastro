import numpy as np

from tdastro.sources.static_source import StaticSource


class StepSource(StaticSource):
    """A static source that is on for a fixed amount of time

    Attributes
    ----------
    brightness : `float`
        The inherent brightness
    t_start : `float`
        The time the step function starts
    t_end : `float`
        The time the step function ends
    """

    def __init__(self, brightness, t_start, t_end, **kwargs):
        """Create a StaticSource object.

        Parameters
        ----------
        brightness : `float`, `function`, `ParameterizedModel`, or `None`
            The inherent brightness
        t_start : `float`
            The time the step function starts
        t_end : `float`
            The time the step function ends
        **kwargs : `dict`, optional
           Any additional keyword arguments.
        """
        super().__init__(brightness, **kwargs)
        self.add_parameter("t_start", t_start, required=True, **kwargs)
        self.add_parameter("t_end", t_end, required=True, **kwargs)

    def __str__(self):
        """Return the string representation of the model."""
        return f"StepSource({self.brightness})_{self.t_start}_to_{self.t_end}"

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

        time_mask = (times >= self.t_start) & (times <= self.t_end)
        flux_density[time_mask] = self.brightness
        return flux_density
