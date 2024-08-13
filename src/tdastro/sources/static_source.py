import numpy as np

from tdastro.sources.physical_model import PhysicalModel


class StaticSource(PhysicalModel):
    """A static source.

    Parameters
    ----------
    brightness : `float`
        The inherent brightness
    **kwargs : `dict`, optional
        Any additional keyword arguments.
    """

    def __init__(self, brightness, **kwargs):
        super().__init__(**kwargs)
        self.add_parameter("brightness", brightness, required=True, **kwargs)

    def _evaluate(self, times, wavelengths, graph_state, **kwargs):
        """Draw effect-free observations for this object.

        Parameters
        ----------
        times : `numpy.ndarray`
            A length T array of timestamps.
        wavelengths : `numpy.ndarray`, optional
            A length N array of wavelengths.
        graph_state : `dict`
            A dictionary mapping graph parameters to their values.
        **kwargs : `dict`, optional
            Any additional keyword arguments.

        Returns
        -------
        flux_density : `numpy.ndarray`
            A length T x N matrix of SED values.
        """
        brightness = self.get_param(graph_state, "brightness")
        return np.full((len(times), len(wavelengths)), brightness)
