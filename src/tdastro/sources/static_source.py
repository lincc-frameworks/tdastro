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
            A length T array of rest frame timestamps.
        wavelengths : `numpy.ndarray`, optional
            A length N array of wavelengths.
        graph_state : `GraphState`
            An object mapping graph parameters to their values.
        **kwargs : `dict`, optional
            Any additional keyword arguments.

        Returns
        -------
        flux_density : `numpy.ndarray`
            A length T x N matrix of SED values.
        """
        params = self.get_local_params(graph_state)
        return np.full((len(times), len(wavelengths)), params["brightness"])
