import numpy as np

from tdastro.sources.physical_model import PhysicalModel


class StaticSource(PhysicalModel):
    """A static source.

    Attributes
    ----------
    brightness : `float`
        The inherent brightness
    """

    def __init__(self, brightness, **kwargs):
        super().__init__(**kwargs)
        self.add_parameter("brightness", brightness, required=True, **kwargs)

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
        return np.full((len(times), len(wavelengths)), self.parameters["brightness"])
