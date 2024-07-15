import numpy as np

from tdastro.effects.effect_model import EffectModel


class WhiteNoise(EffectModel):
    """A white noise model.

    Attributes
    ----------
    scale : `float`
        The scale of the noise.
    """

    def __init__(self, scale, **kwargs):
        super().__init__(**kwargs)
        self.add_parameter("scale", scale, required=True, **kwargs)

    def apply(self, flux_density, wavelengths=None, physical_model=None, **kwargs):
        """Apply the effect to observations (flux_density values)

        Parameters
        ----------
        flux_density : `numpy.ndarray`
            An array of flux density values.
        wavelengths : `numpy.ndarray`, optional
            An array of wavelengths.
        physical_model : `PhysicalModel`
            A PhysicalModel from which the effect may query parameters
            such as redshift, position, or distance.
        **kwargs : `dict`, optional
           Any additional keyword arguments.

        Returns
        -------
        flux_density : `numpy.ndarray`
            The results.
        """
        return np.random.normal(loc=flux_density, scale=self.scale)
