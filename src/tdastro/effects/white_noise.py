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
        self.scale = scale

    def apply(self, flux_density, rng=None, **kwargs):
        """Apply the effect to observations (flux_density values)

        Parameters
        ----------
        flux_density : numpy.ndarray
            A length T X N matrix of flux density values (in nJy).
        rng : numpy.random._generator.Generator, optional
            A random number generator to use for this evaluation. Override
            only for testing purposes.
            Default: None.
        **kwargs : `dict`, optional
           Any additional keyword arguments.

        Returns
        -------
        flux_density : numpy.ndarray
            A length T x N matrix of flux densities after the effect is applied (in nJy).
        """
        if rng is None:
            rng = np.random.default_rng()
        return rng.normal(loc=flux_density, scale=self.scale)
