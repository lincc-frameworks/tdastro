import numpy as np

from tdastro.effects.effect_model import EffectModel


class WhiteNoise(EffectModel):
    """A white noise model.

    Attributes
    ----------
    white_noise_sigma : parameter
        The scale of the noise.
    """

    def __init__(self, white_noise_sigma, **kwargs):
        super().__init__(**kwargs)
        self.add_effect_parameter("white_noise_sigma", white_noise_sigma)

    def apply(
        self,
        flux_density,
        times=None,
        wavelengths=None,
        white_noise_sigma=None,
        rng_info=None,
        **kwargs,
    ):
        """Apply the effect to observations (flux_density values).

        Parameters
        ----------
        flux_density : numpy.ndarray
            A length T X N matrix of flux density values (in nJy).
        times : numpy.ndarray, optional
            A length T array of times (in MJD). Not used for this effect.
        wavelengths : numpy.ndarray, optional
            A length N array of wavelengths (in angstroms). Not used for this effect.
        white_noise_sigma : float, optional
            The scale of the noise. Raises an error if None is provided.
        rng_info : numpy.random._generator.Generator, optional
            A given numpy random number generator to use for this computation. If not
            provided, the function uses the node's random number generator.
        **kwargs : `dict`, optional
           Any additional keyword arguments. This includes all of the
           parameters needed to apply the effect.

        Returns
        -------
        flux_density : numpy.ndarray
            A length T x N matrix of flux densities after the effect is applied (in nJy).
        """
        if white_noise_sigma is None:
            raise ValueError("white_noise_sigma must be provided")

        if rng_info is None:
            rng_info = np.random.default_rng()
        return rng_info.normal(loc=flux_density, scale=white_noise_sigma)
