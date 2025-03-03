import numpy as np

from tdastro.effects.effect_model import EffectModel


class DropEffect(EffectModel):
    """An effect that drops the observation with a given probability.
    Used to model items falling in the telescope's field of view, but
    not on a CCD (e.g. chip gaps)

    Attributes
    ----------
    drop_probability : float
        The probability that an observation is dropped.
    """

    def __init__(self, drop_probability, **kwargs):
        if drop_probability < 0 or drop_probability > 1:
            raise ValueError("drop_probability must be between 0 and 1.")
        self.drop_probability = drop_probability
        super().__init__(rest_frame=False, **kwargs)

    def apply(
        self,
        flux_density,
        times=None,
        wavelengths=None,
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
        if rng_info is None:
            rng_info = np.random.default_rng()

        # Generate a random number for each observed time and drop the observation (all)
        # wavelengths for that time) if the random number is less than the drop probability.
        drop_mask1d = rng_info.uniform(size=flux_density.shape[0]) < self.drop_probability
        drop_mask2d = np.repeat(drop_mask1d[:, np.newaxis], flux_density.shape[1], axis=1)
        return np.where(drop_mask2d, 0.0, flux_density)
