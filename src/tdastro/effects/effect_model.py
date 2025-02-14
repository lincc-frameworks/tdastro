"""The base EffectModel class used for all effects."""


class EffectModel:
    """A physical or systematic effect to apply to an observation."""

    def __init__(self, **kwargs):
        pass

    def apply(self, flux_density, **kwargs):
        """Apply the effect to observations (flux_density values)

        Parameters
        ----------
        flux_density : numpy.ndarray
            A length T X N matrix of flux density values (in nJy).
        **kwargs : `dict`, optional
           Any additional keyword arguments.

        Returns
        -------
        flux_density : numpy.ndarray
            A length T x N matrix of flux densities after the effect is applied (in nJy).
        """
        raise NotImplementedError()
