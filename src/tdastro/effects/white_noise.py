import numpy as np

from tdastro.effects.effect_model import EffectModel


class WhiteNoise(EffectModel):
    """A white noise model.

    Attributes
    ----------
    _rng : `numpy.random._generator.Generator`
        This object's random number generator.

    Parameters
    ----------
    scale : `float`
        The scale of the noise.
    """

    def __init__(self, scale, **kwargs):
        self._rng = np.random.default_rng()
        super().__init__(**kwargs)
        self.add_parameter("scale", scale, required=True, **kwargs)

    def _update_object_seed(self):
        """Update the object seed to the new value."""
        super()._update_object_seed()
        self._rng = np.random.default_rng()

    def apply(self, flux_density, wavelengths=None, graph_state=None, **kwargs):
        """Apply the effect to observations (flux_density values)

        Parameters
        ----------
        flux_density : `numpy.ndarray`
            An array of flux density values (in nJy).
        wavelengths : `numpy.ndarray`, optional
            An array of wavelengths (in angstroms).
        graph_state : `GraphState`
            An object mapping graph parameters to their values.
        **kwargs : `dict`, optional
           Any additional keyword arguments.

        Returns
        -------
        flux_density : `numpy.ndarray`
            The results (in nJy).
        """
        params = self.get_local_params(graph_state)
        return np.random.normal(loc=flux_density, scale=params["scale"])
