"""The base EffectModel class used for all effects."""

from tdastro.base_models import ParameterizedNode


class EffectModel(ParameterizedNode):
    """A physical or systematic effect to apply to an observation."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def apply(self, flux_density, wavelengths=None, graph_state=None, **kwargs):
        """Apply the effect to observations (flux_density values)

        Parameters
        ----------
        flux_density : `numpy.ndarray`
            A length T X N matrix of flux density values (in nJy).
        wavelengths : `numpy.ndarray`, optional
            A length N array of wavelengths (in angstroms).
        graph_state : `GraphState`
            An object mapping graph parameters to their values.
        **kwargs : `dict`, optional
           Any additional keyword arguments.

        Returns
        -------
        flux_density : `numpy.ndarray`
            A length T x N matrix of flux densities after the effect is applied (in nJy).
        """
        raise NotImplementedError()
