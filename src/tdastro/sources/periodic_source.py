from abc import ABC, abstractmethod

from tdastro.base_models import PhysicalModel


class PeriodicSource(PhysicalModel, ABC):
    """A periodic source.

    Attributes
    ----------
    period : `float`
        The period of the source, in days.
    epoch : `float`
        The epoch of the zero phase, date.
    """

    def __init__(self, period, epoch, **kwargs):
        super().__init__(**kwargs)
        self.add_parameter("period", period, required=True, **kwargs)
        self.add_parameter("epoch", epoch, required=True, **kwargs)

    @abstractmethod
    def _evaluate_phases(self, phases, wavelengths, **kwargs):
        """Draw effect-free observations for this object, as a function of phase.

        Parameters
        ----------
        phases : `numpy.ndarray`
            A length T array of phases, in the range [0, 1].
        wavelengths : `numpy.ndarray`, optional
            A length N array of wavelengths.
        **kwargs : `dict`, optional
              Any additional keyword arguments.

        Returns
        -------
        flux_density : `numpy.ndarray`
            A length T x N matrix of SED values.
        """
        raise NotImplementedError()

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
        phases = (times - self.epoch) % self.period / self.period
        flux_density = self._evaluate_phases(phases, wavelengths, **kwargs)

        return flux_density
