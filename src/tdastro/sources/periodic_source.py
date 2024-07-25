from abc import ABC, abstractmethod

from tdastro.sources.physical_model import PhysicalModel


class PeriodicSource(PhysicalModel, ABC):
    """A periodic source.

    Parameters
    ----------
    period : `float`
        The period of the source, in days.
    t0 : `float`
        The t0 of the zero phase, date. Could be date of the minimum or maximum light
        or any other reference time point.
    **kwargs : `dict`, optional
        Any additional keyword arguments.
    """

    def __init__(self, period, t0, **kwargs):
        super().__init__(**kwargs)
        self.add_parameter("period", period, required=True, **kwargs)
        self.add_parameter("t0", t0, required=True, **kwargs)

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
        period = self.parameters["period"]
        phases = (times - self.parameters["t0"]) % period / period
        flux_density = self._evaluate_phases(phases, wavelengths, **kwargs)

        return flux_density
