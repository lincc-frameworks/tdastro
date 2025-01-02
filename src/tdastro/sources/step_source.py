import numpy as np

from tdastro.sources.static_source import StaticSource


class StepSource(StaticSource):
    """A static source that is on for a fixed amount of time

    Parameterized values include:
      * brightness - The inherent brightness
      * dec - The object's declination in degrees. [from PhysicalModel]
      * distance - The object's luminosity distance in pc. [from PhysicalModel]
      * ra - The object's right ascension in degrees. [from PhysicalModel]
      * redshift - The object's redshift. [from PhysicalModel]
      * t0 - The time the step function starts, in MJD.
      * t1- The time the step function ends, in MJD.

    Parameters
    ----------
    brightness : `float`
        The inherent brightness
    t0 : `float`
        The time the step function starts, in MJD.
    t1 : `float`
        The time the step function ends, in MJD.
    **kwargs : `dict`, optional
        Any additional keyword arguments.
    """

    def __init__(self, brightness, t1, **kwargs):
        super().__init__(brightness, **kwargs)

        # t0 is added in the PhysicalModel constructor.
        self.add_parameter("t1", t1, **kwargs)

    def compute_flux(self, times, wavelengths, graph_state, **kwargs):
        """Draw effect-free observations for this object.

        Parameters
        ----------
        times : `numpy.ndarray`
            A length T array of rest frame timestamps.
        wavelengths : `numpy.ndarray`, optional
            A length N array of wavelengths (in angstroms).
        graph_state : `GraphState`
            An object mapping graph parameters to their values.
        **kwargs : `dict`, optional
           Any additional keyword arguments.

        Returns
        -------
        flux_density : `numpy.ndarray`
            A length T x N matrix of SED values (in nJy).
        """
        flux_density = np.zeros((len(times), len(wavelengths)))
        params = self.get_local_params(graph_state)

        time_mask = (times >= params["t0"]) & (times <= params["t1"])
        flux_density[time_mask] = params["brightness"]
        return flux_density
