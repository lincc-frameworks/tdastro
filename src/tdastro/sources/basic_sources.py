"""A collection of toy source models that can be used for testing."""

import numpy as np

from tdastro.sources.physical_model import PhysicalModel


class StaticSource(PhysicalModel):
    """A static source.

    Parameterized values include:
      * brightness - The inherent brightness
      * dec - The object's declination in degrees. [from PhysicalModel]
      * distance - The object's luminosity distance in pc. [from PhysicalModel]
      * ra - The object's right ascension in degrees. [from PhysicalModel]
      * redshift - The object's redshift. [from PhysicalModel]
      * t0 - No effect for static model. [from PhysicalModel]

    Parameters
    ----------
    brightness : float
        The inherent brightness
    **kwargs : dict, optional
        Any additional keyword arguments.
    """

    def __init__(self, brightness, **kwargs):
        super().__init__(**kwargs)
        self.add_parameter("brightness", brightness, **kwargs)

    def compute_flux(self, times, wavelengths, graph_state, **kwargs):
        """Draw effect-free observations for this object.

        Parameters
        ----------
        times : numpy.ndarray
            A length T array of rest frame timestamps.
        wavelengths : numpy.ndarray, optional
            A length N array of wavelengths (in angstroms).
        graph_state : GraphState
            An object mapping graph parameters to their values.
        **kwargs : dict, optional
            Any additional keyword arguments.

        Returns
        -------
        flux_density : numpy.ndarray
            A length T x N matrix of SED values (in nJy).
        """
        params = self.get_local_params(graph_state)
        return np.full((len(times), len(wavelengths)), params["brightness"])


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
    brightness : float
        The inherent brightness
    t0 : float
        The time the step function starts, in MJD.
    t1 : float
        The time the step function ends, in MJD.
    **kwargs : dict, optional
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
        times : numpy.ndarray
            A length T array of rest frame timestamps.
        wavelengths : numpy.ndarray, optional
            A length N array of wavelengths (in angstroms).
        graph_state : GraphState
            An object mapping graph parameters to their values.
        **kwargs : dict, optional
           Any additional keyword arguments.

        Returns
        -------
        flux_density : numpy.ndarray
            A length T x N matrix of SED values (in nJy).
        """
        flux_density = np.zeros((len(times), len(wavelengths)))
        params = self.get_local_params(graph_state)

        time_mask = (times >= params["t0"]) & (times <= params["t1"])
        flux_density[time_mask] = params["brightness"]
        return flux_density


class SinWaveSource(PhysicalModel):
    """A source that emits a sine wave.

    flux = brightness * sin(2 * pi * frequency * (time - t0))

    Parameterized values include:
      * brightness - The inherent brightness
      * frequency - The frequence of the sine wave.
      * dec - The object's declination in degrees. [from PhysicalModel]
      * distance - The object's luminosity distance in pc. [from PhysicalModel]
      * ra - The object's right ascension in degrees. [from PhysicalModel]
      * redshift - The object's redshift. [from PhysicalModel]
      * t0 - The start of the sine wave's period. [from PhysicalModel]

    Parameters
    ----------
    brightness : float
        The inherent brightness
    frequency : float
        The frequence of the sine wave.
    **kwargs : dict, optional
        Any additional keyword arguments.
    """

    def __init__(self, brightness, frequency, **kwargs):
        super().__init__(**kwargs)
        self.add_parameter("brightness", brightness, **kwargs)
        self.add_parameter("frequency", frequency, **kwargs)

    def compute_flux(self, times, wavelengths, graph_state, **kwargs):
        """Draw effect-free observations for this object.

        Parameters
        ----------
        times : numpy.ndarray
            A length T array of rest frame timestamps.
        wavelengths : numpy.ndarray, optional
            A length N array of wavelengths (in angstroms).
        graph_state : GraphState
            An object mapping graph parameters to their values.
        **kwargs : dict, optional
            Any additional keyword arguments.

        Returns
        -------
        flux_density : numpy.ndarray
            A length T x N matrix of SED values (in nJy).
        """
        params = self.get_local_params(graph_state)
        phases = 2.0 * np.pi * params["frequency"] * (times - params["t0"])
        single_wave = params["brightness"] * np.sin(phases)
        return np.tile(single_wave[:, np.newaxis], (1, len(wavelengths)))
