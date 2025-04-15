"""Functions for extrapolating flux past the end of a model's range of valid
wavelengths using flux = f(time, wavelengths).
"""

import numpy as np


class WaveExtrapolationModel:
    """The base class for the wavelength extrapolation methods."""

    def __init__(self):
        pass

    def __call__(self, last_wave, last_flux, query_waves):
        """Evaluate the extrapolation given the last valid points(s)
        and a list of new query points.

        Parameters
        ----------
        last_wave : float
            The last valid wavelength (in AA).
        last_flux : numpy.ndarray
            A length T array of the last valid flux values at each time
            at the last valid wavelength (in nJy).
        query_waves : numpy.ndarray
            A length W array of the query wavelengths (in AA) at which to extrapolate.

        Returns
        -------
        flux : numpy.ndarray
            A T x W matrix of extrapolated values.
        """
        last_flux = np.asarray(last_flux)
        num_times = last_flux.shape[0]

        query_waves = np.asarray(query_waves)
        num_waves = query_waves.shape[0]
        return np.zeros((num_times, num_waves))


class ConstantExtrapolation(WaveExtrapolationModel):
    """Extrapolate using a constant value in nJy.

    Attributes
    ----------
    value : float
        The value (in nJy) to use for the extrapolation.
    """

    def __init__(self, value=0.0):
        super().__init__()

        if value < 0:
            raise ValueError("Extrapolation value must be positive.")
        self.value = value

    def __call__(self, last_wave, last_flux, query_waves):
        """Evaluate the extrapolation given the last valid points(s)
        and a list of new query points.

        Parameters
        ----------
        last_wave : float
            The last valid wavelength (in AA).
        last_flux : numpy.ndarray
            A length T array of the last valid flux values at each time
            at the last valid wavelength (in the model's units).
        query_waves : numpy.ndarray
            A length W array of the query wavelengths (in AA) at which to extrapolate.

        Returns
        -------
        flux : numpy.ndarray
            A T x W matrix of extrapolated values.
        """
        last_flux = np.asarray(last_flux)
        num_times = last_flux.shape[0]

        query_waves = np.asarray(query_waves)
        num_waves = query_waves.shape[0]
        return np.full((num_times, num_waves), self.value)


class LastValueExtrapolation(WaveExtrapolationModel):
    """Extrapolate using the last valid value."""

    def __init__(self):
        super().__init__()

    def __call__(self, last_wave, last_flux, query_waves):
        """Evaluate the extrapolation given the last valid points(s)
        and a list of new query points.

        Parameters
        ----------
        last_wave : float
            The last valid wavelength (in AA).
        last_flux : numpy.ndarray
            A length T array of the last valid flux values at each time
            at the last valid wavelength (in nJy).
        query_waves : numpy.ndarray
            A length W array of the query wavelengths (in AA) at which to extrapolate.

        Returns
        -------
        flux : numpy.ndarray
            A T x W matrix of extrapolated values (in nJy).
        """
        last_flux = np.asarray(last_flux)
        query_waves = np.asarray(query_waves)
        num_waves = query_waves.shape[0]

        return np.tile(last_flux[:, np.newaxis], (1, num_waves))


class LinearDecay(WaveExtrapolationModel):
    """Linear decay of the flux using the last valid point(s) down to zero.

    Attributes
    ----------
    decay_width : float
        The width of the decay region in Angstroms. The flux is
        linearly decreased to zero over this range.
    """

    def __init__(self, decay_width=100.0):
        super().__init__()

        if decay_width <= 0:
            raise ValueError("decay_width must be positive.")
        self.decay_width = decay_width

    def __call__(self, last_wave, last_flux, query_waves):
        """Evaluate the extrapolation given the last valid points(s)
        and a list of new query points.

        Parameters
        ----------
        last_wave : float
            The last valid wavelength (in AA).
        last_flux : numpy.ndarray
            A length T array of the last valid flux values at each time
            at the last valid wavelength (in nJy).
        query_waves : numpy.ndarray
            A length W array of the query wavelengths (in AA) at which to extrapolate.

        Returns
        -------
        flux : numpy.ndarray
            A T x W matrix of extrapolated values (in nJy).
        """
        last_flux = np.asarray(last_flux)
        query_waves = np.asarray(query_waves)
        dist = np.abs(query_waves - last_wave)

        multiplier = np.clip(1.0 - (dist / self.decay_width), 0.0, 1.0)
        flux = last_flux[:, np.newaxis] * multiplier[np.newaxis, :]
        return flux


class ExponentialDecay(WaveExtrapolationModel):
    """Exponential decay of the flux using the last valid point(s) down to zero.

    f(t, λ) = f(t, λ_last) * exp(- rate * |λ - λ_last|)

    Attributes
    ----------
    rate : float
        The decay rate in the exponential function.
    """

    def __init__(self, rate):
        super().__init__()

        if rate < 0:
            raise ValueError("Decay rate must be zero or positive.")
        self.rate = rate

    def __call__(self, last_wave, last_flux, query_waves):
        """Evaluate the extrapolation given the last valid points(s)
        and a list of new query points.

        Parameters
        ----------
        last_wave : float
            The last valid wavelength (in AA).
        last_flux : numpy.ndarray
            A length T array of the last valid flux values at each time
            at the last valid wavelength (in nJy)
        query_waves : numpy.ndarray
            A length W array of the query wavelengths (in AA) at which to extrapolate.

        Returns
        -------
        flux : numpy.ndarray
            A T x W matrix of extrapolated values (in nJy).
        """
        last_flux = np.asarray(last_flux)
        query_waves = np.asarray(query_waves)
        dist = np.abs(query_waves - last_wave)

        multiplier = np.exp(-self.rate * dist)
        flux = last_flux[:, np.newaxis] * multiplier[np.newaxis, :]
        return flux
