import types

import numpy as np

from scipy.interpolate import RectBivariateSpline

from tdastro.base_models import PhysicalModel


class SplineModel(PhysicalModel):
    """A time series model define by sample points where the intermediate
    points are fit by a spline. Based on sncosmo's TimeSeriesSource:
    https://github.com/sncosmo/sncosmo/blob/v2.10.1/sncosmo/models.py

    Attributes
    ----------
    _times : `numpy.ndarray`
        A length T array containing the times at which the data was sampled.
    _wavelengths : `numpy.ndarray`
        A length W array containing the wavelengths at which the data was sampled.
    _spline : `RectBivariateSpline`
        The spline object for predicting the flux from a given (time, wavelength).
    name : `str`
        The name of the model being used.
    amplitude : `float`
        A unitless scaling parameter for the flux density values.
    """

    def __init__(
        self,
        times,
        wavelengths,
        flux,
        amplitude=1.0,
        time_degree=3,
        wave_degree=3,
        name=None,
        **kwargs,
    ):
        """Create the SplineModel from a grid of (timestep, wavelength, flux) points.

        Parameters
        ----------
        times : `numpy.ndarray`
            A length T array containing the times at which the data was sampled.
        wavelengths : `numpy.ndarray`
            A length W array containing the wavelengths at which the data was sampled.
        flux : `numpy.ndarray`
            A shape (T, W) matrix with flux values for each pair of time and wavelength.
            Fluxes provided in erg / s / cm^2 / Angstrom.
        amplitude : `float`
            A unitless scaling parameter for the flux density values. Default = 1.0
        time_degree : `int`
            The polynomial degree to use in the time dimension.
        wave_degree : `int`
            The polynomial degree to use in the wavelength dimension.
        name : `str`, optional
            The name of the model.
        **kwargs : `dict`, optional
           Any additional keyword arguments.
        """
        super().__init__(**kwargs)

        self.name = name
        self.amplitude = amplitude
        self._times = times
        self._wavelengths = wavelengths
        self._spline = RectBivariateSpline(times, wavelengths, flux, kx=time_degree, ky=wave_degree)

    def _evaluate(self, times, wavelengths):
        """Draw effect-free observations for this object.

        Parameters
        ----------
        times : `numpy.ndarray`
            A length N array of timestamps.
        wavelengths : `numpy.ndarray`, optional
            A length N array of wavelengths.
        **kwargs : `dict`, optional
           Any additional keyword arguments.

        Returns
        -------
        flux_density : `numpy.ndarray`
            A length N-array of flux densities.
        """
        return self.amplitude * self._spline(times, wavelengths, grid=False)
