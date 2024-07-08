"""The SplineModel represents SED functions as a two dimensional grid
of (time, wavelength) -> flux value that is interpolated using a 2D spline.

It is adapted from sncosmo's TimeSeriesSource model:
https://github.com/sncosmo/sncosmo/blob/v2.10.1/sncosmo/models.py
"""

from scipy.interpolate import RectBivariateSpline

from tdastro.base_models import PhysicalModel


class SplineModel(PhysicalModel):
    """A time series model defined by sample points where the intermediate
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
        amplitude : `float`, `function`, `ParameterizedModel`, or `None`
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

        # Set the attributes that can be changed (e.g. sampled)
        self.add_parameter("amplitude", amplitude, **kwargs)

        # These parameters are directly set, because they cannot be changed once
        # the object is created.
        self.name = name
        self._times = times
        self._wavelengths = wavelengths
        self._spline = RectBivariateSpline(times, wavelengths, flux, kx=time_degree, ky=wave_degree)

    def __str__(self):
        """Return the string representation of the model."""
        return f"SplineSource{self.name}"

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
        return self.amplitude * self._spline(times, wavelengths, grid=True)
