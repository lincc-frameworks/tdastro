"""The SplineModel represents SED functions as a two dimensional grid
of (time, wavelength) -> flux value that is interpolated using a 2D spline.

It is adapted from sncosmo's TimeSeriesSource model:
https://github.com/sncosmo/sncosmo/blob/v2.10.1/sncosmo/models.py
"""

from scipy.interpolate import RectBivariateSpline

from tdastro.sources.physical_model import PhysicalModel


class SplineModel(PhysicalModel):
    """A time series model defined by sample points where the intermediate
    points are fit by a spline. Based on sncosmo's TimeSeriesSource:
    https://github.com/sncosmo/sncosmo/blob/v2.10.1/sncosmo/models.py

    Parameterized values include:
      * amplitude - A unitless scaling parameter for the flux density values.
      * dec - The object's declination in degrees. [from PhysicalModel]
      * distance - The object's luminosity distance in pc. [from PhysicalModel]
      * ra - The object's right ascension in degrees. [from PhysicalModel]
      * redshift - The object's redshift. [from PhysicalModel]
      * t0 - The t0 of the zero phase, date. [from PhysicalModel]

    Attributes
    ----------
    _times : `numpy.ndarray`
        A length T array containing the times at which the data was sampled.
    _wavelengths : `numpy.ndarray`
        A length W array containing the wavelengths at which the data was sampled
        (in angstroms).
    _spline : `RectBivariateSpline`
        The spline object for predicting the flux from a given (time, wavelength).
    name : `str`
        The name of the model being used.

    Parameters
    ----------
    times : `numpy.ndarray`
        A length T array containing the times at which the data was sampled.
    wavelengths : `numpy.ndarray`
        A length W array containing the wavelengths at which the data was sampled
        (in angstroms).
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
        super().__init__(**kwargs)

        # Set the attributes that can be changed (e.g. sampled)
        self.add_parameter("amplitude", amplitude, **kwargs)

        # These parameters are directly set, because they cannot be changed once
        # the object is created.
        self._times = times
        self._wavelengths = wavelengths
        self._spline = RectBivariateSpline(times, wavelengths, flux, kx=time_degree, ky=wave_degree)

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
        params = self.get_local_params(graph_state)
        return params["amplitude"] * self._spline(times, wavelengths, grid=True)
