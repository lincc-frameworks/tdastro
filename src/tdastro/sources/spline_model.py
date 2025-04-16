"""The SplineModel represents SED functions as a two dimensional grid
of (time, wavelength) -> flux value that is interpolated using a 2D spline.

It is adapted from sncosmo's TimeSeriesSource model:
https://github.com/sncosmo/sncosmo/blob/v2.10.1/sncosmo/models.py
"""

from scipy.interpolate import RectBivariateSpline

from tdastro.sources.physical_model import PhysicalModel
from tdastro.utils.io_utils import read_grid_data


class SplineModel(PhysicalModel):
    """A time series model defined by sample points where the intermediate
    points are fit by a spline. Based on sncosmo's TimeSeriesSource:
    https://github.com/sncosmo/sncosmo/blob/v2.10.1/sncosmo/models.py

    The model is defined by a 2D grid of flux values as a function of time
    and wavelength. Time is given in decimal days since t0 and wavelength
    is given in angstroms.

    Parameterized values include:
      * amplitude - A unitless scaling parameter for the flux density values.
      * dec - The object's declination in degrees. [from PhysicalModel]
      * distance - The object's luminosity distance in pc. [from PhysicalModel]
      * ra - The object's right ascension in degrees. [from PhysicalModel]
      * redshift - The object's redshift. [from PhysicalModel]
      * t0 - The time corresponding to the t=0.0 in the model data [from PhysicalModel]

    Attributes
    ----------
    _times : numpy.ndarray
        A length T array containing the times relative to t0 at which the data
        was sampled.
    _wavelengths : numpy.ndarray
        A length W array containing the wavelengths at which the data was sampled
        (in angstroms).
    _spline : RectBivariateSpline
        The spline object for predicting the flux from a given (time, wavelength).

    Parameters
    ----------
    times : numpy.ndarray
        A length T array containing the times at which the data was sampled.
    wavelengths : numpy.ndarray
        A length W array containing the wavelengths at which the data was sampled
        (in angstroms).
    flux : numpy.ndarray
        A shape (T, W) matrix with flux values for each pair of time and wavelength.
        Fluxes provided in erg / s / cm^2 / Angstrom.
    amplitude : parameter
        A unitless scaling parameter for the flux density values. Default = 1.0
    time_degree : int
        The polynomial degree to use in the time dimension.
    wave_degree : int
        The polynomial degree to use in the wavelength dimension.
    **kwargs : dict, optional
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
        **kwargs,
    ):
        super().__init__(**kwargs)

        # Set the attributes that can be changed (e.g. sampled)
        # t0 is set in the parent class.
        self.add_parameter("amplitude", amplitude, **kwargs)

        # These parameters are directly set, because they cannot be changed once
        # the object is created.
        self._times = times
        self._wavelengths = wavelengths
        self._spline = RectBivariateSpline(times, wavelengths, flux, kx=time_degree, ky=wave_degree)

    @classmethod
    def from_file(
        cls,
        input_file,
        format="ascii",
        amplitude=1.0,
        time_degree=3,
        wave_degree=3,
        **kwargs,
    ):
        """Create a SplineModel from a file where the data consists
        of three columns: time, wavelength, and the corresponding flux.

        The format of the data should be:
            time_0, wave_0, flux_0_0
            time_0, wave_1, flux_0_1
            ...
            time_0, wave_N, flux_0_N
            time_1, wave_0, flux_1_0
            time_1, wave_1, flux_1_1
            ...
            time_M, wave_N, flux_M_N
        The time and wavelength values must be sorted in ascending order.

        Parameters
        ----------
        input_file : str
            The input data file.
        format : str, optional
            The format of the input data such as 'ascii', 'ascii.ecsv', or 'fits'.
            Default = 'ascii'
        amplitude : parameter
            A unitless scaling parameter for the flux density values. Default = 1.0
        time_degree : int
            The polynomial degree to use in the time dimension.
        wave_degree : int
            The polynomial degree to use in the wavelength dimension.
        **kwargs : dict, optional
            Any additional keyword arguments.

        Returns
        -------
        SplineModel
            An instance of the SplineModel class.
        """
        times, wavelengths, flux = read_grid_data(input_file, format=format, validate=True)
        return cls(times, wavelengths, flux, amplitude, time_degree, wave_degree, **kwargs)

    def minwave(self):
        """Get the minimum wavelength of the model.

        Returns
        -------
        minwave : float or None
            The minimum wavelength of the model (in angstroms) or None
            if the model does not have a defined minimum wavelength.
        """
        return self._wavelengths[0]

    def maxwave(self):
        """Get the maximum wavelength of the model.

        Returns
        -------
        maximum : float or None
            The maximum wavelength of the model (in angstroms) or None
            if the model does not have a defined maximum wavelength.
        """
        return self._wavelengths[-1]

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
        t0 = params["t0"]
        if t0 is None:
            t0 = 0.0

        fluxes = params["amplitude"] * self._spline(times - t0, wavelengths, grid=True)
        return fluxes
