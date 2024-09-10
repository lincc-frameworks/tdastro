import logging
import os
import socket
import urllib.request
from urllib.error import HTTPError, URLError

import numpy as np
import scipy.integrate

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class PassbandGroup:
    """A group of passbands.

    Attributes
    ----------
    passbands : dict
        A dictionary of Passband objects. Note the keys are the full names of the passbands (eg, "LSST_u").
    """

    def __init__(self, preset=None, passbands=None):
        """Initialize a PassbandGroup object.

        Parameters
        ----------
        preset : str, optional
            A pre-defined set of passbands to load.
        passbands : list, optional
            A list of Passband objects assigned to the group.
        """
        self.passbands = {}

        if preset is not None:
            self._load_preset(preset)

        if passbands is not None:
            for passband in passbands:
                self.passbands[passband.full_name] = passband

    def __str__(self) -> str:
        """Return a string representation of the PassbandGroup."""
        return (
            f"PassbandGroup containing {len(self.passbands)} passbands: "
            f"{', '.join([passband.full_name for passband in self.passbands.values()])}"
        )

    def _load_preset(self, preset: str) -> None:
        """Load a pre-defined set of passbands.

        Parameters
        ----------
        preset : str
            The name of the pre-defined set of passbands to load.
        """
        if preset == "LSST":
            self.passbands = {
                "LSST_u": Passband("LSST", "u"),
                "LSST_g": Passband("LSST", "g"),
                "LSST_r": Passband("LSST", "r"),
                "LSST_i": Passband("LSST", "i"),
                "LSST_z": Passband("LSST", "z"),
                "LSST_y": Passband("LSST", "y"),
            }
        else:
            raise ValueError(f"Unknown passband preset: {preset}")

    def set_transmission_table_grids(self, wave_grid: float | int | None) -> None:
        """Set the wave grid attribute for all passbands in the group.

        Parameters
        ----------
        wave_grid : float | int | None
            The grid of wavelengths (Angstrom) to which the flux density matrix and transmission table should
            be interpolated. If a float or int is given, wave_grid will be converted to a numpy array with a
            wave_grid step matching the boundaries of the transmission table. If None, the transmission table
            will later be interpolated to a grid with the same boundaries, but matching the step of the flux
            density matrix.
        """
        for _, passband in self.passbands.items():
            passband.set_transmission_table_grid(wave_grid)

    def fluxes_to_bandfluxes(
        self,
        flux_density_matrix: np.ndarray,
        flux_wavelengths: np.ndarray,
        extrapolate_mode: str = "raise",
        spline_degree: int = 1,
    ) -> np.ndarray:
        """Calculate bandfluxes for all passbands in the group.

        Parameters
        ----------
        flux_density_matrix : np.ndarray
            A 2D array of flux densities where rows are times and columns are wavelengths.
        flux_wavelengths : np.ndarray
            An array of wavelengths in Angstroms corresponding to the columns of flux_density_matrix.
        extrapolate_mode : str, optional
            The mode of extrapolation to use, passed to scipy.interpolate.InterpolatedUnivariateSpline.
            Default is "raise", which will raise a ValueError if the interpolation goes out of bounds. Other
            options are "const" (pads with boundary value) and "zeros" (pads with 0).
        spline_degree : int, optional
            The degree of the spline to use for interpolation; must be 1 <= spline_degree <= 5. Default is 1,
            which is linear interpolation.

        Returns
        -------
        dict
            A dictionary of bandfluxes with passband names as keys.
        """
        bandfluxes = {}
        for full_name, passband in self.passbands.items():
            bandfluxes[full_name] = passband.fluxes_to_bandflux(
                flux_density_matrix,
                flux_wavelengths,
                extrapolate_mode=extrapolate_mode,
                spline_degree=spline_degree,
            )
        return bandfluxes


class Passband:
    """A passband contains information about its transmission curve and calculates its normalization.

    Attributes
    ----------
    survey : str
        The survey to which the passband belongs: eg, "LSST".
    filter_name : str
        The name of the passband's filter: eg, "u".
    full_name : str
        The full name of the passband. This is the survey and filter concatenated: eg, "LSST_u".
    _wave_grid : np.ndarray | float | None
        The grid of wavelengths (Angstrom) to which the flux density matrix and transmission table should be
        interpolated. Use the _set_wave_grid_attr method to set this value. Default step is 5.0.
    table_path : str
        The path to the transmission table file.
    table_url : str
        The URL to download the transmission table file.
    _loaded_table : np.ndarray
        A 2D array of wavelengths and transmissions. This is the table loaded from the file, and is neither
        interpolated nor normalized.
    processed_transmission_table : np.ndarray
        A 2D array where the first col is wavelengths (Angstrom) and the second col is transmission values.
        This table is both interpolated to the _wave_grid and normalized to calculate phi_b(λ).
    """

    def __init__(
        self,
        survey: str,
        filter_name: str,
        wave_grid: float = 5.0,
        table_path: str | None = None,
        table_url: str | None = None,
    ):
        """Initialize a Passband object.

        Parameters
        ----------
        survey : str
            The survey to which the passband belongs: eg, "LSST".
        filter_name : str
            The filter_name of the passband: eg, "u".
        wave_grid : np.ndarray | float | int | None, optional
            The grid of wavelengths (Angstrom) to which the flux density matrix and transmission table should
            be interpolated. A float or int will be converted to a numpy array with a grid step matching the
            boundaries of the transmission table. If None, the transmission table will later be interpolated
            to a grid maintaining the same boundaries, but matching the step of the flux density matrix.
            Default is 5.0.
        table_path : str, optional
            The path to the transmission table file. If None, the table path will be set to a default path;
            if no file exists at this location, the file will be downloaded from table_url.
        table_url : str, optional
            The URL to download the transmission table file. If None, the table URL will be set to
            a default URL based on the survey and filter_name. Default is None.
        """
        self.survey = survey
        self.filter_name = filter_name
        self.full_name = f"{survey}_{filter_name}"

        self.table_path = table_path
        self.table_url = table_url

        self._load_transmission_table()
        self._set_wave_grid_attr(wave_grid)
        self._process_transmission_table()

    def __str__(self) -> str:
        """Return a string representation of the Passband."""
        return f"Passband: {self.full_name}"

    def _load_transmission_table(self) -> None:
        """Load a transmission table from a file (or download it if it doesn't exist). Table must have 2
        columns: wavelengths and transmissions; wavelengths must be strictly increasing.

        Returns
        -------
        np.ndarray
            A 2D array of wavelengths and transmissions.
        """
        # Check if the table file exists locally, and download it if it does not
        if self.table_path is None:
            self.table_path = os.path.join(
                os.path.dirname(__file__), f"passbands/{self.survey}/{self.filter_name}.dat"
            )
            os.makedirs(os.path.dirname(self.table_path), exist_ok=True)
        if not os.path.exists(self.table_path):
            self._download_transmission_table()

        # Load the table file
        try:
            loaded_table = np.loadtxt(self.table_path)
        except OSError as e:
            raise OSError(f"Error reading transmission table from file: {e}") from e

        # Check that the table has the correct shape
        if loaded_table.size == 0 or loaded_table.shape[1] != 2:
            raise ValueError("Transmission table must have exactly 2 columns.")

        # Check that wavelengths are strictly increasing
        if not np.all(np.diff(loaded_table[:, 0]) > 0):
            raise ValueError("Wavelengths in transmission table must be strictly increasing.")

        self._loaded_table = loaded_table

    def _download_transmission_table(self) -> bool:
        """Download a transmission table from a URL.

        Returns
        -------
        bool
            True if the download was successful, False otherwise.

        Raises
        ------
        NotImplementedError
            If no table_url has been set and the survey is not supported for transmission table download.
        """
        if self.table_url is None:
            if self.survey == "LSST":
                # Consider: https://github.com/lsst/throughputs/blob/main/baseline/filter_g.dat
                # Or check with Mi's link (unless that was above)
                self.table_url = (
                    f"http://svo2.cab.inta-csic.es/svo/theory/fps3/getdata.php"
                    f"?format=ascii&id=LSST/LSST.{self.filter_name}"
                )
            else:
                raise NotImplementedError(
                    f"Transmission table download is not yet implemented for survey: {self.survey}."
                )
        try:
            socket.setdefaulttimeout(10)
            urllib.request.urlretrieve(self.table_url, self.table_path)
            if os.path.getsize(self.table_path) == 0:
                logging.error(f"Transmission table downloaded for {self.full_name} is empty.")
                return False
            else:
                logging.info(f"Downloaded: {self.full_name} transmission table.")
                return True
        except HTTPError as e:
            logging.error(f"HTTP error occurred when downloading table for {self.full_name}: {e}")
            return False
        except URLError as e:
            logging.error(f"URL error occurred when downloading table for {self.full_name}: {e}")
            return False

    def _set_wave_grid_attr(self, wave_grid: np.ndarray | float | int | None) -> None:
        """Set the wave grid attribute. Keeps _wave_grid as None or array-type, even when set with a float.

        Note: this only sets the attribute; it does NOT process transmission table. Use
        set_transmission_table_grid to set the attribute and process the transmission table in one step.

        Parameters
        ----------
        wave_grid : np.ndarray | float | int | None
            The grid of wavelengths (Angstrom) to which the flux density matrix and transmission table should
            be interpolated. A float or int will be converted to a numpy array with a grid step matching the
            boundaries of the transmission table. If None, the transmission table will later be interpolated
            to a grid maintaining the same boundaries, but matching the step of the flux density matrix.
        """
        if wave_grid is None:
            self._wave_grid = wave_grid

        elif isinstance(wave_grid, np.ndarray):
            # Check that the wave grid is valid
            if wave_grid.size <= 1:
                raise ValueError("Wave grid must have at least two values.")
            elif wave_grid.ndim != 1:
                raise ValueError("Wave grid must be a 1D array.")
            elif not np.all(np.diff(wave_grid) > 0):
                raise ValueError("Wave grid must have strictly increasing values.")

            # Truncate the wave grid to fit the transmission table boundaries
            if self._loaded_table is None:
                raise ValueError("Transmission table must be loaded before setting wave grid.")
            lower_bound, upper_bound = self._loaded_table[0, 0], self._loaded_table[-1, 0]
            lower_index = np.searchsorted(wave_grid, lower_bound, side="left")
            upper_index = np.searchsorted(wave_grid, upper_bound, side="right")
            wave_grid = wave_grid[lower_index:upper_index]
            if wave_grid.size <= 1:
                raise ValueError(
                    "Wave grid has fewer than two values after truncation to fit within the transmission "
                    "table's boundaries."
                )
            # TODO If I'm truncating here, should I also interpolate? at that point, why let them give
            # an array at all? ask about this in the morning.

            self._wave_grid = wave_grid

        elif isinstance(wave_grid, (float, int)):
            # Get the bounds of our new grid from the transmission table
            if self._loaded_table is None:
                raise ValueError("Transmission table must be loaded before setting wave grid.")
            lower_bound, upper_bound = self._loaded_table[0, 0], self._loaded_table[-1, 0]

            # Generate new grid; include the original upper bound if step size is divisor of target interval.
            # We use linspace instead of arange, as numpy recommends this for non-integer step sizes; plus,
            # arange generates values within a half-open interval, and we usually want the upper bound.
            self._wave_grid = np.linspace(
                lower_bound, upper_bound, int((upper_bound - lower_bound) / wave_grid) + 1
            )

        else:
            raise ValueError("Wave grid must be a 1D numpy array, float, int, or None.")

    def set_transmission_table_grid(self, new_wave_grid: np.ndarray | float | int | None) -> None:
        """Updates _wave_grid attr and sets the transmission table to the new grid.

        Parameters
        ----------
        new_wave_grid : np.ndarray | float | int | None
            The grid of wavelengths (Angstrom) to which the flux density matrix and transmission table should
            be interpolated. A float or int will be converted to a numpy array with a grid step matching the
            boundaries of the transmission table. If None, the transmission table will later be interpolated
            to a grid maintaining the same boundaries, but matching the step of the flux density matrix.
        """
        self._set_wave_grid_attr(new_wave_grid)
        self._process_transmission_table()

    def _process_transmission_table(self) -> None:
        """Process the transmission table: downsample or interpolate, then normalize."""
        interpolated_transmissions = self._interpolate_or_downsample_transmission_table(self._loaded_table)
        normalized_transmissions = self._normalize_interpolated_transmission_table(interpolated_transmissions)
        self.processed_transmission_table = normalized_transmissions

    def _interpolate_or_downsample_transmission_table(self, transmission_table: np.ndarray) -> np.ndarray:
        """Interpolate or downsample a transmission table to a desired grid.

        Note that if self._wave_grid is None, this method will return the original transmission table, as
        we will later interpolate the transmission table to match the flux density matrix.

        Parameters
        ----------
        transmission_table : np.ndarray
            A 2D array of wavelengths and transmissions.

        Returns
        -------
        np.ndarray
            A 2D array of wavelengths and transmissions.

        Raises
        ------
        ValueError
            If the transmission table does not have exactly 2 columns.
        """
        if self._wave_grid is None:
            return transmission_table

        if isinstance(self._wave_grid, (float, int)):
            self._set_wave_grid_attr(self._wave_grid)

        if transmission_table.shape[1] != 2:
            raise ValueError("Transmission table must have exactly 2 columns.")

        spline = scipy.interpolate.InterpolatedUnivariateSpline(
            transmission_table[:, 0], transmission_table[:, 1], ext="raise", k=1
        )
        interpolated_transmissions = spline(self._wave_grid)
        return np.column_stack((self._wave_grid, interpolated_transmissions))

    def _normalize_interpolated_transmission_table(self, transmission_table: np.ndarray) -> np.ndarray:
        """Calculate the value of phi_b for all wavelengths in a transmission table.

        This is eq. 8 from "On the Choice of LSST Flux Units" (Ivezić et al.):

        φ_b(λ) = S_b(λ)λ⁻¹ / ∫ S_b(λ)λ⁻¹ dλ

        where S_b(λ) is the system response of the passband. Note we use transmission table as our S_b(λ).

        Parameters
        ----------
        transmission_table : np.ndarray
            A 2D array of wavelengths and transmissions.

        Returns
        -------
        np.ndarray
            A 2D array of wavelengths and normalized transmissions.
        Raises
        ------
        ValueError
            If the transmission table is the wrong size or the calculated denominator is zero.
        """
        if transmission_table.size == 0:
            raise ValueError("Transmission table is empty; cannot normalize.")
        elif transmission_table.shape[0] == 1:
            raise ValueError(f"Cannot normalize transmission table with only one row: {transmission_table}.")
        elif transmission_table.ndim != 2 or transmission_table.shape[1] != 2:
            raise ValueError("Transmission table must be 2D array with exactly 2 columns.")

        wavelengths_angstrom = transmission_table[:, 0]
        transmissions = transmission_table[:, 1]
        # Calculate the numerators and denominator
        numerators = transmissions / wavelengths_angstrom
        denominator = scipy.integrate.trapezoid(numerators, x=wavelengths_angstrom)

        if denominator == 0:
            raise ValueError(
                "Denominator is zero; cannot normalize transmission table. "
                f"Consider checking the transmission table for band {self.full_name}. "
                f"With wave grid set to: {self._wave_grid}, transmission table wavelengths are: "
                f"{transmission_table[:, 0]}."
            )

        # Calculate phi_b for each wavelength
        normalized_transmissions = numerators / denominator
        return np.column_stack((wavelengths_angstrom, normalized_transmissions))

    def _interpolate_flux_densities(
        self,
        _flux_density_matrix: np.ndarray,
        _flux_wavelengths: np.ndarray,
        extrapolate_mode: str = "raise",
        spline_degree: int = 1,
    ) -> tuple:
        """Interpolate the flux density matrix to match the transmission table, which matches self._wave_grid.

        Parameters
        ----------
        _flux_density_matrix : np.ndarray
            A 2D array of flux densities where rows are times and columns are wavelengths.
        _flux_wavelengths : np.ndarray
            An array of wavelengths in Angstroms corresponding to the columns of _flux_density_matrix.
        extrapolate_mode : str, optional
            The mode of extrapolation to use, passed to scipy.interpolate.InterpolatedUnivariateSpline.
            Default is "raise", which will raise a ValueError if the interpolation goes out of bounds. Other
            options are "const" (pads with boundary value) and "zeros" (pads with 0).
        spline_degree : int, optional
            The degree of the spline to use for interpolation; must be 1 <= spline_degree <= 5. Default is 1,
            which is linear interpolation.

        Returns
        -------
        tuple
            A tuple containing the interpolated flux density matrix and interpolated wavelengths.

        Raises
        ------
        ValueError
            If the wavelengths in the flux density matrix are not strictly increasing, if the spline degree is
            invalid, or if the _wave_grid attribute is not set and the flux density matrix wavelengths (which
            would then be used to determine the grid step) do not have a uniform step.
        """
        # Make sure all given wavelengths are strictly increasing
        if not np.all(np.diff(_flux_wavelengths) > 0):
            raise ValueError("Wavelengths corresponding to flux density matrix must be strictly increasing.")

        # Make sure the degree of the spline is valid
        if (spline_degree < 1) or (spline_degree > 5):
            raise ValueError("Spline degree must be between 0 and 5.")
        elif spline_degree > len(_flux_wavelengths):
            raise ValueError(
                "Spline degree is greater than the number of wavelengths. Add more wavelengths to the flux "
                "density matrix or reduce the spline_degree parameter."
            )

        # Make sure the wave grid we're interpolating to is an array (if not None)
        if isinstance(self._wave_grid, (float, int)):
            self._set_wave_grid_attr(self._wave_grid)

        # If wave grid is None, use the flux density matrix's grid step to interpolate the transmission table
        elif self._wave_grid is None:
            if not np.allclose(np.diff(_flux_wavelengths), np.diff(_flux_wavelengths)[0]):
                raise ValueError(
                    "Flux density wavelengths must have a uniform step to interpolate transmission to match. "
                    "Alternatively, provide a target grid step with the set_transmission_table_grid method."
                )
            # Update transmission table step to match the flux density matrix step
            # (but keep the same transmission table boundaries)
            flux_grid_step = np.diff(_flux_wavelengths)[0]
            self.set_transmission_table_grid(flux_grid_step)

        # Interpolate the flux density matrix
        if not np.array_equal(_flux_wavelengths, self._wave_grid):
            # Initialize an array to store interpolated flux densities
            interpolated_flux_density_matrix = np.empty((len(_flux_density_matrix), len(self._wave_grid)))

            # Interpolate each row individually
            for i, row in enumerate(_flux_density_matrix):
                spline = scipy.interpolate.InterpolatedUnivariateSpline(
                    _flux_wavelengths, row, ext=extrapolate_mode, k=spline_degree
                )
                interpolated_flux_density_matrix[i, :] = spline(self._wave_grid)

            flux_density_matrix = interpolated_flux_density_matrix
            flux_wavelengths = self._wave_grid
        else:
            flux_density_matrix = _flux_density_matrix
            flux_wavelengths = _flux_wavelengths

        return (flux_density_matrix, flux_wavelengths)

    def fluxes_to_bandflux(
        self, _flux_density_matrix, _flux_wavelengths, extrapolate_mode="raise", spline_degree=1
    ) -> np.ndarray:
        """Calculate the bandflux for a given set of flux densities.

        This is eq. 7 from "On the Choice of LSST Flux Units" (Ivezić et al.):

        F_b = ∫ f(λ)φ_b(λ) dλ

        where f(λ) is the specific flux of an object at the top of the atmosphere, and φ_b(λ) is the
        normalized system response for given band b."

        Parameters
        ----------
        flux_density_matrix : np.ndarray
            A 2D array of flux densities where rows are times and columns are wavelengths.
        flux_wavelengths : np.ndarray
            An array of wavelengths in Angstroms corresponding to the columns of flux_density_matrix.
        extrapolate_mode : str, optional
            The mode of extrapolation to use, passed to scipy.interpolate.InterpolatedUnivariateSpline.
            Default is "raise", which will raise a ValueError if the interpolation goes out of bounds. Other
            options are "const" (pads with boundary value) and "zeros" (pads with 0).
        spline_degree : int, optional
            The degree of the spline to use for interpolation; must be 1 <= spline_degree <= 5. Default is 1,
            which is linear interpolation.

        Returns
        -------
        np.ndarray
            An array of bandfluxes.
        """
        # Interpolate flux density matrix to match the transmission table, which matches self._wave_grid
        flux_density_matrix, flux_wavelengths = self._interpolate_flux_densities(
            _flux_density_matrix, _flux_wavelengths, extrapolate_mode, spline_degree
        )

        # Calculate the bandflux as ∫ f(λ)φ_b(λ) dλ,
        # where f(λ) is the in-band flux density and φ_b(λ) is the normalized system response
        integrand = flux_density_matrix * self.processed_transmission_table[:, 1]
        bandfluxes = scipy.integrate.trapezoid(integrand, x=flux_wavelengths)

        return bandfluxes
