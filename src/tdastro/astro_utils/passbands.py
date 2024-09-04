import logging
import os
import socket
import urllib.request
from urllib.error import HTTPError, URLError

import numpy as np
import scipy.integrate
from scipy.interpolate import CubicSpline

from tdastro.astro_utils import interpolation

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class PassbandGroup:
    """A group of passbands.

    Attributes
    ----------
    passbands : dict
        A dictionary of Passband objects.
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
            self.passbands = {}
            for passband in passbands:
                self.passbands[passband.label] = passband  # This overrides any preset passbands

    def __str__(self) -> str:
        """Return a string representation of the PassbandGroup."""
        return (
            f"PassbandGroup containing {len(self.passbands)} passbands: "
            f"{', '.join([passband.full_name for passband in self.passbands.values()])}"
        )

    def _load_preset(self, preset):
        """Load a pre-defined set of passbands.

        Parameters
        ----------
        preset : str
            The name of the pre-defined set of passbands to load.
        """
        if preset == "LSST":
            self.passbands = {
                "u": Passband("LSST", "u"),
                "g": Passband("LSST", "g"),
                "r": Passband("LSST", "r"),
                "i": Passband("LSST", "i"),
                "z": Passband("LSST", "z"),
                "y": Passband("LSST", "y"),
            }
        else:
            raise ValueError(f"Unknown passband preset: {preset}")

    def set_wave_grids(self, wave_grids: dict) -> None:
        """Set the wave grid attribute for all passbands in the group.

        Parameters
        ----------
        wave_grids : dict
            A dictionary of passband labels and wave grids.
        """
        for label, wave_grid in wave_grids.items():
            self.passbands[label].set_wave_grid(wave_grid)

    def fluxes_to_bandfluxes(self, flux_density_matrix, flux_wavelengths) -> np.ndarray:
        """Calculate bandfluxes for all passbands in the group.

        Parameters
        ----------
        flux_density_matrix : np.ndarray
            A 2D array of flux densities where rows are times and columns are wavelengths.
        flux_wavelengths : np.ndarray
            An array of wavelengths in Angstroms corresponding to the columns of flux_density_matrix.
        target_grid_step : float, optional
            The unit of the grid (in Angstrom) to which the flux density matrix and normalized transmission
            table should be interpolated.

        Returns
        -------
        dict
            A dictionary of bandfluxes with passband labels as keys.
        """
        bandfluxes = {}
        for label, passband in self.passbands.items():
            bandfluxes[label] = passband.fluxes_to_bandflux(flux_density_matrix, flux_wavelengths)
        return bandfluxes


class Passband:
    """A passband contains information about its transmission curve and calculates its normalization.

    Attributes
    ----------
    label : str
        The label of the passband. This is the filter name: eg, "u".
    survey : str
        The survey to which the passband belongs: eg, "LSST".
    full_name : str
        The full name of the passband. This is the survey and label concatenated: eg, "LSST_u".
    wave_grid : np.ndarray | float | None
        The grid of wavelengths (Angstrom) to which the flux density matrix and transmission table should be
        interpolated. If a float is given, wave_grid will be converted to a numpy array with a wave_grid step
        matching the boundaries of the transmission table. If None, the transmission table will later be
        interpolated to a grid with the same boundaries, but matching the step of the flux density matrix.
        Default is 5.0.
    table_path : str
        The path to the transmission table file.
    table_url : str
        The URL to download the transmission table file.
    transmission_table : np.ndarray
        A 2D array where the first col is wavelengths (Angstrom) and the second col is transmission values.
    normalized_transmission_table : np.ndarray
        A 2D array of wavelengths and normalized transmissions.
    """

    def __init__(self, survey, label, wave_grid=5.0, table_path=None, table_url=""):
        self.label = label
        self.survey = survey
        self.full_name = f"{survey}_{label}"

        self.table_path = table_path
        self.table_url = table_url
        loaded_table = self._get_transmission_table()

        self.set_wave_grid(wave_grid, bounds=(loaded_table[0, 0], loaded_table[-1, 0]))

        self.processed_transmission_table = self._process_transmission_table(loaded_table)

    def __str__(self) -> str:
        """Return a string representation of the Passband."""
        return f"Passband: {self.full_name}"

    def set_wave_grid(self, wave_grid: np.ndarray | float | None, bounds=None) -> None:
        """Set the wave grid attribute, which is used for interpolation.

        Parameters
        ----------
        wave_grid : np.ndarray | float | None
            The grid of wavelengths (Angstrom) to which the flux density matrix and transmission table should
            be interpolated. A float will be converted to a numpy array with a grid step matching the
            boundaries of the transmission table. If None, the transmission table will later be interpolated
            to a grid maintaining the same boundaries, but matching the step of the flux density matrix.
        bounds : tuple, optional
            The lower and upper bounds of the grid. If None, the bounds will be set to the minimum and
            maximum wavelengths in the transmission table.
        """
        self.wave_grid = wave_grid
        print(f"SET WAVE GRID: {self.wave_grid}")

        if bounds is None and self.processed_transmission_table is None:
            raise ValueError("Wave grid bounds must be provided if transmission table is not loaded.")
        elif bounds is None:
            bounds = (self.processed_transmission_table[0, 0], self.processed_transmission_table[-1, 0])

        print(f"BOUNDS: {bounds}")

        # TODO should this check:
        # 1. the bounds are within the transmission table
        # 2. if the wave_grid was given as an array, but bounds were also given, that they match

        if isinstance(self.wave_grid, (float, int)):
            self.wave_grid = interpolation.create_grid(bounds, self.wave_grid)
            print(f"     TO ARRAY: {self.wave_grid}")

        print(f"     WAVEGRID: {self.wave_grid} (type: {type(self.wave_grid)})")

    def _get_transmission_table(self) -> np.ndarray:
        # Check if the table file exists locally, and download it if it doesn't
        if self.table_path is None:
            self.table_path = os.path.join(
                os.path.dirname(__file__), f"passbands/{self.survey}/{self.label}.dat"
            )
            os.makedirs(os.path.dirname(self.table_path), exist_ok=True)
        if not os.path.exists(self.table_path):
            self._download_transmission_table()

        # Load the table
        return self._load_transmission_table()

    def _download_transmission_table(self) -> bool:
        """Download a transmission table from a URL.

        Returns
        -------
        bool
            True if the download was successful, False otherwise.
        """
        if self.table_url == "":
            if self.survey == "LSST":
                # TODO consider: https://github.com/lsst/throughputs/blob/main/baseline/filter_g.dat
                # Or check with Mi's link (unless that was above)
                self.table_url = (
                    f"http://svo2.cab.inta-csic.es/svo/theory/fps3/getdata.php"
                    f"?format=ascii&id=LSST/LSST.{self.label}"
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

    def _load_transmission_table(self) -> np.ndarray:
        """Load a transmission table from a file.

        Returns
        -------
        np.ndarray
            A 2D array of wavelengths and transmissions.
        """
        # Load the table file
        loaded_table = np.loadtxt(self.table_path)

        with np.printoptions(precision=3, suppress=True):
            print(f"LOADED TABLE:\n{loaded_table}")

        # Check that the table has the correct shape
        if loaded_table.shape[1] != 2:
            raise ValueError("Transmission table must have exactly 2 columns.")

        # Check that wavelengths are strictly increasing
        if not np.all(np.diff(loaded_table[:, 0]) > 0):
            raise ValueError("Wavelengths in transmission table must be strictly increasing.")

        return loaded_table

    def _process_transmission_table(self, loaded_table) -> np.ndarray:
        """Process the transmission table: downsample or interpolate, then normalize.

        Parameters
        ----------
        loaded_table : np.ndarray
            A 2D array of wavelengths and transmissions.

        Returns
        -------
        np.ndarray
            A 2D array of wavelengths and normalized transmissions.
        """
        interpolated_transmissions = self._interpolate_or_downsample_transmission_table(loaded_table)
        normalized_transmissions = self._normalize_transmission_table(interpolated_transmissions)
        # with np.printoptions(precision=3, suppress=True):
        #     print(f"INTERPOLATED TRANSMISSIONS: {interpolated_transmissions}")
        #     print(f"NORMALIZED TRANSMISSIONS: {normalized_transmissions}")
        return normalized_transmissions

    def _interpolate_or_downsample_transmission_table(self, loaded_table) -> np.ndarray:
        """Interpolate or downsample a transmission table to a desired grid.

        Returns
        -------
        np.ndarray
            A 2D array of wavelengths and transmissions.
        """
        if self.wave_grid is not None:
            if type(self.wave_grid) == float:
                self.wave_grid = interpolation.create_grid(loaded_table[:, 0], self.wave_grid)
            interpolated_transmissions = interpolation.interpolate_transmission_table(
                loaded_table, self.wave_grid
            )
        return interpolated_transmissions

    def _normalize_transmission_table(self, transmission_table) -> np.ndarray:
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
        """
        wavelengths_angstrom = transmission_table[:, 0]
        transmissions = transmission_table[:, 1]
        # Calculate the numerators and denominator
        numerators = transmissions / wavelengths_angstrom
        denominator = scipy.integrate.trapezoid(numerators, x=wavelengths_angstrom)
        # Calculate phi_b for each wavelength
        normalized_transmissions = numerators / denominator
        return np.column_stack((wavelengths_angstrom, normalized_transmissions))

    def set_transmission_table_to_new_grid(self, new_wave_grid) -> None:
        """Set the transmission table to a new grid.

        A public wrapper for _process_transmission_table, first setting the wave grid and loading the
        unprocessed transmission table.

        Parameters
        ----------
        new_wave_grid : np.ndarray | float | None
            The grid of wavelengths (Angstrom) to which the flux density matrix and transmission table should
            be interpolated. A float will be converted to a numpy array with a grid step matching the
            boundaries of the transmission table. If None, the transmission table will later be interpolated
            to a grid maintaining the same boundaries, but matching the step of the flux density matrix.
        """
        self.wave_grid = new_wave_grid

        if self.wave_grid is None:
            return

        self.wave_grid = interpolation.create_grid(self.processed_transmission_table[:, 0], new_wave_grid)

        loaded_table = self._load_transmission_table()
        self.processed_transmission_table = self._process_transmission_table(loaded_table)

    def _interpolate_flux_densities(self, _flux_density_matrix, _flux_wavelengths) -> tuple:
        """Interpolate the flux density matrix to match the transmission table, which matches self.wave_grid.

        Parameters
        ----------
        _flux_density_matrix : np.ndarray
            A 2D array of flux densities where rows are times and columns are wavelengths.
        _flux_wavelengths : np.ndarray
            An array of wavelengths in Angstroms corresponding to the columns of _flux_density_matrix.

        Returns
        -------
        tuple
            A tuple containing the interpolated flux density matrix and interpolated wavelengths.
        """
        # Make sure all given wavelengths are strictly increasing
        if not np.all(np.diff(_flux_wavelengths) > 0):
            raise ValueError("Wavelengths corresponding to flux density matrix must be strictly increasing.")

        # Get the target grid, if it is not already set; and, convert to array.
        if self.wave_grid is None:
            if not np.allclose(np.diff(_flux_wavelengths), np.diff(_flux_wavelengths)[0]):
                raise ValueError(
                    "Flux density wavelengths must have a uniform step for interpolation. "
                    "Alternatively, provide a target grid step."
                )
            self.wave_grid = interpolation.create_grid(
                self.processed_transmission_table[:, 0], np.diff(_flux_wavelengths)[0]
            )
        elif type(self.wave_grid) == float:
            self.wave_grid = interpolation.create_grid(
                self.processed_transmission_table[:, 0], self.wave_grid
            )

        # Interpolate the flux density matrix
        if not np.array_equal(_flux_wavelengths, self.wave_grid):
            interpolated_flux_density_matrix = []
            for row in _flux_density_matrix:
                flux_density_spline = CubicSpline(
                    _flux_wavelengths, row, bc_type="not-a-knot", extrapolate=True
                )
                interpolated_flux_density_matrix.append(flux_density_spline(self.wave_grid))
            flux_density_matrix = np.array(interpolated_flux_density_matrix)
            flux_wavelengths = self.wave_grid
        else:
            flux_density_matrix = _flux_density_matrix
            flux_wavelengths = _flux_wavelengths

        return (flux_density_matrix, flux_wavelengths)

    def fluxes_to_bandflux(self, _flux_density_matrix, _flux_wavelengths) -> np.ndarray:
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

        Returns
        -------
        np.ndarray
            An array of bandfluxes.
        """
        # Interpolate flux density matrix to match the transmission table, which matches self.wave_grid
        flux_density_matrix, flux_wavelengths = self._interpolate_flux_densities(
            _flux_density_matrix, _flux_wavelengths
        )

        # Calculate the bandflux as ∫ f(λ)φ_b(λ) dλ,
        # where f(λ) is the in-band flux density and φ_b(λ) is the normalized system response
        integrand = flux_density_matrix * self.processed_transmission_table[:, 1]
        bandfluxes = scipy.integrate.trapezoid(integrand, x=flux_wavelengths)

        return bandfluxes
