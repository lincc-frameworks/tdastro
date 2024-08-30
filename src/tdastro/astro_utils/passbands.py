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

    def fluxes_to_bandfluxes(
        self, flux_density_matrix, wavelengths_angstrom, target_grid_step=None
    ) -> np.ndarray:
        """Calculate bandfluxes for all passbands in the group.

        Parameters
        ----------
        flux_density_matrix : np.ndarray
            A 2D array of flux densities where rows are times and columns are wavelengths.
        wavelengths_angstrom : np.ndarray
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
            bandfluxes[label] = passband.fluxes_to_bandflux(
                flux_density_matrix, wavelengths_angstrom, target_grid_step=target_grid_step
            )
        return bandfluxes


class Passband:
    """A passband contains information about its transmission curve and calculates its normalization.

    Attributes
    ----------
    label : str
        The label of the passband.
    survey : str
        The survey to which the passband belongs.
    full_name : str
        The full name of the passband.
    table_path : str
        The path to the transmission table file.
    table_url : str
        The URL to download the transmission table file.
    transmission_table : np.ndarray
        A 2D array where the first col is wavelengths (Angstrom) and the second col is transmission values.
    normalized_transmission_table : np.ndarray
        A 2D array of wavelengths and normalized transmissions.
    """

    def __init__(self, survey, label, table_path=None, table_url=""):
        self.label = label
        self.survey = survey
        self.full_name = f"{survey}_{label}"
        self.table_path = table_path
        self.table_url = table_url
        self.transmission_table = self._load_transmission_table()
        self.normalized_transmission_table = self._normalize_transmission_table()

    def __str__(self) -> str:
        """Return a string representation of the Passband."""
        return f"Passband: {self.full_name}"

    def _load_transmission_table(self) -> np.ndarray:
        """Load a transmission table from a file or URL.

        Returns
        -------
        np.ndarray
            A 2D array where the first column is wavelengths (Angstrom) and the second column is transmission
            values.
        """
        if self.table_path is None:
            self.table_path = os.path.join(
                os.path.dirname(__file__), f"passbands/{self.survey}/{self.label}.dat"
            )
            os.makedirs(os.path.dirname(self.table_path), exist_ok=True)
        if not os.path.exists(self.table_path):
            self._download_transmission_table()
        return np.loadtxt(self.table_path)

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

    def _normalize_transmission_table(self) -> np.ndarray:
        """Calculate the value of phi_b for all wavelengths in a transmission table.

        This is eq. 8 from "On the Choice of LSST Flux Units" (Ivezić et al.):

        φ_b(λ) = S_b(λ)λ⁻¹ / ∫ S_b(λ)λ⁻¹ dλ

        where S_b(λ) is the system response of the passband.

        Notes
        -----
        - We use transmission table here to represent S_b(λ).

        Returns
        -------
        np.ndarray
            A 2D array of wavelengths and normalized transmissions.
        """
        wavelengths_angstrom = self.transmission_table[:, 0]
        transmissions = self.transmission_table[:, 1]
        # Calculate the numerators and denominator
        numerators = transmissions / wavelengths_angstrom
        denominator = scipy.integrate.trapezoid(numerators, x=wavelengths_angstrom)
        # Calculate phi_b for each wavelength
        normalized_transmissions = numerators / denominator
        return np.column_stack((wavelengths_angstrom, normalized_transmissions))

    def _interpolate(
        self, initial_fluxes, initial_wavelengths, initial_norm_trans_table, target_grid_step=None
    ):
        """Interpolate the flux density matrix and normalized transmission table to the same grid.

        Parameters
        ----------
        initial_fluxes : np.ndarray
            A 2D array of flux densities where rows are times and columns are wavelengths.
        initial_wavelengths : np.ndarray
            An array of wavelengths in Angstroms corresponding to the columns of initial_fluxes.
        initial_norm_trans_table : np.ndarray
            A 2D array of wavelengths and normalized transmissions.
        target_grid_step : int, optional
            The unit of the grid (in Angstrom) to which the flux density matrix and normalized transmission
            table should be interpolated.

        Returns
        -------
        tuple of np.ndarray
            A tuple of interpolated flux densities, wavelengths, and normalized transmission table.
        """
        if target_grid_step is None:
            # If grids already match, we can just return the intial arrays
            if np.array_equal(initial_wavelengths, initial_norm_trans_table[:, 0]):
                return (initial_fluxes, initial_wavelengths, initial_norm_trans_table)
            # Otherwise, if the given wavelength grid is uniform, set grid step to the step size
            elif interpolation.has_uniform_step(initial_wavelengths):
                target_grid_step = interpolation.get_grid_step(initial_wavelengths)
            # Otherwise, ask for target grid step
            else:
                raise ValueError(
                    "Interpolation needed; grids do not match. Consider using the target_grid_step parameter."
                )

        interp_wavelengths = initial_wavelengths
        interp_fluxes = initial_fluxes
        if target_grid_step != interpolation.get_grid_step(initial_wavelengths):
            interp_wavelengths = interpolation.interpolate_wavelengths(initial_wavelengths, target_grid_step)
            interp_fluxes = interpolation.interpolate_matrix_along_wavelengths(
                initial_fluxes, initial_wavelengths, interp_wavelengths
            )

        interp_norm_trans_table = initial_norm_trans_table
        if target_grid_step != interpolation.get_grid_step(initial_norm_trans_table[:, 0]):
            interp_norm_trans_table = interpolation.interpolate_transmission_table(
                initial_norm_trans_table, interp_wavelengths
            )

        return (interp_fluxes, interp_wavelengths, interp_norm_trans_table)

    def fluxes_to_bandflux(
        self, _flux_density_matrix, _wavelengths_angstrom, target_grid_step=None
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
        wavelengths_angstrom : np.ndarray
            An array of wavelengths in Angstroms corresponding to the columns of flux_density_matrix.
        target_grid_step : float, optional
            The unit of the grid (in Angstrom) to which the flux density matrix and normalized transmission
            table should be interpolated.

        Returns
        -------
        np.ndarray
            An array of bandfluxes.
        """
        ##### Make sure wavelengths are strictly increasing in both cases
        if not np.all(np.diff(_wavelengths_angstrom) > 0):
            raise ValueError("Wavelengths must be strictly increasing.")
        if not np.all(np.diff(self.normalized_transmission_table[:, 0]) > 0):
            raise ValueError("Wavelengths must be strictly increasing.")

        ##### Pre-interpolation information
        print(
            f"Interp {self.full_name}: {_flux_density_matrix.shape}, {_wavelengths_angstrom.shape}, "
            f"{self.normalized_transmission_table.shape} -> ",
            end="",
        )

        ##### Determine the new grid
        if target_grid_step is None:
            raise ValueError("Target grid step must be provided.")
            # TODO determine if we would want to automatically match the step of the given flux density matrix
            # Ordinarily, this would be good, but consider cases where the flux density matrix is not uniform
            # or is coarser than the transmission table
        target_grid = np.arange(
            self.normalized_transmission_table[0, 0],
            self.normalized_transmission_table[-1, 0] + target_grid_step,
            target_grid_step,
        )

        ##### Match flux density interval to target_grid interval, and interpolate if necessary
        flux_density_matrix = []
        for row in _flux_density_matrix:
            flux_density_spline = CubicSpline(
                _wavelengths_angstrom, row, bc_type="not-a-knot", extrapolate=True
            )
            flux_density_matrix.append(flux_density_spline(target_grid))
        flux_density_matrix = np.array(flux_density_matrix)
        wavelengths_angstrom = target_grid

        ##### Match transmission table to target_grid interval, and interpolate if necessary
        normalized_transmission_spline = CubicSpline(
            self.normalized_transmission_table[:, 0],
            self.normalized_transmission_table[:, 1],
            bc_type="not-a-knot",
            extrapolate=True,
        )
        interpolated_transmissions = normalized_transmission_spline(target_grid)
        normalized_transmission_table = np.column_stack((target_grid, interpolated_transmissions))

        ##### Post-interpolation information
        print(
            f"{flux_density_matrix.shape}, {wavelengths_angstrom.shape}, "
            f"{normalized_transmission_table.shape}"
        )

        # Get only the flux densities that are within the passband's wavelength range
        passband_wavelengths = normalized_transmission_table[:, 0]
        passband_wavelengths_indices = np.searchsorted(
            wavelengths_angstrom, passband_wavelengths
        )  # TODO this should now be unnecessary
        passband_flux_density_matrix = flux_density_matrix[:, passband_wavelengths_indices]

        # Calculate the bandflux as ∫ f(λ)φ_b(λ) dλ,
        # where f(λ) is the in-band flux density and φ_b(λ) is the normalized system response
        integrand = passband_flux_density_matrix * normalized_transmission_table[:, 1]
        bandfluxes = scipy.integrate.trapezoid(integrand, x=passband_wavelengths)

        return bandfluxes
