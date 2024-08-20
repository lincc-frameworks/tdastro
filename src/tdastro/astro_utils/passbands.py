import logging
import os
import socket
import urllib.request
from urllib.error import HTTPError, URLError

import numpy as np
import scipy.integrate

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

    def fluxes_to_bandfluxes(self, flux_density_matrix, wavelengths_angstrom) -> np.ndarray:
        """Calculate bandfluxes for all passbands in the group.

        Parameters
        ----------
        flux_density_matrix : np.ndarray
            A 2D array of flux densities where rows are times and columns are wavelengths.
        wavelengths_angstrom : np.ndarray
            An array of wavelengths in Angstroms corresponding to the columns of flux_density_matrix.

        Returns
        -------
        dict
            A dictionary of bandfluxes with passband labels as keys.
        """
        bandfluxes = {}
        for label, passband in self.passbands.items():
            bandfluxes[label] = passband.fluxes_to_bandflux(flux_density_matrix, wavelengths_angstrom)
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

    def fluxes_to_bandflux(
        self, flux_density_matrix, wavelengths_angstrom, interpolate_to_grid=None
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
        interpolate_to_grid : int, optional
            The unit of the grid (in Angstrom) to which the flux density matrix and normalized transmission
            table should be interpolated.

        Returns
        -------
        np.ndarray
            An array of bandfluxes.
        """
        n_t_table = self.normalized_transmission_table

        # Interpolate the flux density matrix and normalized transmission table to the same grid, if needed
        if interpolate_to_grid is not None:
            wavelengths_angstrom = interpolation.to_grid(
                interpolate_to_grid, wavelengths_angstrom, flux_density_matrix
            )

        elif not interpolation.grids_have_same_step(wavelengths_angstrom, n_t_table[:, 0]):
            raise ValueError(
                f"Interpolation needed: grids do not match. "
                f"Flux density grid step size is{interpolation.get_grid_step(wavelengths_angstrom)}; "
                f"Transmission table grid step size is "
                f"{interpolation.get_grid_step(self.normalized_transmission_table[:, 0])}. "
                f"Consider using the interpolate_to_grid parameter."
            )

        # TODO replace the following with forthcoming exterpolation astro_utils code
        # For now, min check: the bounds of the flux density grid are within the transmission table bounds
        if (
            wavelengths_angstrom[-1] < self.normalized_transmission_table[-1, 0]
            or wavelengths_angstrom[0] > self.normalized_transmission_table[0, 0]
        ):
            raise ValueError(
                f"Extrapolation needed: flux density interval [{wavelengths_angstrom[0]}, "
                f"{wavelengths_angstrom[-1]}] is outside the transmission table interval "
                f"[{self.normalized_transmission_table[0, 0]}, {self.normalized_transmission_table[-1, 0]}]."
            )

        # Get only the flux densities that are within the passband's wavelength range
        passband_wavelengths = self.normalized_transmission_table[:, 0]
        passband_wavelengths_indices = np.searchsorted(wavelengths_angstrom, passband_wavelengths)
        passband_flux_density_matrix = flux_density_matrix[:, passband_wavelengths_indices]

        # Calculate the bandflux as ∫ f(λ)φ_b(λ) dλ,
        # where f(λ) is the in-band flux density and φ_b(λ) is the normalized system response
        integrand = passband_flux_density_matrix * self.normalized_transmission_table[:, 1]
        bandfluxes = scipy.integrate.trapezoid(integrand, x=passband_wavelengths)

        return bandfluxes
