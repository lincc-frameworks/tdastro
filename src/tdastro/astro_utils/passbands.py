import logging
import os
import socket
import urllib.request
from pathlib import Path
from urllib.error import HTTPError, URLError

import numpy as np
import scipy.integrate

from tdastro.sources.physical_model import PhysicalModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class Passbands:
    """Class to handle the transmission tables and normalization for different passbands.

    Attributes
    ----------
    bands : list
        A list of strings representing the band names (eg, ["u", "g", "r", "i", "z", "y"]).
    data_path : str
        Path to the directory where the bandpass data is stored.
    transmission_tables : dict
        A dictionary with band names as keys and transmission tables as values. Transmission tables
        are 2D arrays with wavelengths (Angstrom) in the first column and transmission strengths in
        the second.
    normalized_system_response_tables : dict
        A dictionary with band names as keys and normalized system response tables as values. These
        tables are 2D arrays with wavelengths (Angstrom) in the first column and normalized system
        response values in the second.
    """

    def __init__(self, bands=None):
        if bands is None:
            self.bands = []
        else:
            self.bands = bands
        self.data_path = None
        self.transmission_tables = {}
        self.normalized_system_response_tables = {}

    def _get_data_path(self):
        """Find/create a data directory for the bandpass data, located in same dir as passbands.py.

        Note
        ----
        This is a temporary solution to be replaced when we implement Pooch throughout the project.
        """
        data_dir = os.path.join(Path(__file__).parent, "band_data")
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        self.data_path = data_dir

    def _download_transmission_table(self, band_name: str, output_file: str, timeout: int = 5) -> bool:
        """Download a transmission table for a single band and save it to a file.

        Files used are from:
        http://svo2.cab.inta-csic.es/svo/theory/fps3/index.php?mode=browse&gname=LSST

        Note
        ----
        This is a temporary solution to be replaced when we implement Pooch throughout the project.

        Parameters
        ----------
        band_name : str
            Name of the band for which the transmission table is to be downloaded.
        output_file : str
            Path to the file where the transmission table will be saved.
        timeout : int
            Time in seconds to wait for the file to be downloaded and created.

        Returns
        -------
        bool
            True if the transmission table is successfully downloaded, False otherwise.
        """
        try:
            socket.setdefaulttimeout(timeout)

            url = (
                f"http://svo2.cab.inta-csic.es/svo/theory/fps3/getdata.php"
                f"?format=ascii&id=LSST/LSST.{band_name}"
            )

            urllib.request.urlretrieve(url, output_file)

            if os.path.getsize(output_file) == 0:
                logging.error(f"Transmission table for {band_name} is empty.")
                return False

            logging.info(f"Downloaded: {band_name}-band transmission table.")
            return True
        except HTTPError as e:
            logging.error(f"HTTP error occurred when downloading transmission table for {band_name}: {e}")
            return False
        except URLError as e:
            logging.error(f"URL error occurred when downloading transmission table for {band_name}: {e}")
            return False

    def _load_transmission_table(self, band_name: str, file_path: str) -> bool:
        """Load a transmission table from a file.

        Note
        ----
        This is a temporary solution to be replaced when we implement Pooch throughout the project.

        Parameters
        ----------
        band_name : str
            Name of the band for which the transmission table is to be loaded.
        file_path : str
            Path to the file containing the transmission table.

        Returns
        -------
        bool
            True if the transmission table is successfully loaded, False otherwise.
        """
        try:
            self.transmission_tables[band_name] = np.loadtxt(file_path)
            logging.info(f"Loaded: {band_name}-band transmission table.")
            return True
        except ValueError as e:
            logging.error(f"Value error when loading transmission table for {band_name}-band: {e}")
            return False
        except OSError as e:
            logging.error(f"OS error when loading transmission table for {band_name}-band: {e}")
            return False

    def load_all_transmission_tables(self) -> None:
        """Load all bandpass tables from a file."""
        self._get_data_path()

        for band_name in self.bands:
            file_path = f"{self.data_path}/LSST_LSST.{band_name}.dat"

            # Download the file, if it doesn't exist
            if not os.path.exists(file_path) and not self._download_transmission_table(band_name, file_path):
                logging.error(f"Transmission table for {band_name} could not be downloaded.")
                continue

            # Load the file contents
            self._load_transmission_table(band_name, file_path)

    def _phi_b(self, transmission_table: np.ndarray) -> np.ndarray:
        """Calculate the value of phi_b for all wavelengths in a transmission table.

        This is eq. 8 from "On the Choice of LSST Flux Units" (Ivezić et al.):

        φb(λ) = Sb(λ)λ⁻¹ / ∫ Sb(λ)λ⁻¹ dλ

        Parameters
        ----------
        transmission_table : np.ndarray
            A 2D array with wavelengths (Angstrom) in the first column and transmission strengths in
            the second.

        Returns
        -------
        np.ndarray
            An array of phi_b values for all wavelengths in the transmission table.
        """
        # No interpolation in this version
        transmission_values = transmission_table[:, 1]
        wavelengths_angstrom = transmission_table[:, 0]
        # Calculate the numerator
        numerators = transmission_values / wavelengths_angstrom
        # Perform trapezoidal integration over the wavelengths to get the denominators
        denominator = scipy.integrate.trapezoid(numerators, x=wavelengths_angstrom)
        # Calculate phi_b for all wavelengths
        return numerators / denominator

    def calculate_normalized_system_response_tables(self) -> None:
        """Calculate the normalized system response tables for all bands."""
        for band_name in self.bands:
            normalized_wavelengths_angstroms = self._phi_b(self.transmission_tables[band_name])
            self.normalized_system_response_tables[band_name] = np.column_stack(
                (self.transmission_tables[band_name][:, 0], normalized_wavelengths_angstroms)
            )

    def _get_in_band_flux(self, flux: np.ndarray, band: str) -> float:
        """Calculate the in-band flux for a given flux and band id.

        In-band flux is calculated as the integral of the product of the flux and the normalized system
        response over the wavelengths in the band.

        Note
        ----
        This version requires the input fluxes to match the wavelength grid in the normalized system
        response table. A spline model can handle this by using the band's wavelength column as the
        input wavelengths--but, this will be inconvenient for other types of models.

        Let's consider adding another version that interpolates the normalized system response table
        to the wavelengths of the input fluxes.

        Parameters
        ----------
        flux : np.ndarray
            Array of flux values evaluated at the wavelengths in the normalized system response table.
        band : str
            Name of the band for which the in-band flux is to be calculated. Used to select the corresponding
            normalized system response table.

        Returns
        -------
        float
            The in-band flux value.
        """
        passband_wavelengths = self.normalized_system_response_tables[band][:, 0]
        integrand = flux * self.normalized_system_response_tables[band][:, 1]
        return scipy.integrate.trapezoid(integrand, x=passband_wavelengths)

    def get_all_in_band_fluxes(self, model: PhysicalModel, times) -> np.ndarray:
        """Calculate the in-band fluxes for all bands.

        Note
        ----
        The given model needs to be interpolable or otherwise able to be evaluated with arbitrary wavelengths.

        Parameters
        ----------
        model : PhysicalModel
            A model object that can evaluate fluxes at arbitrary wavelengths.
        times : np.ndarray
            Array of times at which to evaluate the model.

        Returns
        -------
        np.ndarray
            A 2D array with in-band fluxes for all bands at all times.
        """
        # Prepare an array to hold the flux values
        flux_matrix = np.zeros((len(times), len(self.bands)))

        # Apply _get_in_band_flux to all fluxes
        for i, band in enumerate(self.bands):
            wavelengths_in_band = self.normalized_system_response_tables[band][:, 0]
            # Evaluate the spline model for all times and wavelengths in the band
            # Note we'll be double-dipping here, in that some wavelengths do exist in multiple bands
            all_fluxes = model.evaluate(times, wavelengths_in_band)

            # Compute the in-band fluxes for each time
            in_band_fluxes = np.apply_along_axis(self._get_in_band_flux, 1, all_fluxes, band)
            flux_matrix[:, i] = in_band_fluxes
        return flux_matrix
