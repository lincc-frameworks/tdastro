import os
import time
import urllib.request
from pathlib import Path
from urllib.error import HTTPError, URLError

import numpy as np
import scipy.integrate

from tdastro.sources.physical_model import PhysicalModel


class Passbands:
    """Class to handle the transmission tables and normalization for different passbands."""

    def __init__(self, bands=None):
        if bands is None:
            self.bands = ["u", "g", "r", "i", "z", "y"]
        else:
            self.bands = bands
        self.data_path = None
        self.transmission_tables = {}
        self.normalized_system_response_tables = {}

    def _get_data_path(self):
        """Find/create a data directory for the bandpass data, located in same dir as passbands.py.

        NOTE! This is a temporary solution to be replaced when we implement Pooch throughout the project."""
        # script_path = os.path.realpath(__file__)
        # script_dir = os.path.dirname(script_path)
        # data_dir = os.path.join(script_dir, "band_data")

        data_dir = os.path.join(Path(__file__).parent, "band_data")
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        self.data_path = data_dir

    def _download_transmission_table(self, band_name: str, output_file: str, timeout: int = 30) -> bool:
        """Download a transmission table for a single band and save it to a file.

        Files used are from: http://svo2.cab.inta-csic.es/svo/theory/fps3/index.php?mode=browse&gname=LSST

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
            url = (
                f"http://svo2.cab.inta-csic.es/svo/theory/fps3/getdata.php"
                f"?format=ascii&id=LSST/LSST.{band_name}"
            )
            urllib.request.urlretrieve(url, output_file)
            start_time = time.time()
            while not os.path.exists(output_file):
                if time.time() - start_time > timeout:
                    print(f"Timeout reached while waiting for {output_file} to be created.")
                    return False
                time.sleep(1)
            print(f"Downloaded: {band_name}-band transmission table.")
            return True
        except HTTPError as e:
            print(f"HTTP error occurred when downloading transmission table for {band_name}: {e}")
            return False
        except URLError as e:
            print(f"URL error occurred when downloading transmission table for {band_name}: {e}")
            return False

    def _load_transmission_table(self, band_name: str, file_path: str) -> bool:
        """Load a transmission table from a file.

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
            print(f"Loaded: {band_name}-band transmission table.")
            return True
        except ValueError as e:
            print(f"Value error when loading transmission table for {band_name}-band: {e}")
            return False
        except OSError as e:
            print(f"OS error when loading transmission table for {band_name}-band: {e}")
            return False

    def load_all_transmission_tables(self) -> None:
        """Load all bandpass tables from a file."""
        self._get_data_path()

        for band_name in self.bands:
            file_path = f"{self.data_path}/LSST_LSST.{band_name}.dat"

            # Download the file, if it doesn't exist
            if not os.path.exists(file_path) and not self._download_transmission_table(band_name, file_path):
                print(f"Transmission table for {band_name} could not be downloaded.")
                continue

            # Load the file contents
            self._load_transmission_table(band_name, file_path)

    def _phi_b(self, transmission_table: np.ndarray) -> np.ndarray:
        """Calculate the value of phi_b for all wavelengths in a transmission table.

        This is eq. 8 from the LSST Flux Units paper. (TODO write out the equation/proper citation)

        Parameters
        ----------
        transmission_table : np.ndarray
            A 2D array with wavelengths (Angstroms) in the first column and transmission strengths in the
            second.

        Returns
        -------
        np.ndarray
            An array of phi_b values for all wavelengths in the transmission table.
        """
        # No interpolation in this version
        transmission_values = transmission_table[:, 1]
        wavelengths = transmission_table[:, 0]
        # Calculate the numerator
        numerators = transmission_values / wavelengths
        # Perform trapezoidal integration over the wavelengths to get the denominators
        denominators = scipy.integrate.trapezoid(
            transmission_values / wavelengths, x=transmission_table[:, 0]
        )
        # Calculate phi_b for all wavelengths
        return numerators / denominators

    def calculate_normalized_system_response_tables(self) -> None:
        """Calculate the normalized system response tables for all bands."""
        for band_name in self.bands:
            normalized_wavelengths_angstroms = self._phi_b(self.transmission_tables[band_name])
            self.normalized_system_response_tables[band_name] = np.column_stack(
                (self.transmission_tables[band_name][:, 0], normalized_wavelengths_angstroms)
            )

    def _get_in_band_flux(self, flux: np.ndarray, normalized_system_response_table: np.ndarray) -> float:
        """Calculate the in-band flux for a given flux and normalized system response table.

        NOTE this version requires the input fluxes to match the wavelength grid in the normalized system
        response table. A spline model can handle this by using the band's wavelength column as the
        input wavelengths, but, this will be inconvenient for other types of models.

        TODO may want to make another version that interpolates the fluxes to the wavelength grid.

        Parameters
        ----------
        flux : np.ndarray
            Array of flux values evaluated at the wavelengths in the normalized system response table.
        normalized_system_response_table : np.ndarray
            A 2D array with wavelengths (Angstroms) in the first column and normalized system response values
            in the second.

        Returns
        -------
        float
            The in-band flux value.
        """
        passband_wavelengths = normalized_system_response_table[:, 0]
        integrand = flux * normalized_system_response_table[:, 1]
        return scipy.integrate.trapezoid(integrand, x=passband_wavelengths)

    def get_all_in_band_fluxes(self, model: PhysicalModel, times) -> np.ndarray:
        """Calculate the in-band fluxes for all bands.

        NOTE given model needs to be interpolated or otherwise able to evaluate with arbitrary wavelengths.

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
            in_band_fluxes = np.apply_along_axis(
                self._get_in_band_flux, 1, all_fluxes, self.normalized_system_response_tables[band]
            )
            flux_matrix[:, i] = in_band_fluxes
        return flux_matrix
