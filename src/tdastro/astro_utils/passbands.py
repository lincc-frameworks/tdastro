import asyncio
import os
import urllib.request

import matplotlib.pyplot as plt
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
        self.data_path = ""
        self.transmission_tables = {}
        self.normalized_system_response_tables = {}

    def _get_data_path(self):
        """Find/create a data directory for the bandpass data, located next to passbands.py.

        This is a temporary solution to be replaced when we implement Pooch throughout the project."""
        script_path = os.path.realpath(__file__)
        script_dir = os.path.dirname(script_path)
        data_dir = os.path.join(script_dir, "band_data")
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        self.data_path = data_dir

    async def _download_single_transmission_table(self, band_name: str, output_file: str) -> bool:
        """Download the transmission table for the given band.

        Files used are from: http://svo2.cab.inta-csic.es/svo/theory/fps3/index.php?mode=browse&gname=LSST

        Parameters
        ----------
        band_name : str
            Name of the band for which the transmission table is to be downloaded.
        output_file : str
            Path to the file where the transmission table is to be saved.

        Returns
        -------
        bool
            True if the download is successful, False otherwise.
        """
        url = (
            f"http://svo2.cab.inta-csic.es/svo/theory/fps3/getdata.php?format=ascii&id=LSST/LSST.{band_name}"
        )

        def download():
            try:
                urllib.request.urlretrieve(url, output_file)
                return True
            except Exception as e:
                print(f"Error downloading {band_name} band: {e}")
                return False

        return await asyncio.to_thread(download)

    async def _load_single_transmission_table(self, band_name: str) -> bool:
        """Load a transmission table from a file, downloading it first if necessary.

        Parameters
        ----------
        band_name : str
            Name of the band for which the transmission table is to be loaded.

        Returns
        -------
        bool
            True if the transmission table is successfully loaded, False otherwise.
        """
        file_path = f"{self.data_path}/LSST_LSST.{band_name}.dat"

        # If we need to download the file
        if not os.path.exists(file_path):
            if not await self._download_single_transmission_table(band_name):
                return False
            while not os.path.exists(file_path):
                await asyncio.sleep(1)

        # Load the file contents
        try:
            self.transmission_tables[band_name] = np.loadtxt(file_path)
            print(f"Transmission table for {band_name} loaded.")
            return True

        # If there's an error loading the file
        except Exception as e:
            print(f"Error loading transmission table for {band_name}: {e}")
            return False

    async def load_all_transmission_tables(self) -> None:
        """Load all bandpass tables from a file."""
        for band_name in self.bands:
            self._load_single_transmission_table(band_name)

    def _phi_b(self, transmission_table: np.ndarray) -> np.ndarray:
        """Calculate the value of phi_b for all wavelengths in a transmission table.

        This is eq. 8 from the LSST Flux Units paper. (TODO write out the equation/proper citation)"""
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
        for band_name in ["u", "g", "r", "i", "z", "y"]:
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
        """
        passband_wavelengths = normalized_system_response_table[:, 0]
        integrand = flux * normalized_system_response_table[:, 1]
        return scipy.integrate.trapezoid(integrand, x=passband_wavelengths)

    def get_all_in_band_fluxes(self, model: PhysicalModel, times) -> np.ndarray:
        """Calculate the in-band fluxes for all bands.

        NOTE model needs to be interpolated or somehow able to evaluate with arbitrary wavelengths.
        """
        bands = ["u", "g", "r", "i", "z", "y"]

        # Prepare an array to hold the flux values
        flux_matrix = np.zeros((len(times), len(bands)))

        # Vectorized computation
        for i, band in enumerate(bands):
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

    # Plotting functions

    def plot_transmission_tables(self) -> None:
        """Plot all transmission tables."""
        plt.title("Transmission Tables")
        plt.xlabel("Wavelength (Angstroms)")
        plt.ylabel("Transmission")
        plt.legend(self.bands)
        for band_id in self.bands:
            plt.plot(self.transmission_tables[band_id][:, 0], self.transmission_tables[band_id][:, 1])
        plt.show()

    def plot_single_transmission_table(self, band_name: str) -> None:
        """Plot a single transmission table."""
        plt.title(f"{band_name}-band Transmission Table")
        plt.xlabel("Wavelength (Angstroms)")
        plt.ylabel("Transmission")
        plt.plot(self.transmission_tables[band_name][:, 0], self.transmission_tables[band_name][:, 1])
        plt.show()

    def plot_normalized_system_response_tables(self) -> None:
        """Plot all normalized system response tables."""
        plt.title("Normalized System Response Tables")
        plt.xlabel("Wavelength (Angstroms)")
        plt.ylabel("Normalized System Response (inverse Angstroms I guess? check this whole axis label)")
        plt.legend(self.bands)
        for band_id in self.bands:
            plt.plot(
                self.normalized_system_response_tables[band_id][:, 0],
                self.normalized_system_response_tables[band_id][:, 1],
            )
        plt.show()

    def plot_single_normalized_system_response_table(self, band_name: str) -> None:
        """Plot a single normalized system response table."""
        # TODO maybe reduce duplication with the plotting functions
        plt.title(f"{band_name}-band Normalized System Response Table")
        plt.xlabel("Wavelength (Angstroms)")
        plt.ylabel("Normalized System Response (inverse Angstroms I guess? check this whole axis label)")
        plt.plot(
            self.normalized_system_response_tables[band_name][:, 0],
            self.normalized_system_response_tables[band_name][:, 1],
        )
        plt.show()
