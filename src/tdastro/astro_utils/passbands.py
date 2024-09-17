import logging
import os
import socket
import urllib.parse
import urllib.request
from typing import Literal, Optional
from urllib.error import HTTPError, URLError

import numpy as np
import scipy.integrate

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class PassbandGroup:
    """A group of passbands.

    Attributes
    ----------
    passbands : dict of Passband
        A dictionary of Passband objects, where the keys are the full_names of the passbands (eg, "LSST_u").
    waves : np.ndarray
        The union of all wavelengths in the passbands.
    """

    def __init__(
        self,
        preset: str = None,
        passband_parameters: list = None,
        **kwargs,
    ):
        """Construct a PassbandGroup object.

        Parameters
        ----------
        preset : str, optional
            A pre-defined set of passbands to load.
        passband_parameters : list of dict, optional
            A list of dictionaries of passband parameters used to create Passband objects.
            Each dictionary must contain the following:
            - survey : str
            - filter_name : str
            Dictionaries may also contain the following optional parameters:
            - table_path : str
            - table_url : str
            - delta_wave : float
            - trim_percentile : float
            - units : str (either 'nm' or 'A')
            If survey is not LSST (or other survey with defined defaults), either a table_path or table_url
            must be provided.
        **kwargs
            Additional keyword arguments to pass to the Passband constructor.
        """
        self.passbands = {}

        if preset is not None:
            self._load_preset(preset, **kwargs)

        if passband_parameters is not None:
            for parameters in passband_parameters:
                # Add any missing parameters from kwargs
                for key, value in kwargs.items():
                    if key not in parameters:
                        parameters[key] = value
                passband = Passband(**parameters)
                self.passbands[passband.full_name] = passband

        self._update_waves()

    def __str__(self) -> str:
        """Return a string representation of the PassbandGroup."""
        return (
            f"PassbandGroup containing {len(self.passbands)} passbands: "
            f"{', '.join(self.passbands.keys())}"
        )

    def _load_preset(self, preset: str, **kwargs) -> None:
        """Load a pre-defined set of passbands.

        Parameters
        ----------
        preset : str
            The name of the pre-defined set of passbands to load.
        **kwargs
            Additional keyword arguments to pass to the Passband constructor.
        """
        if preset == "LSST":
            self.passbands = {
                "LSST_u": Passband("LSST", "u", **kwargs),
                "LSST_g": Passband("LSST", "g", **kwargs),
                "LSST_r": Passband("LSST", "r", **kwargs),
                "LSST_i": Passband("LSST", "i", **kwargs),
                "LSST_z": Passband("LSST", "z", **kwargs),
                "LSST_y": Passband("LSST", "y", **kwargs),
            }
        else:
            raise ValueError(f"Unknown passband preset: {preset}")

    def _update_waves(self) -> None:
        """Update the group's wave attribute to be the union of all wavelengths in the passbands."""
        if len(self.passbands) == 0:
            self.waves = np.array([])
        else:
            self.waves = np.unique(np.concatenate([passband.waves for passband in self.passbands.values()]))

    def process_transmission_tables(
        self, delta_wave: Optional[float] = 5.0, trim_percentile: Optional[float] = 0.1
    ):
        """Process the transmission tables for all passbands in the group; recalculate group's wave attribute.

        Parameters
        ----------
        delta_wave : float or None, optional
            The grid step of the wave grid. Default is 5.0.
        trim_percentile : float or None, optional
            The percentile to trim the transmission table by. For example, if trim_percentile is 0.1, the
            transmission table will be trimmed to include only the central 80% of rows.
        """
        for passband in self.passbands.values():
            passband.process_transmission_table(delta_wave, trim_percentile)

        self._update_waves()

    def fluxes_to_bandfluxes(self, flux_density_matrix: np.ndarray) -> np.ndarray:
        """Calculate bandfluxes for all passbands in the group.

        Parameters
        ----------
        flux_density_matrix : np.ndarray
            A 2D array of flux densities where rows are times and columns are wavelengths.

        Returns
        -------
        dict of np.ndarray
            A dictionary of bandfluxes with passband names as keys and np.ndarrays of bandfluxes as values.
        """
        if flux_density_matrix.size == 0 or len(self.waves) != len(flux_density_matrix[0]):
            flux_density_matrix_num_cols = 0 if flux_density_matrix.size == 0 else len(flux_density_matrix[0])
            raise ValueError(
                f"PassbandGroup mismatched grids: Flux density matrix has {flux_density_matrix_num_cols} "
                f"columns, which does not match transmission table's {len(self.waves)} rows. Check that the "
                f"flux density matrix was calculated on the same grid as the transmission tables, which can "
                f"be accessed via the Passband's or PassbandGroup's waves attribute."
            )

        bandfluxes = {}
        for full_name, passband in self.passbands.items():
            # We only want the fluxes that are in the passband's wavelength range
            # So, find the indices in the group's wave grid that are in the passband's wave grid
            lower, upper = passband.waves[0], passband.waves[-1]
            lower_index, upper_index = np.searchsorted(self.waves, [lower, upper])
            # Check that this is the right grid after all (check will fail if the grid is not regular, or
            # passbands are overlapping)
            if np.array_equal(self.waves[lower_index : upper_index + 1], passband.waves):
                in_band_fluxes = flux_density_matrix[:, lower_index : upper_index + 1]
            else:
                indices = np.unique(np.searchsorted(passband.waves, self.waves))
                if indices[-1] == len(passband.waves):
                    indices = indices[:-1]
                in_band_fluxes = flux_density_matrix[:, indices]

            bandfluxes[full_name] = passband.fluxes_to_bandflux(in_band_fluxes)
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
    table_path : str
        The path to the transmission table file.
    table_url : str
        The URL to download the transmission table file.
    waves : np.ndarray
        The wavelengths of the transmission table.
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
        delta_wave: Optional[float] = 5.0,
        trim_percentile: Optional[float] = 0.1,
        table_path: Optional[str] = None,
        table_url: Optional[str] = None,
        units: Optional[Literal["nm", "A"]] = "A",
        force_download: bool = False,
    ):
        """Construct a Passband object.

        Parameters
        ----------
        survey : str
            The survey to which the passband belongs: eg, "LSST".
        filter_name : str
            The filter_name of the passband: eg, "u".
        delta_wave : float or None, optional
            The grid step of the wave grid, in angstroms.
            It is typically used to downsample transmission using linear interpolation.
            Default is 5 angstroms. If `None` the original grid is used.
        trim_percentile : float or None, optional
            The percentile to trim the transmission table by. For example, if trim_percentile is 0.1, the
            transmission table will be trimmed to include only the central 80% of the area under the
            transmission curve.
        waves : np.ndarray
            The wavelengths of the transmission table. To be used when evaluating models to generate fluxes
            that will be passed to fluxes_to_bandflux.
        table_path : str, optional
            The path to the transmission table file. If None, the table path will be set to a default path;
            if no file exists at this location, the file will be downloaded from table_url.
        table_url : str, optional
            The URL to download the transmission table file. If None, the table URL will be set to
            a default URL based on the survey and filter_name. Default is None.
        units : Literal['nm','A'], optional
            Denotes whether the wavelength units of the table are nanometers ('nm') or Angstroms ('A').
            By default 'A'. Does not affect the output units of the class, only the interpretation of the
            provided passband table.
        force_download : bool, optional
            If True, the transmission table will be downloaded even if it already exists locally. Default is
            False.
        """
        self.survey = survey
        self.filter_name = filter_name
        self.full_name = f"{survey}_{filter_name}"

        self.table_path = table_path
        self.table_url = table_url
        self.units = units

        self._load_transmission_table(force_download=force_download)
        self.process_transmission_table(delta_wave, trim_percentile)

    def __str__(self) -> str:
        """Return a string representation of the Passband."""
        return f"Passband: {self.full_name}"

    def _load_transmission_table(self, force_download: bool = False) -> None:
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
        if force_download or not os.path.exists(self.table_path):
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

        # Correct for units
        if self.units == "nm":
            loaded_table[
                :, 0
            ] *= 10.0  # Multiply the first column (wavelength) by 10.0 to convert to Angstroms

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
                # TODO switch to files at: https://github.com/lsst/throughputs/blob/main/baseline/total_g.dat
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
            logging.info(f"Retrieving {self.table_url}", flush=True)
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

    def process_transmission_table(
        self, delta_wave: Optional[float] = 5.0, trim_percentile: Optional[float] = 0.1
    ):
        """Process the transmission table.

        Parameters
        ----------
        delta_wave : Optional[float] = None
            The grid step of the wave grid. Default is 5.0 Angstroms.
        trim_percentile : Optional[float] = None
            The percentile to trim the transmission table by. For example, if trim_percentile is 0.1, the
            transmission table will be trimmed to include only the central 80% of rows.
        """
        interpolated_table = self._interpolate_transmission_table(self._loaded_table, delta_wave)
        trimmed_table = self._trim_transmission_by_percentile(interpolated_table, trim_percentile)
        self.processed_transmission_table = self._normalize_transmission_table(trimmed_table)

        self.waves = self.processed_transmission_table[:, 0]

    def _interpolate_transmission_table(self, table: np.ndarray, delta_wave: Optional[float]) -> np.ndarray:
        """Interpolate the transmission table to a new wave grid.

        Parameters
        ----------
        table : np.ndarray
            A 2D array of wavelengths and transmissions.
        delta_wave : float or None
            The grid step of the wave grid.

        Returns
        -------
        np.ndarray
            A 2D array of wavelengths and transmissions.
        """
        # Don't interpolate if delta_wave is None or the table is already on the desired grid
        if delta_wave is None:
            return table
        if np.allclose(np.diff(table[:, 0]), delta_wave):
            return table

        # Regrid wavelengths to the new wave grid
        wavelengths = table[:, 0]
        lower_bound, upper_bound = wavelengths[0], wavelengths[-1]
        new_wavelengths = np.linspace(
            lower_bound, upper_bound, int((upper_bound - lower_bound) / delta_wave) + 1
        )

        # Interpolate the transmission table to the new wave grid
        spline = scipy.interpolate.InterpolatedUnivariateSpline(table[:, 0], table[:, 1], ext="raise", k=1)
        interpolated_transmissions = spline(new_wavelengths)
        return np.column_stack((new_wavelengths, interpolated_transmissions))

    def _trim_transmission_by_percentile(
        self, table: np.ndarray, trim_percentile: Optional[float]
    ) -> np.ndarray:
        """Trim the transmission table such that it only includes the central (1 - 2*trim_percentile) of rows.

        E.g., if trim_percentile is 0.1, the transmission table will be trimmed to include only the central
        80% of rows.

        Parameters
        ----------
        table : np.ndarray
            A 2D array of wavelengths and transmissions.
        trim_percentile : float
            The percentile to trim the transmission table by. For example, if trim_percentile is 0.1, the
            transmission table will be trimmed to include only the central 80% of rows.

        Returns
        -------
        np.ndarray
            A 2D array of wavelengths and transmissions.
        """
        if trim_percentile is None or trim_percentile == 0.0:
            return table
        if trim_percentile < 0 or trim_percentile > 0.5:
            raise ValueError("Trim percentile must be between 0 and 0.5.")

        # Separate wavelengths and transmissions
        wavelengths = table[:, 0]
        transmissions = table[:, 1]

        # Calculate the cumulative sum of the transmission values (area under the curve)
        cumulative_area = scipy.integrate.cumulative_trapezoid(transmissions, x=wavelengths)

        # Normalize cumulative area to range from 0 to 1
        cumulative_area /= cumulative_area[-1]

        # Find indices where the cumulative area exceeds the trim percentiles
        lower_bound = np.searchsorted(cumulative_area, trim_percentile)
        upper_bound = np.searchsorted(cumulative_area, 1 - trim_percentile)

        # Trim the table to the desired range
        trimmed_table = table[lower_bound : upper_bound + 1]

        return trimmed_table

    def _normalize_transmission_table(self, transmission_table: np.ndarray) -> np.ndarray:
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

    def fluxes_to_bandflux(
        self,
        flux_density_matrix: np.ndarray,
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

        Returns
        -------
        np.ndarray
            An array of bandfluxes with length flux_density_matrix, where each element is the bandflux
            at the corresponding time.
        """
        if flux_density_matrix.size == 0 or len(self.waves) != len(flux_density_matrix[0]):
            flux_density_matrix_num_cols = 0 if flux_density_matrix.size == 0 else len(flux_density_matrix[0])
            raise ValueError(
                f"Passband mismatched grids: Flux density matrix has {flux_density_matrix_num_cols} "
                f"columns, which does not match the {len(self.waves)} rows in band {self.full_name}'s "
                f"transmission table. Check that the flux density matrix was calculated on the same grid as "
                f"the transmission tables, which can be accessed via the Passband's or PassbandGroup's waves "
                f"attribute."
            )

        # Calculate the bandflux as ∫ f(λ)φ_b(λ) dλ,
        # where f(λ) is the in-band flux density and φ_b(λ) is the normalized system response
        integrand = flux_density_matrix * self.processed_transmission_table[:, 1]
        bandfluxes = scipy.integrate.trapezoid(integrand, x=self.waves)

        return bandfluxes
