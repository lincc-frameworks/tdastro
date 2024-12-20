import logging
import socket
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Literal, Optional, Union
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
    table_dir : str
        The path to the directory containing the passband tables.
    waves : np.ndarray
        The union of all wavelengths in the passbands.
    _in_band_wave_indices : dict
        A dictionary mapping the passband name (eg, "LSST_u") to the indices of that specific
        passband's wavelengths in the full waves list.
    """

    def __init__(
        self,
        preset: str = None,
        passband_parameters: Optional[list] = None,
        table_dir: Optional[Union[str, Path]] = None,
        given_passbands: list = None,
        **kwargs,
    ):
        """Construct a PassbandGroup object.

        Parameters
        ----------
        preset : str, optional
            A pre-defined set of passbands to load. If using a preset, passband_parameters will be ignored.
        passband_parameters : list of dict, optional
            A list of dictionaries of passband parameters used to create Passband objects.
            Each dictionary must contain the following:
            - survey : str
            - filter_name : str
            Dictionaries may also contain the following optional parameters:
            - table_path : str or Path
            - table_url : str
            - delta_wave : float
            - trim_quantile : float
            - units : str (either 'nm' or 'A')
            If survey is not LSST (or other survey with defined defaults), either a table_path or table_url
            must be provided.
        table_dir : str, optional
            The path to the directory containing the passband tables. If a table_path has not been specified
            in the passband_parameters dictionary, table paths will be set to
            {table_dir}/{survey}/{filter_name}.dat. If None, the table path will be set to a default path.
        given_passbands : list, optional
            A list of Passband objects from which to create the PassbandGroup. These
            overwrite any passbands with the same full name provided by either
            preset or passband_parameters.
        **kwargs
            Additional keyword arguments to pass to the Passband constructor.
        """
        self.passbands = {}
        self._in_band_wave_indices = {}

        if preset is None and passband_parameters is None and given_passbands is None:
            raise ValueError(
                "PassbandGroup must be initialized with one of a preset, a list"
                "of Passband objects, or passband_parameters."
            )

        if preset is not None:
            self._load_preset(preset, table_dir=table_dir, **kwargs)

        elif passband_parameters is not None:
            for parameters in passband_parameters:
                # Add any missing parameters from kwargs
                for key, value in kwargs.items():
                    if key not in parameters:
                        parameters[key] = value

                # Set the table path if it is not already set and a table_dir is provided
                if "table_path" not in parameters:
                    if table_dir is not None:
                        parameters["table_path"] = Path(
                            table_dir,
                            parameters["survey"],
                            f"{parameters['filter_name']}.dat",
                        )
                elif isinstance(parameters["table_path"], str):
                    parameters["table_path"] = Path(parameters["table_path"])

                passband = Passband(**parameters)
                self.passbands[passband.full_name] = passband

        # Load any additional passbands from the given list.
        if given_passbands is not None:
            for pb_obj in given_passbands:
                self.passbands[pb_obj.full_name] = pb_obj

        # Compute the unique points and bounds for the group.
        self._update_waves()

    def __str__(self) -> str:
        """Return a string representation of the PassbandGroup."""
        return (
            f"PassbandGroup containing {len(self.passbands)} passbands: "
            f"{', '.join(self.passbands.keys())}"
        )

    def __len__(self) -> int:
        return len(self.passbands)

    def _load_preset(self, preset: str, table_dir: Optional[str], **kwargs) -> None:
        """Load a pre-defined set of passbands.

        Parameters
        ----------
        preset : str
            The name of the pre-defined set of passbands to load.
        table_dir : str, optional
            The path to the directory containing the passband tables. If no table_path has been specified in
            the PassbandGroup's passband_parameters and table_dir is not None, table paths will be set to
            table_dir/{survey}/{filter_name}.dat.
        **kwargs
            Additional keyword arguments to pass to the Passband constructor.
        """
        if preset == "LSST":
            for filter_name in ["u", "g", "r", "i", "z", "y"]:
                if table_dir is None:
                    self.passbands[f"LSST_{filter_name}"] = Passband("LSST", filter_name, **kwargs)
                else:
                    table_path = Path(table_dir, "LSST", f"{filter_name}.dat")
                    self.passbands[f"LSST_{filter_name}"] = Passband(
                        "LSST",
                        filter_name,
                        table_path=table_path,
                        **kwargs,
                    )
        else:
            raise ValueError(f"Unknown passband preset: {preset}")

    def _update_waves(self, threshold=1e-5) -> None:
        """Update the group's wave attribute to be the union of all wavelengths in
        the passbands and update the group's _in_band_wave_indices attribute, which is
        the indices of the group's wave grid that are in the passband's wave grid.

        Eg, if a group's waves are [11, 12, 13, 14, 15] and a single band's are [13, 14],
        we get [2, 3].

        The indices are stored in the passband's _in_band_wave_indices attribute as either
        a tuple of two ints (lower, upper) or a 1D np.ndarray of ints.

        Parameters
        ----------
        threshold : float
            The threshold for merging two "close" wavelengths. This is used to
            avoid problems with numerical precision.
            Default: 1e-5
        """
        if len(self.passbands) == 0:
            self.waves = np.array([])
        else:
            # Compute the unique wavelengths (accounting for floating point error) by
            # sorting the union of all the waves, computing the gaps between each adjacent
            # pair (using 1e8 for before the first), and saving the points with a gap
            # larger than the threshold.
            all_waves = np.concatenate([passband.waves for passband in self.passbands.values()])
            sorted_waves = np.sort(all_waves)
            gap_sizes = np.insert(sorted_waves[1:] - sorted_waves[:-1], 0, 1e8)
            self.waves = sorted_waves[gap_sizes >= threshold]

        # Update the mapping of each passband's wavelengths to the corresponding indices in the
        # unioned list of all wavelengths.
        self._in_band_wave_indices = {}
        for name, passband in self.passbands.items():
            # We only want the fluxes that are in the passband's wavelength range
            # So, find the indices in the group's wave grid that are in the passband's wave grid
            lower, upper = passband.waves[0], passband.waves[-1]
            lower_index, upper_index = np.searchsorted(self.waves, [lower, upper])

            # Check that this is the right grid after all (check will fail if passbands overlap and passbands
            # do not happen to be on the same phase of the grid; eg, even if the step is 10, if the first
            # passband starts at 100 and the second at 105, the passbands won't share the same grid)
            if np.array_equal(self.waves[lower_index : upper_index + 1], passband.waves):
                indices = slice(lower_index, upper_index + 1)
            else:
                indices = np.searchsorted(self.waves, passband.waves)
            self._in_band_wave_indices[name] = indices

    def wave_bounds(self):
        """Get the minimum and maximum wavelength for this group.

        Returns
        -------
        min_wave : float
            The minimum wavelength.
        max_wave : float
            The maximum wavelength.
        """
        min_wave = np.min(self.waves)
        max_wave = np.max(self.waves)
        return min_wave, max_wave

    def process_transmission_tables(
        self, delta_wave: Optional[float] = 5.0, trim_quantile: Optional[float] = 1e-3
    ):
        """Process the transmission tables for all passbands in the group; recalculate group's wave attribute.

        Parameters
        ----------
        delta_wave : float or None, optional
            The grid step of the wave grid. Default is 5.0.
        trim_quantile : float or None, optional
            The quantile to trim the transmission table by. For example, if trim_quantile is 1e-3, the
            transmission table will be trimmed to include only the central 99.8% of rows.
        """
        for passband in self.passbands.values():
            passband.process_transmission_table(delta_wave, trim_quantile)

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
            indices = self._in_band_wave_indices[full_name]

            if indices is None:
                raise ValueError(
                    f"Passband {full_name} does not have _in_band_wave_indices set. "
                    "This should have been calculated in PassbandGroup._update_waves."
                )

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
        The wavelengths of the transmission table. To be used when evaluating models to generate fluxes
        that will be passed to fluxes_to_bandflux.
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
        trim_quantile: Optional[float] = 1e-3,
        table_path: Optional[Union[str, Path]] = None,
        table_url: Optional[str] = None,
        table_values: Optional[np.array] = None,
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
        trim_quantile : float or None, optional
            The quantile to trim the transmission table by. For example, if trim_quantile is 1e-3, the
            transmission table will be trimmed to include only the central 99.8% of the area under the
            transmission curve.
        table_path : str, optional
            The path to the transmission table file. If None, the table path will be set to a default path;
            if no file exists at this location, the file will be downloaded from table_url.
        table_url : str, optional
            The URL to download the transmission table file. If None, the table URL will be set to
            a default URL based on the survey and filter_name. Default is None.
        table_values : np.ndarray, optional
            A 2D array of wavelengths and transmissions. Used to provide a given table.
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

        self.table_path = Path(table_path) if table_path is not None else None
        self.table_url = table_url
        self.units = units

        if table_values is not None:
            if table_values.shape[1] != 2:
                raise ValueError("Passband requires an input table with exactly two columns.")
            if table_path is not None or table_url is not None:
                raise ValueError("Multiple inputs given for passband table.")
            self._loaded_table = np.copy(table_values)
        else:
            self._load_transmission_table(force_download=force_download)

        # Check that wavelengths are strictly increasing
        if not np.all(np.diff(self._loaded_table[:, 0]) > 0):
            raise ValueError("Wavelengths in transmission table must be strictly increasing.")

        # Preprocess the passband.
        self._standardize_units()
        self.process_transmission_table(delta_wave, trim_quantile)

    def __str__(self) -> str:
        """Return a string representation of the Passband."""
        return f"Passband: {self.full_name}"

    def _standardize_units(self):
        """Convert the units into Angstroms."""
        if self.units == "nm":
            # Multiply the first column (wavelength) by 10.0 to convert to Angstroms
            self._loaded_table[:, 0] *= 10.0
        elif self.units != "A":
            raise ValueError(f"Unknown Passband units {self.units}")
        self.units = "A"

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
            self.table_path = Path(
                Path(__file__).parent,
                "passbands",
                self.survey,
                f"{self.filter_name}.dat",
            )
        self.table_path.parent.mkdir(parents=True, exist_ok=True)
        if force_download or not self.table_path.exists():
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
                self.table_url = f"https://github.com/lsst/throughputs/blob/main/baseline/total_{self.filter_name}.dat?raw=true"
            else:
                raise NotImplementedError(
                    f"Transmission table download is not yet implemented for survey: {self.survey}."
                )
        try:
            socket.setdefaulttimeout(10)
            logging.info(f"Retrieving {self.table_url}")
            urllib.request.urlretrieve(self.table_url, self.table_path)
            if self.table_path.stat().st_size == 0:
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
        self, delta_wave: Optional[float] = 5.0, trim_quantile: Optional[float] = 1e-3
    ):
        """Process the transmission table.

        Parameters
        ----------
        delta_wave : Optional[float] = 5.0
            The grid step of the wave grid. Default is 5.0 Angstroms.
        trim_quantile : Optional[float] = 1e-3
            The quantile to trim the transmission table by. For example, if trim_quantile is 1e-3, the
            transmission table will be trimmed to include only the central 99.8% of rows.
        """
        interpolated_table = self._interpolate_transmission_table(self._loaded_table, delta_wave)
        trimmed_table = self._trim_transmission_by_quantile(interpolated_table, trim_quantile)
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

    def _trim_transmission_by_quantile(self, table: np.ndarray, trim_quantile: Optional[float]) -> np.ndarray:
        """Trim the transmission table so that it only includes the central (100 - 2*trim_quartile)% of rows.

        E.g., if trim_quantile is 1e-3, the transmission table will be trimmed to include only the central
        99.8% of rows.

        Parameters
        ----------
        table : np.ndarray
            A 2D array of wavelengths and transmissions.
        trim_quantile : float
            The quantile to trim the transmission table by. For example, if trim_quantile is 1e-3, the
            transmission table will be trimmed to include only the central 99.8% of rows. Must be greater than
            or equal to 0 and less than 0.5.

        Returns
        -------
        np.ndarray
            A 2D array of wavelengths and transmissions.
        """
        if trim_quantile is None or trim_quantile == 0.0:
            return table
        if trim_quantile < 0 or trim_quantile >= 0.5:
            raise ValueError(f"Trim quantile must be between 0 and 0.5; got {trim_quantile}.")

        # Separate wavelengths and transmissions
        wavelengths = table[:, 0]
        transmissions = table[:, 1]

        # Calculate the cumulative sum of the transmission values (area under the curve)
        cumulative_area = scipy.integrate.cumulative_trapezoid(transmissions, x=wavelengths)

        # Normalize cumulative area to range from 0 to 1
        cumulative_area /= cumulative_area[-1]

        # Find indices where the cumulative area exceeds the trim quantiles
        lower_bound = max(np.searchsorted(cumulative_area, trim_quantile, side="right") - 1, 0)
        upper_bound = np.searchsorted(cumulative_area, 1 - trim_quantile)

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

        where f(λ) is the flux density of an object at the top of the atmosphere, and φ_b(λ) is the
        normalized system response for given band b."

        Parameters
        ----------
        flux_density_matrix : np.ndarray
            A 2D or 3D array of flux densities. If the array is 2D it contains a single sample where
            the rows are the T times and columns are M wavelengths. If the array is 3D it contains S
            samples and the values are indexed as (sample_num, time, wavelength).

        Returns
        -------
        bandfluxes : np.ndarray
            A 1D or 2D array. If the flux_density_matrix contains a single sample (2D input) then
            the function returns a 1D length T array where each element is the bandflux
            at the corresponding time. Otherwise the function returns a size S x T array where
            each entry corresponds to the value for a given sample at a given time.
        """
        if flux_density_matrix.size == 0:
            raise ValueError("Empty flux density matrix used.")
        if len(flux_density_matrix.shape) == 2:
            w_axis = 1
            flux_density_matrix_num_cols = flux_density_matrix.shape[1]
        elif len(flux_density_matrix.shape) == 3:
            w_axis = 2
            flux_density_matrix_num_cols = flux_density_matrix.shape[2]
        else:
            raise ValueError("Invalid flux density matrix. Must be 2 or 3-dimensional.")

        # Check the number of wavelengths match.
        if len(self.waves) != flux_density_matrix_num_cols:
            raise ValueError(
                f"Passband mismatched grids: Flux density matrix has {flux_density_matrix_num_cols} "
                f"columns, which does not match the {len(self.waves)} rows in band {self.full_name}'s "
                f"transmission table. Check that the flux density matrix was calculated on the same grid as "
                f"the transmission tables, which can be accessed via the Passband's or PassbandGroup's waves "
                f"attribute."
            )

        # Calculate the bandflux as ∫ f(λ)φ_b(λ) dλ,
        # where f(λ) is the flux density and φ_b(λ) is the normalized system response
        integrand = flux_density_matrix * self.processed_transmission_table[:, 1]
        bandfluxes = scipy.integrate.trapezoid(integrand, x=self.waves, axis=w_axis)
        return bandfluxes
