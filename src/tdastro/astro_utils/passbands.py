import logging
import socket
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Literal, Optional, Union
from urllib.error import HTTPError, URLError

import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate

import tdastro

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# Set default colors for plotting to match:
# https://community.lsst.org/t/lsst-filter-profiles/1463
_lsst_filter_plot_colors = {
    "u": "purple",
    "g": "blue",
    "r": "green",
    "i": "yellow",
    "z": "orange",
    "y": "red",
}


class PassbandGroup:
    """A group of passbands.

    The given passbands can come from a single survey or multiple surveys. As such, a given filter may appear
    in multiple passbands in a passband group. For example we might generate data from a combination of Rubin
    and DECCAM and want to use both their r filters. Thus the primary mapping is done by the passband's full
    name “{SURVEY}_{FILTER}”. Lookups by filter name are permitted in cases where the filter only occurs once
    in the group and thus the desired passband is unambiguous.

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
    _filter_to_name : dict
        A dictionary mapping the filter name to a list of matching passband full names.
    """

    def __init__(
        self,
        preset: str = None,
        passband_parameters: Optional[list] = None,
        table_dir: Optional[Union[str, Path]] = None,
        given_passbands: Optional[list] = None,
        filters_to_load: Optional[list] = None,
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
        filters_to_load : list, optional
            A list of filters to include in this PassbandGroup. If None, includes all filters.
            Otherwise drops filters that do not occur and throws an error if a filter is missing.
            Used for loading a subset of the filters.
        **kwargs
            Additional keyword arguments to pass to the Passband constructor.
        """
        self.passbands = {}
        self._in_band_wave_indices = {}
        self._filter_to_name = {}

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

        # Prune any filters that are not on the given list and check for any missing filters.
        # We match on either the full name or the filter name.
        if filters_to_load is not None:
            filters_to_load = set(filters_to_load)
            filters_remaining = filters_to_load.copy()
            all_bands = list(self.passbands.keys())

            for pb_name in all_bands:
                pb_obj = self.passbands[pb_name]
                if pb_name in filters_to_load:
                    filters_remaining.discard(pb_name)
                elif pb_obj.filter_name in filters_to_load:
                    filters_remaining.discard(pb_obj.filter_name)
                else:
                    del self.passbands[pb_name]

            if len(filters_remaining) != 0:
                raise ValueError(f"The following filters were not found: {filters_remaining}")

        # Compute the unique points and bounds for the group.
        self._update_internal_data()

    def __str__(self) -> str:
        """Return a string representation of the PassbandGroup."""
        return (
            f"PassbandGroup containing {len(self.passbands)} passbands: "
            f"{', '.join(self.passbands.keys())}"
        )

    def __len__(self) -> int:
        return len(self.passbands)

    def __getitem__(self, key):
        """Return the passband corresponding to a full name or filter name."""
        if key in self.passbands:
            return self.passbands[key]
        elif key in self._filter_to_name:
            # If we are looking up the passband by filter name, we check
            # that the filter only appears in a single Passband object.
            pb_list = self._filter_to_name[key]
            if len(pb_list) > 1:
                raise KeyError(
                    f"Filter {key} corresponds to multiple passbands: {pb_list}.\n"
                    "Lookup the passband by full name."
                )
            return self.passbands[pb_list[0]]
        else:
            raise KeyError(f"Unknown passband {key}")

    def __contains__(self, key):
        if key in self.passbands:
            return True
        elif key in self._filter_to_name:
            return True
        return False

    @classmethod
    def from_dir(
        cls,
        dir_path: Union[str, Path],
        filters: Optional[list] = None,
        delta_wave: Optional[float] = 5.0,
        trim_quantile: Optional[float] = 1e-3,
        units: Optional[Literal["nm", "A"]] = "A",
    ):
        """Load the passbands from a directory where the directorty name corresponds
        to the survey and the file names correspond to the filters:
        path_to_survey_dir/survey_name/filter_name.dat

        Parameters
        ----------
        dir_path : str or Path
            The path to the passband files including the survey directory.
        filters : list, set, or None, optional
            A list of filters to load.
        delta_wave : float or None, optional
            The grid step of the wave grid, in angstroms.
            It is typically used to downsample transmission using linear interpolation.
            Default is 5 angstroms. If `None` the original grid is used.
        trim_quantile : float or None, optional
            The quantile to trim the transmission table by. For example, if trim_quantile is 1e-3, the
            transmission table will be trimmed to include only the central 99.8% of the area under the
            transmission curve.
        units : Literal['nm','A'], optional
            Denotes whether the wavelength units of the table are nanometers ('nm') or Angstroms ('A').
            By default 'A'. Does not affect the output units of the class, only the interpretation of the
            provided passband table.
        """
        dir_path = Path(dir_path)
        if not dir_path.is_dir():
            raise ValueError(f"{dir_path} is not a valid directory.")

        # Iterate through the files in the directory.
        all_params = []
        for entry in dir_path.iterdir():
            if entry.is_file():
                filter_name = entry.stem
                params = {
                    "survey": dir_path.name,
                    "filter_name": filter_name,
                    "table_path": dir_path / entry,
                    "delta_wave": delta_wave,
                    "trim_quantile": trim_quantile,
                    "units": units,
                }
                all_params.append(params)

        # Do the actual loading. The subsetting of filters happens here.
        return PassbandGroup(passband_parameters=all_params, filters_to_load=filters)

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
        logger.info(f"Loading passbands from preset {preset}")
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

    def _update_internal_data(self) -> None:
        """Update the cached internal data."""
        # Update the mapping of filter name to full_name.
        self._filter_to_name = {}
        for full_name, pb_obj in self.passbands.items():
            if pb_obj.filter_name in self._filter_to_name:
                logger.info("Multiple passband objects detected for filter {pb_obj.filter_name}")
                self._filter_to_name[pb_obj.filter_name].append(full_name)
            else:
                self._filter_to_name[pb_obj.filter_name] = [full_name]

        self._update_waves()

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

    def mask_by_filter(self, filters):
        """Compute a mask for whether a given observations is of interest for
        for a given analysis. For example this could be used to remove unneeded
        observations from an OpSim or other data set.

        Parameters
        ----------
        filters : list-like
            A length T array of filter names or full names.

        Returns
        -------
        mask : numpy.ndarray
            A length T array of Booleans indicating whether the filter is of interest.
        """
        filters = np.asarray(filters)
        full_name_mask = np.isin(filters, list(self.passbands.keys()))
        filter_name_mask = np.isin(filters, list(self._filter_to_name.keys()))
        return full_name_mask | filter_name_mask

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

        self._update_internal_data()

    def fluxes_to_bandfluxes(self, flux_density_matrix: np.ndarray) -> np.ndarray:
        """Calculate bandfluxes for all passbands in the group.

        Parameters
        ----------
        flux_density_matrix : np.ndarray
            A 2D array of flux densities where rows are times and columns are wavelengths.

        Returns
        -------
        dict of np.ndarray
            A dictionary of bandfluxes with passband full names as keys and np.ndarrays of
            bandfluxes as values.
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
                    "This should have been calculated in PassbandGroup._update_internal_data."
                )

            in_band_fluxes = flux_density_matrix[:, indices]

            bandfluxes[full_name] = passband.fluxes_to_bandflux(in_band_fluxes)
        return bandfluxes

    def plot(self, ax=None, figure=None):
        """Plot the PassbandGroup on a single plot.

        Parameters
        ----------
        ax : matplotlib.pyplot.Axes or None, optional
            Axes, None by default.
        figure : matplotlib.pyplot.Figure or None
            Figure, None by default.
        """
        if ax is None:
            if figure is None:
                figure = plt.figure()
            ax = figure.add_axes([0, 0, 1, 1])

        # Plot each passband.
        for pb_obj in self.passbands.values():
            pb_obj.plot(ax=ax, plot_loaded=False)
        ax.legend()


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
        self.units = units

        if table_values is not None:
            # If the data values are given directly, use those.
            if table_values.shape[1] != 2:
                raise ValueError("Passband requires an input table with exactly two columns.")
            if table_path is not None or table_url is not None:
                raise ValueError("Multiple inputs given for passband table.")
            self._loaded_table = np.copy(table_values)
        else:
            # Load the data from a file, possibly downloading it if needed.
            if table_path is None:
                # If no path is given, use the default.
                table_path = Path(
                    tdastro._TDASTRO_BASE_DATA_DIR,
                    "passbands",
                    self.survey,
                    f"{self.filter_name}.dat",
                )
            else:
                table_path = Path(table_path)

            # If no URL is given, try loading a default.
            if table_url is None:
                table_url = _get_passband_url(self.survey, self.filter_name)

            self._loaded_table = Passband.load_transmission_table(
                table_path,
                table_url=table_url,
                force_download=force_download,
            )

        # Check that wavelengths are strictly increasing
        if not np.all(np.diff(self._loaded_table[:, 0]) > 0):
            raise ValueError("Wavelengths in transmission table must be strictly increasing.")

        # Preprocess the passband.
        self._standardize_units()
        self.process_transmission_table(delta_wave, trim_quantile)

    def __str__(self) -> str:
        """Return a string representation of the Passband."""
        return f"Passband: {self.full_name}"

    def __eq__(self, other) -> bool:
        """Determine if two passbands have equal values for the processed tables."""
        if self.units != other.units:
            return False

        # Check that they are using the same wavelengths.
        if len(self.waves) != len(other.waves):
            return False
        if not np.allclose(self.waves, other.waves):
            return False

        # Check that they have the (approximately) same transmission tables.
        if self.processed_transmission_table.shape != other.processed_transmission_table.shape:
            return False
        if not np.allclose(self.processed_transmission_table, other.processed_transmission_table):
            return False

        return True

    def _standardize_units(self):
        """Convert the units into Angstroms."""
        if self.units == "nm":
            # Multiply the first column (wavelength) by 10.0 to convert to Angstroms
            self._loaded_table[:, 0] *= 10.0
        elif self.units != "A":
            raise ValueError(f"Unknown Passband units {self.units}")
        self.units = "A"

    @staticmethod
    def load_transmission_table(
        table_path: Union[str, Path],
        table_url: Optional[str] = None,
        force_download: bool = False,
    ) -> np.ndarray:
        """Load a transmission table from a file (or download it if it doesn't exist).
        Table must have 2 columns: wavelengths and transmissions; wavelengths must be
        strictly increasing.

        Parameters
        ----------
        table_path : str or Path
            The path to the transmission table file. If no file exists at this location, the
            file will be downloaded from table_url.
        table_url : str, optional
            The URL to download the transmission table file. If None, the table URL will be set to
            a default URL based on the survey and filter_name. Default is None.
        force_download : bool, optional
            If True, the transmission table will be downloaded even if it already exists locally.
            Default is False.

        Returns
        -------
        np.ndarray
            A 2D array of wavelengths and transmissions.
        """
        logger.info(f"Loading passband from file: {table_path}")

        table_path.parent.mkdir(parents=True, exist_ok=True)
        if force_download or not table_path.exists():
            Passband.download_transmission_table(table_path, table_url)

        # Load the table file
        try:
            loaded_table = np.loadtxt(table_path)
        except OSError as e:
            raise OSError(f"Error reading transmission table from file: {e}") from e

        # Check that the table has the correct shape
        if loaded_table.size == 0 or loaded_table.shape[1] != 2:
            raise ValueError("Transmission table must have exactly 2 columns.")
        # Check that wavelengths are strictly increasing
        if not np.all(np.diff(loaded_table[:, 0]) > 0):
            raise ValueError("Wavelengths in transmission table must be strictly increasing.")

        return loaded_table

    @staticmethod
    def download_transmission_table(
        table_path: Union[str, Path],
        table_url: str,
    ) -> bool:
        """Download a transmission table from a URL.

        Parameters
        ----------
        table_path : str or Path
            The path to the transmission table file. This is where the downloaded file
            will be written.
        table_url : str
            The URL to download the transmission table file.

        Returns
        -------
        bool
            True if the download was successful, False otherwise.
        """
        # Check that there is a valid URL for the download.
        if table_url is None:
            raise ValueError("No URL given for table download.")
        logger.info(f"Downloading passband file from {table_url} to {table_path}")

        # Create the directory in which to save the file if it does not already exist.
        table_path = Path(table_path)
        table_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            socket.setdefaulttimeout(10)
            logger.info(f"Retrieving {table_url}")
            urllib.request.urlretrieve(table_url, table_path)
            if table_path.stat().st_size == 0:
                logger.error(f"Transmission table downloaded from {table_url} is empty.")
                return False
            else:
                logger.info(f"Downloaded transmission table {table_url}.")
                return True
        except HTTPError as e:
            logger.error(f"HTTP error occurred when downloading table {table_url}: {e}")
            return False
        except URLError as e:
            logger.error(f"URL error occurred when downloading table {table_url}: {e}")
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

    def plot(self, ax=None, figure=None, color=None, plot_loaded=False):
        """Plot the passband.

        Parameters
        ----------
        ax : matplotlib.pyplot.Axes or None, optional
            Axes, None by default.
        figure : matplotlib.pyplot.Figure or None
            Figure, None by default.
        color : str or None, optional
            The color of the curve.
        plot_loaded : bool
            Also plot the loaded table as a dashed line. Used for debugging.
        """
        if ax is None:
            if figure is None:
                figure = plt.figure()
            ax = figure.add_axes([0, 0, 1, 1])

        # If the color is provided, we use that. Otherwise we try
        # the LSST filter colors (or default to black).
        if color is None:
            color = _lsst_filter_plot_colors.get(self.filter_name, "black")

        ax.plot(
            self.processed_transmission_table[:, 0],  # X values are the wavelength
            self.processed_transmission_table[:, 1],  # Y values are the transmission values.
            color=color,
            label=self.full_name,
        )
        if plot_loaded:
            ax.plot(
                self._loaded_table[:, 0],  # X values are the wavelength
                self._loaded_table[:, 1],  # Y values are the transmission values.
                color=color,
                linestyle="--",
            )

        ax.set_xlabel(r"Wavelength, $\AA$")
        ax.set_ylabel("Transmission Value")
        ax.set_ylim(0, None)


# --- Helper Functions ----------------------------------------------------


def _get_passband_url(survey: str, filter: str) -> Union[str, None]:
    """Get the URL to download passband information.

    Parameters
    ----------
    survey : str,
        The passband's survey.
    filter : str,
        The passband's filter.

    Returns
    -------
    url : str or None
        Returns a string with the URL if the survey's data location is known.
        Otherwise returns None.
    """
    if survey == "LSST":
        return f"https://github.com/lsst/throughputs/blob/main/baseline/total_{filter}.dat?raw=true"
    return None
