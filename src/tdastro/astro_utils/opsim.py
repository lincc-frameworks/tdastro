from __future__ import annotations  # "type1 | type2" syntax in Python <3.10

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import KDTree

from tdastro.astro_utils.mag_flux import mag2flux
from tdastro.astro_utils.noise_model import poisson_bandflux_std
from tdastro.astro_utils.zeropoint import (
    _lsstcam_extinction_coeff,
    _lsstcam_zeropoint_per_sec_zenith,
    flux_electron_zeropoint,
)
from tdastro.consts import GAUSS_EFF_AREA2FWHM_SQ

_rubin_opsim_colnames = {
    "airmass": "airmass",
    "dec": "fieldDec",
    "exptime": "visitExposureTime",
    "filter": "filter",
    "ra": "fieldRA",
    "time": "observationStartMJD",
    "zp": "zp_nJy",  # We add this column to the table
}
"""Default mapping of short column names to Rubin OpSim column names."""

LSSTCAM_PIXEL_SCALE = 0.2
"""The pixel scale for the LSST camera in arcseconds per pixel."""

_lsstcam_readout_noise = 8.8
"""The standard deviation of the count of readout electrons per pixel for the LSST camera.

The value is from
https://smtn-002.lsst.io/v/OPSIM-1171/index.html
"""

_lsstcam_dark_current = 0.2
"""The dark current for the LSST camera in electrons per second per pixel.

The value is from
https://smtn-002.lsst.io/v/OPSIM-1171/index.html
"""

_lsstcam_view_radius = 1.75
"""The angular radius of the observation field (in degrees)."""


# Suppress "no docstring", because we define it via an attribute.
class OpSim:  # noqa: D101
    __doc__ = f"""A wrapper class around the opsim table with cached data for efficiency.

    Parameters
    ----------
    table : `dict` or `pandas.core.frame.DataFrame`
        The table with all the OpSim information.
    colmap : `dict`
        A mapping of short column names to their names in the underlying table.
        Defaults to the Rubin OpSim column names, stored in `_rubin_opsim_colnames`:
        {_rubin_opsim_colnames}
    ext_coeff : `dict` or None, optional
        Mapping of filter names to extinction coefficients. Defaults to
        the Rubin OpSim values, stored in `_rubin_extinction_coeff`:
        {_lsstcam_extinction_coeff}
    zp_per_sec : `dict` or None, optional
        Mapping of filter names to zeropoints at zenith. Defaults to
        the Rubin OpSim values, stored in `_rubin_zeropoint_per_sec_zenith`:
        {_lsstcam_zeropoint_per_sec_zenith}
    pixel_scale : `float` or None, optional
        The pixel scale for the LSST camera in arcseconds per pixel. Defaults to
        the Rubin OpSim value, see _rubin_pixel_scale, stored in `_rubin_pixel_scale`:
        {LSSTCAM_PIXEL_SCALE}
    read_noise : `float` or None, optional
        The readout noise for the LSST camera in electrons per pixel. Defaults to
        the Rubin OpSim value, stored in `_rubin_readout_noise`:
        {_lsstcam_readout_noise}
    dark_current : `float` or None, optional
        The dark current for the LSST camera in electrons per second per pixel. Defaults to
        the Rubin OpSim value, stored in `_rubin_dark_current`:
        {_lsstcam_dark_current}
    radius : float or None, optional
        The angular radius of the observations (in degrees).
        Defaults to the Rubin value, stored in `_lsstcam_view_radius`: {_lsstcam_view_radius}

    Attributes
    ----------
    table : `pandas.core.frame.DataFrame`
        The table with all the OpSim information.
    colmap : `dict`
        A mapping of short column names to their names in the underlying table.
    _kd_tree : `scipy.spatial.KDTree` or None
        A kd_tree of the OpSim pointings for fast spatial queries. We use the scipy
        kd-tree instead of astropy's functions so we can directly control caching.
    pixel_scale : `float` or None, optional
        The pixel scale for the LSST camera in arcseconds per pixel.
    read_noise : `float` or None, optional
        The readout noise for the LSST camera in electrons per pixel.
    dark_current : `float` or None, optional
        The dark current for the LSST camera in electrons per second per pixel.
    ext_coeff : `dict` or None, optional
        Mapping of filter names to extinction coefficients. Defaults to
        the Rubin OpSim values.
    zp_per_sec : `dict` or None, optional
        Mapping of filter names to zeropoints at zenith. Defaults to
        the Rubin OpSim values.
    radius : float
        The angular radius of the observations (in degrees).
    """

    _required_names = ["ra", "dec", "time"]

    # Class constants for the column names.
    def __init__(
        self,
        table,
        colmap=None,
        *,
        ext_coeff=None,
        zp_per_sec=None,
        pixel_scale=None,
        read_noise=None,
        dark_current=None,
        radius=None,
    ):
        if isinstance(table, dict):
            self.table = pd.DataFrame(table)
        else:
            self.table = table

        # Basic validity checking on the column map names.
        self.colmap = _rubin_opsim_colnames.copy() if colmap is None else colmap
        if "zp" not in self.colmap:
            self.colmap["zp"] = "zp_nJy"
        for name in self._required_names:
            if name not in self.colmap:
                raise KeyError(f"The column name map is missing key={name}")

        self.pixel_scale = LSSTCAM_PIXEL_SCALE if pixel_scale is None else pixel_scale
        self.read_noise = _lsstcam_readout_noise if read_noise is None else read_noise
        self.dark_current = _lsstcam_dark_current if dark_current is None else dark_current
        self.ext_coeff = _lsstcam_extinction_coeff if ext_coeff is None else ext_coeff
        self.zp_per_sec = _lsstcam_zeropoint_per_sec_zenith if zp_per_sec is None else zp_per_sec
        self.radius = _lsstcam_view_radius if radius is None else radius

        # Build the kd-tree.
        self._kd_tree = None
        self._build_kd_tree()

        # If we are not given zero point data, try to derive it from the other columns.
        if not self.has_columns("zp"):
            if self.has_columns(["filter", "airmass", "exptime"]):
                self._assign_zero_points(ext_coeff=self.ext_coeff, zp_per_sec=self.zp_per_sec)
            else:
                raise ValueError(
                    "OpSim must include either a zero point column or the columns "
                    "needed to derive it (filter, airmass, and exposure time)."
                )

    def __len__(self):
        return len(self.table)

    def __getitem__(self, key):
        """Access the underlying opsim table."""
        # Auto apply the colmap if possible.
        if self.colmap is not None and key in self.colmap:
            return self.table[self.colmap[key]]
        return self.table[key]

    @property
    def columns(self):
        """Get the column names."""
        return self.table.columns

    def has_columns(self, columns):
        """Checks whether OpSim has a column or columns while accounting
        for the colmap.

        Parameters
        ----------
        columns : `str` or iterable
            The column name or column names to check.

        Returns
        -------
        `bool`
            True if and only if all the columns are contained in the table.
        """
        if isinstance(columns, str):
            return self.colmap.get(columns, columns) in self.table.columns

        return all(self.colmap.get(col, col) in self.table.columns for col in columns)

    def _build_kd_tree(self):
        """Construct the KD-tree from the opsim table."""
        ra_rad = np.radians(self.table[self.colmap["ra"]].to_numpy())
        dec_rad = np.radians(self.table[self.colmap["dec"]].to_numpy())
        # Convert the pointings to Cartesian coordinates on a unit sphere.
        x = np.cos(dec_rad) * np.cos(ra_rad)
        y = np.cos(dec_rad) * np.sin(ra_rad)
        z = np.sin(dec_rad)
        cart_coords = np.array([x, y, z]).T

        # Construct the kd-tree.
        self._kd_tree = KDTree(cart_coords)

    def _assign_zero_points(self, *, ext_coeff: dict[str, float] | None, zp_per_sec: dict[str, float] | None):
        """Assign instrumental zero points in nJy to the OpSim tables.

        Parameters
        ----------
        ext_coeff : dict[str, float], optional
            Atmospheric extinction coefficient for each bandpass.
            Keys are the bandpass names, values are the coefficients.
            If None, the LSST coefficients are used.
        zp_per_sec : dict[str, float], optional
             The instrumental zeropoint for each bandpass in AB magnitudes,
             i.e. the magnitude that produces 1 electron in a 1-second exposure.
             Keys are the bandpass names, values are the zeropoints.
             If None, the LSST zeropoints are used.
        """
        zp_values = flux_electron_zeropoint(
            ext_coeff=ext_coeff,
            instr_zp_mag=zp_per_sec,
            band=self.table[self.colmap.get("filter", "filter")],
            airmass=self.table[self.colmap.get("airmass", "airmass")],
            exptime=self.table[self.colmap.get("exptime", "visitExposureTime")],
        )
        self.add_column(self.colmap.get("zp", "zp"), zp_values, overwrite=True)

    @classmethod
    def from_db(cls, filename, sql_query="SELECT * FROM observations", colmap=_rubin_opsim_colnames):
        """Create an OpSim object from the data in an opsim db file.

        Parameters
        ----------
        filename : `str`
            The name of the opsim db file.
        sql_query : `str`
            The SQL query to use when loading the table.
            Default = "SELECT * FROM observations"
        colmap : `dict`
            A mapping of short column names to their names in the underlying table.
            Defaults to the Rubin opsim column names.

        Returns
        -------
        opsim : `OpSim`
            A table with all of the pointing data.

        Raise
        -----
        ``FileNotFoundError`` if the file does not exist.
        ``ValueError`` if unable to load the table.
        """
        if not Path(filename).is_file():
            raise FileNotFoundError(f"opsim file {filename} not found.")
        con = sqlite3.connect(f"file:{filename}?mode=ro", uri=True)

        # Read the table.
        try:
            opsim = pd.read_sql_query(sql_query, con)
        except Exception:
            raise ValueError("Opsim database read failed.") from None

        # Close the connection.
        con.close()

        return OpSim(opsim, colmap=colmap)

    def add_column(self, colname, values, overwrite=False):
        """Add a column to the current opsim table.

        Parameters
        ----------
        colname : `str`
            The name of the new column.
        values : `int`, `float`, `str`, `list`, or `numpy.ndarray`
            The value(s) to add.
        overwrite : `bool`
            Overwrite the column is it already exists.
            Default: False
        """
        colname = self.colmap.get(colname, colname)
        if colname in self.table.columns and not overwrite:
            raise KeyError(f"Column {colname} already exists.")

        # If the input is a scalar, turn it into an array of the correct length
        if np.isscalar(values):
            values = np.full((len(self.table)), values)
        self.table[colname] = values

    def write_opsim_table(self, filename, tablename="observations", overwrite=False):
        """Write out an opsim database to a given SQL table.

        Parameters
        ----------
        filename : `str`
            The name of the opsim db file.
        tablename : `str`
            The table to which to write.
            Default = "observations"
        overwrite : `bool`
            Overwrite the existing DB file.
            Default = False

        Raise
        -----
        ``FileExistsError`` if the file already exists and ``overwrite`` is ``False``.
        """
        if_exists = "replace" if overwrite else "fail"

        con = sqlite3.connect(filename)
        try:
            self.table.to_sql(tablename, con, if_exists=if_exists)
        except Exception:
            raise ValueError("Opsim database write failed.") from None

        con.close()

    def time_bounds(self):
        """Returns the min and max times for all observations in the OpSim.

        Returns
        -------
        t_min, t_max : float, float
            The min and max times for all observations in the OpSim.
        """
        t_min = self.table[self.colmap["time"]].min()
        t_max = self.table[self.colmap["time"]].max()
        return t_min, t_max

    def filter_rows(self, rows):
        """Filter the rows in the OpSim to only include those indices that are provided
        in a list of row indices (integers) or marked True in a mask.

        Parameters
        ----------
        rows : numpy.ndarray
            Either a Boolean array of the same length as the table or list of integer
            row indices to keep.

        Returns
        -------
        new_opsim : OpSim
            A new OpSim object with the reduced rows.
        """
        # Check if we are dealing with a mask of a list of indices.
        rows = np.asarray(rows)
        if rows.dtype == bool:
            if len(rows) != len(self.table):
                raise ValueError(
                    f"Mask length mismatch. Expected {len(self.table)} rows, but found {len(rows)}."
                )
            mask = rows
        else:
            mask = np.full((len(self.table),), False)
            mask[rows] = True

        # Do the actual filtering and generate a new OpSim. This automatically creates
        # the cached data, such as the KD-tree.
        new_table = self.table[mask]
        new_opsim = OpSim(
            new_table,
            colmap=self.colmap,
            ext_coeff=self.ext_coeff,
            zp_per_sec=self.zp_per_sec,
            pixel_scale=self.pixel_scale,
            read_noise=self.read_noise,
            dark_current=self.dark_current,
        )
        return new_opsim

    def range_search(self, query_ra, query_dec, radius=None):
        """Return the indices of the opsim pointings that fall within the field
        of view of the query point(s).

        Parameters
        ----------
        query_ra : `float` or `numpy.ndarray`
            The query right ascension (in degrees).
        query_dec : `float` or `numpy.ndarray`
            The query declination (in degrees).
        radius : `float` or None, optional
            The angular radius of the observation (in degrees). If None
            uses the default radius for the OpSim.

        Returns
        -------
        inds : `list[int]` or `list[numpy.ndarray]`
            Depending on the input, this is either a list of indices for a single query point
            or a list of arrays (of indices) for an array of query points.
        """
        radius = self.radius if radius is None else radius

        # Transform the query point(s) to 3-d Cartesian coordinate(s).
        ra_rad = np.radians(query_ra)
        dec_rad = np.radians(query_dec)
        x = np.cos(dec_rad) * np.cos(ra_rad)
        y = np.cos(dec_rad) * np.sin(ra_rad)
        z = np.sin(dec_rad)
        cart_query = np.array([x, y, z]).T

        # Adjust the angular radius to a cartesian search radius and perform the search.
        adjusted_radius = 2.0 * np.sin(0.5 * np.radians(radius))
        return self._kd_tree.query_ball_point(cart_query, adjusted_radius)

    def get_observations(self, query_ra, query_dec, radius=None, cols=None):
        """Return the observation information when the query point falls within
        the field of view of a pointing in the survey.

        Parameters
        ----------
        query_ra : `float`
            The query right ascension (in degrees).
        query_dec : `float`
            The query declination (in degrees).
        radius : `float` or None, optional
            The angular radius of the observation (in degrees). If None
            uses the default radius for the OpSim.
        cols : `list`
            A list of the names of columns to extract. If `None` returns all the
            columns.

        Returns
        -------
        results : `dict`
            A dictionary mapping the given column name to a numpy array of values.
        """
        neighbors = self.range_search(query_ra, query_dec, radius)

        results = {}
        if cols is None:
            cols = self.table.columns.to_list()
        for col in cols:
            # Allow the user to specify either the original or mapped column names.
            table_col = self.colmap.get(col, col)
            if table_col not in self.table.columns:
                raise KeyError(f"Unrecognized column name {table_col}")
            results[col] = self.table[table_col][neighbors].to_numpy()
        return results

    def bandflux_error_point_source(self, bandflux, index):
        """Compute observational bandflux error for a point source

        Parameters
        ----------
        bandflux : array_like of float
            Band bandflux of the point source in nJy.
        index : array_like of int
            The index of the observation in the OpSim table.

        Returns
        -------
        flux_err : array_like of float
            Simulated bandflux noise in nJy.
        """
        observations = self.table.iloc[index]

        # By the effective FWHM definition, see
        # https://smtn-002.lsst.io/v/OPSIM-1171/index.html
        footprint = GAUSS_EFF_AREA2FWHM_SQ * observations["seeingFwhmEff"] ** 2

        # table value is in mag/arcsec^2
        sky_njy = mag2flux(observations["skyBrightness"])

        return poisson_bandflux_std(
            bandflux,
            pixel_scale=self.pixel_scale,
            total_exposure_time=observations["visitExposureTime"],
            exposure_count=observations["numExposures"],
            footprint=footprint,
            sky=sky_njy,
            zp=observations["zp_nJy"],
            readout_noise=self.read_noise,
            dark_current=self.dark_current,
        )


def create_random_opsim(num_obs, seed=None):
    """Create a random OpSim pointings drawn uniformly from (RA, dec).

    Parameters
    ----------
    num_obs : int
        The size of the OpSim to generate.
    seed : int
        The seed to used for random number generation. If None then
        uses a default random number generator.
        Default: None

    Returns
    -------
    opsim_data : OpSim
        The OpSim data structure.
    seed : int, optional
        The seed for the random number generator.
    """
    if num_obs <= 0:
        raise ValueError("Number of observations must be greater than zero.")

    rng = np.random.default_rng() if seed is None else np.random.default_rng(seed=seed)

    # Generate the (RA, dec) pairs uniformly on the surface of a sphere.
    ra = np.degrees(rng.uniform(0.0, 2.0 * np.pi, size=num_obs))
    dec = np.degrees(np.arccos(2.0 * rng.uniform(0.0, 1.0, size=num_obs) - 1.0) - (np.pi / 2.0))

    # Generate the information needed to compute zeropoint.
    airmass = rng.uniform(1.3, 1.7, size=num_obs)
    filter = rng.choice(["u", "g", "r", "i", "z", "y"], size=num_obs)

    input_data = {
        "observationStartMJD": 0.05 * np.arange(num_obs),
        "fieldRA": ra,
        "fieldDec": dec,
        "airmass": airmass,
        "filter": filter,
        "visitExposureTime": 29.0 * np.ones(num_obs),
    }

    opsim = OpSim(
        input_data,
        ext_coeff=_lsstcam_extinction_coeff,
        zp_per_sec=_lsstcam_zeropoint_per_sec_zenith,
        pixel_scale=LSSTCAM_PIXEL_SCALE,
        read_noise=_lsstcam_readout_noise,
        dark_current=_lsstcam_dark_current,
    )
    return opsim


def opsim_add_random_data(opsim_data, colname, min_val=0.0, max_val=1.0):
    """Add a column composed of random uniform data. Used for testing.

    Parameters
    ----------
    opsim_data : OpSim
        The OpSim data structure to modify.
    colname : `str`
        The name of the new column to add.
    min_val : `float`
        The minimum value of the uniform range.
        Default: 0.0
    max_val : `float`
        The maximum value of the uniform range.
        Default: 1.0
    """
    values = np.random.uniform(low=min_val, high=max_val, size=len(opsim_data))
    opsim_data.add_column(colname, values)


def oversample_opsim(
    opsim: OpSim,
    *,
    pointing: tuple[float, float] = (200, -50),
    search_radius: float = 1.75,
    delta_t: float = 0.01,
    time_range: tuple[float | None, float | None] = (None, None),
    bands: list[str] | None = None,
    strategy: str = "darkest_sky",
):
    """Single-pointing oversampled OpSim table.

    It includes observations for a single pointing only,
    but with very high time resolution. The observations
    would alternate between the bands.

    Parameters
    ----------
    opsim : OpSim
        The OpSim table to oversample.
    pointing : tuple of RA and Dec in degrees
        The pointing to use for the oversampled table.
    search_radius : float, optional
        The search radius for the oversampled table in degrees.
        The default is the half of the LSST's field of view.
    delta_t : float, optional
        The time between observations in days.
    time_range : tuple or floats or Nones, optional
        The start and end times of the observations in MJD.
        `None` means to use the minimum (maximum) time in
        all the observations found for the given pointing.
        Time is being samples as np.arange(*time_range, delta_t).
    bands : list of str or None, optional
        The list of bands to include in the oversampled table.
        The default is to include all bands found for the given pointing.
    strategy : str, optional
        The strategy to select prototype observations.
        - "darkest_sky" selects the observations with the minimal sky brightness
          (maximum "skyBrightness" value) in each band. This is the default.
        - "random" selects the observations randomly. Fixed seed is used.

    """
    ra, dec = pointing
    observations = opsim.table.iloc[opsim.range_search(ra, dec, search_radius)]
    if len(observations) == 0:
        raise ValueError("No observations found for the given pointing.")

    time_min, time_max = time_range
    if time_min is None:
        time_min = np.min(observations["observationStartMJD"])
    if time_max is None:
        time_max = np.max(observations["observationStartMJD"])
    if time_min >= time_max:
        raise ValueError(f"Invalid time_range: start > end: {time_min} > {time_max}")

    uniq_bands = np.unique(observations["filter"])
    if bands is None:
        bands = uniq_bands
    elif not set(bands).issubset(uniq_bands):
        raise ValueError(f"Invalid bands: {bands}")

    new_times = np.arange(time_min, time_max, delta_t)
    n = len(new_times)
    if n < len(bands):
        raise ValueError("Not enough time points to cover all bands.")

    new_table = pd.DataFrame(
        {
            # Just in case, to not have confusion with the original table
            "observationId": opsim.table["observationId"].max() + 1 + np.arange(n),
            "observationStartMJD": new_times,
            "fieldRA": ra,
            "fieldDec": dec,
            "filter": np.tile(bands, n // len(bands)),
        }
    )
    other_columns = [column for column in observations.columns if column not in new_table.columns]

    if strategy == "darkest_sky":
        for band in bands:
            # MAXimum magnitude is MINimum brightness (darkest sky)
            idxmax = observations["skyBrightness"][observations["filter"] == band].idxmax()
            idx = new_table.index[new_table["filter"] == band]
            darkest_sky_obs = pd.DataFrame.from_records([observations.loc[idxmax]] * idx.size, index=idx)
            new_table.loc[idx, other_columns] = darkest_sky_obs[other_columns]
    elif strategy == "random":
        rng = np.random.default_rng(0)
        for band in bands:
            single_band_obs = observations[observations["filter"] == band]
            idx = new_table.index[new_table["filter"] == band]
            random_obs = single_band_obs.sample(idx.size, replace=True, random_state=rng).set_index(idx)
            new_table.loc[idx, other_columns] = random_obs[other_columns]
    else:
        raise ValueError(f"Invalid strategy: {strategy}")

    return OpSim(
        new_table,
        colmap=opsim.colmap,
        pixel_scale=opsim.pixel_scale,
        read_noise=opsim.read_noise,
        dark_current=opsim.dark_current,
    )
