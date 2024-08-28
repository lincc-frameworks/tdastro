import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import KDTree

from tdastro.astro_utils.zeropoint import (
    _lsstcam_extinction_coeff,
    _lsstcam_zeropoint_per_sec_zenith,
    flux_electron_zeropoint,
)

_rubin_opsim_colnames = {
    "time": "observationStartMJD",
    "ra": "fieldRA",
    "dec": "fieldDec",
    "zp": "zp_nJy",  # We add this column to the table
}
"""Default mapping of short column names to Rubin OpSim column names."""

LSSTCAM_PIXEL_SCALE = 0.2
"""The pixel scale for the LSST camera in arcseconds per pixel."""

_lsstcam_readout_noise = 8.8
"""The readout noise for the LSST camera in electrons per pixel.

The value is from
https://smtn-002.lsst.io/v/OPSIM-1171/index.html
"""

_lsstcam_dark_current = 0.2
"""The dark current for the LSST camera in electrons per second per pixel.

The value is from
https://smtn-002.lsst.io/v/OPSIM-1171/index.html
"""


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

    Attributes
    ----------
    table : `dict` or `pandas.core.frame.DataFrame`
        The table with all the OpSim information.
    colmap : `dict`
        A mapping of short column names to their names in the underlying table.
    _kd_tree : `scipy.spatial.KDTree` or None
        A kd_tree of the OpSim pointings for fast spatial queries. We use the scipy
        kd-tree instead of astropy's functions so we can directly control caching.
    _pixel_scale : `float` or None, optional
        The pixel scale for the LSST camera in arcseconds per pixel.
    _read_noise : `float` or None, optional
        The readout noise for the LSST camera in electrons per pixel.
    _dark_current : `float` or None, optional
        The dark current for the LSST camera in electrons per second per pixel.
    """

    _required_names = ["ra", "dec", "time", "zp"]

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

        self._ext_coeff = _lsstcam_extinction_coeff.copy() if ext_coeff is None else ext_coeff
        self.ext_coeff_getter = np.vectorize(self._ext_coeff.get)

        self._zp_per_sec = _lsstcam_zeropoint_per_sec_zenith.copy() if zp_per_sec is None else zp_per_sec
        self.zp_per_sec_getter = np.vectorize(self._zp_per_sec.get)

        self.pixel_scale = LSSTCAM_PIXEL_SCALE if pixel_scale is None else pixel_scale
        self.read_noise = _lsstcam_readout_noise if read_noise is None else read_noise
        self.dark_current = _lsstcam_dark_current if dark_current is None else dark_current

        # Build the kd-tree.
        self._kd_tree = None
        self._build_kd_tree()

        if self.colmap["zp"] not in self.table.columns:
            self._assign_zero_points(col_name=self.colmap["zp"])

    def __len__(self):
        return len(self.table)

    def __getitem__(self, key):
        """Access the underlying opsim table."""
        return self.table[key]

    def get_column(self, colname, fail_on_missing=True):
        """Get a specific column from the table, using the mapping of
        column names if needed.

        Parameters
        ----------
        colname : `str`
            The name of the column. Can be either the name in the original table
            or the one provided in the colmap.
        fail_on_missing : `bool`
            Raise an error if the column is not found.

        Return
        ------
        column: `pandas.core.series.Series` or `None`
            Return the column if it exists Otherwise returns None or raises a
            KeyError depending on the setting of fail_on_missing.

        Raises
        ------
        Raises a KeyError of the column is not in the data and fail_on_missing is True.
        """
        # If the name is in the column map,
        if colname in self.colmap:
            colname = self.colmap[colname]

        # Get the column from the table.
        if colname in self.table.columns:
            return self.table[colname]
        else:
            if fail_on_missing:
                raise KeyError(f"Column {colname} not found in OpSim")
            return None

    def _build_kd_tree(self):
        """Construct the KD-tree from the opsim table."""
        ra_rad = np.radians(self.get_column("ra").to_numpy())
        dec_rad = np.radians(self.get_column("dec").to_numpy())
        # Convert the pointings to Cartesian coordinates on a unit sphere.
        x = np.cos(dec_rad) * np.cos(ra_rad)
        y = np.cos(dec_rad) * np.sin(ra_rad)
        z = np.sin(dec_rad)
        cart_coords = np.array([x, y, z]).T

        # Construct the kd-tree.
        self._kd_tree = KDTree(cart_coords)

    def _assign_zero_points(self, col_name):
        """Assign instrumental zero points in nJy to the OpSim tables"""
        self.table[col_name] = flux_electron_zeropoint(
            self.get_column("filter"),
            self.get_column("airmass"),
            self.get_column("visitExposureTime"),
        )

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

    def filter_on_value(self, indices, colname, value):
        """Filter a list of indices to those matching the given filter.

        For example to extract observations that are only in the red band,
        we could use:
            matching_inds = my_opsim.filter_on_value(indices, "filter", "r")

        Parameters
        ----------
        indices : `numpy.ndarray`
            An array of indices to check.
        colname : `str`
            The name of column on which to match.
        value : `str`
            The value to match.

        Returns
        -------
        result : `numpy.ndarray`
            An array containing the indices whose value in column colname
            matches the given value.
        """
        if len(indices) == 0:
            return np.array([])
        indices = np.array(indices)

        col_vals = self.get_column(colname)
        return indices[col_vals[indices] == value]

    def range_search(self, query_ra, query_dec, radius, filter_name=None):
        """Return the indices of the opsim pointings that fall within the field
        of view of the query point(s).

        Parameters
        ----------
        query_ra : `float` or `numpy.ndarray`
            The query right ascension (in degrees).
        query_dec : `float` or `numpy.ndarray`
            The query declination (in degrees).
        radius : `float`
            The angular radius of the observation (in degrees).
        filter_name : `str`, optional
            If provided, only returns observations in that filter.

        Returns
        -------
        inds : `numpy.ndarray[int]` or `numpy.ndarray[numpy.ndarray[int]]`
            Depending on the input, this is either an array of indices for a single query point
            or am array of arrays (of indices) for an array of query points.
        """
        # Transform the query point(s) to 3-d Cartesian coordinate(s).
        ra_rad = np.radians(query_ra)
        dec_rad = np.radians(query_dec)
        x = np.cos(dec_rad) * np.cos(ra_rad)
        y = np.cos(dec_rad) * np.sin(ra_rad)
        z = np.sin(dec_rad)
        cart_query = np.array([x, y, z]).T

        # Adjust the angular radius to a cartesian search radius and perform the search.
        adjusted_radius = 2.0 * np.sin(0.5 * np.radians(radius))
        inds = np.array(self._kd_tree.query_ball_point(cart_query, adjusted_radius))

        # Do a post search match on filter if needed.
        if filter_name is not None:
            if isinstance(query_ra, float):
                inds = self.filter_on_value(inds, "filter", filter_name)
            else:
                # We filter each row individually.
                for i, row in enumerate(inds):
                    inds[i] = self.filter_on_value(row, "filter", filter_name)

        return inds

    def get_observed_times(self, query_ra, query_dec, radius, filter_name=None):
        """Return the times when the query point falls within the field of view of
        a pointing in the survey.

        Parameters
        ----------
        query_ra : `float` or `numpy.ndarray`
            The query right ascension (in degrees).
        query_dec : `float` or `numpy.ndarray`
            The query declination (in degrees).
        radius : `float`
            The angular radius of the observation (in degrees).
        filter_name : `str`, optional
            If provided, only returns observations in that filter.

        Returns
        -------
        results : `numpy.ndarray`
            Depending on the input, this is either an array of times (for a single query point)
            or an array of arrays of times (for multiple query points).
        """
        neighbors = self.range_search(query_ra, query_dec, radius, filter_name)
        times = self.table[self.colmap["time"]].to_numpy()

        if isinstance(query_ra, float):
            if len(neighbors) > 0:
                return times[neighbors]
            else:
                return np.array([])
        else:
            num_queries = len(query_ra)
            results = np.full((num_queries), None)
            for i in range(num_queries):
                if len(neighbors[i]) > 0:
                    results[i] = times[neighbors[i]]
                else:
                    results[i] = np.array([])
        return results
