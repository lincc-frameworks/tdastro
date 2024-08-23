import sqlite3
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.spatial import KDTree

from tdastro.astro_utils.mag_flux import mag2flux

_rubin_opsim_colnames = {
    "time": "observationStartMJD",
    "ra": "fieldRA",
    "dec": "fieldDec",
    "zp": "zp_nJy",  # We add this column to the table
}
"""Default mapping of short column names to Rubin OpSim column names."""

# See _flux_zeropoint for discription of the following two dictionaries
# https://community.lsst.org/t/release-of-v3-4-simulations/8548/12
_lsstcam_extinction_coeff = {
    "u": -0.458,
    "g": -0.208,
    "r": -0.122,
    "i": -0.074,
    "z": -0.057,
    "y": -0.095,
}
"""The extinction coefficients for the LSST filters.

Values are from
https://community.lsst.org/t/release-of-v3-4-simulations/8548/12
"""
_lsstcam_zeropoint_per_sec_zenith = {
    "u": 26.524,
    "g": 28.508,
    "r": 28.361,
    "i": 28.171,
    "z": 27.782,
    "y": 26.818,
}
"""The zeropoints for the LSST filters at zenith

This is magnitude that produces 1 electron in a 1 second exposure,
see _assign_zero_points() docs for more details.

Values are from
https://community.lsst.org/t/release-of-v3-4-simulations/8548/12
"""

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
    _ext_coeff : `dict` or None, optional
        Mapping of filter names to extinction coefficients.
    _zp_per_sec : `dict` or None, optional
        Mapping of filter names to zeropoints at zenith.
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

    def _magnitude_electron_zeropoint(
        self, filtername: npt.ArrayLike, airmass: npt.ArrayLike, exptime: npt.ArrayLike
    ) -> npt.ArrayLike:
        """Photometric zeropoint (magnitude that produces 1 electron) for
        LSST bandpasses (v1.9), using a standard atmosphere scaled
        for different airmasses and scaled for exposure times.

        Parameters
        ----------
        filtername : ndarray of str
            The filter for which to return the photometric zeropoint.
        airmass : ndarray of float
            The airmass at which to return the photometric zeropoint.
        exptime : ndarray of float
            The exposure time for which to return the photometric zeropoint.

        Returns
        -------
        ndarray of float
            AB mags that produces 1 electron.

        Notes
        -----
        Typically, zeropoints are defined as the magnitude of a source
        which would produce 1 count in a 1 second exposure -
        here we use *electron* counts, not ADU counts.

        Authored by Lynne Jones:
        https://community.lsst.org/t/release-of-v3-4-simulations/8548/12
        """
        # calculated with syseng_throughputs v1.9
        return (
            self.zp_per_sec_getter(filtername)
            + self.ext_coeff_getter(filtername) * (airmass - 1)
            + 2.5 * np.log10(exptime)
        )

    def _flux_electron_zeropoint(
        self, filtername: npt.ArrayLike, airmass: npt.ArrayLike, exptime: npt.ArrayLike
    ) -> npt.ArrayLike:
        """Flux (nJy) per electron for LSST bandpasses

        Parameters
        ----------
        filtername : nparray of str
            The filter for which to return the photometric zeropoint.
        airmass : ndarray of float
            The airmass at which to return the photometric zeropoint.
        exptime : ndarray of float
            The exposure time for which to return the photometric zeropoint.

        Returns
        -------
        ndarray of float
            Flux (nJy) per electron.
        """

        mag_zp_electron = self._magnitude_electron_zeropoint(filtername, airmass, exptime)
        return mag2flux(mag_zp_electron)

    def _assign_zero_points(self, col_name):
        """Assign instrumental zero points in nJy to the OpSim tables"""
        self.table[col_name] = self._flux_electron_zeropoint(
            self.table[self.colmap["filter"]],
            self.table[self.colmap["airmass"]],
            self.table[self.colmap["visitExposureTime"]],
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

    def range_search(self, query_ra, query_dec, radius):
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

        Returns
        -------
        inds : `list[int]` or `list[numpy.ndarray]`
            Depending on the input, this is either a list of indices for a single query point
            or a list of arrays (of indices) for an array of query points.
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
        return self._kd_tree.query_ball_point(cart_query, adjusted_radius)

    def get_observed_times(self, query_ra, query_dec, radius):
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

        Returns
        -------
        results : `numpy.ndarray`
            Depending on the input, this is either an array of times (for a single query point)
            or an array of arrays of times (for multiple query points).
        """
        neighbors = self.range_search(query_ra, query_dec, radius)
        times = self.table[self.colmap["time"]].to_numpy()

        if isinstance(query_ra, float):
            return times[neighbors]
        else:
            num_queries = len(query_ra)
            results = np.full((num_queries), None)
            for i in range(num_queries):
                results[i] = times[neighbors[i]]
        return results
