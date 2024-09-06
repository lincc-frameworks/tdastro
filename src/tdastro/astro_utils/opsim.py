from __future__ import annotations  # "type1 | type2" syntax in Python <3.10

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import KDTree

from tdastro.astro_utils.mag_flux import mag2flux
from tdastro.astro_utils.noise_model import poisson_flux_std
from tdastro.astro_utils.zeropoint import (
    _lsstcam_extinction_coeff,
    _lsstcam_zeropoint_per_sec_zenith,
    flux_electron_zeropoint,
)
from tdastro.consts import GAUSS_EFF_AREA2FWHM_SQ

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
"""The standard deviation of the count of readout electrons per pixel for the LSST camera.

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

        # Build the kd-tree.
        self._kd_tree = None
        self._build_kd_tree()

        if self.colmap["zp"] not in self.table.columns:
            self._assign_zero_points(col_name=self.colmap["zp"], ext_coeff=ext_coeff, zp_per_sec=zp_per_sec)

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

    def _assign_zero_points(
        self, col_name: str, *, ext_coeff: dict[str, float] | None, zp_per_sec: dict[str, float] | None
    ):
        """Assign instrumental zero points in nJy to the OpSim tables"""
        self.table[col_name] = flux_electron_zeropoint(
            ext_coeff=ext_coeff,
            instr_zp_mag=zp_per_sec,
            band=self.table[self.colmap.get("filter", "filter")],
            airmass=self.table[self.colmap.get("airmass", "airmass")],
            exptime=self.table[self.colmap.get("exptime", "visitExposureTime")],
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

    def get_observations(self, query_ra, query_dec, radius, cols=None):
        """Return the observation information when the query point falls within
        the field of view of a pointing in the survey.

        Parameters
        ----------
        query_ra : `float`
            The query right ascension (in degrees).
        query_dec : `float`
            The query declination (in degrees).
        radius : `float`
            The angular radius of the observation (in degrees).
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

    def flux_err_point_source(self, flux, index):
        """Compute observational flux error for a point source

        Parameters
        ----------
        flux : array_like of float
            Flux of the point source in nJy.
        index : array_like of int
            The index of the observation in the OpSim table.

        Returns
        -------
        flux_err : array_like of float
            Simulated flux noise in nJy.
        """
        observations = self.table.iloc[index]

        # By the effective FWHM definition, see
        # https://smtn-002.lsst.io/v/OPSIM-1171/index.html
        footprint = GAUSS_EFF_AREA2FWHM_SQ * observations["seeingFwhmEff"] ** 2

        # table value is in mag/arcsec^2
        sky_njy = mag2flux(observations["skyBrightness"])

        return poisson_flux_std(
            flux,
            pixel_scale=self.pixel_scale,
            total_exposure_time=observations["visitExposureTime"],
            exposure_count=observations["numExposures"],
            footprint=footprint,
            sky=sky_njy,
            zp=observations["zp_nJy"],
            readout_noise=self.read_noise,
            dark_current=self.dark_current,
        )
