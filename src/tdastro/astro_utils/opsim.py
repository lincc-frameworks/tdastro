import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import KDTree

_rubin_opsim_colnames = {
    "time": "observationStartMJD",
    "ra": "fieldRA",
    "dec": "fieldDec",
}


class OpSim:
    """A wrapper class around the opsim table with cached data for efficiency.

    Attributes
    ----------
    table : `dict` or `pandas.core.frame.DataFrame`
        The table with all the opsim information.
    colmap : `dict`
        A mapping of short column names to their names in the underlying table.
        Defaults to the Rubin opsim column names.
    _kd_tree : `scipy.spatial.KDTree` or None
        A kd_tree of the opsim pointings for fast spatial queries. We use the scipy
        kd-tree instead of astropy's functions so we can directly control caching.
    """

    _required_names = ["ra", "dec", "time"]

    # Class constants for the column names.
    def __init__(self, table, colmap=_rubin_opsim_colnames):
        if isinstance(table, dict):
            self.table = pd.DataFrame(table)
        else:
            self.table = table

        # Basic validity checking on the column map names.
        self.colmap = colmap
        for name in self._required_names:
            if name not in colmap:
                raise KeyError(f"The column name map is missing key={name}")

        # Build the kd-tree.
        self._kd_tree = None
        self._build_kd_tree()

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
