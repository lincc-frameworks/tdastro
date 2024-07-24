import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.coordinates import angular_separation
from astropy.table import Table


def load_opsim_table(filename, sql_query="SELECT * FROM observations"):
    """Load the data in an opsim db file.

    Parameters
    ----------
    filename : `str`
        The name of the opsim db file.
    sql_query : `str`
        The SQL query to use when loading the table.
        Default = "SELECT * FROM observations"

    Returns
    -------
    opsim : `pandas.core.frame.DataFrame`
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

    return opsim


def write_opsim_table(opsim, filename, tablename="observations", overwrite=False):
    """Write out an opsim database to a given SQL table.

    Parameters
    ----------
    opsim : `pandas.core.frame.DataFrame`
        A table with all of the pointing data.
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
        opsim.to_sql(tablename, con, if_exists=if_exists)
    except Exception:
        raise ValueError("Opsim database write failed.") from None

    con.close()


def pointings_from_opsim(
    opsim,
    time_colname="observationStartMJD",
    ra_colname="fieldRA",
    dec_colname="fieldDec",
):
    """Create an astropy table of the minimal pointing information from the
    and opsim data frame.

    Parameters
    ----------
    opsim : `pandas.core.frame.DataFrame`
        A table with all of the opsim data.
    time_colname : `str`
        The column name of the timestamp.
        Default = "observationStartMJD"
    ra_colname : `str`
        The column name of the RA information.
        Default = "fieldRA"
    dec_colname : `str`
        The column name of the dec information.
        Default = "fieldDec"

    Returns
    -------
    pointings : `astropy.table.Table`
        A table with a timestamp, RA, and Dec for each pointing.
    """
    for colname in [time_colname, ra_colname, dec_colname]:
        if colname not in opsim.columns:
            raise KeyError(f"Column {colname} not found in opsim data frame with columns {opsim.columns}")

    pointings = Table()
    pointings["time"] = opsim[time_colname].to_numpy()
    pointings["ra"] = opsim[ra_colname].to_numpy()
    pointings["dec"] = opsim[dec_colname].to_numpy()
    return pointings


def get_pointings_matched_times(pointings, ra, dec, fov):
    """Get the time stamp of all the pointings that match the given (RA, dec).

    Note
    ----
    This is a slow, exhaustive implementation for testing and comparison. We
    need to implement something faster for the real system.

    Parameters
    ----------
    pointings : `astropy.table.Table`
        A table with "time", "ra", and "dec" for each pointing.
    ra : `float`
        The query right ascension (in degrees).
    dec : `float`
        The query declination (in degrees).
    fov : `float`
        The angular radius of the observation (in degrees).

    Returns
    -------
    times : `numpy.ndarray`
        The times where the query observation was within the field of view.
    """
    pt_ra = np.radians(pointings["ra"])
    pt_dec = np.radians(pointings["dec"])
    query_ra = np.full(pt_ra.shape, np.radians(ra))
    query_dec = np.full(pt_dec.shape, np.radians(dec))
    dist = angular_separation(pt_ra, pt_dec, query_ra, query_dec)

    times = pointings["time"][dist <= np.radians(fov)]
    return times
