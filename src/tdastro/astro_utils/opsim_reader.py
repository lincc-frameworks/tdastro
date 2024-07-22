import sqlite3
from pathlib import Path

import pandas as pd


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

    # Close the connection and return the data frame.
    con.close()

    return opsim
