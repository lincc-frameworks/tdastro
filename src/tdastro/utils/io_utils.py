import logging
from pathlib import Path

import numpy as np
from astropy.table import Table


def read_grid_data(input_file, format="ascii", validate=False):
    """Read 2-d grid data from a text, csv, ecsv, or fits file.

    Each line is of the form 'x0 x1 value' where x0 and x1 are the grid
    coordinates and value is the grid value. The rows should be sorted by
    increasing x0 and, within an x0 value, increasing x1.

    Parameters
    ----------
    input_file : `str` or file-like object
        The input data file.
    format : `str`
        The file format. Should be one of the formats supported by
        astropy Tables such as 'ascii', 'ascii.ecsv', or 'fits'.
        Default = 'ascii'
    validate : `bool`
        Perform additional validation on the input data.
        Default = False

    Returns
    -------
    x0 : `numpy.ndarray`
        A 1-d array with the values along the x-axis of the grid.
    x1 : `numpy.ndarray`
        A 1-d array with the values along the y-axis of the grid.
    values : `numpy.ndarray`
        A 2-d array with the values at each point in the grid with
        shape (len(x0), len(x1)).

    Raises
    ------
    ``ValueError`` if any data validation fails.
    """
    logging.debug(f"Loading file {input_file} (format={format})")
    if not Path(input_file).is_file():
        raise FileNotFoundError(f"File {input_file} not found.")

    data = Table.read(input_file, format=format, comment=r"\s*#")
    if len(data.colnames) != 3:
        raise ValueError(
            f"Incorrect format for grid data in {input_file} with format {format}. "
            f"Expected 3 columns but found {len(data.colnames)}."
        )
    x0_col = data.colnames[0]
    x1_col = data.colnames[1]
    v_col = data.colnames[2]

    # Get the values along the x0 and x1 dimensions.
    x0 = np.sort(np.unique(data[x0_col].data))
    x1 = np.sort(np.unique(data[x1_col].data))

    # Get the array of values.
    if len(data) != len(x0) * len(x1):
        raise ValueError(
            f"Incomplete data for {input_file} with format {format}. Expected "
            f"{len(x0) * len(x1)} entries but found {len(data)}."
        )

    # If we are validating, loop through the entire table and check that
    # the x0 and x1 values are in the expected order.
    if validate:
        counter = 0
        for i in range(len(x0)):
            for j in range(len(x1)):
                if data[x0_col][counter] != x0[i]:
                    raise ValueError(f"Incorrect x0 ordering in {input_file} at row={counter}.")
                if data[x1_col][counter] != x1[j]:
                    raise ValueError(f"Incorrect x0 ordering in {input_file} at row={counter}.")

    # Build the values matrix.
    values = data[v_col].data.reshape((len(x0), len(x1)))

    return x0, x1, values
