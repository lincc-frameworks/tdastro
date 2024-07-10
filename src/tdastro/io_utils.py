import numpy as np
from astropy.table import Table


def read_grid_data(input_file, format="ascii"):
    """Read 2-d grid data from a text, csv, ecsv, or fits file.

    Each line is of the form 'x0 x1 value' where x0 and x1 are the grid
    coordinates and value is the grid value. The rows should be sorted by
    increasing x0 and, within an x0 value, increasing x1.

    Parameters
    ----------
    input_file : `str` or file-like object
        The input data file.
    format : `str`
        The file format. Should be one of ascii, csv, ecsv,
        or fits.

    Returns
    -------
    x0 : `numpy.ndarray`
        A 1-d array with the values along the x-axis of the grid.
    x1 : `numpy.ndarray`
        A 1-d array with the values along the y-axis of the grid.
    values : `numpy.ndarray`
        A 2-d array with the values at each point in the grid with
        shape (len(x0), len(x1)).
    """
    data = Table.read(input_file, format=format)
    if len(data.colnames) != 3:
        raise ValueError(
            f"Incorrect format for grid data in {input_file} with format {format}. "
            f"Expected 3 columns but found {len(data.colnames)}."
        )

    # Get the values along the x0 and x1 dimensions.
    x0 = np.sort(np.unique(data[data.colnames[0]].data))
    x1 = np.sort(np.unique(data[data.colnames[1]].data))

    # Get the array of values.
    if len(data) != len(x0) * len(x1):
        raise ValueError(
            f"Incomplete data for {input_file} with format {format}. Expected "
            f"{len(x0) * len(x1)} entries but found {len(data)}."
        )
    values = data[data.colnames[2]].data.reshape((len(x0), len(x1)))

    return x0, x1, values
