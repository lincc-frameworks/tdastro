import gzip
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
    input_file : str or file-like object
        The input data file.
    format : str
        The file format. Should be one of the formats supported by
        astropy Tables such as 'ascii', 'ascii.ecsv', or 'fits'.
        Default: 'ascii'
    validate : bool
        Perform additional validation on the input data.
        Default: False

    Returns
    -------
    x0 : numpy.ndarray
        A 1-d array with the values along the x-axis of the grid.
    x1 : numpy.ndarray
        A 1-d array with the values along the y-axis of the grid.
    values : numpy.ndarray
        A 2-d array with the values at each point in the grid with
        shape (len(x0), len(x1)).

    Raises
    ------
    ValueError if any data validation fails.
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
                    raise ValueError(
                        f"Incorrect x0 ordering in {input_file} at line={counter}."
                        f"Expected {x0[i]} but found {data[x0_col][counter]}."
                    )
                if data[x1_col][counter] != x1[j]:
                    raise ValueError(
                        f"Incorrect x1 ordering in {input_file} at line={counter}. "
                        f"Expected {x1[j]} but found {data[x1_col][counter]}."
                    )
                counter += 1

    # Build the values matrix.
    values = data[v_col].data.reshape((len(x0), len(x1)))

    return x0, x1, values


def _read_lclib_data_from_open_file(input_file):
    """Read SNANA's lclib data from a text file.

    Parameters
    ----------
    input_file : file
        The input data file containing SNANA's lclib data.

    Returns
    -------
    curves : list of astropy.table.Table
        A list of Astropy Tables, each representing a light curve.
    """
    colnames = []
    curves = []
    meta = {}
    current_model = {}
    parnames = []
    in_doc_block = False

    for l_num, line in enumerate(input_file):
        # Strip out the trailing comment. Then skip lines that are either
        # empty or do not contain a key-value pair.
        line = line.split("#")[0].strip()
        if not line or ":" not in line:
            continue

        # Split the line into key and value.
        key, value = line.split(":", 1)
        value = value.strip()

        # Handle the keys corresponding to a documentation block.
        if key == "DOCUMENTATION":
            in_doc_block = True
        elif key == "DOCUMENTATION_END":
            in_doc_block = False
        if in_doc_block:
            # If we are in a documentation block, just continue to the next line.
            continue

        if key == "COMMENT":
            continue  # Skip comments.
        elif key == "FILTERS":
            # Create a list of data columns with time and each filter.
            colnames = ["time"]
            for c in value:
                colnames.append(c)
        elif key == "END_EVENT":
            curr_id = meta.get("id", "")
            if curr_id != value:
                raise ValueError(f"Event ID mismatch (line {l_num}): found {value}, expected {curr_id}.")

            # Save the table we have so far.
            curves.append(Table(current_model, meta=meta))
        elif key == "START_EVENT":
            if len(colnames) == 0:
                raise ValueError(f"Error on line= {l_num}: No filters defined.")

            # Start a new light curve, but resetting the lists of data from the columns.
            current_model["type"] = []  # Initialize the type list.
            for col in colnames:
                current_model[col] = []
            meta["id"] = value
        elif key == "S" or key == "T":
            # Save an observation or template to the current lightcurve.
            current_model["type"].append(key)  # Get the type from the key.

            # Get the time and magnitudes from the columns.
            col_vals = value.split()
            if len(col_vals) != len(colnames):
                raise ValueError(f"Expected {len(colnames)} values on line={l_num}: {col_vals}")
            for col_idx, col in enumerate(colnames):
                current_model[col].append(float(col_vals[col_idx]))
        elif key == "MODEL_PARNAMES" or key == "MODEL_PARAMETERS":
            parnames = value.split(",")
        elif key == "PARVAL":
            if "," in value:
                all_vals = value.split(",")
            else:
                all_vals = value.split()

            if len(all_vals) != len(parnames):
                raise ValueError(f"Expected {len(parnames)} parameter values on line={l_num}: {all_vals}")
            meta["PARVAL"] = {key: value for key, value in zip(parnames, all_vals, strict=False)}
        else:
            # Save everything else to the meta dictionary.
            meta[key] = value

    return curves


def read_lclib_data(input_file):
    """Read SNANA's LCLIB data from a text file.

    Parameters
    ----------
    input_file : str or Path
        The path to the SNANA LCLIB data file.

    Returns
    -------
    curves : list of astropy.table.Table
        A list of Astropy Tables, each representing a light curve.
    """
    input_file = Path(input_file)
    logging.debug(f"Loading SNANA LCLIB data from {input_file}")
    if not input_file.is_file():
        raise FileNotFoundError(f"File {input_file} not found.")

    # Use the file suffix to determine how to read the file.
    suffix = input_file.suffix.lower()
    if suffix in [".gz", ".gzip"]:
        # Open as a gzipped text file.
        with gzip.open(input_file, "rt") as file_ptr:
            curves = _read_lclib_data_from_open_file(file_ptr)
    elif suffix in [".dat", ".txt", ".text"]:
        # Try to open the file as a regular text file.
        with open(input_file, "r") as file_ptr:
            curves = _read_lclib_data_from_open_file(file_ptr)
    else:
        raise ValueError(f"Unsupported file format: {suffix}.")

    return curves
