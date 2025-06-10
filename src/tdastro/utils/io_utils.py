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
        Default = 'ascii'
    validate : bool
        Perform additional validation on the input data.
        Default = False

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


def _read_snana_lc_from_text_file(file_ptr):
    """Read a SNANA outut lightcurve from a file.

    Parameters
    ----------
    file_ptr : file object
        The file pointer to read the SNANA light curve from.

    Returns
    -------
    lc : astropy.table.Table
        An Astropy Table containing the light curve data.
    spec_tables : list
        A list of spectra tables (if present in the file).
    """
    line_num = 0

    # Data for the observation table.
    meta = {}
    table_data = {}
    col_names = []

    # Data for any spectrum tables.
    spec_col_names = []
    spec_tables = []
    current_spec_data = {}
    current_spec_meta = {}

    for line in file_ptr:
        # Strip out the trailing comment. Then skip lines that are either
        # empty or do not contain a key-value pair.
        line = line.split("#")[0].strip()
        if not line or ":" not in line:
            continue

        # Split the line into key and value.
        key, value = line.split(":", 1)

        # The SNANA output file consists of a few blocks: general metadata,
        # observations in a table, and possibly multiple spectrum tables.
        # We determine which block we are in by the key.
        if key == "VARLIST":
            # If we are reading the observation table header, initialize the table_data
            col_names = value.split()
            for var_name in col_names:
                table_data[var_name] = []
        elif key == "OBS":
            # If we are reading the obsertvation table data, split the line into
            # values for each variable.
            values = value.split()
            if len(values) != len(col_names):
                raise ValueError(
                    f"Data error at line {line_num}: Expected {len(col_names)} values "
                    f" but found {len(values)}."
                )
            for var_name, val in zip(col_names, values, strict=False):
                table_data[var_name].append(val)
        elif key == "VARNAMES_SPEC":
            # If we are reading a spectrum table header, initialize the spec_col_names
            spec_col_names = value.split()
        elif key == "SPECTRUM_ID":
            # If we are starting a new spectrum table, save the current one
            # and reset the current_spec_data.
            if len(current_spec_data) > 0:
                spec_tables.append(Table(current_spec_data, meta=current_spec_meta))
            current_spec_data = {key: [] for key in spec_col_names}
            current_spec_meta = {"SPECTRUM_ID": value.strip()}
        elif key.startswith("SPECTRUM_"):
            # Save the spectrum metadata.
            current_spec_meta[key] = value.strip()
        elif key == "SPEC":
            # Save the spectrum data lines.
            values = value.split()
            if len(values) != len(spec_col_names):
                raise ValueError(
                    f"Data error at line {line_num}: Expected {len(spec_col_names)} values "
                    f" but found {len(values)}."
                )
            for var_name, val in zip(spec_col_names, values, strict=False):
                current_spec_data[var_name].append(val)
        else:
            # Save everything else to the meta dictionary.
            meta[key] = value.strip()

        line_num += 1

    # Put the observation data and the last spectrum data into tables.
    lc = Table(table_data, meta=meta)
    if len(current_spec_data) > 0:
        spec_tables.append(Table(current_spec_data, meta=current_spec_meta))

    return lc, spec_tables


def read_snana_lc(file_path):
    """Read a SNANA output light curve from a file.

    Parameters
    ----------
    file_path : str or Path
        The path to the SNANA light curve file.

    Returns
    -------
    lc : astropy.table.Table
        An Astropy Table containing the light curve data.
    spec_tables : list
        A list of spectra tables (if present in the file).
    """
    logging.debug(f"Loading SNANA light curve from {file_path}")
    input_file = Path(file_path)
    if not input_file.is_file():
        raise FileNotFoundError(f"File {file_path} not found.")

    if ".fits" in file_path.suffixes:
        lc = None
        raise NotImplementedError("Reading SNANA light curves from FITS files is not implemented yet.")
    elif file_path.suffix == ".gz":
        # Open as a gzipped text file.
        with gzip.open(input_file, "rt") as file_ptr:
            lc, spec_tables = _read_snana_lc_from_text_file(file_ptr)
    else:
        # Try to open the file as a regular text file.
        with open(input_file, "r") as file_ptr:
            lc, spec_tables = _read_snana_lc_from_text_file(file_ptr)

    return lc, spec_tables
