import numpy as np


def get_grid_step(grid: np.ndarray) -> float:
    """Get the step size of a grid if it is uniform; otherwise, return None.

    Parameters
    ----------
    grid : np.ndarray
        The grid.

    Returns
    -------
    float or None
        The step size of the grid if it is uniform, otherwise None.
    """
    if np.allclose(np.diff(grid), np.diff(grid)[0]):
        return np.diff(grid)[0]
    return None


def grids_have_same_step(grid1: np.ndarray, grid2: np.ndarray) -> bool:
    """Check if two grids have the same step size.

    Parameters
    ----------
    grid1 : np.ndarray
        The first grid.
    grid2 : np.ndarray
        The second grid.

    Returns
    -------
    bool
        True if the grids have the same step size, False otherwise.
    """
    # If both grids have the same length, we can account for non-uniform grids
    if len(grid1) == len(grid2):
        return np.allclose(np.diff(grid1), np.diff(grid2))
    # If the grids have different lengths, they must be uniform to have the same step size
    return get_grid_step(grid1) is not None and get_grid_step(grid1) == get_grid_step(grid2)


def interpolate_wavelengths(wavelengths: np.ndarray, new_grid_step: float) -> np.ndarray:
    """Interpolate a grid of wavelengths to a new grid with a given step size.

    TODO add note about whether the upper bound is included in the new grid

    Parameters
    ----------
    wavelengths : np.ndarray
        The grid of wavelengths (must be strictly increasing).
    new_grid_step : float
        The step size of the new grid.

    Returns
    -------
    np.ndarray
        The interpolated grid.
    """
    # Check that wavelengths are strictly increasing
    if not np.all(np.diff(wavelengths) > 0):
        raise ValueError("Wavelengths must be strictly increasing.")
    new_grid = np.arange(wavelengths[0], wavelengths[-1] + new_grid_step, new_grid_step)
    return new_grid


def interpolate_transmission_table(table: np.ndarray, new_wavelengths: np.ndarray) -> np.ndarray:
    """Interpolate a 2 column table (such that the first column is x and the second column is y; eg,
    a transmission table) to a new grid of x values.

    Parameters
    ----------
    table : np.ndarray
        The table to interpolate. Must be a 2D array with shape (n, 2).
    new_wavelengths : np.ndarray
        The new wavelengths grid to interpolate to. Must be a 1D array of length m.

    Returns
    -------
    np.ndarray
        The interpolated table, with shape (m, 2).
    """
    if table.shape[1] != 2:
        raise ValueError("Table must have exactly 2 columns.")
    interpolated_transmissions = np.interp(new_wavelengths, table[:, 0], table[:, 1])
    return np.column_stack((new_wavelengths, interpolated_transmissions))


def interpolate_matrix_along_wavelengths(
    matrix: np.ndarray, wavelengths: np.ndarray, new_wavelengths: np.ndarray
) -> np.ndarray:
    """Interpolate a matrix with columns corresponding to a vector of wavelengths.

    Parameters
    ----------
    matrix : np.ndarray
        A 2D matrix to interpolate, with size (n, m).
    wavelengths : np.ndarray
        The wavelengths corresponding to the matrix, with size m.
    new_wavelengths : np.ndarray
        The new grid of wavelengths to interpolate to, with size p.

    Returns
    -------
    np.ndarray
        The interpolated matrix, with size (n, p).
    """
    if matrix.shape[1] != len(wavelengths):
        raise ValueError(
            f"Matrix has {matrix.shape[1]} columns, but {len(wavelengths)} wavelengths were given."
        )
    new_matrix = np.array([np.interp(new_wavelengths, wavelengths, row) for row in matrix])
    return new_matrix


"""
TODO
- Different interpolation methods?
"""
