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


def create_grid(bounds: np.ndarray | tuple, grid_step: float, increasing=True) -> np.ndarray:
    """Create a new grid with given bounds and grid step.

    If the interval between the original grid's upper and lower bounds is a multiple of the new grid step,
    the upper bound will be included in the new grid.

    Parameters
    ----------
    bounds : np.ndarray | tuple
        The bounds of the grid. If a tuple, it should be (lower_bound, upper_bound). If an array, the first
        and last elements will be used as the lower and upper bounds, respectively.
    grid_step : float
        The step size of the new grid.
    increasing : bool, optional
        Whether the grid should include checks that bounds are increasing. Default is True.

    Returns
    -------
    np.ndarray
        The new grid.
    """
    # Get bounds
    if isinstance(bounds, tuple):
        lower_bound, upper_bound = bounds
    else:
        lower_bound, upper_bound = bounds[0], bounds[-1]

    # Check lower bound is less than upper bound
    if increasing and lower_bound >= upper_bound:
        raise ValueError("Lower bound must be less than upper bound, or disable 'increasing' check.")

    # Generate new grid, including the original upper bound if step size is a divisor of target interval
    if (upper_bound - lower_bound) % grid_step == 0:
        new_grid = np.arange(lower_bound, upper_bound + grid_step, grid_step)
    else:
        new_grid = np.arange(lower_bound, upper_bound, grid_step)
    return new_grid


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
