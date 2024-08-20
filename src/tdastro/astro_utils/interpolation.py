import numpy as np


def get_grid_step(grid: np.ndarray) -> float:
    """Get the step size of a grid.

    Parameters
    ----------
    grid : np.ndarray
        The grid.

    Returns
    -------
    float
        The step size of the grid.
    """
    # Get the step sizes of the grid
    step_size = np.diff(grid)

    # Check if the step sizes are consistent across the grid
    if not np.allclose(step_size, step_size[0]):
        raise ValueError("The grid step sizes are not consistent.")

    # Return the step size
    return step_size[0]


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
    return np.isclose(get_grid_step(grid1), get_grid_step(grid2))
