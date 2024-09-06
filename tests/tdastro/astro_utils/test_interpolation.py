import numpy as np
import pytest
from tdastro.astro_utils.interpolation import (
    create_grid,
    get_grid_step,
    interpolate_matrix_along_wavelengths,
    interpolate_transmission_table,
)


def test_get_grid_step():
    """Test get_grid_step function."""
    # Test uniform grid
    grid = np.array([1, 2, 3])
    grid_step = get_grid_step(grid)
    assert grid_step == 1

    # Test non-uniform grid
    grid = np.array([1, 2, 4])
    grid_step = get_grid_step(grid)
    assert grid_step is None

    # Test non-uniform grid with equal differences
    grid = np.array([1, 2, 2])
    grid_step = get_grid_step(grid)
    assert grid_step is None


def test_create_grid():
    """Test create_grid function."""
    wavelengths = np.array([1, 2, 3])
    new_grid_step = 0.5
    interp_wavelengths = create_grid(wavelengths, new_grid_step)
    np.testing.assert_allclose(interp_wavelengths, np.array([1.0, 1.5, 2.0, 2.5, 3.0]))

    # Test non-uniform grid
    wavelengths = np.array([1, 2, 4])
    new_grid_step = 0.5
    interp_wavelengths = create_grid(wavelengths, new_grid_step)
    np.testing.assert_allclose(interp_wavelengths, np.array([1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]))

    # Test not strictly increasing grid
    wavelengths = np.array([1, 2, 1])
    new_grid_step = 0.5
    with pytest.raises(ValueError):
        create_grid(wavelengths, new_grid_step)

    # Test upper bound is included where the step size is a divisor of the upper bound
    wavelengths = np.array([1, 4])
    new_grid_step = 1
    interp_wavelengths = create_grid(wavelengths, new_grid_step)
    np.testing.assert_allclose(interp_wavelengths, np.array([1.0, 2.0, 3.0, 4.0]))

    # Test upper bound not included where the step size is not a divisor of the upper bound
    wavelengths = np.array([1, 4])
    new_grid_step = 2
    interp_wavelengths = create_grid(wavelengths, new_grid_step)
    np.testing.assert_allclose(interp_wavelengths, np.array([1.0, 3.0]))


def test_interpolate_transmission_table():
    """Test interpolate_transmission_table function."""
    wavelengths = np.array([0, 10, 20])
    transmissions = np.array([1, 2, 3])
    table = np.stack((wavelengths, transmissions), axis=-1)
    new_wavelengths = np.linspace(0, 20, 7)
    interp_transmissions = interpolate_transmission_table(table, new_wavelengths)
    np.testing.assert_allclose(
        interp_transmissions,
        np.array(
            [
                [0.0, 1.0],
                [3.33333333, 1.33333333],
                [6.66666667, 1.66666667],
                [10.0, 2.0],
                [13.33333333, 2.33333333],
                [16.66666667, 2.66666667],
                [20.0, 3.0],
            ]
        ),
    )

    # TODO test edge cases
    # TODO moved ths into passband's _interpolate_or_downsample_transmission_table


def test_interpolate_matrix_along_wavelengths():
    """Test interpolate_matrix_along_wavelengths function."""
    fluxes = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    wavelengths = np.array([1, 2, 3])
    new_grid_step = 0.5
    interp_wavelengths = create_grid(wavelengths, new_grid_step)
    interp_fluxes = interpolate_matrix_along_wavelengths(fluxes, wavelengths, interp_wavelengths)
    np.testing.assert_allclose(
        interp_fluxes,
        np.array([[1.0, 1.5, 2.0, 2.5, 3.0], [4.0, 4.5, 5.0, 5.5, 6.0], [7.0, 7.5, 8.0, 8.5, 9.0]]),
    )
