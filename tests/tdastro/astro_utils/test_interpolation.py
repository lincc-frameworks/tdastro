import numpy as np
import pytest
from tdastro.astro_utils.interpolation import (
    has_uniform_step,
    interpolate_matrix_along_wavelengths,
    interpolate_transmission_table,
    interpolate_wavelengths,
)
from tdastro.astro_utils.passbands import Passband


def test_has_uniform_step():
    """Test has_uniform_step function."""
    a1 = np.array([1, 2, 3])
    assert has_uniform_step(a1)

    a2 = np.array([1, 2, 3, 40])
    assert not has_uniform_step(a2)

    a3 = np.array([])
    assert has_uniform_step(a3)

    a4 = np.array([1])
    assert has_uniform_step(a4)


def test_interpolate_wavelengths():
    """Test interpolate_wavelengths function."""
    wavelengths = np.array([1, 2, 3])
    new_grid_step = 0.5
    interp_wavelengths = interpolate_wavelengths(wavelengths, new_grid_step)
    assert np.allclose(interp_wavelengths, np.array([1.0, 1.5, 2.0, 2.5, 3.0]))

    # Test non-uniform grid # TODO would this be better as its own unit test? what's the standard for this?
    wavelengths = np.array([1, 2, 4])
    new_grid_step = 0.5
    interp_wavelengths = interpolate_wavelengths(wavelengths, new_grid_step)
    assert np.allclose(interp_wavelengths, np.array([1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]))

    # Test not strictly increasing grid
    wavelengths = np.array([1, 2, 1])
    new_grid_step = 0.5
    with pytest.raises(ValueError):
        interpolate_wavelengths(wavelengths, new_grid_step)


def test_interpolate_matrix_along_wavelengths():
    """Test interpolate_matrix_along_wavelengths function."""
    fluxes = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    wavelengths = np.array([1, 2, 3])
    new_grid_step = 0.5
    interp_wavelengths = interpolate_wavelengths(wavelengths, new_grid_step)
    interp_fluxes = interpolate_matrix_along_wavelengths(fluxes, wavelengths, interp_wavelengths)
    assert np.allclose(
        interp_fluxes,
        np.array([[1.0, 1.5, 2.0, 2.5, 3.0], [4.0, 4.5, 5.0, 5.5, 6.0], [7.0, 7.5, 8.0, 8.5, 9.0]]),
    )  # TODO change to np assert? (and above)


def test_interpolate_transmission_table():
    """Test interpolate_transmission_table function."""
    wavelengths = np.array([0, 10, 20])
    transmissions = np.array([1, 2, 3])
    table = np.stack((wavelengths, transmissions), axis=-1)
    new_wavelengths = np.linspace(0, 20, 7)
    interp_transmissions = interpolate_transmission_table(table, new_wavelengths)
    assert np.allclose(
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


def test_interpolate_passband(tmp_path):
    """Test interpolate_passband function."""
    # Initial values
    flux_density_matrix = np.array([[1.0, 1.0, 1.0], [1.0, 10.0, 1.0], [1.0, 0.0, 1.0]])
    wavelengths_angstrom = np.array([100.0, 105.0, 110.0])

    # New values
    new_grid_step = 1.0

    # Make passband
    survey = "TEST"
    band_label = "a"
    table_path = tmp_path / f"{survey}_{band_label}.dat"
    transmission_table = "100 0.5\n" "110 1.0\n"
    with open(table_path, "w") as f:
        f.write(transmission_table)

    a_band = Passband(survey, band_label, table_path=table_path)

    # Interpolate
    interpolated_matrix, interpolated_vector, interpolated_t_table = a_band._interpolate(
        flux_density_matrix,
        wavelengths_angstrom,
        a_band.normalized_transmission_table,
        target_grid_step=new_grid_step,
    )

    # Assert shape/size
    assert interpolated_matrix.shape == (3, 11)
    assert len(interpolated_vector) == 11
    assert len(interpolated_t_table) == 11

    # Assert values
    expected_matrix = np.array(
        [
            [1.0] * 11,
            [1.0, 2.8, 4.6, 6.4, 8.2, 10.0, 8.2, 6.4, 4.6, 2.8, 1.0],
            [1.0, 0.8, 0.6, 0.4, 0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        ]
    )
    assert np.allclose(interpolated_matrix, expected_matrix)

    # TODO add a test with a weird grid upper bound
