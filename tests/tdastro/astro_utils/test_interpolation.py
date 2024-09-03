import numpy as np
import pytest
from tdastro.astro_utils.interpolation import (
    create_grid,
    interpolate_matrix_along_wavelengths,
    interpolate_transmission_table,
)
from tdastro.astro_utils.passbands import Passband


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


def _make_passband(transmission_table, path):
    """Helper function to create a Passband object from a transmission table."""
    survey = "TEST"
    band_label = "a"
    table_path = path / f"{survey}_{band_label}.dat"
    with open(table_path, "w") as f:
        f.write(transmission_table)
    a_band = Passband(survey, band_label, table_path=table_path)
    return a_band


def test_interpolate_passband_simple(tmp_path):
    """Test _interpolate function."""
    flux_density_matrix = np.array([[1.0, 1.0, 1.0], [1.0, 10.0, 1.0], [1.0, 0.0, 1.0]])
    wavelengths_angstrom = np.array([100.0, 105.0, 110.0])

    transmission_table = "100 0.5\n" "110 1.0\n"
    a_band = _make_passband(transmission_table, tmp_path)

    # Interpolate to grid of 1.0
    new_grid_step = 1.0
    interpolated_fluxes, interpolated_wavelengths, interpolated_t_table = a_band._interpolate(
        flux_density_matrix,
        wavelengths_angstrom,
        new_grid_step,
    )

    # Assert shape/size
    assert interpolated_fluxes.shape == (3, 11)
    assert len(interpolated_wavelengths) == 11
    assert len(interpolated_t_table) == 11

    # Assert values
    np.testing.assert_allclose(interpolated_wavelengths, np.arange(100, 111))
    np.testing.assert_allclose(interpolated_fluxes[0], [1.0] * 11)
    assert np.isclose(interpolated_fluxes[1][0], 1.0)
    assert np.isclose(interpolated_fluxes[1][5], 10.0)
    assert np.isclose(interpolated_fluxes[1][10], 1.0)
    assert np.isclose(interpolated_fluxes[2][0], 1.0)
    assert np.isclose(interpolated_fluxes[2][5], 0.0)
    assert np.isclose(interpolated_fluxes[2][10], 1.0)

    # Check the original data was not modified
    assert flux_density_matrix.shape == (3, 3)
    assert wavelengths_angstrom.shape == (3,)


def test_interpolate_passband_coarser_grid(tmp_path):
    """Test _interpolate function if target grid is coarser than current grid."""
    flux_density_matrix = np.array([[1.0, 1.0, 1.0], [1.0, 10.0, 1.0], [1.0, 0.0, 1.0]])
    wavelengths_angstrom = np.array([100.0, 105.0, 110.0])

    transmission_table = "100 0.5\n" "110 1.0\n"
    a_band = _make_passband(transmission_table, tmp_path)

    # Interpolate to grid of 10.0
    new_grid_step = 10.0
    interpolated_fluxes, interpolated_wavelengths, interpolated_t_table = a_band._interpolate(
        flux_density_matrix,
        wavelengths_angstrom,
        new_grid_step,
    )

    # Assert shape/size
    assert interpolated_fluxes.shape == (3, 2)
    assert len(interpolated_wavelengths) == 2
    assert len(interpolated_t_table) == 3

    # Assert values
    assert np.isclose(interpolated_wavelengths[0], 100.0)
    assert np.isclose(interpolated_wavelengths[1], 110.0)
    np.testing.assert_allclose(interpolated_fluxes[0], [1.0, 1.0])
    np.testing.assert_allclose(interpolated_fluxes[1], [1.0, 1.0])
    np.testing.assert_allclose(interpolated_fluxes[2], [1.0, 1.0])


def test_interpolate_passband_none_needed(tmp_path):
    """Test _interpolate function if no interpolation is needed for target grid."""
    flux_density_matrix = np.array([[1.0, 1.0, 1.0], [1.0, 10.0, 1.0], [1.0, 0.0, 1.0]])
    wavelengths_angstrom = np.array([100.0, 105.0, 110.0])

    transmission_table = "100 0.5\n" "110 1.0\n"
    a_band = _make_passband(transmission_table, tmp_path)

    # Interpolate to grid of 5.0 (no interpolation)
    new_grid_step = 5.0
    interpolated_fluxes, interpolated_wavelengths, interpolated_t_table = a_band._interpolate(
        flux_density_matrix,
        wavelengths_angstrom,
        new_grid_step,
    )

    # Assert shape/size
    assert interpolated_fluxes.shape == (3, 3)
    assert len(interpolated_wavelengths) == 3
    assert len(interpolated_t_table) == 3

    # Assert values
    np.testing.assert_allclose(interpolated_wavelengths, wavelengths_angstrom)
    np.testing.assert_allclose(interpolated_fluxes, flux_density_matrix)
    np.testing.assert_allclose(interpolated_t_table, np.array([100.0, 0.5], [110.0, 1.0]))

    # Assert scipy.interpolate.CubicSpline was not called
    # TODO


def test_extrapolate_fluxes(tmp_path):
    """Test flux matrix extrapolation performed in _interpolate function."""
    flux_density_matrix = np.array([[1.0, 1.0, 1.0], [1.0, 10.0, 1.0], [1.0, 0.0, 1.0]])
    wavelengths_angstrom = np.array([100.0, 105.0, 110.0])

    transmission_table = "100 0.5\n105 0.75\n110 1.0\n115 0.75\n120 0.5\n"
    a_band = _make_passband(transmission_table, tmp_path)

    # Interpolate to grid of 5.0 (no interpolation needed, but extrapolation is)
    new_grid_step = 5.0
    interpolated_fluxes, interpolated_wavelengths, interpolated_t_table = a_band._interpolate(
        flux_density_matrix,
        wavelengths_angstrom,
        new_grid_step,
    )

    # Assert shape/size
    assert interpolated_fluxes.shape == (3, 5)
    assert len(interpolated_wavelengths) == 5
    assert len(interpolated_t_table) == 5

    # Assert values
    np.testing.assert_allclose(interpolated_wavelengths, [100.0, 105.0, 110.0, 115.0, 120.0])
    np.testing.assert_allclose(interpolated_fluxes[0], [1.0, 1.0, 1.0, 1.0, 1.0])
    np.testing.assert_allclose(interpolated_fluxes[1][:3], [1.0, 10.0, 1.0])
    np.testing.assert_allclose(interpolated_fluxes[2][:3], [1.0, 0.0, 1.0])
    np.testing.assert_allclose(interpolated_t_table, a_band.normalized_transmission_table)


def test_truncate_fluxes(tmp_path):
    """Test flux matrix truncation performed in _interpolate function."""
    flux_density_matrix = np.array([[1.0, 1.0, 1.0], [1.0, 10.0, 1.0], [1.0, 0.0, 1.0]])
    wavelengths_angstrom = np.array([100.0, 105.0, 110.0])

    transmission_table = "100 0.5\n105 0.75\n"
    a_band = _make_passband(transmission_table, tmp_path)

    # Interpolate to grid of 5.0 (no interpolation needed, but truncation is)
    new_grid_step = 5.0
    interpolated_fluxes, interpolated_wavelengths, interpolated_t_table = a_band._interpolate(
        flux_density_matrix,
        wavelengths_angstrom,
        new_grid_step,
    )

    # Assert shape/size
    assert interpolated_fluxes.shape == (3, 2)
    assert len(interpolated_wavelengths) == 2
    assert len(interpolated_t_table) == 2

    # Assert values
    np.testing.assert_allclose(interpolated_wavelengths, [100.0, 105.0])
    np.testing.assert_allclose(interpolated_fluxes[0], [1.0, 1.0])
    np.testing.assert_allclose(interpolated_fluxes[1], [1.0, 10.0])
    np.testing.assert_allclose(interpolated_fluxes[2], [1.0, 0.0])
    np.testing.assert_allclose(interpolated_t_table, a_band.normalized_transmission_table)


def test_truncate_and_interpolate_fluxes(tmp_path):
    """Test flux matrix truncation performed in _interpolate function."""
    flux_density_matrix = np.array([[1.0, 1.0, 1.0], [1.0, 10.0, 1.0], [1.0, 0.0, 1.0]])
    wavelengths_angstrom = np.array([100.0, 105.0, 110.0])

    transmission_table = "100 0.5\n105 0.75\n"
    a_band = _make_passband(transmission_table, tmp_path)

    # Interpolate to grid of 1.0 (both interpolation and truncation needed)
    new_grid_step = 1.0
    interpolated_fluxes, interpolated_wavelengths, interpolated_t_table = a_band._interpolate(
        flux_density_matrix,
        wavelengths_angstrom,
        new_grid_step,
    )

    # Assert shape/size
    assert interpolated_fluxes.shape == (3, 6)
    assert len(interpolated_wavelengths) == 6
    assert len(interpolated_t_table) == 6

    # Assert values
    np.testing.assert_allclose(interpolated_wavelengths, [100.0, 101.0, 102.0, 103.0, 104.0, 105.0])
    np.testing.assert_allclose(interpolated_fluxes[0], [1.0] * 6)
    assert np.isclose(interpolated_fluxes[1][0], 1.0)
    assert np.isclose(interpolated_fluxes[1][5], 10.0)
    assert np.isclose(interpolated_fluxes[2][0], 1.0)
    assert np.isclose(interpolated_fluxes[2][5], 0.0)
    assert np.isclose(interpolated_t_table[0][0], 100.0)
    assert np.isclose(interpolated_t_table[0][1], a_band.normalized_transmission_table[0][1])
    assert np.isclose(interpolated_t_table[5][0], 105.0)
    assert np.isclose(interpolated_t_table[5][1], a_band.normalized_transmission_table[1][1])


def test_extrapolate_out_of_range(tmp_path):
    """Test flux matrix extrapolation when given fluxes are completely out of range for passband."""
    flux_density_matrix = np.array([[1.0, 1.0, 1.0], [1.0, 10.0, 1.0], [1.0, 0.0, 1.0]])
    wavelengths_angstrom = np.array([200.0, 205.0, 210.0])

    transmission_table = "100 0.5\n110 1.0"
    a_band = _make_passband(transmission_table, tmp_path)

    # Interpolate to grid of 5.0 (no interpolation needed, but extrapolation is)
    new_grid_step = 5.0
    interpolated_fluxes, interpolated_wavelengths, interpolated_t_table = a_band._interpolate(
        flux_density_matrix,
        wavelengths_angstrom,
        new_grid_step,
    )

    # Well, here we see the results of the extrapolation.
    # Don't love the  [-3959. -3590. -3239.] fluxes.
    # TODO
    with np.printoptions(precision=2, suppress=True):
        print(interpolated_fluxes)
        print(interpolated_wavelengths)
        print(interpolated_t_table)
