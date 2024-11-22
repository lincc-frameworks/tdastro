import numpy as np
import pytest
from tdastro.utils.io_utils import read_grid_data


def test_read_grid_data_good(grid_data_good_file):
    """Test that we can read a well formatted grid data file."""
    x0, x1, values = read_grid_data(grid_data_good_file, format="ascii.csv")
    x0_expected = np.array([0.0, 1.0, 2.0])
    x1_expected = np.array([1.0, 1.5])
    values_expected = np.array([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]])

    np.testing.assert_allclose(x0, x0_expected, atol=1e-5)
    np.testing.assert_allclose(x1, x1_expected, atol=1e-5)
    np.testing.assert_allclose(values, values_expected, atol=1e-5)


def test_read_grid_data_bad(grid_data_bad_file):
    """Test that we correctly handle a badly formatted grid data file."""
    # We load without a problem is validation is off.
    x0, x1, values = read_grid_data(grid_data_bad_file, format="ascii")
    assert values.shape == (3, 2)

    with pytest.raises(ValueError):
        _, _, _ = read_grid_data(grid_data_bad_file, format="ascii", validate=True)

    # We fail when loading a nonexistent file.
    with pytest.raises(FileNotFoundError):
        _, _, _ = read_grid_data("no_such_file_here", format="ascii", validate=True)
