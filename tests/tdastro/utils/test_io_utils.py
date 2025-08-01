import numpy as np
import pytest
from tdastro.utils.io_utils import read_grid_data, read_lclib_data


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


def test_read_lclib_data(test_data_dir):
    """Test reading a SNANA LCLIB data from a text file."""
    lc_file = test_data_dir / "test_lclib_data.TEXT"
    curves = read_lclib_data(lc_file)
    assert len(curves) == 3

    expected_cols = ["type", "time", "u", "g", "r", "i", "z"]
    expected_len = [20, 20, 15]
    expected_param = [
        {"TYPE": "1", "OTHER": "1"},
        {"TYPE": "1", "OTHER": "2"},
        {"TYPE": "6", "OTHER": "7"},
    ]
    for idx, curve in enumerate(curves):
        assert len(curve) == expected_len[idx]
        assert int(curve.meta["id"]) == idx
        assert curve.meta["RECUR_CLASS"] == "RECUR-PERIODIC"
        for key, value in expected_param[idx].items():
            assert curve.meta["PARVAL"][key] == value

        # We did not pick up anything in the documentation block.
        assert "PURPOSE" not in curve.meta

        # Check that the expected columns are present and type is "S" or "T"
        for col in expected_cols:
            assert col in curve.colnames
        assert np.all((curve["type"].data == "S") | (curve["type"].data == "T"))
