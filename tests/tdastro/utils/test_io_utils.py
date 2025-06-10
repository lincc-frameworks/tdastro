import numpy as np
import pytest
from tdastro.utils.io_utils import read_grid_data, read_snana_lc


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


def test_read_snana_lc(test_data_dir):
    """Test reading a SNANA light curve from a text file."""
    lc_file = test_data_dir / "test_snana_lc.dat"
    lc, spec = read_snana_lc(lc_file)
    assert len(lc) == 10
    assert len(spec) == 2
    assert len(spec[0]) == 5
    assert len(spec[1]) == 8

    # Check that the lightcurve has the expected columns.
    assert "MJD" in lc.colnames
    assert "FLT" in lc.colnames
    assert "FIELD" in lc.colnames
    assert "FLUXCAL" in lc.colnames
    assert "FLUXCALERR" in lc.colnames
    assert "MAG" in lc.colnames
    assert "MAGERR" in lc.colnames

    # Check that the specrum tables have the expected columns.
    expected_spec_cols = ["LAMMIN", "LAMMAX", "FLAM", "FLAMERR", "SPECFLAG"]
    for spec_table in spec:
        for col_name in expected_spec_cols:
            assert col_name in spec_table.colnames
