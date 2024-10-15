import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from tdastro.astro_utils.mag_flux import mag2flux
from tdastro.astro_utils.opsim import (
    OpSim,
    opsim_add_random_data,
    oversample_opsim,
)


def test_create_opsim():
    """Create a minimal OpSim object and perform basic queries."""
    values = {
        "observationStartMJD": np.array([0.0, 1.0, 2.0, 3.0, 4.0]),
        "fieldRA": np.array([15.0, 30.0, 15.0, 0.0, 60.0]),
        "fieldDec": np.array([-10.0, -5.0, 0.0, 5.0, 10.0]),
        "zp_nJy": np.ones(5),
    }
    pdf = pd.DataFrame(values)

    ops_data = OpSim(pdf)
    assert len(ops_data) == 5
    assert len(ops_data.columns) == 4

    # We can query which columns the OpSim has.
    assert ops_data.has_columns("fieldRA")
    assert ops_data.has_columns("dec")
    assert not ops_data.has_columns("is_good_obs")
    assert ops_data.has_columns(["ra", "dec", "time"])
    assert not ops_data.has_columns(["ra", "dec", "vacation_time"])

    # We can access columns directly as though it was a table.
    assert np.allclose(ops_data["fieldRA"], values["fieldRA"])
    assert np.allclose(ops_data["fieldDec"], values["fieldDec"])
    assert np.allclose(ops_data["observationStartMJD"], values["observationStartMJD"])

    # We can create an OpSim directly from the dictionary as well.
    ops_data2 = OpSim(pdf)
    assert len(ops_data2) == 5
    assert len(ops_data.columns) == 4
    assert np.allclose(ops_data2["fieldRA"], values["fieldRA"])
    assert np.allclose(ops_data2["fieldDec"], values["fieldDec"])
    assert np.allclose(ops_data2["observationStartMJD"], values["observationStartMJD"])


def test_create_opsim_custom_names():
    """Create a minimal OpSim object from alternate column names."""
    values = {
        "custom_time": np.array([0.0, 1.0, 2.0, 3.0, 4.0]),
        "custom_ra": np.array([15.0, 30.0, 15.0, 0.0, 60.0]),
        "custom_dec": np.array([-10.0, -5.0, 0.0, 5.0, 10.0]),
        "custom_zp": np.ones(5),
    }

    # Load fails if we use the default colmap.
    with pytest.raises(KeyError):
        _ = OpSim(values)

    # Load succeeds if we pass in a customer dictionary.
    colmap = {"ra": "custom_ra", "dec": "custom_dec", "time": "custom_time", "zp": "custom_zp"}
    ops_data = OpSim(values, colmap)
    assert len(ops_data) == 5


def test_opsim_add_columns():
    """Create a minimal OpSim object and perform basic queries."""
    values = {
        "observationStartMJD": np.array([0.0, 1.0, 2.0, 3.0, 4.0]),
        "fieldRA": np.array([15.0, 30.0, 15.0, 0.0, 60.0]),
        "fieldDec": np.array([-10.0, -5.0, 0.0, 5.0, 10.0]),
        "zp_nJy": np.ones(5),
    }
    pdf = pd.DataFrame(values)

    ops_data = OpSim(pdf)
    assert len(ops_data) == 5
    assert len(ops_data.columns) == 4

    # We can add a column of constant values from a scalar.
    assert "new_column1" not in ops_data.columns
    ops_data.add_column("new_column1", 10)
    assert "new_column1" in ops_data.columns
    assert np.allclose(ops_data["new_column1"], [10, 10, 10, 10, 10])

    # We can add a column of data from a list or array.
    assert "new_column2" not in ops_data.columns
    ops_data.add_column("new_column2", [1, 2, 3, 4, 5])
    assert "new_column2" in ops_data.columns
    assert np.allclose(ops_data["new_column2"], [1, 2, 3, 4, 5])

    # We fail if we add a column with the incorrect number of values.
    with pytest.raises(ValueError):
        ops_data.add_column("new_column3", [1, 2, 3, 4, 5, 6])

    # We fail if we try to overwrite a current column unless we
    # set overwrite=True.
    with pytest.raises(KeyError):
        ops_data.add_column("new_column1", 12)
    assert np.allclose(ops_data["new_column1"], [10, 10, 10, 10, 10])

    ops_data.add_column("new_column2", 12, overwrite=True)
    assert np.allclose(ops_data["new_column2"], [12, 12, 12, 12, 12])

    # We can add random data.
    opsim_add_random_data(ops_data, "new_column3", min_val=1.0, max_val=5.0)
    values = ops_data["new_column3"]
    assert len(np.unique(values)) == 5
    assert np.all(values >= 1.0)
    assert np.all(values <= 5.0)


def test_read_small_opsim(opsim_small):
    """Read in a small OpSim file from the testing data directory."""
    ops_data = OpSim.from_db(opsim_small)
    assert len(ops_data) == 300


def test_write_read_opsim():
    """Create a minimal opsim data frame, test that we can write it,
    and test that we can correctly read it back in."""

    # Create a fake opsim data frame with just time, RA, and dec.
    values = {
        "observationStartMJD": np.array([0.0, 1.0, 2.0, 3.0, 4.0]),
        "fieldRA": np.array([15.0, 30.0, 15.0, 0.0, 60.0]),
        "fieldDec": np.array([-10.0, -5.0, 0.0, 5.0, 10.0]),
        "zp_nJy": np.ones(5),
    }
    ops_data = OpSim(pd.DataFrame(values))

    with tempfile.TemporaryDirectory() as dir_name:
        file_path = Path(dir_name, "test_write_read_opsim.db")

        # The opsim does not exist until we write it.
        assert not file_path.is_file()
        with pytest.raises(FileNotFoundError):
            _ = OpSim.from_db(file_path)

        # We can write the opsim db.
        ops_data.write_opsim_table(file_path)
        assert file_path.is_file()

        # We can reread the opsim db.
        ops_data2 = OpSim.from_db(file_path)
        assert len(ops_data2) == 5
        assert np.allclose(values["observationStartMJD"], ops_data2["observationStartMJD"].to_numpy())
        assert np.allclose(values["fieldRA"], ops_data2["fieldRA"].to_numpy())
        assert np.allclose(values["fieldDec"], ops_data2["fieldDec"].to_numpy())

        # We cannot overwrite unless we set overwrite=True
        with pytest.raises(ValueError):
            ops_data.write_opsim_table(file_path, overwrite=False)
        ops_data.write_opsim_table(file_path, overwrite=True)


def test_opsim_range_search():
    """Test that we can extract the time, ra, and dec from an opsim data frame."""
    # Create a fake opsim data frame with just time, RA, and dec.
    values = {
        "observationStartMJD": np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]),
        "fieldRA": np.array([15.0, 15.0, 15.01, 15.0, 25.0, 24.99, 60.0, 5.0]),
        "fieldDec": np.array([-10.0, 10.0, 10.01, 9.99, 10.0, 9.99, -5.0, -1.0]),
        "zp_nJy": np.ones(8),
    }
    ops_data = OpSim(values)

    # Test single queries.
    assert set(ops_data.range_search(15.0, 10.0, 0.5)) == set([1, 2, 3])
    assert set(ops_data.range_search(25.0, 10.0, 0.5)) == set([4, 5])
    assert set(ops_data.range_search(15.0, 10.0, 100.0)) == set([0, 1, 2, 3, 4, 5, 6, 7])
    assert set(ops_data.range_search(15.0, 10.0, 1e-6)) == set([1])
    assert set(ops_data.range_search(15.02, 10.0, 1e-6)) == set()

    # Test a batched query.
    query_ra = np.array([15.0, 25.0, 15.0])
    query_dec = np.array([10.0, 10.0, 5.0])
    neighbors = ops_data.range_search(query_ra, query_dec, 0.5)
    assert len(neighbors) == 3
    assert set(neighbors[0]) == set([1, 2, 3])
    assert set(neighbors[1]) == set([4, 5])
    assert set(neighbors[2]) == set()


def test_opsim_get_observations():
    """Test that we can extract the time, ra, and dec from an opsim data frame."""
    # Create a fake opsim data frame with just time, RA, and dec.
    values = {
        "observationStartMJD": np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]),
        "fieldRA": np.array([15.0, 15.0, 15.01, 15.0, 25.0, 24.99, 60.0, 5.0]),
        "fieldDec": np.array([-10.0, 10.0, 10.01, 9.99, 10.0, 9.99, -5.0, -1.0]),
        "zp_nJy": np.ones(8),
    }
    ops_data = OpSim(values)

    # Test basic queries (all columns).
    obs = ops_data.get_observations(15.0, 10.0, 0.5)
    assert len(obs) == 4
    assert np.allclose(obs["observationStartMJD"], [1.0, 2.0, 3.0])

    obs = ops_data.get_observations(25.0, 10.0, 0.5)
    assert np.allclose(obs["observationStartMJD"], [4.0, 5.0])

    obs = ops_data.get_observations(15.0, 10.0, 100.0)
    assert np.allclose(obs["observationStartMJD"], [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])

    obs = ops_data.get_observations(15.0, 10.0, 1e-6)
    assert np.allclose(obs["observationStartMJD"], [1.0])

    obs = ops_data.get_observations(15.02, 10.0, 1e-6)
    assert len(obs["observationStartMJD"]) == 0

    # Test we can get a subset of columns.
    obs = ops_data.get_observations(15.0, 10.0, 0.5, cols=["observationStartMJD", "zp_nJy"])
    assert len(obs) == 2
    assert np.allclose(obs["observationStartMJD"], [1.0, 2.0, 3.0])

    # Test we can use the colmap names.
    obs = ops_data.get_observations(15.0, 10.0, 0.5, cols=["time", "ra"])
    assert len(obs) == 2
    assert np.allclose(obs["time"], [1.0, 2.0, 3.0])

    # Test we fail with an unrecognized column name.
    with pytest.raises(KeyError):
        _ = ops_data.get_observations(15.0, 10.0, 0.5, cols=["time", "custom_col"])


def test_opsim_docstring():
    """Test if OpSim class has a docstring"""
    assert OpSim.__doc__ is not None
    assert len(OpSim.__doc__) > 100


def test_read_opsim_shorten(opsim_shorten):
    """Read in a shorten OpSim file from the testing data directory."""
    ops_data = OpSim.from_db(opsim_shorten)
    assert len(ops_data) == 100


def test_opsim_flux_err_point_source(opsim_shorten):
    """Check if OpSim.flux_err_point_source is consistent with fiveSigmaDepth."""
    ops_data = OpSim.from_db(opsim_shorten)
    # fiveSigmaDepth is the 5-sigma limiting magnitude.
    flux = mag2flux(ops_data.table["fiveSigmaDepth"])
    expected_flux_err = flux / 5.0

    flux_err = ops_data.bandflux_error_point_source(flux, index=np.arange(len(ops_data)))

    # Tolerance is very high, we should investigate why the values are so different.
    np.testing.assert_allclose(flux_err, expected_flux_err, rtol=0.2)


def test_oversample_opsim(opsim_shorten):
    """Test that we can oversample an OpSim file."""
    opsim = OpSim.from_db(opsim_shorten)

    bands = ["g", "r"]
    ra, dec = 205.0, -57.0
    time_range = 60_000.0, 60_010.0
    delta_t = 0.01

    for strategy in ["darkest_sky", "random"]:
        oversampled = oversample_opsim(
            opsim,
            pointing=(ra, dec),
            time_range=time_range,
            delta_t=delta_t,
            bands=bands,
            strategy=strategy,
        )
        assert set(opsim.table.columns) == set(oversampled.table.columns), "columns are not the same"
        np.testing.assert_allclose(
            np.diff(oversampled["observationStartMJD"]), delta_t, err_msg="delta_t is not correct"
        )
        np.testing.assert_allclose(oversampled["fieldRA"], ra, err_msg="RA is not correct")
        np.testing.assert_allclose(oversampled["fieldDec"], dec, err_msg="Dec is not correct")
        assert np.all(oversampled["observationStartMJD"] >= time_range[0]), "time range is not correct"
        assert np.all(oversampled["observationStartMJD"] <= time_range[1]), "time range is not correct"
        assert set(oversampled["filter"]) == set(bands), "oversampled table has the wrong bands"
        assert (
            oversampled["skyBrightness"].unique().size >= oversampled["filter"].unique().size
        ), "there should be at least as many skyBrightness values as bands"
        assert oversampled["skyBrightness"].isna().sum() == 0, "skyBrightness has NaN values"


def test_fixture_oversampled_observations(oversampled_observations):
    """Test the fixture oversampled_observations."""
    assert len(oversampled_observations) == 36_500
    assert set(oversampled_observations["filter"]) == {"g", "r"}
    assert oversampled_observations["skyBrightness"].isna().sum() == 0
    assert oversampled_observations["skyBrightness"].unique().size >= 2
    assert np.all(oversampled_observations["observationStartMJD"] >= 61406.0)
    assert np.all(oversampled_observations["observationStartMJD"] <= 61771.0)
    np.testing.assert_allclose(oversampled_observations["fieldRA"], 0.0)
    np.testing.assert_allclose(oversampled_observations["fieldDec"], 0.0)
    np.testing.assert_allclose(np.diff(oversampled_observations["observationStartMJD"]), 0.01)
