import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from tdastro.astro_utils.opsim import (
    OpSim,
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

    # We can access columns directly as though it was a table.
    assert np.allclose(ops_data["fieldRA"], values["fieldRA"])
    assert np.allclose(ops_data["fieldDec"], values["fieldDec"])
    assert np.allclose(ops_data["observationStartMJD"], values["observationStartMJD"])

    # We can create an OpSim directly from the dictionary as well.
    ops_data2 = OpSim(pdf)
    assert len(ops_data2) == 5
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
        filename = os.path.join(dir_name, "test_write_read_opsim.db")

        # The opsim does not exist until we write it.
        assert not Path(filename).is_file()
        with pytest.raises(FileNotFoundError):
            _ = OpSim.from_db(filename)

        # We can write the opsim db.
        ops_data.write_opsim_table(filename)
        assert Path(filename).is_file()

        # We can reread the opsim db.
        ops_data2 = OpSim.from_db(filename)
        assert len(ops_data2) == 5
        assert np.allclose(values["observationStartMJD"], ops_data2["observationStartMJD"].to_numpy())
        assert np.allclose(values["fieldRA"], ops_data2["fieldRA"].to_numpy())
        assert np.allclose(values["fieldDec"], ops_data2["fieldDec"].to_numpy())

        # We cannot overwrite unless we set overwrite=True
        with pytest.raises(ValueError):
            ops_data.write_opsim_table(filename, overwrite=False)
        ops_data.write_opsim_table(filename, overwrite=True)


def test_obsim_range_search():
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
