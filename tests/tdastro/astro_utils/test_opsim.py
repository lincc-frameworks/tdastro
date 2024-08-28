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


def test_opsim_filter_on_value():
    """Check that we can filter indices based on matching data in another column."""
    values = {
        "observationStartMJD": np.array([0.0, 1.0, 2.0, 3.0, 4.0]),
        "fieldRA": np.array([15.0, 30.0, 15.0, 0.0, 60.0]),
        "fieldDec": np.array([-10.0, -5.0, 0.0, 5.0, 10.0]),
        "filter": np.array(["r", "g", "r", "g", "r"]),
        "zp_nJy": np.ones(5),
    }
    ops_data = OpSim(values)

    assert np.array_equal(
        ops_data.filter_on_value(np.array([0, 1, 2, 3, 4]), "filter", "r"),
        np.array([0, 2, 4]),
    )
    assert np.array_equal(
        ops_data.filter_on_value(np.array([0, 1, 2, 3, 4]), "filter", "g"),
        np.array([1, 3]),
    )
    assert np.array_equal(
        ops_data.filter_on_value(np.array([1, 2, 4]), "filter", "r"),
        np.array([2, 4]),
    )
    assert np.array_equal(
        ops_data.filter_on_value(np.array([1, 2, 4]), "filter", "g"),
        np.array([1]),
    )

    # Use column mapping "time" -> "observationStartMJD"
    assert np.array_equal(
        ops_data.filter_on_value(np.array([0, 1, 2, 3, 4]), "time", 1.0),
        np.array([1]),
    )

    # Fail with no matching column
    with pytest.raises(KeyError):
        _ = (ops_data.filter_on_value(np.array([0, 1, 2, 3, 4]), "unknown_colname", 1.0),)


def test_obsim_range_search():
    """Test that we can extract the time, ra, and dec from an opsim data frame."""
    # Create a fake opsim data frame with just time, RA, and dec.
    values = {
        "observationStartMJD": np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]),
        "fieldRA": np.array([15.0, 15.0, 15.01, 15.0, 25.0, 24.99, 60.0, 5.0]),
        "fieldDec": np.array([-10.0, 10.0, 10.01, 9.99, 10.0, 9.99, -5.0, -1.0]),
        "zp_nJy": np.ones(8),
        "filter": np.array(["r", "g", "r", "g", "r", "g", "r", "g"]),
    }
    ops_data = OpSim(values)

    # Test single queries.
    assert set(ops_data.range_search(15.0, 10.0, 0.5)) == set([1, 2, 3])
    assert set(ops_data.range_search(25.0, 10.0, 0.5)) == set([4, 5])
    assert set(ops_data.range_search(15.0, 10.0, 100.0)) == set([0, 1, 2, 3, 4, 5, 6, 7])
    assert set(ops_data.range_search(15.0, 10.0, 1e-6)) == set([1])
    assert set(ops_data.range_search(15.02, 10.0, 1e-6)) == set()

    # Test single query with filter matching.
    assert set(ops_data.range_search(15.0, 10.0, 0.5, filter_name="r")) == set([2])
    assert set(ops_data.range_search(15.0, 10.0, 0.5, filter_name="g")) == set([1, 3])
    assert set(ops_data.range_search(15.0, 10.0, 100.0, filter_name="r")) == set([0, 2, 4, 6])
    assert set(ops_data.range_search(15.0, 10.0, 100.0, filter_name="b")) == set()

    # Test a batched query.
    query_ra = np.array([15.0, 25.0, 15.0])
    query_dec = np.array([10.0, 10.0, 5.0])
    neighbors = ops_data.range_search(query_ra, query_dec, 0.5)
    assert len(neighbors) == 3
    assert set(neighbors[0]) == set([1, 2, 3])
    assert set(neighbors[1]) == set([4, 5])
    assert set(neighbors[2]) == set()

    # Test a batched query with filter matching.
    query_ra = np.array([15.0, 25.0, 15.0])
    query_dec = np.array([10.0, 10.0, 5.0])
    neighbors = ops_data.range_search(query_ra, query_dec, 0.5, filter_name="g")
    assert len(neighbors) == 3
    assert set(neighbors[0]) == set([1, 3])
    assert set(neighbors[1]) == set([5])
    assert set(neighbors[2]) == set()


def test_opsim_get_observed_times():
    """Test that we can extract the time, ra, and dec from an opsim data frame."""
    # Create a fake opsim data frame with just time, RA, and dec.
    values = {
        "observationStartMJD": np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]),
        "fieldRA": np.array([15.0, 15.0, 15.01, 15.0, 25.0, 24.99, 60.0, 5.0]),
        "fieldDec": np.array([-10.0, 10.0, 10.01, 9.99, 10.0, 9.99, -5.0, -1.0]),
        "zp_nJy": np.ones(8),
    }
    ops_data = OpSim(values)

    assert np.allclose(ops_data.get_observed_times(15.0, 10.0, 0.5), [1.0, 2.0, 3.0])
    assert np.allclose(ops_data.get_observed_times(25.0, 10.0, 0.5), [4.0, 5.0])
    assert np.allclose(
        ops_data.get_observed_times(15.0, 10.0, 100.0),
        [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
    )
    assert np.allclose(ops_data.get_observed_times(15.0, 10.0, 1e-6), [1.0])
    assert len(ops_data.get_observed_times(15.02, 10.0, 1e-6)) == 0

    # Test a batched query.
    query_ra = np.array([15.0, 25.0, 15.0])
    query_dec = np.array([10.0, 10.0, 5.0])
    times = ops_data.get_observed_times(query_ra, query_dec, 0.5)
    assert len(times) == 3
    assert np.allclose(times[0], [1.0, 2.0, 3.0])
    assert np.allclose(times[1], [4.0, 5.0])
    assert len(times[2]) == 0


def test_opsim_docstring():
    """Test if OpSim class has a docstring"""
    assert OpSim.__doc__ is not None
    assert len(OpSim.__doc__) > 100
