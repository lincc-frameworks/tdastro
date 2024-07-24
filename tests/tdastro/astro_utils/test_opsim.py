import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from tdastro.astro_utils.opsim import load_opsim_table, pointings_from_opsim, write_opsim_table


def test_write_read_opsim():
    """Create a minimal opsim data frame, test that we can write it,
    and test that we can correctly read it back in."""

    # Create a fake opsim data frame with just time, RA, and dec.
    values = {
        "observationStartMJD": np.array([0.0, 1.0, 2.0, 3.0, 4.0]),
        "fieldRA": np.array([15.0, 30.0, 15.0, 0.0, 60.0]),
        "fieldDec": np.array([-10.0, -5.0, 0.0, 5.0, 10.0]),
    }
    opsim = pd.DataFrame(values)

    with tempfile.TemporaryDirectory() as dir_name:
        filename = os.path.join(dir_name, "test_write_read_opsim.db")

        # The opsim does not exist until we write it.
        assert not Path(filename).is_file()
        with pytest.raises(FileNotFoundError):
            _ = load_opsim_table(filename)

        # We can write the opsim db.
        write_opsim_table(opsim, filename)
        assert Path(filename).is_file()

        # We can reread the opsim db.
        opsim2 = load_opsim_table(filename)
        assert len(opsim2) == 5
        assert np.allclose(values["observationStartMJD"], opsim2["observationStartMJD"].to_numpy())
        assert np.allclose(values["fieldRA"], opsim2["fieldRA"].to_numpy())
        assert np.allclose(values["fieldDec"], opsim2["fieldDec"].to_numpy())

        # We cannot overwrite unless we set overwrite=True
        with pytest.raises(ValueError):
            write_opsim_table(opsim, filename, overwrite=False)
        write_opsim_table(opsim, filename, overwrite=True)


def test_pointings_from_opsim():
    """Test that we can extract the time, ra, and dec from an opsim data frame."""

    # Create a fake opsim data frame with just time, RA, and dec.
    values = {
        "custom_time_name": np.array([0.0, 1.0, 2.0, 3.0, 4.0]),
        "custom_ra_name": np.array([15.0, 30.0, 15.0, 0.0, 60.0]),
        "custom_dec_name": np.array([-10.0, -5.0, 0.0, 5.0, 10.0]),
    }
    opsim = pd.DataFrame(values)

    pointings = pointings_from_opsim(
        opsim,
        time_colname="custom_time_name",
        ra_colname="custom_ra_name",
        dec_colname="custom_dec_name",
    )
    assert len(pointings) == 5
    assert np.allclose(values["custom_time_name"], pointings["time"])
    assert np.allclose(values["custom_ra_name"], pointings["ra"])
    assert np.allclose(values["custom_dec_name"], pointings["dec"])

    # We fail if we give the wrong column names.
    with pytest.raises(KeyError):
        _ = pointings_from_opsim(opsim)
