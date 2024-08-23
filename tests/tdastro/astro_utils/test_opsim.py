import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from tdastro.astro_utils.mag_flux import flux2mag
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


def test_magnitude_electron_zeropoint():
    """Test that instrumental zeropoints are correct"""
    opsim = OpSim({"fieldRA": [], "fieldDec": [], "observationStartMJD": [], "zp_nJy": []})

    # Reproducing magnitude corresponding to S/N=5 from
    # https://smtn-002.lsst.io/v/OPSIM-1171/index.html
    fwhm_eff = {
        "u": 0.92,
        "g": 0.87,
        "r": 0.83,
        "i": 0.80,
        "z": 0.78,
        "y": 0.76,
    }
    fwhm_eff_getter = np.vectorize(fwhm_eff.get)
    sky_brightness = {
        "u": 23.05,
        "g": 22.25,
        "r": 21.20,
        "i": 20.46,
        "z": 19.61,
        "y": 18.60,
    }
    sky_brightness_getter = np.vectorize(sky_brightness.get)

    bands = list(fwhm_eff.keys())
    exptime = 30
    airmass = 1
    s2n = 5
    zp = opsim._magnitude_electron_zeropoint(bands, airmass, exptime)
    sky_count_per_arcsec_sq = np.power(10.0, -0.4 * (sky_brightness_getter(bands) - zp))
    readout_per_arcsec_sq = opsim.read_noise**2 / opsim.pixel_scale**2
    dark_per_arcsec_sq = opsim.dark_current * exptime / opsim.pixel_scale**2
    count_per_arcsec_sq = sky_count_per_arcsec_sq + readout_per_arcsec_sq + dark_per_arcsec_sq

    area = 2.266 * fwhm_eff_getter(bands) ** 2  # effective seeing area in arcsec^2
    n_background = count_per_arcsec_sq * area
    # sky-dominated regime would be n_signal = s2n * np.sqrt(n_background)
    n_signal = 0.5 * (s2n**2 + np.sqrt(s2n**4 + 4 * s2n**2 * n_background))
    mag_signal = zp - 2.5 * np.log10(n_signal)

    m5_desired = {
        "u": 23.70,
        "g": 24.97,
        "r": 24.52,
        "i": 24.13,
        "z": 23.56,
        "y": 22.55,
    }
    assert list(m5_desired) == bands
    m5_desired_getter = np.vectorize(m5_desired.get)

    np.testing.assert_allclose(mag_signal, m5_desired_getter(bands), atol=0.1)


def test_flux_electron_zeropoint():
    """Test that flux zeropoints are correct"""
    # Here we just check that magnitude-flux conversion is correct
    opsim = OpSim({"fieldRA": [], "fieldDec": [], "observationStartMJD": [], "zp_nJy": []})
    airmass = np.array([1, 1.5, 2]).reshape(-1, 1, 1)
    exptime = np.array([30, 38, 45]).reshape(1, -1, 1)
    bands = ["u", "g", "r", "i", "z", "y"]
    mag = opsim._magnitude_electron_zeropoint(bands, airmass, exptime)
    flux = opsim._flux_electron_zeropoint(bands, airmass, exptime)
    np.testing.assert_allclose(mag, flux2mag(flux), rtol=1e-10)
