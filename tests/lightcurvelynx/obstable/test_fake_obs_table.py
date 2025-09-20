import numpy as np
import pandas as pd
import pytest
from lightcurvelynx.obstable.fake_obs_table import FakeObsTable


def test_create_fake_obs_table_consts():
    """Create a minimal FakeObsTable object with given defaults."""
    values = {
        "time": np.array([0.0, 1.0, 2.0, 3.0, 4.0]),
        "ra": np.array([15.0, 30.0, 15.0, 0.0, 60.0]),
        "dec": np.array([-10.0, -5.0, 0.0, 5.0, 10.0]),
        "filter": np.array(["r", "g", "r", "i", "g"]),
    }
    pdf = pd.DataFrame(values)

    zp_per_band = {"g": 26.0, "r": 27.0, "i": 28.0}
    ops_data = FakeObsTable(pdf, zp_per_band=zp_per_band, fwhm=2.0, sky_background=100.0)
    assert len(ops_data) == 5

    # We use the defaults when we do not provide values in the table.
    assert np.allclose(ops_data["ra"], values["ra"])
    assert np.allclose(ops_data["dec"], values["dec"])
    assert np.allclose(ops_data["time"], values["time"])
    assert np.array_equal(ops_data["filter"], values["filter"])
    assert np.allclose(ops_data["fwhm"], [2.0] * 5)
    assert np.allclose(ops_data["sky_background"], [100.0] * 5)
    assert np.allclose(ops_data["exptime"], [30.0] * 5)
    assert np.allclose(ops_data["nexposure"], [1] * 5)
    assert np.allclose(ops_data["zp"], [27.0, 26.0, 27.0, 28.0, 26.0])

    assert ops_data.survey_values["dark_current"] == 0
    assert ops_data.survey_values["nexposure"] == 1
    assert ops_data.survey_values["sky_background"] == 100
    assert ops_data.survey_values["radius"] is None
    assert ops_data.survey_values["read_noise"] == 0
    assert ops_data.survey_values["survey_name"] == "FAKE_SURVEY"

    # We can compute noise.
    flux_error = ops_data.bandflux_error_point_source(
        np.array([100.0, 200.0, 300.0, 400.0, 500.0]),  # Fluxes in nJy
        np.arange(5),  # Indices of the observations
    )
    assert len(flux_error) == 5
    assert np.all(flux_error > 0)
    assert len(np.unique(flux_error)) > 1  # Not all the same

    # We can override the defaults.
    ops_data = FakeObsTable(
        pdf,
        zp_per_band=zp_per_band,
        exptime=60.0,
        fwhm=2.5,
        nexposure=2,
        radius=1.0,
        read_noise=5.0,
        sky_background=150.0,
        survey_name="MY_SURVEY",
    )
    assert len(ops_data) == 5

    # We use the defaults when we do not provide values in the table.
    assert np.allclose(ops_data["ra"], values["ra"])
    assert np.allclose(ops_data["dec"], values["dec"])
    assert np.allclose(ops_data["time"], values["time"])
    assert np.array_equal(ops_data["filter"], values["filter"])
    assert np.allclose(ops_data["fwhm"], [2.5] * 5)
    assert np.allclose(ops_data["sky_background"], [150.0] * 5)
    assert np.allclose(ops_data["exptime"], [60.0] * 5)
    assert np.allclose(ops_data["nexposure"], [2] * 5)
    assert np.allclose(ops_data["zp"], [27.0, 26.0, 27.0, 28.0, 26.0])

    assert ops_data.survey_values["dark_current"] == 0
    assert ops_data.survey_values["nexposure"] == 2
    assert ops_data.survey_values["sky_background"] == 150
    assert ops_data.survey_values["radius"] == 1.0
    assert ops_data.survey_values["read_noise"] == 5.0
    assert ops_data.survey_values["survey_name"] == "MY_SURVEY"

    # We can compute noise.
    flux_error2 = ops_data.bandflux_error_point_source(
        np.array([100.0, 200.0, 300.0, 400.0, 500.0]),  # Fluxes in nJy
        np.arange(5),  # Indices of the observations
    )
    assert len(flux_error2) == 5
    assert np.all(flux_error2 > 0)
    assert len(np.unique(flux_error2)) > 1  # Not all the same
    assert np.any(flux_error2 != flux_error)  # Different from before


def test_create_fake_obs_table_non_consts():
    """Test that if we specify values in columns, we use those instead of the defaults."""
    values = {
        "time": np.array([0.0, 1.0, 2.0, 3.0, 4.0]),
        "ra": np.array([15.0, 30.0, 15.0, 0.0, 60.0]),
        "dec": np.array([-10.0, -5.0, 0.0, 5.0, 10.0]),
        "filter": np.array(["r", "g", "r", "i", "g"]),
        "fwhm": np.array([2.0, 2.5, 3.0, 2.0, 1.5]),
        "sky_background": np.array([100.0, 150.0, 200.0, 100.0, 50.0]),
        "exptime": np.array([20.0, 30.0, 40.0, 20.0, 10.0]),
        "nexposure": np.array([1, 2, 3, 1, 1]),
        "zp": np.array([10.0, 11.0, 12.0, 13.0, 14.0]),
    }
    pdf = pd.DataFrame(values)

    zp_per_band = {"g": 26.0, "r": 27.0, "i": 28.0}
    ops_data = FakeObsTable(pdf, zp_per_band=zp_per_band, fwhm=2.0, sky_background=100.0)
    assert len(ops_data) == 5
    assert np.allclose(ops_data["ra"], values["ra"])
    assert np.allclose(ops_data["dec"], values["dec"])
    assert np.allclose(ops_data["time"], values["time"])
    assert np.array_equal(ops_data["filter"], values["filter"])
    assert np.allclose(ops_data["fwhm"], values["fwhm"])
    assert np.allclose(ops_data["sky_background"], values["sky_background"])
    assert np.allclose(ops_data["exptime"], values["exptime"])
    assert np.allclose(ops_data["nexposure"], values["nexposure"])
    assert np.allclose(ops_data["zp"], values["zp"])

    # We can compute noise.
    flux_error = ops_data.bandflux_error_point_source(
        np.array([100.0, 200.0, 300.0, 400.0, 500.0]),  # Fluxes in nJy
        np.arange(5),  # Indices of the observations
    )
    assert len(flux_error) == 5
    assert np.all(flux_error > 0)
    assert len(np.unique(flux_error)) > 1  # Not all the same


def test_create_fake_obs_table_cols_fail():
    """Test that we raise errors when we do not provide required values."""
    values = {
        "time": np.array([0.0, 1.0, 2.0, 3.0, 4.0]),
        "ra": np.array([15.0, 30.0, 15.0, 0.0, 60.0]),
        "dec": np.array([-10.0, -5.0, 0.0, 5.0, 10.0]),
        "filter": np.array(["r", "g", "r", "i", "g"]),
    }
    pdf = pd.DataFrame(values)

    # Missing fwhm.
    zp_per_band = {"g": 26.0, "r": 27.0, "i": 28.0}
    with pytest.raises(ValueError):
        _ = FakeObsTable(pdf, zp_per_band=zp_per_band, sky_background=100.0)

    # Missing sky_background.
    with pytest.raises(ValueError):
        _ = FakeObsTable(pdf, zp_per_band=zp_per_band, fwhm=2.0)

    # Missing or invalid exptime.
    with pytest.raises(ValueError):
        _ = FakeObsTable(pdf, zp_per_band=zp_per_band, fwhm=2.0, sky_background=100.0, exptime=None)
    with pytest.raises(ValueError):
        _ = FakeObsTable(pdf, zp_per_band=zp_per_band, fwhm=2.0, sky_background=100.0, exptime=-10.0)

    # Missing or invalid nexposure.
    with pytest.raises(ValueError):
        _ = FakeObsTable(pdf, zp_per_band=zp_per_band, fwhm=2.0, sky_background=100.0, nexposure=None)
    with pytest.raises(ValueError):
        _ = FakeObsTable(pdf, zp_per_band=zp_per_band, fwhm=2.0, sky_background=100.0, nexposure=-1)


def test_create_fake_obs_table_zp_fail():
    """Test that we raise errors when we do not provide required values."""
    values = {
        "time": np.array([0.0, 1.0, 2.0, 3.0, 4.0]),
        "ra": np.array([15.0, 30.0, 15.0, 0.0, 60.0]),
        "dec": np.array([-10.0, -5.0, 0.0, 5.0, 10.0]),
    }
    pdf = pd.DataFrame(values)

    # No filters from which to computer zp.
    zp_per_band = {"g": 26.0, "r": 27.0, "i": 28.0}
    with pytest.raises(ValueError):
        _ = FakeObsTable(pdf, zp_per_band=zp_per_band, fwhm=2.0, sky_background=100.0)

    # Mismatched filters.
    values["filter"] = np.array(["r", "g", "r", "i", "z"])
    pdf = pd.DataFrame(values)

    zp_per_band = {"g": 26.0, "r": 27.0}
    with pytest.raises(ValueError):
        _ = FakeObsTable(pdf, zp_per_band=zp_per_band, fwhm=2.0, sky_background=100.0)


def test_create_fake_obs_table_const_flux_error():
    """Test that we can provide a constant flux error."""
    values = {
        "time": np.array([0.0, 1.0, 2.0, 3.0, 4.0]),
        "ra": np.array([15.0, 30.0, 15.0, 0.0, 60.0]),
        "dec": np.array([-10.0, -5.0, 0.0, 5.0, 10.0]),
        "filter": np.array(["r", "g", "r", "r", "g"]),
    }
    pdf = pd.DataFrame(values)

    zp_per_band = {"g": 26.0, "r": 27.0, "i": 28.0}
    const_flux_error = 5.0  # nJy

    # We can construct the table even without providing fwhm, sky_background, exptime, or nexposure,
    # since we do not need them to compute noise.
    ops_data = FakeObsTable(pdf, zp_per_band=zp_per_band, const_flux_error=const_flux_error)
    assert len(ops_data) == 5

    flux_error = ops_data.bandflux_error_point_source(
        np.array([100.0, 200.0, 300.0, 400.0, 500.0]),  # Fluxes in nJy
        np.arange(5),  # Indices of the observations
    )
    assert np.allclose(flux_error, const_flux_error)

    # We can also use a dictionary to provide per-band constant flux errors.
    const_flux_error = {"g": 3.0, "r": 4.0}
    ops_data = FakeObsTable(pdf, zp_per_band=zp_per_band, const_flux_error=const_flux_error)
    assert len(ops_data) == 5

    flux_error = ops_data.bandflux_error_point_source(
        np.array([100.0, 200.0, 300.0, 400.0, 500.0]),  # Fluxes in nJy
        np.arange(5),  # Indices of the observations
    )
    assert np.allclose(flux_error, [4.0, 3.0, 4.0, 4.0, 3.0])
