import numpy as np
import pytest
from tdastro.astro_utils.passbands import PassbandGroup
from tdastro.graph_state import GraphState
from tdastro.math_nodes.given_sampler import GivenValueList
from tdastro.models.basic_models import ConstantSEDModel
from tdastro.models.static_sed_model import StaticBandfluxModel
from tdastro.obstable.opsim import OpSim
from tdastro.simulate import get_time_windows, simulate_lightcurves


def test_get_time_windows():
    """Test the get_time_windows function with various inputs."""
    assert get_time_windows(None, None) == (None, None)
    assert get_time_windows(0.0, None) == (None, None)
    assert get_time_windows(None, (1.0, 2.0)) == (None, None)

    result = get_time_windows(0.0, (1.0, 2.0))
    assert np.array_equal(result[0], np.array([-1.0]))
    assert np.array_equal(result[1], np.array([2.0]))

    result = get_time_windows(1.0, (None, 2.0))
    assert result[0] is None
    assert np.array_equal(result[1], np.array([3.0]))

    result = get_time_windows(-10.0, (1.0, None))
    assert np.array_equal(result[0], np.array([-11.0]))
    assert result[1] is None

    result = get_time_windows(np.array([0.0, 1.0, 2.0]), (1.0, 2.0))
    assert np.array_equal(result[0], np.array([-1.0, 0.0, 1.0]))
    assert np.array_equal(result[1], np.array([2.0, 3.0, 4.0]))

    with pytest.raises(ValueError):
        get_time_windows(0.0, (1.0, 2.0, 3.0))


def test_simulate_lightcurves(test_data_dir):
    """Test an end to end run of simulating the light curves."""
    # Load the OpSim data.
    opsim_db = OpSim.from_db(test_data_dir / "opsim_small.db")

    # Load the passband data for the griz filters only.
    passband_group = PassbandGroup.from_preset(
        preset="LSST",
        table_dir=test_data_dir / "passbands",
        filters=["g", "r", "i", "z"],
    )

    # Create a constant SED model with known brightnesses and RA, dec
    # values that match the opsim.
    given_brightness = [1000.0, 2000.0, 5000.0, 1000.0, 100.0]
    source = ConstantSEDModel(
        brightness=GivenValueList(given_brightness),
        t0=0.0,
        ra=GivenValueList(opsim_db["ra"].values[0:5]),
        dec=GivenValueList(opsim_db["dec"].values[0:5]),
        redshift=0.0,
        node_label="source",
    )

    results = simulate_lightcurves(
        source,
        5,
        opsim_db,
        passband_group,
        obstable_save_cols=["observationId", "zp_nJy"],
        param_cols=["source.brightness"],
    )
    assert len(results) == 5
    assert np.all(results["nobs"].values >= 1)
    for idx in range(5):
        num_obs = results["nobs"][idx]
        assert len(results.loc[idx]["lightcurve"]["flux"]) == num_obs

        # Check that we pulled the metadata from the opsim.
        assert len(results.loc[idx]["lightcurve"]["observationId"]) == num_obs
        assert len(np.unique(results.loc[idx]["lightcurve"]["observationId"])) == num_obs
        assert np.all(results.loc[idx]["lightcurve"]["observationId"] >= 0)
        assert len(results.loc[idx]["lightcurve"]["zp_nJy"]) == num_obs

        # Check that we extract one of the parameters.
        assert results["source_brightness"][idx] == given_brightness[idx]

    # Check that we saved and can reassemble the GraphStates
    assert "params" in results
    state = GraphState.from_list(results["params"].values)
    assert state.num_samples == 5
    assert np.allclose(state["source.ra"], opsim_db["ra"].values[0:5])
    assert np.allclose(state["source.dec"], opsim_db["dec"].values[0:5])

    # Check that we fail if we try to save a parameter column that doesn't exist.
    source2 = ConstantSEDModel(brightness=10.0, t0=0.0, ra=1.0, dec=-1.0, redshift=0.0, node_label="source2")
    with pytest.raises(KeyError):
        _ = simulate_lightcurves(
            source2,
            1,
            opsim_db,
            passband_group,
            param_cols=["source.unknown_parameter"],
        )


def test_simulate_bandfluxes(test_data_dir):
    """Test an end to end run of simulating a bandflux model."""
    # Create a toy observation table with two pointings.
    num_obs = 6
    obsdata = {
        "time": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
        "ra": [0.0, 0.0, 180.0, 0.0, 180.0, 0.0],
        "dec": [10.0, 10.0, -10.0, 10.0, -10.0, 10.0],
        "filter": ["g", "r", "g", "r", "z", "z"],
        "zp": [1.0] * num_obs,
        "seeing": [1.12] * num_obs,
        "skybrightness": [20.0] * num_obs,
        "exptime": [29.2] * num_obs,
        "nexposure": [2] * num_obs,
    }
    obstable = OpSim(obsdata)

    # Load the passband data for the griz filters only.
    passband_group = PassbandGroup.from_preset(
        preset="LSST",
        table_dir=test_data_dir / "passbands",
        filters=["g", "r", "i", "z"],
    )

    # Create a static bandflux model and simulate 2 runs.
    model = StaticBandfluxModel({"g": 1.0, "i": 2.0, "r": 3.0, "y": 4.0, "z": 5.0}, ra=0.0, dec=10.0)
    results = simulate_lightcurves(model, 2, obstable, passband_group)
    assert len(results) == 2
    assert np.allclose(results["lightcurve"][0]["flux_perfect"], [1.0, 3.0, 3.0, 5.0])
    assert np.allclose(results["lightcurve"][1]["flux_perfect"], [1.0, 3.0, 3.0, 5.0])


def test_simulate_single_lightcurve(test_data_dir):
    """Test an end to end run of simulating a single light curve."""
    # Load the OpSim data.
    opsim_db = OpSim.from_db(test_data_dir / "opsim_small.db")

    # Load the passband data for the griz filters only.
    passband_group = PassbandGroup.from_preset(
        preset="LSST",
        table_dir=test_data_dir / "passbands",
        filters=["g", "r", "i", "z"],
    )

    # Create a constant SED model with known brightnesses and RA, dec
    # values that match the opsim.
    given_brightness = [1000.0, 2000.0, 5000.0, 1000.0, 100.0]
    source = ConstantSEDModel(
        brightness=GivenValueList(given_brightness),
        t0=0.0,
        ra=GivenValueList(opsim_db["ra"].values[0:5]),
        dec=GivenValueList(opsim_db["dec"].values[0:5]),
        redshift=0.0,
        node_label="source",
    )

    results = simulate_lightcurves(
        source,
        1,
        opsim_db,
        passband_group,
        obstable_save_cols=["observationId", "zp_nJy"],
        param_cols=["source.brightness"],
    )
    assert len(results) == 1

    # Check that we saved and can reassemble the GraphStates
    assert "params" in results
    state = GraphState.from_list(results["params"].values)
    assert state.num_samples == 1
    assert state["source.ra"] == opsim_db["ra"].values[0]
    assert state["source.dec"] == opsim_db["dec"].values[0]


def test_simulate_with_time_window(test_data_dir):
    """Test an end to end run of simulating with a limited time window."""
    # Create a toy OpSim database with two pointings over a series of tiems.
    # Create a fake opsim data frame with just time, RA, and dec.
    values = {
        "observationStartMJD": np.arange(50.0),
        "fieldRA": np.array([15.0 if i % 2 == 0 else 180.0 for i in range(50)]),
        "fieldDec": np.array([10.0 if i % 2 == 0 else -10.0 for i in range(50)]),
        "filter": np.full(50, "g"),
        # We add the remaining values so the OpSim can compute noise, but they are
        # arbitrary and not tested in this test.
        "zp_nJy": np.ones(50),
        "seeingFwhmEff": np.ones(50) * 0.7,
        "visitExposureTime": np.ones(50) * 30.0,
        "numExposures": np.ones(50) * 1,
        "skyBrightness": np.full(50, 20.0),
    }
    opsim_db = OpSim(values)

    # Load the passband data for the griz filters only.
    passband_group = PassbandGroup.from_preset(
        preset="LSST",
        table_dir=test_data_dir / "passbands",
        filters=["g", "r", "i", "z"],
    )

    # Create a constant SED model with known brightnesses and RA, dec
    # values that match the opsim.
    source = ConstantSEDModel(
        brightness=1000.0,
        t0=GivenValueList([20.0, 15.0]),
        ra=15.0,
        dec=10.0,
        redshift=0.0,
        node_label="source",
    )

    results = simulate_lightcurves(
        source,
        2,
        opsim_db,
        passband_group,
        time_window_offset=(5.0, 10.0),
    )
    assert len(results) == 2

    # We should simulate the observations that are only within the time window, (15.0, 30.0) for the
    # first samples and (10.0, 25.0) for the second sample, and at the matching RA/Dec (the even indices).
    assert np.array_equal(
        results["lightcurve"][0]["mjd"],
        np.array([16.0, 18.0, 20.0, 22.0, 24.0, 26.0, 28.0, 30.0]),
    )
    assert np.array_equal(
        results["lightcurve"][1]["mjd"],
        np.array([10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0]),
    )


def test_simulate_multiple_surveys(test_data_dir):
    """Test an end to end run of simulating a single light curve from multiple surveys."""
    # The first survey points at two locations in the sky in the "g" and "r" bands.
    obsdata1 = {
        "time": [0.0, 1.0, 2.0, 3.0],
        "ra": [0.0, 0.0, 180.0, 180.0],
        "dec": [10.0, 10.0, -10.0, -10.0],
        "filter": ["g", "r", "g", "r"],
        "zp": [0.4, 0.5, 0.6, 0.7],
        "seeing": [1.12, 1.12, 1.12, 1.12],
        "skybrightness": [20.0, 20.0, 20.0, 20.0],
        "exptime": [29.2, 29.2, 29.2, 29.2],
        "nexposure": [2, 2, 2, 2],
        "custom_col": [1, 1, 1, 1],
    }
    obstable1 = OpSim(obsdata1)
    passband_group1 = PassbandGroup.from_preset(
        preset="LSST",
        table_dir=test_data_dir / "passbands",
        filters=["g", "r"],
    )

    # The second survey points at two locations on the sky in the "r" and "z" bands.
    obsdata2 = {
        "time": [0.5, 1.5, 2.5, 3.5],
        "ra": [0.0, 90.0, 0.0, 90.0],
        "dec": [10.0, -10.0, 10.0, -10.0],
        "filter": ["r", "z", "r", "z"],
        "zp": [0.05, 0.1, 0.2, 0.3],
        "seeing": [1.12, 1.12, 1.12, 1.12],
        "skybrightness": [20.0, 20.0, 20.0, 20.0],
        "exptime": [29.2, 29.2, 29.2, 29.2],
        "nexposure": [2, 2, 2, 2],
    }
    obstable2 = OpSim(obsdata2)
    passband_group2 = PassbandGroup.from_preset(
        preset="LSST",
        table_dir=test_data_dir / "passbands",
        filters=["r", "z"],
    )

    # Create a constant SED model with known brightnesses and RA, dec values that
    # match the (0.0, 10.0) pointing.
    model = ConstantSEDModel(brightness=100.0, t0=0.0, ra=0.0, dec=10.0, redshift=0.0, node_label="source")
    results = simulate_lightcurves(
        model,
        1,
        [obstable1, obstable2],
        [passband_group1, passband_group2],
        obstable_save_cols=["zp", "custom_col"],
    )
    assert len(results) == 1
    assert results["nobs"][0] == 4

    # Check that the light curve was simulated correctly, including saving the zeropoint information
    # from each ObsTable.
    lightcurve = results["lightcurve"][0]
    assert np.allclose(lightcurve["mjd"], np.array([0.0, 1.0, 0.5, 2.5]))
    assert np.allclose(lightcurve["zp"], np.array([0.4, 0.5, 0.05, 0.2]))
    assert np.array_equal(lightcurve["filter"], np.array(["g", "r", "r", "r"]))
    assert np.array_equal(lightcurve["survey_idx"], np.array([0, 0, 1, 1]))

    # The custom column should only exist for observations from one of the surveys.
    assert np.all(lightcurve["custom_col"][0:2] == 1)
    assert np.all(np.isnan(lightcurve["custom_col"][2:4]))

    # We fail if we pass in lists of different lengths.
    with pytest.raises(ValueError):
        simulate_lightcurves(
            model,
            1,
            [obstable1, obstable2],
            passband_group1,
        )

    # We fail if we try to use a bandflux only model with multiple surveys.
    model2 = StaticBandfluxModel({"g": 1.0, "i": 1.0, "r": 1.0, "y": 1.0, "z": 1.0})
    with pytest.raises(ValueError):
        simulate_lightcurves(
            model2,
            1,
            [obstable1, obstable2],
            [passband_group1, passband_group2],
        )
