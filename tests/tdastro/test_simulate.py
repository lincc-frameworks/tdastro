import numpy as np
import pytest
from tdastro.astro_utils.passbands import PassbandGroup
from tdastro.graph_state import GraphState
from tdastro.math_nodes.given_sampler import GivenValueList
from tdastro.opsim.opsim import OpSim
from tdastro.simulate import get_time_windows, simulate_lightcurves
from tdastro.sources.basic_models import ConstantSED


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
    """Test an end to end run of simulating the lightcurves."""
    # Load the OpSim data.
    opsim_db = OpSim.from_db(test_data_dir / "opsim_small.db")

    # Load the passband data for the griz filters only.
    passband_group = PassbandGroup.from_preset(
        preset="LSST",
        table_dir=test_data_dir / "passbands",
        filters=["g", "r", "i", "z"],
    )

    # Create a static source with known brightnesses and RA, dec
    # values that match the opsim.
    given_brightness = [1000.0, 2000.0, 5000.0, 1000.0, 100.0]
    source = ConstantSED(
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
    source2 = ConstantSED(brightness=10.0, t0=0.0, ra=1.0, dec=-1.0, redshift=0.0, node_label="source2")
    with pytest.raises(KeyError):
        _ = simulate_lightcurves(
            source2,
            1,
            opsim_db,
            passband_group,
            param_cols=["source.unknown_parameter"],
        )

    # Check that we fail if we try to save an ObsTable column that doesn't exist.
    with pytest.raises(KeyError):
        _ = simulate_lightcurves(
            source2,
            1,
            opsim_db,
            passband_group,
            obstable_save_cols=["unknown_column"],
        )


def test_simulate_single_lightcurve(test_data_dir):
    """Test an end to end run of simulating a single lightcurves."""
    # Load the OpSim data.
    opsim_db = OpSim.from_db(test_data_dir / "opsim_small.db")

    # Load the passband data for the griz filters only.
    passband_group = PassbandGroup.from_preset(
        preset="LSST",
        table_dir=test_data_dir / "passbands",
        filters=["g", "r", "i", "z"],
    )

    # Create a static source with known brightnesses and RA, dec
    # values that match the opsim.
    given_brightness = [1000.0, 2000.0, 5000.0, 1000.0, 100.0]
    source = ConstantSED(
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

    # Create a static source with known brightnesses and RA, dec
    # values that match the opsim.
    source = ConstantSED(
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
