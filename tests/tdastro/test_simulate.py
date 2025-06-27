import numpy as np
from tdastro.astro_utils.passbands import PassbandGroup
from tdastro.math_nodes.given_sampler import GivenValueList
from tdastro.opsim.opsim import OpSim
from tdastro.simulate import simulate_lightcurves
from tdastro.sources.basic_sources import StaticSource


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
    source = StaticSource(
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
        opsim_save_cols=["observationId", "zp_nJy"],
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
