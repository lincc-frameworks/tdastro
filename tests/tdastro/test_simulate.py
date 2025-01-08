import numpy as np
from tdastro.astro_utils.opsim import OpSim
from tdastro.astro_utils.passbands import PassbandGroup
from tdastro.math_nodes.given_sampler import GivenValueList
from tdastro.simulate import simulate_lightcurves
from tdastro.sources.static_source import StaticSource


def test_simulate_lightcurves(test_data_dir):
    """Test an end to end run of simulating the lightcurves."""
    # Load the OpSim data.
    opsim_db = OpSim.from_db(test_data_dir / "opsim_small.db")

    # Load the passband data for the griz filters only.
    passband_group = PassbandGroup(
        preset="LSST",
        table_dir=test_data_dir / "passbands",
        filters_to_load=["g", "r", "i", "z"],
    )

    # Create a static source with known brightnesses and RA, dec
    # values that match the opsim.
    source = StaticSource(
        brightness=GivenValueList([1000.0, 2000.0, 5000.0, 1000.0, 100.0]),
        t0=0.0,
        ra=GivenValueList(opsim_db["ra"].values[0:5]),
        dec=GivenValueList(opsim_db["dec"].values[0:5]),
        redshift=0.0,
    )

    results = simulate_lightcurves(source, 5, opsim_db, passband_group)
    assert len(results) == 5
    assert np.all(results["nobs"].values >= 1)
    for idx in range(5):
        assert len(results.loc[idx]["lightcurve"]["flux"]) == results["nobs"][idx]
