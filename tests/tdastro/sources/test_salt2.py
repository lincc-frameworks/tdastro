import os

import numpy as np
from sncosmo.models import SALT2Source
from tdastro.sources.salt2_jax import SALT2JaxModel


def test_salt2_model(test_data_dir):
    """Test loading a SALT2 object from a file and querying it."""
    dir_name = os.path.join(test_data_dir, "truncated-salt2-h17")
    model = SALT2JaxModel(x0=0.5, x1=0.2, c=1.0, model_dir=dir_name)

    assert model._colorlaw is not None
    assert model._m0_model is not None
    assert model._m1_model is not None

    # Test compared to values computed via sncosmo's implementation that
    # fall within the range of the truncated grid. We multiple by 1e12
    # for comparison precision purposes.
    times = np.array([1.0, 2.1, 3.9, 4.0])
    waves = np.array([4000.0, 4102.0, 4200.0])
    expected_times_1e12 = np.array(
        [
            [0.12842110, 0.17791164, 0.17462753],
            [0.12287933, 0.17060205, 0.17152248],
            [0.11121435, 0.15392100, 0.16234423],
            [0.11051545, 0.15288580, 0.16170497],
        ]
    )

    flux = model.evaluate(times, waves)
    assert np.allclose(flux * 1e12, expected_times_1e12)


def test_salt2_model_parity(test_data_dir):
    """Test loading a SALT2 object from a file and test we get the same
    results as the sncosmo version.
    """
    dir_name = os.path.join(test_data_dir, "truncated-salt2-h17")
    td_model = SALT2JaxModel(x0=0.4, x1=0.3, c=1.1, model_dir=dir_name)
    sn_model = SALT2Source(modeldir=dir_name)
    sn_model.set(x0=0.4, x1=0.3, c=1.1)

    # Test compared to values computed via sncosmo's implementation that
    # fall within the range of the truncated grid. We multiple by 1e12
    # for comparison precision purposes.
    times = np.arange(-1.0, 15.0, 0.01)
    waves = np.arange(3800.0, 4200.0, 0.5)

    flux_td = td_model.evaluate(times, waves)
    flux_sn = sn_model._flux(times, waves)
    assert np.allclose(flux_td * 1e12, flux_sn * 1e12)
