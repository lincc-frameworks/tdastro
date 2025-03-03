import numpy as np
import pytest
from tdastro.effects.drop_observations import DropEffect


def test_drop_effect() -> None:
    """Test that we can create and sample a DropEffect object."""
    num_times = 500
    num_wavelengths = 400
    times = np.linspace(0, 100, num_times)
    wavelengths = np.linspace(1000, 200, num_wavelengths)
    fluxes = np.full((num_times, num_wavelengths), 100.0)

    drop_effect = DropEffect(drop_probability=0.1)
    rng_info = np.random.default_rng(100)
    values = drop_effect.apply(fluxes, times, wavelengths, rng_info=rng_info)
    assert values.shape == (num_times, num_wavelengths)
    assert np.sum(values == 0.0) > 0.05 * num_times * num_wavelengths
    assert np.sum(values == 0.0) < 0.15 * num_times * num_wavelengths
    assert len(np.unique(values)) == 2

    with pytest.raises(ValueError):
        DropEffect(drop_probability=-0.1)
    with pytest.raises(ValueError):
        DropEffect(drop_probability=1.1)
