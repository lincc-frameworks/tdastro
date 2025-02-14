import numpy as np
from tdastro.effects.white_noise import WhiteNoise
from tdastro.sources.basic_sources import StaticSource


def test_white_noise() -> None:
    """Test that we can sample and create a WhiteNoise object."""
    model = StaticSource(brightness=100.0)
    times = np.array([1, 2, 3, 5, 10])
    wavelengths = np.array([100.0, 200.0, 300.0])

    # We start with noiseless fluxes.
    values = model.evaluate(times, wavelengths)
    assert values.shape == (5, 3)
    assert np.all(values == 100.0)

    # We can apply noise.
    white_noise = WhiteNoise(scale=0.1)
    values = white_noise.apply(values)
    assert not np.all(values == 100.0)
    assert np.all(np.abs(values - 100.0) <= 1.0)
