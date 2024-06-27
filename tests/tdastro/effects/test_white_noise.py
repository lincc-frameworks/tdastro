import numpy as np
from tdastro.effects.white_noise import WhiteNoise
from tdastro.sources.static_source import StaticSource


def brightness_generator():
    """A test generator function."""
    return 10.0 + 0.5 * np.random.rand(1)


def test_white_noise() -> None:
    """Test that we can sample and create a WhiteNoise object."""
    model = StaticSource(brightness=brightness_generator)
    model.add_effect(WhiteNoise(scale=0.01))

    times = np.array([1, 2, 3, 5, 10])
    wavelengths = np.array([100.0, 200.0, 300.0])

    values = model.evaluate(times, wavelengths)
    assert values.shape == (5, 3)
    assert not np.all(values == 10.0)
    assert np.all(np.abs(values - 10.0) < 1.0)
