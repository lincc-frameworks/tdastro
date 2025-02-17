import numpy as np
import pytest
from tdastro.effects.white_noise import WhiteNoise
from tdastro.sources.basic_sources import StaticSource


def test_white_noise() -> None:
    """Test that we can sample and create a WhiteNoise object."""
    values = np.full((5, 3), 100.0)

    # We can apply noise the noise.
    white_noise = WhiteNoise(white_noise_sigma=0.1)
    values = white_noise.apply(values, white_noise_sigma=0.1)
    assert not np.all(values == 100.0)
    assert np.all(np.abs(values - 100.0) <= 1.0)

    # We can override the default value using the parameters.
    values = white_noise.apply(values, white_noise_sigma=20.0)
    assert not np.all(values == 100.0)
    assert not np.all(np.abs(values - 100.0) <= 1.0)

    # We fail if we do not pass in the expected parameters.
    with pytest.raises(ValueError):
        _ = white_noise.apply(values)


def test_static_source_white_noise() -> None:
    """Test that we can sample and create a StaticSource object with white noise."""
    white_noise = WhiteNoise(white_noise_sigma=0.1)
    model = StaticSource(
        brightness=10.0,
        node_label="my_static_source",
        effects=[white_noise],
        seed=100,
    )
    state = model.sample_parameters()

    times = np.array([1, 2, 3, 4, 5, 10])
    wavelengths = np.array([100.0, 200.0, 300.0])
    values = model.evaluate(times, wavelengths, state)
    assert values.shape == (6, 3)

    # We get noisy values around 10.0.
    assert len(np.unique(values)) > 10
    assert np.all(np.abs(values - 10.0) < 3.0)

    # Test that if we pass in an rng, we control the randomness.
    values1 = model.evaluate(times, wavelengths, state, rng_info=np.random.default_rng(100))
    values2 = model.evaluate(times, wavelengths, state, rng_info=np.random.default_rng(100))
    assert not np.any(values1 == 10.0)
    assert np.all(values1 == values2)
