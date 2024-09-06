import numpy as np
from tdastro.effects.white_noise import WhiteNoise
from tdastro.sources.static_source import StaticSource
from tdastro.util_nodes.np_random import NumpyRandomFunc


def test_white_noise() -> None:
    """Test that we can sample and create a WhiteNoise object."""
    model = StaticSource(brightness=100.0)
    model.add_effect(WhiteNoise(scale=0.01))

    times = np.array([1, 2, 3, 5, 10])
    wavelengths = np.array([100.0, 200.0, 300.0])

    values = model.evaluate(times, wavelengths)
    assert values.shape == (5, 3)
    assert not np.all(values == 100.0)
    assert np.all(np.abs(values - 100.0) <= 1.0)


def test_white_noise_random() -> None:
    """Test that we can resample effects to change their parameters."""
    rand_generator = NumpyRandomFunc("uniform", low=1.0, high=2.0)
    model = StaticSource(brightness=10.0)
    model.add_effect(WhiteNoise(scale=rand_generator))
    state1 = model.sample_parameters()
    scale = model.effects[0].get_param(state1, "scale")
    state2 = model.sample_parameters()
    assert model.effects[0].get_param(state2, "scale") != scale
