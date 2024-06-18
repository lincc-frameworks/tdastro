import numpy as np
import pytest

from tdastro.effects.white_noise import DistanceBasedWhiteNoise, WhiteNoise
from tdastro.sources.static_source import StaticSource

def brightness_generator():
    return 10.0 + 0.5 * np.random.rand(1)

def test_white_noise() -> None:
    model = StaticSource(brightness=brightness_generator)
    model.add_effect(WhiteNoise(scale=0.01))

    values = model.observe(np.array([1, 2, 3, 4, 5]))
    assert len(values) == 5
    assert not np.all(values == 10.0) 
    assert np.all((np.abs(values - 10.0) < 1.0))

def test_distance_based_white_noise() -> None:
    model1 = StaticSource(brightness=10.0, distance=10.0)
    model1.add_effect(DistanceBasedWhiteNoise(scale=0.01, dist_multiplier=0.05))

    values = model1.observe(np.array([1, 2, 3, 4, 5]))
    assert len(values) == 5
    assert not np.all(values == 10.0) 

    # Fail if distance is not specified.
    model2 = StaticSource(brightness=10.0)
    with pytest.raises(AttributeError):
        model2.add_effect(DistanceBasedWhiteNoise(scale=0.01, dist_multiplier=0.05))
