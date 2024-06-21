import numpy as np
from tdastro.effects.white_noise import WhiteNoise
from tdastro.sources.static_source import StaticSource


def brightness_generator():
    return 10.0 + 0.5 * np.random.rand(1)


def test_white_noise() -> None:
    model = StaticSource(brightness=brightness_generator)
    model.add_effect(WhiteNoise(scale=0.01))

    values = model.evaluate(np.array([1, 2, 3, 4, 5]))
    assert len(values) == 5
    assert not np.all(values == 10.0)
    assert np.all((np.abs(values - 10.0) < 1.0))
