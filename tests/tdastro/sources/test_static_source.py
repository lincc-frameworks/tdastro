import random

import numpy as np
from tdastro.sources.static_source import StaticSource


def _sampler_fun(magnitude, **kwargs):
    """Return a random value between 0 and magnitude.

    Parameters
    ----------
    magnitude : `float`
        The range of brightness magnitude
    **kwargs : `dict`, optional
        Absorbs additional parameters
    """
    return magnitude * random.random()


def test_static_source() -> None:
    """Test that we can sample and create a StaticSource object."""
    model = StaticSource(brightness=10.0)
    assert model.brightness == 10.0
    assert model.ra is None
    assert model.dec is None
    assert model.distance is None

    times = np.array([1, 2, 3, 4, 5, 10])
    wavelengths = np.array([100.0, 200.0, 300.0])

    values = model.evaluate(times, wavelengths)
    assert values.shape == (6, 3)
    assert np.all(values == 10.0)


def test_static_source_host() -> None:
    """Test that we can sample and create a StaticSource object with properties
    derived from the host object."""
    host = StaticSource(brightness=15.0, ra=1.0, dec=2.0, distance=3.0)
    model = StaticSource(brightness=10.0, ra=host, dec=host, distance=host)
    assert model.brightness == 10.0
    assert model.ra == 1.0
    assert model.dec == 2.0
    assert model.distance == 3.0


def test_static_source_resample() -> None:
    """Check that we can call resample on the model parameters."""
    model = StaticSource(brightness=_sampler_fun, magnitude=100.0)

    num_samples = 100
    values = np.zeros((num_samples, 1))
    for i in range(num_samples):
        model.sample_parameters(magnitude=100.0)
        values[i] = model.brightness

    # Check that the values fall within the expected bounds.
    assert np.all(values >= 0.0)
    assert np.all(values <= 100.0)

    # Check that the values are not all the same.
    assert not np.all(values == values[0])
