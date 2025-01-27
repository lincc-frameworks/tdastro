import random

import numpy as np
from tdastro.base_models import FunctionNode
from tdastro.sources.basic_sources import SinWaveSource, StaticSource, StepSource


def _sampler_fun(magnitude, offset=0.0, **kwargs):
    """Return a random value between 0 and magnitude.

    Parameters
    ----------
    magnitude : float
        The range of brightness magnitude
    offset : float, optional
        The offset.
    **kwargs : dict, optional
        Absorbs additional parameters
    """
    return magnitude * random.random() + offset


def test_static_source() -> None:
    """Test that we can sample and create a StaticSource object."""
    model = StaticSource(brightness=10.0, node_label="my_static_source")
    state = model.sample_parameters()
    assert model.get_param(state, "brightness") == 10.0
    assert model.get_param(state, "ra") is None
    assert model.get_param(state, "dec") is None
    assert model.get_param(state, "distance") is None
    assert str(model) == "my_static_source"

    times = np.array([1, 2, 3, 4, 5, 10])
    wavelengths = np.array([100.0, 200.0, 300.0])

    values = model.evaluate(times, wavelengths, state)
    assert values.shape == (6, 3)
    assert np.all(values == 10.0)

    # We can set a value we have already added.
    model.set_parameter("brightness", 5.0)
    values = model.evaluate(times, wavelengths)
    assert values.shape == (6, 3)
    assert np.all(values == 5.0)


def test_static_source_pytree():
    """Test that the PyTree only contains brightness."""
    model = StaticSource(brightness=10.0, node_label="my_static_source")
    state = model.sample_parameters()

    pytree = model.build_pytree(state)
    assert pytree["my_static_source"]["brightness"] == 10.0
    assert pytree["my_static_source"]["t0"] is None
    assert len(pytree["my_static_source"]) == 2
    assert len(pytree) == 1


def test_static_source_white_noise() -> None:
    """Test that we can sample and create a StaticSource object with white noise."""
    model = StaticSource(
        brightness=10.0,
        white_noise_sigma=1.0,
        node_label="my_static_source",
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


def test_static_source_host() -> None:
    """Test that we can sample and create a StaticSource object with properties
    derived from the host object."""
    host = StaticSource(brightness=15.0, ra=1.0, dec=2.0, distance=3.0)
    model = StaticSource(brightness=10.0, ra=host.ra, dec=host.dec, distance=host.distance)
    state = model.sample_parameters()

    assert model.get_param(state, "brightness") == 10.0
    assert model.get_param(state, "ra") == 1.0
    assert model.get_param(state, "dec") == 2.0
    assert model.get_param(state, "distance") == 3.0
    assert str(model) == "StaticSource_0"

    # Test that we have given a different name to the host.
    assert str(host) == "StaticSource_1"


def test_static_source_resample() -> None:
    """Check that we can call resample on the model parameters."""
    model = StaticSource(brightness=_sampler_fun, magnitude=100.0)

    num_samples = 100
    values = np.zeros((num_samples, 1))
    for i in range(num_samples):
        state = model.sample_parameters()
        values[i] = model.get_param(state, "brightness")

    # Check that the values fall within the expected bounds.
    assert np.all(values >= 0.0)
    assert np.all(values <= 100.0)

    # Check that the values are not all the same.
    assert not np.all(values == values[0])


def test_step_source() -> None:
    """Test that we can sample and create a StepSource object."""
    host = StaticSource(brightness=150.0, ra=1.0, dec=2.0, distance=3.0)
    model = StepSource(brightness=15.0, t0=1.0, t1=2.0, ra=host.ra, dec=host.dec, distance=host.distance)
    state = model.sample_parameters()

    param_values = model.get_local_params(state)
    assert param_values["brightness"] == 15.0
    assert param_values["t0"] == 1.0
    assert param_values["t1"] == 2.0
    assert param_values["ra"] == 1.0
    assert param_values["dec"] == 2.0
    assert param_values["distance"] == 3.0

    times = np.array([0.0, 1.0, 2.0, 3.0])
    wavelengths = np.array([100.0, 200.0])
    expected = np.array([[0.0, 0.0], [15.0, 15.0], [15.0, 15.0], [0.0, 0.0]])

    values = model.evaluate(times, wavelengths, state)
    assert values.shape == (4, 2)
    assert np.array_equal(values, expected)


def test_step_source_resample() -> None:
    """Check that we can call resample on the model parameters."""
    random.seed(1111)

    model = StepSource(
        brightness=FunctionNode(_sampler_fun, magnitude=100.0),
        t0=0.0,
        t1=FunctionNode(_sampler_fun, magnitude=5.0, offset=1.0),
    )

    num_samples = 1000
    brightness_vals = np.zeros((num_samples, 1))
    t_end_vals = np.zeros((num_samples, 1))
    t_start_vals = np.zeros((num_samples, 1))
    for i in range(num_samples):
        state = model.sample_parameters()
        brightness_vals[i] = model.get_param(state, "brightness")
        t_end_vals[i] = model.get_param(state, "t1")
        t_start_vals[i] = model.get_param(state, "t0")

    # Check that the values fall within the expected bounds.
    assert np.all(brightness_vals >= 0.0)
    assert np.all(brightness_vals <= 100.0)
    assert np.all(t_start_vals == 0.0)
    assert np.all(t_end_vals >= 1.0)
    assert np.all(t_end_vals <= 6.0)

    # Check that the expected values are close.
    assert abs(np.mean(brightness_vals) - 50.0) < 5.0
    assert abs(np.mean(t_end_vals) - 3.5) < 0.5

    # Check that the brightness or end values are not all the same.
    assert not np.all(brightness_vals == brightness_vals[0])
    assert not np.all(t_end_vals == t_end_vals[0])


def test_sin_wave_source() -> None:
    """Test that we can sample and create a SinWaveSource object."""
    model = SinWaveSource(brightness=15.0, frequency=1.0, t0=0.0)
    state = model.sample_parameters()

    times = np.array([0.0, 1.0 / 12.0, 1.0 / 4.0, 5.0 / 12.0, 0.5])
    wavelengths = np.array([100.0, 200.0])
    expected = np.array([[0.0, 0.0], [7.5, 7.5], [15.0, 15.0], [7.5, 7.5], [0.0, 0.0]])

    values = model.evaluate(times, wavelengths, state)
    assert values.shape == (5, 2)
    assert np.allclose(values, expected)
