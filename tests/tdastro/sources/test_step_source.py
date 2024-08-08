import random

import numpy as np
from tdastro.base_models import FunctionNode
from tdastro.sources.static_source import StaticSource
from tdastro.sources.step_source import StepSource


def _sample_brightness(magnitude, **kwargs):
    """Return a random value between 0 and magnitude

    Parameters
    ----------
    magnitude : `float`
        The range of brightness magnitude
    **kwargs : `dict`, optional
        Absorbs additional parameters
    """
    return magnitude * random.random()


def _sample_end(duration):
    """Return a random value between 1 and 1 + duration

    Parameters
    ----------
    duration : `float`
        The range of time lengths
    **kwargs : `dict`, optional
        Absorbs additional parameters
    """
    return duration * random.random() + 1.0


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
        brightness=FunctionNode(_sample_brightness, magnitude=100.0),
        t0=0.0,
        t1=FunctionNode(_sample_end, duration=5.0),
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
