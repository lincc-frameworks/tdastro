import os

import numpy as np
from tdastro.sources.spline_model import SplineModel
from tdastro.utils.wave_extrapolate import ConstantExtrapolation


def test_spline_model_flat() -> None:
    """Test that we can sample and create a flat SplineModel object."""
    times = np.linspace(1.0, 5.0, 20)
    wavelengths = np.linspace(100.0, 500.0, 25)
    fluxes = np.full((len(times), len(wavelengths)), 1.0)
    model = SplineModel(times, wavelengths, fluxes)
    assert str(model) == "SplineModel"

    test_times = np.array([0.0, 1.0, 2.0, 3.0, 10.0])
    test_waves = np.array([0.0, 100.0, 150.0, 200.0, 500.0, 1000.0])

    state = model.sample_parameters()
    values = model.evaluate(test_times, test_waves, state)
    assert values.shape == (5, 6)

    # The first and last times are outside the range of the model, so they should be 0.0
    # and the rest should be 1.0.
    expected = np.full_like(values, 1.0)
    expected[:, 0] = 0.0
    expected[:, -1] = 0.0
    np.testing.assert_array_almost_equal(values, expected)

    model2 = SplineModel(times, wavelengths, fluxes, amplitude=5.0, node_label="test")
    assert str(model2) == "test"

    state2 = model2.sample_parameters()
    values2 = model2.evaluate(test_times, test_waves, state2)
    assert values2.shape == (5, 6)

    # The first and last times are outside the range of the model, so they should be 0.0
    # and the rest should be 5.0.
    expected2 = np.full_like(values2, 5.0)
    expected2[:, 0] = 0.0
    expected2[:, -1] = 0.0
    np.testing.assert_array_almost_equal(values2, expected2)


def test_spline_model_interesting() -> None:
    """Test that we can sample and create a flat SplineModel object."""
    times = np.array([1.0, 2.0, 3.0])
    wavelengths = np.array([10.0, 20.0, 30.0])
    fluxes = np.array([[1.0, 5.0, 1.0], [5.0, 10.0, 5.0], [1.0, 5.0, 3.0]])
    model = SplineModel(
        times,
        wavelengths,
        fluxes,
        time_degree=1,
        wave_degree=1,
        wave_extrapolation=ConstantExtrapolation(value=0.1),
    )
    state = model.sample_parameters()

    test_times = np.array([1.0, 1.5, 2.0, 3.0])
    test_waves = np.array([10.0, 15.0, 20.0, 30.0, 40.0, 50.0])
    values = model.evaluate(test_times, test_waves, state)
    assert values.shape == (4, 6)

    # The last two wavelengths are outside the range of the model, so they should be 0.1
    expected = np.array(
        [
            [1.0, 3.0, 5.0, 1.0, 0.1, 0.1],
            [3.0, 5.25, 7.5, 3.0, 0.1, 0.1],
            [5.0, 7.5, 10.0, 5.0, 0.1, 0.1],
            [1.0, 3.0, 5.0, 3.0, 0.1, 0.1],
        ]
    )
    np.testing.assert_array_almost_equal(values, expected)


def test_spline_model_interesting_t0() -> None:
    """Test that we can sample and create a flat SplineModel object."""
    t0_times = np.array([1.0, 2.0, 3.0])
    wavelengths = np.array([10.0, 20.0, 30.0])
    fluxes = np.array([[1.0, 5.0, 1.0], [5.0, 10.0, 5.0], [1.0, 5.0, 3.0]])
    model = SplineModel(
        t0_times,
        wavelengths,
        fluxes,
        time_degree=1,
        wave_degree=1,
        t0=60676.0,
    )
    state = model.sample_parameters()

    # Test times correspond to t0 + [1.0, 1.5, 2.0, 3.0]
    test_times = np.array([60677.0, 60677.5, 60678.0, 60679.0])
    test_waves = np.array([10.0, 15.0, 20.0, 30.0])
    values = model.evaluate(test_times, test_waves, state)
    assert values.shape == (4, 4)

    expected = np.array(
        [[1.0, 3.0, 5.0, 1.0], [3.0, 5.25, 7.5, 3.0], [5.0, 7.5, 10.0, 5.0], [1.0, 3.0, 5.0, 3.0]]
    )
    np.testing.assert_array_almost_equal(values, expected)


def test_spine_model_from_file(test_data_dir):
    """Test that we can create a SplineModel from a file."""
    filename = os.path.join(test_data_dir, "truncated-salt2-h17/salt2_template_0.dat")

    model = SplineModel.from_file(filename)
    assert len(model._times) == 26
    assert len(model._wavelengths) == 401
