import numpy as np
from tdastro.sources.spline_model import SplineModel


def test_spline_model_flat() -> None:
    """Test that we can sample and create a flat SplineModel object."""
    times = np.linspace(1.0, 5.0, 20)
    wavelengths = np.linspace(100.0, 500.0, 25)
    fluxes = np.full((len(times), len(wavelengths)), 1.0)
    model = SplineModel(times, wavelengths, fluxes)

    test_times = np.array([0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 10.0])
    test_waves = np.array([100.0, 200.0, 200.0, 0.0, 200.0, 1000.0, 200.0, 200.0])

    values = model.evaluate(test_times, test_waves)
    expected = np.array([1.0] * len(test_times))
    np.testing.assert_array_almost_equal(values, expected)

    model2 = SplineModel(times, wavelengths, fluxes, amplitude=5.0)
    values2 = model2.evaluate(test_times, test_waves)
    expected2 = np.array([5.0] * len(test_times))
    np.testing.assert_array_almost_equal(values2, expected2)


def test_spline_model_interesting() -> None:
    """Test that we can sample and create a flat SplineModel object."""
    times = np.array([1.0, 2.0, 3.0])
    wavelengths = np.array([10.0, 20.0, 30.0])
    fluxes = np.array([[1.0, 5.0, 1.0], [5.0, 10.0, 5.0], [1.0, 5.0, 3.0]])
    model = SplineModel(times, wavelengths, fluxes, time_degree=1, wave_degree=1)

    test_times = np.array([1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.5, 2.0])
    test_waves = np.array([10.0, 10.0, 10.0, 20.0, 20.0, 20.0, 30.0, 30.0, 30.0, 10.0, 15.0])
    values = model.evaluate(test_times, test_waves)

    expected = np.array([1.0, 5.0, 1.0, 5.0, 10.0, 5.0, 1.0, 5.0, 3.0, 3.0, 7.5])
    np.testing.assert_array_almost_equal(values, expected)
