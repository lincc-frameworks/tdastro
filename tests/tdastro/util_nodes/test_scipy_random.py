import numpy as np
from tdastro.util_nodes.np_random import NumpyRandomFunc
from tdastro.util_nodes.scipy_random import NumericalInversePolynomialFunc


class FlatDist:
    """A flat (uniform) distribution used for testing.

    Attributes
    ----------
    min_val : `float`
        The minimum value of the distribution.
    max_val : `float`
        The maximum value of the distribution.
    width : `float`
        The width of the distribution.
    """

    def __init__(self, min_val, max_val):
        if min_val >= max_val:
            raise ValueError("Invalid flat_dist bounds.")
        self.min_val = min_val
        self.max_val = max_val
        self.width = self.max_val - self.min_val

    def pdf(self, xval):
        """The pdf of the distribution.

        Parameters
        ----------
        xval: `float`
            The x value to evaluate.

        Returns
        -------
        result : `float`
            The value of the pdf function at the given points.
        """
        if xval <= self.min_val:
            return 0.0
        elif xval >= self.max_val:
            return 0.0
        else:
            return 1.0 / self.width


def test_numerical_inverse_polynomial_func_object():
    """Test that we can generate numbers from a uniform distribution."""
    dist = FlatDist(min_val=0.25, max_val=0.75)
    scipy_node = NumericalInversePolynomialFunc(dist, seed=100)

    # Test that we get a uniform distribution.
    num_bins = 10
    num_samples = 10_000
    counts = [0] * num_bins
    all_values = [0] * num_samples
    for i in range(10_000):
        value = scipy_node.generate()
        assert 0.25 <= value <= 0.75

        all_values[i] = value
        bin = int((value - 0.25) * (num_bins / 0.5))
        counts[bin] += 1

    for idx in range(num_bins):
        assert np.abs(counts[idx] - 10_000 / num_bins) < 200.0

    # If we reuse the same seed, we get the same values.
    dist2 = FlatDist(min_val=0.25, max_val=0.75)
    scipy_node2 = NumericalInversePolynomialFunc(dist2, seed=100)
    all_values2 = [0] * num_samples
    for i in range(10_000):
        all_values2[i] = scipy_node2.generate()
    assert np.allclose(all_values, all_values2)

    # If we use a different seed, we get different values.
    dist3 = FlatDist(min_val=0.25, max_val=0.75)
    scipy_node3 = NumericalInversePolynomialFunc(dist3, seed=101)
    all_values3 = [0] * num_samples
    for i in range(10_000):
        all_values3[i] = scipy_node3.generate()
    assert not np.allclose(all_values, all_values3)

    # We can batch generate results.
    state = scipy_node.sample_parameters(num_samples=500)
    samples = scipy_node.get_param(state, "function_node_result")
    assert len(samples) == 500
    assert len(np.unique(samples)) > 50
    assert np.abs(np.mean(samples) - 0.5) < 0.01


def test_numerical_inverse_polynomial_func_class():
    """Test that we can generate numbers from a uniform distribution."""
    scipy_node = NumericalInversePolynomialFunc(
        FlatDist,
        min_val=NumpyRandomFunc("uniform", low=0.0, high=0.4, seed=101),
        max_val=NumpyRandomFunc("uniform", low=0.6, high=1.0, seed=102),
        seed=100,
    )

    # Check that when we sample both the values and the distribution's parameters.
    num_samples = 1000
    states = scipy_node.sample_parameters(num_samples=num_samples)
    values = scipy_node.get_param(states, "function_node_result")
    min_vals = scipy_node.get_param(states, "min_val")
    max_vals = scipy_node.get_param(states, "max_val")
    assert len(np.unique(values)) == num_samples
    assert len(np.unique(min_vals)) == num_samples
    assert len(np.unique(max_vals)) == num_samples

    # Test that the generated values are consistent with the distributions.
    assert np.all(values >= min_vals)
    assert np.all(values <= max_vals)
