import numpy as np
from tdastro.util_nodes.np_random import NumpyRandomFunc


def test_numpy_random_uniform():
    """Test that we can generate numbers from a uniform distribution."""
    np_node = NumpyRandomFunc("uniform", seed=100)

    values = np.array([np_node.generate() for _ in range(10_000)])
    assert len(np.unique(values)) > 10
    assert np.all(values <= 1.0)
    assert np.all(values >= 0.0)
    assert np.abs(np.mean(values) - 0.5) < 0.01

    # If we reuse the seed, we get the same numbers.
    np_node2 = NumpyRandomFunc("uniform", seed=100)
    values2 = np.array([np_node2.generate() for _ in range(10_000)])
    assert np.allclose(values, values2)

    # If we use a different seed, we get different numbers
    np_node2 = NumpyRandomFunc("uniform", seed=101)
    values2 = np.array([np_node2.generate() for _ in range(10_000)])
    assert not np.allclose(values, values2)

    # We can change the range.
    np_node3 = NumpyRandomFunc("uniform", low=10.0, high=20.0)
    values = np.array([np_node3.generate() for _ in range(10_000)])
    assert len(np.unique(values)) > 10
    assert np.all(values <= 20.0)
    assert np.all(values >= 10.0)
    assert np.abs(np.mean(values) - 15.0) < 0.5

    # We can override the range dynamically.
    values = np.array([np_node3.generate(low=2.0) for _ in range(10_000)])
    assert len(np.unique(values)) > 10
    assert np.all(values <= 20.0)
    assert np.all(values >= 2.0)
    assert np.abs(np.mean(values) - 11.0) < 0.5


def test_numpy_random_normal():
    """Test that we can generate numbers from a normal distribution."""
    np_node = NumpyRandomFunc("normal", loc=100.0, scale=10.0, seed=100)

    values = np.array([np_node.generate() for _ in range(10_000)])
    assert np.abs(np.mean(values) - 100.0) < 0.5
    assert np.abs(np.std(values) - 10.0) < 0.5

    # If we reuse the seed, we get the same number.
    np_node2 = NumpyRandomFunc("normal", loc=100.0, scale=10.0, seed=100)
    values2 = np.array([np_node2.generate() for _ in range(10_000)])
    assert np.allclose(values, values2)
