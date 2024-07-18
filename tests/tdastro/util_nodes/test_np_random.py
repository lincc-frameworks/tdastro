import numpy as np
from tdastro.util_nodes.np_random import NumpyRandomFunc


def test_numpy_random_uniform():
    """Test that we can generate numbers from a uniform distribution."""
    np_node = NumpyRandomFunc("uniform", graph_base_seed=100)

    values = np.array([np_node.compute() for _ in range(10_000)])
    assert len(np.unique(values)) > 10
    assert np.all(values <= 1.0)
    assert np.all(values >= 0.0)
    assert np.abs(np.mean(values) - 0.5) < 0.01

    # If we reuse the seed, we get the same number.
    np_node2 = NumpyRandomFunc("uniform", graph_base_seed=100)
    values2 = np.array([np_node2.compute() for _ in range(10_000)])
    assert np.allclose(values, values2)

    # We can change the range.
    np_node3 = NumpyRandomFunc("uniform", graph_base_seed=101, low=10.0, high=20.0)
    values = np.array([np_node3.compute() for _ in range(10_000)])
    assert len(np.unique(values)) > 10
    assert np.all(values <= 20.0)
    assert np.all(values >= 10.0)
    assert np.abs(np.mean(values) - 15.0) < 0.05

    # We can override the range dynamically.
    values = np.array([np_node3.compute(low=2.0) for _ in range(10_000)])
    assert len(np.unique(values)) > 10
    assert np.all(values <= 20.0)
    assert np.all(values >= 2.0)
    assert np.abs(np.mean(values) - 11.0) < 0.05


def test_numpy_random_normal():
    """Test that we can generate numbers from a normal distribution."""
    np_node = NumpyRandomFunc("normal", loc=100.0, scale=10.0, graph_base_seed=100)

    values = np.array([np_node.compute() for _ in range(10_000)])
    assert np.abs(np.mean(values) - 100.0) < 0.5
    assert np.abs(np.std(values) - 10.0) < 0.5

    # If we reuse the seed, we get the same number.
    jax_node2 = NumpyRandomFunc("normal", loc=100.0, scale=10.0, graph_base_seed=100)
    values2 = np.array([jax_node2.compute() for _ in range(10_000)])
    assert np.allclose(values, values2)
