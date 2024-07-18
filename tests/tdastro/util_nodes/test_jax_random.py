import jax.random
import numpy as np
from tdastro.util_nodes.jax_random import JaxRandomFunc, JaxRandomNormal


def test_jax_random_uniform():
    """Test that we can generate numbers from a uniform distribution."""
    jax_node = JaxRandomFunc(jax.random.uniform, graph_base_seed=100)

    values = np.array([jax_node.compute() for _ in range(10_000)])
    assert len(np.unique(values)) > 10
    assert np.all(values <= 1.0)
    assert np.all(values >= 0.0)
    assert np.abs(np.mean(values) - 0.5) < 0.01

    # If we reuse the seed, we get the same number.
    jax_node2 = JaxRandomFunc(jax.random.uniform, graph_base_seed=100)
    values2 = np.array([jax_node2.compute() for _ in range(10_000)])
    assert np.allclose(values, values2)

    # We can change the range.
    jax_node3 = JaxRandomFunc(jax.random.uniform, graph_base_seed=101, minval=10.0, maxval=20.0)
    values = np.array([jax_node3.compute() for _ in range(10_000)])
    assert len(np.unique(values)) > 10
    assert np.all(values <= 20.0)
    assert np.all(values >= 10.0)
    assert np.abs(np.mean(values) - 15.0) < 0.05

    # We can override the range dynamically.
    values = np.array([jax_node3.compute(minval=2.0) for _ in range(10_000)])
    assert len(np.unique(values)) > 10
    assert np.all(values <= 20.0)
    assert np.all(values >= 2.0)
    assert np.abs(np.mean(values) - 11.0) < 0.05


def test_jax_random_normal():
    """Test that we can generate numbers from a normal distribution."""
    jax_node = JaxRandomNormal(loc=100.0, scale=10.0, graph_base_seed=100)

    values = np.array([jax_node.compute() for _ in range(10_000)])
    assert np.abs(np.mean(values) - 100.0) < 0.5
    assert np.abs(np.std(values) - 10.0) < 0.5

    # If we reuse the seed, we get the same number.
    jax_node2 = JaxRandomNormal(loc=100.0, scale=10.0, graph_base_seed=100)
    values2 = np.array([jax_node2.compute() for _ in range(10_000)])
    assert np.allclose(values, values2)
