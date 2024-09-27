import jax.random
import numpy as np
import pytest
from tdastro.rand_nodes.jax_random import JaxRandomFunc, JaxRandomNormal, build_jax_keys_from_nodes


def test_jax_random_uniform():
    """Test that we can generate numbers from a uniform distribution."""
    jax_node = JaxRandomFunc(jax.random.uniform)
    jax_node.set_graph_positions()
    keys = build_jax_keys_from_nodes(jax_node, base_seed=100)

    values = np.array([jax_node.generate(rng_info=keys) for _ in range(10_000)])
    assert len(np.unique(values)) > 10
    assert np.all(values <= 1.0)
    assert np.all(values >= 0.0)
    assert np.abs(np.mean(values) - 0.5) < 0.01

    # We run into a seeding error if we try to use a second node without
    # anything to make its seed unique.
    jax_node2 = JaxRandomFunc(jax.random.uniform)
    jax_node2.set_graph_positions()
    assert jax_node.node_hash == jax_node2.node_hash
    with pytest.raises(ValueError):
        _ = build_jax_keys_from_nodes([jax_node, jax_node2])

    # If we set the label for the second node, we get a different seed.
    jax_node2 = JaxRandomFunc(jax.random.uniform, node_label="second")
    jax_node2.set_graph_positions()
    assert jax_node.node_hash != jax_node2.node_hash
    keys = build_jax_keys_from_nodes([jax_node, jax_node2], base_seed=100)
    assert keys[jax_node.node_hash] != keys[jax_node2.node_hash]

    # We should get different numbers from the generators.
    values2 = np.array([jax_node2.generate(rng_info=keys) for _ in range(10_000)])
    assert not np.allclose(values, values2)

    # If we reuse the seed, we get the same numbers. We force overwrite the keys.
    keys[jax_node.node_hash] = jax.random.key(100)
    keys[jax_node2.node_hash] = jax.random.key(100)
    values = np.array([jax_node.generate(rng_info=keys) for _ in range(10_000)])
    values2 = np.array([jax_node2.generate(rng_info=keys) for _ in range(10_000)])
    assert np.allclose(values, values2)

    # We can change the range.
    jax_node3 = JaxRandomFunc(jax.random.uniform, minval=10.0, maxval=20.0)
    jax_node3.set_graph_positions()
    keys3 = build_jax_keys_from_nodes(jax_node3, base_seed=100)
    values = np.array([jax_node3.generate(rng_info=keys3) for _ in range(10_000)])
    assert len(np.unique(values)) > 10
    assert np.all(values <= 20.0)
    assert np.all(values >= 10.0)
    assert np.abs(np.mean(values) - 15.0) < 0.5

    # We can override the range dynamically.
    values = np.array([jax_node3.generate(rng_info=keys3, minval=2.0) for _ in range(10_000)])
    assert len(np.unique(values)) > 10
    assert np.all(values <= 20.0)
    assert np.all(values >= 2.0)
    assert np.abs(np.mean(values) - 11.0) < 0.5


def test_jax_random_uniform_multiple():
    """Test that we can generate multiple numbers at once from a uniform distribution in JAX."""
    jax_node = JaxRandomFunc(jax.random.uniform)
    jax_node.set_graph_positions()
    keys = build_jax_keys_from_nodes(jax_node, base_seed=100)

    state = jax_node.sample_parameters(rng_info=keys, num_samples=10_000)
    samples = jax_node.get_param(state, "function_node_result")
    assert len(samples) == 10_000
    assert len(np.unique(samples)) > 1_000
    assert np.abs(np.mean(samples) - 0.5) < 0.01


def test_jax_random_normal():
    """Test that we can generate numbers from a normal distribution."""
    jax_node = JaxRandomNormal(loc=100.0, scale=10.0)
    jax_node.set_graph_positions()
    keys = build_jax_keys_from_nodes(jax_node, base_seed=100)

    values = np.array([jax_node.generate(rng_info=keys) for _ in range(1000)])
    assert np.abs(np.mean(values) - 100.0) < 0.5
    assert np.abs(np.std(values) - 10.0) < 0.5

    # If we reuse the seed, we get the same numbers.
    jax_node2 = JaxRandomNormal(loc=100.0, scale=10.0)
    jax_node2.set_graph_positions()
    keys2 = build_jax_keys_from_nodes(jax_node2, base_seed=100)
    assert keys[jax_node.node_hash] == keys2[jax_node2.node_hash]

    values2 = np.array([jax_node2.generate(rng_info=keys2) for _ in range(1000)])
    assert np.allclose(values, values2)
