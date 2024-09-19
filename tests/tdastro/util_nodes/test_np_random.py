import numpy as np
import pytest
from tdastro.util_nodes.np_random import NumpyRandomFunc, build_rngs_from_hashes, build_rngs_from_nodes


def test_build_rngs_from_hashes():
    """Test that we can create seeded random number generators from hash values."""
    rngs = build_rngs_from_hashes([0, 1, 2, 3, 4, 5, 6, 7], base_seed=0)
    values = [gen.integers(0, 10000) for gen in rngs.values()]
    assert len(np.unique(values) == len(rngs))

    # If we reuse a hash id, we get the same values.
    rngs2 = build_rngs_from_hashes([3, 5], base_seed=0)
    assert rngs2[3].integers(0, 10000) == values[3]
    assert rngs2[5].integers(0, 10000) == values[5]

    # But not if we change the base seed.
    rngs3 = build_rngs_from_hashes([3, 5], base_seed=1)
    assert rngs3[3].integers(0, 10000) != values[3]
    assert rngs3[5].integers(0, 10000) != values[5]

    # We fail if we have duplicate hash_values.
    with pytest.raises(ValueError):
        _ = build_rngs_from_hashes([1, 2, 1])


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

    # But we can override the seed and get the same results again.
    np_node2.set_seed(100)
    values2 = np.array([np_node2.generate() for _ in range(10_000)])
    assert np.allclose(values, values2)

    # We can change the range.
    np_node3 = NumpyRandomFunc("uniform", low=10.0, high=20.0, seed=100)
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


def test_numpy_random_uniform_multi():
    """Test that we can many generate numbers at once from a uniform distribution."""
    np_node = NumpyRandomFunc("uniform", seed=100)
    state = np_node.sample_parameters(num_samples=10_000)
    samples = np_node.get_param(state, "function_node_result")
    assert len(samples) == 10_000
    assert len(np.unique(samples)) > 1_000
    assert np.abs(np.mean(samples) - 0.5) < 0.01


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


def test_numpy_random_given_rng():
    """Test that we can generate numbers from a uniform distribution."""
    np_node1 = NumpyRandomFunc("uniform", seed=100, node_label="node1")
    np_node1.set_graph_positions()

    np_node2 = NumpyRandomFunc("uniform", seed=100, node_label="node2")
    np_node2.set_graph_positions()

    # The first value generated is the same because they are using the node's seed.
    assert np.abs(np_node1.generate() - np_node2.generate()) < 1e-12

    # But if we use a dictionary of rngs, we will generate different ones
    # because the hash values are different.
    rngs = build_rngs_from_nodes([np_node1, np_node2], base_seed=100)
    assert np.abs(np_node1.generate(rng_info=rngs) - np_node2.generate(rng_info=rngs)) > 1e-12

    # We correctly fail if we don't have the correct hashes.
    with pytest.raises(KeyError):
        _ = np_node1.generate(rng_info={})
