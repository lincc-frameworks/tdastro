import numpy as np
import pytest
from tdastro.graph_state import GraphState


def test_create_single_sample_graph_state():
    """Test that we can create and access a single sample GraphState."""
    state = GraphState()
    assert len(state) == 0
    assert state.num_samples == 1

    state.set("a", "v1", 1.0)
    state.set("a", "v2", 2.0)
    state.set("b", "v1", 3.0)
    assert len(state) == 3
    assert state["a"]["v1"] == 1.0
    assert state["a"]["v2"] == 2.0
    assert state["b"]["v1"] == 3.0

    with pytest.raises(KeyError):
        _ = state["a"]["v3"]
    with pytest.raises(KeyError):
        _ = state["c"]["v1"]

    # Check tht we can get all the values for a specific node.
    a_vals = state.get_node_state("a")
    assert len(a_vals) == 2
    assert a_vals["v1"] == 1.0
    assert a_vals["v2"] == 2.0

    # Can't access the second value of a single element sample.
    with pytest.raises(ValueError):
        _ = state.get_node_state("a", 2)

    b_vals = state.get_node_state("b")
    assert len(b_vals) == 1
    assert b_vals["v1"] == 3.0

    with pytest.raises(KeyError):
        _ = state.get_node_state("c")

    # We can extract a single sample.
    new_state = state.extract_single_sample(0)
    assert len(new_state) == 3
    assert new_state.num_samples == 1
    assert state["a"]["v1"] == 1.0
    assert state["a"]["v2"] == 2.0
    assert state["b"]["v1"] == 3.0

    # We can overwrite settings.
    state.set("a", "v1", 10.0)
    assert len(state) == 3
    assert state["a"]["v1"] == 10.0


def test_create_multi_sample_graph_state():
    """Test that we can create and access a multi-sample GraphState."""
    state = GraphState(5)
    assert len(state) == 0
    assert state.num_samples == 5

    state.set("a", "v1", 1.0)
    state.set("a", "v2", np.array([2.0, 2.5, 3.0, 3.5, 4.0]))
    state.set("b", "v1", np.array([-2.0, -2.5, -3.0, -3.5, -4.0]))
    assert len(state) == 3
    assert np.allclose(state["a"]["v1"], [1.0, 1.0, 1.0, 1.0, 1.0])
    assert np.allclose(state["a"]["v2"], [2.0, 2.5, 3.0, 3.5, 4.0])
    assert np.allclose(state["b"]["v1"], [-2.0, -2.5, -3.0, -3.5, -4.0])

    # Check tht we can get all the values for a specific node and sample.
    a_vals = state.get_node_state("a")
    assert len(a_vals) == 2
    assert a_vals["v1"] == 1.0
    assert a_vals["v2"] == 2.0

    a_vals = state.get_node_state("a", 1)
    assert len(a_vals) == 2
    assert a_vals["v1"] == 1.0
    assert a_vals["v2"] == 2.5

    # We can extract a single sample.
    new_state = state.extract_single_sample(3)
    assert len(new_state) == 3
    assert new_state.num_samples == 1
    assert new_state["a"]["v1"] == 1.0
    assert new_state["a"]["v2"] == 3.5
    assert new_state["b"]["v1"] == -3.5

    # Error if we send an array of the wrong length.
    with pytest.raises(ValueError):
        state.set("b", "v2", [-2.0, -2.5, -3.0, -3.5, -4.0, 1.0])


def test_create_multi_sample_graph_state_no_reference():
    """Test that we can create and access a multi-sample GraphState with variables
    that were created from each other."""
    state = GraphState(5)
    state.set("a", "v1", 1.0)
    state.set("a", "v2", np.array([2.0, 2.5, 3.0, 3.5, 4.0]))
    state.set("b", "v1", state["a"]["v2"])
    assert len(state) == 3
    assert np.allclose(state["a"]["v1"], [1.0, 1.0, 1.0, 1.0, 1.0])
    assert np.allclose(state["a"]["v2"], [2.0, 2.5, 3.0, 3.5, 4.0])
    assert np.allclose(state["b"]["v1"], [2.0, 2.5, 3.0, 3.5, 4.0])

    # Check that (c, v1) is not just pointing to the data held in (a, v2),
    # but is its own copy.
    state["a"]["v2"][2] = 5.0
    assert np.allclose(state["a"]["v2"], [2.0, 2.5, 5.0, 3.5, 4.0])
    assert np.allclose(state["b"]["v1"], [2.0, 2.5, 3.0, 3.5, 4.0])
