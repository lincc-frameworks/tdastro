import os
import tempfile
from pathlib import Path

import numpy as np
import pytest
from astropy.table import Table
from tdastro.graph_state import GraphState, transpose_dict_of_list


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

    # We can create a human readable string representation of the GraphState.
    debug_str = str(state)
    assert debug_str == "a:\n    v1: 1.0\n    v2: 2.0\nb:\n    v1: 3.0"

    # Check that we can get all the values for a specific node.
    a_vals = state.get_node_state("a")
    assert len(a_vals) == 2
    assert a_vals["v1"] == 1.0
    assert a_vals["v2"] == 2.0

    # Can't access an out-of-bounds sample_num.
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

    # Test we cannot use a name containing the separator as a substring.
    with pytest.raises(ValueError):
        state.set("a|>b", "v1", 10.0)
    with pytest.raises(ValueError):
        state.set("b", "v1|>v3", 10.0)


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


def test_create_multi_sample_graph_state_reference():
    """Test that we can create and access a multi-sample GraphState with variables
    that were created from each other (with and without references)."""
    state = GraphState(5)
    state.set("a", "v1", 1.0)
    state.set("a", "v2", np.array([2.0, 2.5, 3.0, 3.5, 4.0]))
    state.set("b", "v1", state["a"]["v2"])
    assert len(state) == 3
    assert np.allclose(state["a"]["v1"], [1.0, 1.0, 1.0, 1.0, 1.0])
    assert np.allclose(state["a"]["v2"], [2.0, 2.5, 3.0, 3.5, 4.0])
    assert np.allclose(state["b"]["v1"], [2.0, 2.5, 3.0, 3.5, 4.0])

    # Check that (b, v1) is just pointing to the data held in (a, v2).
    state["a"]["v2"][2] = 5.0
    assert np.allclose(state["a"]["v2"], [2.0, 2.5, 5.0, 3.5, 4.0])
    assert np.allclose(state["b"]["v1"], [2.0, 2.5, 5.0, 3.5, 4.0])

    # If we set force_copy to True then (b, v1) should point to its own
    # copy of the data and not inherit any changes to (a, v2).
    state2 = GraphState(5)
    state2.set("a", "v1", 1.0)
    state2.set("a", "v2", np.array([2.0, 2.5, 3.0, 3.5, 4.0]))
    state2.set("b", "v1", state2["a"]["v2"], force_copy=True)
    state2["a"]["v2"][2] = 5.0
    assert np.allclose(state2["a"]["v2"], [2.0, 2.5, 5.0, 3.5, 4.0])
    assert np.allclose(state2["b"]["v1"], [2.0, 2.5, 3.0, 3.5, 4.0])


def test_graph_state_equal():
    """Test that we use == on GraphStates."""
    state1 = GraphState(num_samples=2)
    state1.set("a", "v1", [1.0, 2.0])
    state1.set("a", "v2", [2.0, 3.0])
    state1.set("b", "v1", [3.0, 4.0])

    state2 = GraphState(num_samples=2)
    state2.set("a", "v1", [1.0, 2.0])
    state2.set("a", "v2", [2.0, 3.0])
    state2.set("b", "v1", [3.0, 4.0])

    assert state1 == state2

    state2.set("b", "v1", [3.0, 5.0])
    assert state1 != state2

    state2.set("b", "v1", [3.0, 4.0])
    assert state1 == state2

    state2.set("b", "v2", [3.0, 4.0])
    assert state1 != state2

    state3 = GraphState(num_samples=3)
    state3.set("a", "v1", [1.0, 2.0, 3.0])
    state3.set("a", "v2", [2.0, 3.0, 4.0])
    state3.set("b", "v1", [3.0, 4.0, 5.0])
    assert state3 != state1
    assert state3 != state2

    # Test that equality works with a single sample (not arrays).
    state4 = GraphState(num_samples=1)
    state4.set("a", "v1", 1.0)
    state4.set("a", "v2", 2.0)
    state4.set("b", "v1", 3.0)

    state5 = GraphState(num_samples=1)
    state5.set("a", "v1", 1.0)
    state5.set("a", "v2", 2.0)
    state5.set("b", "v1", 3.0)

    assert state4 == state5

    state5.set("a", "v2", 2.5)
    assert state4 != state5


def test_graph_state_fixed():
    """Test that we respected the 'fixed' flag for GraphState."""
    state = GraphState()
    assert len(state) == 0
    state.set("a", "v1", 1.0)
    state.set("a", "v2", 2.0)
    state.set("b", "v1", 3.0, fixed=True)
    assert len(state) == 3
    assert state["a"]["v1"] == 1.0
    assert state["a"]["v2"] == 2.0
    assert state["b"]["v1"] == 3.0

    # Try changing each of the states. Only two should actually change.
    state.set("a", "v1", 4.0)
    state.set("a", "v2", 5.0)
    state.set("b", "v1", 6.0)
    assert state["a"]["v1"] == 4.0
    assert state["a"]["v2"] == 5.0
    assert state["b"]["v1"] == 3.0


def test_graph_state_update():
    """Test that we can update a single sample GraphState."""
    state = GraphState()
    state.set("a", "v1", 1.0)
    state.set("a", "v2", 2.0)
    state.set("b", "v1", 3.0)

    state2 = GraphState()
    state2.set("a", "v1", 4.0)
    state2.set("a", "v3", 5.0)
    state2.set("c", "v1", 6.0)
    state2.set("c", "v2", 7.0)

    assert len(state) == 3
    assert len(state2) == 4

    # We set 3 new parameters and overwrite one.
    state.update(state2)
    assert len(state) == 6
    assert state["a"]["v1"] == 4.0
    assert state["a"]["v2"] == 2.0
    assert state["a"]["v3"] == 5.0
    assert state["b"]["v1"] == 3.0
    assert state["c"]["v1"] == 6.0
    assert state["c"]["v2"] == 7.0

    # We set 3 new parameters and overwrite one.
    state3 = {"a": {"v2": 8.0, "v4": 9.0}, "d": {"v1": 10.0}}
    state.update(state3)
    assert len(state) == 8
    assert state["a"]["v1"] == 4.0
    assert state["a"]["v2"] == 8.0
    assert state["a"]["v3"] == 5.0
    assert state["a"]["v4"] == 9.0
    assert state["b"]["v1"] == 3.0
    assert state["c"]["v1"] == 6.0
    assert state["c"]["v2"] == 7.0
    assert state["d"]["v1"] == 10.0

    # Test we cannot update with mismatched number of samples.
    state4 = GraphState(num_samples=2)
    state4.set("e", "v1", 1.0)
    with pytest.raises(ValueError):
        state.update(state4)


def test_graph_state_from_table():
    """Test that we can create a GraphState from an AstroPy Table."""
    input = Table(
        {
            GraphState.extended_param_name("a", "v1"): [1.0, 2.0, 3.0],
            GraphState.extended_param_name("a", "v2"): [4.0, 5.0, 6.0],
            GraphState.extended_param_name("b", "v1"): [7.0, 8.0, 9.0],
        }
    )
    state = GraphState.from_table(input)

    assert len(state) == 3
    assert state.num_samples == 3
    np.testing.assert_allclose(state["a"]["v1"], [1.0, 2.0, 3.0])
    np.testing.assert_allclose(state["a"]["v2"], [4.0, 5.0, 6.0])
    np.testing.assert_allclose(state["b"]["v1"], [7.0, 8.0, 9.0])


def test_graph_state_to_dict():
    """Test that we can create a dictionary from a GraphState."""
    state = GraphState(num_samples=3)
    state.set("a", "v1", [1.0, 2.0, 3.0])
    state.set("a", "v2", [3.0, 4.0, 5.0])
    state.set("b", "v1", [6.0, 7.0, 8.0])

    result = state.to_dict()
    assert len(result) == 3
    np.testing.assert_allclose(
        result[GraphState.extended_param_name("a", "v1")].data,
        [1.0, 2.0, 3.0],
    )
    np.testing.assert_allclose(
        result[GraphState.extended_param_name("a", "v2")].data,
        [3.0, 4.0, 5.0],
    )
    np.testing.assert_allclose(
        result[GraphState.extended_param_name("b", "v1")].data,
        [6.0, 7.0, 8.0],
    )


def test_graph_state_to_table():
    """Test that we can create an AstroPy Table from a GraphState."""
    state = GraphState(num_samples=3)
    state.set("a", "v1", [1.0, 2.0, 3.0])
    state.set("a", "v2", [3.0, 4.0, 5.0])
    state.set("b", "v1", [6.0, 7.0, 8.0])

    result = state.to_table()
    assert len(result) == 3
    np.testing.assert_allclose(
        result[GraphState.extended_param_name("a", "v1")].data,
        [1.0, 2.0, 3.0],
    )
    np.testing.assert_allclose(
        result[GraphState.extended_param_name("a", "v2")].data,
        [3.0, 4.0, 5.0],
    )
    np.testing.assert_allclose(
        result[GraphState.extended_param_name("b", "v1")].data,
        [6.0, 7.0, 8.0],
    )


def test_graph_state_update_multi():
    """Test that we can update a single sample GraphState."""
    state = GraphState(num_samples=3)
    state.set("a", "v1", [1.0, 2.0, 3.0])
    state.set("a", "v2", [3.0, 4.0, 5.0])
    state.set("b", "v1", [6.0, 7.0, 8.0])

    state2 = GraphState(num_samples=3)
    state2.set("a", "v1", [9.0, 10.0, 11.0])
    state2.set("c", "v1", [12.0, 13.0, 14.0])

    assert len(state) == 3
    assert len(state2) == 2

    # We set one new parameter and overwrite one.
    state.update(state2)
    assert len(state) == 4
    assert np.allclose(state["a"]["v1"], [9.0, 10.0, 11.0])
    assert np.allclose(state["a"]["v2"], [3.0, 4.0, 5.0])
    assert np.allclose(state["b"]["v1"], [6.0, 7.0, 8.0])
    assert np.allclose(state["c"]["v1"], [12.0, 13.0, 14.0])

    # If we add a parameter with sample_size = 1, we correctly expand it out.
    state3 = {"a": {"v2": 15.0}, "d": {"v1": 16.0}}
    state.update(state3)
    assert len(state) == 5
    assert np.allclose(state["a"]["v1"], [9.0, 10.0, 11.0])
    assert np.allclose(state["a"]["v2"], [15.0, 15.0, 15.0])
    assert np.allclose(state["b"]["v1"], [6.0, 7.0, 8.0])
    assert np.allclose(state["c"]["v1"], [12.0, 13.0, 14.0])
    assert np.allclose(state["d"]["v1"], [16.0, 16.0, 16.0])

    # Test we cannot update with mismatched number of samples.
    state4 = GraphState(num_samples=2)
    state4.set("e", "v1", 1.0)
    with pytest.raises(ValueError):
        state.update(state4)


def test_graph_to_from_file():
    """Test that we can create an AstroPy Table from a GraphState."""
    state = GraphState(num_samples=3)
    state.set("a", "v1", [1.0, 2.0, 3.0])
    state.set("a", "v2", [3.0, 4.0, 5.0])
    state.set("b", "v1", [6.0, 7.0, 8.0])

    with tempfile.TemporaryDirectory() as dir_name:
        file_path = os.path.join(dir_name, "state.ecsv")
        assert not Path(file_path).is_file()

        state.save_to_file(file_path)
        assert Path(file_path).is_file()

        state2 = GraphState.from_file(file_path)
        assert state == state2

        # Cannot overwrite with it set to False, but works when set to True.
        with pytest.raises(OSError):
            state.save_to_file(file_path)
        state.save_to_file(file_path, overwrite=True)


def test_transpose_dict_of_list():
    """Test the transpose_dict_of_list helper function"""
    input_dict = {
        "a": [1, 2, 3],
        "b": [4, 5, 6],
        "c": [7, 8, 9],
    }
    expected = [
        {"a": 1, "b": 4, "c": 7},
        {"a": 2, "b": 5, "c": 8},
        {"a": 3, "b": 6, "c": 9},
    ]
    output_list = transpose_dict_of_list(input_dict, 3)
    assert len(output_list) == 3
    for i in range(3):
        assert expected[i] == output_list[i]

    # We fail if num_elem does not match the list lengths.
    with pytest.raises(ValueError):
        _ = transpose_dict_of_list(input_dict, 0)
    with pytest.raises(ValueError):
        _ = transpose_dict_of_list(input_dict, 4)
