import numpy as np
import pandas as pd
import pytest
from astropy.table import Table
from tdastro.base_models import FunctionNode
from tdastro.graph_state import GraphState
from tdastro.math_nodes.given_sampler import GivenSampler, TableSampler


def _test_func(value1, value2):
    """Return the sum of the two parameters.

    Parameters
    ----------
    value1 : `float`
        The first parameter.
    value2 : `float`
        The second parameter.
    """
    return value1 + value2


def test_given_sampler():
    """Test that we can retrieve numbers from a GivenSampler."""
    given_node = GivenSampler([1.0, 1.5, 2.0, 2.5, 3.0, -1.0, 3.5])

    # Check that we generate the correct result and save it in the GraphState.
    state1 = GraphState(num_samples=2)
    results = given_node.compute(state1)
    assert np.array_equal(results, [1.0, 1.5])
    assert np.array_equal(given_node.get_param(state1, "function_node_result"), [1.0, 1.5])

    state2 = GraphState(num_samples=1)
    results = given_node.compute(state2)
    assert results == 2.0
    assert given_node.get_param(state2, "function_node_result") == 2.0

    state3 = GraphState(num_samples=2)
    results = given_node.compute(state3)
    assert np.array_equal(results, [2.5, 3.0])
    assert np.array_equal(given_node.get_param(state3, "function_node_result"), [2.5, 3.0])

    # Check that GivenSampler raises an error when it has run out of samples.
    state4 = GraphState(num_samples=4)
    with pytest.raises(IndexError):
        _ = given_node.compute(state4)

    # Resetting the GivenSampler starts back at the beginning.
    given_node.reset()
    state5 = GraphState(num_samples=6)
    results = given_node.compute(state5)
    assert np.array_equal(results, [1.0, 1.5, 2.0, 2.5, 3.0, -1.0])
    assert np.array_equal(
        given_node.get_param(state5, "function_node_result"),
        [1.0, 1.5, 2.0, 2.5, 3.0, -1.0],
    )


def test_test_given_sampler_compound():
    """Test that we can use the GivenSampler as input into another node."""
    values = [1.0, 1.5, 2.0, 2.5, 3.0, -1.0, 3.5, 4.0, 10.0, -2.0]
    given_node = GivenSampler(values)

    # Create a function node that takes the next value and adds 2.0.
    compound_node = FunctionNode(_test_func, value1=given_node, value2=2.0)
    for val in values:
        state = compound_node.sample_parameters()
        assert compound_node.get_param(state, "function_node_result") == val + 2.0

    # Reset the given node and try generating all the samples.
    given_node.reset()
    state2 = compound_node.sample_parameters(num_samples=8)
    assert np.array_equal(
        compound_node.get_param(state2, "function_node_result"),
        [3.0, 3.5, 4.0, 4.5, 5.0, 1.0, 5.5, 6.0],
    )


@pytest.mark.parametrize("test_data_type", ["dict", "ap_table", "pd_df"])
def test_table_sampler(test_data_type):
    """Test that we can retrieve numbers from a TableSampler from a
    dictionary, AstroPy Table, and Panda's DataFrame."""
    raw_data_dict = {
        "A": [1, 2, 3, 4, 5, 6, 7, 8],
        "B": [1, 1, 1, 1, 1, 1, 1, 1],
        "C": [3, 4, 5, 6, 7, 8, 9, 10],
    }

    # Convert the data type depending on the parameterized value.
    if test_data_type == "dict":
        data = raw_data_dict
    elif test_data_type == "ap_table":
        data = Table(raw_data_dict)
    elif test_data_type == "pd_df":
        data = pd.DataFrame(raw_data_dict)
    else:
        data = None

    # Create the table sampler from the data.
    table_node = TableSampler(data, node_label="node")
    state = table_node.sample_parameters(num_samples=2)
    assert len(state) == 3
    assert np.allclose(state["node"]["A"], [1, 2])
    assert np.allclose(state["node"]["B"], [1, 1])
    assert np.allclose(state["node"]["C"], [3, 4])

    state = table_node.sample_parameters(num_samples=1)
    assert len(state) == 3
    assert state["node"]["A"] == 3
    assert state["node"]["B"] == 1
    assert state["node"]["C"] == 5

    state = table_node.sample_parameters(num_samples=4)
    assert len(state) == 3
    assert np.allclose(state["node"]["A"], [4, 5, 6, 7])
    assert np.allclose(state["node"]["B"], [1, 1, 1, 1])
    assert np.allclose(state["node"]["C"], [6, 7, 8, 9])

    # We go past the end of the data.
    with pytest.raises(IndexError):
        _ = table_node.sample_parameters(num_samples=4)

    # We can reset and sample from the beginning.
    table_node.reset()
    state = table_node.sample_parameters(num_samples=2)
    assert len(state) == 3
    assert np.allclose(state["node"]["A"], [1, 2])
    assert np.allclose(state["node"]["B"], [1, 1])
    assert np.allclose(state["node"]["C"], [3, 4])
