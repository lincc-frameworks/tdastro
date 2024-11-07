import random

import jax
import numpy as np
import pytest
from tdastro.base_models import FunctionNode, ParameterizedNode, ParameterSource
from tdastro.math_nodes.single_value_node import SingleVariableNode


def _sampler_fun(**kwargs):
    """Return a random value between 0 and 1.0.

    Parameters
    ----------
    **kwargs : `dict`, optional
        Absorbs additional parameters
    """
    return random.random()


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


class PairModel(ParameterizedNode):
    """A test class for the ParameterizedNode.

    Parameters
    ----------
    value1 : `float`
        The first value.
    value2 : `float`
        The second value.
    value_sum : `float`
        The sum of the two values.
    **kwargs : `dict`, optional
        Any additional keyword arguments.
    """

    def __init__(self, value1, value2, **kwargs):
        super().__init__(**kwargs)
        self.add_parameter("value1", value1, **kwargs)
        self.add_parameter("value2", value2, **kwargs)
        self.add_parameter(
            "value_sum",
            FunctionNode(_test_func, value1=self.value1, value2=self.value2),
            **kwargs,
        )


def test_parameter_source():
    """Test the ParameterSource creation and setter functions."""
    source = ParameterSource("test")
    assert source.parameter_name == "test"
    assert source.node_name == ""
    assert source.source_type == ParameterSource.UNDEFINED
    assert source.dependency is None
    assert source.value is None

    source.set_as_constant(10.0)
    assert source.parameter_name == "test"
    assert source.source_type == ParameterSource.CONSTANT
    assert source.dependency is None
    assert source.value == 10.0

    with pytest.raises(ValueError):
        source.set_as_constant(_test_func)

    node = SingleVariableNode("A", 20.0)
    source.set_as_parameter(node, "A")
    assert source.source_type == ParameterSource.MODEL_PARAMETER
    assert source.dependency is node
    assert source.value == "A"

    func = FunctionNode(_test_func, value1=0.1, value2=0.4)
    source.set_as_function(func)
    assert source.source_type == ParameterSource.FUNCTION_NODE
    assert source.dependency is func


def test_parameterized_node():
    """Test that we can sample and create a PairModel object."""
    # Simple addition
    model1 = PairModel(value1=0.5, value2=0.5)
    assert str(model1) == "PairModel"

    state = model1.sample_parameters()
    assert model1.get_param(state, "value1") == 0.5
    assert model1.get_param(state, "value2") == 0.5
    assert model1.get_param(state, "value_sum") == 1.0

    # We can also access the parameters using their names and the graph state.
    assert model1.value1(state) == 0.5
    assert model1.value2(state) == 0.5
    assert model1.value_sum(state) == 1.0

    # We use the default for un-assigned parameters.
    assert model1.get_param(state, "value3") is None
    assert model1.get_param(state, "value4", 1.0) == 1.0

    # If we set a position (and there is no node_label), the position shows up in the name.
    model1.node_pos = 100
    model1._update_node_string()
    assert str(model1) == "PairModel_100"

    # Use value1=model.value and value2=1.0
    model2 = PairModel(value1=model1.value1, value2=1.0, node_label="test")
    assert str(model2) == "test"

    state = model2.sample_parameters()
    assert model2.get_param(state, "value1") == 0.5
    assert model2.get_param(state, "value2") == 1.0
    assert model2.get_param(state, "value_sum") == 1.5

    # Compute value1 from model2's result and value2 from the sampler function.
    # The sampler function is auto-wrapped in a FunctionNode.
    model3 = PairModel(value1=model2.value_sum, value2=_sampler_fun)
    state = model3.sample_parameters()
    rand_val = model3.get_param(state, "value2")
    assert model3.get_param(state, "value_sum") == pytest.approx(1.5 + rand_val)

    # Compute value1 from model3's result (which is itself the result for model2 +
    # a random value) and value2 = -1.0.
    model4 = PairModel(value1=model3.value_sum, value2=-1.0)
    state = model4.sample_parameters()
    rand_val = model3.get_param(state, "value2")
    assert model4.get_param(state, "value_sum") == pytest.approx(0.5 + rand_val)

    # We can resample and it should change the result.
    new_state = model4.sample_parameters()
    rand_val = model3.get_param(new_state, "value2")

    # Nothing changes in model1 or model2
    assert model1.get_param(new_state, "value1") == 0.5
    assert model1.get_param(new_state, "value2") == 0.5
    assert model1.get_param(new_state, "value_sum") == 1.0
    assert model2.get_param(new_state, "value1") == 0.5
    assert model2.get_param(new_state, "value2") == 1.0
    assert model2.get_param(new_state, "value_sum") == 1.5

    # Models 3 and 4 use the data from the new random value.
    assert model3.get_param(new_state, "value_sum") == pytest.approx(1.5 + rand_val)
    assert model4.get_param(new_state, "value_sum") == pytest.approx(0.5 + rand_val)
    assert model4.get_param(state, "value_sum") != model4.get_param(new_state, "value_sum")


def test_parameterized_node_label_collision():
    """Test that throw an error when two nodes use the same label."""
    node_a = SingleVariableNode("A", 10.0, node_label="A")
    node_b = SingleVariableNode("B", 20.0, node_label="B")
    pair1 = PairModel(value1=node_a.A, value2=node_b.B, node_label="pair1")
    pair2 = PairModel(value1=pair1.value_sum, value2=node_b.B, node_label="pair2")

    # No collision even though node_b is referenced twice.
    state = pair2.sample_parameters()
    assert state["pair1"]["value_sum"] == 30.0
    assert state["pair2"]["value_sum"] == 50.0

    # We run into a problem if we reuse a label. Label "A" collides multiple
    # levels above.
    node_c = SingleVariableNode("C", 5.0, node_label="C")
    pair3 = PairModel(value1=pair2.value_sum, value2=node_c.C, node_label="A")
    with pytest.raises(ValueError):
        _ = pair3.sample_parameters()


def test_parameterized_node_modify():
    """Test that we can modify the parameters in a node."""
    model = PairModel(value1=0.5, value2=0.5)

    # We cannot add a parameter a second time.
    with pytest.raises(KeyError):
        model.add_parameter("value1", 5.0)

    # We can set the parameter.
    model.set_parameter("value1", 5.0)

    # We cannot set a value that hasn't been added.
    with pytest.raises(KeyError):
        model.set_parameter("brightness", 5.0)


def test_parameterized_node_build_pytree():
    """Test that we can extract the PyTree of a graph."""
    model1 = PairModel(value1=0.5, value2=1.5, node_label="A")
    model2 = PairModel(value1=model1.value1, value2=3.0, node_label="B")
    graph_state = model2.sample_parameters()

    pytree = model2.build_pytree(graph_state)
    assert pytree["A"]["value1"] == 0.5
    assert pytree["A"]["value2"] == 1.5
    assert pytree["B"]["value2"] == 3.0

    # Manually set value2 to allow_gradient to False and check that it no
    # longer appears in the pytree.
    model1.setters["value2"].allow_gradient = False

    pytree = model2.build_pytree(graph_state)
    assert pytree["A"]["value1"] == 0.5
    assert pytree["B"]["value2"] == 3.0
    assert "value2" not in pytree["A"]

    # If we set node B's value1 to allow the gradient, it will appear and
    # neither of node A's value will appear (because the gradient stops at
    # B.value1).
    model1.setters["value2"].allow_gradient = True
    model2.setters["value1"].allow_gradient = True

    pytree = model2.build_pytree(graph_state)
    assert "A" not in pytree
    assert pytree["B"]["value1"] == 0.5
    assert pytree["B"]["value2"] == 3.0


def test_function_node_basic():
    """Test that we can create and query a FunctionNode."""
    my_func = FunctionNode(_test_func, value1=1.0, value2=2.0)
    state = my_func.sample_parameters()

    assert my_func.compute(state) == 3.0
    assert my_func.compute(state, value2=3.0) == 4.0
    assert my_func.compute(state, value2=3.0, unused_param=5.0) == 4.0
    assert my_func.compute(state, value2=3.0, value1=1.0) == 4.0
    assert str(my_func) == "FunctionNode:_test_func_0"


def test_function_node_chain():
    """Test that we can create and query a chained FunctionNode."""
    func1 = FunctionNode(_test_func, value1=1.0, value2=1.0)
    func2 = FunctionNode(_test_func, value1=func1, value2=3.0)
    state = func2.sample_parameters()

    assert func2.compute(state) == 5.0


def test_no_resample_functions():
    """Test that if we use the same node as dependencies in two other nodes, we do not resample it."""
    rand_val = FunctionNode(_sampler_fun)
    node_a = SingleVariableNode("A", rand_val)
    node_b = PairModel(value1=node_a.A, value2=rand_val)
    state = node_b.sample_parameters()

    # All the values should be the same (no resampling of A.)
    assert node_b.get_param(state, "value1") == node_a.get_param(state, "A")
    assert node_b.get_param(state, "value2") == node_a.get_param(state, "A")

    # They should change when we resample.
    new_state = node_b.sample_parameters()
    assert node_a.get_param(state, "A") != node_a.get_param(new_state, "A")
    assert node_b.get_param(new_state, "value1") == node_a.get_param(new_state, "A")
    assert node_b.get_param(new_state, "value2") == node_a.get_param(new_state, "A")


def test_np_sampler_method():
    """Test that we can wrap numpy random functions."""
    rng = np.random.default_rng(1001)
    my_func = FunctionNode(rng.normal, loc=10.0, scale=1.0, outputs=["val"])

    # Sample 1000 times with the default values. Check that we are near
    # the expected mean and not everything is equal.
    vals = []
    for _ in range(1000):
        state = my_func.sample_parameters()
        vals.append(my_func.get_param(state, "val"))
    assert abs(np.mean(vals) - 10.0) < 1.0
    assert not np.all(vals == vals[0])


def test_function_node_obj():
    """Test that we can create and query a FunctionNode that depends on
    another ParameterizedNode.
    """
    # The model depends on the function.
    func = FunctionNode(_test_func, value1=5.0, value2=6.0)
    model = PairModel(value1=10.0, value2=func)
    state = model.sample_parameters()
    assert model.get_param(state, "value_sum") == 21.0

    # Function depends on the model's value2 parameter.
    model = PairModel(value1=-10.0, value2=17.5)
    func = FunctionNode(_test_func, value1=5.0, value2=model.value2, outputs=["res"])
    state = func.sample_parameters()

    assert model.get_param(state, "value_sum") == 7.5
    assert func.get_param(state, "res") == 22.5

    # We can always override the node's parameters with kwargs.
    assert func.compute(state, value1=1.0, value2=4.0) == 5.0


def test_function_node_multi():
    """Test that we can query a function node with multiple outputs."""

    def _test_func2(value1, value2):
        return (value1 + value2, value1 - value2)

    my_func = FunctionNode(_test_func2, outputs=["sum_res", "diff_res"], value1=5.0, value2=6.0)
    model = PairModel(value1=my_func.sum_res, value2=my_func.diff_res)
    state = model.sample_parameters()

    assert model.get_param(state, "value1") == 11.0
    assert model.get_param(state, "value2") == -1.0
    assert model.get_param(state, "value_sum") == 10.0


def test_function_node_jax():
    """Test that we can perform a JAX grad computation on a graph of function nodes."""

    def _test_func2(value1, value2):
        return value1 / value2

    # Create a function of a / b + c
    div_node = FunctionNode(_test_func2, value1=4.0, value2=0.5, node_label="div")
    sum_node = FunctionNode(_test_func, value1=1.0, value2=div_node, node_label="sum")
    graph_state = sum_node.sample_parameters()

    pytree = sum_node.build_pytree(graph_state)
    gr_func = jax.value_and_grad(sum_node.resample_and_compute)
    values, gradients = gr_func(pytree)
    assert values == 9.0
    assert gradients["sum"]["value1"] == 1.0
    assert gradients["div"]["value1"] == 2.0
    assert gradients["div"]["value2"] == -16.0
