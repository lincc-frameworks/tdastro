import random

import numpy as np
import pytest
from tdastro.base_models import FunctionNode, ParameterizedNode, ParameterSource, SingleVariableNode


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
        self.add_parameter("value1", value1, required=True, **kwargs)
        self.add_parameter("value2", value2, required=True, **kwargs)
        self.add_parameter(
            "value_sum",
            FunctionNode(_test_func, value1=self.value1, value2=self.value2),
            required=True,
            **kwargs,
        )


def test_parameter_source():
    """Test the ParameterSource creation and setter functions."""
    source = ParameterSource("test")
    assert source.parameter_name == "test"
    assert source.full_name == "test"
    assert source.source_type == ParameterSource.UNDEFINED
    assert source.dependency is None
    assert source.value is None
    assert not source.fixed
    assert not source.required

    source.set_as_constant(10.0)
    assert source.parameter_name == "test"
    assert source.full_name == "test"
    assert source.source_type == ParameterSource.CONSTANT
    assert source.dependency is None
    assert source.value == 10.0
    assert not source.fixed
    assert not source.required

    source.set_name("my_var", "my_node")
    assert source.parameter_name == "my_var"
    assert source.full_name == "my_node.my_var"

    with pytest.raises(ValueError):
        source.set_as_constant(_test_func)

    node = SingleVariableNode("A", 20.0)
    source.set_as_parameter(node, "A")
    assert source.source_type == ParameterSource.MODEL_PARAMETER
    assert source.dependency is node
    assert source.value == "A"
    assert not source.fixed
    assert not source.required

    func = FunctionNode(_test_func, value1=0.1, value2=0.4)
    source.set_as_function(func)
    assert source.source_type == ParameterSource.FUNCTION_NODE
    assert source.dependency is func
    assert not source.fixed
    assert not source.required


def test_parameterized_node():
    """Test that we can sample and create a PairModel object."""
    # Simple addition
    model1 = PairModel(value1=0.5, value2=0.5)
    assert str(model1) == "test_base_models.PairModel"

    state = model1.sample_parameters()
    assert model1.get_param(state, "value1") == 0.5
    assert model1.get_param(state, "value2") == 0.5
    assert model1.get_param(state, "value_sum") == 1.0

    # Use value1=model.value and value2=1.0
    model2 = PairModel(value1=model1.value1, value2=1.0, node_label="test")
    assert str(model2) == "test"

    state = model2.sample_parameters()
    assert model2.get_param(state, "value1") == 0.5
    assert model2.get_param(state, "value2") == 1.0
    assert model2.get_param(state, "value_sum") == 1.5

    # If we set an ID it shows up in the name.
    model2._node_pos = 100
    model2._update_node_string()
    assert str(model2) == "100:test"

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


def test_parameterized_node_get_dependencies():
    """Test that we can extract the parameters of a graph of ParameterizedNode."""
    model1 = PairModel(value1=0.5, value2=1.5, node_identifier="1")
    assert len(model1.direct_dependencies) == 1

    model2 = PairModel(value1=model1.value1, value2=3.0, node_identifier="2")
    assert len(model2.direct_dependencies) == 2
    assert model1 in model2.direct_dependencies

    model3 = PairModel(value1=model1.value1, value2=model2.value_sum, node_identifier="3")
    assert len(model3.direct_dependencies) == 3
    assert model1 in model3.direct_dependencies
    assert model2 in model3.direct_dependencies


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


def test_parameterized_node_seed():
    """Test that we can set a random seed for the entire graph."""
    # Left unspecified we use full random seeds.
    model_a = PairModel(value1=0.5, value2=0.5)
    model_b = PairModel(value1=0.5, value2=0.5)
    assert model_a._object_seed != model_b._object_seed

    # If we specify a seed, the results are the same objects with
    # the same name (class + node_id + node_label) and different
    # otherwise. Everything starts with a default node_id = None.
    model_a = PairModel(value1=0.5, value2=0.5, node_label="A")
    model_b = PairModel(value1=0.5, value2=0.5, node_label="B")
    model_c = PairModel(value1=0.5, value2=0.5, node_label="A")
    model_d = SingleVariableNode("value1", 0.5, node_label="A")
    model_e = SingleVariableNode("value1", 0.5, node_label="C")

    model_a.set_seed(graph_base_seed=10)
    model_b.set_seed(graph_base_seed=10)
    model_c.set_seed(graph_base_seed=10)
    model_d.set_seed(graph_base_seed=10)
    model_e.set_seed(graph_base_seed=10)

    assert model_a._object_seed != model_b._object_seed
    assert model_a._object_seed == model_c._object_seed
    assert model_a._object_seed == model_d._object_seed
    assert model_a._object_seed != model_e._object_seed

    assert model_b._object_seed != model_a._object_seed
    assert model_b._object_seed != model_c._object_seed
    assert model_b._object_seed != model_d._object_seed
    assert model_b._object_seed != model_e._object_seed

    assert model_c._object_seed == model_a._object_seed
    assert model_c._object_seed != model_b._object_seed
    assert model_c._object_seed == model_d._object_seed
    assert model_c._object_seed != model_e._object_seed

    assert model_d._object_seed == model_a._object_seed
    assert model_d._object_seed != model_b._object_seed
    assert model_d._object_seed == model_c._object_seed
    assert model_d._object_seed != model_e._object_seed


def test_parameterized_node_base_seed_fail():
    """Test that we can set a random seed for the entire graph."""
    model_a = PairModel(value1=0.5, value2=0.5)
    model_a.set_seed(graph_base_seed=10)
    model_a.sample_parameters()

    model_b = PairModel(value1=1.5, value2=0.5)
    model_b.set_seed(graph_base_seed=10)
    model_b.sample_parameters()

    # The models have the same string and seed, but that's desired since they
    # are not in the same graph.
    assert str(model_a) == str(model_b)
    assert model_a._object_seed == model_b._object_seed

    # But if we change the graph to link them, we don't want them
    # to have the same seed.
    model_b.set_parameter("value1", model_a.value_sum)
    with pytest.raises(KeyError):
        model_b.sample_parameters()

    # We need to reset the node IDs (their positions within the graph)
    # so they have unique identifiers and seeds.
    model_b.update_graph_information()
    assert str(model_a) != str(model_b)
    assert model_a._object_seed != model_b._object_seed
    model_b.sample_parameters()


def test_single_variable_node():
    """Test that we can create and query a SingleVariableNode."""
    node = SingleVariableNode("A", 10.0)
    assert str(node) == "tdastro.base_models.SingleVariableNode"

    state = node.sample_parameters()
    assert node.get_param(state, "A") == 10


def test_function_node_basic():
    """Test that we can create and query a FunctionNode."""
    my_func = FunctionNode(_test_func, value1=1.0, value2=2.0)
    state = my_func.sample_parameters()

    assert my_func.compute(state) == 3.0
    assert my_func.compute(state, value2=3.0) == 4.0
    assert my_func.compute(state, value2=3.0, unused_param=5.0) == 4.0
    assert my_func.compute(state, value2=3.0, value1=1.0) == 4.0
    assert str(my_func) == "0:tdastro.base_models.FunctionNode:_test_func"


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
